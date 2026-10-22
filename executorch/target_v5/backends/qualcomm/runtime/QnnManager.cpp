/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/QnnBackendOptions.h>
#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/backends/qualcomm/runtime/QnnManager.h>
#include <executorch/backends/qualcomm/runtime/SharedBuffer.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnBackendCommon.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnCustomProtocol.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnImplementation.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>

namespace executorch {
namespace backends {
namespace qnn {

using executorch::runtime::Error;

bool CompareExportedInput(
    const std::shared_ptr<TensorWrapper>& a,
    const std::shared_ptr<TensorWrapper>& b) {
  // Using the order of the nodes as external_id in AOT
  // to extract the right arg from *args at runtime
  int numA = std::stoi(a->GetName().substr(a->GetName().find('_') + 1));
  int numB = std::stoi(b->GetName().substr(b->GetName().find('_') + 1));
  return numA < numB;
}

int ExtractMutableBufferNumber(const std::string& name) {
  std::string prefix = "mutbuf_";
  size_t startPos = name.find(prefix);
  if (startPos != std::string::npos) {
    startPos += prefix.length();
    return std::stoi(name.substr(startPos));
  }
  return -1;
}

// Only new, explicitly numbered outputs are reordered. Old context binaries
// retain their original order, including tensor-dump and mutable-buffer paths.
static int ExportedOutputOrder(const std::string& name) {
  constexpr const char* prefix = "output_order_";
  if (name.compare(0, std::strlen(prefix), prefix) != 0) {
    return -1;
  }
  return std::stoi(name.substr(std::strlen(prefix)));
}

static void SortExportedOutputs(
    std::vector<std::shared_ptr<TensorWrapper>>& tensors) {
  std::stable_sort(tensors.begin(), tensors.end(), [](const auto& a, const auto& b) {
    const int lhs = ExportedOutputOrder(a->GetName());
    const int rhs = ExportedOutputOrder(b->GetName());
    if (lhs < 0) return false;
    if (rhs < 0) return true;
    return lhs < rhs;
  });
}

QnnManager::~QnnManager() {
  Destroy();
}

QnnManager::QnnManager(
    const QnnExecuTorchOptions* options,
    const QnnExecuTorchContextBinary& qnn_executorch_context_binary)
    : qnn_context_blob_(qnn_executorch_context_binary), options_(options) {
  QnnExecuTorchBackendType backend_type =
      options->backend_options()->backend_type();

  if (get_option(options_->log_level(), QNN_RUNTIME_LOG_LEVEL) >=
      QnnExecuTorchLogLevel::kLogLevelInfo) {
    QNN_EXECUTORCH_LOG_INFO(
        "soc_model in soc_info: %s",
        EnumNameQcomChipset(options_->soc_info()->soc_model()));
    QNN_EXECUTORCH_LOG_INFO(
        "backend_type: %s", EnumNameQnnExecuTorchBackendType(backend_type));
    QNN_EXECUTORCH_LOG_INFO(
        "library_path: %s", options->library_path()->str().c_str());
    QNN_EXECUTORCH_LOG_INFO("dump intermediate outputs: %s", IsTensorDump());
    QNN_EXECUTORCH_LOG_INFO(
        "log_level: %s",
        EnumNameQnnExecuTorchLogLevel(
            get_option(options_->log_level(), QNN_RUNTIME_LOG_LEVEL)));
    QNN_EXECUTORCH_LOG_INFO(
        "profile_level: %s",
        EnumNameQnnExecuTorchProfileLevel(
            get_option(options_->profile_level(), QNN_RUNTIME_PROFILE_LEVEL)));
    QNN_EXECUTORCH_LOG_INFO(
        "the size of qnn context binary: %d",
        qnn_executorch_context_binary.nbytes);
    QNN_EXECUTORCH_LOG_INFO(
        "Is on-device graph construction: %d", options->online_prepare());
    QNN_EXECUTORCH_LOG_INFO(
        "Enable shared buffer: %d", options->shared_buffer());
    QNN_EXECUTORCH_LOG_INFO(
        "The number of op packages: %d",
        options_->op_package_options() &&
                options_->op_package_options()->op_package_infos()
            ? options_->op_package_options()->op_package_infos()->size()
            : 0);
  }

  backend_params_ptr_ = std::make_unique<BackendConfigParameters>();
  backend_bundle_ptr_ = std::make_shared<QnnBackendBundle>();

  qnn_dlc_manager_ =
      std::make_shared<QnnDlcManager>(qnn_context_blob_, options_);
}

namespace {
// zero-copy 진단/수정 (2026-09-15 KST).
// 같은 버퍼(예: 모든 샤드가 쓰는 attention mask, input_pos)를 여러 context 에 등록하면
// ION 은 "fd 1개 = memHandle 1개" 규칙을 어기게 된다. QNN_SHM_EXCLUSIVE_CTX=1 이면
// 버퍼를 처음 등록한 context 만 memHandle 로 쓰고, 다른 context 는 RAW 로 넘긴다.
std::mutex& ShmOwnerMutex() {
  static std::mutex mu;
  return mu;
}
std::unordered_map<void*, const void*>& ShmOwners() {
  static std::unordered_map<void*, const void*> owners;
  return owners;
}
// QNN_SHM_CTX_MASK=<hex>: 실행 순서로 매긴 context 번호 i 의 비트가 켜진 context 만 등록한다(진단용).
int ShmContextIndex(const void* manager) {
  static std::unordered_map<const void*, int> index;
  auto it = index.find(manager);
  if (it != index.end()) {
    return it->second;
  }
  const int value = static_cast<int>(index.size());
  index.emplace(manager, value);
  return value;
}
bool ShmContextAllowed(const void* manager) {
  static const unsigned long long mask = [] {
    const char* v = std::getenv("QNN_SHM_CTX_MASK");
    return v == nullptr ? ~0ULL : std::strtoull(v, nullptr, 16);
  }();
  const int i = ShmContextIndex(manager);
  return i >= 64 || ((mask >> i) & 1ULL) != 0;
}
bool ShmExclusiveContext() {
  static const bool enabled = [] {
    const char* v = std::getenv("QNN_SHM_EXCLUSIVE_CTX");
    return v != nullptr && std::strcmp(v, "1") == 0;
  }();
  return enabled;
}

// mutable buffer 를 context 단위로 공유하고 고정한다 (2026-09-17 KST).
// 상류 구현은 그래프마다 따로 할당하므로 prefill 이 갱신한 상태를 decode 가 못 본다.
// 또 new char[] 라 초기화되지 않고, RAW 라 공유 버퍼(zero-copy) 대상도 아니다.
//   QNN_MUTBUF_SHARE=1  같은 context 안의 그래프들이 같은 메모리를 쓴다 + 0 으로 초기화
//   QNN_MUTBUF_SHM=1    그 메모리를 rpcmem 에서 잡고 memHandle 로 등록한다
// 주의: 헤더에 데이터 멤버를 추가하지 않는다. 다른 정적 라이브러리 객체와 배치가 어긋나
//       segfault 가 난다(2026-09-15 경험). 상태는 아래 표에 둔다.
bool MutBufShareEnabled() {
  static const bool enabled = [] {
    const char* v = std::getenv("QNN_MUTBUF_SHARE");
    return v != nullptr && std::strcmp(v, "1") == 0;
  }();
  return enabled;
}
bool MutBufShmEnabled() {
  static const bool enabled = [] {
    const char* v = std::getenv("QNN_MUTBUF_SHM");
    return v != nullptr && std::strcmp(v, "1") == 0;
  }();
  return enabled;
}
struct MutBufEntry {
  void* ptr;
  std::uint32_t bytes;
  bool from_shm;
  Qnn_MemHandle_t handle;
  Qnn_DataType_t dtype;
  std::int32_t zero_point;
};
// 양자화된 상태의 "실수 0" 은 zero point 다 (QNN 의 offset = -zero_point).
// 0 으로 채우면 -zero_point*scale (러너 실측 약 -29) 에서 시작해 첫 토큰부터 틀어진다.
// kv_manager.cpp 의 fill_state_buffer 와 같은 규칙이다.
std::int32_t MutBufZeroPoint(const std::shared_ptr<TensorWrapper>& tensor_wrapper) {
  const Qnn_QuantizeParams_t qp = tensor_wrapper->GetQuantizeParams();
  if (qp.encodingDefinition == QNN_DEFINITION_DEFINED &&
      qp.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
    return -qp.scaleOffsetEncoding.offset;
  }
  return 0;
}
void MutBufFill(
    void* ptr,
    std::uint32_t bytes,
    Qnn_DataType_t dtype,
    std::int32_t zero_point) {
  switch (dtype) {
    case QNN_DATATYPE_UFIXED_POINT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
      std::fill_n(
          static_cast<std::uint16_t*>(ptr),
          bytes / sizeof(std::uint16_t),
          static_cast<std::uint16_t>(zero_point));
      return;
    case QNN_DATATYPE_UFIXED_POINT_8:
    case QNN_DATATYPE_SFIXED_POINT_8:
      std::fill_n(
          static_cast<std::uint8_t*>(ptr),
          bytes,
          static_cast<std::uint8_t>(zero_point));
      return;
    default:
      std::memset(ptr, 0, bytes);
      return;
  }
}
std::mutex& MutBufMutex() {
  static std::mutex mu;
  return mu;
}
std::unordered_map<const void*, std::unordered_map<int, MutBufEntry>>&
MutBufTable() {
  static std::unordered_map<const void*, std::unordered_map<int, MutBufEntry>>
      table;
  return table;
}
// 이 context 의 id 번 mutable buffer 가 쓸 메모리. 이미 만들어 둔 것이 있으면 재사용해
// 두 그래프(kv_forward/prefill_forward)가 같은 메모리를 보게 한다.
void* BindMutableBuffer(
    QnnManager* self,
    int id,
    const std::shared_ptr<TensorWrapper>& tensor_wrapper) {
  std::lock_guard<std::mutex> lk(MutBufMutex());
  auto& per_context = MutBufTable()[static_cast<const void*>(self)];
  auto it = per_context.find(id);
  static const bool kDebug = std::getenv("QNN_MUTBUF_DEBUG") != nullptr;
  if (kDebug) {
    QNN_EXECUTORCH_LOG_WARN(
        "[MUTBUF] ctx=%p id=%d tensor=%s bytes=%u dtype=%d zp=%d %s",
        static_cast<const void*>(self), id, tensor_wrapper->GetName().c_str(),
        tensor_wrapper->GetBytes(), static_cast<int>(tensor_wrapper->GetDataType()),
        MutBufZeroPoint(tensor_wrapper),
        it == per_context.end() ? "NEW"
            : (it->second.bytes == tensor_wrapper->GetBytes() ? "REUSE" : "SIZE-MISMATCH->NEW"));
  }
  if (it != per_context.end() &&
      it->second.bytes == tensor_wrapper->GetBytes()) {
    tensor_wrapper->FillDataBuffer(it->second.ptr);
    if (it->second.handle != nullptr) {
      // 같은 메모리를 두 번 등록하면 QNN 이 "Mem handle exists already" 로 거부한다
      // (2026-09-17 기기 실측). 먼저 받은 핸들을 그대로 물려준다.
      tensor_wrapper->SetMemHandle(it->second.handle);
    }
    return it->second.ptr;
  }
  void* ptr = nullptr;
  bool from_shm = false;
  if (MutBufShmEnabled()) {
    ptr = SharedBuffer::GetSharedBufferManager().AllocMem(
        tensor_wrapper->GetBytes(), /*alignment=*/4096);
    from_shm = ptr != nullptr;
  }
  if (ptr != nullptr) {
    tensor_wrapper->FillDataBuffer(ptr);
  } else {
    tensor_wrapper->AllocateDataBuffer();
    ptr = const_cast<void*>(tensor_wrapper->GetStaticTensorData());
  }
  if (ptr == nullptr) {
    return nullptr;
  }
  // 상류는 초기화하지 않는다. 첫 실행이 쓰레기 상태에서 시작하지 않게 채운다.
  const Qnn_DataType_t dtype = tensor_wrapper->GetDataType();
  const std::int32_t zero_point = MutBufZeroPoint(tensor_wrapper);
  MutBufFill(ptr, tensor_wrapper->GetBytes(), dtype, zero_point);
  Qnn_MemHandle_t handle = nullptr;
  if (from_shm && self->RegisterMem(ptr, tensor_wrapper) == Error::Ok) {
    handle = tensor_wrapper->GetMemHandle();
  }
  per_context[id] = MutBufEntry{
      ptr, tensor_wrapper->GetBytes(), from_shm, handle, dtype, zero_point};
  return ptr;
}
void ReleaseMutableBuffers(const void* owner) {
  std::lock_guard<std::mutex> lk(MutBufMutex());
  auto it = MutBufTable().find(owner);
  if (it == MutBufTable().end()) {
    return;
  }
  for (auto& entry : it->second) {
    if (entry.second.from_shm) {
      SharedBuffer::GetSharedBufferManager().FreeMem(entry.second.ptr);
    }
  }
  MutBufTable().erase(it);
}
} // namespace

// 요청 사이에 델리게이트가 쥔 상태를 비운다. 러너가 QnnExecuTorch.h 의 C API 로 부른다.
size_t ResetMutableBuffers(int fill) {
  std::lock_guard<std::mutex> lk(MutBufMutex());
  size_t count = 0;
  for (auto& context : MutBufTable()) {
    for (auto& entry : context.second) {
      if (entry.second.ptr != nullptr) {
        if (fill < 0) {
          // 기본: 각 버퍼의 zero point 로 되돌린다 (= 실수 0).
          MutBufFill(
              entry.second.ptr,
              entry.second.bytes,
              entry.second.dtype,
              entry.second.zero_point);
        } else {
          std::memset(entry.second.ptr, fill, entry.second.bytes);
        }
        ++count;
      }
    }
  }
  return count;
}

Error QnnManager::RegisterMem(
    void* data_ptr,
    const std::shared_ptr<TensorWrapper>& tensor_wrapper) {
  // Not enable shared buffer
  if (!options_->shared_buffer()) {
    return Error::Internal;
  }
  if (backend_params_ptr_->qnn_mem_manager_ptr_ != nullptr) {
    // 실제 I/O 를 매핑하기 직전에 예약해 둔 주소공간을 돌려준다.
    backend_params_ptr_->qnn_mem_manager_ptr_->ReleaseVaReservation();
  }
  {
    std::lock_guard<std::mutex> lk(ShmOwnerMutex());
    if (!ShmContextAllowed(static_cast<const void*>(this))) {
      return Error::Internal;
    }
  }
  if (ShmExclusiveContext()) {
    std::lock_guard<std::mutex> lk(ShmOwnerMutex());
    auto it = ShmOwners().find(data_ptr);
    if (it != ShmOwners().end() && it->second != static_cast<const void*>(this)) {
      return Error::Internal;
    }
  }
  Error status = RegisterMemImpl(data_ptr, tensor_wrapper);
  if (status == Error::Ok && ShmExclusiveContext()) {
    std::lock_guard<std::mutex> lk(ShmOwnerMutex());
    ShmOwners().emplace(data_ptr, static_cast<const void*>(this));
  }
  return status;
}

Error QnnManager::RegisterMemImpl(
    void* data_ptr,
    const std::shared_ptr<TensorWrapper>& tensor_wrapper) {
  SharedBuffer& shared_buffer_manager = SharedBuffer::GetSharedBufferManager();

  if (backend_params_ptr_->qnn_mem_manager_ptr_ == nullptr) {
    QNN_EXECUTORCH_LOG_WARN(
        "Backend %s doesn't supported shared buffer.",
        EnumNameQnnExecuTorchBackendType(
            options_->backend_options()->backend_type()));
    return Error::Internal;
  }

  void* custom_mem_base = shared_buffer_manager.GetCustomMemBase(data_ptr);
  if (custom_mem_base != nullptr) {
    return RegisterCustomMem(data_ptr, custom_mem_base, tensor_wrapper);
  }
  return RegisterIonMem(data_ptr, tensor_wrapper);
}

Error QnnManager::RegisterIonMem(
    void* data_ptr,
    const std::shared_ptr<TensorWrapper>& tensor_wrapper) {
  SharedBuffer& shared_buffer_manager = SharedBuffer::GetSharedBufferManager();
  if (!shared_buffer_manager.IsAllocated(data_ptr)) {
    // It means two scenarios here:
    // 1. the input and output partitioned graph
    // 2. Actually, user doesn't allocate shared buffer with
    // QnnExecuTorchAllocCustomMem API
    return Error::Internal;
  } else if (backend_params_ptr_->qnn_mem_manager_ptr_->IsRegistered(
                 tensor_wrapper->GetMemHandle(), data_ptr)) {
    if (get_option(options_->log_level(), QNN_RUNTIME_LOG_LEVEL) >=
        QnnExecuTorchLogLevel::kLogLevelInfo)
      QNN_EXECUTORCH_LOG_INFO(
          "Tensor name %s has been registered shared memory.",
          tensor_wrapper->GetName().c_str());
    return Error::Ok;
  }

  int32_t mem_fd = shared_buffer_manager.MemToFd(data_ptr);
  if (mem_fd == -1) {
    QNN_EXECUTORCH_LOG_WARN(
        "Tensor name %s is failed to get file descriptor.",
        tensor_wrapper->GetName().c_str());
    return Error::Internal;
  }
  ET_CHECK_OR_RETURN_ERROR(
      backend_params_ptr_->qnn_mem_manager_ptr_->RegisterIonMem(
          tensor_wrapper, mem_fd, data_ptr) == Error::Ok,
      Internal,
      "Fail to register to shared memory.");

  return Error::Ok;
}

Error QnnManager::RegisterCustomMem(
    void* data_ptr,
    void* custom_mem_base,
    const std::shared_ptr<TensorWrapper>& tensor_wrapper) {
  if (backend_params_ptr_->qnn_mem_manager_ptr_->IsRegistered(
          tensor_wrapper->GetMemHandle(), data_ptr)) {
    if (get_option(options_->log_level(), QNN_RUNTIME_LOG_LEVEL) >=
        QnnExecuTorchLogLevel::kLogLevelInfo)
      QNN_EXECUTORCH_LOG_INFO(
          "Tensor name %s has been registered shared memory.",
          tensor_wrapper->GetName().c_str());
    return Error::Ok;
  }

  CustomMemTensorInfo info{
      custom_mem_base,
      data_ptr,
      static_cast<size_t>(
          static_cast<char*>(data_ptr) - static_cast<char*>(custom_mem_base)),
      tensor_wrapper->GetBytes(),
      tensor_wrapper->GetDims(),
      tensor_wrapper->GetRank(),
      qnn_dtype_to_scalar_type_[tensor_wrapper->GetDataType()]};

  Qnn_MemHandle_t pre_registered_handle =
      backend_params_ptr_->qnn_mem_manager_ptr_->GetPreRegisteredHandle(info);
  // If this memory block has already been registered, we can use it directly.
  // This applies when running llama in lookahead mode with the same AR-N model
  // handling both the prompt processor and the token generator.
  if (pre_registered_handle != nullptr) {
    if (get_option(options_->log_level(), QNN_RUNTIME_LOG_LEVEL) >=
        QnnExecuTorchLogLevel::kLogLevelInfo) {
      QNN_EXECUTORCH_LOG_INFO(
          "Tensor name %s found a pre-registered memHandle.",
          tensor_wrapper->GetName().c_str());
    }
    return backend_params_ptr_->qnn_mem_manager_ptr_->SetMemHandle(
        tensor_wrapper, data_ptr, pre_registered_handle);
  }

  SharedBuffer& shared_buffer_manager = SharedBuffer::GetSharedBufferManager();

  size_t tensor_offset = info.pos;
  size_t total_custom_mem_size =
      shared_buffer_manager.GetAllocatedSize(custom_mem_base);

  int32_t mem_fd = shared_buffer_manager.MemToFd(custom_mem_base);
  // Note: If obtaining the file descriptor fails, it may be due to memory not
  // being released with QnnExecuTorchFreeCustomMem. In this situation, we could
  // consider adding a map to monitor it.
  if (mem_fd == -1) {
    QNN_EXECUTORCH_LOG_WARN(
        "Tensor name %s failed to get file descriptor.",
        tensor_wrapper->GetName().c_str());
    return Error::Internal;
  }

  ET_CHECK_OR_RETURN_ERROR(
      backend_params_ptr_->qnn_mem_manager_ptr_->RegisterCustomMem(
          tensor_wrapper,
          mem_fd,
          data_ptr,
          total_custom_mem_size,
          tensor_offset,
          info) == Error::Ok,
      Internal,
      "Fail to register to shared memory.");

  return Error::Ok;
}

Error QnnManager::InitBackend() {
  // Get or create the shared backend bundle
  Error err = QnnBackendUnifiedRegistry::GetInstance().GetOrCreateBackendBundle(
      options_, backend_bundle_ptr_);
  ET_CHECK_OR_RETURN_ERROR(
      err == Error::Ok,
      Internal,
      "Fail to get or create shared Qnn backend bundle. Error code: %d",
      static_cast<int>(err));
  return Error::Ok;
}

Error QnnManager::InitContext(
    std::optional<std::vector<std::string>> graph_names) {
  if (backend_params_ptr_->backend_init_state_ ==
      BackendInitializeState::UNINITIALIZED) {
    QNN_EXECUTORCH_LOG_INFO(
        "Initialize Qnn backend "
        "parameters for Qnn executorch backend type %d",
        options_->backend_options()->backend_type());
    backend_params_ptr_ = QnnBackendFactory().Create(
        backend_bundle_ptr_->implementation.get(),
        backend_bundle_ptr_->system_implementation.get(),
        backend_bundle_ptr_->qnn_backend_ptr.get(),
        backend_bundle_ptr_->qnn_device_ptr.get(),
        qnn_context_blob_,
        options_,
        qnn_dlc_manager_.get());
    ET_CHECK_OR_RETURN_ERROR(
        backend_params_ptr_ != nullptr,
        Internal,
        "Failed to load Qnn backend.");
    // Note: For online_prepare or deserialization, the graph name will be
    // obtained from the binary.
    ET_CHECK_OR_RETURN_ERROR(
        backend_params_ptr_->qnn_backend_cache_ptr_->Configure(
            graph_names.value_or(std::vector<std::string>{})) == Error::Ok,
        Internal,
        "Fail to configure Qnn backend cache");
    ET_CHECK_OR_RETURN_ERROR(
        backend_params_ptr_->qnn_context_ptr_->Configure() == Error::Ok,
        Internal,
        "Fail to configure Qnn context");
    for (const std::string& graph_name :
         backend_params_ptr_->qnn_context_ptr_->GetGraphNames()) {
      ET_CHECK_OR_RETURN_ERROR(
          backend_params_ptr_->qnn_graph_ptr_->Configure(graph_name) ==
              Error::Ok,
          Internal,
          "Fail to configure Qnn graph");
    }

    backend_params_ptr_->backend_init_state_ =
        BackendInitializeState::INITIALIZED;
#if defined(__ANDROID__)
    // context 생성 직후 I/O 매핑 공간 예약 (QnnMemManager::ReserveVa 설명 참고).
    const char* reserve_mb = std::getenv("QNN_SHM_RESERVE_MB");
    if (reserve_mb != nullptr && options_->shared_buffer() &&
        backend_params_ptr_->qnn_mem_manager_ptr_ != nullptr) {
      const long mb = std::strtol(reserve_mb, nullptr, 10);
      if (mb > 0) {
        backend_params_ptr_->qnn_mem_manager_ptr_->ReserveVa(
            static_cast<size_t>(mb) << 20);
      }
    }
#endif
  }

  if (IsOnlinePrepare() &&
      backend_params_ptr_->qnn_backend_cache_ptr_->GetCacheState() ==
          QnnBackendCache::SERIALIZE) {
    // Set up DLC environment at AOT time
    // Check whether the QNN version supports the DLC format.
    Qnn_ApiVersion_t qnn_version = {QNN_VERSION_INIT};
    backend_bundle_ptr_->implementation->GetQnnInterface()
        .qnn_backend_get_api_version(&qnn_version);

    ET_CHECK_OR_RETURN_ERROR(
        qnn_dlc_manager_->SetUpDlcEnvironment(
            qnn_version.coreApiVersion,
            graph_names.value_or(std::vector<std::string>{})) == Error::Ok,
        Internal,
        "Fail to setup Dlc environment");
  }
  return Error::Ok;
}

Error QnnManager::InitContextCache() {
  if (backend_params_ptr_->backend_init_state_ ==
      BackendInitializeState::UNINITIALIZED) {
    QNN_EXECUTORCH_LOG_INFO(
        "Initialize Qnn backend "
        "parameters for Qnn executorch backend type %d",
        options_->backend_options()->backend_type());
    backend_params_ptr_ = QnnBackendFactory().Create(
        backend_bundle_ptr_->implementation.get(),
        backend_bundle_ptr_->system_implementation.get(),
        backend_bundle_ptr_->qnn_backend_ptr.get(),
        backend_bundle_ptr_->qnn_device_ptr.get(),
        qnn_context_blob_,
        options_,
        qnn_dlc_manager_.get());
    ET_CHECK_OR_RETURN_ERROR(
        backend_params_ptr_ != nullptr,
        Internal,
        "Failed to load Qnn backend.");
    // Note: For online_prepare or deserialization, the graph name will be
    // obtained from the binary.
    ET_CHECK_OR_RETURN_ERROR(
        backend_params_ptr_->qnn_backend_cache_ptr_->Configure({}) == Error::Ok,
        Internal,
        "Fail to configure Qnn backend cache");

    backend_params_ptr_->backend_init_state_ =
        BackendInitializeState::INITIALIZED;
  }
  return Error::Ok;
}

Error QnnManager::AllocateTensor(const std::string& graph_name) {
  std::vector<Qnn_Tensor_t> input_tensors =
      backend_params_ptr_->qnn_context_ptr_->GetGraphInputs(graph_name);
  std::vector<Qnn_Tensor_t> output_tensors =
      backend_params_ptr_->qnn_context_ptr_->GetGraphOutputs(graph_name);

  // Mapping memory address for the input and output of mutable buffer
  std::unordered_map<int, const void*> mutable_buffer_id_to_memory_map;

  for (auto& tensor : input_tensors) {
    std::shared_ptr<TensorWrapper> tensor_wrapper = CreateTensorWrapper(tensor);
    tensor_wrapper->UpdateQnnTensorMeta(tensor);

    int mutable_buffer_id =
        ExtractMutableBufferNumber(tensor_wrapper->GetName());
    if (mutable_buffer_id != -1) {
      // Delegate maintains the memory for mutable buffer
      if (MutBufShareEnabled()) {
        mutable_buffer_id_to_memory_map[mutable_buffer_id] =
            BindMutableBuffer(this, mutable_buffer_id, tensor_wrapper);
      } else {
        tensor_wrapper->AllocateDataBuffer();
        mutable_buffer_id_to_memory_map[mutable_buffer_id] =
            tensor_wrapper->GetStaticTensorData();
      }
    }
    input_tensors_[graph_name].emplace_back(std::move(tensor_wrapper));
  }
  if (!options_->is_from_context_binary()) {
    std::sort(
        input_tensors_[graph_name].begin(),
        input_tensors_[graph_name].end(),
        CompareExportedInput);
  }
  for (size_t i = 0; i < output_tensors.size(); ++i) {
    std::shared_ptr<TensorWrapper> tensor_wrapper =
        CreateTensorWrapper(output_tensors[i]);
    tensor_wrapper->UpdateQnnTensorMeta(output_tensors[i]);
    const std::string& tensor_name = tensor_wrapper->GetName();
    // this is required by identifying shared buffer mechanism
    // info might be missed if context binary came from qnn_converter
    if (options_->is_from_context_binary() &&
        tensor_name.find("output_") == std::string::npos) {
      tensor_wrapper->SetName("output_" + tensor_name);
    }
    if (IsTensorDump()) {
      tensor_wrapper->AllocateDataBuffer();
    }
    int mutable_buffer_id =
        ExtractMutableBufferNumber(tensor_wrapper->GetName());
    if (mutable_buffer_id != -1 &&
        mutable_buffer_id_to_memory_map.find(mutable_buffer_id) !=
            mutable_buffer_id_to_memory_map.end()) {
      // Fill the same memory for I/O of mutable buffer
      if (MutBufShareEnabled()) {
        BindMutableBuffer(this, mutable_buffer_id, tensor_wrapper);
      } else {
        tensor_wrapper->FillDataBuffer(
            mutable_buffer_id_to_memory_map[mutable_buffer_id]);
      }
    }
    output_tensors_[graph_name].emplace_back(std::move(tensor_wrapper));
  }
  if (!options_->is_from_context_binary()) {
    SortExportedOutputs(output_tensors_[graph_name]);
  }
  return Error::Ok;
}

Error QnnManager::AllocateTensor(
    const std::string& graph_name,
    std::vector<std::shared_ptr<TensorWrapper>>& inputs,
    std::vector<std::shared_ptr<TensorWrapper>>& outputs) {
  input_tensors_[graph_name] = std::move(inputs);
  // TODO: suuport per-tensor dump in online prepare mode
  //       should be achievable with some pre-process
  if (!options_->is_from_context_binary()) {
    std::sort(
        input_tensors_[graph_name].begin(),
        input_tensors_[graph_name].end(),
        CompareExportedInput);
  }
  output_tensors_[graph_name] = std::move(outputs);
  if (!options_->is_from_context_binary()) {
    SortExportedOutputs(output_tensors_[graph_name]);
  }
  return Error::Ok;
}

Error QnnManager::Execute(
    const std::string& graph_name,
    const std::vector<Qnn_Tensor_t>& input_tensor_structs,
    std::vector<Qnn_Tensor_t>& output_tensor_structs,
    executorch::runtime::EventTracer* event_tracer) {
  Qnn_ErrorHandle_t error = QNN_SUCCESS;

  error = backend_params_ptr_->qnn_graph_ptr_->GraphExecute(
      graph_name, input_tensor_structs, output_tensor_structs);

  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR(
        "qnn_graph_execute failed. Error %d", QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }
  if (IsTensorDump()) {
    // TODO: Need to handle the graph which is partitioned.
    // Maybe we could use graph name.
    for (std::size_t out_idx = 0; out_idx < output_tensor_structs.size();
         ++out_idx) {
      const Qnn_Tensor_t& output_tensor = output_tensor_structs[out_idx];
      std::vector<executorch::aten::SizesType> sizes(
          QNN_TENSOR_VER_PTR(output_tensor)->dimensions,
          QNN_TENSOR_VER_PTR(output_tensor)->dimensions +
              QNN_TENSOR_VER_PTR(output_tensor)->rank);

      // Compute contiguous strides from sizes (e.g. [2,3,4] -> [12,4,1]).
      std::vector<executorch::aten::StridesType> stride_size(sizes.size());
      if (!sizes.empty()) {
        stride_size.back() = 1;
        for (int i = sizes.size() - 2; i >= 0; --i) {
          stride_size[i] = stride_size[i + 1] * sizes[i + 1];
        }
      }
      // Avoid using from_blob as it significantly increases shared library
      // size.
      executorch::aten::TensorImpl tensor_impl(
          qnn_dtype_to_scalar_type_[QNN_TENSOR_VER_PTR(output_tensor)
                                        ->dataType],
          sizes.size(),
          sizes.data(),
          QNN_TENSOR_VER_PTR(output_tensor)->clientBuf.data,
          nullptr,
          stride_size.data());

      executorch::runtime::event_tracer_log_output_delegate<
          executorch::aten::Tensor>(
          event_tracer,
          QNN_TENSOR_VER_PTR(output_tensor)->name,
          /*delegate_debug_id=*/
          static_cast<executorch::runtime::DebugHandle>(-1),
          executorch::aten::Tensor(&tensor_impl));
    }
  }

  return Error::Ok;
}

Error QnnManager::ProfileExecuteData(
    const std::string& graph_name,
    executorch::runtime::EventTracer* event_tracer) {
  Qnn_ErrorHandle_t error = QNN_SUCCESS;
  if (std::getenv("QNN_PROFILE_COMPACT") != nullptr || get_option(options_->profile_level(), QNN_RUNTIME_PROFILE_LEVEL) !=
      QnnExecuTorchProfileLevel::kProfileOff) {
    error = backend_params_ptr_->qnn_graph_ptr_->ProfileExecuteData(
        graph_name, event_tracer);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          " Failed to profile. Error %d", QNN_GET_ERROR_CODE(error));
      return Error::Internal;
    }
  }
  return Error::Ok;
}

void QnnManager::Destroy() {
  if (backend_params_ptr_ != nullptr &&
      backend_params_ptr_->qnn_mem_manager_ptr_ != nullptr) {
    backend_params_ptr_->qnn_mem_manager_ptr_->ReleaseVaReservation();
  }
  {
    std::lock_guard<std::mutex> lk(ShmOwnerMutex());
    for (auto it = ShmOwners().begin(); it != ShmOwners().end();) {
      it = it->second == static_cast<const void*>(this) ? ShmOwners().erase(it) : std::next(it);
    }
  }
  ReleaseMutableBuffers(static_cast<const void*>(this));
  qnn_dlc_manager_->Destroy();
  backend_params_ptr_.reset(new BackendConfigParameters());
  backend_bundle_ptr_.reset(new QnnBackendBundle());
}

void QnnManager::DestroyContext() {
  backend_params_ptr_.reset(new BackendConfigParameters());
  qnn_dlc_manager_->Destroy();
}

bool QnnManager::IsNodeSupportedByBackend(
    std::vector<std::shared_ptr<OpWrapper>>& op_wrappers) {
  Qnn_ErrorHandle_t error = QNN_SUCCESS;
  for (std::shared_ptr<OpWrapper>& op_wrapper : op_wrappers) {
    for (const auto& param : op_wrapper->GetParams()) {
      // unused?
      // auto* p_tensor_param = dynamic_cast<TensorParamWrapper*>(param.get());
      if (param->PopulateQnnParam() != Error::Ok) {
        QNN_EXECUTORCH_LOG_WARN(
            "Qnn Backend op validation failed "
            "with PopulateQnnParam: %d",
            QNN_GET_ERROR_CODE(error));
        return false;
      }
    }

    error = backend_bundle_ptr_->qnn_backend_ptr->BackendValidateOpConfig(
        op_wrapper->GetOpConfig());
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_WARN(
          "Qnn Backend op validation failed with error: %d",
          QNN_GET_ERROR_CODE(error));

      return false;
    }
  }
  return true;
}

Error QnnManager::GetContextBinary(
    QnnExecuTorchContextBinary& qnn_executorch_context_binary) {
  if (IsOnlinePrepare() &&
      qnn_dlc_manager_->backend_params_ptr_->qnn_context_ptr_.get() !=
          nullptr) {
    ET_CHECK_OR_RETURN_ERROR(
        qnn_dlc_manager_->backend_params_ptr_->qnn_context_ptr_
                ->GetContextBinary(qnn_executorch_context_binary) == Error::Ok,
        Internal,
        "Fail to get context binary.");
  }

  else {
    ET_CHECK_OR_RETURN_ERROR(
        backend_params_ptr_->qnn_context_ptr_->GetContextBinary(
            qnn_executorch_context_binary) == Error::Ok,
        Internal,
        "Fail to get context binary.");
  }
  return Error::Ok;
}

Error QnnManager::CompileDlc() {
  Qnn_ErrorHandle_t error;
  auto graphs = qnn_dlc_manager_->GetQnnDlcGraphInfoPtr();
  uint32_t num_graphs = qnn_dlc_manager_->GetQnnDlcGraphInfoNum();
  for (uint32_t i = 0; i < num_graphs; ++i) {
    auto& graphInfo = graphs[i].graphInfoV1;
    Qnn_GraphHandle_t graphHandle;
    error = backend_bundle_ptr_->implementation->GetQnnInterface()
                .qnn_graph_retrieve(
                    backend_params_ptr_->qnn_context_ptr_->GetHandle(),
                    graphInfo.graphName,
                    &graphHandle);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Failed to retrieve graph %s. Error %d.",
          graphInfo.graphName,
          QNN_GET_ERROR_CODE(error));
      return Error::Internal;
    }
    backend_params_ptr_->qnn_graph_ptr_->SetGraphHandle(
        graphInfo.graphName, graphHandle);
    error =
        backend_params_ptr_->qnn_graph_ptr_->GraphFinalize(graphInfo.graphName);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Failed to finalize Qnn Graph with error: %d",
          QNN_GET_ERROR_CODE(error));
      return Error::Internal;
    }

    std::vector<std::shared_ptr<TensorWrapper>> graph_inputs, graph_outputs,
        tensors;

    // Mapping memory address for the input and output of mutable buffer
    std::unordered_map<int, const void*> mutable_buffer_id_to_memory_map;
    for (uint32_t i = 0; i < graphInfo.numGraphInputs; ++i) {
      auto tw = CreateTensorWrapper(graphInfo.graphInputs[i]);
      tw->UpdateQnnTensorMeta(graphInfo.graphInputs[i]);

      int mutable_buffer_id = ExtractMutableBufferNumber(tw->GetName());
      if (mutable_buffer_id != -1) {
        // Delegate maintains the memory for mutable buffer
        if (MutBufShareEnabled()) {
          mutable_buffer_id_to_memory_map[mutable_buffer_id] =
              BindMutableBuffer(this, mutable_buffer_id, tw);
        } else {
          tw->AllocateDataBuffer();
          mutable_buffer_id_to_memory_map[mutable_buffer_id] =
              tw->GetStaticTensorData();
        }
      }
      graph_inputs.push_back(tw);
    }
    for (uint32_t i = 0; i < graphInfo.numGraphOutputs; ++i) {
      auto tw = CreateTensorWrapper(graphInfo.graphOutputs[i]);
      tw->UpdateQnnTensorMeta(graphInfo.graphOutputs[i]);
      int mutable_buffer_id = ExtractMutableBufferNumber(tw->GetName());
      if (mutable_buffer_id != -1 &&
          mutable_buffer_id_to_memory_map.find(mutable_buffer_id) !=
              mutable_buffer_id_to_memory_map.end()) {
        // Fill the same memory for I/O of mutable buffer
        if (MutBufShareEnabled()) {
          BindMutableBuffer(this, mutable_buffer_id, tw);
        } else {
          tw->FillDataBuffer(mutable_buffer_id_to_memory_map[mutable_buffer_id]);
        }
      }
      graph_outputs.push_back(tw);
    }

    ET_CHECK_OR_RETURN_ERROR(
        AllocateTensor(graphInfo.graphName, graph_inputs, graph_outputs) ==
            Error::Ok,
        Internal,
        "Fail to allocate tensor for Dlc with graph_name: %s",
        graphInfo.graphName);
  }

  return Error::Ok;
}

Error QnnManager::Compile(
    const std::string& graph_name,
    std::vector<std::shared_ptr<OpWrapper>>& op_wrappers) {
  Qnn_ErrorHandle_t error = QNN_SUCCESS;
  QnnGraph* qnn_graph_ptr = backend_params_ptr_->qnn_graph_ptr_.get();

  if (IsOnlinePrepare() &&
      qnn_dlc_manager_->backend_params_ptr_->qnn_graph_ptr_.get() != nullptr) {
    qnn_graph_ptr = qnn_dlc_manager_->backend_params_ptr_->qnn_graph_ptr_.get();
  }
  for (std::shared_ptr<OpWrapper>& op_wrapper : op_wrappers) {
    for (const auto& tensor_wrapper : op_wrapper->GetInputTensors()) {
      ET_CHECK_OR_RETURN_ERROR(
          qnn_graph_ptr->EnsureTensorInQnnGraph(graph_name, tensor_wrapper) ==
              Error::Ok,
          Internal,
          "Tensor name %s isn't added to Qnn Graph",
          tensor_wrapper->GetName().c_str());
    }
    for (const auto& tensor_wrapper : op_wrapper->GetOutputTensors()) {
      ET_CHECK_OR_RETURN_ERROR(
          qnn_graph_ptr->EnsureTensorInQnnGraph(graph_name, tensor_wrapper) ==
              Error::Ok,
          Internal,
          "Tensor name %s isn't added to Qnn Graph",
          tensor_wrapper->GetName().c_str());
    }
    for (const auto& param : op_wrapper->GetParams()) {
      auto* p_tensor_param = dynamic_cast<TensorParamWrapper*>(param.get());
      if (p_tensor_param != nullptr) {
        ET_CHECK_OR_RETURN_ERROR(
            qnn_graph_ptr->EnsureTensorInQnnGraph(
                graph_name, p_tensor_param->GetTensorWrapper()) == Error::Ok,
            Internal,
            "Param tensor name %s isn't added to Qnn Graph",
            p_tensor_param->GetName().c_str());
      }
      ET_CHECK_OR_RETURN_ERROR(
          param->PopulateQnnParam() == Error::Ok,
          Internal,
          "Fail to configure Qnn backend");
    }

    error = qnn_graph_ptr->GraphAddNode(graph_name, op_wrapper->GetOpConfig());
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Failed to add node to Qnn Graph with error: %d",
          QNN_GET_ERROR_CODE(error));
      return Error::Internal;
    }
  }
  error = qnn_graph_ptr->GraphFinalize(graph_name);
  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Failed to finalize Qnn Graph with error: %d",
        QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }
  return Error::Ok;
}

} // namespace qnn
} // namespace backends
} // namespace executorch
void* QnnExecuTorchAllocCustomMem(size_t bytes, size_t alignment) {
  void* buffer_ptr =
      executorch::backends::qnn::SharedBuffer::GetSharedBufferManager()
          .AllocMem(bytes, alignment);
  return buffer_ptr;
}

void QnnExecuTorchFreeCustomMem(void* buffer_ptr) {
  executorch::backends::qnn::SharedBuffer::GetSharedBufferManager().FreeMem(
      buffer_ptr);
}

void QnnExecuTorchAddCustomMemTensorAddr(void* tensor_addr, void* custom_mem) {
  executorch::backends::qnn::SharedBuffer::GetSharedBufferManager()
      .AddCusomMemTensorAddr(tensor_addr, custom_mem);
}

size_t QnnExecuTorchResetMutableBuffers(int fill) {
  return executorch::backends::qnn::ResetMutableBuffers(fill);
}
