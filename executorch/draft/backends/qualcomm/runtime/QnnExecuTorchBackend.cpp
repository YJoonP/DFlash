/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/aot/wrappers/TensorWrapper.h>
#include <executorch/backends/qualcomm/qc_compiler_spec_generated.h>
#include <executorch/backends/qualcomm/runtime/QnnBackendOptions.h>
#include <executorch/backends/qualcomm/runtime/QnnExecuTorchBackend.h>
#include <executorch/backends/qualcomm/runtime/QnnManager.h>
#include <executorch/backends/qualcomm/runtime/SharedBuffer.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnCustomProtocol.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <executorch/backends/qualcomm/runtime/backends/QnnMemManager.h>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>
namespace executorch {
namespace backends {
namespace qnn {

namespace {
// zero-copy 대형 모델 지원 (2026-09-15 KST).
// QNN_SHM_STAGE_RAW=1 이면 공유 버퍼로 등록되지 않는 delegate 입출력
// (샤드 사이 hidden state, 첫 샤드가 계산해 넘기는 RoPE cos/sin 같은 ExecuTorch 계획 메모리)을
// 백엔드가 가진 rpcmem 스테이징 버퍼에 복사해 memHandle 로 넘긴다.
// 채택 모델에서 RAW 입력이 섞인 실행(두 번째 샤드부터)은 skel 1100 이나 SSR 로 실패했고,
// 입력을 모두 memHandle 로 만든 첫 샤드는 성공했다.
bool StageRawEnabled() {
  static const bool enabled = [] {
    const char* v = std::getenv("QNN_SHM_STAGE_RAW");
    return v != nullptr && std::strcmp(v, "1") == 0;
  }();
  return enabled;
}
std::mutex& StagingMutex() {
  static std::mutex mu;
  return mu;
}
std::unordered_map<const TensorWrapper*, void*>& StagingTable() {
  static std::unordered_map<const TensorWrapper*, void*> table;
  return table;
}
void* GetStaging(const TensorWrapper* tensor) {
  std::lock_guard<std::mutex> lk(StagingMutex());
  auto it = StagingTable().find(tensor);
  if (it != StagingTable().end()) {
    return it->second;
  }
  void* ptr = SharedBuffer::GetSharedBufferManager().AllocMem(
      tensor->GetBytes(), /*alignment=*/4096);
  if (ptr != nullptr) {
    StagingTable().emplace(tensor, ptr);
  }
  return ptr;
}
void FreeStaging(const std::vector<std::shared_ptr<TensorWrapper>>& tensors) {
  std::lock_guard<std::mutex> lk(StagingMutex());
  for (const auto& tensor : tensors) {
    auto it = StagingTable().find(tensor.get());
    if (it != StagingTable().end()) {
      SharedBuffer::GetSharedBufferManager().FreeMem(it->second);
      StagingTable().erase(it);
    }
  }
}
} // namespace

using namespace qnn_delegate;
using executorch::runtime::ArrayRef;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::BackendInitContext;
using executorch::runtime::CompileSpec;
using executorch::runtime::DelegateHandle;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::Result;
using executorch::runtime::Span;

// ========== Public method implementations =========================
constexpr const char* QNN_COMPILE_SPEC = "qnn_compile_spec";
Result<DelegateHandle*> QnnExecuTorchBackend::init(
    BackendInitContext& context,
    FreeableBuffer* processed,
    ArrayRef<CompileSpec> compile_specs) const {
  // covert SizedBuffer to qnn ExecuTorch option
  QnnExecuTorchContextBinary qnn_context_blob;
  const qnn_delegate::QnnExecuTorchOptions* qnn_executorch_options = nullptr;
  auto [status, signature, ctx_size, ctx_bin] =
      QnnContextCustomProtocol().DeserializeContextCustomBuffer(
          const_cast<void*>(processed->data()));
  if (status == Error::Ok) {
    QNN_EXECUTORCH_LOG_INFO(
        "Deserializing processed data using QnnContextCustomProtocol");
    // After this stage, qnn_context_blob.nbytes & qnn_context_blob.buffer will
    // only store qnn_context_binary.
    qnn_context_blob.nbytes = ctx_size;
    qnn_context_blob.buffer = ctx_bin;
  } else {
    // This buffer will be verified again in QnnBackendCache.
    QNN_EXECUTORCH_LOG_INFO("Deserializing processed data using Dlc");
    qnn_context_blob.buffer = const_cast<void*>(processed->data());
    qnn_context_blob.nbytes = processed->size();
  }

  // convert CompileSpec to qnn ExecuTorch option
  for (auto& compile_spec : compile_specs) {
    if (std::strcmp(compile_spec.key, QNN_COMPILE_SPEC) == 0)
      qnn_executorch_options =
          GetQnnExecuTorchOptions(compile_spec.value.buffer);
    else
      QNN_EXECUTORCH_LOG_WARN("unknown argument: %s", compile_spec.key);
  }

  // Create QnnManager
  MemoryAllocator* runtime_allocator = context.get_runtime_allocator();
  QnnManager* qnn_manager = runtime_allocator->allocateInstance<QnnManager>();
  if (qnn_manager == nullptr) {
    return Error::MemoryAllocationFailed;
  }

  // NOTE: Since we use placement new and since this type is not trivially
  // destructible, we must call the destructor manually in destroy().
  new (qnn_manager) QnnManager(qnn_executorch_options, qnn_context_blob);
  // TODO: this is a temporal solution for multi-graph support, will be
  //       removed once framework starts to accept runtime configuration
  // ---
  // check if current context binary has already been initialized
  // return cached one for reducing memory footprint

  auto iter = delegate_map_.find(signature);
  if (iter != delegate_map_.end()) {
    QNN_EXECUTORCH_LOG_INFO(
        "Use cached delegate handle for current method: %s",
        context.get_method_name());
    return iter->second;
  }

  ET_CHECK_OR_RETURN_ERROR(
      qnn_manager->InitBackend() == Error::Ok,
      Internal,
      "Fail to initialize Qnn Manager");
  ET_CHECK_OR_RETURN_ERROR(
      qnn_manager->InitContext() == Error::Ok,
      Internal,
      "Fail to initialize Qnn Manager");

  if (qnn_manager->IsOnlinePrepare()) {
    ET_CHECK_OR_RETURN_ERROR(
        qnn_manager->CompileDlc() == Error::Ok,
        Internal,
        "Fail to compile binary in Dlc format");
  } else {
    for (const std::string& graph_name : qnn_manager->GetGraphNames()) {
      ET_CHECK_OR_RETURN_ERROR(
          qnn_manager->AllocateTensor(graph_name) == Error::Ok,
          Internal,
          "Fail to allocate tensor");
    }
  }
  add_cached_delegate(signature, qnn_manager);

#ifndef __hexagon__
  // This backend does not need its processed data after Init.
  processed->Free();
#endif

  return qnn_manager;
}

Error QnnExecuTorchBackend::execute(
    BackendExecutionContext& context,
    DelegateHandle* handle,
    Span<EValue*> args) const {
  ET_CHECK_OR_RETURN_ERROR(
      delegate_map_rev_.count(handle) != 0,
      Internal,
      "DelegateHandle has been deleted");
  QnnManager* qnn_manager = static_cast<QnnManager*>(handle);

  std::string method_name = context.get_method_name();
  const auto& input_tensors = qnn_manager->GetGraphInputs(method_name);
  const auto& output_tensors = qnn_manager->GetGraphOutputs(method_name);
  std::vector<Qnn_Tensor_t> input_tensor_structs;
  std::vector<Qnn_Tensor_t> output_tensor_structs;

  int args_index = 0;
  input_tensor_structs.reserve(input_tensors.size());
  output_tensor_structs.reserve(output_tensors.size());
  // 진단용 구간 계측: QNN_BACKEND_TIMING=1 (2026-09-15 KST)
  static const bool kTiming = std::getenv("QNN_BACKEND_TIMING") != nullptr;
  using Clock = std::chrono::steady_clock;
  const auto t_begin = kTiming ? Clock::now() : Clock::time_point();
  double reg_us = 0.0;
  auto timed_register = [&](void* ptr, const std::shared_ptr<TensorWrapper>& t) {
    if (!kTiming) {
      return qnn_manager->RegisterMem(ptr, t);
    }
    const auto a = Clock::now();
    Error e = qnn_manager->RegisterMem(ptr, t);
    reg_us += std::chrono::duration<double, std::micro>(Clock::now() - a).count();
    return e;
  };
  // 스테이징 출력: 실행 뒤 (ExecuTorch 버퍼, 스테이징 버퍼, 바이트) 순서로 복사한다.
  std::vector<std::pair<void*, std::pair<void*, size_t>>> staged_outputs;
  const bool stage_raw = StageRawEnabled();
  for (const auto& input_tensor : input_tensors) {
    if (input_tensor->GetName().find("mutbuf_") == std::string::npos) {
      auto& arg_tensor = args[args_index]->toTensor();
      if (timed_register(arg_tensor.mutable_data_ptr(), input_tensor) !=
          Error::Ok) {
        void* staging = stage_raw ? GetStaging(input_tensor.get()) : nullptr;
        if (staging != nullptr &&
            timed_register(staging, input_tensor) == Error::Ok) {
          std::memcpy(
              staging,
              arg_tensor.const_data_ptr(),
              std::min<size_t>(arg_tensor.nbytes(), input_tensor->GetBytes()));
        } else {
          // update data ptr only should be fine
          input_tensor->FillDataBuffer(arg_tensor.const_data_ptr());
          // use the real input shape instead of nominal one to make sure
          // dynamic shape is functional
          auto dims = arg_tensor.sizes();
          input_tensor->SetDims(dims.data(), dims.size());
        }
      }
      args_index++;
    }
    input_tensor_structs.emplace_back(input_tensor->CloneTensorStruct());
  }

  for (const auto& output_tensor : output_tensors) {
    // pos=0 limits the search to the prefix
    if (output_tensor->GetName().rfind("output_", 0) == 0 &&
        output_tensor->GetName().find("mutbuf_") == std::string::npos) {
      auto& arg_tensor = args[args_index]->toTensor();
      void* mutable_data_ptr = arg_tensor.mutable_data_ptr();
      if (timed_register(mutable_data_ptr, output_tensor) !=
          Error::Ok) {
        void* staging = stage_raw ? GetStaging(output_tensor.get()) : nullptr;
        if (staging != nullptr &&
            timed_register(staging, output_tensor) == Error::Ok) {
          staged_outputs.push_back(
              {mutable_data_ptr,
               {staging,
                std::min<size_t>(arg_tensor.nbytes(), output_tensor->GetBytes())}});
        } else {
          output_tensor->FillDataBuffer(mutable_data_ptr);
        }
      }
      args_index++;
    }
    output_tensor_structs.push_back(output_tensor->CloneTensorStruct());
  }

  const auto t_prep = kTiming ? Clock::now() : Clock::time_point();
  ET_CHECK_OR_RETURN_ERROR(
      qnn_manager->Execute(
          method_name,
          input_tensor_structs,
          output_tensor_structs,
          context.event_tracer()) == Error::Ok,
      Internal,
      "Fail to execute graph");
  const auto t_exec = kTiming ? Clock::now() : Clock::time_point();
  for (const auto& copy : staged_outputs) {
    std::memcpy(copy.first, copy.second.first, copy.second.second);
  }
  if (kTiming) {
    // shard(=delegate) 별 실행 시간. 256 호출마다 출력하고 초기화.
    struct ShardAcc { int index; double exec_ms; uint64_t n; };
    static thread_local std::unordered_map<const void*, ShardAcc> s_shard;
    auto sit = s_shard.find(handle);
    if (sit == s_shard.end()) {
      sit = s_shard.emplace(handle, ShardAcc{static_cast<int>(s_shard.size()), 0.0, 0}).first;
    }
    sit->second.exec_ms += std::chrono::duration<double, std::milli>(t_exec - t_prep).count();
    sit->second.n += 1;
    static thread_local double s_prep = 0, s_reg = 0, s_exec = 0, s_post = 0;
    static thread_local uint64_t s_calls = 0;
    const auto t_end = Clock::now();
    s_prep += std::chrono::duration<double, std::milli>(t_prep - t_begin).count();
    s_reg += reg_us / 1000.0;
    s_exec += std::chrono::duration<double, std::milli>(t_exec - t_prep).count();
    s_post += std::chrono::duration<double, std::milli>(t_end - t_exec).count();
    if (++s_calls % 256 == 0) {
      QNN_EXECUTORCH_LOG_WARN(
          "[BK_TIMING] calls %llu prep %.3f (register %.3f) exec %.3f post %.3f ms/call, mem_register total %llu",
          static_cast<unsigned long long>(s_calls), s_prep / s_calls, s_reg / s_calls,
          s_exec / s_calls, s_post / s_calls,
          static_cast<unsigned long long>(QnnMemRegisterCallCount()));
      char line[512];
      int off = 0;
      std::vector<std::pair<int, double>> shards;
      for (auto& kv : s_shard) {
        if (kv.second.n > 0) {
          shards.push_back({kv.second.index, kv.second.exec_ms / kv.second.n});
        }
        kv.second.exec_ms = 0;
        kv.second.n = 0;
      }
      std::sort(shards.begin(), shards.end());
      for (auto& sh : shards) {
        off += std::snprintf(line + off, sizeof(line) - off, " %d:%.3f", sh.first, sh.second);
        if (off >= static_cast<int>(sizeof(line)) - 16) break;
      }
      QNN_EXECUTORCH_LOG_WARN("[BK_SHARD]%s", line);
      s_prep = s_reg = s_exec = s_post = 0;
      s_calls = 0;
    }
  }
  ET_CHECK_OR_RETURN_ERROR(
      qnn_manager->ProfileExecuteData(method_name, context.event_tracer()) ==
          Error::Ok,
      Internal,
      "Fail to profile graph");

  return Error::Ok;
}

void QnnExecuTorchBackend::destroy(DelegateHandle* handle) const {
  if (handle != nullptr && delegate_map_rev_.count(handle)) {
    QnnManager* qnn_manager = static_cast<QnnManager*>(handle);
    std::vector<std::shared_ptr<TensorWrapper>> staged;
    if (StageRawEnabled()) {
      for (const std::string& graph_name : qnn_manager->GetGraphNames()) {
        const auto& inputs = qnn_manager->GetGraphInputs(graph_name);
        const auto& outputs = qnn_manager->GetGraphOutputs(graph_name);
        staged.insert(staged.end(), inputs.begin(), inputs.end());
        staged.insert(staged.end(), outputs.begin(), outputs.end());
      }
    }
    qnn_manager->Destroy();
    // memHandle 은 Destroy 에서 해제되므로 그 뒤에 스테이징 메모리를 푼다.
    FreeStaging(staged);
    erase_cached_delegate(handle);
  }
}

executorch::runtime::Error QnnExecuTorchBackend::set_option(
    executorch::runtime::BackendOptionContext& context,
    const executorch::runtime::Span<executorch::runtime::BackendOption>&
        backend_options) {
  std::lock_guard<std::mutex> guard(runtime_option_mutex_);
  size_t matches = backend_options.size();
  for (const auto& option : backend_options) {
    if (strcmp(option.key, QNN_RUNTIME_LOG_LEVEL) == 0) {
      if (auto* val = std::get_if<int>(&option.value)) {
        qnn_runtime_log_level_.value = *val;
        qnn_runtime_log_level_.is_set = true;
      }
    } else if (strcmp(option.key, QNN_RUNTIME_HTP_PERFORMANCE_MODE) == 0) {
      if (auto* val = std::get_if<int>(&option.value)) {
        qnn_runtime_performance_mode_.value = *val;
        qnn_runtime_performance_mode_.is_set = true;
      }
    } else if (strcmp(option.key, QNN_RUNTIME_PROFILE_LEVEL) == 0) {
      if (auto* val = std::get_if<int>(&option.value)) {
        qnn_runtime_profile_level_.value = *val;
        qnn_runtime_profile_level_.is_set = true;
      }
    } else if (strcmp(option.key, QNN_RUNTIME_LPAI_FPS) == 0) {
      if (auto* val = std::get_if<int>(&option.value)) {
        qnn_runtime_lpai_fps_.value = *val;
        qnn_runtime_lpai_fps_.is_set = true;
      }
    } else if (strcmp(option.key, QNN_RUNTIME_LPAI_FTRT_RATIO) == 0) {
      if (auto* val = std::get_if<int>(&option.value)) {
        qnn_runtime_lpai_ftrt_ratio_.value = *val;
        qnn_runtime_lpai_ftrt_ratio_.is_set = true;
      }
    } else if (strcmp(option.key, QNN_RUNTIME_LPAI_CLIENT_PERF_TYPE) == 0) {
      if (auto* val = std::get_if<int>(&option.value)) {
        qnn_runtime_lpai_client_perf_type_.value = *val;
        qnn_runtime_lpai_client_perf_type_.is_set = true;
      }
    } else if (strcmp(option.key, QNN_RUNTIME_LPAI_AFFINITY) == 0) {
      if (auto* val = std::get_if<int>(&option.value)) {
        qnn_runtime_lpai_affinity_.value = *val;
        qnn_runtime_lpai_affinity_.is_set = true;
      }
    } else if (strcmp(option.key, QNN_RUNTIME_LPAI_CORE_SELECTION) == 0) {
      if (auto* val = std::get_if<int>(&option.value)) {
        qnn_runtime_lpai_core_selection_.value = *val;
        qnn_runtime_lpai_core_selection_.is_set = true;
      }
    } else if (strcmp(option.key, QNN_RUNTIME_HEAP_PROFILING_PATH) == 0) {
      if (auto* val =
              std::get_if<std::array<char, runtime::kMaxOptionValueLength>>(
                  &option.value)) {
        qnn_runtime_heap_profiling_path_.value = *val;
        qnn_runtime_heap_profiling_path_.is_set = true;
      }
    } else {
      ET_LOG(
          Error,
          "Unable to set the following runtime option for QnnExecuTorchBackend: %s.",
          option.key);
      matches--;
    }
  }

  ET_CHECK_OR_RETURN_ERROR(
      matches == backend_options.size(),
      Internal,
      "Some set options are not supported by QnnExecuTorchBackend. %zu options provided but only %zu is supported.",
      backend_options.size(),
      matches);

  return Error::Ok;
}

executorch::runtime::Error QnnExecuTorchBackend::get_option(
    executorch::runtime::BackendOptionContext& context,
    executorch::runtime::Span<executorch::runtime::BackendOption>&
        backend_options) {
  std::lock_guard<std::mutex> guard(runtime_option_mutex_);
  size_t matches = backend_options.size();
  for (size_t i = 0; i < backend_options.size(); ++i) {
    // Set the value to what was stored by set_option
    if (strcmp(backend_options[i].key, QNN_RUNTIME_LOG_LEVEL) == 0 &&
        qnn_runtime_log_level_.is_set) {
      backend_options[i].value = qnn_runtime_log_level_.value;
    } else if (
        strcmp(backend_options[i].key, QNN_RUNTIME_HTP_PERFORMANCE_MODE) == 0 &&
        qnn_runtime_performance_mode_.is_set) {
      backend_options[i].value = qnn_runtime_performance_mode_.value;
    } else if (
        strcmp(backend_options[i].key, QNN_RUNTIME_PROFILE_LEVEL) == 0 &&
        qnn_runtime_profile_level_.is_set) {
      backend_options[i].value = qnn_runtime_profile_level_.value;
    } else if (
        strcmp(backend_options[i].key, QNN_RUNTIME_LPAI_FPS) == 0 &&
        qnn_runtime_lpai_fps_.is_set) {
      backend_options[i].value = qnn_runtime_lpai_fps_.value;
    } else if (
        strcmp(backend_options[i].key, QNN_RUNTIME_LPAI_FTRT_RATIO) == 0 &&
        qnn_runtime_lpai_ftrt_ratio_.is_set) {
      backend_options[i].value = qnn_runtime_lpai_ftrt_ratio_.value;
    } else if (
        strcmp(backend_options[i].key, QNN_RUNTIME_LPAI_CLIENT_PERF_TYPE) ==
            0 &&
        qnn_runtime_lpai_client_perf_type_.is_set) {
      backend_options[i].value = qnn_runtime_lpai_client_perf_type_.value;
    } else if (
        strcmp(backend_options[i].key, QNN_RUNTIME_LPAI_AFFINITY) == 0 &&
        qnn_runtime_lpai_affinity_.is_set) {
      backend_options[i].value = qnn_runtime_lpai_affinity_.value;
    } else if (
        strcmp(backend_options[i].key, QNN_RUNTIME_LPAI_CORE_SELECTION) == 0 &&
        qnn_runtime_lpai_core_selection_.is_set) {
      backend_options[i].value = qnn_runtime_lpai_core_selection_.value;
    } else if (
        strcmp(backend_options[i].key, QNN_RUNTIME_HEAP_PROFILING_PATH) == 0 &&
        qnn_runtime_heap_profiling_path_.is_set) {
      backend_options[i].value = qnn_runtime_heap_profiling_path_.value;
    } else {
      // either runtime never called set_option or key does not exist
      matches--;
    }
  }

  if (matches != backend_options.size()) {
    return Error::Internal;
  }
  return Error::Ok;
}

bool QnnExecuTorchBackend::is_available() const {
  return true;
}

void QnnExecuTorchBackend::add_cached_delegate(
    const std::int64_t& signature,
    executorch::runtime::DelegateHandle* handle) const {
  std::lock_guard<std::mutex> guard(mutex_);
  delegate_map_[signature] = handle;
  delegate_map_rev_[handle] = signature;
}

void QnnExecuTorchBackend::erase_cached_delegate(
    executorch::runtime::DelegateHandle* handle) const {
  std::lock_guard<std::mutex> guard(mutex_);
  auto iter = delegate_map_rev_.find(handle);
  if (iter == delegate_map_rev_.end()) {
    return;
  }
  delegate_map_.erase(iter->second);
  delegate_map_rev_.erase(handle);
}

namespace {
#if !defined(_MSC_VER)
auto cls = QnnExecuTorchBackend();
executorch::runtime::Backend backend{QNN_BACKEND, &cls};
// On non-Windows platforms a single copy of executorch_core is shared between
// the DLL and the exe, so the static initializer writes into the correct
// registered_backends[] array.
static auto success_with_compiler = register_backend(backend);
#endif
} // namespace

#if defined(_MSC_VER)
static QnnExecuTorchBackend cls_msvc;
static executorch::runtime::Backend backend_msvc{QNN_BACKEND, &cls_msvc};
#endif

} // namespace qnn
} // namespace backends
} // namespace executorch

#if defined(_MSC_VER)
// On Windows the DLL and the exe each carry a private copy of executorch_core
// globals. The runner calls this function at startup, passing its own
// register_backend() so registration happens in the exe's address space.
// See QnnExecuTorch.h for the required register_fn signature.
void QnnExecuTorchBackendRegister(void* register_fn) {
  using RegisterFn =
      executorch::runtime::Error (*)(const executorch::runtime::Backend&);
  if (register_fn == nullptr) {
    ET_LOG(Error, "QnnExecuTorchBackendRegister called with null register_fn");
    return;
  }
  executorch::runtime::Error err = reinterpret_cast<RegisterFn>(register_fn)(
      executorch::backends::qnn::backend_msvc);
  if (err != executorch::runtime::Error::Ok) {
    ET_LOG(
        Error,
        "Failed to register QNN backend: 0x%x",
        static_cast<uint32_t>(err));
  }
}
#else
void QnnExecuTorchBackendRegister(void*) {
  // No-op: ELF symbol interposition already shares a single backend registry
  // between the .so and the exe, so the static initializer above suffices.
}
#endif
