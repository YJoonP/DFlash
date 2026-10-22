/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/backends/QnnMemManager.h>

#include <atomic>
#include <memory>
#include <mutex>

namespace executorch {
namespace backends {
namespace qnn {

using executorch::runtime::Error;

namespace {
std::atomic<uint64_t> g_mem_register_calls{0};
} // namespace
uint64_t QnnMemRegisterCallCount() {
  return g_mem_register_calls.load();
}

bool QnnMemManager::IsRegistered(Qnn_MemHandle_t handle, void* mem_ptr) {
  auto it = registered_map_.find(handle);
  if (it != registered_map_.end()) {
    return it->second == mem_ptr;
  }
  return false;
}

Error QnnMemManager::RegisterIonMem(
    const std::shared_ptr<TensorWrapper>& tensor_wrapper,
    int32_t mem_fd,
    void* mem_ptr) {
  const QnnInterface& qnn_interface = implementation_->GetQnnInterface();
  Qnn_MemDescriptor_t descriptor = {
      {tensor_wrapper->GetRank(), tensor_wrapper->GetDims(), nullptr},
      tensor_wrapper->GetDataType(),
      QNN_MEM_TYPE_ION,
      {{mem_fd}}};
  Qnn_MemHandle_t handle = nullptr;
  Qnn_ErrorHandle_t error = QNN_SUCCESS;
  g_mem_register_calls.fetch_add(1);
  error = qnn_interface.qnn_mem_register(
      context_->GetHandle(),
      &descriptor,
      /*numDescriptors=*/1,
      &handle);
  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_WARN(
        "Tensor %s is failed to register shared memory. Error %d",
        tensor_wrapper->GetName().c_str(),
        QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }
  tensor_wrapper->SetMemHandle(handle);
  registered_map_.insert({handle, mem_ptr});
  if (log_level_ >= QnnExecuTorchLogLevel::kLogLevelInfo) {
    QNN_EXECUTORCH_LOG_INFO(
        "Tensor %s is successfully registered to ION shared memory.",
        tensor_wrapper->GetName().c_str());
  }

  return Error::Ok;
}

Error QnnMemManager::RegisterCustomMem(
    const std::shared_ptr<TensorWrapper>& tensor_wrapper,
    int32_t mem_fd,
    void* mem_ptr,
    size_t total_custom_mem_size,
    size_t tensor_offset,
    const CustomMemTensorInfo& info) {
  const QnnInterface& qnn_interface = implementation_->GetQnnInterface();
  Qnn_MemDescriptor_t descriptor = {
      {tensor_wrapper->GetRank(), tensor_wrapper->GetDims(), nullptr},
      tensor_wrapper->GetDataType(),
      QNN_MEM_TYPE_CUSTOM,
      {{mem_fd}}};
  Qnn_MemHandle_t handle = nullptr;
  Qnn_ErrorHandle_t error = QNN_SUCCESS;

  QnnMemHtp_Descriptor_t htp_descriptor;
  htp_descriptor.type = QNN_HTP_MEM_SHARED_BUFFER;
  htp_descriptor.size = total_custom_mem_size;

  QnnHtpMem_SharedBufferConfig_t htpSharedBuffConfig = {mem_fd, tensor_offset};
  htp_descriptor.sharedBufferConfig = htpSharedBuffConfig;

  descriptor.customInfo = &htp_descriptor;

  g_mem_register_calls.fetch_add(1);
  error = qnn_interface.qnn_mem_register(
      context_->GetHandle(),
      &descriptor,
      /*numDescriptors=*/1,
      &handle);
  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_WARN(
        "Tensor %s is failed to register shared memory. Error %d",
        tensor_wrapper->GetName().c_str(),
        QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }
  tensor_wrapper->SetMemHandle(handle);
  pre_registered_handles_.insert({info, handle});
  registered_map_.insert({handle, mem_ptr});
  if (log_level_ >= QnnExecuTorchLogLevel::kLogLevelInfo) {
    QNN_EXECUTORCH_LOG_INFO(
        "Tensor %s is successfully registered to custom shared memory.",
        tensor_wrapper->GetName().c_str());
  }
  return Error::Ok;
}

namespace {
struct VaReservation {
  void* ptr{nullptr};
  Qnn_MemHandle_t handle{nullptr};
  uint32_t dims[1]{0};
};
std::mutex& VaReservationMutex() {
  static std::mutex mu;
  return mu;
}
std::unordered_map<const QnnMemManager*, std::unique_ptr<VaReservation>>& VaReservations() {
  static std::unordered_map<const QnnMemManager*, std::unique_ptr<VaReservation>> table;
  return table;
}
} // namespace

Error QnnMemManager::ReserveVa(size_t bytes) {
  if (bytes == 0) {
    return Error::Ok;
  }
  std::lock_guard<std::mutex> lk(VaReservationMutex());
  if (VaReservations().count(this) != 0) {
    return Error::Ok;
  }
  SharedBuffer& shared_buffer = SharedBuffer::GetSharedBufferManager();
  constexpr size_t kPage = 4096;
  bytes = (bytes + kPage - 1) / kPage * kPage;
  void* ptr = shared_buffer.AllocMem(bytes, kPage);
  if (ptr == nullptr) {
    QNN_EXECUTORCH_LOG_WARN("VA reservation: rpcmem alloc of %zu bytes failed", bytes);
    return Error::Internal;
  }
  int32_t fd = shared_buffer.MemToFd(ptr);
  if (fd == -1) {
    QNN_EXECUTORCH_LOG_WARN("VA reservation: rpcmem_to_fd failed");
    shared_buffer.FreeMem(ptr);
    return Error::Internal;
  }
  auto reservation = std::make_unique<VaReservation>();
  reservation->dims[0] = static_cast<uint32_t>(bytes);
  Qnn_MemDescriptor_t descriptor = {
      {1, reservation->dims, nullptr},
      QNN_DATATYPE_UINT_8,
      QNN_MEM_TYPE_ION,
      {{fd}}};
  Qnn_MemHandle_t handle = nullptr;
  const QnnInterface& qnn_interface = implementation_->GetQnnInterface();
  Qnn_ErrorHandle_t error = qnn_interface.qnn_mem_register(
      context_->GetHandle(), &descriptor, /*numDescriptors=*/1, &handle);
  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_WARN(
        "VA reservation: register %zu bytes failed. Error %d",
        bytes,
        QNN_GET_ERROR_CODE(error));
    shared_buffer.FreeMem(ptr);
    return Error::Internal;
  }
  reservation->ptr = ptr;
  reservation->handle = handle;
  VaReservations().emplace(this, std::move(reservation));
  QNN_EXECUTORCH_LOG_WARN("VA reservation: %zu bytes mapped for context I/O", bytes);
  return Error::Ok;
}

void QnnMemManager::ReleaseVaReservation() {
  std::unique_ptr<VaReservation> reservation;
  {
    std::lock_guard<std::mutex> lk(VaReservationMutex());
    auto it = VaReservations().find(this);
    if (it == VaReservations().end()) {
      return;
    }
    reservation = std::move(it->second);
    VaReservations().erase(it);
  }
  const QnnInterface& qnn_interface = implementation_->GetQnnInterface();
  Qnn_ErrorHandle_t error =
      qnn_interface.qnn_mem_de_register(&reservation->handle, /*numHandles=*/1);
  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_WARN(
        "VA reservation: de-register failed. Error %d", QNN_GET_ERROR_CODE(error));
  }
  SharedBuffer::GetSharedBufferManager().FreeMem(reservation->ptr);
}

void* QnnMemManager::GetPreRegisteredHandle(const CustomMemTensorInfo& info) {
  auto it = pre_registered_handles_.find(info);
  if (it == pre_registered_handles_.end()) {
    return nullptr;
  }
  return it->second;
}

Error QnnMemManager::SetMemHandle(
    const std::shared_ptr<TensorWrapper>& tensor_wrapper,
    void* mem_ptr,
    Qnn_MemHandle_t handle) {
  tensor_wrapper->SetMemHandle(handle);
  registered_map_.insert({handle, mem_ptr});
  return Error::Ok;
}

void QnnMemManager::DeRegisterMem() {
  const QnnInterface& qnn_interface = implementation_->GetQnnInterface();
  Qnn_ErrorHandle_t error = QNN_SUCCESS;

  for (auto& it : registered_map_) {
    error = qnn_interface.qnn_mem_de_register(&it.first, /*numHandles=*/1);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_WARN(
          "Failed to de-register shared memory. Error %d",
          QNN_GET_ERROR_CODE(error));
    }
  }
  registered_map_.clear();
}

} // namespace qnn
} // namespace backends
} // namespace executorch
