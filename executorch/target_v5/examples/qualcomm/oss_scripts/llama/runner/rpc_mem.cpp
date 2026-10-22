/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/rpc_mem.h>
#include <executorch/runtime/core/memory_allocator.h>
using executorch::runtime::MemoryAllocator;
using executorch::runtime::TensorInfo;

namespace example {
namespace {
// HTP 의 DMA 는 정렬된 주소를 요구한다. 상류 bump allocator 는 텐서 크기만 누적해서,
// 크기가 정렬 배수가 아닌 텐서(예: ar_len=1 의 input_pos = 4 바이트) 다음부터
// 이후 모든 텐서가 어긋난다. 일반 llama 는 텐서 크기가 대부분 2 의 거듭제곱이라
// 드러나지 않지만, 하이브리드 모델(Qwen3.5)에서는 zero-copy 실행이
// QnnDsp skelExecute err 1100 으로 실패한다.
constexpr size_t kIoAlignment = 128;  // 4096 도 시험했으나(2026-09-08) --shared_buffer err 1100 은 그대로였다
// 정렬 패딩이 들어가는 만큼 총량에 여유를 준다(할당 개수 x 정렬 상한).
constexpr size_t kAlignmentSlack = 1u << 20;  // 1 MB

inline size_t align_up(size_t v, size_t a) {
  return (v + a - 1) & ~(a - 1);
}
} // namespace

RpcMem::RpcMem(
    const size_t total_cache_size,
    const size_t total_prompt_processor_io_size,
    const size_t total_token_generator_io_size)
    : calculated_offsets_(0) {
  size_t total_bytes = total_cache_size + total_prompt_processor_io_size +
      total_token_generator_io_size;
  total_bytes += kAlignmentSlack;
  total_bytes_ = total_bytes;
  ET_LOG(
      Info,
      "RpcMem: cache=%zu prompt_io=%zu token_io=%zu total=%zu bytes (align %zu)",
      total_cache_size,
      total_prompt_processor_io_size,
      total_token_generator_io_size,
      total_bytes,
      kIoAlignment);
  shared_buffer_base_ptr_ = QnnExecuTorchAllocCustomMem(
      total_bytes, MemoryAllocator::kDefaultAlignment);
}
RpcMem::RpcMem(
    const size_t total_cache_size,
    const size_t total_prompt_processor_io_size,
    const size_t total_token_generator_io_size,
    const size_t total_embedding_processor_io_size,
    const size_t total_embedding_generator_io_size)
    : calculated_offsets_(0) {
  size_t total_bytes = total_cache_size + total_prompt_processor_io_size +
      total_token_generator_io_size + total_embedding_processor_io_size +
      total_embedding_generator_io_size;
  shared_buffer_base_ptr_ = QnnExecuTorchAllocCustomMem(
      total_bytes, MemoryAllocator::kDefaultAlignment);
}
RpcMem::~RpcMem() {
  QnnExecuTorchFreeCustomMem(shared_buffer_base_ptr_);
}

std::byte* RpcMem::allocate(size_t data_size) {
  // 각 텐서를 정렬 경계에서 시작시킨다(위 kIoAlignment 주석 참조).
  calculated_offsets_ = align_up(calculated_offsets_, kIoAlignment);
  // The bump allocator has no bounds check upstream: an under-reported total
  // silently hands out pointers past the shared block and segfaults later.
  ET_CHECK_MSG(
      total_bytes_ == 0 || calculated_offsets_ + data_size <= total_bytes_,
      "RpcMem overrun: requested %zu bytes at offset %zu, capacity %zu",
      data_size,
      calculated_offsets_,
      total_bytes_);
  std::byte* data_ptr = static_cast<std::byte*>(shared_buffer_base_ptr_);
  data_ptr += calculated_offsets_;
  // Record the position of the data pointer
  io_pos_map_[data_ptr] = calculated_offsets_;
  calculated_offsets_ += data_size;
  return data_ptr;
}

void RpcMem::add_memory_info(
    void* data_ptr,
    size_t data_size,
    TensorInfo tensor_info) {
  if (auto it = io_pos_map_.find(static_cast<std::byte*>(data_ptr));
      it == io_pos_map_.end()) {
    ET_LOG(Error, "Shared buffer pointer %p is not found", data_ptr);
  }
  if (binded_tensor_addr_set_.find(data_ptr) == binded_tensor_addr_set_.end()) {
    QnnExecuTorchAddCustomMemTensorAddr(data_ptr, shared_buffer_base_ptr_);
    binded_tensor_addr_set_.insert(data_ptr);
  }
};

} // namespace example
