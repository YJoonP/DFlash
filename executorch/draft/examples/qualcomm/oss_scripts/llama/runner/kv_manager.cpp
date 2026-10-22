/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/kv_manager.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/assert.h>
#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <numeric>
#include <thread>
#include <vector>

using executorch::runtime::MethodMeta;
using executorch::runtime::Result;
using executorch::runtime::TensorInfo;
namespace example {
namespace {
// 2026-09-15 KST: 상태 캐시 복사(출력 버퍼 -> 입력 버퍼)를 여러 스레드로 나눈다.
// zero-copy 에서 QNN 은 상태 입력/출력 버퍼가 매 토큰 같고 서로 다를 때 가장 빠르다(ping-pong·in-place 는 느림).
// 그 경우 남는 비용이 상태 약 26MB 복사(단일 스레드 약 1.1ms/토큰)라서 이를 병렬화한다.
// QNN_STATE_COPY_THREADS=N (기본 1 = 기존 단일 memcpy). Run 은 모든 복사가 끝난 뒤 반환한다.
class StateCopyPool {
 public:
  struct Job {
    void* dst;
    const void* src;
    size_t n;
  };
  static StateCopyPool* Get() {
    static StateCopyPool* pool = []() -> StateCopyPool* {
      const char* v = std::getenv("QNN_STATE_COPY_THREADS");
      int threads = v ? std::atoi(v) : 1;
      if (threads <= 1) return nullptr;
      return new StateCopyPool(std::min(threads, 8));
    }();
    return pool;
  }
  void Run(const std::vector<Job>& jobs) {
    const size_t bins = workers_.size() + 1;
    for (auto& b : bins_) b.clear();
    std::vector<size_t> load(bins, 0);
    std::vector<size_t> order(jobs.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return jobs[a].n > jobs[b].n; });
    for (size_t idx : order) {
      size_t best = std::min_element(load.begin(), load.end()) - load.begin();
      bins_[best].push_back(jobs[idx]);
      load[best] += jobs[idx].n;
    }
    {
      std::lock_guard<std::mutex> lk(mu_);
      pending_ = workers_.size();
      ++generation_;
    }
    cv_.notify_all();
    for (const auto& j : bins_[0]) std::memcpy(j.dst, j.src, j.n);
    std::unique_lock<std::mutex> lk(mu_);
    done_cv_.wait(lk, [&] { return pending_ == 0; });
  }

 private:
  explicit StateCopyPool(int threads) : bins_(threads) {
    for (int i = 1; i < threads; ++i) {
      workers_.emplace_back([this, i] { Loop(i); });
    }
    ET_LOG(Info, "state copy pool: %d threads", threads);
  }
  void Loop(int bin) {
    uint64_t seen = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return generation_ != seen; });
        seen = generation_;
      }
      for (const auto& j : bins_[bin]) std::memcpy(j.dst, j.src, j.n);
      {
        std::lock_guard<std::mutex> lk(mu_);
        --pending_;
      }
      done_cv_.notify_all();
    }
  }
  std::vector<std::vector<Job>> bins_;
  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable cv_, done_cv_;
  uint64_t generation_{0};
  size_t pending_{0};
};
} // namespace


namespace {
// Graph inputs end with the cache slots (k_0..k_{n-1}, v_0..v_{n-1}), so their
// start index follows from the input count and holds whether or not the model
// carries an extra window-attention-mask input.

// TensorInfo::sizes() returns a Span, which has no operator==.
template <typename SpanT>
bool same_shape(const SpanT& a, const SpanT& b) {
  if (a.size() != b.size())
    return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i])
      return false;
  }
  return true;
}

void fill_mask(
    executorch::aten::ScalarType scalar_type,
    std::byte* buf,
    size_t size,
    bool use_pos_value) {
  if (use_pos_value) {
    switch (scalar_type) {
      case executorch::aten::ScalarType::UInt16:
        std::fill_n(reinterpret_cast<uint16_t*>(buf), size, 65535u);
        break;
      case executorch::aten::ScalarType::Byte:
        std::fill_n(reinterpret_cast<uint8_t*>(buf), size, 255u);
        break;
      case executorch::aten::ScalarType::Float:
        std::fill_n(reinterpret_cast<float*>(buf), size, 0.0);
        break;
      default:
        ET_CHECK_MSG(
            false,
            "Unsupported scalar type %s",
            executorch::runtime::toString(scalar_type));
        break;
    }
  } else {
    switch (scalar_type) {
      case executorch::aten::ScalarType::UInt16:
        std::fill_n(reinterpret_cast<uint16_t*>(buf), size, 0u);
        break;
      case executorch::aten::ScalarType::Byte:
        std::fill_n(reinterpret_cast<uint8_t*>(buf), size, 0u);
        break;
      // -65535 acts as the additive "very negative" attention-mask value;
      // chosen as a large finite negative so masked positions effectively
      // zero out after softmax without relying on -inf.
      case executorch::aten::ScalarType::Float:
        std::fill_n(reinterpret_cast<float*>(buf), size, -65535.0);
        break;
      default:
        ET_CHECK_MSG(
            false,
            "Unsupported scalar type %s",
            executorch::runtime::toString(scalar_type));
        break;
    }
  }
}
} // namespace

KVManager::KVManager(Metadata metadata, std::unique_ptr<MethodMeta> method_meta)
    : metadata_(metadata) {
  Result<TensorInfo> attention_mask = method_meta->input_tensor_meta(1);
  attention_mask_dtype_ = attention_mask->scalar_type();

  // inputs: [tokens, atten_mask, (window_atten_mask), (input_pos), k_caches...,
  // v_caches...].
  // outputs: [logits, k_cache_0..n-1, v_cache_0..n-1]
  // k_cache shape: [bsz, n_kv_head, head_dim, ar_len]
  // v_cache v_cache shape: [bsz, n_kv_head, ar_len, head_dim]
  k_cache_.resize(metadata_.num_layers);
  v_cache_.resize(metadata_.num_layers);
  k_cache_in_nbytes_.resize(metadata_.num_layers);
  k_cache_out_nbytes_.resize(metadata_.num_layers);
  v_cache_in_nbytes_.resize(metadata_.num_layers);
  v_cache_out_nbytes_.resize(metadata_.num_layers);
  is_state_layer_.assign(metadata_.num_layers, false);
  state_dtype_.assign(
      metadata_.num_layers, executorch::aten::ScalarType::Undefined);
  if (state_k_zero_point_.empty()) {
    state_k_zero_point_.assign(metadata_.num_layers, -1);
    state_v_zero_point_.assign(metadata_.num_layers, -1);
  }
  const size_t first_cache_input_idx =
      method_meta->num_inputs() - 2 * metadata_.num_layers;

  for (int layer = 0; layer < metadata_.num_layers; ++layer) {
    // k output: index 1 + layer
    Result<TensorInfo> k_out = method_meta->output_tensor_meta(1 + layer);
    // v output: index 1 + num_layers + layer
    Result<TensorInfo> v_out =
        method_meta->output_tensor_meta(1 + metadata_.num_layers + layer);

    // Recurrent-state slots (Qwen3.5 Gated DeltaNet) are detected from the
    // graph signature alone, so no extra .pte metadata is required: a KV cache
    // shrinks/grows between input (max_cache_len) and output (ar_len), while a
    // state slot is returned with exactly the same shape it was given.
    // Detected before the dtype checks below: state slots may carry a different
    // scalar type than the (8-bit) KV caches, which is expected, not an error.
    Result<TensorInfo> k_in =
        method_meta->input_tensor_meta(first_cache_input_idx + layer);
    Result<TensorInfo> v_in = method_meta->input_tensor_meta(
        first_cache_input_idx + metadata_.num_layers + layer);
    const bool same_k = same_shape(k_in->sizes(), k_out->sizes());
    const bool same_v = same_shape(v_in->sizes(), v_out->sizes());
    if (same_k && same_v && metadata_.max_cache_len != metadata_.max_ar_len) {
      is_state_layer_[layer] = true;
      has_state_layer_ = true;
      state_dtype_[layer] = k_in->scalar_type();
      k_head_dim_.push_back(0);
      v_head_dim_.push_back(0);
      // whole-buffer slots: input and output are the same size
      k_cache_in_nbytes_[layer] = k_in->nbytes();
      k_cache_out_nbytes_[layer] = k_out->nbytes();
      v_cache_in_nbytes_[layer] = v_in->nbytes();
      v_cache_out_nbytes_[layer] = v_out->nbytes();
      continue;
    }

    if (kv_cache_dtype_ == executorch::aten::ScalarType::Undefined) {
      kv_cache_dtype_ = k_out->scalar_type();
    }
    ET_CHECK_MSG(
        k_out->scalar_type() == kv_cache_dtype_,
        "Mixed scalar type of kv_cache is not allowed (k layer %d)",
        layer);
    ET_CHECK_MSG(
        v_out->scalar_type() == kv_cache_dtype_,
        "Mixed scalar type of kv_cache is not allowed (v layer %d)",
        layer);

    // k_cache shape: [1, n_kv, head_dim, ar_len] -> head_dim at dim 2
    // v_cache shape: [1, n_kv, ar_len, head_dim] -> head_dim at dim 3
    const int64_t k_head_dim = k_out->sizes()[2];
    const int64_t v_head_dim = v_out->sizes()[3];
    ET_CHECK_MSG(
        k_head_dim == v_head_dim,
        "K/V head_dim mismatch at layer %d (k %d, v %d). Expected output order [logits, k_0..n-1, v_0..n-1] with k [1, n_kv, head_dim, ar_len] and v [1, n_kv, ar_len, head_dim]",
        layer,
        static_cast<int>(k_head_dim),
        static_cast<int>(v_head_dim));
    k_head_dim_.push_back(k_head_dim);
    v_head_dim_.push_back(v_head_dim);

    const size_t dtype_size = getDtypeSize(kv_cache_dtype_);
    k_cache_out_nbytes_[layer] =
        metadata_.num_heads * k_head_dim * metadata_.max_ar_len * dtype_size;
    v_cache_out_nbytes_[layer] =
        metadata_.num_heads * v_head_dim * metadata_.max_ar_len * dtype_size;
    k_cache_in_nbytes_[layer] =
        metadata_.num_heads * k_head_dim * metadata_.max_cache_len * dtype_size;
    v_cache_in_nbytes_[layer] =
        metadata_.num_heads * v_head_dim * metadata_.max_cache_len * dtype_size;
  }

  ET_CHECK_MSG(
      kv_cache_dtype_ != executorch::aten::ScalarType::Undefined,
      "kv_cache_dtype was not detected from method outputs");

  // Total cache size across all layers (K + V, input + output)
  total_cache_size_ = 0;
  for (int layer = 0; layer < metadata_.num_layers; ++layer) {
    total_cache_size_ += k_cache_in_nbytes_[layer] +
        k_cache_out_nbytes_[layer] + v_cache_in_nbytes_[layer] +
        v_cache_out_nbytes_[layer];
  }
};

void KVManager::init_attention_mask(
    std::byte* attention_mask,
    const std::vector<int32_t>& attention_map,
    int32_t ar_len,
    int32_t n_past) {
  ET_CHECK_MSG(
      attention_map.size() <= ar_len,
      "The size of attention_map (%zu) doesn't match with ar_len (%d)",
      attention_map.size(),
      ar_len);
  // Clear the attention mask
  fill_mask(
      attention_mask_dtype_,
      attention_mask,
      ar_len * metadata_.context_len,
      /*use_pos_value=*/false);

  // SMART_MASK requires special handling of attention mask
  std::byte* past_ptr = attention_mask;
  std::byte* new_ptr = attention_mask +
      (metadata_.context_len - ar_len) * getDtypeSize(attention_mask_dtype_);
  // All inputs will necessarily attend to n_past and itself
  for (int i = 0; i < ar_len; i++) {
    // Iterate across ar_len
    if (attention_map[i] < 0) {
      // If negative, attend to only past tokens
      fill_mask(
          attention_mask_dtype_,
          past_ptr,
          n_past,
          /*use_pos_value=*/true);
    } else {
      // If positive, copy attention map from (relative to 0th input) parent
      // Parent token index
      const int32_t pidx = attention_map[i];
      std::byte* parent_ptr = attention_mask +
          pidx * metadata_.context_len * getDtypeSize(attention_mask_dtype_);
      std::memcpy(
          past_ptr,
          parent_ptr,
          metadata_.context_len * getDtypeSize(attention_mask_dtype_));
    }
    // Attend to itself
    fill_mask(
        attention_mask_dtype_,
        new_ptr + i * getDtypeSize(attention_mask_dtype_),
        1,
        /*use_pos_value=*/true);
    past_ptr += metadata_.context_len * getDtypeSize(attention_mask_dtype_);
    new_ptr += metadata_.context_len * getDtypeSize(attention_mask_dtype_);
  }
}

void KVManager::init_attention_mask(
    std::byte* attention_mask,
    const std::vector<int32_t>& attention_map,
    int32_t ar_len,
    int32_t n_past,
    int32_t sliding_window,
    const std::vector<int32_t>& position_offset) {
  ET_CHECK_MSG(
      attention_map.size() <= ar_len,
      "The size of attention_map (%zu) doesn't match with ar_len (%d)",
      attention_map.size(),
      ar_len);
  // Clear the attention mask
  fill_mask(
      attention_mask_dtype_,
      attention_mask,
      ar_len * metadata_.context_len,
      /*use_pos_value=*/false);

  // SMART_MASK requires special handling of attention mask
  std::byte* past_ptr = attention_mask;
  std::byte* new_ptr = attention_mask +
      (metadata_.context_len - ar_len) * getDtypeSize(attention_mask_dtype_);
  // All inputs will necessarily attend to n_past and itself
  for (int i = 0; i < ar_len; i++) {
    // Iterate across ar_len
    if (attention_map[i] < 0) {
      // If negative, attend to only past tokens
      fill_mask(
          attention_mask_dtype_,
          past_ptr,
          n_past,
          /*use_pos_value=*/true);
    } else {
      // If positive, copy attention map from (relative to 0th input) parent
      // Parent token index
      const int32_t pidx = attention_map[i];
      std::byte* parent_ptr = attention_mask +
          pidx * metadata_.context_len * getDtypeSize(attention_mask_dtype_);
      std::memcpy(
          past_ptr,
          parent_ptr,
          metadata_.context_len * getDtypeSize(attention_mask_dtype_));
    }
    // Attend to itself
    fill_mask(
        attention_mask_dtype_,
        new_ptr + i * getDtypeSize(attention_mask_dtype_),
        1,
        /*use_pos_value=*/true);

    // mask by limitation of sliding_window
    int32_t available_context_len = position_offset.empty()
        ? sliding_window - (i + 1) - n_past
        : sliding_window - (position_offset[i] + 1) - n_past;
    // if available_context_len is less than 0, it means we need to mask some
    // tokens in the past to avoid exceeding the sliding window
    if (available_context_len < 0) {
      fill_mask(
          attention_mask_dtype_,
          past_ptr,
          -available_context_len,
          /*use_pos_value=*/false);
    }

    past_ptr += metadata_.context_len * getDtypeSize(attention_mask_dtype_);
    new_ptr += metadata_.context_len * getDtypeSize(attention_mask_dtype_);
  }
}

void KVManager::update_attention_mask(
    std::byte* attention_mask,
    int32_t ar_len,
    int32_t n_past,
    int32_t n_update) {
  std::byte* cur_ptr =
      attention_mask + n_past * getDtypeSize(attention_mask_dtype_);

  for (int i = 0; i < ar_len; i++) {
    fill_mask(attention_mask_dtype_, cur_ptr, n_update, /*use_pos_value=*/true);
    cur_ptr += metadata_.context_len * getDtypeSize(attention_mask_dtype_);
  }
}

void KVManager::update_attention_mask(
    std::byte* attention_mask,
    int32_t ar_len,
    int32_t n_past,
    int32_t n_update,
    int32_t sliding_window,
    const std::vector<int32_t>& position_offset) {
  std::byte* cur_ptr =
      attention_mask + n_past * getDtypeSize(attention_mask_dtype_);

  for (int i = 0; i < ar_len; i++) {
    fill_mask(attention_mask_dtype_, cur_ptr, n_update, /*use_pos_value=*/true);
    int32_t available_cache_len = position_offset.empty()
        ? sliding_window - (i + 1)
        : sliding_window - (position_offset[i] + 1);
    if (n_past + n_update > available_cache_len) {
      fill_mask(
          attention_mask_dtype_,
          cur_ptr - n_past * getDtypeSize(attention_mask_dtype_),
          n_past + n_update - available_cache_len,
          /*use_pos_value=*/false);
    }
    cur_ptr += metadata_.context_len * getDtypeSize(attention_mask_dtype_);
  }
}

void KVManager::set_state_zero_points(
    std::vector<int32_t> k_zero_points,
    std::vector<int32_t> v_zero_points) {
  state_k_zero_point_ = std::move(k_zero_points);
  state_v_zero_point_ = std::move(v_zero_points);
}

// Recurrent state must start at the real value 0. With float buffers that is
// just an all-zero buffer, but a quantized (uint16/uint8) state stores 0 as its
// zero point: the encoding is asymmetric, so leaving the buffer zeroed would
// start the model at -zero_point*scale (measured around -29) instead of 0 and
// corrupt the first tokens. Fill with the zero point when we know it.
void KVManager::fill_state_buffer(
    std::byte* buf,
    size_t nbytes,
    executorch::aten::ScalarType dtype,
    int32_t zero_point) {
  switch (dtype) {
    case executorch::aten::ScalarType::Float:
      std::fill_n(reinterpret_cast<float*>(buf), nbytes / sizeof(float), 0.0f);
      return;
    case executorch::aten::ScalarType::UInt16:
      ET_CHECK_MSG(
          zero_point >= 0,
          "Quantized recurrent state needs its zero point; the .pte is missing "
          "get_state_output_<i>_zero_point");
      std::fill_n(
          reinterpret_cast<uint16_t*>(buf),
          nbytes / sizeof(uint16_t),
          static_cast<uint16_t>(zero_point));
      return;
    case executorch::aten::ScalarType::Byte:
      ET_CHECK_MSG(
          zero_point >= 0,
          "Quantized recurrent state needs its zero point; the .pte is missing "
          "get_state_output_<i>_zero_point");
      std::fill_n(
          reinterpret_cast<uint8_t*>(buf),
          nbytes,
          static_cast<uint8_t>(zero_point));
      return;
    default:
      ET_CHECK_MSG(
          false,
          "Unsupported recurrent-state scalar type %s",
          executorch::runtime::toString(dtype));
  }
}

void KVManager::init_cache(IMemAlloc* buffer_manager, int32_t ar_len) {
  cur_ar_len_ = ar_len;
  for (int layer = 0; layer < metadata_.num_layers; ++layer) {
    k_cache_[layer].buffer =
        buffer_manager->allocate(k_cache_in_nbytes_[layer]);
    k_cache_[layer].output_buffer =
        buffer_manager->allocate(k_cache_out_nbytes_[layer]);
    v_cache_[layer].buffer =
        buffer_manager->allocate(v_cache_in_nbytes_[layer]);
    v_cache_[layer].output_buffer =
        buffer_manager->allocate(v_cache_out_nbytes_[layer]);
    if (!is_state_layer_[layer])
      continue;
    fill_state_buffer(
        k_cache_[layer].buffer,
        k_cache_in_nbytes_[layer],
        state_dtype_[layer],
        state_k_zero_point_[layer]);
    fill_state_buffer(
        v_cache_[layer].buffer,
        v_cache_in_nbytes_[layer],
        state_dtype_[layer],
        state_v_zero_point_[layer]);
    // 진단: 상태 슬롯이 기대한 dtype/zero_point/크기로 잡혔는지 실행 중 확인한다.
    if (layer < 3 || layer == metadata_.num_layers - 2) {
      ET_LOG(
          Info,
          "[STATEDBG] layer=%d dtype=%s k_nbytes=%zu v_nbytes=%zu k_zp=%d v_zp=%d "
          "k_first=%u v_first=%u",
          layer,
          executorch::runtime::toString(state_dtype_[layer]),
          k_cache_in_nbytes_[layer],
          v_cache_in_nbytes_[layer],
          state_k_zero_point_[layer],
          state_v_zero_point_[layer],
          state_dtype_[layer] == executorch::aten::ScalarType::UInt16
              ? (unsigned)*reinterpret_cast<uint16_t*>(k_cache_[layer].buffer)
              : 0u,
          state_dtype_[layer] == executorch::aten::ScalarType::UInt16
              ? (unsigned)*reinterpret_cast<uint16_t*>(v_cache_[layer].buffer)
              : 0u);
    }
  }
}

void KVManager::rearrange_cache(int32_t ar_len_dst) {
  // Don't need to rearrange if cur_ar_len_ is equal to target ar_len
  if (cur_ar_len_ == ar_len_dst)
    return;
  for (int layer = 0; layer < metadata_.num_layers; ++layer) {
    // Recurrent state does not depend on ar_len: nothing to rearrange.
    if (is_state_layer_[layer])
      continue;
    rearrange_key(k_cache_[layer], ar_len_dst, k_head_dim_[layer]);
    rearrange_value(v_cache_[layer], ar_len_dst, v_head_dim_[layer]);
  }
  // rearrange done.
  cur_ar_len_ = ar_len_dst;
}

void KVManager::rearrange_key(
    KVCache& k_cache,
    int32_t ar_len_dst,
    int64_t head_dim) {
  const int32_t src_cache_num = (cur_ar_len_ == metadata_.context_len)
      ? metadata_.context_len
      : metadata_.context_len - cur_ar_len_;
  const int32_t dst_cache_num = metadata_.context_len - ar_len_dst;
  std::byte* k_cache_in_read_ptr = k_cache.buffer;
  std::byte* k_cache_in_write_ptr = k_cache.buffer;
  size_t src_cache_nbytes = src_cache_num * getDtypeSize(kv_cache_dtype_);
  size_t dst_cache_nbytes = dst_cache_num * getDtypeSize(kv_cache_dtype_);
  if (src_cache_num > dst_cache_num) {
    // copy from first dimension
    for (int i = 0; i < head_dim * metadata_.num_heads; i++) {
      std::memmove(k_cache_in_write_ptr, k_cache_in_read_ptr, dst_cache_nbytes);
      k_cache_in_read_ptr += src_cache_nbytes;
      k_cache_in_write_ptr += dst_cache_nbytes;
    }
  } else {
    k_cache_in_read_ptr +=
        (head_dim * metadata_.num_heads - 1) * src_cache_nbytes;
    k_cache_in_write_ptr +=
        (head_dim * metadata_.num_heads - 1) * dst_cache_nbytes;
    // copy from last dimension
    for (int i = 0; i < head_dim * metadata_.num_heads; i++) {
      std::memmove(k_cache_in_write_ptr, k_cache_in_read_ptr, src_cache_nbytes);
      k_cache_in_read_ptr -= src_cache_nbytes;
      k_cache_in_write_ptr -= dst_cache_nbytes;
    }
  }
}

void KVManager::rearrange_value(
    KVCache& v_cache,
    int32_t ar_len_dst,
    int64_t head_dim) {
  const int32_t src_cache_num = (cur_ar_len_ == metadata_.context_len)
      ? metadata_.context_len
      : metadata_.context_len - cur_ar_len_;
  const int32_t dst_cache_num = metadata_.context_len - ar_len_dst;
  std::byte* v_cache_in_read_ptr = v_cache.buffer;
  std::byte* v_cache_in_write_ptr = v_cache.buffer;
  size_t src_cache_nbytes = src_cache_num * getDtypeSize(kv_cache_dtype_);
  size_t dst_cache_nbytes = dst_cache_num * getDtypeSize(kv_cache_dtype_);
  if (src_cache_num > dst_cache_num) {
    // copy from first dimension
    for (int i = 0; i < metadata_.num_heads; i++) {
      std::memmove(
          v_cache_in_write_ptr,
          v_cache_in_read_ptr,
          dst_cache_nbytes * head_dim);
      v_cache_in_read_ptr += src_cache_nbytes * head_dim;
      v_cache_in_write_ptr += dst_cache_nbytes * head_dim;
    }
  } else {
    v_cache_in_read_ptr +=
        head_dim * (metadata_.num_heads - 1) * src_cache_nbytes;
    v_cache_in_write_ptr +=
        head_dim * (metadata_.num_heads - 1) * dst_cache_nbytes;
    // copy from last dimension
    for (int i = 0; i < metadata_.num_heads; i++) {
      std::memmove(
          v_cache_in_write_ptr,
          v_cache_in_read_ptr,
          src_cache_nbytes * head_dim);
      v_cache_in_read_ptr -= src_cache_nbytes * head_dim;
      v_cache_in_write_ptr -= dst_cache_nbytes * head_dim;
    }
  }
}

void KVManager::update_cache(
    int32_t ar_len,
    int32_t n_past,
    int32_t n_update,
    const std::vector<bool>& selected,
    bool skip_state_copy) {
  ET_CHECK_MSG(
      cur_ar_len_ == ar_len,
      "Current AR length (%d) is not matched with target AR length (%d). Please rearrange cache first.",
      cur_ar_len_,
      ar_len);
  StateCopyPool* pool = skip_state_copy ? nullptr : StateCopyPool::Get();
  std::vector<StateCopyPool::Job> state_jobs;
  for (int layer = 0; layer < metadata_.num_layers; ++layer) {
    if (is_state_layer_[layer]) {
      if (skip_state_copy) continue;
      if (pool != nullptr) {
        if (k_cache_[layer].buffer != k_cache_[layer].output_buffer)
          state_jobs.push_back({k_cache_[layer].buffer, k_cache_[layer].output_buffer, k_cache_in_nbytes_[layer]});
        if (v_cache_[layer].buffer != v_cache_[layer].output_buffer)
          state_jobs.push_back({v_cache_[layer].buffer, v_cache_[layer].output_buffer, v_cache_in_nbytes_[layer]});
        continue;
      }
      // Whole-buffer replacement: the graph already produced the next state.
      update_state(k_cache_[layer], k_cache_in_nbytes_[layer]);
      update_state(v_cache_[layer], v_cache_in_nbytes_[layer]);
      continue;
    }
    update_key(k_cache_[layer], n_past, n_update, selected, k_head_dim_[layer]);
    update_value(
        v_cache_[layer], n_past, n_update, selected, v_head_dim_[layer]);
  }
  if (!state_jobs.empty()) pool->Run(state_jobs);
}

void KVManager::update_state(KVCache& cache, size_t nbytes) {
  if (cache.buffer != cache.output_buffer)
    std::memcpy(cache.buffer, cache.output_buffer, nbytes);
}

void KVManager::update_key(
    KVCache& k_cache,
    int32_t n_past,
    int32_t n_update,
    const std::vector<bool>& selected,
    int64_t head_dim) {
  std::byte* write_ptr = k_cache.buffer;
  std::byte* read_ptr = k_cache.output_buffer;
  const int32_t copy_size = n_update * getDtypeSize(kv_cache_dtype_);
  const int32_t iter_size = (cur_ar_len_ == metadata_.context_len)
      ? metadata_.context_len * getDtypeSize(kv_cache_dtype_)
      : (metadata_.context_len - cur_ar_len_) * getDtypeSize(kv_cache_dtype_);
  const int32_t out_size = cur_ar_len_ * getDtypeSize(kv_cache_dtype_);
  const int32_t past_size = n_past * getDtypeSize(kv_cache_dtype_);
  const int32_t n_iter = head_dim * metadata_.num_heads;

  write_ptr += past_size;
  if (selected.empty()) {
    for (int i = 0; i < n_iter; ++i) {
      std::memcpy(write_ptr, read_ptr, copy_size);
      write_ptr += iter_size;
      read_ptr += out_size;
    }
  } else {
    std::vector<int32_t> true_indices(n_update);
    for (int i = 0, j = 0; i < selected.size() && j < n_update; ++i) {
      if (selected[i]) {
        true_indices[j++] = i;
      }
    }
    for (int i = 0; i < n_iter; ++i) {
      auto wp = write_ptr, rp = read_ptr;
      for (auto ind : true_indices) {
        std::memmove(
            wp,
            rp + ind * getDtypeSize(kv_cache_dtype_),
            getDtypeSize(kv_cache_dtype_));
        wp += getDtypeSize(kv_cache_dtype_);
      }
      write_ptr += iter_size;
      read_ptr += out_size;
    }
  }
}

void KVManager::update_value(
    KVCache& v_cache,
    int32_t n_past,
    int32_t n_update,
    const std::vector<bool>& selected,
    int64_t head_dim) {
  std::byte* write_ptr = v_cache.buffer;
  std::byte* read_ptr = v_cache.output_buffer;
  const int32_t copy_size = n_update * head_dim * getDtypeSize(kv_cache_dtype_);
  const int32_t past_size = n_past * head_dim * getDtypeSize(kv_cache_dtype_);
  const int32_t n_iter = metadata_.num_heads;
  const int32_t iter_size = (cur_ar_len_ == metadata_.context_len)
      ? metadata_.context_len * head_dim * getDtypeSize(kv_cache_dtype_)
      : (metadata_.context_len - cur_ar_len_) * head_dim *
          getDtypeSize(kv_cache_dtype_);
  const int32_t out_size =
      cur_ar_len_ * head_dim * getDtypeSize(kv_cache_dtype_);

  write_ptr += past_size;

  if (selected.empty()) {
    for (int i = 0; i < n_iter; i++) {
      std::memcpy(write_ptr, read_ptr, copy_size);
      write_ptr += iter_size;
      read_ptr += out_size;
    }
  } else {
    int32_t update_times = n_update;
    for (int i = 0; i < n_iter; ++i) {
      auto wp = write_ptr, rp = read_ptr;
      for (auto sel : selected) {
        if (sel) {
          std::memcpy(wp, rp, head_dim * getDtypeSize(kv_cache_dtype_));
          wp += head_dim * getDtypeSize(kv_cache_dtype_);
          update_times--;
          if (update_times == 0)
            break;
        }
        rp += head_dim * getDtypeSize(kv_cache_dtype_);
      }
      write_ptr += iter_size;
      read_ptr += out_size;
    }
  }
}

} // namespace example
