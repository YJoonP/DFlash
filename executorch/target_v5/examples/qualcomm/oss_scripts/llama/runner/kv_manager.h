/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/imem_alloc.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/utils.h>
#include <cstdint>
#include <memory>
#include <vector>

namespace example {

// Structure to hold key-value cache buffers
struct KVCache {
  std::byte* buffer;
  std::byte* output_buffer;
};

/**
 * @class KVManager
 * @brief Class for kv cache update, rearrangement, and buffer allocatation.
 */
class KVManager {
 public:
  struct Metadata {
    int32_t context_len;
    int32_t max_ar_len;
    int32_t max_cache_len;
    int64_t num_heads;
    int64_t num_layers;
  };
  KVManager(
      Metadata metadata,
      std::unique_ptr<executorch::runtime::MethodMeta> method_meta);

  /**
   * @brief Allocate buffer for KV cache and set the cur_ar_len_.
   * @param buffer_manager Pointer to IMemAlloc instance; by default, it uses a
   * shared buffer with RPC memory.
   * @param ar_len Length of input tokens.
   */
  void init_cache(IMemAlloc* buffer_manager, int32_t ar_len);

  /**
   * @brief Provide the quantization zero points of the recurrent-state slots.
   *
   * Must be called before init_cache() when the state I/O is quantized.
   * Vectors are indexed by layer; -1 means "unknown" (buffer is left zeroed).
   */
  void set_state_zero_points(
      std::vector<int32_t> k_zero_points,
      std::vector<int32_t> v_zero_points);

  /**
   * @brief Switch key and value cache from AR-cur to AR-dst.
   * @param ar_len_dst Target length of input tokens.
   */
  void rearrange_cache(int32_t ar_len_dst);

  /**
   * @brief Initialize attention mask based on kv manager mode, and attention
   * map.
   * For example,
   * ar_len = 4, CL = 6, n_past = 0,
   * attention map: {-1, 0, 1, 2} and SMART_MASK.
   * Attention_mask will be:
   * [     0     0 65535     0     0     0 ]
   * [     0     0 65535 65535     0     0 ]
   * [     0     0 65535 65535 65535     0 ]
   * [     0     0 65535 65535 65535 65535 ]
   * @param attention_mask Pointer to the attention mask array to be
   * initialized.
   * @param attention_map Vector containing the attention map values. The shape
   * of attention map should be [ar_len].
   * @param ar_len Length of input tokens.
   * @param n_past Number of past elements in the cache.
   */
  void init_attention_mask(
      std::byte* attention_mask,
      const std::vector<int32_t>& attention_map,
      int32_t ar_len,
      int32_t n_past);

  /**
   * @brief Initialize attention mask based on kv manager mode, and attention
   * map.
   * For example,
   * ar_len = 4, CL = 6, n_past = 0,
   * attention map: {-1, 0, 1, 2} and SMART_MASK.
   * Attention_mask will be:
   * [     0     0 65535     0     0     0 ]
   * [     0     0 65535 65535     0     0 ]
   * [     0     0 65535 65535 65535     0 ]
   * [     0     0 65535 65535 65535 65535 ]
   * @param attention_mask Pointer to the attention mask array to be
   * initialized.
   * @param attention_map Vector containing the attention map values. The shape
   * of attention map should be [ar_len].
   * @param ar_len Length of input tokens.
   * @param n_past Number of past elements in the cache.
   * @param sliding_window Length of sliding window for sliding window attention
   * mask
   * @param position_offset (optional) attention mask position offset of
   */
  void init_attention_mask(
      std::byte* attention_mask,
      const std::vector<int32_t>& attention_map,
      int32_t ar_len,
      int32_t n_past,
      int32_t sliding_window,
      const std::vector<int32_t>& position_offset = {});

  /**
   * @brief Update attention mask based on kv manager mode, and n_update.
   * @param attention_mask Pointer to the attention mask array to be
   * initialized.
   * @param ar_len Length of input tokens.
   * @param n_past Number of past elements in the cache.
   * @param n_update Number of elements to be updated.
   */
  void update_attention_mask(
      std::byte* attention_mask,
      int32_t ar_len,
      int32_t n_past,
      int32_t n_update);

  /**
   * @brief Update attention mask based on kv manager mode, and n_update.
   * @param attention_mask Pointer to the attention mask array to be
   * initialized.
   * @param ar_len Length of input tokens.
   * @param n_past Number of past elements in the cache.
   * @param n_update Number of elements to be updated.
   * @param sliding_window Length of sliding window for sliding window attention
   * mask
   * @param position_offset (optional) attention mask position offset of
   * lookahead decoder
   */
  void update_attention_mask(
      std::byte* attention_mask,
      int32_t ar_len,
      int32_t n_past,
      int32_t n_update,
      int32_t sliding_window,
      const std::vector<int32_t>& position_offset = {});

  /**
   * @brief Based on cur_ar_len_ to update cache
   * @param ar_len Length of input tokens.
   * @param n_past Number of past elements in the cache.
   * @param n_update Number of elements to be updated.
   * @param selected Indicate which position to be updated
   * @param skip_state_copy Decode caller owns recurrent-state buffer rotation
   * and restores canonical buffers on exit; attention KV updates still run.
   */
  void update_cache(
      int32_t ar_len,
      int32_t n_past,
      int32_t n_update,
      const std::vector<bool>& selected,
      bool skip_state_copy = false);

  const std::vector<KVCache>& get_k_cache_() const {
    return k_cache_;
  }
  const std::vector<KVCache>& get_v_cache_() const {
    return v_cache_;
  }

  inline const size_t total_cache_size_in_bytes() const {
    return total_cache_size_;
  }

 private:
  // Helper functions to rearrange and update key and value caches

  void rearrange_key(KVCache& k_cache, int32_t ar_len_dst, int64_t head_dim);

  void rearrange_value(KVCache& v_cache, int32_t ar_len_dst, int64_t head_dim);

  void update_key(
      KVCache& k_cache,
      int32_t n_past,
      int32_t n_update,
      const std::vector<bool>& selected,
      int64_t head_dim);

  void update_value(
      KVCache& v_cache,
      int32_t n_past,
      int32_t n_update,
      const std::vector<bool>& selected,
      int64_t head_dim);

  // metadata
  Metadata metadata_;
  size_t total_cache_size_;
  int32_t cur_ar_len_;
  executorch::aten::ScalarType attention_mask_dtype_ =
      executorch::aten::ScalarType::Undefined;
  executorch::aten::ScalarType kv_cache_dtype_ =
      executorch::aten::ScalarType::Undefined;
  // Store start pointer of k and v cache for input and output
  // input: layer -> head * head_dim * max_cache_len
  // output: layer -> head * head_dim * max_ar_len
  std::vector<KVCache> k_cache_;
  std::vector<KVCache> v_cache_;
  // Per-layer buffer sizes and head_dim (may differ when head_dim varies across
  // layers)
  std::vector<size_t> k_cache_in_nbytes_;
  std::vector<size_t> k_cache_out_nbytes_;
  std::vector<size_t> v_cache_in_nbytes_;
  std::vector<size_t> v_cache_out_nbytes_;
  std::vector<int64_t> k_head_dim_;
  std::vector<int64_t> v_head_dim_;
  // Qwen3.5 hybrid (Gated DeltaNet) support: layers whose two cache slots hold
  // recurrent state (conv_state / recurrent_state) rather than K/V caches.
  // Such slots are whole-buffer replacements: input shape == output shape, the
  // update is a straight copy of the graph output back into the input buffer,
  // and there is nothing to rearrange when ar_len changes.
  std::vector<bool> is_state_layer_;
  bool has_state_layer_ = false;
  // 상태 슬롯의 스칼라 타입(입력 기준). float32 면 0 바이트가 곧 0.0 이라
  // 별도 초기화가 필요 없지만, uint16 로 태깅해서 내보낸 모델에서는
  // 0 바이트가 실제로는 -zero_point*scale 을 뜻하므로 zero_point 로 채워야 한다.
  std::vector<executorch::aten::ScalarType> state_dtype_;
  // 층별 상태 슬롯의 zero_point (k 슬롯, v 슬롯). -1 이면 값을 모른다는 뜻.
  std::vector<int32_t> state_k_zero_point_;
  std::vector<int32_t> state_v_zero_point_;

  void update_state(KVCache& cache, size_t nbytes);
  void fill_state_buffer(
      std::byte* buf,
      size_t nbytes,
      executorch::aten::ScalarType dtype,
      int32_t zero_point);

 public:
  /** @brief True if the given layer's slots carry recurrent state. */
  bool is_state_layer(int32_t layer) const {
    return layer < static_cast<int32_t>(is_state_layer_.size())
        ? static_cast<bool>(is_state_layer_[layer])
        : false;
  }
  /** @brief True if the model has any recurrent-state layer. */
  bool has_state_layer() const {
    return has_state_layer_;
  }
};
} // namespace example
