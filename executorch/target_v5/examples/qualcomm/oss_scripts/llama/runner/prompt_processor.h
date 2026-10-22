/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/attention_sink_rope_runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/cache_utils.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/decoder_runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/imem_alloc.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/kv_manager.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/utils.h>
#include <memory>
#include <string>

namespace example {
/**
 * @class PromptProcessor
 * @brief Class for processing prompts using decoder and key-value manager.
 */

class PromptProcessor {
 public:
  struct Metadata {
    int32_t context_len;
    int64_t num_heads;
    int64_t num_layers;
    int32_t ar_len;
    int32_t vocab_size;
    bool use_int64_token;
    int sliding_window;
    CacheMode cache_mode;
    // PTE exports prefill_forward with valid/conv_idx inputs (get_prefill_valid == 1).
    bool prefill_valid = false;
    // ... plus a last_row input; prefill emits only the last valid row's logits.
    bool prefill_last_row = false;
  };
  PromptProcessor(
      DecoderRunner* decoder_runner,
      KVManager* kv_manager,
      const std::string& method_name,
      Metadata metadata,
      std::unique_ptr<executorch::extension::MethodMeta> method_meta);

  virtual ~PromptProcessor() = default;

  /**
   * @brief Initialize I/O tensor and allocate I/O data buffer.
   * @param buffer_manager Pointer to IMemAlloc instance; by default, it uses a
   * shared buffer with RPC memory.
   * @param method_meta Method metadata.
   */
  void init_io(
      IMemAlloc* buffer_manager,
      executorch::runtime::Result<executorch::runtime::MethodMeta> method_meta);

  /**
   * @brief Get the all logits generated
   *
   * @return std::vector<std::byte>& all the logits generated
   */
  virtual const std::vector<std::byte>& get_all_logits();

  /**
   * Prefill an LLM Module with the given text input.
   * @param prompt_tokens The text prompt tokens to the LLM Module. Encoded by
   * tokenizer.
   * @param start_pos The starting position in KV cache of the input in the LLM
   * Module.
   * @param dump_logits Used to save all logits. Only enable when analyzing
   * accuracy.
   * @return The next token of the LLM Module after prefill.
   */
  executorch::runtime::Result<uint64_t> prefill(
      std::vector<uint64_t> prompt_tokens,
      int64_t start_pos,
      bool dump_logits,
      AttentionSinkRopeRunner* attention_sink_rope_runner);
  /**
   * @brief Get total I/O size in bytes (excluding the KV cache size)
   * @return Total I/O size in bytes.
   */
  inline const size_t total_prompt_processor_io_size_in_bytes() const {
    return input_toks_.size + input_pos_.size + attention_mask_.size +
        window_attention_mask_.size + logits_.size + prefill_valid_.size +
        conv_index_.size + last_row_.size;
  }

 protected:
  // If the cache length is zero, it indicates a BERT model, which does not use
  // position ids or KV cache inputs.
  bool is_bert() const {
    return metadata_.context_len == metadata_.ar_len;
  }
  /**
   * @brief Fill in I/O buffers with prompt token and position.
   * @param prompt_tokens Vector of prompt tokens.
   * @param prompt_pos Position of the prompt.
   * @param start_pos Starting position.
   */
  void prepare_io(
      const std::vector<uint64_t>& prompt_tokens,
      int64_t prompt_pos,
      int64_t start_pos);
  /**
   * @brief Mark the first n_valid rows of the chunk as prompt rows and point
   * the conv-state gather at the K-1 columns after the last of them.
   */
  void fill_prefill_valid(int32_t n_valid);
  DecoderRunner* decoder_runner_;
  KVManager* kv_manager_;
  std::string method_name_;

  // metadata
  Metadata metadata_;

  // inputs and outputs
  TensorStruct<int64_t> input_toks_;
  TensorStruct<int32_t> input_pos_;
  TensorStructRaw attention_mask_;
  TensorStructRaw window_attention_mask_;
  TensorStructRaw logits_;
  // QNN_PREFILL_VALID: row mask and conv-state gather index of the last chunk.
  TensorStruct<int32_t> prefill_valid_;
  TensorStruct<int32_t> conv_index_;
  TensorStruct<int32_t> last_row_;
  // Rows of logits the graph emits per call (ar_len, or 1 with prefill_last_row).
  int32_t logits_rows_ = 0;
  int32_t conv_index_len_ = 0;
  bool prefill_valid_legacy_ = false;
  int32_t phantom_token_ = 0;

  // layer -> TensorImpl
  std::vector<std::unique_ptr<executorch::aten::TensorImpl>> k_cache_in_;
  std::vector<std::unique_ptr<executorch::aten::TensorImpl>> v_cache_in_;
  std::vector<std::unique_ptr<executorch::aten::TensorImpl>> k_cache_out_;
  std::vector<std::unique_ptr<executorch::aten::TensorImpl>> v_cache_out_;

  std::vector<executorch::runtime::EValue> inputs_;
  std::vector<executorch::aten::Tensor> input_tensors_;
  std::vector<executorch::aten::Tensor> output_tensors_;
  // Used for attention sink to evict KV cache.
  std::vector<executorch::runtime::EValue> cache_inputs_;

  // Unused by default, only used when dump_logits_path is provided.
  std::vector<std::byte> prompt_all_logits_;
};
} // namespace example
