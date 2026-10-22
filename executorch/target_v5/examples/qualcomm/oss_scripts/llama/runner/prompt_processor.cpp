/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/safe_numerics.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/prompt_processor.h>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
using executorch::aten::TensorImpl;
using executorch::runtime::EValue;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;
using executorch::runtime::TensorInfo;
namespace example {

PromptProcessor::PromptProcessor(
    DecoderRunner* decoder_runner,
    KVManager* kv_manager,
    const std::string& method_name,
    Metadata metadata,
    std::unique_ptr<MethodMeta> method_meta)
    : decoder_runner_(decoder_runner),
      kv_manager_(kv_manager),
      method_name_(method_name),
      metadata_(metadata) {
  k_cache_in_.resize(metadata_.num_layers);
  v_cache_in_.resize(metadata_.num_layers);
  k_cache_out_.resize(metadata_.num_layers);
  v_cache_out_.resize(metadata_.num_layers);
  // Calculate I/O size
  Result<TensorInfo> attention_mask = method_meta->input_tensor_meta(1);
  Result<TensorInfo> logits = method_meta->output_tensor_meta(0);
  input_toks_.size = metadata_.ar_len * sizeof(int64_t);
  if (is_bert()) {
    input_pos_.size = 0;
  } else {
    input_pos_.size = metadata_.ar_len * sizeof(int32_t);
  }

  attention_mask_.dtype = attention_mask->scalar_type();
  attention_mask_.size = metadata_.ar_len * metadata_.context_len *
      attention_mask_.getElementSize();
  switch (metadata_.cache_mode) {
    case CacheMode::StaticCahce:
      window_attention_mask_.size = 0;
      break;
    case CacheMode::HybridCache: {
      Result<TensorInfo> window_attention_mask =
          method_meta->input_tensor_meta(2);
      window_attention_mask_.dtype = window_attention_mask->scalar_type();
      window_attention_mask_.size = metadata_.ar_len * metadata_.context_len *
          window_attention_mask_.getElementSize();
      break;
    }
    default:
      ET_CHECK_MSG(false, "Unsupported llama cache mode");
      break;
  }

  logits_.dtype = logits->scalar_type();
  // Size from the graph: ar_len rows, or 1 row when prefill emits only the last valid row.
  logits_.size = logits->nbytes();
  logits_rows_ = static_cast<int32_t>(
      logits_.size / (metadata_.vocab_size * logits_.getElementSize()));
  ET_CHECK_MSG(
      logits_rows_ == metadata_.ar_len ||
          (metadata_.prefill_last_row && logits_rows_ == 1),
      "unexpected prefill logits rows %d",
      logits_rows_);

  prefill_valid_.size = 0;
  conv_index_.size = 0;
  last_row_.size = 0;
  if (metadata_.prefill_valid) {
    // Two int32 inputs after the caches: valid (1, ar_len, 1, 1) and conv_idx (K-1).
    const size_t base = 2 +
        (metadata_.cache_mode == CacheMode::HybridCache ? 1 : 0) +
        (is_bert() ? 0 : 1 + 2 * metadata_.num_layers);
    const size_t n_extra = metadata_.prefill_last_row ? 3 : 2;
    ET_CHECK_MSG(
        method_meta->num_inputs() == base + n_extra,
        "prefill_valid PTE: expected %zu inputs, method has %zu",
        base + n_extra,
        method_meta->num_inputs());
    Result<TensorInfo> valid = method_meta->input_tensor_meta(base);
    Result<TensorInfo> conv_idx = method_meta->input_tensor_meta(base + 1);
    ET_CHECK_MSG(
        valid->scalar_type() == executorch::aten::ScalarType::Int &&
            valid->nbytes() == metadata_.ar_len * sizeof(int32_t),
        "prefill_valid input must be int32 with ar_len elements");
    ET_CHECK_MSG(
        conv_idx->scalar_type() == executorch::aten::ScalarType::Int &&
            conv_idx->sizes().size() == 1 && conv_idx->sizes()[0] >= 1 &&
            conv_idx->sizes()[0] <= metadata_.ar_len,
        "conv_idx input must be a 1-D int32 tensor");
    conv_index_len_ = conv_idx->sizes()[0];
    prefill_valid_.size = metadata_.ar_len * sizeof(int32_t);
    conv_index_.size = conv_index_len_ * sizeof(int32_t);
    if (metadata_.prefill_last_row) {
      Result<TensorInfo> last = method_meta->input_tensor_meta(base + 2);
      ET_CHECK_MSG(
          last->scalar_type() == executorch::aten::ScalarType::Int &&
              last->nbytes() == sizeof(int32_t),
          "last_row input must be one int32");
      last_row_.size = sizeof(int32_t);
    }
    const char* legacy = std::getenv("QNN_PREFILL_VALID_LEGACY");
    prefill_valid_legacy_ = legacy != nullptr && std::string(legacy) == "1";
    const char* phantom = std::getenv("QNN_PREFILL_PHANTOM_TOKEN");
    phantom_token_ = phantom != nullptr ? std::atoi(phantom) : 0;
    ET_CHECK_MSG(
        phantom_token_ >= 0 && phantom_token_ < metadata_.vocab_size,
        "QNN_PREFILL_PHANTOM_TOKEN %d out of [0, %d)",
        phantom_token_,
        metadata_.vocab_size);
    ET_LOG(
        Info,
        "Prompt Processor: prefill_valid inputs %zu/%zu, conv_idx len %d, mode %s, phantom token %d",
        base,
        base + 1,
        conv_index_len_,
        prefill_valid_legacy_ ? "legacy" : "fixed",
        phantom_token_);
  }
};

void PromptProcessor::fill_prefill_valid(int32_t n_valid) {
  const int32_t n = prefill_valid_legacy_ ? metadata_.ar_len : n_valid;
  for (int32_t t = 0; t < metadata_.ar_len; ++t) {
    prefill_valid_.data[t] = t < n ? 1 : 0;
  }
  for (int32_t j = 0; j < conv_index_len_; ++j) {
    conv_index_.data[j] = n + j;
  }
  if (metadata_.prefill_last_row) {
    // Legacy mode only re-enables the padded rows' state updates; the first
    // token is still read from the last prompt row, as before.
    last_row_.data[0] = n_valid - 1;
  }
}

void PromptProcessor::init_io(
    IMemAlloc* buffer_manager,
    Result<MethodMeta> method_meta) {
  size_t idx = 0;
  input_tensors_.reserve(method_meta->num_inputs());
  output_tensors_.reserve(method_meta->num_outputs());
  // [I]: input_tokens
  Result<TensorInfo> input_toks = method_meta->input_tensor_meta(idx++);
  input_toks_.data =
      reinterpret_cast<int64_t*>(buffer_manager->allocate(input_toks_.size));
  std::memset(input_toks_.data, 0, input_toks_.size);
  input_toks_.tensor = std::make_unique<TensorImpl>(
      input_toks->scalar_type(),
      input_toks->sizes().size(),
      const_cast<TensorImpl::SizesType*>(input_toks->sizes().data()),
      input_toks_.data,
      const_cast<TensorImpl::DimOrderType*>(input_toks->dim_order().data()));
  input_tensors_.emplace_back(input_toks_.tensor.get());
  buffer_manager->add_memory_info(
      input_toks_.data, input_toks_.size, input_toks.get());

  // [I]: attention_mask
  Result<TensorInfo> attention_mask = method_meta->input_tensor_meta(idx++);
  attention_mask_.data = buffer_manager->allocate(attention_mask_.size);
  attention_mask_.tensor = std::make_unique<TensorImpl>(
      attention_mask->scalar_type(),
      attention_mask->sizes().size(),
      const_cast<TensorImpl::SizesType*>(attention_mask->sizes().data()),
      attention_mask_.data,
      const_cast<TensorImpl::DimOrderType*>(
          attention_mask->dim_order().data()));
  input_tensors_.emplace_back(attention_mask_.tensor.get());
  buffer_manager->add_memory_info(
      attention_mask_.data, attention_mask_.size, attention_mask.get());

  // [I]: sliding window attention_mask
  if (metadata_.cache_mode == CacheMode::HybridCache) {
    Result<TensorInfo> window_attention_mask =
        method_meta->input_tensor_meta(idx++);
    window_attention_mask_.data =
        buffer_manager->allocate(window_attention_mask_.size);
    window_attention_mask_.tensor = std::make_unique<TensorImpl>(
        window_attention_mask->scalar_type(),
        window_attention_mask->sizes().size(),
        const_cast<TensorImpl::SizesType*>(
            window_attention_mask->sizes().data()),
        window_attention_mask_.data,
        const_cast<TensorImpl::DimOrderType*>(
            window_attention_mask->dim_order().data()));
    input_tensors_.emplace_back(window_attention_mask_.tensor.get());
    buffer_manager->add_memory_info(
        window_attention_mask_.data,
        window_attention_mask_.size,
        window_attention_mask.get());
  }

  if (!is_bert()) {
    // [I]: input_pos
    Result<TensorInfo> input_pos = method_meta->input_tensor_meta(idx++);
    input_pos_.data =
        reinterpret_cast<int32_t*>(buffer_manager->allocate(input_pos_.size));
    input_pos_.tensor = std::make_unique<TensorImpl>(
        input_pos->scalar_type(),
        input_pos->sizes().size(),
        const_cast<TensorImpl::SizesType*>(input_pos->sizes().data()),
        input_pos_.data,
        const_cast<TensorImpl::DimOrderType*>(input_pos->dim_order().data()));
    input_tensors_.emplace_back(input_pos_.tensor.get());
    buffer_manager->add_memory_info(
        input_pos_.data, input_pos_.size, input_pos.get());

    // [I] kv_cache
    // Prepare the vector of EValue for kv cache to evict token
    cache_inputs_.reserve(2 * metadata_.num_layers);
    size_t index = idx; // bypass input_tokens, atten_mask, input_pos
    for (int cache_group = 0; cache_group < 2; ++cache_group) {
      std::vector<std::unique_ptr<TensorImpl>>& cache =
          (cache_group == 0 ? k_cache_in_ : v_cache_in_);
      std::vector<KVCache> cache_ptrs = (cache_group == 0)
          ? kv_manager_->get_k_cache_()
          : kv_manager_->get_v_cache_();
      for (int layer = 0; layer < metadata_.num_layers; ++layer, ++index) {
        Result<TensorInfo> kv_cache = method_meta->input_tensor_meta(index);

        cache[layer] = std::make_unique<TensorImpl>(
            kv_cache->scalar_type(),
            kv_cache->sizes().size(),
            const_cast<TensorImpl::SizesType*>(kv_cache->sizes().data()),
            cache_ptrs[layer].buffer,
            const_cast<TensorImpl::DimOrderType*>(
                kv_cache->dim_order().data()));
        input_tensors_.emplace_back(cache[layer].get());
        cache_inputs_.emplace_back(input_tensors_.back());
        buffer_manager->add_memory_info(
            cache_ptrs[layer].buffer, cache[layer]->nbytes(), kv_cache.get());
      }
    }

    // [I] prefill_valid, conv_idx (QNN_PREFILL_VALID PTEs only)
    if (metadata_.prefill_valid) {
      std::vector<TensorStruct<int32_t>*> extras = {&prefill_valid_, &conv_index_};
      if (metadata_.prefill_last_row) {
        extras.push_back(&last_row_);
      }
      for (TensorStruct<int32_t>* io : extras) {
        Result<TensorInfo> info = method_meta->input_tensor_meta(index++);
        io->data = reinterpret_cast<int32_t*>(buffer_manager->allocate(io->size));
        io->tensor = std::make_unique<TensorImpl>(
            info->scalar_type(),
            info->sizes().size(),
            const_cast<TensorImpl::SizesType*>(info->sizes().data()),
            io->data,
            const_cast<TensorImpl::DimOrderType*>(info->dim_order().data()));
        input_tensors_.emplace_back(io->tensor.get());
        buffer_manager->add_memory_info(io->data, io->size, info.get());
      }
      fill_prefill_valid(metadata_.ar_len);
    }
  }

  // [O]: logits
  Result<TensorInfo> logits = method_meta->output_tensor_meta(0);
  logits_.data = buffer_manager->allocate(logits_.size);
  logits_.tensor = std::make_unique<TensorImpl>(
      logits->scalar_type(),
      logits->sizes().size(),
      const_cast<TensorImpl::SizesType*>(logits->sizes().data()),
      logits_.data,
      const_cast<TensorImpl::DimOrderType*>(logits->dim_order().data()));
  output_tensors_.emplace_back(logits_.tensor.get());
  buffer_manager->add_memory_info(logits_.data, logits_.size, logits.get());

  // [O] kv_cache
  size_t index = 1;
  for (int cache_group = 0; cache_group < 2; ++cache_group) {
    std::vector<std::unique_ptr<TensorImpl>>& cache =
        (cache_group == 0 ? k_cache_out_ : v_cache_out_);
    std::vector<KVCache> cache_ptrs = (cache_group == 0)
        ? kv_manager_->get_k_cache_()
        : kv_manager_->get_v_cache_();
    for (int layer = 0; layer < metadata_.num_layers; ++layer, ++index) {
      Result<TensorInfo> kv_cache = method_meta->output_tensor_meta(index);
      cache[layer] = std::make_unique<TensorImpl>(
          kv_cache->scalar_type(),
          kv_cache->sizes().size(),
          const_cast<TensorImpl::SizesType*>(kv_cache->sizes().data()),
          cache_ptrs[layer].output_buffer,
          const_cast<TensorImpl::DimOrderType*>(kv_cache->dim_order().data()));
      output_tensors_.emplace_back(cache[layer].get());
      buffer_manager->add_memory_info(
          cache_ptrs[layer].output_buffer,
          cache[layer]->nbytes(),
          kv_cache.get());
    }
  }
  // Prepare the vector of EValue to run inference
  inputs_.reserve(input_tensors_.size());
  for (auto& input_tensor : input_tensors_) {
    inputs_.emplace_back(std::move(input_tensor));
  }
}

const std::vector<std::byte>& PromptProcessor::get_all_logits() {
  return prompt_all_logits_;
}

void PromptProcessor::prepare_io(
    const std::vector<uint64_t>& prompt_tokens,
    int64_t prompt_pos,
    int64_t start_pos) {
  for (int i = 0; i < metadata_.ar_len; i++) {
    if (!is_bert()) {
      // Prepare pos data
      input_pos_.data[i] = start_pos + i;
    }

    // Prepare input token data
    if (prompt_pos + i < prompt_tokens.size()) {
      // Support CPU 4-bit embedding, which requires int64 input.
      // However, for QNN embedding, only int32 input is needed.
      // Therefore, we need to cast to the correct type to write the data.
      if (metadata_.use_int64_token) {
        input_toks_.data[i] = prompt_tokens[prompt_pos + i];
      } else {
        int32_t* input_toks_ptr = reinterpret_cast<int32_t*>(input_toks_.data);
        input_toks_ptr[i] = static_cast<int32_t>(prompt_tokens[prompt_pos + i]);
      }
    } else if (metadata_.prefill_valid && !prefill_valid_legacy_) {
      // Padded rows are masked out; write a fixed id so no stale or
      // uninitialized value ever reaches the embedding.
      if (metadata_.use_int64_token) {
        input_toks_.data[i] = phantom_token_;
      } else {
        reinterpret_cast<int32_t*>(input_toks_.data)[i] = phantom_token_;
      }
    }
  }
}

Result<uint64_t> PromptProcessor::prefill(
    std::vector<uint64_t> prompt_tokens,
    int64_t start_pos,
    bool dump_logits,
    AttentionSinkRopeRunner* attention_sink_rope_runner) {
  ET_CHECK_MSG(!prompt_tokens.empty(), "Prompt cannot be null");

  int64_t shifted_pos = start_pos;
  bool enable_attention_sink = attention_sink_rope_runner != nullptr;

  // Calculate number of blocks
  int32_t num_prompt_tokens = prompt_tokens.size();
  if (is_bert()) {
    ET_CHECK_MSG(
        start_pos == 0, "Bert model doesn't support multi-turn conversation.");
  } else if (!enable_attention_sink) {
    int64_t end_pos = 0;
    ET_CHECK_MSG(
        !c10::add_overflows(
            start_pos, static_cast<int64_t>(num_prompt_tokens), &end_pos) &&
            end_pos <= static_cast<int64_t>(metadata_.context_len) -
                    static_cast<int64_t>(metadata_.ar_len),
        "The sequence length exceeds the maximum limit that the prompt processor can handle.");
  }

  // store the token
  int64_t cur_token;
  int64_t prompt_pos = 0;
  int32_t n_update = metadata_.ar_len;
  int num_iters = 1 + ((num_prompt_tokens - 1) / metadata_.ar_len);
  ET_LOG(
      Info,
      "Prompt Processor: total %d prompt tokens (AR-%d * %d iters)",
      num_prompt_tokens,
      metadata_.ar_len,
      num_iters);

  // Initialize attention sink rope runner if given and update position
  // accordingly
  if (enable_attention_sink) {
    ET_CHECK_MSG(
        attention_sink_rope_runner->set_outputs(method_name_, cache_inputs_) ==
            executorch::runtime::Error::Ok,
        "Failed to set output tensor for module %s",
        method_name_.c_str());
    shifted_pos =
        shifted_pos - attention_sink_rope_runner->get_position_shift();
  }

  // Rearrange KV cache first
  kv_manager_->rearrange_cache(metadata_.ar_len);
  std::vector<int32_t> attention_map(metadata_.ar_len);
  std::iota(attention_map.begin(), attention_map.end(), -1);
  // Initialize attention mask with current position
  kv_manager_->init_attention_mask(
      attention_mask_.data, attention_map, metadata_.ar_len, shifted_pos);
  // Initialize window attention mask with current position
  if (metadata_.cache_mode == CacheMode::HybridCache) {
    kv_manager_->init_attention_mask(
        window_attention_mask_.data,
        attention_map,
        metadata_.ar_len,
        shifted_pos,
        metadata_.sliding_window);
  }

  // Initialize the output of the module
  ET_CHECK_MSG(
      decoder_runner_->set_outputs(method_name_, output_tensors_) ==
          executorch::runtime::Error::Ok,
      "Failed to set output tensor for module %s",
      method_name_.c_str());

  for (int i = 0; i < num_iters; ++i) {
    // The current position plus the future generated cache exceeds the cache
    // size, which means we need to remove eviction_batch_size key-value cache
    // entries to make room for new tokens.
    if (enable_attention_sink &&
        shifted_pos + metadata_.ar_len >
            metadata_.context_len - metadata_.ar_len) {
      attention_sink_rope_runner->evict_token(method_name_, cache_inputs_);
      shifted_pos =
          shifted_pos - attention_sink_rope_runner->get_eviction_batch_size();
      // Initialize attention mask with current position
      kv_manager_->init_attention_mask(
          attention_mask_.data, attention_map, metadata_.ar_len, shifted_pos);
      // Initialize window attention mask with current position
      if (metadata_.cache_mode == CacheMode::HybridCache) {
        kv_manager_->init_attention_mask(
            window_attention_mask_.data,
            attention_map,
            metadata_.ar_len,
            shifted_pos,
            metadata_.sliding_window);
      }
    }

    // Fill in the token and position data
    prepare_io(prompt_tokens, prompt_pos, shifted_pos);
    if (metadata_.prefill_valid) {
      fill_prefill_valid(
          i == num_iters - 1 ? 1 + ((num_prompt_tokens - 1) % metadata_.ar_len)
                             : metadata_.ar_len);
    }

    // Run inference
    auto step_result = decoder_runner_->step(method_name_, inputs_);
    ET_CHECK_MSG(
        !metadata_.prefill_valid || step_result.ok(),
        "prefill_forward failed (0x%x)",
        static_cast<unsigned>(step_result.error()));
    if (dump_logits) {
      prompt_all_logits_.insert(
          prompt_all_logits_.end(),
          logits_.data,
          logits_.data + logits_.size);
    }
    // In the last run, offset to the meaningful logits.
    if (i == num_iters - 1) {
      n_update = 1 + ((num_prompt_tokens - 1) % metadata_.ar_len);
    }
    // Update KV Cache with the output results
    kv_manager_->update_cache(metadata_.ar_len, shifted_pos, n_update, {});

    // Update attention mask with current position
    kv_manager_->update_attention_mask(
        attention_mask_.data, metadata_.ar_len, shifted_pos, n_update);
    if (metadata_.cache_mode == CacheMode::HybridCache) {
      kv_manager_->update_attention_mask(
          window_attention_mask_.data,
          metadata_.ar_len,
          shifted_pos,
          n_update,
          metadata_.sliding_window);
    }
    prompt_pos += metadata_.ar_len;
    shifted_pos += metadata_.ar_len;
  }

  cur_token = decoder_runner_->logits_to_token(
      output_tensors_[0],
      logits_rows_ == 1
          ? 0
          : (num_prompt_tokens + metadata_.ar_len - 1) % metadata_.ar_len);
  return cur_token;
}

} // namespace example
