/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/token_generator.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <numeric>

using executorch::aten::TensorImpl;
using executorch::runtime::EValue;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;
using executorch::runtime::TensorInfo;

namespace example {
namespace {
// Only decode TensorImpl pointers move. KVManager and prefill retain canonical
// addresses. Restore them on every exit, including execution errors and EOS.
class DecodeStateBuffers {
 public:
  void add(TensorImpl* input, TensorImpl* output) {
    ET_CHECK_MSG(input->nbytes() == output->nbytes(), "State buffer size mismatch");
    pairs_.push_back({input, output, input->mutable_data(), output->mutable_data()});
  }
  void rotate() {
    for (auto& p : pairs_) {
      void* previous = p.input->mutable_data();
      p.input->set_data(p.output->mutable_data());
      p.output->set_data(previous);
    }
  }
  ~DecodeStateBuffers() {
    for (auto& p : pairs_) {
      if (p.input->mutable_data() != p.canonical_input)
        std::memcpy(p.canonical_input, p.input->data(), p.input->nbytes());
      p.input->set_data(p.canonical_input);
      p.output->set_data(p.canonical_output);
    }
  }
 private:
  struct Pair {
    TensorImpl* input;
    TensorImpl* output;
    void* canonical_input;
    void* canonical_output;
  };
  std::vector<Pair> pairs_;
};
// 2026-09-15 KST: decode 상태를 제자리(in-place)로 갱신한다. 상태 출력 텐서가 입력 버퍼를 그대로 가리키게
// 해서 CPU 복사(update_state)도, 매 토큰 버퍼 교대(ping-pong)도 없앤다. 공유 버퍼(zero-copy)에서는 QNN 이
// 매 실행 같은 memHandle 을 보게 되어, 핸들이 바뀔 때 드는 실행당 비용이 사라진다.
// 끝나면 출력 텐서를 원래 버퍼로 되돌리고, 최신 상태는 입력 버퍼(프리필이 읽는 쪽)에 남는다.
class DecodeStateInplace {
 public:
  void add(TensorImpl* input, TensorImpl* output) {
    ET_CHECK_MSG(input->nbytes() == output->nbytes(), "State buffer size mismatch");
    pairs_.push_back({output, output->mutable_data()});
    output->set_data(input->mutable_data());
  }
  bool empty() const { return pairs_.empty(); }
  ~DecodeStateInplace() {
    for (auto& p : pairs_) p.output->set_data(p.canonical_output);
  }
 private:
  struct Pair {
    TensorImpl* output;
    void* canonical_output;
  };
  std::vector<Pair> pairs_;
};
} // namespace

TokenGenerator::TokenGenerator(
    tokenizers::Tokenizer* tokenizer,
    DecoderRunner* decoder_runner,
    KVManager* kv_manager,
    const std::string& method_name,
    std::unique_ptr<std::unordered_set<uint64_t>>&& eos_ids,
    Metadata metadata,
    executorch::llm::Stats* stats,
    std::unique_ptr<MethodMeta> method_meta)
    : tokenizer_(tokenizer),
      decoder_runner_(decoder_runner),
      kv_manager_(kv_manager),
      method_name_(method_name),
      eos_ids_(std::move(eos_ids)),
      stats_(stats),
      metadata_(metadata) {
  k_cache_in_.resize(metadata_.num_layers);
  v_cache_in_.resize(metadata_.num_layers);
  k_cache_out_.resize(metadata_.num_layers);
  v_cache_out_.resize(metadata_.num_layers);

  // Calculate I/O size
  Result<TensorInfo> attention_mask = method_meta->input_tensor_meta(1);
  Result<TensorInfo> logits = method_meta->output_tensor_meta(0);

  input_toks_.size = metadata_.ar_len * sizeof(int64_t);
  input_pos_.size = metadata_.ar_len * sizeof(int32_t);
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
  logits_.size =
      metadata_.ar_len * metadata_.vocab_size * logits_.getElementSize();
}
void TokenGenerator::init_io(
    IMemAlloc* buffer_manager,
    Result<MethodMeta> method_meta) {
  size_t idx = 0;
  input_tensors_.reserve(method_meta->num_inputs());
  output_tensors_.reserve(method_meta->num_outputs());
  // [I]: input_tokens
  Result<TensorInfo> input_toks = method_meta->input_tensor_meta(idx++);
  input_toks_.data =
      reinterpret_cast<int64_t*>(buffer_manager->allocate(input_toks_.size));
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
          const_cast<TensorImpl::DimOrderType*>(kv_cache->dim_order().data()));
      input_tensors_.emplace_back(cache[layer].get());
      cache_inputs_.emplace_back(input_tensors_.back());
      buffer_manager->add_memory_info(
          cache_ptrs[layer].buffer, cache[layer]->nbytes(), kv_cache.get());
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
  index = 1;
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

const std::vector<std::byte>& TokenGenerator::get_all_logits() {
  return token_all_logits_;
}

// This function only considers the case where token_generator_ar_len equals 1.
void TokenGenerator::prepare_io(uint64_t cur_token, int64_t start_pos) {
  // update input_tok
  *input_toks_.data =
      metadata_.use_int64_token ? cur_token : static_cast<int32_t>(cur_token);
  // update position_ids
  *input_pos_.data = static_cast<int32_t>(start_pos);
}

Result<int64_t> TokenGenerator::generate(
    std::vector<uint64_t> tokens,
    int64_t start_pos,
    int32_t seq_len,
    std::function<void(const std::string&)> token_callback,
    bool dump_logits,
    AttentionSinkRopeRunner* attention_sink_rope_runner) {
  ET_CHECK_MSG(
      !tokens.empty(), "Token generation loop shouldn't take empty tokens");
  int64_t pos = start_pos; // position in the sequence
  int shifted_pos = start_pos;
  bool enable_attention_sink = attention_sink_rope_runner != nullptr;
  const char* state_option = std::getenv("QNN_DECODE_STATE_PINGPONG");
  const bool state_pingpong = state_option && std::strcmp(state_option, "1") == 0 &&
      metadata_.ar_len == 1 && !enable_attention_sink && kv_manager_->has_state_layer();
  const char* inplace_option = std::getenv("QNN_DECODE_STATE_INPLACE");
  const bool state_inplace = !state_pingpong && inplace_option &&
      std::strcmp(inplace_option, "1") == 0 && metadata_.ar_len == 1 &&
      !enable_attention_sink && kv_manager_->has_state_layer();
  DecodeStateBuffers state_buffers;
  if (state_pingpong) {
    for (int layer = 0; layer < metadata_.num_layers; ++layer) {
      if (!kv_manager_->is_state_layer(layer)) continue;
      state_buffers.add(k_cache_in_[layer].get(), k_cache_out_[layer].get());
      state_buffers.add(v_cache_in_[layer].get(), v_cache_out_[layer].get());
    }
  }
  DecodeStateInplace state_inplace_buffers;
  if (state_inplace) {
    for (int layer = 0; layer < metadata_.num_layers; ++layer) {
      if (!kv_manager_->is_state_layer(layer)) continue;
      state_inplace_buffers.add(k_cache_in_[layer].get(), k_cache_out_[layer].get());
      state_inplace_buffers.add(v_cache_in_[layer].get(), v_cache_out_[layer].get());
    }
    ET_CHECK_OK_OR_RETURN_ERROR(decoder_runner_->set_outputs(method_name_, output_tensors_));
    ET_LOG(Info, "decode state in-place: output state tensors alias input buffers");
  }

  // Token after prefill
  uint64_t cur_token = tokens.back();
  generated_ids_.clear();
  generated_ids_.push_back(cur_token);
  uint64_t prev_token;
  // Rearrange KV cache first
  kv_manager_->rearrange_cache(metadata_.ar_len);
  std::vector<int32_t> attention_map(metadata_.ar_len);
  std::iota(attention_map.begin(), attention_map.end(), -1);

  // Initialize attention sink rope runner if given and update position
  // accordingly
  if (enable_attention_sink) {
    ET_CHECK_MSG(
        attention_sink_rope_runner->set_outputs(method_name_, cache_inputs_) ==
            executorch::runtime::Error::Ok,
        "Failed to set output tensor for module %s",
        method_name_.c_str());
    shifted_pos = pos - attention_sink_rope_runner->get_position_shift();
  }

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

  // Generate our tokens
  while (pos < seq_len - 1) {
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
    prepare_io(cur_token, shifted_pos);

    // 디코드 한 스텝을 구간별로 계측한다. NPU 그래프 실행(71ms)과 러너 전체(100ms)
    // 사이의 29ms 가 어디서 나오는지 보기 위한 것 — QNN_RUNNER_PROFILE=1 일 때만 집계.
    static const bool kProfile = getenv("QNN_RUNNER_PROFILE") != nullptr;
    static double t_step = 0, t_sample = 0, t_cache = 0, t_mask = 0, t_decode = 0;
    static int64_t n_prof = 0;
    auto now_us = []() {
      return std::chrono::duration_cast<std::chrono::microseconds>(
                 std::chrono::steady_clock::now().time_since_epoch())
          .count();
    };
    int64_t _t0 = kProfile ? now_us() : 0;

    // Run inference
    if (state_pingpong) {
      ET_CHECK_OK_OR_RETURN_ERROR(decoder_runner_->set_outputs(method_name_, output_tensors_));
    }
    auto logits_res = decoder_runner_->step(method_name_, inputs_);
    int64_t _t1 = kProfile ? now_us() : 0;
    if (dump_logits) {
      token_all_logits_.insert(
          token_all_logits_.end(),
          logits_.data,
          logits_.data +
              metadata_.ar_len * metadata_.vocab_size *
                  logits_.getElementSize());
    }
    ET_CHECK_OK_OR_RETURN_ERROR(logits_res.error());
    executorch::aten::Tensor& logits_tensor = logits_res.get();

    prev_token = cur_token;

    stats_->on_sampling_begin();
    cur_token =
        decoder_runner_->logits_to_token(logits_tensor, metadata_.ar_len);
    // Teacher forcing: the runner already fed forced[0]; step k feeds forced[k + 1].
    if (generated_ids_.size() < forced_tokens().size()) {
      cur_token = forced_tokens()[generated_ids_.size()];
    }
    generated_ids_.push_back(cur_token);
    stats_->on_sampling_end();
    int64_t _t2 = kProfile ? now_us() : 0;

    // Update KV Cache with the output results
    kv_manager_->update_cache(
        metadata_.ar_len, shifted_pos, metadata_.ar_len, {}, state_pingpong || state_inplace);
    if (state_pingpong) state_buffers.rotate();
    int64_t _t3 = kProfile ? now_us() : 0;
    // Update attention mask with current position
    kv_manager_->update_attention_mask(
        attention_mask_.data, metadata_.ar_len, shifted_pos, metadata_.ar_len);
    if (metadata_.cache_mode == CacheMode::HybridCache) {
      kv_manager_->update_attention_mask(
          window_attention_mask_.data,
          metadata_.ar_len,
          shifted_pos,
          metadata_.ar_len,
          metadata_.sliding_window);
    }
    int64_t _t4 = kProfile ? now_us() : 0;
    shifted_pos++;
    pos++;

    // print the token as string, decode it with the Tokenizer object
    ET_ASSIGN_OR_RETURN_TOKENIZER(
        decoded_token__, tokenizer_->decode(prev_token, cur_token));
    if (kProfile) {
      int64_t _t5 = now_us();
      t_step += (_t1 - _t0) / 1000.0;
      t_sample += (_t2 - _t1) / 1000.0;
      t_cache += (_t3 - _t2) / 1000.0;
      t_mask += (_t4 - _t3) / 1000.0;
      t_decode += (_t5 - _t4) / 1000.0;
      if (++n_prof % 32 == 0) {
        ET_LOG(
            Info,
            "[PROF] %lld tok: graph %.2f | sample %.2f | cache %.2f | mask %.2f | detok %.2f ms/tok",
            (long long)n_prof,
            t_step / n_prof,
            t_sample / n_prof,
            t_cache / n_prof,
            t_mask / n_prof,
            t_decode / n_prof);
      }
    }
    token_callback(decoded_token__);

    // data-dependent terminating condition: we have n_eos_ number of EOS
    if (eos_ids_->count(cur_token) > 0) {
      printf("\n");
      ET_LOG(Info, "\nReached to the end of generation");
      break;
    }
  }

  // Check if generation was truncated due to seq_len limit (no EOS token)
  if (eos_ids_->count(cur_token) == 0 && pos >= seq_len - 1) {
    printf("\n");
    ET_LOG(
        Info,
        "Warning: Generation stopped at seq_len limit (%d) without reaching EOS token. Response may be incomplete.",
        seq_len);
    if (!enable_attention_sink) {
      if (seq_len >= metadata_.context_len) {
        ET_LOG(
            Info,
            "- seq_len (%d) already equals compiled max_context_len (%d). Consider recompiling with larger --max_context_len.",
            seq_len,
            metadata_.context_len);
      } else {
        ET_LOG(
            Info,
            "- seq_len (%d) is less than compiled max_context_len (%d). Consider increasing --seq_len (up to %d).",
            seq_len,
            metadata_.context_len,
            metadata_.context_len);
      }
    }
  }

  return pos - start_pos;
}

} // namespace example
