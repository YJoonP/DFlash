/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A llama 3.2 runner that includes preprocessing and post processing
// logic. The module takes in a string as input and emits a string as output.

#include <c10/util/safe_numerics.h>
#include <executorch/examples/models/llama/runner/runner.h>
#include <executorch/examples/models/llama/tokenizer/llama_tiktoken.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/client_mem.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/lhd_token_generator.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/rpc_mem.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/runner.h>
#include <executorch/extension/llm/runner/text_llm_runner.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/log.h>
#include <pytorch/tokenizers/hf_tokenizer.h>
#include <pytorch/tokenizers/llama2c_tokenizer.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>

using executorch::extension::llm::get_rss_bytes;
using executorch::extension::llm::print_report;
using executorch::extension::llm::Stats;
using executorch::extension::llm::time_in_ms;
using executorch::runtime::Error;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;
namespace llm = ::executorch::extension::llm;

namespace example {
namespace {
void print_performance_report(
    const Stats& stats,
    const std::string& performance_output_path) {
  // For now, we just print the total inference time for CI, can save more info
  // in future if needed.
  std::ofstream outfile(performance_output_path.c_str());
  if (outfile.is_open()) {
    double num_tok = 0;
    if (stats.num_generated_tokens == 0) {
      // For cases like evaluate perplexity where prompt_len == cache_len
      num_tok = ((stats.num_prompt_tokens)) /
          (double)(stats.prompt_eval_end_ms - stats.inference_start_ms) *
          stats.SCALING_FACTOR_UNITS_PER_SECOND;
    } else {
      num_tok = (stats.num_generated_tokens) /
          (double)(stats.inference_end_ms - stats.inference_start_ms) *
          stats.SCALING_FACTOR_UNITS_PER_SECOND;
    }

    outfile << num_tok;
    outfile.close();
  } else {
    ET_LOG(Error, "Error saving the inference speed file");
  }
}

void save_logits(
    const std::string& dump_logits_path,
    const std::vector<std::byte>& prefill_logits,
    const std::vector<std::byte>& decode_logits) {
  std::ofstream outFile(dump_logits_path.c_str(), std::ios::binary);
  if (outFile.is_open()) {
    outFile.write(
        reinterpret_cast<const char*>(prefill_logits.data()),
        prefill_logits.size());

    outFile.write(
        reinterpret_cast<const char*>(decode_logits.data()),
        decode_logits.size());
    outFile.close();
  } else {
    ET_CHECK_MSG(false, "Error saving the dump logits file");
  }
}

// PPL 평가용 축약 덤프. 위치(행) 하나당 vocab 개 logits 대신
//   lse = logsumexp(x), target_logit = x[target], argmax, target
// 만 남긴다. x = (q - zero_point) * scale (uint16 양자화 출력) 또는 float 원값.
// 평가기(device_evaluator.py)는 log p(target) = target_logit - lse 로 복원한다.
// 파일 형식: "CLG1" + int32 n_rows + int32 vocab + n_rows * CompactRow.
struct CompactRow {
  float lse;
  float target_logit;
  int32_t argmax;
  int32_t target;
};

template <typename T>
CompactRow reduce_row(
    const T* row,
    int32_t vocab,
    double scale,
    int32_t zp,
    int64_t target) {
  int32_t amax = 0;
  for (int32_t v = 1; v < vocab; ++v) {
    if (row[v] > row[amax]) { // 동률이면 첫 번째 (torch.argmax 와 동일)
      amax = v;
    }
  }
  const double xmax = (static_cast<double>(row[amax]) - zp) * scale;
  double sum = 0.0;
  for (int32_t v = 0; v < vocab; ++v) {
    sum += std::exp((static_cast<double>(row[v]) - zp) * scale - xmax);
  }
  CompactRow r;
  r.lse = static_cast<float>(xmax + std::log(sum));
  r.argmax = amax;
  r.target = (target >= 0 && target < vocab) ? static_cast<int32_t>(target) : -1;
  r.target_logit = r.target >= 0
      ? static_cast<float>((static_cast<double>(row[r.target]) - zp) * scale)
      : std::numeric_limits<float>::quiet_NaN();
  return r;
}

void save_logits_compact(
    const std::string& path,
    const std::vector<std::byte>& prefill_logits,
    const std::vector<std::byte>& decode_logits,
    executorch::aten::ScalarType dtype,
    int32_t vocab,
    double scale,
    int32_t zp,
    const std::vector<int64_t>& targets) {
  const size_t elem = getDtypeSize(dtype);
  const size_t row_bytes = static_cast<size_t>(vocab) * elem;
  const size_t total_rows =
      (prefill_logits.size() + decode_logits.size()) / row_bytes;
  const int32_t n_rows =
      static_cast<int32_t>(std::min(total_rows, targets.size()));
  std::ofstream out(path.c_str(), std::ios::binary);
  ET_CHECK_MSG(out.is_open(), "Error saving the compact logits file");
  out.write("CLG1", 4);
  out.write(reinterpret_cast<const char*>(&n_rows), sizeof(n_rows));
  out.write(reinterpret_cast<const char*>(&vocab), sizeof(vocab));
  for (int32_t r = 0; r < n_rows; ++r) {
    const size_t off = static_cast<size_t>(r) * row_bytes;
    const std::byte* p = off < prefill_logits.size()
        ? prefill_logits.data() + off
        : decode_logits.data() + (off - prefill_logits.size());
    CompactRow cr;
    switch (dtype) {
      case executorch::aten::ScalarType::UInt16:
        cr = reduce_row(
            reinterpret_cast<const uint16_t*>(p), vocab, scale, zp, targets[r]);
        break;
      case executorch::aten::ScalarType::Float:
        cr = reduce_row(
            reinterpret_cast<const float*>(p), vocab, scale, zp, targets[r]);
        break;
      default:
        ET_CHECK_MSG(false, "compact logits dump: unsupported logits dtype");
    }
    out.write(reinterpret_cast<const char*>(&cr), sizeof(cr));
  }
  out.close();
  ET_LOG(
      Info,
      "compact logits saved: %d rows x vocab %d -> %s",
      n_rows,
      vocab,
      path.c_str());
}

} // namespace

Runner::Runner(
    std::unique_ptr<executorch::extension::Module> module,
    const std::string& decoder_model_version,
    const std::string& model_path,
    const std::string& tokenizer_path,
    const std::string& dump_logits_path,
    const std::string& performance_output_path,
    const float temperature,
    const int eval_mode,
    const bool shared_buffer,
    const int ngram,
    const int window,
    const int gcap,
    std::unique_ptr<tokenizers::Tokenizer> tokenizer,
    std::unique_ptr<executorch::extension::Module> attention_sink_rope_module)
    : module_(std::move(module)),
      attention_sink_rope_module_(std::move(attention_sink_rope_module)),
      ngram_(ngram),
      window_(window),
      gcap_(gcap),
      tokenizer_path_(tokenizer_path),
      performance_output_path_(performance_output_path),
      dump_logits_path_(dump_logits_path),
      temperature_(temperature),
      eval_mode_(static_cast<EvalMode>(eval_mode)),
      shared_buffer_(shared_buffer),
      tokenizer_(std::move(tokenizer)) {
  stats_.reset();

  if (decoder_model_version == "llama2") {
    decoder_model_version_ = DecoderModelVersion::kLlama2;
  } else if (decoder_model_version == "llama3") {
    decoder_model_version_ = DecoderModelVersion::kLlama3;
  } else if (decoder_model_version == "gemma") {
    decoder_model_version_ = DecoderModelVersion::kGemma;
  } else if (decoder_model_version == "gemma2") {
    decoder_model_version_ = DecoderModelVersion::kGemma2;
    cache_mode_ = CacheMode::HybridCache;
  } else if (decoder_model_version == "gemma3") {
    decoder_model_version_ = DecoderModelVersion::kGemma3;
    cache_mode_ = CacheMode::HybridCache;
  } else if (decoder_model_version == "gemma4") {
    decoder_model_version_ = DecoderModelVersion::kGemma4;
    cache_mode_ = CacheMode::HybridCache;
  } else if (decoder_model_version == "granite") {
    decoder_model_version_ = DecoderModelVersion::kGranite;
  } else if (decoder_model_version == "phi_4_mini") {
    decoder_model_version_ = DecoderModelVersion::kPhi4;
  } else if (decoder_model_version == "qwen2_5") {
    decoder_model_version_ = DecoderModelVersion::kQwen2_5;
  } else if (decoder_model_version == "qwen3") {
    decoder_model_version_ = DecoderModelVersion::kQwen3;
  } else if (decoder_model_version == "qwen3_5") {
    decoder_model_version_ = DecoderModelVersion::kQwen3_5;
  } else if (decoder_model_version == "smollm2_135m") {
    decoder_model_version_ = DecoderModelVersion::kSmollm2_135m;
  } else if (decoder_model_version == "smollm3") {
    decoder_model_version_ = DecoderModelVersion::kSmollm3;
  } else if (decoder_model_version == "codegen") {
    decoder_model_version_ = DecoderModelVersion::kCodegen;
  } else if (decoder_model_version == "glm") {
    decoder_model_version_ = DecoderModelVersion::kGlm;
  } else {
    ET_CHECK_MSG(false, "Unsupported Decoder Model");
  }

  ET_LOG(Info, "creating module: model_path=%s", model_path.c_str());
  ET_LOG(Info, "creating runner: tokenizer_path=%s", tokenizer_path_.c_str());
  ET_LOG(Info, "eval mode=%d", eval_mode_);
}

bool Runner::is_loaded() const {
  return module_->is_loaded() && tokenizer_ && decoder_runner_ &&
      prompt_processor_ && token_generator_ && kv_manager_ && buffer_manager_;
}

Error Runner::load() {
  if (is_loaded()) {
    return Error::Ok;
  }

  std::string token_generator_method_name, prompt_processor_method_name;
  std::vector<std::string> method_names;
  switch (eval_mode_) {
    case EvalMode::kKVCached:
      prompt_processor_method_name = "kv_forward";
      token_generator_method_name = "kv_forward";
      method_names.emplace_back(token_generator_method_name);
      break;
    case EvalMode::kHybrid:
    case EvalMode::kLookaheadDecoding:
      prompt_processor_method_name = "prefill_forward";
      token_generator_method_name = "kv_forward";
      method_names.emplace_back(prompt_processor_method_name);
      method_names.emplace_back(token_generator_method_name);
      break;
    case EvalMode::kUnsupported:
      ET_CHECK_MSG(false, "Unsupported llama evaluation mode");
      break;
  }
  auto eos_ids = std::make_unique<std::unordered_set<uint64_t>>();
  if (tokenizer_ != nullptr) {
    eos_ids->insert(tokenizer_->encode("<|eot_id|>", 0, 0).get()[0]);
    eos_ids->insert(tokenizer_->encode("<|eot|>", 0, 0).get()[0]);
    eos_ids->insert(tokenizer_->encode("<|end_of_text|>", 0, 0).get()[0]);
  } else {
    tokenizer_ = llm::load_tokenizer(tokenizer_path_);
    if (tokenizer_ == nullptr) {
      ET_LOG(
          Error, "Failed to load tokenizer with %s", tokenizer_path_.c_str());
      return Error::Internal;
    }
    eos_ids->insert(tokenizer_->eos_tok());
  }
  if (decoder_model_version_ == DecoderModelVersion::kLlama3) {
    eos_ids->insert(tokenizer_->encode("<|eot_id|>", 0, 0).get()[0]);
  } else if (decoder_model_version_ == DecoderModelVersion::kPhi4) {
    eos_ids->insert(tokenizer_->encode("<|end|>", 0, 0).get()[0]);
  } else if (
      decoder_model_version_ == DecoderModelVersion::kQwen3 ||
      decoder_model_version_ == DecoderModelVersion::kQwen3_5 ||
      decoder_model_version_ == DecoderModelVersion::kSmollm2_135m ||
      decoder_model_version_ == DecoderModelVersion::kSmollm3) {
    eos_ids->insert(tokenizer_->encode("<|im_end|>", 0, 0).get()[0]);
  } else if (
      decoder_model_version_ == DecoderModelVersion::kGemma ||
      decoder_model_version_ == DecoderModelVersion::kGemma2 ||
      decoder_model_version_ == DecoderModelVersion::kGemma3) {
    eos_ids->insert(tokenizer_->encode("<end_of_turn>", 0, 0).get()[0]);
  } else if (decoder_model_version_ == DecoderModelVersion::kGemma4) {
    eos_ids->insert(tokenizer_->encode("<turn|>", 0, 0).get()[0]);
  } else if (decoder_model_version_ == DecoderModelVersion::kCodegen) {
    eos_ids->insert(tokenizer_->encode("<|endoftext|>", 0, 0).get()[0]);
  } else if (decoder_model_version_ == DecoderModelVersion::kGlm) {
    eos_ids->insert(tokenizer_->encode("<|user|>", 0, 0).get()[0]);
  }

  Result<MethodMeta> method_meta =
      module_->method_meta(token_generator_method_name);

  // For some tokenizer.json, runtime vocab_size might be different, use output
  // shape to get vocab size.
  int32_t vocab_size = method_meta->output_tensor_meta(0)->sizes()[2];
  vocab_size_ = vocab_size;
  logits_dtype_ = method_meta->output_tensor_meta(0)->scalar_type();
  decoder_runner_ =
      std::make_unique<DecoderRunner>(module_.get(), vocab_size, temperature_);

  ET_CHECK_OK_OR_RETURN_ERROR(decoder_runner_->load(method_names));

  ET_LOG(Info, "Reading metadata from model");
  // retrieve any method meta, can be either prefill or kv
  ET_ASSIGN_OR_RETURN(num_layers_evalue__, module_->get("get_n_layers"));
  int64_t num_layers = num_layers_evalue__.toScalar().to<int64_t>();

  ET_CHECK_MSG(num_layers != -1, "Could not retrieve num layers");
  // k_cache: [1, n_heads, head_dim, seq_len]
  // Hybrid models (e.g. Qwen3.5) place recurrent-state slots in the leading
  // cache positions. A state slot comes back with exactly the shape it was
  // given while a KV cache does not, so scan for the first real KV layer
  // instead of assuming layer 0 is one.
  const size_t first_cache_input_idx =
      method_meta->num_inputs() - 2 * num_layers;
  int64_t num_heads = -1;
  for (int64_t layer = 0; layer < num_layers; ++layer) {
    auto k_out_sizes = method_meta->output_tensor_meta(1 + layer)->sizes();
    auto k_in_sizes =
        method_meta->input_tensor_meta(first_cache_input_idx + layer)->sizes();
    bool same = k_out_sizes.size() == k_in_sizes.size();
    for (size_t i = 0; same && i < k_out_sizes.size(); ++i)
      same = k_out_sizes[i] == k_in_sizes[i];
    if (!same) {
      num_heads = k_out_sizes[1];
      break;
    }
  }
  ET_CHECK_MSG(
      num_heads != -1, "Could not find a KV cache layer to read n_heads from");
  ET_LOG(
      Info,
      "n_kv_heads=%d (from first non-state cache layer)",
      static_cast<int>(num_heads));
  bool use_int64_token = method_meta->input_tensor_meta(0)->scalar_type() ==
      executorch::aten::ScalarType::Long;

  // Use attention mask length to retrieve AR length and context length
  // Cache len equals to context_len - ar_len
  int32_t prompt_processor_ar_len = 0;
  int32_t token_generator_ar_len = 0;
  int32_t max_cache_len = 0;
  int32_t max_ar_len = 0;
  // atten mask: [1, 1, AR-N, CL]
  auto atten_mask_meta_token = method_meta->input_tensor_meta(1);
  token_generator_ar_len = atten_mask_meta_token->sizes()[2];
  context_len_ = atten_mask_meta_token->sizes()[3];
  if (eval_mode_ == EvalMode::kKVCached) {
    prompt_processor_ar_len = token_generator_ar_len;
  } else if (
      eval_mode_ == EvalMode::kHybrid ||
      eval_mode_ == EvalMode::kLookaheadDecoding) {
    auto atten_mask_meta_prompt =
        module_->method_meta(prompt_processor_method_name)
            ->input_tensor_meta(1);
    prompt_processor_ar_len = atten_mask_meta_prompt->sizes()[2];
  }
  if (prompt_processor_ar_len == context_len_)
    max_cache_len = context_len_;
  else
    max_cache_len = context_len_ -
        std::min(token_generator_ar_len, prompt_processor_ar_len);
  max_ar_len = std::max(token_generator_ar_len, prompt_processor_ar_len);

  // Load the sliding window size if the model supports it.
  // This is used to configure the attention mask for models with window
  // attention
  int32_t sliding_window = context_len_;
  if (module_->method_names()->count("get_sliding_window") > 0) {
    ET_ASSIGN_OR_RETURN(
        sliding_window_evalue__, module_->get("get_sliding_window"));
    sliding_window = sliding_window_evalue__.toInt();
  }
  // Gemma4 uses YOCO: only self-decoder layers have KV I/O; use
  // get_n_self_layers when available so KVManager allocates the correct number
  // of caches.
  int64_t num_kv_layers = num_layers;
  if (module_->method_names()->count("get_n_self_layers") > 0) {
    ET_ASSIGN_OR_RETURN(num_layers_evalue__, module_->get("get_n_self_layers"));
    num_kv_layers = num_layers_evalue__.toScalar().to<int64_t>();
  }
  kv_manager_ = std::make_unique<KVManager>(
      KVManager::Metadata{
          context_len_, max_ar_len, max_cache_len, num_heads, num_kv_layers},
      std::make_unique<MethodMeta>(
          std::move(module_->method_meta(token_generator_method_name).get())));

  if (attention_sink_rope_module_ != nullptr) {
    attention_sink_rope_runner_ = std::make_unique<AttentionSinkRopeRunner>(
        attention_sink_rope_module_.get());
    ET_CHECK_OK_OR_RETURN_ERROR(
        attention_sink_rope_runner_->load(method_names));
  }

  prompt_processor_ = std::make_unique<PromptProcessor>(
      decoder_runner_.get(),
      kv_manager_.get(),
      prompt_processor_method_name,
      PromptProcessor::Metadata{
          context_len_,
          num_heads,
          num_kv_layers,
          prompt_processor_ar_len,
          vocab_size,
          use_int64_token,
          sliding_window,
          cache_mode_},
      std::make_unique<MethodMeta>(
          std::move(module_->method_meta(prompt_processor_method_name).get())));
  if (eval_mode_ == EvalMode::kLookaheadDecoding) {
    token_generator_ = std::make_unique<LhdTokenGenerator>(
        tokenizer_.get(),
        decoder_runner_.get(),
        kv_manager_.get(),
        token_generator_method_name,
        std::move(eos_ids),
        LhdTokenGenerator::Metadata{
            context_len_,
            num_heads,
            num_kv_layers,
            token_generator_ar_len,
            vocab_size,
            use_int64_token,
            ngram_,
            window_,
            gcap_,
            sliding_window,
            cache_mode_},
        &stats_,
        std::make_unique<MethodMeta>(std::move(
            module_->method_meta(token_generator_method_name).get())));
  } else {
    token_generator_ = std::make_unique<TokenGenerator>(
        tokenizer_.get(),
        decoder_runner_.get(),
        kv_manager_.get(),
        token_generator_method_name,
        std::move(eos_ids),
        TokenGenerator::Metadata{
            context_len_,
            num_heads,
            num_kv_layers,
            token_generator_ar_len,
            vocab_size,
            use_int64_token,
            sliding_window,
            cache_mode_},
        &stats_,
        std::make_unique<MethodMeta>(std::move(
            module_->method_meta(token_generator_method_name).get())));
  }

  buffer_manager_ = std::make_unique<ClientMem>();
  if (shared_buffer_) {
    buffer_manager_ = std::make_unique<RpcMem>(
        kv_manager_->total_cache_size_in_bytes(),
        prompt_processor_->total_prompt_processor_io_size_in_bytes(),
        token_generator_->total_token_generator_io_size_in_bytes());
  }
  // Quantized recurrent state (Qwen3.5 Gated DeltaNet): the buffers must start
  // at the real value 0, which for an asymmetric encoding is the zero point,
  // not an all-zero buffer. The exporter writes one method per cache slot;
  // outputs are ordered [logits, k_0..k_{L-1}, v_0..v_{L-1}], so layer L uses
  // index L for its k slot and num_layers+L for its v slot.
  if (kv_manager_->has_state_layer()) {
    std::vector<int32_t> k_zp(num_layers, -1), v_zp(num_layers, -1);
    for (int64_t layer = 0; layer < num_layers; ++layer) {
      if (!kv_manager_->is_state_layer(static_cast<int32_t>(layer)))
        continue;
      auto k_res = module_->get(
          "get_state_output_" + std::to_string(layer) + "_zero_point");
      auto v_res = module_->get(
          "get_state_output_" + std::to_string(num_layers + layer) +
          "_zero_point");
      if (k_res.ok())
        k_zp[layer] = static_cast<int32_t>(k_res.get().toScalar().to<int64_t>());
      if (v_res.ok())
        v_zp[layer] = static_cast<int32_t>(v_res.get().toScalar().to<int64_t>());
    }
    kv_manager_->set_state_zero_points(std::move(k_zp), std::move(v_zp));
  }

  ET_LOG(Info, "creating io_memory");
  // prepare io
  kv_manager_->init_cache(buffer_manager_.get(), prompt_processor_ar_len);
  prompt_processor_->init_io(
      buffer_manager_.get(),
      module_->method_meta(prompt_processor_method_name));
  token_generator_->init_io(
      buffer_manager_.get(), module_->method_meta(token_generator_method_name));
  return Error::Ok;
}

Error Runner::generate(
    const std::string& prompt,
    const llm::GenerationConfig& config,
    std::function<void(const std::string&)> token_callback,
    std::function<void(const Stats&)> stats_callback) {
  return generate_from_prompt_or_file(
      prompt, false, config, token_callback, stats_callback);
}

Error Runner::generate_from_prompt_or_file(
    const std::string& prompt,
    bool tokenized_prompt,
    const llm::GenerationConfig& config,
    std::function<void(const std::string&)> token_callback,
    std::function<void(const Stats&)> stats_callback) {
  ET_CHECK_MSG(!prompt.empty(), "prompt cannot be null");
  if (!is_loaded()) {
    stats_.model_load_start_ms = time_in_ms();
    ET_CHECK_OK_OR_RETURN_ERROR(load());
    stats_.model_load_end_ms = time_in_ms();
  }
  stats_.inference_start_ms = time_in_ms();

  int32_t seq_len = config.seq_len;
  if (attention_sink_rope_runner_ == nullptr && seq_len > context_len_) {
    ET_LOG(
        Info,
        "Warning: Requested seq_len (%d) exceeds compiled max_context_len (%d) without attention sink. Clamping to %d.",
        seq_len,
        context_len_,
        context_len_);
    seq_len = context_len_;
  } else if (seq_len <= 0) {
    ET_LOG(
        Info,
        "Warning: Invalid seq_len (%d). Using compiled max_context_len (%d).",
        seq_len,
        context_len_);
    seq_len = context_len_;
  }
  int32_t n_bos = (cur_pos_ == 0) ? 1 : 0;

  // encode the (string) prompt into tokens sequence
  std::vector<uint64_t> prompt_tokens;
  if (tokenized_prompt) {
    std::ifstream inFile(prompt, std::ios::binary);
    if (inFile.is_open()) {
      // Get file size
      inFile.seekg(0, std::ios::end);
      size_t fileSize = inFile.tellg();
      inFile.seekg(0, std::ios::beg);

      // Resize vector and read raw data
      prompt_tokens.resize(fileSize / sizeof(uint64_t));

      inFile.read(reinterpret_cast<char*>(prompt_tokens.data()), fileSize);
      inFile.close();
    } else {
      ET_CHECK_MSG(
          false,
          "Unable to read tokenized prompt from file: %s",
          prompt.c_str());
    }
  } else {
    tokenizers::Result<std::vector<uint64_t>> encode_res =
        tokenizer_->encode(prompt, n_bos, 0);
    ET_CHECK_TK_OK_OR_RETURN_ERROR(
        encode_res.error(), "failed to encode prompt %s", prompt.c_str());
    prompt_tokens = encode_res.get();
  }
  int num_prompt_tokens = prompt_tokens.size();
  ET_CHECK_MSG(num_prompt_tokens >= 1, "Expected at least 1 prompt token");
  int64_t end_pos = 0;
  ET_CHECK_MSG(
      !c10::add_overflows(
          cur_pos_, static_cast<int64_t>(num_prompt_tokens), &end_pos) &&
          end_pos < static_cast<int64_t>(seq_len),
      "sequence length exceeded - please increase the seq_len value");

  // Prompt Processor first
  if (token_callback && config.echo) {
    token_callback(prompt);
  }
  bool dump_logits = dump_logits_path_.empty() ? false : true;
  auto prefill_res = prompt_processor_->prefill(
      prompt_tokens, cur_pos_, dump_logits, attention_sink_rope_runner_.get());
  ET_CHECK_OK_OR_RETURN_ERROR(prefill_res.error());
  uint64_t cur_token = prefill_res.get();
  cur_pos_ += num_prompt_tokens;
  stats_.first_token_ms = time_in_ms();
  stats_.prompt_eval_end_ms = time_in_ms();

  // print the first token from prefill. No prev_token so use cur_token for
  // it.
  if (token_callback) {
    ET_ASSIGN_OR_RETURN_TOKENIZER(
        decoded_token__, tokenizer_->decode(cur_token, cur_token));
    token_callback(decoded_token__);
  }
  ET_LOG(
      Info,
      "RSS after prompt prefill: %f MiB (0 if unsupported)",
      get_rss_bytes() / 1024.0 / 1024.0);

  // start the main loop
  prompt_tokens.push_back(cur_token);
  ET_ASSIGN_OR_RETURN(
      num_generated_tokens,
      token_generator_->generate(
          prompt_tokens,
          cur_pos_,
          seq_len,
          token_callback,
          dump_logits,
          attention_sink_rope_runner_.get()));
  stats_.inference_end_ms = time_in_ms();
  ET_LOG(
      Info,
      "RSS after finishing text generation: %f MiB (0 if unsupported)",
      get_rss_bytes() / 1024.0 / 1024.0);
  cur_pos_ += num_generated_tokens;
  if (cur_pos_ == seq_len) {
    ET_LOG(Info, "Sequence length (%i tokens) reached!", seq_len);
  }

  stats_.num_prompt_tokens = num_prompt_tokens;
  stats_.num_generated_tokens = num_generated_tokens;
  print_report(stats_);
  if (!dump_token_ids_path_.empty()) {
    // 1행: 프롬프트 id (prefill 출력 토큰 제외), 2행: 생성 id. 공백 구분.
    std::ofstream f(dump_token_ids_path_.c_str());
    if (f.is_open()) {
      for (int i = 0; i < num_prompt_tokens; ++i) f << prompt_tokens[i] << (i + 1 < num_prompt_tokens ? " " : "");
      f << "\n";
      const auto& g = token_generator_->get_generated_ids();
      for (size_t i = 0; i < g.size(); ++i) f << g[i] << (i + 1 < g.size() ? " " : "");
      f << "\n";
    }
  }
  print_performance_report(stats_, performance_output_path_);
  if (dump_logits && dump_logits_compact_) {
    // 정답 토큰: 파일이 있으면 그것, 없으면 prompt 를 한 칸 민 것(마지막 행은 정답 없음).
    std::vector<int64_t> targets;
    if (!dump_targets_path_.empty()) {
      std::ifstream tf(dump_targets_path_, std::ios::binary);
      ET_CHECK_MSG(
          tf.is_open(),
          "Unable to read targets from file: %s",
          dump_targets_path_.c_str());
      tf.seekg(0, std::ios::end);
      const size_t n = static_cast<size_t>(tf.tellg()) / sizeof(uint64_t);
      tf.seekg(0, std::ios::beg);
      std::vector<uint64_t> raw(n);
      tf.read(reinterpret_cast<char*>(raw.data()), n * sizeof(uint64_t));
      targets.assign(raw.begin(), raw.end());
    } else {
      for (int i = 0; i < num_prompt_tokens; ++i) {
        targets.push_back(
            i + 1 < num_prompt_tokens ? static_cast<int64_t>(prompt_tokens[i + 1])
                                      : -1);
      }
    }
    save_logits_compact(
        dump_logits_path_,
        prompt_processor_->get_all_logits(),
        token_generator_->get_all_logits(),
        logits_dtype_,
        vocab_size_,
        logits_scale_,
        logits_zero_point_,
        targets);
  } else if (dump_logits) {
    save_logits(
        dump_logits_path_,
        prompt_processor_->get_all_logits(),
        token_generator_->get_all_logits());
  }
  if (stats_callback) {
    stats_callback(stats_);
  }
  return Error::Ok;
}

Result<DecoderModelVersion> Runner::get_decoder_model_version() {
  if (!is_loaded()) {
    stats_.model_load_start_ms = time_in_ms();
    ET_CHECK_OK_OR_RETURN_ERROR(load());
    stats_.model_load_end_ms = time_in_ms();
  }
  return decoder_model_version_;
}

} // namespace example
