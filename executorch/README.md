# ExecuTorch changes

Everything here is relative to upstream ExecuTorch commit
[`c27baa8031`](https://github.com/pytorch/executorch/commit/c27baa8031862fb7528bb87616d6793a032a1257)
("Qualcomm AI Engine Direct - [GenAI Pipeline] PR6"). There are two source trees, because the target and the draft were
exported from different snapshots of our fork.

| | Target (Qwen3.5-4B), final v5 | Draft (z-lab/Qwen3.5-4B-DFlash) |
|---|---|---|
| Patch | `patches/target_v5.patch` (53 files, +5802 / −202) | `patches/draft.patch` (56 files, +5337 / −171) |
| Changed files, full text | `target_v5/` | `draft/` |
| Used for | every target export in `results/` (recompute v5, ckpt10 v5) | the b = 16 draft of both phones |

`target_v5/` and `draft/` hold the full text of each added or changed file, under the same path as in ExecuTorch, so the
code can be read here. Applying a patch to a clean checkout gives exactly these files:

```bash
git clone https://github.com/pytorch/executorch.git && cd executorch
git checkout c27baa8031862fb7528bb87616d6793a032a1257
git submodule update --init --recursive
git apply /path/to/DFlash/executorch/patches/target_v5.patch      # or draft.patch, in a second checkout
# then build the QNN backend for the host (backends/qualcomm/scripts/build.sh, QAIRT 2.49) into build-x86/
```

The files are the ones we exported with, except that paths of our server in defaults and comments were replaced (for
example the default of `QNN_DELTA_OPPKG_DIR`, which the export script always sets). Upstream ExecuTorch is
BSD-licensed (`LICENSE`); the original copyright headers are kept.

## Main pieces (target)

| File | What it does |
|---|---|
| `examples/qualcomm/oss_scripts/llama/model/gated_deltanet.py` | Gated DeltaNet layer for the static QNN graph: explicit conv / recurrent state inputs and outputs, causal conv1d as a depthwise conv2d, the deferred-state path (the accepted rows of the previous verify call are committed at the start of the next call) |
| `.../model/delta_rule_op.py` | The delta-rule recurrence as one custom op family (`qwen35::delta_rule_*`, CPU reference implementations), its QNN op-package registration, and the graph passes that put every ingredient path on one quantization grid (`QNN_DEFER_EXACT*`) |
| `.../model/static_llama.py` | Hybrid decoder (24 DeltaNet + 8 attention layers) with the deferred-state inputs and outputs (`valid_prev`, ingredients, `valid_cur`, `conv_idx_prev`, the ckpt10 checkpoint states) |
| `.../model/snapshot_outputs.py` | The state-saving ("snapshot") baseline: exposes the per-row states of the verify call |
| `.../wrappers/llm_wrappers.py` | Export pipeline: calibration, custom 16-bit IO, encoding passes, uneven sharding, the quantized-graph dump read by `scripts/check/` |
| `.../static_llm_quant_recipe.py` | Quantization recipe: 16-bit activations, 4-bit linear weights in blocks of 32, 8-bit lm_head, 8-bit KV cache, 16-bit DeltaNet state |
| `extension/llm/custom_ops/model_sharding.py` | Explicit shard boundaries (`QNN_SHARD_LAYERS=2,6,…,30`), so the hidden states the draft reads (layers 1, 5, …, 29) are shard outputs |
| `backends/qualcomm/builders/op_custom_op.py`, `quantizer/`, `partition/` | Per-output encodings of multi-output custom ops, custom annotations on the recipe path, placement of graph-output sinks |
| `backends/qualcomm/runtime/*` | Zero copy (shared buffers) for large models, rpcmem flags, profiling events used in the latency breakdown |
| `examples/qualcomm/oss_scripts/llama/runner/*`, `qnn_llama_runner.cpp` | ExecuTorch runner used for the autoregressive (AR) baseline: hybrid state handling, Qwen3.5 chat template, NEON argmax |

## Draft-only pieces

| File | What it does |
|---|---|
| `examples/qualcomm/oss_scripts/llama/model/dflash_draft.py` | DFlash draft as one graph per step: block = [last accepted token, MASK × (b − 1)]; the 8 target hidden states (layers 1, 5, …, 29) go through `fc` into a second KV source of every draft layer |
| `examples/models/qwen3_5_dflash/convert_weights.py`, `config/4b_dflash_config.json` | z-lab checkpoint → static-model names; the tied embedding / lm_head come from the target checkpoint |
| `examples/qualcomm/oss_scripts/llama/dump_target_hidden.py` | Target hidden states of the calibration prompts (the draft's calibration input) |

The draft path started from the Qwen3-4B DFlash export in [QNN_Exporter](https://github.com/mkjsym/QNN_Exporter)
(`dflash_draft.py`, and the context-extraction / params helpers in `scripts/export/`).

## Export switches

The measured exports set the environment variables in `scripts/export/export_target_v5.sh`; the exact list of each export
is in `results/exports/*/export_env.txt`. The ones specific to DFlash verification:

| Variable | Meaning |
|---|---|
| `QNN_DEFER_STATE=1` | The verify graph does not write the DeltaNet state; it outputs the ingredients (k, v, β, decay, conv input) and commits the accepted rows in the next call |
| `QNN_DEFER_EXACT=1`, `QNN_DEFER_EXACT_U16IO=1` | The commit reads the ingredients as 16-bit codes on the same encoding the verify side used, so the recomputed state equals the verified one |
| `QNN_DEFER_EXACT_ONE_ENC=1`, `QNN_STATE_IO_FROM_Q=1`, `QNN_DEFER_SPLIT_Q_ON_GROUP=1` | One quantization grid along every ingredient path, state inputs read on their stored encoding, split query slices on the group encoding |
| `QNN_DEFER_CHECKPOINT_ROW=10` | ckpt10: the verify op is split at row 10 and the row-10 state is an output; `0` = plain recompute |
| `QNN_DEFER_EXACT_CONV=0` | v5: no conv-window encoding change (it cost a requantize per DeltaNet layer and was not needed for exactness) |
