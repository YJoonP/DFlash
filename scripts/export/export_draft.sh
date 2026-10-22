#!/usr/bin/env bash
# Export the DFlash draft z-lab/Qwen3.5-4B-DFlash (4-bit weights in blocks of 32, 16-bit activations) and turn it into
# the runner's draft folder: ART_DIR/ctx_draft/{forward_0.bin, forward_0_json.json, params.json, sha256.txt}, plus the
# fp16 token-embedding table the runner uses for the draft block (ART_DIR/draft_q35_embd_f16.bin, 1.2 GiB).
# usage: ART_DIR=<out> bash export_draft.sh SOC [BLOCK=16] [CTX=1024]     SOC = SM8850 (S26 Ultra) | SM8750 (S25)
# Needs (see scripts/env.sh): ET_DRAFT (ExecuTorch + executorch/patches/draft.patch), CKPT_DIR/qwen3_5_4b_dflash_meta.pth
# (examples/models/qwen3_5_dflash/convert_weights.py), CKPT_DIR/qwen3_5_4b_meta.pth (tied embedding), and
# HIDDEN_DUMP = target hidden states of the calibration prompts, made by
#   $PY $ET_DRAFT/examples/qualcomm/oss_scripts/llama/dump_target_hidden.py --calib calib_specbench39.json --out <file>
set -euo pipefail
SOC=${1:?SM8850 or SM8750}; BLOCK=${2:-16}; CTX=${3:-1024}
SCR=$(cd "$(dirname "$0")" && pwd)
source "$SCR/../env.sh"
: "${ET_DRAFT:?set ET_DRAFT}" "${CKPT_DIR:?set CKPT_DIR}" "${HIDDEN_DUMP:?set HIDDEN_DUMP}"
ART=${ART_DIR:?ART_DIR required}; mkdir -p "$ART"
for v in $(env | grep -oE '^QNN_[A-Z0-9_]+' | grep -vx QNN_SDK_ROOT); do unset $v; done
export ET=$ET_DRAFT PYTHONPATH=$(dirname "$ET_DRAFT")
export QNN_DFLASH_WEIGHT_BLOCK_SIZE=32 QNN_EXPORT_CPU_THREADS=8 QNN_SKIP_QDQ_EP=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE
T=$TMPROOT/export_draft_$$; mkdir -p "$T"
cd "$ET"
echo "[$(date '+%F %T KST')] draft export -> $ART" | tee "$ART/export.log"
TMPDIR=$T $PY examples/qualcomm/oss_scripts/llama/llama.py \
  --compile_only --soc_model "$SOC" --artifact "$ART" --build_folder "$ET/build-x86" \
  --decoder_model qwen3_5-4b-dflash --checkpoint "$CKPT_DIR/qwen3_5_4b_dflash_meta.pth" \
  --params "$ET/examples/models/qwen3_5_dflash/config/4b_dflash_config.json" \
  --model_mode hybrid --max_seq_len "$CTX" --prefill_ar_len "$BLOCK" \
  --calib_samples "$SCR/calib_specbench39.json" --dflash_hidden "$HIDDEN_DUMP" --temperature 0 \
  --prompt 'What is the capital of France?' 2>&1 | tee -a "$ART/export.log"
rm -rf "$T"
grep -q "Finish compile_only" "$ART/export.log" && grep -q "'block_size': (1, 32, 1, 1)" "$ART/export.log" \
  || { echo "draft export failed (see $ART/export.log)"; exit 1; }
PTE=$(ls "$ART"/*.pte | head -1)
bash "$SCR/extract_context.sh" "$PTE" "$ART/ctx_draft"
$PY "$SCR/make_params.py" --draft --block-size "$BLOCK" --out "$ART/ctx_draft/params.json" > /dev/null
$PY - "$ART/ctx_draft/params.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1])); p['weight_block_size'] = 32; json.dump(p, open(sys.argv[1], 'w'), indent=1)
PY
EMB="$ART/draft_q35_embd_f16.bin"
[ -f "$EMB" ] || $PY "$SCR/dump_draft_embd.py" --ckpt "$CKPT_DIR/qwen3_5_4b_meta.pth" --out "$EMB"
sha256sum "$PTE" "$ART/ctx_draft/forward_0.bin" "$EMB" | tee "$ART/ctx_draft/sha256.txt"
echo "[$(date '+%F %T KST')] draft context: $ART/ctx_draft (embedding: $EMB)"
