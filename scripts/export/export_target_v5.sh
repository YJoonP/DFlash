#!/usr/bin/env bash
# Export the Qwen3.5-4B target for DFlash verification, final version v5: the accepted rows of a verify call are committed
# by exact recomputation inside the next call (deferred state), every ingredient path on one quantization grid, and no
# conv-window change. SM8850 = Galaxy S26 Ultra (Hexagon v81), SM8750 = Galaxy S25 (Hexagon v79).
# usage: ART_DIR=<out> [CKPT_ROW=10|0] [U16IO=1|0] [GATE=<status.json>] [DELETE_PTE=1] bash export_target_v5.sh SOC MODE W
#   SOC   SM8850 | SM8750
#   MODE  full32 (the model) | diag8 (first 8 layers, for quick checks)
#   W     verify width (draft block size b): 16 | 8
#   CKPT_ROW=0    plain recompute (R16v5 in results/)    CKPT_ROW=10  ckpt10 split (K16on / K16off), the default
#   U16IO=0       float32 ingredient IO instead of 16-bit codes (built once, not measured)
#   GATE          status.json of another export: calibrate now, convert and compile only after that export ended
#   DELETE_PTE=1  remove the .pte at the end (the runner reads ctx_target/, not the .pte)
# Writes ART_DIR/ctx_target/{forward_<i>.bin, forward_<i>_json.json, params.json, shard_io_map.json, sha256.txt},
# ART_DIR/qdq_prefill.jsonl (input of scripts/check/*), ART_DIR/export_env.txt and ART_DIR/status.json.
# Needs about 100 GiB of available memory for full32 (60 GiB for diag8); run one export at a time.
set -uo pipefail
SOC=${1:?SM8850 or SM8750}; MODE=${2:?full32 or diag8}; W=${3:?16 or 8}
SCR=$(cd "$(dirname "$0")" && pwd)
source "$SCR/../env.sh"
: "${ET:?set ET (see scripts/env.sh)}" "${CKPT_DIR:?set CKPT_DIR}" "${OPPKG_X86:?set OPPKG_X86}"
ART=${ART_DIR:?ART_DIR required}; mkdir -p $ART
[ -e $ART/hybrid_llama_qnn.pte ] && { echo "existing export kept: $ART"; exit 5; }
for v in $(env | grep -oE '^QNN_[A-Z0-9_]+' | grep -vx QNN_SDK_ROOT); do unset $v; done
export PYTHONPATH=$(dirname "$ET")
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE QNN_EXPORT_CPU_THREADS=8
export QNN_DELTA_KERNEL=1 QNN_STATE_IO_16BIT=1 QNN_SKIP_QDQ_EP=1 QNN_DELTA_HEAD_GROUPS=4 QNN_DELTA_LOCAL_STATE=all
export QNN_DELTA_LOCAL_CONV_STATE=decode QNN_L2NORM_RMS=decode_direct QNN_DELTA_DECODE=off QNN_DELTA_STATE_SHARED_ENC=2
export QNN_DEFER_STATE=1 QNN_NORM_FOLD_FP32=1 QNN_CUSTOM_IO_DEBUG=1 QNN_DEFER_EXACT=1 QNN_DEFER_EXACT_U16IO=${U16IO:-1} QNN_DEFER_CHECKPOINT_ROW=${CKPT_ROW:-10} QNN_DEFER_EXACT_ONE_ENC=1 QNN_STATE_IO_FROM_Q=1 QNN_DEFER_SPLIT_Q_ON_GROUP=1 QNN_DEFER_EXACT_CONV=0
# OPPKG_DEVICE_DIR: phone folder of the op package recorded in the .pte (read by the ExecuTorch AR runner only; the
# DFlash runner gets the package from QNN_OP_PACKAGES). The default is the folder of our runs.
export QNN_DELTA_OPPKG_DIR=$OPPKG_X86 QNN_DELTA_OPPKG_DEVICE_DIR=${OPPKG_DEVICE_DIR:-/data/local/tmp/whdgns/q35_snapshot_w8_20261010/lib}
export QNN_QDQ_DUMP=$ART/qdq_prefill.jsonl
if [ "$MODE" = diag8 ]; then
  export QNN_SHARD_LAYERS=2,6; CKPT=$CKPT_DIR/qwen3_5_4b_meta_8L.pth; PARAMS=$ET/examples/models/qwen3_5/config/4b_config_diag8.json; NEED=60
else
  export QNN_SHARD_LAYERS=2,6,10,14,18,22,26,30; CKPT=$CKPT_DIR/qwen3_5_4b_meta.pth; PARAMS=$ET/examples/models/qwen3_5/config/4b_config.json; NEED=100
fi
avail=$(free -g | awk '/^Mem:/{print $7}'); [ "$avail" -ge $NEED ] || { echo "available ${avail}G < ${NEED}G, not starting"; exit 3; }
cp $SCR/calib_samples_b${W}.json $ART/calib_samples.json
env | grep -E "^QNN_|^PYTHONPATH|^OMP_NUM" | sort > $ART/export_env.txt
echo "start(KST): $(date '+%F %T') $SOC $MODE b$W avail=${avail}G"
cd $ET
ENTRY=examples/qualcomm/oss_scripts/llama/llama.py
if [ -n "${GATE:-}" ]; then export QNN_SNAPSHOT_EXPORT_GATE=$GATE; ENTRY=$SCR/export_entry.py; fi
echo '{"status": "exporting"}' > $ART/status.json
T=$TMPROOT/export_target_$$; mkdir -p $T
TMPDIR=$T $PY $ENTRY --compile_only --soc_model $SOC --artifact $ART --build_folder $ET/build-x86 \
  --decoder_model qwen3_5-4b --checkpoint $CKPT --params $PARAMS --model_mode hybrid --max_seq_len 1024 \
  --prefill_ar_len $W --calib_samples $ART/calib_samples.json --temperature 0 --prompt 'What is the capital of France?'
rc=$?; rm -rf $T; echo "export end(KST): $(date '+%F %T') rc=$rc"; [ $rc -eq 0 ] || { echo '{"status": "failed"}' > $ART/status.json; exit $rc; }
PTE=$ART/hybrid_llama_qnn.pte
bash $SCR/extract_context.sh "$PTE" "$ART/ctx_target" > $ART/extract.log 2>&1 || { echo "extract failed"; exit 6; }
$PY $SCR/make_params.py --target --out "$ART/ctx_target/params.json" --verify-ar-len $W --defer-masked 1 --target-config $PARAMS --shard-layers $QNN_SHARD_LAYERS > /dev/null || { echo "make_params failed"; exit 7; }
$PY $SCR/make_shard_io_map.py --pte "$PTE" --ctx "$ART/ctx_target" --ckpt-groups $([ "${CKPT_ROW:-10}" = 0 ] && echo 0 || echo 4) || { echo "shard map failed"; exit 8; }
sha256sum "$PTE" > "$ART/ctx_target/sha256.txt"
[ "${DELETE_PTE:-0}" = 1 ] && rm -f "$PTE" && echo "pte removed (DELETE_PTE=1); ctx_target kept"
echo '{"status": "exported"}' > $ART/status.json
echo "finish end(KST): $(date '+%F %T') ok"
