#!/bin/bash
# Exactness test of a ckpt10 target on the S26 Ultra (results/s26_v5_20261013/diag): teacher-forced replay of the prompt-81
# AR token stream ($PH/logs/diag/ar_ids_81.txt) with a diagnosis build of the runner (DFLASH_REPLAY_SUMS = logit
# fingerprint per position, DIAG_DUMP_STATE_* = committed DeltaNet state dump):
#   off g16 / on g16 : same tokens and grouping; "on" starts each commit from the row-10 checkpoint. Exact <=> identical
#                      logit fingerprints and identical committed states (dump after the 2nd prompt call, shard 0)
#   off g1 / off g4  : grouping dependence of the plain recompute
# usage: S26_SERIAL=<adb serial> [ADB=adb] [PY=python3] [RESULTS_DIR=<folder>] test_exact_device.sh TAG PHONE_CTX_DIR
TAG=${1:?tag}; CTX=${2:?phone ctx dir}
PH=/data/local/tmp/whdgns/q35_s26_20261012
D=${RESULTS_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)/results/s26_v5_20261013}/diag/exact_$TAG
PY=${PY:-python3}
ADB=${ADB:-adb}; SERIAL=${S26_SERIAL:?set S26_SERIAL}
mkdir -p $D
adbs() { timeout 600 $ADB -s $SERIAL shell "$@"; }
adbs "mkdir -p $PH/logs/diag/$TAG; rm -f $PH/logs/diag/$TAG/*"
run() {   # name group extra-env
  adbs "cd $PH && timeout 300 env 'ADSP_LIBRARY_PATH=$PH/lib_v1p;$PH/qnnlib' QNN_OP_PACKAGES=$PH/lib_v1p/libQnnDeltaRuleOpPackage_HTP.so:DeltaRuleOpPackageInterfaceProvider:HTP $3 DIAG_DUMP_STATE_CALL=2 DIAG_DUMP_STATE_PATH=$PH/logs/diag/$TAG/$1 LD_LIBRARY_PATH=$PH/lib:$PH/qnnlib DFLASH_AR=1 DFLASH_REPLAY_IDS=$PH/logs/diag/ar_ids_81.txt DFLASH_REPLAY_GROUP=$2 DFLASH_REPLAY_SUMS=$PH/logs/diag/$TAG/sums_$1.txt DFLASH_CHAT=1 DFLASH_THINK=1 DFLASH_BLOCK=16 $PH/runner_diag4 --qnn --multi-context --ctx-dir $CTX --params $CTX/params.json --tokenizer $PH/Qwen3.5-4B-q4_0-outq4.gguf --log-level 1 -n 128 -c 1024 --temp 0.0 -f $PH/inputs/prompt_81.txt > $PH/logs/diag/$TAG/run_$1.log 2>&1; echo $1 exit \$?; grep 'REPLAY\]' $PH/logs/diag/$TAG/run_$1.log"
  timeout 60 $ADB -s $SERIAL shell "cat $PH/logs/diag/$TAG/sums_$1.txt" > $D/sums_$1.txt
}
run off_g16 16 ""
run on_g16 16 "QNN_DEFER_CHECKPOINT_ROW=10"
run off_g1 1 ""
run off_g4 4 ""
for f in off_g16_s0_rec_L0 on_g16_s0_rec_L0 off_g16_s0_rec_L1 on_g16_s0_rec_L1; do
  timeout 120 $ADB -s $SERIAL pull $PH/logs/diag/$TAG/$f.bin $D/$f.bin > /dev/null 2>&1
done
$PY - $D <<'PY'
import sys, numpy as np
D = sys.argv[1]
L = lambda n: [l.split() for l in open(f'{D}/sums_{n}.txt') if l.strip()]
for a, b in (('off_g16', 'on_g16'), ('off_g1', 'off_g4'), ('off_g1', 'off_g16')):
    x, y = L(a), L(b)
    d = [i for i, (p, q) in enumerate(zip(x, y)) if p[3:] != q[3:]]
    print(f'{a} vs {b}: {len(d)} of {min(len(x), len(y))} positions differ (logit fingerprint); first {d[:6]}')
for Lr in (0, 1):
    try:
        a = np.fromfile(f'{D}/off_g16_s0_rec_L{Lr}.bin', dtype=np.uint16).astype(np.int32)
        b = np.fromfile(f'{D}/on_g16_s0_rec_L{Lr}.bin', dtype=np.uint16).astype(np.int32)
    except FileNotFoundError:
        continue
    d = b - a
    print(f'committed state after the 2nd prompt call, layer {Lr}: {np.count_nonzero(d)} of {d.size} codes differ, max |diff| {np.abs(d).max()}')
PY
