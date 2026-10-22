#!/usr/bin/env bash
# .pte -> 런타임이 읽는 forward_<i>.bin + forward_<i>_json.json  (QNN_Exporter/scripts/extract_context.sh 판)
#   bash extract_context.sh <pte> <out_dir> [params.json]
# ExecuTorch 페이로드에는 QNN 컨텍스트 앞에 포장 헤더(0x5678abcd)가 붙어 있어 dump_context_from_pte
# (StripProtocol)로 벗긴다. 샤드 i 의 컨텍스트 하나에 kv_forward/prefill_forward 두 그래프가 함께 들어 있으므로
# kv_forward_<i>.bin 만 forward_<i>.bin 으로 쓴다. 중간 파일은 $TMPROOT 에서 만들고 지운다. ET 는 패치한 ExecuTorch.
set -euo pipefail
source "$(dirname "$0")/../env.sh"
: "${ET:?set ET}"
PTE=$(readlink -f "${1:?pte path}"); OUT=$(readlink -m "${2:?out dir}"); PARAMS=${3:-}
# OUT 은 반드시 절대경로로. 아래에서 cd "$ET" 를 하므로 상대경로면 엉뚱한 곳에 만들어진다 (2026-09-16 실측).
Q="$QNN_SDK_ROOT/bin/x86_64-linux-clang"
TMP=$TMPROOT/extract_ctx_$$
mkdir -p "$TMP"; trap 'rm -rf "$TMP"' EXIT
ln -s "$PTE" "$TMP/model.pte"
cd "$ET"
$PY -c "
from executorch.backends.qualcomm.utils.utils import dump_context_from_pte
fs = dump_context_from_pte('$TMP/model.pte')
print('dumped', len(fs), 'binaries')" 2>&1 | grep -E "dumped|Error|Traceback"
rm -rf "$OUT"; mkdir -p "$OUT"
for f in "$TMP"/kv_forward_*.bin; do
  i=$(basename "$f" .bin); i=${i#kv_forward_}
  p="$TMP/prefill_forward_$i.bin"
  if [ -f "$p" ] && [ "$(stat -c %s "$p")" != "$(stat -c %s "$f")" ]; then
    echo "warning: shard $i kv/prefill blobs differ in size (separate contexts?) -> keeping both" >&2
    mv "$p" "$OUT/prefill_forward_$i.bin"
  fi
  mv "$f" "$OUT/forward_$i.bin"
  "$Q/qnn-context-binary-utility" --context_binary "$OUT/forward_$i.bin" --json_file "$OUT/forward_${i}_json.json" >/dev/null
done
[ -n "$PARAMS" ] && cp "$PARAMS" "$OUT/params.json"
$PY - "$OUT" <<'PY'
import json, sys, glob
out = sys.argv[1]
for f in sorted(glob.glob(out + "/forward_*_json.json"), key=lambda p: int(p.split("_")[-2])):
    j = json.load(open(f))
    g = j["info"]["graphs"]
    print(f"{f.split('/')[-1]}: graphs={[x['info']['graphName'] for x in g]} "
          f"inputs={[len(x['info']['graphInputs']) for x in g]} outputs={[len(x['info']['graphOutputs']) for x in g]}")
PY
echo "$OUT: $(du -sh "$OUT" | cut -f1)"
