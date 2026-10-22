# 2026-10-13 11:4x KST. Two-grid meeting points of an export: every tensor whose store encoding (the Q after its producer)
# differs from a read encoding (a DQ user) -> QNN inserts a requantize there. Reads ART/qdq_prefill.jsonl (QNN_QDQ_DUMP).
# 10-11 exact conv export (C16): 46 points = 24 conv windows [1, 8192, 19] (cat -> conv), 16 attention K/V cats,
# 6 state inputs (layers 16/21 groups 0-2). v4 must have no conv-window and no state-input point.
# usage: check_og.py ART_DIR [EXPORT_LOG]   prints the points by kind and PASS / FAIL (v4 rule)
import collections, json, re, sys
from pathlib import Path
art = Path(sys.argv[1])
Q = 'quantized_decomposed.quantize_per_tensor.default'
DQ = 'quantized_decomposed.dequantize_per_tensor.default'
nodes = {}
for line in open(art / 'qdq_prefill.jsonl'):
    n = json.loads(line); nodes[n['name']] = n
ref = lambda a: a['node'] if isinstance(a, dict) and 'node' in a else None
points = []
for name, n in nodes.items():
    if n['target'] != Q:
        continue
    enc = tuple(n['args'][1:3]); prod = ref(n['args'][0]); p = nodes.get(prod, {})
    for u in n['users']:
        d = nodes.get(u)
        if d and d['target'] == DQ and tuple(d['args'][1:3]) != enc:
            shape = p.get('shape'); tgt = p.get('target', prod)
            if tgt == 'aten.cat.default' and shape and shape[1:2] == [8192]:
                kind = 'conv window'
            elif p.get('op') == 'placeholder' and shape and len(shape) == 4 and shape[2:] == [128, 128]:
                kind = 'state input'
            elif 'transpose' in str(tgt) or 'permute' in str(tgt):
                kind = 'attention K/V'
            else:
                kind = f'other ({tgt} {shape})'
            rel = abs(d['args'][1] - enc[0]) / enc[0]
            points.append((kind, p.get('layer'), rel))
by = collections.Counter(k for k, _, _ in points)
print(f'{art.name}: {len(points)} two-grid points')
for k, v in by.most_common():
    rels = sorted(r for kk, _, r in points if kk == k)
    print(f'  {v:3d}  {k:40s} scale diff (relative) {rels[0]:.1e} .. {rels[-1]:.1e}')
if len(sys.argv) > 2:
    m = re.findall(r'QNN_DEFER_EXACT: ingredient inputs put on their output encodings: (\{.*\})', Path(sys.argv[2]).read_text(errors='ignore'))
    print('exact pass:', m[-1] if m else 'no QNN_DEFER_EXACT line')
bad = by.get('conv window', 0) + by.get('state input', 0)
print('PASS' if bad == 0 else f'FAIL: {bad} conv-window / state-input points left')
