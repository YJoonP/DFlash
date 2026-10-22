# 2026-10-11 KST. Check an exact-recompute export (QNN_DEFER_EXACT=1):
#  1) quantized graph dump: for every commit/verify op pair and ingredient (k, v, beta, decay) the commit-side slice
#     source (graph input) has the same stored/read encoding as the verify-side slice source; same for x at the
#     conv-window cats; kernel group encodings equal on both sides;
#  2) context binaries: every ingredient graph input/output is UFIXED_POINT_16 and input encoding == output encoding
#     for the same role (the codes cross the call boundary unchanged).
# usage: check_exact.py ART_DIR
import json, math, sys
from pathlib import Path
art = Path(sys.argv[1])
Q = 'quantized_decomposed.quantize_per_tensor.default'
DQ = 'quantized_decomposed.dequantize_per_tensor.default'
MASK = 'qwen35.delta_rule_state16_masked.default'
nodes, order = {}, []
for line in open(art / 'qdq_prefill.jsonl'):
    n = json.loads(line); nodes[n['name']] = n; order.append(n['name'])
ref = lambda a: a['node'] if isinstance(a, dict) and 'node' in a else None
enc = lambda n: (tuple(nodes[n]['args'][1:3]))
back = lambda n: ref(nodes[n]['args'][0])
res = {'pairs': 0, 'group_equal': 0, 'src_equal': 0, 'src_diff': 0, 'x_equal': 0, 'x_diff': 0}
for v in order:
    if nodes[v]['target'] != MASK:
        continue
    st = back(back(ref(nodes[v]['args'][5])))
    if 'getitem' not in nodes[st]['target'] or nodes[ref(nodes[st]['args'][0])]['target'] != MASK:
        continue
    c = ref(nodes[st]['args'][0]); res['pairs'] += 1
    for i in (1, 2, 3, 4):
        dv, dc = ref(nodes[v]['args'][i]), ref(nodes[c]['args'][i])
        qv, qc = back(dv), back(dc)
        res['group_equal'] += (enc(qv), enc(dv)) == (enc(qc), enc(dc))
        dav, dac = ref(nodes[back(qv)]['args'][0]), ref(nodes[back(qc)]['args'][0])
        same = (enc(back(dav)), enc(dav)) == (enc(back(dac)), enc(dac))
        res['src_equal' if same else 'src_diff'] += 1
for g in order:
    if nodes[g]['target'] != 'aten.index_select.default':
        continue
    cat_c = back(back(ref(nodes[g]['args'][0])))
    dxin = ref(nodes[cat_c]['args'][0][1])
    cat_v, frontier = None, list(nodes[g]['users'])
    for _ in range(8):
        nxt = []
        for u in frontier:
            if nodes[u]['target'] == 'aten.cat.default' and u != cat_c:
                cat_v = cat_v or u
            elif nodes[u]['target'] in (Q, DQ):
                nxt.extend(nodes[u]['users'])
        frontier = nxt
    dxt = ref(nodes[cat_v]['args'][0][1])
    same = (enc(back(dxin)), enc(dxin)) == (enc(back(dxt)), enc(dxt))
    res['x_equal' if same else 'x_diff'] += 1
print('graph dump:', json.dumps(res))
ctx = art / 'ctx_target'
if (ctx / 'shard_io_map.json').exists():
    m = json.loads((ctx / 'shard_io_map.json').read_text())
    io = {}
    for sh in m['shards']:
        s = sh['shard']
        g = next(g['info'] for g in json.loads((ctx / f'forward_{s}_json.json').read_text())['info']['graphs']
                 if g['info']['graphName'] == 'prefill_forward')
        for side, key in (('inputs', 'graphInputs'), ('outputs', 'graphOutputs')):
            for t in g[key]:
                t = t['info']; role = sh[side][t['name']]['role']
                if role.startswith('ing_'):
                    io[(side, role)] = (s, t['dataType'], t.get('quantizeParams', {}).get('scaleOffset'), t['dimensions'])
    roles = sorted({r for (_, r) in io})
    bad = []
    nbytes = {'inputs': 0, 'outputs': 0}
    for r in roles:
        i, o = io.get(('inputs', r)), io.get(('outputs', r))
        if not i or not o or i[1] != 'QNN_DATATYPE_UFIXED_POINT_16' or o[1] != 'QNN_DATATYPE_UFIXED_POINT_16' or i[2] != o[2] or i[3] != o[3] or i[0] != o[0]:
            bad.append((r, i, o))
        for side, t in (('inputs', i), ('outputs', o)):
            if t:
                nbytes[side] += math.prod(t[3]) * (2 if t[1].endswith('_16') else 4)
    print(f'context IO: {len(roles)} ingredient roles, {len(bad)} not u16 with equal in/out encoding; bytes {nbytes}')
    for b in bad[:6]:
        print('   ', b)
