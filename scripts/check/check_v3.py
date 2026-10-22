# 2026-10-13 10:5x KST. Strict checks for the v3 exports (QNN_DEFER_EXACT_ONE_ENC + QNN_STATE_IO_FROM_Q +
# QNN_DEFER_SPLIT_Q_ON_GROUP), on top of check_exact_ckpt.py (split exports) or check_exact.py (plain recompute):
#   S1 every state input (placeholder -> Q -> DQ users) is read on the Q's encoding (10-13: layers 16/21 groups 0-2 were not)
#   S2 every ingredient chain (k, v, beta, decay) has a single encoding from the producer's output to every delta op
#   S3 split exports: the query row slices of head / tail ops are on the query group encoding
#   S4 split exports: every checkpoint output is exactly on its layer's state_rec input encoding (no tolerance)
# usage: check_v3.py ART_DIR      prints PASS / FAIL lines
import json, sys
from pathlib import Path
art = Path(sys.argv[1])
Q = 'quantized_decomposed.quantize_per_tensor.default'
DQ = 'quantized_decomposed.dequantize_per_tensor.default'
MASK = 'qwen35.delta_rule_state16_masked.default'
SL = ('aten.slice.Tensor', 'aten.slice_copy.Tensor')
nodes, order = {}, []
for line in open(art / 'qdq_prefill.jsonl'):
    n = json.loads(line); nodes[n['name']] = n; order.append(n['name'])
ref = lambda a: a['node'] if isinstance(a, dict) and 'node' in a else None
tgt = lambda n: nodes[n]['target'] if n in nodes else None
enc = lambda n: tuple(nodes[n]['args'][1:3])
back = lambda n: ref(nodes[n]['args'][0])
fails = []


def state_src(op):
    st = back(back(ref(nodes[op]['args'][5])))
    if st and 'getitem' in (tgt(st) or '') and tgt(ref(nodes[st]['args'][0])) == MASK:
        return ref(nodes[st]['args'][0])


def chain(dn):
    out, d = [], dn
    while True:
        qn = back(d); src = back(qn); out.append((enc(qn), enc(d)))
        if tgt(src) in SL and tgt(ref(nodes[src]['args'][0])) == DQ:
            d = ref(nodes[src]['args'][0]); continue
        return out, src


ops = [n for n in order if tgt(n) == MASK]
commits = [n for n in ops if state_src(n) is None]
verifies = [n for n in ops if state_src(n) in commits]          # unsplit verify op or split head
tails = [n for n in ops if state_src(n) in verifies]
split = bool(tails)
# S1: state inputs read on the Q's encoding (follow the commit ops' state_in chain to the placeholder)
s1_bad, s1_n = [], 0
for c in commits:
    d = ref(nodes[c]['args'][5]); qn = back(d); src = back(qn)
    if tgt(src) in SL:
        dio = ref(nodes[src]['args'][0]); qio = back(dio)
        s1_n += 1
        if enc(dio) != enc(qio):
            s1_bad.append((c, enc(qio), enc(dio)))
print(f'S1 state input dequantizes on the input Q encoding: {s1_n - len(s1_bad)}/{s1_n}')
for b in s1_bad[:4]:
    print('   ', b)
if s1_bad:
    fails.append(f'S1 {len(s1_bad)}')
# S2: ingredient chains on one encoding (verify/head/tail and commit sides)
s2_bad, s2_n = 0, 0
for op in ops:
    for i in (1, 2, 3, 4):
        ch, _ = chain(ref(nodes[op]['args'][i]))
        s2_n += 1
        if len({e for pair in ch for e in pair}) != 1:
            s2_bad += 1
print(f'S2 ingredient chains on a single encoding: {s2_n - s2_bad}/{s2_n}')
if s2_bad:
    fails.append(f'S2 {s2_bad}')
# S3: query row slices on the group encoding (split only)
if split:
    s3_bad, s3_n = 0, 0
    for op in verifies + tails:
        ch, _ = chain(ref(nodes[op]['args'][0]))
        if len(ch) >= 3:
            s3_n += 1
            if ch[0] != ch[1]:
                s3_bad += 1
    print(f'S3 query row slices on the group encoding: {s3_n - s3_bad}/{s3_n}')
    if s3_bad or not s3_n:
        fails.append(f'S3 {s3_bad}/{s3_n}')
# S4: checkpoints exactly on the state input encoding (context binaries)
ctx = art / 'ctx_target'
if split and (ctx / 'shard_io_map.json').exists():
    m = json.loads((ctx / 'shard_io_map.json').read_text())
    so = lambda t: (lambda e: (e['scale'], e['offset']) if e else None)((t.get('quantizeParams') or {}).get('scaleOffset'))
    state_in, ck = {}, {}
    for sh in m['shards']:
        g = next(g['info'] for g in json.loads((ctx / f"forward_{sh['shard']}_json.json").read_text())['info']['graphs']
                 if g['info']['graphName'] == m.get('method', 'prefill_forward'))
        for t in g['graphInputs']:
            t = t['info']; r = sh['inputs'][t['name']]['role']
            if r.startswith('state_rec:'):
                state_in[int(r.split(':')[1])] = so(t)
        for t in g['graphOutputs']:
            t = t['info']; r = sh['outputs'][t['name']]['role']
            if r.startswith('state_ckpt_group:'):
                ck[(int(r.split(':')[1]), int(r.split(':')[2]))] = so(t)
    s4_bad = [(k, v, state_in.get(k[0])) for k, v in ck.items() if v != state_in.get(k[0])]
    print(f'S4 checkpoint outputs exactly on the state input encoding: {len(ck) - len(s4_bad)}/{len(ck)}')
    for b in s4_bad[:4]:
        print('   ', b)
    if s4_bad or not ck:
        fails.append(f'S4 {len(s4_bad)}/{len(ck)}')
print('FAIL: ' + '; '.join(fails) if fails else 'PASS')
