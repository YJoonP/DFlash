# 2026-10-12 23:5x KST. Check of an exact-recompute export with the checkpoint split (QNN_DEFER_EXACT=1 +
# QNN_DEFER_CHECKPOINT_ROW=c): check_exact.py and the earlier checkpoint-context check in one, adapted to the split
# verify op.
#  1) quantized graph dump, per DeltaNet head group (commit op, head op = rows [0, c), tail op = rows [c, T)) and
#     ingredient k, v, beta, decay:
#       - the commit reads with the head's encoding (Q and DQ in front of the op), the tail reads with the head's too
#         (the commit replays rows [0, a) or [c, a) from one ingredient tensor with one encoding);
#       - the commit's graph input (stored codes) is on the encoding of the graph output that feeds it;
#       - every requantization on the verify side between that output and the op is replayable: each intermediate
#         encoding equals the output encoding or the op's read encoding (then q(q(x)) == the commit's single q(x)).
#     x at the conv-window cats as in check_exact.py.
#  2) context binaries: ingredient graph IO u16 with input encoding == output encoding (if the export used u16 IO);
#     4 checkpoint outputs per DeltaNet layer (role state_ckpt_group:L:g), UFIXED_POINT_16 [1, 8, 128, 128] on the
#     scale/offset of that layer's state_rec input (the runtime feeds a checkpoint back as the base state).
#  3) export log: CUSTOM_STATE_ENC in == out; mode-2 prefill forced count; the QNN_DEFER_EXACT counts.
# usage: check_exact_ckpt.py ART_DIR [EXPORT_LOG]
import json, math, re, sys
from pathlib import Path
art = Path(sys.argv[1])
log = Path(sys.argv[2]) if len(sys.argv) > 2 else None
Q = 'quantized_decomposed.quantize_per_tensor.default'
DQ = 'quantized_decomposed.dequantize_per_tensor.default'
MASK = 'qwen35.delta_rule_state16_masked.default'
SLICES = ('aten.slice.Tensor', 'aten.slice_copy.Tensor')
fails = []
nodes, order = {}, []
for line in open(art / 'qdq_prefill.jsonl'):
    n = json.loads(line); nodes[n['name']] = n; order.append(n['name'])
ref = lambda a: a['node'] if isinstance(a, dict) and 'node' in a else None
enc = lambda n: tuple(nodes[n]['args'][1:3])
back = lambda n: ref(nodes[n]['args'][0])
tgt = lambda n: nodes[n]['target'] if n in nodes else ('placeholder' if n else None)


def state_src(op):
    """the masked op feeding op's state input (via getitem 1 -> Q -> DQ), or None (graph input)."""
    st = back(back(ref(nodes[op]['args'][5])))
    if st and 'getitem' in tgt(st) and tgt(ref(nodes[st]['args'][0])) == MASK:
        return ref(nodes[st]['args'][0])
    return None


def chain(dn):
    """op input DQ -> list of (Q enc, DQ enc) from the op back through every slice; last = the source's (Q, DQ)
    and the source node."""
    out, d = [], dn
    while True:
        qn = back(d); src = back(qn)
        out.append((enc(qn), enc(d)))
        if tgt(src) in SLICES and tgt(ref(nodes[src]['args'][0])) == DQ:
            d = ref(nodes[src]['args'][0]); continue
        return out, src


ops = [n for n in order if tgt(n) == MASK]
commits = [n for n in ops if state_src(n) is None]
heads = [n for n in ops if state_src(n) in commits]
tails = [n for n in ops if state_src(n) in heads]
res = {'commit_ops': len(commits), 'head_ops': len(heads), 'tail_ops': len(tails), 'read_equal': 0, 'read_diff': 0,
       'tail_equal': 0, 'tail_diff': 0, 'stored_equal': 0, 'stored_diff': 0, 'chain_ok': 0, 'chain_bad': 0,
       'x_equal': 0, 'x_diff': 0}
if not (len(commits) == len(heads) == len(tails) > 0):
    fails.append(f"ops commit/head/tail {len(commits)}/{len(heads)}/{len(tails)}")
tail_of = {state_src(t): t for t in tails}
bad_examples = []
for h in heads:
    c, t = state_src(h), tail_of.get(h)
    for i in (1, 2, 3, 4):
        ch, src_h = chain(ref(nodes[h]['args'][i]))
        cc, src_c = chain(ref(nodes[c]['args'][i]))
        read_h, read_c = ch[0], cc[0]
        res['read_equal' if read_h == read_c else 'read_diff'] += 1
        if t:
            ct, src_t = chain(ref(nodes[t]['args'][i]))
            res['tail_equal' if (ct[0] == read_h and src_t == src_h) else 'tail_diff'] += 1
        # stored: the commit's graph input (Q, DQ) == the verify-side source output (Q, DQ)
        res['stored_equal' if cc[-1] == ch[-1] else 'stored_diff'] += 1
        # replayable verify chain: every intermediate encoding is the source's or the op's read encoding
        ok = all(e[0] in (ch[-1][0], read_h[0]) for e in ch[1:-1]) and len(cc) == 2
        res['chain_ok' if ok else 'chain_bad'] += 1
        if not ok and len(bad_examples) < 4:
            bad_examples.append((h, i, ch, cc))
for g in order:
    if tgt(g) != 'aten.index_select.default':
        continue
    cat_c = back(back(ref(nodes[g]['args'][0])))
    dxin = ref(nodes[cat_c]['args'][0][1])
    cat_v, frontier = None, list(nodes[g]['users'])
    for _ in range(8):
        nxt = []
        for u in frontier:
            if tgt(u) == 'aten.cat.default' and u != cat_c:
                cat_v = cat_v or u
            elif tgt(u) in (Q, DQ):
                nxt.extend(nodes[u]['users'])
        frontier = nxt
    dxt = ref(nodes[cat_v]['args'][0][1])
    same = (enc(back(dxin)), enc(dxin)) == (enc(back(dxt)), enc(dxt))
    res['x_equal' if same else 'x_diff'] += 1
print('graph dump:', json.dumps(res))
for b in bad_examples:
    print('   chain example', b)
for k in ('read_diff', 'tail_diff', 'stored_diff', 'chain_bad', 'x_diff'):
    if res[k]:
        fails.append(f'{k} {res[k]}')

ctx = art / 'ctx_target'
if (ctx / 'shard_io_map.json').exists():
    params = json.load(open(ctx / 'params.json'))
    delta = [i for i, t in enumerate(params['layer_types']) if t == 'linear_attention']
    m = json.loads((ctx / 'shard_io_map.json').read_text())
    io, state_in, ckpt = {}, {}, {}
    so = lambda t: (lambda e: (e['scale'], e['offset']) if e else None)((t.get('quantizeParams') or {}).get('scaleOffset'))
    for sh in m['shards']:
        s = sh['shard']
        g = next(g['info'] for g in json.loads((ctx / f'forward_{s}_json.json').read_text())['info']['graphs']
                 if g['info']['graphName'] == m.get('method', 'prefill_forward'))
        for side, key in (('inputs', 'graphInputs'), ('outputs', 'graphOutputs')):
            for t in g[key]:
                t = t['info']; role = sh[side][t['name']]['role']
                if role.startswith('ing_'):
                    io[(side, role)] = (s, t['dataType'], so(t), t['dimensions'])
                elif side == 'inputs' and role.startswith('state_rec:'):
                    state_in[int(role.split(':')[1])] = so(t)
                elif side == 'outputs' and role.startswith('state_ckpt_group:'):
                    _, L, gi = role.split(':')
                    ckpt[(int(L), int(gi))] = (t['dataType'], t['dimensions'], so(t), s)
    roles = sorted({r for (_, r) in io})
    u16 = all(v[1] == 'QNN_DATATYPE_UFIXED_POINT_16' for v in io.values())
    bad = []
    for r in roles:
        i, o = io.get(('inputs', r)), io.get(('outputs', r))
        if not i or not o or i[1] != o[1] or i[3] != o[3] or i[0] != o[0] or (u16 and i[2] != o[2]):
            bad.append((r, i, o))
    print(f'context IO: {len(roles)} ingredient roles ({"u16" if u16 else "not all u16 (float32 ingredient IO?)"}), '
          f'{len(bad)} with in/out dtype, shape, shard or (u16) encoding differing')
    for b in bad[:6]:
        print('   ', b)
    if bad:
        fails.append(f'{len(bad)} ingredient roles in != out')
    print(f'checkpoint outputs {len(ckpt)} (expected {4 * len(delta)})')
    if len(ckpt) != 4 * len(delta):
        fails.append(f'checkpoint count {len(ckpt)}')
    # the ckpt5 rule (10-10 15:20 KST): |scale'/scale - 1| <= 1e-5 and |offset' - offset| <= 2 is restored exactly by
    # the runner's offset shift; anything else fails
    tol = lambda e, s: e and s and abs(e[0] - s[0]) / s[0] <= 1e-5 and abs(e[1] - s[1]) <= 2
    cbad = [(k, v) for k, v in ckpt.items() if v[0] != 'QNN_DATATYPE_UFIXED_POINT_16' or v[1] != [1, 8, 128, 128]
            or not tol(v[2], state_in.get(k[0]))]
    cadj = [(k, v) for k, v in ckpt.items() if (k, v) not in cbad and v[2] != state_in.get(k[0])]
    print(f'checkpoint dtype/shape/encoding beyond tolerance: {len(cbad)}; within tolerance but not equal: {len(cadj)}')
    for k, v in (cbad + cadj)[:6]:
        print('   ', k, v, 'state_in', state_in.get(k[0]))
    if cbad:
        fails.append(f'{len(cbad)} checkpoint outputs off the state input encoding')
else:
    print('context IO: no ctx_target/shard_io_map.json (graph dump only)')
if log and log.exists():
    text = log.read_text(errors='ignore')
    e = re.findall(r'CUSTOM_STATE_ENC \S+ in\(scale=([^,]+), offset=(-?\d+)\) out\(scale=([^,]+), offset=(-?\d+)\)', text)
    neq = [x for x in e if (x[0], x[1]) != (x[2], x[3])]
    forced = re.findall(r'shared state encoding mode 2 \(prefill\): (\d+) delta-op state outputs forced', text)
    exact = re.findall(r'QNN_DEFER_EXACT: ingredient inputs put on their output encodings: (\{.*\})', text)
    print(f'export log: CUSTOM_STATE_ENC {len(e)} lines, in != out {len(neq)}; mode-2 prefill forced {forced}; '
          f'exact counts {exact[-1] if exact else None}')
    if neq:
        fails.append(f'{len(neq)} CUSTOM_STATE_ENC in != out')
print('FAIL: ' + '; '.join(fails) if fails else 'PASS')
