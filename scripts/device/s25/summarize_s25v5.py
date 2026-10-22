# Summary of a run_s25v5.py session: per config the per-step latency breakdown (runner counters + HTP profile), the
# final table (speedup against the S25 AR runs of 10-05), checkpoint use per call ([HybridCkpt]) and token identity.
# Reads <TAG>_*_run.json and the matching .log files (extract the folder's logs.tar.xz first). Per-step value =
# (cumulative counter at the end - at the end of the prompt calls) / decode steps, pooled over prompts (step-weighted). The
# prompt calls are found by count (k0 = total calls - steps, checked = ceil(prompt tokens / width)), not by the
# '[Q35-DRV] prompt decoded' line: that line goes to stderr (flushed) while [Breakdown] goes to buffered stdout.
# HTP BASIC events per execute: 400 QNN execute, 3001 RPC (host side), 3002 QNN accelerator (HTP side), 3004 accelerator,
# 3012 accelerator excluding wait, 8001 HVX threads.
# usage: [RESULTS_DIR=<session folder>] python3 summarize_s25v5.py [TAG]      (TAG default: main)
import json, os, re, sys
from pathlib import Path
OUT = Path(os.environ.get('RESULTS_DIR', Path(__file__).resolve().parents[3] / 'results/s25_vs_old_adopted_20261014'))
TAG = sys.argv[1] if len(sys.argv) > 1 else 'main'
BD = re.compile(r'\[Breakdown\] calls=(\d+) kv_us=(\d+) prep_us=([\d,]+) exec_us=([\d,]+) post_us=([\d,]+) '
                r'prof_exec_us=([\d,]+) host_rpc_us=[\d,]+ htp_rpc_us=[\d,]+ accel_us=[\d,]+ misc_us=[\d,]+ '
                r'prof_stale=([\d,]+) prof_types=(\S+)')
MAP = re.compile(r'\[HybridSnapshotMap\] register_us=(\d+) deregister_us=(\d+) graph_us=(\d+) registered_bytes=(\d+) staging_copy_us=(\d+)')
SEL = re.compile(r'\[HybridSnapshot\] selections=(\d+) select_us=(\d+)')
TYPES = (3001, 3002, 3004, 3012, 8001)


def parse(m):
    lst = lambda s: [int(x) for x in s.split(',')]
    v = {'calls': int(m.group(1)), 'kv': int(m.group(2)), 'prep': lst(m.group(3)), 'exec': lst(m.group(4)),
         'post': lst(m.group(5)), 'q400': lst(m.group(6)), 'stale': lst(m.group(7))}
    per = {t: [] for t in TYPES}
    for shard in m.group(8).split(';'):
        kv = dict(x.split('=') for x in shard.split(':', 1)[1].split('/') if x)
        for t in TYPES:
            per[t].append(int(kv.get(str(t), 0)))
    for t in TYPES:
        v[f't{t}'] = per[t]
    return v


def at_index_and_end(rx, text, conv, k):
    b = [conv(m) for m in rx.finditer(text)]
    return (b[k - 1] if k >= 1 and len(b) >= k else None), (b[-1] if b else None)


def diff(e, s):
    out = {}
    for k, x in e.items():
        y = s.get(k) if s else None
        out[k] = [a - (y[i] if y else 0) for i, a in enumerate(x)] if isinstance(x, list) else x - (y or 0)
    return out


acc, checks = {}, []
for p in sorted(OUT.glob(f'{TAG}_*_run.json')):
    j = json.loads(p.read_text()); cfg = j['config']; m = j['metrics']; n = m['n_steps']
    text = (OUT / (p.name[:-9] + '.log')).read_text(errors='ignore')
    # stderr STAGE lines can land inside a buffered stdout line (main_R8_81: '[Q35-DRV] prompt decoded' split a
    # [Breakdown] line); dropping them rejoins the halves. Any other damage fails the call-count check below.
    text = re.sub(r'\[Q35-DRV\][^\n]*\n', '', text)
    allbd = [parse(mm) for mm in BD.finditer(text)]
    if not allbd:
        raise SystemExit(f'no v3 [Breakdown] line in {p.name}')
    k0 = allbd[-1]['calls'] - n
    width = 16 if '16' in cfg else 8
    if k0 != -(-m['n_past'] // width) or allbd[k0 - 1]['calls'] != k0:
        raise SystemExit(f'{p.name}: prompt calls {k0} != ceil({m["n_past"]}/{width})')
    s, e = allbd[k0 - 1], allbd[-1]
    d = diff(e, s)
    checks.append((p.name, n, d['calls'], sum(d['stale'])))
    r = acc.setdefault(cfg, {'steps': 0, 'draft': 0.0, 'verify': 0.0, 'tok': 0, 'prompts': 0, 'ids': []})
    r['steps'] += n; r['draft'] += m['draft_ms'] * n; r['verify'] += m['verify_ms'] * n
    r['tok'] += m['n_tokens'] - 1; r['prompts'] += 1; r['ids'].append(j['id'])
    for k, x in d.items():
        if isinstance(x, list):
            r[k] = [a + b for a, b in zip(r.get(k, [0] * len(x)), x)]
        else:
            r[k] = r.get(k, 0) + x
    # one [HybridSnapshotMap] line per shard execute (9 per call); one selection per call after the first, made at
    # the start of the next call (so k0 - 1 selections belong to the prompt calls).
    for rx, names, k in ((MAP, ('register', 'deregister', None, None, 'staging'), 9 * k0), (SEL, (None, 'select'), k0 - 1)):
        s2, e2 = at_index_and_end(rx, text, lambda mm: list(map(int, mm.groups())), k)
        if e2:
            for i, nm in enumerate(names):
                if nm:
                    r[nm] = r.get(nm, 0) + e2[i] - (s2[i] if s2 else 0)

res = {}
for cfg, r in acc.items():
    st = r['steps']
    ms = lambda v: (sum(v) if isinstance(v, list) else v) / 1e3 / st
    sh = lambda v: [x / 1e3 / st for x in v]
    o = {'prompts': r['prompts'], 'ids': r['ids'], 'steps': st, 'tok_per_step': r['tok'] / st,
         'step_ms': (r['draft'] + r['verify']) / st, 'draft_ms': r['draft'] / st, 'verify_ms': r['verify'] / st}
    q400, h, t, a, x = ms(r['q400']), ms(r['t3001']), ms(r['t3002']), ms(r['t3004']), ms(r['t3012'])
    o['verify_split_ms'] = {
        'shard input prep (runner)': ms(r['prep']),
        'graph execute call (runner-timed)': ms(r['exec']),
        'shard output handling (runner)': ms(r['post']),
        'KV write-back (runner)': ms(r['kv']),
    }
    o['verify_split_ms']['rest of verify (driver, acceptance, state select, ...)'] = o['verify_ms'] - sum(o['verify_split_ms'].values())
    o['graph_split_ms'] = {
        'NPU compute excl. wait (3012)': x,
        'NPU wait inside execute (3004-3012)': a - x,
        'NPU-side call overhead (3002-3004)': t - a,
        'CPU<->NPU RPC overhead (3001-3002)': h - t,
        'QNN CPU-side work incl. RAW copies (400-3001)': q400 - h,
        'loader around QNN execute (exec-400)': ms(r['exec']) - q400,
    }
    o['per_shard_ms'] = {'exec': sh(r['exec']), 'npu_excl_wait': sh(r['t3012']), 'qnn_cpu_work': [a_ - b_ for a_, b_ in zip(sh(r['q400']), sh(r['t3001']))],
                         'prep': sh(r['prep']), 'post': sh(r['post'])}
    o['hvx_threads_mean'] = sum(r['t8001']) / max(1, sum(1 for _ in r['t8001'])) / (r['calls'] or 1)
    o['snapshot_ms'] = {k: r[k] / 1e3 / st for k in ('staging', 'select', 'register', 'deregister') if k in r}
    res[cfg] = o
(OUT / f'summary_v2_{TAG}.json').write_text(json.dumps({'checks': checks, 'configs': res}, indent=2))
bad = [c for c in checks if abs(c[1] - c[2]) > 1 or c[3]]
print('call-count check (run, steps, decode verify calls, stale):', 'all ok' if not bad else bad)
for cfg in sorted(res):
    if cfg not in res:
        continue
    o = res[cfg]
    print(f"== {cfg}: prompts {o['ids']}, {o['steps']} steps, {o['tok_per_step']:.2f} tok/step | step {o['step_ms']:.2f} ms = draft {o['draft_ms']:.2f} + verify {o['verify_ms']:.2f}")
    for k, v in o['verify_split_ms'].items():
        print(f"   verify | {k:55s} {v:7.2f}")
    for k, v in o['graph_split_ms'].items():
        print(f"   graph  | {k:55s} {v:7.2f}")
    for k, v in o['snapshot_ms'].items():
        print(f"   snap   | {k:55s} {v:7.2f}")
    for k in ('exec', 'npu_excl_wait', 'qnn_cpu_work', 'post'):
        print(f"   shard {k:14s}: " + ' '.join(f'{v:5.2f}' for v in o['per_shard_ms'][k]))


# ---- the user's table (same session) ------------------------------------------------------------------------------
AR = {8: (56.14, 17.81), 16: (55.90, 17.89)}   # S25 AR 10-05 runs (per-token ms, tok/s), as in the S25 summaries since 10-05
NAMES = {'O16': '옛 채택 (10-02, 정확성 수정 없음)', 'C16': 'exact conv (10-11)', 'R16v5': '재계산 v5 (S25)', 'K16off': 'ckpt10 v5, 체크포인트 끔', 'K16on': 'ckpt10 v5, 10행 상태에서 이어서',
         'S16': '저장 방식 (전부 고정, S26 main 세션)'}
runs = {}
for p in sorted(OUT.glob(f'{TAG}_*_run.json')):
    j = json.loads(p.read_text())
    if not j.get('exit_code'):
        runs.setdefault(j['config'], {})[j['id']] = j
for p in sorted((OUT.parent / 'runs').glob('main_S16_*_run.json')):   # snapshot b = 16 from the S26 main session
    j = json.loads(p.read_text())
    if not j.get('exit_code'):
        runs.setdefault('S16', {})[j['id']] = j
print('\n| Tree Size | Method | Verification Latency (ms) | Drafting Latency (ms) | AL | TPS (token/sec) | Speedup (×AR) | prompts |')
print('|---|---|---|---|---|---|---|---|')
w = 16
print(f'| b = {w} | AR (S25 10-05) | {AR[w][0]:.2f} (토큰당) | – | 1.00 | {AR[w][1]:.2f} | 1.00× | 18 |')
for cfg in ('O16', 'C16', 'R16v5', 'K16off', 'K16on'):
    if cfg not in runs:
        continue
    m = [r['metrics'] for r in runs[cfg].values()]
    st = sum(x['n_steps'] for x in m)
    ver = sum(x['verify_ms'] * x['n_steps'] for x in m) / st
    dr = sum(x['draft_ms'] * x['n_steps'] for x in m) / st
    al = sum(x['n_tokens'] for x in m) / st
    tps = sum(x['n_tokens'] - 1 for x in m) / sum(x['decode_s'] for x in m)
    print(f'| b = {w} | {NAMES[cfg]} | {ver:.2f} | {dr:.2f} | {al:.3f} | {tps:.2f} | {tps / AR[w][1]:.2f}× | {len(m)} |')

# ---- checkpoint use: last [HybridCkpt] calls/used/rows line per run (all calls incl. the prompt) ----------------------
CK = re.compile(r'\[HybridCkpt\] calls=(\d+) used=(\d+) rows_full=(\d+) rows_from_ckpt=(\d+)')
print('\ncheckpoint use (all target calls incl. prompt; rows = committed rows):')
for cfg in ('K16on', 'K16off'):
    tot = [0, 0, 0, 0]
    for p in sorted(OUT.glob(f'{TAG}_{cfg}_*_run.json')):
        hits = CK.findall((OUT / (p.name[:-9] + '.log')).read_text(errors='ignore'))
        if hits:
            tot = [a + int(b) for a, b in zip(tot, hits[-1])]
    if tot[0]:
        print(f'  {cfg}: calls {tot[0]}, from the row-10 state {tot[1]} ({100 * tot[1] / tot[0]:.1f}%), rows committed from the '
              f'committed state {tot[2]}, rows committed after the checkpoint {tot[3]}')

# ---- token identity --------------------------------------------------------------------------------------------
def same(a, b):
    ids = sorted(set(a) & set(b))
    diff = [i for i in ids if a[i] != b[i]]
    return f'{len(ids) - len(diff)}/{len(ids)} identical' + (f' (differ: {diff})' if diff else '')
tok = {c: {i: r['token_ids'] for i, r in d.items()} for c, d in runs.items()}
main_r16 = {}
for p in sorted((OUT.parent / 'runs').glob('main_R16_*_run.json')):
    j = json.loads(p.read_text())
    if not j.get('exit_code'):
        main_r16[j['id']] = j['token_ids']
print('\ntoken identity:')
for a, b, note in (('K16on', 'K16off', 'checkpoint on vs off, same export (must be identical)'),
                   ('K16off', 'R16v5', 'ckpt10 v5 vs plain recompute v5 (separate exports, must be identical)'),
                   ('R16v5', 'C16', 'v5 vs exact conv (different encodings: not expected identical)'),
                   ('K16on', 'S16', 'ckpt10 vs snapshot')):
    if a in tok and b in tok:
        print(f'  {a} vs {b} ({note}): {same(tok[a], tok[b])}')
if 'R16' in tok and main_r16:
    print(f'  R16 this session vs R16 S26 main session (hostpath runner): {same(tok["R16"], main_r16)}')
