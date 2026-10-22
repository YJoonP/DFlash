# Summary of a run_spec6.py session: the latency breakdown of summarize_v5.py (step-weighted per-step values, call-count
# check), a per-prompt table for the six Spec-Bench prompts (K16on), token identity of 81 / 161 / 401 against the 10-13
# K16on runs (results/s26_v5_20261013), and the decoded outputs (outputs_<TAG>_<CFG>.txt). Speedup uses the S26 AR mean of
# 10-13 (20.07 tok/s over 18 prompts, range 19.66-20.27; AR was not run for 101 / 121 / 141). Reads the .log files
# (extract the folder's logs.tar.xz first); decoding the outputs needs the Qwen/Qwen3.5-4B tokenizer.
# usage: [RESULTS_DIR=<session folder>] python3 summarize_spec6.py [TAG]      (TAG default: main)
import json, os, re, sys
from pathlib import Path
OUT = Path(os.environ.get('RESULTS_DIR', Path(__file__).resolve().parents[3] / 'results/s26_spec6_20261016'))
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



# ---- per-prompt table ---------------------------------------------------------------------------------------------
AR_TPS, AR_DIR = 20.07, OUT.parent / 's26_ar_20261013'
NAME = {81: 'Writing', 101: 'Reasoning', 121: 'Coding', 141: 'STEM', 161: 'Translation', 401: 'Math reasoning'}
man = {r['id']: r for r in json.loads((OUT / 'inputs/manifest.json').read_text())}
order = [int(i) for i in (OUT / 'inputs/ids.txt').read_text().split()]
runs = {}
for p in sorted(OUT.glob(f'{TAG}_*_run.json')):
    j = json.loads(p.read_text())
    if not j.get('exit_code'):
        runs.setdefault(j['config'], {})[j['id']] = j
ar = {}
for i in order:
    p = AR_DIR / f'main_AR16_{i}_run.json'
    if p.exists():
        ar[i] = json.loads(p.read_text())['decode_tps']
for cfg, d in runs.items():
    print(f'\n== {cfg} per prompt (thinking template, greedy, 128 tokens; TPS = (tokens - 1) / decode time)')
    print('| question_id | 분야 | 프롬프트 토큰 | 검증 횟수 | AL | 검증 (ms) | 초안 (ms) | TPS (tok/s) | 배속 (×AR 20.07) | 시작 (KST) | SoC 시작 °C |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    for i in order:
        if i not in d:
            print(f'| {i} | {NAME[i]} | {man[i]["expected_n_past"]} | (없음) |')
            continue
        m = d[i]['metrics']
        tps = (m['n_tokens'] - 1) / m['decode_s']
        print(f"| {i} | {NAME[i]} | {m['n_past']} | {m['n_steps']} | {m['n_tokens'] / m['n_steps']:.3f} | {m['verify_ms']:.2f} | "
              f"{m['draft_ms']:.2f} | {tps:.2f} | {tps / AR_TPS:.2f}× | {d[i]['started_kst'][11:16]} | {d[i]['thermal_before']['soc_mC'] / 1000:.1f} |")
    m = [d[i]['metrics'] for i in order if i in d]
    st = sum(x['n_steps'] for x in m)
    tps = sum(x['n_tokens'] - 1 for x in m) / sum(x['decode_s'] for x in m)
    print(f"| 전체 ({len(m)}) | – | – | {st} | {sum(x['n_tokens'] for x in m) / st:.3f} | {sum(x['verify_ms'] * x['n_steps'] for x in m) / st:.2f} | "
          f"{sum(x['draft_ms'] * x['n_steps'] for x in m) / st:.2f} | {tps:.2f} | {tps / AR_TPS:.2f}× | | |")
    print('AR per prompt (10-13, same prompts only): ' + ', '.join(f'{i} {v:.2f}' for i, v in ar.items()))

# ---- token identity vs the 10-13 K16on runs (same export, runner and switches) -----------------------------------------
old = {}
for p in sorted((OUT.parent / 's26_v5_20261013').glob('main_K16on_*_run.json')):
    j = json.loads(p.read_text())
    if not j.get('exit_code'):
        old[j['id']] = j
if 'K16on' in runs:
    both = [i for i in order if i in runs['K16on'] and i in old]
    same_ids = [i for i in both if runs['K16on'][i]['token_ids'] == old[i]['token_ids']]
    print(f'\nK16on tokens vs 10-13 K16on: {len(same_ids)}/{len(both)} identical ({both})')
    for i in both:
        a, b = runs['K16on'][i]['metrics'], old[i]['metrics']
        print(f"  {i}: TPS {(a['n_tokens'] - 1) / a['decode_s']:.2f} vs {(b['n_tokens'] - 1) / b['decode_s']:.2f}, "
              f"verify {a['verify_ms']:.2f} vs {b['verify_ms']:.2f} ms, draft {a['draft_ms']:.2f} vs {b['draft_ms']:.2f} ms")

# ---- decoded outputs ---------------------------------------------------------------------------------------------
try:
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B', local_files_only=True)
    for cfg, d in runs.items():
        with (OUT / f'outputs_{TAG}_{cfg}.txt').open('w') as f:
            for i in order:
                if i in d:
                    f.write(f"===== {i} ({NAME[i]}) {len(d[i]['token_ids'])} tokens\n[prompt] {man[i]['prompt_head']}...\n"
                            f"{tk.decode(d[i]['token_ids'])}\n\n")
        print(f'decoded outputs: outputs_{TAG}_{cfg}.txt')
except Exception as e:
    print('decode skipped:', e)
