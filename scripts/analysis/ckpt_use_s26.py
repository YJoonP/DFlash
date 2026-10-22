# 2026-10-13 19:1x KST (S26). Split the main_* runs of the session into
# prompt calls (first k0 calls, 16 rows each) and decode steps. Per phase: NPU time per call (event 3012 summed over the
# 9 shards, from the cumulative [Breakdown] counters), and for K16on the checkpoint use from the cumulative [HybridCkpt]
# lines (used = calls that continued from the row-10 state; rows = committed rows: rows_full from the committed state,
# rows_from_ckpt after the checkpoint).
# usage: [RESULTS_DIR=<session folder>] python3 ckpt_use_s26.py   (reads the .log files: extract the folder's logs.tar.xz first)
import os, json, re, sys
from pathlib import Path
OUT = Path(os.environ.get('RESULTS_DIR', Path(__file__).resolve().parents[2] / 'results/s26_v5_20261013'))
BD = re.compile(r'\[Breakdown\] calls=(\d+) .*?prof_types=(\S+)')
CK = re.compile(r'\[HybridCkpt\] calls=(\d+) used=(\d+) rows_full=(\d+) rows_from_ckpt=(\d+)')
def t3012(types):
    return sum(int(dict(x.split('=') for x in sh.split(':', 1)[1].split('/') if x).get('3012', 0)) for sh in types.split(';'))
res = {}
for p in sorted(OUT.glob('main_*_run.json')):
    j = json.loads(p.read_text()); cfg = j['config']; m = j['metrics']; n = m['n_steps']
    text = re.sub(r'\[Q35-DRV\][^\n]*\n', '', (OUT / (p.name[:-9] + '.log')).read_text(errors='ignore'))
    bd = [(int(a), t3012(b)) for a, b in BD.findall(text)]
    k0 = bd[-1][0] - n
    r = res.setdefault(cfg, {'pcalls': 0, 'p_us': 0, 'steps': 0, 'd_us': 0, 'ck': [0] * 6})
    r['pcalls'] += k0; r['p_us'] += bd[k0 - 1][1]; r['steps'] += n; r['d_us'] += bd[-1][1] - bd[k0 - 1][1]
    ck = [tuple(map(int, x)) for x in CK.findall(text)]
    if ck:
        a = next(c for c in ck if c[0] == k0); e = ck[-1]
        # prompt phase: used, rows_full, rows_ckpt; decode phase: the same
        r['ck'] = [r['ck'][0] + a[1], r['ck'][1] + a[2], r['ck'][2] + a[3],
                   r['ck'][3] + e[1] - a[1], r['ck'][4] + e[2] - a[2], r['ck'][5] + e[3] - a[3]]
print('NPU time (3012, all 9 shards) per call:  prompt calls (16 rows committed each) | decode steps')
for cfg, r in res.items():
    print(f"  {cfg:7s} prompt {r['pcalls']:4d} calls {r['p_us'] / r['pcalls'] / 1e3:7.3f} ms | decode {r['steps']:4d} steps {r['d_us'] / r['steps'] / 1e3:7.3f} ms")
c = res['K16on']['ck']
print(f"K16on prompt calls: from the row-10 state {c[0]} of {res['K16on']['pcalls']}; rows committed: from the committed state {c[1]}, after the checkpoint {c[2]}")
print(f"K16on decode steps: from the row-10 state {c[3]} of {res['K16on']['steps']} ({100 * c[3] / res['K16on']['steps']:.1f}%); "
      f"rows committed: from the committed state {c[4]}, after the checkpoint {c[5]} "
      f"-> without the checkpoint these steps would commit {c[5] + 10 * c[3]} rows; per decode step {(c[4] + c[5]) / res['K16on']['steps']:.2f} vs "
      f"{(c[4] + c[5] + 10 * c[3]) / res['K16on']['steps']:.2f} rows")
