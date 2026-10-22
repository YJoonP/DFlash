# 2026-10-13 19:1x KST (S26). K16on decode calls that continued from the row-10 state with 0 rows left to commit (accepted
# exactly 10): per-shard NPU time (3012) against 1-row calls, and where these calls sit (call index, prompt id).
# usage: [RESULTS_DIR=<session folder>] python3 zero_row_calls_s26.py   (reads the .log files: extract the folder's logs.tar.xz first)
import os, collections, json, re
from pathlib import Path
OUT = Path(os.environ.get('RESULTS_DIR', Path(__file__).resolve().parents[2] / 'results/s26_v5_20261013'))
BD = re.compile(r'\[Breakdown\] calls=(\d+) .*?prof_types=(\S+)')
CK = re.compile(r'\[HybridCkpt\] calls=(\d+) used=(\d+) rows_full=(\d+) rows_from_ckpt=(\d+)')
def shards(types):
    return [int(dict(x.split('=') for x in sh.split(':', 1)[1].split('/') if x).get('3012', 0)) for sh in types.split(';')]
acc = collections.defaultdict(lambda: [0.0] * 9); cnt = collections.Counter(); where = []
for p in sorted(OUT.glob('main_K16on_*_run.json')):
    j = json.loads(p.read_text()); n = j['metrics']['n_steps']
    text = re.sub(r'\[Q35-DRV\][^\n]*\n', '', (OUT / (p.name[:-9] + '.log')).read_text(errors='ignore'))
    bd = {int(a): shards(b) for a, b in BD.findall(text)}
    ck = {int(c): (int(u), int(f) + int(r)) for c, u, f, r in CK.findall(text)}
    last = max(bd); k0 = last - n
    for c in range(k0 + 1, last + 1):
        if c - 1 not in bd or c not in ck or c - 1 not in ck:
            continue
        rows = ck[c][1] - ck[c - 1][1]; used = ck[c][0] - ck[c - 1][0]
        key = 'zero rows (from ckpt)' if (used and rows == 0) else ('1 row' if rows == 1 and not used else None)
        if key:
            d = [(a - b) / 1e3 for a, b in zip(bd[c], bd[c - 1])]
            acc[key] = [x + y for x, y in zip(acc[key], d)]; cnt[key] += 1
            if used and rows == 0:
                where.append((j['id'], c - k0))
for k in acc:
    print(f"{k:22s} n={cnt[k]:3d} per shard ms: " + ' '.join(f'{x / cnt[k]:6.2f}' for x in acc[k]) + f"  sum {sum(acc[k]) / cnt[k]:6.2f}")
print('zero-row calls (prompt id, decode step):', where)
