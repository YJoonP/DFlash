# 2026-10-14 02:5x KST (S25). Per call of the session's main_K16off / main_K16on runs: NPU time (event 3012, 9 shards, from
# consecutive cumulative [Breakdown] lines) against the rows the call committed (from consecutive cumulative [HybridCkpt]
# lines: rows_full + rows_from_ckpt). Decode steps only (calls after the k0 prompt calls). Question: does committing fewer
# rows (checkpoint) shorten the NPU time of a decode step, and by how much per row?
# usage: [RESULTS_DIR=<session folder>] python3 commit_rows_vs_npu_s25.py   (reads the .log files: extract the folder's logs.tar.xz first)
import os, collections, json, re
from pathlib import Path
OUT = Path(os.environ.get('RESULTS_DIR', Path(__file__).resolve().parents[2] / 'results/s25_vs_old_adopted_20261014'))
BD = re.compile(r'\[Breakdown\] calls=(\d+) .*?prof_types=(\S+)')
CK = re.compile(r'\[HybridCkpt\] calls=(\d+) used=(\d+) rows_full=(\d+) rows_from_ckpt=(\d+)')
def t3012(types):
    return sum(int(dict(x.split('=') for x in sh.split(':', 1)[1].split('/') if x).get('3012', 0)) for sh in types.split(';'))
for cfg in ('K16off', 'K16on'):
    by = collections.defaultdict(list); byp = collections.defaultdict(list)
    for p in sorted(OUT.glob(f'main_{cfg}_*_run.json')):
        j = json.loads(p.read_text()); n = j['metrics']['n_steps']
        text = re.sub(r'\[Q35-DRV\][^\n]*\n', '', (OUT / (p.name[:-9] + '.log')).read_text(errors='ignore'))
        bd = {int(a): t3012(b) for a, b in BD.findall(text)}
        ck = {int(c): (int(u), int(f) + int(r)) for c, u, f, r in CK.findall(text)}
        last = max(bd); k0 = last - n
        for c in range(2, last + 1):
            if c in bd and c - 1 in bd and c in ck and c - 1 in ck:
                us = bd[c] - bd[c - 1]; rows = ck[c][1] - ck[c - 1][1]; used = ck[c][0] - ck[c - 1][0]
                (by if c > k0 else byp)[(rows, used)].append(us / 1e3)
    print(f'== {cfg}: NPU ms per call by committed rows (used = continued from the row-10 state)')
    for name, d in (('decode', by), ('prompt', byp)):
        cells = [f"{r:2d}{'c' if u else ' '}: {sum(v) / len(v):6.2f} (n={len(v)})" for (r, u), v in sorted(d.items())]
        print(f'  {name}:'); [print('    ' + x) for x in cells]
