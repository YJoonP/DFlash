#!/usr/bin/env python
"""Parser for llama-speculative-dflash-qnn-npu-q35 run summaries (2026-10-02 KST), used by run_dflash_specbench.sh.

The device run writes the full runner log on the phone; only a grep'd summary (TOKDBG ids + [Q35-*] / [REPLAY] lines,
a few KB) comes back over USB. Commands:
  row     <summary> <mode sd|ar|replay> <expected_n_past>   -> JSON of parsed metrics (one line);
                                                               exit 0 ok, 3 ok but n_past != expected, 1 failed
  record  <summary> <expected_n_past> <tsv> <id> <category> <round> <soc_mC> <bat_dC> <kst>
                                                             -> appends one SD row to results.tsv, prints OK|FAIL line
                                                                (same exit codes as row)
  ids     <summary> <out_ids_txt>                            -> writes TOKDBG ids in index order (one per line), prints count
  cmp     <a_ids_txt> <b_ids_txt>                            -> JSON {status PASS|FAIL, n_a, n_b, prefix, first_mismatch, ...};
                                                                exit 0 PASS, 1 FAIL
  summary <results.tsv> <manifest.json> [gate.json]          -> per-prompt / per-category table
"""
import csv
import json
import re
import sys
from pathlib import Path

RE = {
    "n_past": re.compile(r"prompt decoded, n_past=(\d+) id_last=(-?\d+)"),
    "width": re.compile(r"verify width=(\d+)"),
    "links": re.compile(r"inter-shard links: (\d+)"),
    "draft_pre": re.compile(r"draft prefill: (\d+) rows"),
    "sd_tps": re.compile(r"\[Q35-NPU\] decoded (\d+) tokens in ([\d.]+) s \(([\d.]+) t/s\)"),
    "sd_step": re.compile(r"per step: draft ([\d.]+) ms, verify ([\d.]+) ms, total ([\d.]+) ms"),
    "sd_al": re.compile(r"n_steps=(\d+) n_drafted=(\d+) n_accept=(\d+) mean_accept_len=([\d.]+)(?: committed_len=([\d.]+))?"),
    "ar_tps": re.compile(r"\[Q35-AR\] (\d+) tokens in ([\d.]+) s \(([\d.]+) t/s\)"),
    "replay": re.compile(r"\[REPLAY\] group=(\d+) fed=(\d+) next-token matches (\d+)/(\d+) first_mismatch_at=(-?\d+) \(([\d.]+) s\)"),
    "replay_warn": re.compile(r"\[REPLAY\] warning: (.*)"),
    "exit": re.compile(r"^exit=(\d+)", re.M),
    "tok": re.compile(r"\[TOKDBG\] (\d+) (-?\d+)"),
}


COLS = ("id", "category", "round", "exit", "ok", "n_past", "expected_n_past", "template_ok", "n_tokens", "decode_s", "tps",
        "n_steps", "n_drafted", "n_accept", "mean_accept_len", "committed_len", "draft_ms", "verify_ms", "step_ms",
        "verify_width", "soc_mC", "bat_dC", "kst")


def tokdbg(text):
    d = {}
    for m in RE["tok"].finditer(text):
        d[int(m.group(1))] = int(m.group(2))
    if not d:
        return [], True
    n = max(d) + 1
    return [d.get(i, -999) for i in range(n)], all(i in d for i in range(n))


def row(text, mode, expected):
    r = {"mode": mode}
    if m := RE["exit"].search(text): r["exit"] = int(m.group(1))
    if m := RE["n_past"].search(text): r["n_past"], r["id_last"] = int(m.group(1)), int(m.group(2))
    if m := RE["width"].search(text): r["verify_width"] = int(m.group(1))
    if m := RE["links"].search(text): r["links"] = int(m.group(1))
    if m := RE["draft_pre"].search(text): r["draft_prefill_rows"] = int(m.group(1))
    if m := RE["sd_tps"].search(text): r["n_tokens"], r["decode_s"], r["tps"] = int(m.group(1)), float(m.group(2)), float(m.group(3))
    if m := RE["sd_step"].search(text): r["draft_ms"], r["verify_ms"], r["step_ms"] = map(float, m.groups())
    if m := RE["sd_al"].search(text):
        r["n_steps"], r["n_drafted"], r["n_accept"] = int(m.group(1)), int(m.group(2)), int(m.group(3))
        r["mean_accept_len"] = float(m.group(4)); r["committed_len"] = float(m.group(5)) if m.group(5) else None
    if m := RE["ar_tps"].search(text): r["n_tokens"], r["decode_s"], r["tps"] = int(m.group(1)), float(m.group(2)), float(m.group(3))
    if m := RE["replay"].search(text):
        r.update(group=int(m.group(1)), fed=int(m.group(2)), matches=int(m.group(3)), of=int(m.group(4)),
                 first_mismatch_at=int(m.group(5)), replay_s=float(m.group(6)))
    if m := RE["replay_warn"].search(text): r["replay_warning"] = m.group(1)
    ids, complete = tokdbg(text)
    r["tokdbg_n"], r["tokdbg_complete"] = len(ids), complete
    r["expected_n_past"] = expected
    r["template_ok"] = r.get("n_past") == expected if expected >= 0 else None
    need = {"sd": ("n_tokens", "mean_accept_len", "tps"), "ar": ("n_tokens", "tps"), "replay": ("matches",)}[mode]
    r["ok"] = r.get("exit") == 0 and all(k in r for k in need) and complete and len(ids) > 0
    return r


def main():
    cmd = sys.argv[1]
    if cmd == "row":
        text = Path(sys.argv[2]).read_text(errors="ignore")
        r = row(text, sys.argv[3], int(sys.argv[4]))
        print(json.dumps(r))
        sys.exit(0 if r["ok"] and r["template_ok"] is not False else (3 if r["ok"] else 1))
    elif cmd == "record":   # record <summary> <expected_n_past> <tsv> <id> <category> <round> <soc_mC> <bat_dC> <kst>
        text = Path(sys.argv[2]).read_text(errors="ignore")
        r = row(text, "sd", int(sys.argv[3]))
        tsv = Path(sys.argv[4])
        vals = {k: r.get(k) for k in COLS if k in r}
        vals.update(id=sys.argv[5], category=sys.argv[6], round=sys.argv[7], ok=int(r["ok"]), soc_mC=sys.argv[8],
                    bat_dC=sys.argv[9], kst=sys.argv[10])
        new = not tsv.exists() or tsv.stat().st_size == 0
        with open(tsv, "a") as f:
            if new:
                f.write("\t".join(COLS) + "\n")
            f.write("\t".join("" if vals.get(k) is None else str(vals[k]) for k in COLS) + "\n")
        print(("OK " if r["ok"] else "FAIL ") + " ".join(f"{k}={r.get(k)}" for k in
              ("exit", "n_past", "expected_n_past", "n_tokens", "n_steps", "mean_accept_len", "committed_len", "tps",
               "draft_ms", "verify_ms", "verify_width")))
        sys.exit(0 if r["ok"] and r["template_ok"] is not False else (3 if r["ok"] else 1))
    elif cmd == "ids":
        ids, complete = tokdbg(Path(sys.argv[2]).read_text(errors="ignore"))
        Path(sys.argv[3]).write_text("".join(f"{t}\n" for t in ids))
        print(len(ids) if complete else f"{len(ids)} INCOMPLETE")
    elif cmd == "cmp":
        a = [int(x) for x in Path(sys.argv[2]).read_text().split()]
        b = [int(x) for x in Path(sys.argv[3]).read_text().split()]
        n = min(len(a), len(b)); k = 0
        while k < n and a[k] == b[k]:
            k += 1
        res = {"n_a": len(a), "n_b": len(b), "prefix": k,
               "status": "PASS" if k == n and n > 0 and len(a) == len(b) else "FAIL",
               "first_mismatch": None if k == n else k, "a_at": a[k:k + 6] if k < n else [], "b_at": b[k:k + 6] if k < n else []}
        if k == n and len(a) != len(b):
            res["note"] = "same prefix, different length (EOS/limit)"
        print(json.dumps(res))
        sys.exit(0 if res["status"] == "PASS" else 1)
    elif cmd == "summary":
        rows = list(csv.DictReader(open(sys.argv[2]), delimiter="\t"))
        man = {m["id"]: m for m in json.load(open(sys.argv[3]))}
        gate = json.load(open(sys.argv[4])) if len(sys.argv) > 4 and Path(sys.argv[4]).exists() else None
        ok = [r for r in rows if r["ok"] == "1"]
        print(f"성공 {len(ok)}/{len(rows)} 실행 (SD, DFLASH_CHAT=1, greedy; 템플릿 모드는 run_config.json 의 think)")
        print(f"{'id':>4} {'카테고리':<15} {'n_past':>6} {'토큰':>4} {'steps':>5} | {'AL':>6} {'commit':>6} | {'t/s':>6} "
              f"{'draft':>6} {'verify':>7} {'step ms':>7} | 템플릿")
        cats = {}
        for r in rows:
            i = int(r["id"])
            if r["ok"] != "1":
                print(f"{i:>4} {man[i]['category']:<15} 실패 (exit={r['exit']}, round {r['round']})"); continue
            print(f"{i:>4} {man[i]['category']:<15} {r['n_past']:>6} {r['n_tokens']:>4} {r['n_steps']:>5} | {float(r['mean_accept_len']):6.3f} "
                  f"{float(r['committed_len'] or 0):6.3f} | {float(r['tps']):6.2f} {float(r['draft_ms']):6.1f} {float(r['verify_ms']):7.1f} "
                  f"{float(r['step_ms']):7.1f} | {'OK' if r['template_ok'] == 'True' else 'n_past 불일치(기대 ' + r['expected_n_past'] + ')'}")
            c = cats.setdefault(man[i]["category"], {"al": [], "tps": [], "tok": 0, "steps": 0, "s": 0.0})
            c["al"].append(float(r["mean_accept_len"])); c["tps"].append(float(r["tps"]))
            c["tok"] += int(r["n_tokens"]); c["steps"] += int(r["n_steps"]); c["s"] += float(r["decode_s"])
        if cats:
            print("카테고리별 (문항 평균 AL·t/s, 괄호 = 토큰 합/스텝 합, 토큰 합/시간 합):")
            for k, c in cats.items():
                print(f"  {k:<15} AL {sum(c['al'])/len(c['al']):6.3f} ({c['tok']/c['steps']:6.3f})  "
                      f"t/s {sum(c['tps'])/len(c['tps']):6.2f} ({c['tok']/c['s']:6.2f})  n={len(c['al'])}")
            al = [float(r["mean_accept_len"]) for r in ok]; tps = [float(r["tps"]) for r in ok]
            tok = sum(int(r["n_tokens"]) for r in ok); steps = sum(int(r["n_steps"]) for r in ok); s = sum(float(r["decode_s"]) for r in ok)
            cm = [sum(c["al"]) / len(c["al"]) for c in cats.values()]
            print(f"전체: AL 문항 평균 {sum(al)/len(al):.3f}, 카테고리 평균 {sum(cm)/len(cm):.3f}, 토큰/스텝 {tok/steps:.3f} | "
                  f"t/s 문항 평균 {sum(tps)/len(tps):.2f}, 토큰/시간 {tok/s:.2f}")
        if gate:
            print(f"AR==SD 게이트(id {gate.get('id')}): {gate.get('status')} {json.dumps(gate.get('cmp', {}), ensure_ascii=False)}"
                  + (f" | AR(verify 그래프) {gate['ar'].get('tps')} t/s" if gate.get("ar") else ""))
        print("주의: t/s 는 decode 구간만(프롬프트 prefill 제외). AR 기준 속도는 E1(SE2 PTE kv_forward decode)과 비교 — "
              "verify 그래프로 돈 AR(게이트)은 행 1개에 폭 16 호출이라 비관적이다.")
    else:
        sys.exit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
