# Results

Device measurements behind the tables of the README and `docs/REPORT_20261016.md`. All DFlash runs use block size b = 16,
18 Spec-Bench prompts (`inputs_specbench6x3`: MT-Bench 81–83 first turn, WMT14 de→en 161–163, CNN/DM 241–243, Natural
Questions 321–323, GSM8K 401–403, RAG 481–483), the thinking chat template, greedy decoding and 128 new tokens.

| Folder | Phone | Configs | Used for |
|---|---|---|---|
| `s26_v5_20261013` | S26 Ultra (SM8850) | R16v3, R16v5, K16on, K16off | S26 final table; `diag/`: on-device replay test (checkpoint on vs off) |
| `s25_final_20261015` | S25 (SM8750) | S16, R16v5, K16on, K16off | S25 final table, all methods with one runner |
| `s25_vs_old_adopted_20261014` | S25 | O16, C16, R16v5, K16on, K16off | v5 against the earlier exports (NPU time) |
| `s26_snapshot_vs_recompute_20261012` | S26 Ultra | S16, R16 | S26 snapshot (state-saving) number |
| `s26_ar_20261013` | S26 Ultra | AR16 | S26 autoregressive baseline (20.07 tok/s) |
| `s26_spec6_20261016` | S26 Ultra | K16on | six other Spec-Bench prompts (81, 101, 121, 141, 161, 401), first turn |
| `exports/` | | | settings and checks of the measured exports |

Config names: **R16v5** plain recompute v5 · **K16on / K16off** ckpt10 v5 with / without the row-10 checkpoint ·
**S16** snapshot (state-saving) · **O16** old adopted export of 10-02 (no exactness fixes) · **C16** exact conv export of
10-11 · **R16v3** recompute v3 (with the conv-window change) · **R16** (10-12 session) the exact conv export ·
**AR16** autoregressive with the ExecuTorch runner. The 18 prompts of R16v5, K16on and K16off give identical tokens on
both phones.

## Files in a session folder

| File | Contents |
|---|---|
| `main_<CFG>_<id>_run.json` | One run: parsed metrics (TPS, AL, per-step draft / verify ms, …), generated token ids, temperatures before and after, start time (KST) |
| `main_<CFG>_<id>.command.txt` | The exact command executed on the phone (runner environment and arguments) |
| `pre_*`, `smoke_*` | Warm-up and smoke runs before the main session |
| `summary_main.txt`, `summary_v2_main.json` | Output of the session's summarize script (latency breakdown per config, final table, token identity) |
| `measurement_status.json` | Last state written by the measurement loop |
| `inputs/` | Prompts (`prompt_<id>.txt`) and `manifest.json` (token ids, expected prompt length) |
| `logs.tar.xz` | Raw runner logs of every run (and the state dumps of `diag/`); extract with `tar -xJf logs.tar.xz -C <folder>` before running a summarize or analysis script |

`exports/<name>/` holds, per target export, `export_env.txt` (every `QNN_*` switch), the outputs of the checks in
`scripts/check/` (`check_exact.txt`, `check_v3.txt`, `check_og.txt`), and `ctx_target/params.json`,
`shard_io_map.json`, `sha256.txt` (the runner's metadata and the .pte hash). `s25_recompute_v5f` is the variant with
32-bit ingredient IO, exported but not measured. `s25_draft_b16/` and `s26_draft_b16/` describe the two b = 16 drafts.
The model binaries themselves (context binaries, about 4 GB per target) are not in the repository.
