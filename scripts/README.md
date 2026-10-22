# Scripts

| Folder | Contents |
|---|---|
| `env.sh` | Paths the export scripts need (`ET`, `ET_DRAFT`, `QNN_SDK_ROOT`, `PY`, `CKPT_DIR`, `OPPKG_X86`, `TMPROOT`); see the comments in the file |
| `export/` | Target and draft export, context extraction, runner metadata |
| `check/` | Checks of a target export on its quantized-graph dump and context binaries |
| `device/` | Phone measurements (thermal-gated, interleaved), summaries, the on-device exactness test |
| `analysis/` | Per-call NPU time against committed rows, checkpoint use |

## Export

```bash
export ET=<checkout>/executorch            # c27baa8031 + executorch/patches/target_v5.patch, QNN backend built in build-x86/
export ET_DRAFT=<checkout2>/executorch     # c27baa8031 + executorch/patches/draft.patch
export QNN_SDK_ROOT=<QAIRT 2.49.0.260730> PY=<python> CKPT_DIR=<checkpoints> OPPKG_X86=<op package build> TMPROOT=<scratch>

# target, final v5: ckpt10 (CKPT_ROW=10, the default) or plain recompute (CKPT_ROW=0); SM8850 = S26 Ultra, SM8750 = S25
ART_DIR=out/s26_ckpt10_v5 CKPT_ROW=10 bash scripts/export/export_target_v5.sh SM8850 full32 16
python3 scripts/check/check_v3.py out/s26_ckpt10_v5          # S1-S4 must print PASS
python3 scripts/check/check_og.py out/s26_ckpt10_v5          # no conv-window / state-input two-grid points
python3 scripts/check/check_exact_ckpt.py out/s26_ckpt10_v5  # or check_exact.py for CKPT_ROW=0

# draft (b = 16)
HIDDEN_DUMP=<hidden states of calib_specbench39.json> ART_DIR=out/s26_draft_b16 bash scripts/export/export_draft.sh SM8850 16 1024
```

- Checkpoints: `qwen3_5_4b_meta.pth` is Qwen/Qwen3.5-4B in the ExecuTorch (meta) layout
  (`examples/models/qwen3_5/convert_weights.py`); `qwen3_5_4b_dflash_meta.pth` is z-lab/Qwen3.5-4B-DFlash converted with
  `examples/models/qwen3_5_dflash/convert_weights.py` of the draft tree.
- `OPPKG_X86` is an op package build folder holding `config/` (copy `op_package/config_export/` there) and
  `build/x86_64-linux-clang/libQnnDeltaRuleOpPackage.so` (`op_package/*/qhpi/build.sh`).
- A full32 target export needs about 100 GiB of free memory and a few hours; run one at a time.
- Calibration: the target uses two Spec-Bench prompts (`calib_samples_b16.json`, `calib_samples_b8.json`); the draft uses
  39 Spec-Bench prompts (`calib_specbench39.json`, sha256 `4918954761ed…`) and the target hidden states of those prompts.
- Checked against our runs: for all four measured v5 exports the script sets exactly the `QNN_*` variables recorded in
  `results/exports/*/export_env.txt`, and `make_params.py` writes the recorded `params.json` of target and draft.
- `export_entry.py` (used with `GATE=`) lets a second export calibrate while the first one compiles.

## Device measurements

The runner that executes DFlash on the phone (target verify + draft, deferred commit, zero copy) is not part of this
repository; the scripts document how it was driven. Each run's exact phone command is in `results/*/*.command.txt`.

```bash
S26_SERIAL=<adb serial> python3 scripts/device/s26/run_v5.py main R16v5,K16on,K16off
python3 scripts/device/s26/summarize_v5.py main > summary_main.txt
```

- Protocol (`device/bench_lib.py`): configs alternate per prompt; before every run at least 60 s rest and two readings
  with SoC < 38 °C and battery < 33 °C; a USB disconnect stops the session. Thinking chat template, greedy, 128 tokens.
- Results go to `results/<session>/` by default; set `RESULTS_DIR` to a new folder (with `inputs/` copied from
  `results/inputs_specbench6x3`) for a new session, because existing `*_run.json` are skipped.
- Phone paths inside the scripts are the folders of our runs.
- `s26/test_exact_device.sh`: teacher-forced replay with a diagnosis build of the runner (logit fingerprints and committed
  state dumps), checkpoint on vs off.
- Summaries and analyses read the run logs: extract `results/<session>/logs.tar.xz` into the folder first. Run on the
  published data, `summarize_v5.py`, `summarize_s25v5.py`, `summarize_s25fin.py` and `summarize_spec6.py` reproduce the
  recorded `summary_main.txt` files exactly.

| Script | Session (results/) |
|---|---|
| `s26/run_v5.py`, `s26/summarize_v5.py` | `s26_v5_20261013` |
| `s25/run_s25fin.py`, `s25/summarize_s25fin.py` | `s25_final_20261015` |
| `s25/run_s25v5.py`, `s25/summarize_s25v5.py` | `s25_vs_old_adopted_20261014` |
| `s26/make_spec6_inputs.py`, `s26/run_spec6.py`, `s26/summarize_spec6.py` | `s26_spec6_20261016` |
| `analysis/*_s26.py`, `analysis/*_s25.py` | `s26_v5_20261013`, `s25_vs_old_adopted_20261014` |
