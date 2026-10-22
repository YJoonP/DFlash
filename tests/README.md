# CPU tests of the DFlash protocol (2026-09)

Host-side (fp32, CPU) checks written while porting DFlash to the static Qwen3.5 graph. They import the draft source tree
(`ET_DRAFT`, ExecuTorch + `executorch/patches/draft.patch`, folder named `executorch`, QNN backend built), see
`scripts/env.sh`.

| Test | What it checks | Needs |
|---|---|---|
| `test_deferred_state_parity.py` | Deferred commit (`QNN_DEFER_STATE=1`, T-row verify graph) against immediate one-token steps: with random accept counts per step, the logits at accepted positions and the final DeltaNet / conv states must match. A small random model; `QNN_DELTA_KERNEL=0` (unrolled recurrence) or `1` (custom op, CPU reference) | `ET_DRAFT` |
| `test_dflash_draft_parity.py` | Static DFlash draft (`dflash_draft.py`, converted checkpoint) against the z-lab implementation, two steps (empty cache; 12 committed rows), logits max abs error < 2e-2 | `ET_DRAFT`, `CKPT_DIR`, `DFLASH_REF_DIR` (a clone of [z-lab/dflash](https://github.com/z-lab/dflash) with its `model.py`), the z-lab/Qwen3.5-4B-DFlash weights (`DFLASH_HF_DIR`, or downloaded) |
| `dflash_sd_reference.py` | Reference speculative-decoding loop on the CPU (static target + static draft): the SD output must equal the target's greedy AR output, and it reports the acceptance length | `ET_DRAFT`, `CKPT_DIR` |

```bash
ET_DRAFT=<checkout>/executorch QNN_DELTA_KERNEL=1 python3 tests/test_deferred_state_parity.py   # prints PASS
```

`test_deferred_state_parity.py` was re-run on the published draft tree (both `QNN_DELTA_KERNEL` modes): PASS, logits max
error 8.3e-7.
