# DeltaRule op package (Hexagon HTP, QHPI)

The Gated DeltaNet recurrence of Qwen3.5 runs as a custom HTP operation on the HVX vector units. Each call updates the
128 × 128 state of its heads row by row (delta rule), with the state kept as 16-bit integer codes; the 32 value heads are
split into 4 groups of 8 (`QNN_DELTA_HEAD_GROUPS=4`), one op per group.

| Folder | Used by | Operations |
|---|---|---|
| `v1p/` | recompute and ckpt10 targets (all final v5 runs) | `DeltaRule`, `DeltaRuleState16`, `DeltaRuleDecode16`, `DeltaRuleState16Masked` |
| `snapshot/` | the state-saving (snapshot) baseline | the above + `DeltaRuleSnapshots16`, `DeltaRuleReplay16` |
| `config_export/` | read while exporting (`QNN_DELTA_OPPKG_DIR/config`) | op definitions (XML) and the capability flags the export checks |

`DeltaRuleState16Masked` is the verify op of the deferred design: it first commits the accepted rows of the previous call
(`valid_prev`) from the stored ingredients, then runs the new rows.

Sources: `qhpi/DeltaRule.cpp` (op entry points), `qhpi/Interface.cpp` (package interface), `src/ops/dr16m.h` (16-bit
masked kernel), `src/ops/delta_rule_core.h`, `delta_rule_quant_step.h`, `delta_rule_io.h` (shared row update and IO).

## Build

```bash
export QNN_SDK_ROOT=<QAIRT 2.49.0.260730> ANDROID_NDK_ROOT=<android-ndk-r26c>
export DELTA_HEX_TOOLS=<Hexagon SDK 6.6.0.0>/tools/HEXAGON_Tools/19.0.07/Tools
HEX_ARCH=v81 DELTA_INT16=1 DELTA_INT16_PREFILL=1 bash v1p/qhpi/build.sh out/v1p_v81        # S26 Ultra (SM8850)
HEX_ARCH=v79 DELTA_INT16=1 DELTA_INT16_PREFILL=1 bash v1p/qhpi/build.sh out/v1p_v79        # S25 (SM8750)
HEX_ARCH=v81 DELTA_INT16=1 DELTA_INT16_PREFILL=2 bash snapshot/qhpi/build.sh out/snap_v81  # snapshot baseline
```

Each build writes `build/x86_64-linux-clang` (host, used by the export), `build/hexagon-<arch>` (the HTP library, pushed
to the phone as `libQnnDeltaRuleOpPackage_HTP.so`), `build/aarch64-android` and `config/`. The runner registers it with
`QNN_OP_PACKAGES=<dir>/libQnnDeltaRuleOpPackage_HTP.so:DeltaRuleOpPackageInterfaceProvider:HTP` (see any
`results/*/*.command.txt`).

Check of these sources against the measured S26 libraries: rebuilding with the flags above gives byte-identical x86 and
Android libraries; the Hexagon libraries differ only in the order of PLT entries and the embedded link command (the
disassembly of `.text` is identical after replacing addresses).

`config_export/` holds the op definitions the target exports were compiled against: those of the snapshot package
before `DeltaRuleReplay16` was added (all other operations are the same).
