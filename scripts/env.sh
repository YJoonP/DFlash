# Paths shared by the export scripts (sourced by them). Set the variables in the environment before running a script.
#   ET            ExecuTorch checkout at c27baa8031 with executorch/patches/target_v5.patch applied (target export).
#                 The folder must be named "executorch": its parent goes on PYTHONPATH so that `import executorch`
#                 reads this checkout. Build its QNN backend into $ET/build-x86 first (backends/qualcomm/scripts/build.sh).
#   ET_DRAFT      the same with executorch/patches/draft.patch instead (draft export only)
#   QNN_SDK_ROOT  QAIRT 2.49.0.260730
#   PY            python of the ExecuTorch environment (default: python3)
#   CKPT_DIR      converted checkpoints: qwen3_5_4b_meta.pth (target), qwen3_5_4b_dflash_meta.pth (draft),
#                 qwen3_5_4b_meta_8L.pth (first 8 target layers, only for the diag8 mode)
#   OPPKG_X86     DeltaRule op package folder read while compiling the target: config/ (= op_package/config_export)
#                 and build/x86_64-linux-clang/libQnnDeltaRuleOpPackage.so (op_package/*/qhpi/build.sh)
#   TMPROOT       scratch folder for compiler temporaries and context extraction (default: $TMPDIR or /tmp)
: "${QNN_SDK_ROOT:?set QNN_SDK_ROOT (QAIRT 2.49.0.260730)}"
: "${PY:=python3}"
: "${TMPROOT:=${TMPDIR:-/tmp}}"
export QNN_SDK_ROOT PY TMPROOT
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TZ=Asia/Seoul
