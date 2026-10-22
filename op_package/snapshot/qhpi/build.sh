#!/usr/bin/env bash
# Build the DeltaRule QHPI op package for the snapshot (state-saving) target: host x86 (export), Hexagon (HTP) and Android libraries.
# usage: HEX_ARCH=v81|v79 DELTA_INT16=1 DELTA_INT16_PREFILL=2 bash build.sh <out_dir>
#   HEX_ARCH v81 = SM8850 (Galaxy S26 Ultra), v79 = SM8750 (Galaxy S25); the flags above are the ones of the measured builds.
#   needs QNN_SDK_ROOT (QAIRT 2.49), DELTA_HEX_TOOLS (Hexagon SDK 6.6 .../HEXAGON_Tools/19.0.07/Tools), ANDROID_NDK_ROOT (r26c)
set -euo pipefail
SRC=$(cd "$(dirname "$0")" && pwd)
OUT=${1:?output package directory required}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
SDK=${QNN_SDK_ROOT:?set QNN_SDK_ROOT}
HEX=${DELTA_HEX_TOOLS:?set DELTA_HEX_TOOLS}
NDK=${ANDROID_NDK_ROOT:?set ANDROID_NDK_ROOT}
ARCH=${HEX_ARCH:-v81}
if [ "${DELTA_THREADS:-0}" != 0 ]; then
  echo 'SDK 2.49/v79 cannot deserialize QHPI self-slicing (q::sync_zero). Use QNN_DELTA_HEAD_GROUPS=4 instead.' >&2
  exit 2
fi
if [ "${DELTA_INT16:-0}" = 0 ]; then
  echo 'The snapshot package requires DELTA_INT16=1 (or a strict integer mode).' >&2
  exit 2
fi
mkdir -p "$OUT"/build/{x86_64-linux-clang,hexagon-$ARCH,aarch64-android} "$OUT/config"
cp "$SRC/delta_rule_op_package_htp.xml" "$OUT/config/"
COMMON=(-std=c++17 -O2 -fPIC -shared -I"$SDK/include/QNN" -DDELTA_THREADS="${DELTA_THREADS:-0}")
COMMON+=(-DDELTA_DECODE_DIRECT_WRITE="${DELTA_DECODE_DIRECT_WRITE:-0}")
COMMON+=(-DDELTA_STATE16_TILE="${DELTA_STATE16_TILE:-64}")
COMMON+=(-DDELTA_ROW_REQUANT="${DELTA_ROW_REQUANT:-0}")
COMMON+=(-DDELTA_INT16="${DELTA_INT16:-0}")
COMMON+=(-DDELTA_INT16_PREFILL="${DELTA_INT16_PREFILL:-0}")
case "${DELTA_STATE16_FUSE_DECAY:-0}" in
  0|1) ;;
  *) echo 'DELTA_STATE16_FUSE_DECAY must be 0 or 1' >&2; exit 2 ;;
esac
COMMON+=(-DDELTA_STATE16_FUSE_DECAY="${DELTA_STATE16_FUSE_DECAY:-0}")
if [ "${DELTA_DIAGNOSTICS:-0}" = 1 ]; then COMMON+=(-DDELTA_DIAGNOSTICS); fi
SOURCES=("$SRC/Interface.cpp" "$SRC/DeltaRule.cpp")
"${CXX:-g++}" "${COMMON[@]}" "${SOURCES[@]}" -o "$OUT/build/x86_64-linux-clang/libQnnDeltaRuleOpPackage.so"
"$HEX/bin/hexagon-clang++" "${COMMON[@]}" -m$ARCH -mhvx -mhvx-length=128B "${SOURCES[@]}" -o "$OUT/build/hexagon-$ARCH/libQnnDeltaRuleOpPackage.so"
# The Android backend also discovers qhpi_init when registering the package.
# Statically link C++ support so deployment does not require libc++_shared.so.
"$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android30-clang++" "${COMMON[@]}" -static-libstdc++ "${SOURCES[@]}" -L"$SDK/lib/aarch64-android" -lQnnHtp -lQnnHtpPrepare -o "$OUT/build/aarch64-android/libQnnDeltaRuleOpPackage.so"
cat > "$OUT/config/delta_rule_capabilities.json" <<'JSON'
{"implementation":"qhpi","independent_head_workspace":true,"state_dtypes":["float32","uint16"],"state16_workspace_op":true,"decode16_unique_op":true,"requires_output_order_runtime":true,"state16_masked_op":true,"masked_valid_dtype":"int32","state16_snapshots_op":true,"snapshots_workspace_v2":true}
JSON
