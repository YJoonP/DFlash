// Boundary conversion for contiguous QHPI tensors. Recurrence stays float.
#ifndef DELTA_RULE_IO_H
#define DELTA_RULE_IO_H
#include "delta_rule_core.h"

namespace deltarule {
inline void load_u16(const uint16_t* src, float* dst, size_t n, int offset, float scale) {
  size_t i = 0;
#ifdef DELTA_RULE_HAVE_HVX
  const HVX_Vector voffset = Q6_V_vsplat_R(offset);
  const HVX_Vector vscale = vsplat_f(scale);
  for (; i + 64 <= n; i += 64) {
    const HVX_Vector packed = *reinterpret_cast<const HVX_UVector*>(src + i);
    const HVX_VectorPair wide = Q6_Wuw_vunpack_Vuh(packed);
    const HVX_Vector lo = Q6_Vsf_equals_Vw(Q6_Vw_vsub_VwVw(Q6_V_lo_W(wide), voffset));
    const HVX_Vector hi = Q6_Vsf_equals_Vw(Q6_Vw_vsub_VwVw(Q6_V_hi_W(wide), voffset));
    *reinterpret_cast<HVX_UVector*>(dst + i) = Q6_Vsf_vmpy_VsfVsf(lo, vscale);
    *reinterpret_cast<HVX_UVector*>(dst + i + 32) = Q6_Vsf_vmpy_VsfVsf(hi, vscale);
  }
#endif
  for (; i < n; ++i) dst[i] = (static_cast<float>(src[i]) - offset) * scale;
}

inline void store_u16(const float* src, uint16_t* dst, size_t n, float offset, float inv) {
  size_t i = 0;
#ifdef DELTA_RULE_HAVE_HVX
  const HVX_Vector voffset = vsplat_f(offset);
  const HVX_Vector vinv = vsplat_f(inv);
  const HVX_Vector half = vsplat_f(0.5f);
  const HVX_Vector zero = Q6_V_vzero();
  const HVX_Vector maximum = vsplat_f(65535.f);
  auto convert = [&](const float* p) {
    HVX_Vector x = *reinterpret_cast<const HVX_UVector*>(p);
    x = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(x, vinv), voffset);
    // Match the scalar nonnegative half-up conversion after saturation.
    x = Q6_Vsf_vmin_VsfVsf(Q6_Vsf_vmax_VsfVsf(x, zero), maximum);
    return Q6_Vw_equals_Vsf(Q6_Vsf_vadd_VsfVsf(x, half));
  };
  for (; i + 64 <= n; i += 64) {
    const HVX_Vector lo = convert(src + i), hi = convert(src + i + 32);
    *reinterpret_cast<HVX_UVector*>(dst + i) = Q6_Vuh_vpack_VwVw_sat(hi, lo);
  }
#endif
  for (; i < n; ++i) {
    float q = src[i] * inv + offset;
    q = q < 0.f ? 0.f : (q > 65535.f ? 65535.f : q);
    dst[i] = static_cast<uint16_t>(q + 0.5f);
  }
}
// Round a float state onto its uint16 grid and back, with exactly the conversion the
// op applies at a call boundary (store with the output encoding, load with the input
// encoding). Used between rows so a T-row call matches T one-row calls bit for bit.
inline void roundtrip_u16(float* x, size_t n, float out_offset, float out_inv,
                          int in_offset, float in_scale) {
  alignas(128) uint16_t tmp[512];
  for (size_t i = 0; i < n; i += 512) {
    const size_t m = n - i < 512 ? n - i : 512;
    store_u16(x + i, tmp, m, out_offset, out_inv);
    load_u16(tmp, x + i, m, in_offset, in_scale);
  }
}
} // namespace deltarule
#endif
