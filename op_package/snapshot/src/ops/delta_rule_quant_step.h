#ifndef DELTA_RULE_QUANT_STEP_H
#define DELTA_RULE_QUANT_STEP_H
#include "delta_rule_io.h"
#ifndef DELTA_STATE16_TILE
#define DELTA_STATE16_TILE 64
#endif
#ifndef DELTA_STATE16_FUSE_DECAY
#define DELTA_STATE16_FUSE_DECAY 0
#endif

namespace deltarule {
#ifdef DELTA_RULE_HAVE_HVX
// Decode-only streaming path: no full float state materialization. Convert
// packed state in registers, accumulate in float, and quantize the final state.
// The default ordering matches step_hvx rounding; FuseDecay is opt-in and
// changes rounding. Input/output encodings are independent. dv is a multiple
// of 64, so vector loads never read a tail.
template <int NV, bool FuseDecay = (DELTA_STATE16_FUSE_DECAY != 0)>
inline void step_u16_hvx_n(int dk, int dv, const uint16_t* input, uint16_t* output,
    int input_zero, float input_scale, float output_zero, float output_inv,
    const float* q, const float* k, const float* v, float decay, float beta, float* out, float* workspace = nullptr) {
  HVX_Vector accum[NV], delta[NV];
  const HVX_Vector izero = Q6_V_vsplat_R(input_zero);
  // Opt-in: one scalar multiply per head/tile replaces one vector multiply
  // for every state vector in both passes. This changes FP32 rounding, so
  // retain the historical ordering by default and validate recurrent drift.
  const HVX_Vector iscale = vsplat_f(FuseDecay ? input_scale * decay : input_scale);
  const HVX_Vector dec = vsplat_f(decay);
  auto unpack = [&](HVX_Vector words) {
    auto f = Q6_Vsf_equals_Vw(Q6_Vw_vsub_VwVw(words, izero));
    if constexpr (FuseDecay) return Q6_Vsf_vmpy_VsfVsf(f, iscale);
    else return Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(f, iscale), dec);
  };
  for (int n = 0; n < NV; ++n) accum[n] = Q6_V_vzero();
  for (int i = 0; i < dk; ++i) {
    auto ki = vsplat_f(k[i]);
    for (int n = 0; n < NV; n += 2) {
      auto pair = Q6_Wuw_vunpack_Vuh(*reinterpret_cast<const HVX_UVector*>(input + i * dv + n * 32));
      accum[n] = Q6_Vqf32_vadd_Vqf32Vqf32(accum[n], Q6_Vqf32_vmpy_VsfVsf(unpack(Q6_V_lo_W(pair)), ki));
      accum[n+1] = Q6_Vqf32_vadd_Vqf32Vqf32(accum[n+1], Q6_Vqf32_vmpy_VsfVsf(unpack(Q6_V_hi_W(pair)), ki));
    }
  }
  for (int n = 0; n < NV; ++n) {
    auto vn = *reinterpret_cast<const HVX_Vector*>(v + n * 32);
    delta[n] = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vsub_VsfVsf(vn, Q6_Vsf_equals_Vqf32(accum[n])), vsplat_f(beta));
    accum[n] = Q6_V_vzero();
  }
  const HVX_Vector ozero = vsplat_f(output_zero), oinv = vsplat_f(output_inv);
  auto quantize = [&](HVX_Vector x) {
    x = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(x, oinv), ozero);
    x = Q6_Vsf_vmin_VsfVsf(Q6_Vsf_vmax_VsfVsf(x, Q6_V_vzero()), vsplat_f(65535.f));
    return Q6_Vw_equals_Vsf(Q6_Vsf_vadd_VsfVsf(x, vsplat_f(0.5f)));
  };
  for (int i = 0; i < dk; ++i) {
    auto ki = vsplat_f(k[i]), qi = vsplat_f(q[i]);
    for (int n = 0; n < NV; n += 2) {
      auto pair = Q6_Wuw_vunpack_Vuh(*reinterpret_cast<const HVX_UVector*>(input + i * dv + n * 32));
      auto lo = Q6_Vsf_vadd_VsfVsf(unpack(Q6_V_lo_W(pair)), Q6_Vsf_vmpy_VsfVsf(ki, delta[n]));
      auto hi = Q6_Vsf_vadd_VsfVsf(unpack(Q6_V_hi_W(pair)), Q6_Vsf_vmpy_VsfVsf(ki, delta[n+1]));
      if (workspace) {
        *reinterpret_cast<HVX_UVector*>(workspace + i * dv + n * 32) = lo;
        *reinterpret_cast<HVX_UVector*>(workspace + i * dv + (n+1) * 32) = hi;
      }
      accum[n] = Q6_Vqf32_vadd_Vqf32Vqf32(accum[n], Q6_Vqf32_vmpy_VsfVsf(lo, qi));
      accum[n+1] = Q6_Vqf32_vadd_Vqf32Vqf32(accum[n+1], Q6_Vqf32_vmpy_VsfVsf(hi, qi));
      *reinterpret_cast<HVX_UVector*>(output + i * dv + n * 32) = Q6_Vuh_vpack_VwVw_sat(quantize(hi), quantize(lo));
    }
  }
  for (int n = 0; n < NV; ++n)
    *reinterpret_cast<HVX_Vector*>(out + n * 32) = Q6_Vsf_equals_Vqf32(accum[n]);
}

inline void step_u16_hvx(int dk, int dv, const uint16_t* input, uint16_t* output,
    int input_zero, float input_scale, float output_zero, float output_inv,
    const float* q, const float* k, const float* v, float decay, float beta, float* out, float* workspace = nullptr) {
  if (dv == 128 && DELTA_STATE16_TILE == 128)
    step_u16_hvx_n<4>(dk, dv, input, output, input_zero, input_scale, output_zero, output_inv, q,k,v,decay,beta,out,workspace);
  else for (int col = 0; col < dv; col += 64)
    step_u16_hvx_n<2>(dk, dv, input+col, output+col, input_zero, input_scale, output_zero, output_inv, q,k,v+col,decay,beta,out+col,workspace ? workspace+col : nullptr);
}
#endif
} // namespace deltarule
#endif
