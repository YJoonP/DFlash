// dr16m: 설계 1(delta_rule_int16.h) + 설계 2(int_step.h) 병합 시제품 (2026-09-28 KST).
// 생산 op package 는 건드리지 않는다. dk = dv = 128, 입력/출력 상태 인코딩이 같을 때만.
// zp > 32767 (2026-09-29): 2차 통과는 x = u - 32768 과 zp - 32768 로 계산하고 부호 포화 pack 뒤 부호 비트를 뒤집는다
//   x*A + d2*K2 + (zp-32768)(32768-A) = u*A + d2*K2 + zp*(32768-A) - 2^30  ->  반올림 결과가 정확히 -32768 이동.
//
// u = 저장 u16, S = s*(u - zp), x = u ^ 0x8000 = u - 32768 (int16).
// 준비 (k-head 당, 같은 q/k 의 v-head 끼리 공유 가능):
//   X = rne(clamp(x_i * 2^e, +-32767)), e = min(14 - ilogb(max|x|), e1),
//   e1 = B*2^e1 <= Lm 인 최대 정수, B = L1f*(1+2^-15), L1f = 정해진 순서의 float 합 (l1_tree),
//   Lm = floor((2^31-1)/max(zp,65535-zp)) - 64  ->  L1(X)*max(zp,65535-zp) <= 2^31-1 보장
//   zK = sumK * (zp - 32768) mod 2^32
// decay: A = clamp(((uint32)(clamp01(decay)*2^24) + dith9) >> 9, 1, 32768), dith9 = 입력 해시 (DR16_DECAY_DITHER=0 이면 256)
// 1차 통과 (접기 없음, mod 2^32): P_j = sum_i K_i x_ij,  R_j = sum_i Q_i x_ij
// 열 처리: T = P - zK (정확), kv = (A/2^15 * s * 2^-eK) * T, d = beta (v - kv), o = (A/2^15 s 2^-eQ) U + qk d
// 2차 통과 (설계 1 그대로): acc = sat32(u*A + D2_j*K2_i + zp*(2^15 - A)), u' = usat16((acc + 2^14) >> 15)
#ifndef DR16M_H
#define DR16M_H
#include <cstdint>
#include <cstring>
#include <cmath>
#if defined(__hexagon__) && defined(__HVX__)
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#define DR16M_HAVE_HVX 1
#endif
#ifndef DR16_DECAY_DITHER
#define DR16_DECAY_DITHER 1
#endif
#ifndef DR16M_P1_UNROLL
#define DR16M_P1_UNROLL 2
#endif

namespace dr16m {
constexpr int DK = 128, DV = 128;

struct alignas(128) Tables {
  int32_t RK[DK], RQ[DK], R2[DK];
  int32_t W[4][32];
  int32_t sumK, sumQ;
};
struct Stats { long heads = 0, split = 0, hard = 0, rK = 0, rQ = 0, rKmax = 0; };   // rK/rQ: eK/eQ 합 (이름은 이전 판 흔적)

static inline float rne(float x) { return (x + 12582912.0f) - 12582912.0f; }
static inline float clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }
static inline float bits_f(uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }
static inline uint32_t f_bits(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }
static inline float pow2f(int n) { n = n < -126 ? -126 : (n > 127 ? 127 : n); return bits_f(static_cast<uint32_t>(n + 127) << 23); }
static inline int exp_of(float x) {
  const int eb = static_cast<int>((f_bits(x) >> 23) & 0xFF);
  return x > 0 ? (eb == 0 ? -126 : eb - 126) : -30;
}
static inline int ilogb_pos(float m) { return static_cast<int>((f_bits(m) >> 23) & 0xFF) - 127; }
static inline uint32_t mix32(uint32_t x) {
  x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15; x *= 0x846CA68Bu; x ^= x >> 16; return x;
}
// decay -> Q15 정수 A (디더 또는 반올림). 같은 행 입력이면 같은 A (SD == AR).
static inline int decay_q15(float decay, const float* v, int head) {
  uint32_t dith = 256;
#if DR16_DECAY_DITHER
  uint32_t h = mix32(f_bits(v[0]) ^ (0x9E3779B9u * static_cast<uint32_t>(head + 1)));
  h = mix32(h ^ f_bits(v[1]));
  h = mix32(h ^ f_bits(v[2]));
  h = mix32(h ^ f_bits(v[3]) ^ f_bits(decay));
  dith = h >> 23;
#endif
  const float dc = decay > 0.f ? (decay < 1.f ? decay : 1.f) : 0.f;   // NaN -> 0
  const uint32_t F = static_cast<uint32_t>(dc * 16777216.0f);
  const int A = static_cast<int>((F + dith) >> 9);
  return A < 1 ? 1 : (A > 32768 ? 32768 : A);
}
// 호출(층)마다 상수: Lm = floor((2^31-1)/max(zp, 65535-zp)) - 64  (정수, float 로 정확)
static inline float lm_of(int zp) {
  const int Zm = zp > 65535 - zp ? zp : 65535 - zp;
  return static_cast<float>(static_cast<int>(2147483647LL / Zm) - 64);
}
static inline int32_t q16(float x, float sc) { return static_cast<int32_t>(rne(clampf(x * sc, -32767.f, 32767.f))); }
// 128 개 |x| 의 float 합, HVX 와 같은 순서 (레인별 4벡터 순차 합 -> 32레인 회전 나비).
static inline float l1_tree(const float* x) {
  float p[32];
  for (int l = 0; l < 32; ++l) {
    float t = std::fabs(x[l]);
    t = t + std::fabs(x[32 + l]); t = t + std::fabs(x[64 + l]); t = t + std::fabs(x[96 + l]);
    p[l] = t;
  }
  for (int sh = 16; sh >= 1; sh >>= 1) {
    float n[32];
    for (int l = 0; l < 32; ++l) n[l] = p[l] + p[(l + sh) & 31];
    std::memcpy(p, n, sizeof p);
  }
  return p[0];
}
// 지수 e 선택: e = min(14 - ilogb(m), e1),  e1 = B*2^e1 <= Lm 인 최대 정수, B = L1f*(1+2^-15) >= 참 L1.
// 그러면 L1(X) <= 2^e*L1 + 64 <= Lm + 64 = floor((2^31-1)/Zm) 이므로 |sum_i X_i (u_i - zp)| <= 2^31-1.
static inline int pick_e(float m, float L1f, float Lm, float* sc) {
  if (!(m >= 0x1p-40f)) { *sc = 0.f; return 0; }
  const int e0 = 14 - ilogb_pos(m);
  const float B = L1f * 1.000030517578125f;
  int e1 = ilogb_pos(Lm) - ilogb_pos(B);
  if (B * pow2f(e1) > Lm) --e1;
  const int e = e0 < e1 ? e0 : e1;
  *sc = pow2f(e);
  return e;
}
static inline void quant_ref(const float* x, float Lm, int32_t* X, int* e_out, int32_t* sum_out, float* m_out) {
  float m = 0;
  for (int i = 0; i < DK; ++i) m = std::fmax(m, std::fabs(x[i]));
  *m_out = m;
  float sc;
  *e_out = pick_e(m, l1_tree(x), Lm, &sc);
  int32_t s = 0;
  for (int i = 0; i < DK; ++i) { X[i] = q16(x[i], sc); s += X[i]; }
  *sum_out = s;
}
static inline int choose_a(float kappa, float Mlsb, int* pieces) {
  const int e2 = exp_of(kappa), e1 = exp_of(Mlsb);
  const int amax = 15 - e2;
  *pieces = 1;
  if (e1 + e2 > 15) {
    const float dmax = Mlsb * pow2f(15 - amax);
    *pieces = dmax <= 32767.f ? 1 : (dmax <= 65534.f ? 2 : (dmax <= 98301.f ? 3 : 4));
    return amax;
  }
  int a = 15 + e1 - e2;
  a = a >= 0 ? a / 2 : -((-a + 1) / 2);
  return a < -60 ? -60 : (a > 60 ? 60 : a);
}
static inline float qk_dot(const float* q, const float* k) {
  float p[32];
  for (int l = 0; l < 32; ++l) {
    float t = q[l] * k[l];
    float t1 = q[32 + l] * k[32 + l]; t = t + t1;
    float t2 = q[64 + l] * k[64 + l]; t = t + t2;
    float t3 = q[96 + l] * k[96 + l]; t = t + t3;
    p[l] = t;
  }
  for (int sh = 16; sh >= 1; sh >>= 1) {
    float n[32];
    for (int l = 0; l < 32; ++l) n[l] = p[l] + p[(l + sh) & 31];
    std::memcpy(p, n, sizeof p);
  }
  return p[0];
}

// ---------------------------------------------------------------------------
// C 참조 (HVX 와 비트 동일). in == out 허용.
static inline void step_ref(const uint16_t* in, uint16_t* out, int zp, float s, float inv_s, float Lm,
                            const float* q, const float* k, const float* v, float decay, float beta, int head,
                            float* o, Stats* st = nullptr) {
  int32_t K[DK], Q[DK]; int eK, eQ; int32_t sumK, sumQ; float kappa, mq;
  quant_ref(k, Lm, K, &eK, &sumK, &kappa);
  quant_ref(q, Lm, Q, &eQ, &sumQ, &mq);
  const int rK = eK, rQ = eQ;
  const uint32_t zK = static_cast<uint32_t>(sumK) * static_cast<uint32_t>(zp - 32768);
  const uint32_t zQ = static_cast<uint32_t>(sumQ) * static_cast<uint32_t>(zp - 32768);
  const float qk = qk_dot(q, k);
  const int A = decay_q15(decay, v, head);
  const float ad = static_cast<float>(A) * 0x1p-15f;
  const float as = ad * s;
  const float ck = as * pow2f(-eK), cq = as * pow2f(-eQ);
  float d[DV]; float M = 0;
  for (int j = 0; j < DV; ++j) {
    uint32_t p = 0, r = 0;
    for (int i = 0; i < DK; ++i) {
      const int32_t x = static_cast<int32_t>(in[i * DV + j]) - 32768;
      p += static_cast<uint32_t>(K[i] * x); r += static_cast<uint32_t>(Q[i] * x);
    }
    const int32_t T = static_cast<int32_t>(p - zK), U = static_cast<int32_t>(r - zQ);
    const float kv = static_cast<float>(T) * ck;
    const float t = v[j] - kv;
    d[j] = t * beta;
    const float oo = static_cast<float>(U) * cq;
    const float w = qk * d[j];
    o[j] = oo + w;
    M = std::fmax(M, std::fabs(d[j]));
  }
  int np = 1;
  const int a = choose_a(kappa, M * inv_s, &np);
  if (st) { st->heads++; st->split += np >= 2; st->hard += np >= 4; st->rK += rK; st->rQ += rQ; if (rK > st->rKmax) st->rKmax = rK; }
  const float c2 = inv_s * pow2f(15 - a), ka = pow2f(a);
  const float dlim = 32767.f * np;
  int32_t K2[DK], D2p[4][DV];
  for (int i = 0; i < DK; ++i) K2[i] = static_cast<int32_t>(rne(clampf(k[i] * ka, -32767.f, 32767.f)));
  for (int j = 0; j < DV; ++j) {
    int32_t t = static_cast<int32_t>(rne(clampf(d[j] * c2, -dlim, dlim)));
    for (int p = 0; p < np; ++p) {
      const int32_t piece = t > 32767 ? 32767 : (t < -32767 ? -32767 : t);
      D2p[p][j] = piece; t -= piece;
    }
  }
  const bool zb = zp > 32767;
  const int64_t C = static_cast<int64_t>(zb ? zp - 32768 : zp) * (32768 - A);
  auto sat32 = [](int64_t x) { return x > INT32_MAX ? INT32_MAX : (x < INT32_MIN ? INT32_MIN : x); };
  for (int i = 0; i < DK; ++i)
    for (int j = 0; j < DV; ++j) {
      const int64_t u = static_cast<int64_t>(in[i * DV + j]) - (zb ? 32768 : 0);
      int64_t acc = sat32(u * A + static_cast<int64_t>(D2p[0][j]) * K2[i] + C);
      for (int p = 1; p < np; ++p) acc = sat32(acc + static_cast<int64_t>(D2p[p][j]) * K2[i]);
      const int64_t r = ((acc + 16384) >> 15) + (zb ? 32768 : 0);
      out[i * DV + j] = static_cast<uint16_t>(r < 0 ? 0 : (r > 65535 ? 65535 : r));
    }
}

#ifdef DR16M_HAVE_HVX
static inline HVX_Vector splat_f(float f) { return Q6_V_vsplat_R(static_cast<int32_t>(f_bits(f))); }
static inline HVX_Vector rne_v(HVX_Vector x, HVX_Vector magic) { return Q6_Vsf_vsub_VsfVsf(Q6_Vsf_vadd_VsfVsf(x, magic), magic); }
static inline HVX_Vector clamp_v(HVX_Vector x, HVX_Vector lo, HVX_Vector hi) { return Q6_Vsf_vmin_VsfVsf(Q6_Vsf_vmax_VsfVsf(x, lo), hi); }
static inline int32_t lane0(HVX_Vector x) { union { HVX_Vector v; int32_t w[32]; } u; u.v = x; return u.w[0]; }
static inline HVX_Vector reduce_add_w(HVX_Vector x) {
#pragma unroll
  for (int sh = 64; sh >= 4; sh >>= 1) x = Q6_Vw_vadd_VwVw(x, Q6_V_vror_VR(x, sh));
  return x;
}
static inline HVX_Vector reduce_max_w(HVX_Vector x) {
#pragma unroll
  for (int sh = 64; sh >= 4; sh >>= 1) x = Q6_Vw_vmax_VwVw(x, Q6_V_vror_VR(x, sh));
  return x;
}
static inline float as_f(int32_t i) { return bits_f(static_cast<uint32_t>(i)); }

// 2차 통과 첫 곱과 pack. ZB (zp > 32767) 는 x = u - 32768 부호 곱과 부호 포화 pack + 부호 비트 반전.
template <bool ZB> static inline HVX_VectorPair p2_mul(HVX_Vector u, int32_t RD) {
  if constexpr (ZB) return Q6_Ww_vmpy_VhVuh(Q6_V_vxor_VV(u, Q6_V_vsplat_R(0x80008000)), Q6_V_vsplat_R(RD));
  else return Q6_Wuw_vmpy_VuhRuh(u, RD);
}
template <bool ZB> static inline HVX_Vector p2_pack(HVX_Vector hi, HVX_Vector lo) {
  if constexpr (ZB) return Q6_V_vxor_VV(Q6_Vh_vasr_VwVwR_rnd_sat(hi, lo, 15), Q6_V_vsplat_R(0x80008000));
  else return Q6_Vuh_vasr_VwVwR_rnd_sat(hi, lo, 15);
}
static inline HVX_Vector zp_hi(int zp) {
  return Q6_V_vsplat_R(static_cast<int32_t>(static_cast<uint32_t>(zp > 32767 ? zp - 32768 : zp) << 16));
}

template <bool ZB>
__attribute__((noinline)) static void pass2_pieces(const HVX_Vector* src, HVX_Vector* dst, const Tables& tb,
                                                   int32_t RD, int np, int zp) {
  const HVX_Vector ilo = Q6_V_vsplat_R(-32767), ihi = Q6_V_vsplat_R(32767), mask16 = Q6_V_vsplat_R(0xFFFF);
  const HVX_Vector zphi = zp_hi(zp);
  HVX_Vector dzp[4][4];
  for (int n = 0; n < 4; ++n) {
    HVX_Vector w = reinterpret_cast<const HVX_Vector*>(tb.W)[n];
    for (int p = 0; p < 4; ++p) {
      HVX_Vector wa = Q6_Vw_vmin_VwVw(Q6_Vw_vmax_VwVw(w, ilo), ihi);
      dzp[p][n] = p == 0 ? Q6_V_vor_VV(Q6_V_vand_VV(wa, mask16), zphi) : Q6_V_vand_VV(wa, mask16);
      w = Q6_Vw_vsub_VwVw(w, wa);
    }
  }
  for (int i = 0; i < DK; ++i) {
    const int32_t R = tb.R2[i];
    HVX_VectorPair P0 = p2_mul<ZB>(src[2 * i], RD), P1 = p2_mul<ZB>(src[2 * i + 1], RD);
    HVX_Vector acc[4] = {Q6_V_lo_W(P0), Q6_V_hi_W(P0), Q6_V_lo_W(P1), Q6_V_hi_W(P1)};
    for (int n = 0; n < 4; ++n)
      for (int p = 0; p < np; ++p) acc[n] = Q6_Vw_vdmpyacc_VwVhRh_sat(acc[n], dzp[p][n], R);
    dst[2 * i] = p2_pack<ZB>(acc[1], acc[0]);
    dst[2 * i + 1] = p2_pack<ZB>(acc[3], acc[2]);
  }
}

// 2차 통과 본선 (설계 1 의 손 파이프라인 그대로)
template <bool ZB>
__attribute__((always_inline)) static inline void pass2_main(const HVX_Vector* src, HVX_Vector* dst, const int32_t* r2,
                                                             int32_t RD, const HVX_Vector* dz) {
  HVX_VectorPair A0 = p2_mul<ZB>(src[0], RD), A1 = p2_mul<ZB>(src[1], RD);
  HVX_VectorPair B0, B1;
  auto finish = [&](HVX_VectorPair X0, HVX_VectorPair X1, int32_t R, HVX_Vector* d) {
    const HVX_Vector l0 = Q6_Vw_vdmpyacc_VwVhRh_sat(Q6_V_lo_W(X0), dz[0], R);
    const HVX_Vector h0 = Q6_Vw_vdmpyacc_VwVhRh_sat(Q6_V_hi_W(X0), dz[1], R);
    const HVX_Vector l1 = Q6_Vw_vdmpyacc_VwVhRh_sat(Q6_V_lo_W(X1), dz[2], R);
    const HVX_Vector h1 = Q6_Vw_vdmpyacc_VwVhRh_sat(Q6_V_hi_W(X1), dz[3], R);
    d[0] = p2_pack<ZB>(h0, l0);
    d[1] = p2_pack<ZB>(h1, l1);
  };
  const int64_t* r2p = reinterpret_cast<const int64_t*>(r2);
  int64_t Rab = r2p[0];
  for (int i = 0; i < DK - 2; i += 2) {
    const int64_t Rn = r2p[i / 2 + 1];
    const int32_t Ra = static_cast<int32_t>(Rab), Rb = static_cast<int32_t>(Rab >> 32);
    B0 = p2_mul<ZB>(src[2 * i + 2], RD); B1 = p2_mul<ZB>(src[2 * i + 3], RD);
    finish(A0, A1, Ra, dst + 2 * i);
    A0 = p2_mul<ZB>(src[2 * i + 4], RD); A1 = p2_mul<ZB>(src[2 * i + 5], RD);
    finish(B0, B1, Rb, dst + 2 * i + 2);
    Rab = Rn;
  }
  B0 = p2_mul<ZB>(src[2 * DK - 2], RD); B1 = p2_mul<ZB>(src[2 * DK - 1], RD);
  finish(A0, A1, r2[DK - 2], dst + 2 * DK - 4);
  finish(B0, B1, r2[DK - 1], dst + 2 * DK - 2);
}

static inline void step_hvx(const uint16_t* in, uint16_t* out, int zp, float s, float inv_s, float Lm,
                            const float* q, const float* k, const float* v, float decay, float beta, int head,
                            float* o, Tables& tb, Stats* st = nullptr) {
  const HVX_Vector* kv = reinterpret_cast<const HVX_Vector*>(k);
  const HVX_Vector* qv = reinterpret_cast<const HVX_Vector*>(q);
  const HVX_Vector* vv = reinterpret_cast<const HVX_Vector*>(v);
  const HVX_Vector magic = splat_f(12582912.0f), lo32k = splat_f(-32767.f), hi32k = splat_f(32767.f);
  const HVX_Vector mask16 = Q6_V_vsplat_R(0xFFFF), absmask = Q6_V_vsplat_R(0x7FFFFFFF);
  const int A = decay_q15(decay, v, head);      // 스칼라, HVX 준비와 겹쳐 실행
  // --- 준비 (k-head 단위로 공유 가능): max|k|, max|q|, L1(k), L1(q), q.k 를 한 번에 (나무 5개 동시)
  HVX_Vector km = Q6_V_vzero(), qm = km, kl = km, ql = km, p = km;
#pragma unroll
  for (int n = 0; n < 4; ++n) {
    const HVX_Vector ak = Q6_V_vand_VV(kv[n], absmask), aq = Q6_V_vand_VV(qv[n], absmask);
    km = Q6_Vw_vmax_VwVw(km, ak); qm = Q6_Vw_vmax_VwVw(qm, aq);
    const HVX_Vector pr = Q6_Vsf_vmpy_VsfVsf(qv[n], kv[n]);
    if (n == 0) { kl = ak; ql = aq; p = pr; }
    else { kl = Q6_Vsf_vadd_VsfVsf(kl, ak); ql = Q6_Vsf_vadd_VsfVsf(ql, aq); p = Q6_Vsf_vadd_VsfVsf(p, pr); }
  }
#pragma unroll
  for (int sh = 64; sh >= 4; sh >>= 1) {
    km = Q6_Vw_vmax_VwVw(km, Q6_V_vror_VR(km, sh)); qm = Q6_Vw_vmax_VwVw(qm, Q6_V_vror_VR(qm, sh));
    kl = Q6_Vsf_vadd_VsfVsf(kl, Q6_V_vror_VR(kl, sh)); ql = Q6_Vsf_vadd_VsfVsf(ql, Q6_V_vror_VR(ql, sh));
    p = Q6_Vsf_vadd_VsfVsf(p, Q6_V_vror_VR(p, sh));
  }
  const float qk = as_f(lane0(p));
  const float kappa = as_f(lane0(km)), mq = as_f(lane0(qm));
  float sck, scq;
  const int eK = pick_e(kappa, as_f(lane0(kl)), Lm, &sck), eQ = pick_e(mq, as_f(lane0(ql)), Lm, &scq);
  const int rK = eK, rQ = eQ;
  {
    const HVX_Vector vsk = splat_f(sck), vsq = splat_f(scq);
    HVX_Vector sK = Q6_V_vzero(), sQ = sK;
    HVX_Vector* RKv = reinterpret_cast<HVX_Vector*>(tb.RK);
    HVX_Vector* RQv = reinterpret_cast<HVX_Vector*>(tb.RQ);
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      const HVX_Vector wk = Q6_Vw_equals_Vsf(rne_v(clamp_v(Q6_Vsf_vmpy_VsfVsf(kv[n], vsk), lo32k, hi32k), magic));
      const HVX_Vector wq = Q6_Vw_equals_Vsf(rne_v(clamp_v(Q6_Vsf_vmpy_VsfVsf(qv[n], vsq), lo32k, hi32k), magic));
      sK = Q6_Vw_vadd_VwVw(sK, wk); sQ = Q6_Vw_vadd_VwVw(sQ, wq);
      RKv[n] = Q6_V_vor_VV(Q6_V_vand_VV(wk, mask16), Q6_Vw_vasl_VwR(wk, 16));
      RQv[n] = Q6_V_vor_VV(Q6_V_vand_VV(wq, mask16), Q6_Vw_vasl_VwR(wq, 16));
    }
#pragma unroll
    for (int sh = 64; sh >= 4; sh >>= 1) {
      sK = Q6_Vw_vadd_VwVw(sK, Q6_V_vror_VR(sK, sh)); sQ = Q6_Vw_vadd_VwVw(sQ, Q6_V_vror_VR(sQ, sh));
    }
    tb.sumK = lane0(sK); tb.sumQ = lane0(sQ);
  }
  const uint32_t zK = static_cast<uint32_t>(tb.sumK) * static_cast<uint32_t>(zp - 32768);
  const uint32_t zQ = static_cast<uint32_t>(tb.sumQ) * static_cast<uint32_t>(zp - 32768);
  const float ad = static_cast<float>(A) * 0x1p-15f;
  const float as = ad * s;
  const float ck = as * pow2f(-eK), cq = as * pow2f(-eQ);
#if defined(DR16M_STOP) && DR16M_STOP == 1
  o[0] = ck + cq + static_cast<float>(zK ^ zQ ^ tb.RK[5] ^ tb.RQ[9]); return;
#endif
  // --- 1차 통과: 접기 없음, mod 2^32 누산
  const HVX_Vector sign = Q6_V_vsplat_R(0x80008000);
  HVX_VectorPair aK0 = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero()), aK1 = aK0, aQ0 = aK0, aQ1 = aK0;
  const HVX_Vector* src = reinterpret_cast<const HVX_Vector*>(in);
  const int32_t* RK = tb.RK; const int32_t* RQ = tb.RQ;
#pragma clang loop unroll_count(DR16M_P1_UNROLL)
  for (int i = 0; i < DK; ++i) {
    const HVX_Vector x0 = Q6_V_vxor_VV(src[2 * i], sign), x1 = Q6_V_vxor_VV(src[2 * i + 1], sign);
    const int32_t rk = RK[i], rq = RQ[i];
    aK0 = Q6_Ww_vmpyacc_WwVhRh(aK0, x0, rk); aK1 = Q6_Ww_vmpyacc_WwVhRh(aK1, x1, rk);
    aQ0 = Q6_Ww_vmpyacc_WwVhRh(aQ0, x0, rq); aQ1 = Q6_Ww_vmpyacc_WwVhRh(aQ1, x1, rq);
  }
#if defined(DR16M_STOP) && DR16M_STOP == 2
  o[0] = static_cast<float>(lane0(Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(Q6_V_lo_W(aK0), Q6_V_hi_W(aK1)), Q6_Vw_vadd_VwVw(Q6_V_hi_W(aQ0), Q6_V_lo_W(aQ1)))) + lane0(Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(Q6_V_hi_W(aK0), Q6_V_lo_W(aK1)), Q6_Vw_vadd_VwVw(Q6_V_lo_W(aQ0), Q6_V_hi_W(aQ1))))); return;
#endif
  // --- 열 처리 (짝/홀 순서 n = 2c+h)
  const HVX_Vector vck = splat_f(ck), vcq = splat_f(cq), vbeta = splat_f(beta), vqk = splat_f(qk);
  const HVX_Vector vzK = Q6_V_vsplat_R(static_cast<int32_t>(zK)), vzQ = Q6_V_vsplat_R(static_cast<int32_t>(zQ));
  const HVX_Vector aK[4] = {Q6_V_lo_W(aK0), Q6_V_hi_W(aK0), Q6_V_lo_W(aK1), Q6_V_hi_W(aK1)};
  const HVX_Vector aQ[4] = {Q6_V_lo_W(aQ0), Q6_V_hi_W(aQ0), Q6_V_lo_W(aQ1), Q6_V_hi_W(aQ1)};
  HVX_Vector dd[4], oo[4], mmax = Q6_V_vzero();
#pragma unroll
  for (int c = 0; c < 2; ++c) {
    const HVX_VectorPair vd = Q6_W_vdeal_VVR(vv[2 * c + 1], vv[2 * c], -4);
    const HVX_Vector vx[2] = {Q6_V_lo_W(vd), Q6_V_hi_W(vd)};
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int n = 2 * c + h;
      HVX_Vector t = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_equals_Vw(Q6_Vw_vsub_VwVw(aK[n], vzK)), vck);
      t = Q6_Vsf_vsub_VsfVsf(vx[h], t);
      dd[n] = Q6_Vsf_vmpy_VsfVsf(t, vbeta);
      const HVX_Vector u = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_equals_Vw(Q6_Vw_vsub_VwVw(aQ[n], vzQ)), vcq);
      oo[n] = Q6_Vsf_vadd_VsfVsf(u, Q6_Vsf_vmpy_VsfVsf(vqk, dd[n]));
      mmax = Q6_Vw_vmax_VwVw(mmax, Q6_V_vand_VV(dd[n], absmask));
    }
    const HVX_VectorPair os = Q6_W_vshuff_VVR(oo[2 * c + 1], oo[2 * c], -4);
    reinterpret_cast<HVX_UVector*>(o)[2 * c] = Q6_V_lo_W(os);
    reinterpret_cast<HVX_UVector*>(o)[2 * c + 1] = Q6_V_hi_W(os);
  }
  const float M = as_f(lane0(reduce_max_w(mmax)));
  int np = 1;
  const int a = choose_a(kappa, M * inv_s, &np);
  if (st) { st->heads++; st->split += np >= 2; st->hard += np >= 4; st->rK += rK; st->rQ += rQ; if (rK > st->rKmax) st->rKmax = rK; }
  const HVX_Vector vc2 = splat_f(inv_s * pow2f(15 - a)), vka = splat_f(pow2f(a));
  const HVX_Vector cqhi = Q6_V_vsplat_R(static_cast<int32_t>(static_cast<uint32_t>(32768 - A) << 16));
  const HVX_Vector zphi = zp_hi(zp);
  const bool zb = zp > 32767;
#pragma unroll
  for (int n = 0; n < 4; ++n) {
    const HVX_Vector w = Q6_Vw_equals_Vsf(rne_v(clamp_v(Q6_Vsf_vmpy_VsfVsf(kv[n], vka), lo32k, hi32k), magic));
    reinterpret_cast<HVX_Vector*>(tb.R2)[n] = Q6_V_vor_VV(Q6_V_vand_VV(w, mask16), cqhi);
  }
  const int32_t RD = static_cast<int32_t>(static_cast<uint32_t>(A) | (static_cast<uint32_t>(A) << 16));
  HVX_Vector* dst = reinterpret_cast<HVX_Vector*>(out);
  if (__builtin_expect(np > 1, 0)) {
    const HVX_Vector lim_lo = splat_f(-32767.f * np), lim_hi = splat_f(32767.f * np);
#pragma unroll
    for (int n = 0; n < 4; ++n)
      reinterpret_cast<HVX_Vector*>(tb.W)[n] =
          Q6_Vw_equals_Vsf(rne_v(clamp_v(Q6_Vsf_vmpy_VsfVsf(dd[n], vc2), lim_lo, lim_hi), magic));
    if (zb) pass2_pieces<true>(src, dst, tb, RD, np, zp); else pass2_pieces<false>(src, dst, tb, RD, np, zp);
    return;
  }
  HVX_Vector dz[4];
#pragma unroll
  for (int n = 0; n < 4; ++n) {
    const HVX_Vector w = Q6_Vw_equals_Vsf(rne_v(clamp_v(Q6_Vsf_vmpy_VsfVsf(dd[n], vc2), lo32k, hi32k), magic));
    dz[n] = Q6_V_vor_VV(Q6_V_vand_VV(w, mask16), zphi);
  }
#if defined(DR16M_STOP) && DR16M_STOP == 3
  o[1] = static_cast<float>(lane0(Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(dz[0], dz[1]), Q6_Vw_vadd_VwVw(dz[2], dz[3]))) + tb.R2[5]); return;
#endif
  // --- 2차 통과
  if (zb) pass2_main<true>(src, dst, tb.R2, RD, dz); else pass2_main<false>(src, dst, tb.R2, RD, dz);
}

// State commit without computing the discarded attention output.
static inline void step_state_hvx(const uint16_t* in, uint16_t* out, int zp, float s, float inv_s, float Lm,
                            const float* k, const float* v, float decay, float beta, int head,
                            Tables& tb, Stats* st = nullptr) {
  const HVX_Vector* kv = reinterpret_cast<const HVX_Vector*>(k);
  const HVX_Vector* vv = reinterpret_cast<const HVX_Vector*>(v);
  const HVX_Vector magic = splat_f(12582912.0f), lo32k = splat_f(-32767.f), hi32k = splat_f(32767.f);
  const HVX_Vector mask16 = Q6_V_vsplat_R(0xFFFF), absmask = Q6_V_vsplat_R(0x7FFFFFFF);
  const int A = decay_q15(decay, v, head);      // 스칼라, HVX 준비와 겹쳐 실행
  HVX_Vector km = Q6_V_vzero(), kl = km;
#pragma unroll
  for (int n = 0; n < 4; ++n) {
    const HVX_Vector ak = Q6_V_vand_VV(kv[n], absmask);
    km = Q6_Vw_vmax_VwVw(km, ak);
    if (n == 0) { kl = ak; }
    else { kl = Q6_Vsf_vadd_VsfVsf(kl, ak); }
  }
#pragma unroll
  for (int sh = 64; sh >= 4; sh >>= 1) {
    km = Q6_Vw_vmax_VwVw(km, Q6_V_vror_VR(km, sh));
    kl = Q6_Vsf_vadd_VsfVsf(kl, Q6_V_vror_VR(kl, sh));
  }
  const float kappa = as_f(lane0(km));
  float sck;
  const int eK = pick_e(kappa, as_f(lane0(kl)), Lm, &sck);
  const int rK = eK;
  {
    const HVX_Vector vsk = splat_f(sck);
    HVX_Vector sK = Q6_V_vzero();
    HVX_Vector* RKv = reinterpret_cast<HVX_Vector*>(tb.RK);
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      const HVX_Vector wk = Q6_Vw_equals_Vsf(rne_v(clamp_v(Q6_Vsf_vmpy_VsfVsf(kv[n], vsk), lo32k, hi32k), magic));
      sK = Q6_Vw_vadd_VwVw(sK, wk);
      RKv[n] = Q6_V_vor_VV(Q6_V_vand_VV(wk, mask16), Q6_Vw_vasl_VwR(wk, 16));
    }
#pragma unroll
    for (int sh = 64; sh >= 4; sh >>= 1) {
      sK = Q6_Vw_vadd_VwVw(sK, Q6_V_vror_VR(sK, sh));
    }
    tb.sumK = lane0(sK);
  }
  const uint32_t zK = static_cast<uint32_t>(tb.sumK) * static_cast<uint32_t>(zp - 32768);
  const float ad = static_cast<float>(A) * 0x1p-15f;
  const float as = ad * s;
  const float ck = as * pow2f(-eK);
  const HVX_Vector sign = Q6_V_vsplat_R(0x80008000);
  HVX_VectorPair aK0 = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero()), aK1 = aK0;
  const HVX_Vector* src = reinterpret_cast<const HVX_Vector*>(in);
  const int32_t* RK = tb.RK;
#pragma clang loop unroll_count(DR16M_P1_UNROLL)
  for (int i = 0; i < DK; ++i) {
    const HVX_Vector x0 = Q6_V_vxor_VV(src[2 * i], sign), x1 = Q6_V_vxor_VV(src[2 * i + 1], sign);
    const int32_t rk = RK[i];
    aK0 = Q6_Ww_vmpyacc_WwVhRh(aK0, x0, rk); aK1 = Q6_Ww_vmpyacc_WwVhRh(aK1, x1, rk);
  }
  const HVX_Vector vck = splat_f(ck), vbeta = splat_f(beta);
  const HVX_Vector vzK = Q6_V_vsplat_R(static_cast<int32_t>(zK));
  const HVX_Vector aK[4] = {Q6_V_lo_W(aK0), Q6_V_hi_W(aK0), Q6_V_lo_W(aK1), Q6_V_hi_W(aK1)};
  HVX_Vector dd[4], mmax = Q6_V_vzero();
#pragma unroll
  for (int c = 0; c < 2; ++c) {
    const HVX_VectorPair vd = Q6_W_vdeal_VVR(vv[2 * c + 1], vv[2 * c], -4);
    const HVX_Vector vx[2] = {Q6_V_lo_W(vd), Q6_V_hi_W(vd)};
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int n = 2 * c + h;
      HVX_Vector t = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_equals_Vw(Q6_Vw_vsub_VwVw(aK[n], vzK)), vck);
      t = Q6_Vsf_vsub_VsfVsf(vx[h], t);
      dd[n] = Q6_Vsf_vmpy_VsfVsf(t, vbeta);
      mmax = Q6_Vw_vmax_VwVw(mmax, Q6_V_vand_VV(dd[n], absmask));
    }
  }
  const float M = as_f(lane0(reduce_max_w(mmax)));
  int np = 1;
  const int a = choose_a(kappa, M * inv_s, &np);
  if (st) { st->heads++; st->split += np >= 2; st->hard += np >= 4; st->rK += rK; if (rK > st->rKmax) st->rKmax = rK; }
  const HVX_Vector vc2 = splat_f(inv_s * pow2f(15 - a)), vka = splat_f(pow2f(a));
  const HVX_Vector cqhi = Q6_V_vsplat_R(static_cast<int32_t>(static_cast<uint32_t>(32768 - A) << 16));
  const HVX_Vector zphi = zp_hi(zp);
  const bool zb = zp > 32767;
#pragma unroll
  for (int n = 0; n < 4; ++n) {
    const HVX_Vector w = Q6_Vw_equals_Vsf(rne_v(clamp_v(Q6_Vsf_vmpy_VsfVsf(kv[n], vka), lo32k, hi32k), magic));
    reinterpret_cast<HVX_Vector*>(tb.R2)[n] = Q6_V_vor_VV(Q6_V_vand_VV(w, mask16), cqhi);
  }
  const int32_t RD = static_cast<int32_t>(static_cast<uint32_t>(A) | (static_cast<uint32_t>(A) << 16));
  HVX_Vector* dst = reinterpret_cast<HVX_Vector*>(out);
  if (__builtin_expect(np > 1, 0)) {
    const HVX_Vector lim_lo = splat_f(-32767.f * np), lim_hi = splat_f(32767.f * np);
#pragma unroll
    for (int n = 0; n < 4; ++n)
      reinterpret_cast<HVX_Vector*>(tb.W)[n] =
          Q6_Vw_equals_Vsf(rne_v(clamp_v(Q6_Vsf_vmpy_VsfVsf(dd[n], vc2), lim_lo, lim_hi), magic));
    if (zb) pass2_pieces<true>(src, dst, tb, RD, np, zp); else pass2_pieces<false>(src, dst, tb, RD, np, zp);
    return;
  }
  HVX_Vector dz[4];
#pragma unroll
  for (int n = 0; n < 4; ++n) {
    const HVX_Vector w = Q6_Vw_equals_Vsf(rne_v(clamp_v(Q6_Vsf_vmpy_VsfVsf(dd[n], vc2), lo32k, hi32k), magic));
    dz[n] = Q6_V_vor_VV(Q6_V_vand_VV(w, mask16), zphi);
  }
  if (zb) pass2_main<true>(src, dst, tb.R2, RD, dz); else pass2_main<false>(src, dst, tb.R2, RD, dz);
}
#endif
} // namespace dr16m
#endif
