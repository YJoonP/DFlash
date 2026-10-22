// Qwen3.5 Gated DeltaNet 델타 규칙 재귀의 순수 계산부.
//
//   t 스텝마다:  S ← S·decay ;  kv = Sᵀk ;  d = (v − kv)·beta ;  S ← S + k⊗d ;  o = Sᵀq
//
// 구조를 두 겹으로 나눈다.
//   (1) 텐서 입출력: dtype·레이아웃은 호출자 책임이다. QHPI 경로는 연속 벡터의
//       입출력을 HVX로 변환한다. 원소 수가 작아도 scalar Q/DQ 비용은 클 수 있다.
//   (2) 상태 이중 루프: dk×dv 개 원소를 두 번 훑는 무거운 부분. float 스크래치 위에서
//       돌며, Hexagon 에서는 HVX 로 32 개씩 처리한다.
// 재귀 누산은 항상 float 이다 — state 가 양자화돼도 T 스텝 동안 재양자화하지 않는 것이
// 이 커널의 존재 이유다.
#ifndef DELTA_RULE_CORE_H
#define DELTA_RULE_CORE_H

#include <cstddef>
#include <cstdint>

#if defined(__hexagon__) && defined(__HVX__)
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#define DELTA_RULE_HAVE_HVX 1
#endif

namespace deltarule {

struct Dims {
  int B, T, H, dk, dv;
};

// 스텝 하나의 상태 갱신 (스칼라 기준 구현).
//   S: (dk, dv) float, 제자리 갱신.  kt/qt: dk 개.  vt: dv 개.  ot: dv 개 출력.
inline void step_scalar(int dk, int dv, float* S, const float* qt, const float* kt,
                        const float* vt, float dec, float bet, float* ot, float* kv,
                        float* dl, const float* state_src = nullptr) {
  if (!state_src) state_src = S;
  for (int j = 0; j < dv; ++j) kv[j] = 0.f;
  // 1차 통과: S *= decay,  kv = Sᵀk
  for (int i = 0; i < dk; ++i) {
    float* Si = S + static_cast<size_t>(i) * dv;
    const float ki = kt[i];
    for (int j = 0; j < dv; ++j) { const float s = state_src[static_cast<size_t>(i) * dv + j] * dec; Si[j] = s; kv[j] += s * ki; }
  }
  for (int j = 0; j < dv; ++j) { dl[j] = (vt[j] - kv[j]) * bet; ot[j] = 0.f; }
  // 2차 통과: S += k⊗d,  o = Sᵀq
  for (int i = 0; i < dk; ++i) {
    float* Si = S + static_cast<size_t>(i) * dv;
    const float ki = kt[i], qi = qt[i];
    for (int j = 0; j < dv; ++j) { const float s = Si[j] + ki * dl[j]; Si[j] = s; ot[j] += s * qi; }
  }
}

#ifdef DELTA_RULE_HAVE_HVX
// HVX 판. dv 가 32 의 배수이고 S 가 128 바이트 정렬일 때만 쓴다.
// 누산은 qf32 (HVX 내부 부동소수 형식) 로 해서 매 항마다 정규화하지 않는다 —
// 128 항 내적을 스칼라 float 로 누산하는 것보다 오차도 작다.
inline HVX_Vector vsplat_f(float v) {
  union { float f; int32_t i; } u; u.f = v;
  return Q6_V_vsplat_R(u.i);
}

// NV 를 템플릿 인자로 받아 컴파일 타임 상수로 만든다. 런타임 값이면 누산기 배열이
// 벡터 레지스터에 못 남고 스택으로 흘러(allocframe 3072B), (i,n) 마다 vmem 이 2~3 개
// 추가로 발생한다 (2026-09-10 디스어셈블 확인).
template <int NV, bool StoreDecayed = true>
inline void step_hvx_n(int dk, int dv, float* S, const float* qt, const float* kt,
                       const float* vt, float dec, float bet, float* ot,
                       const float* state_src = nullptr) {
  if (!state_src) state_src = S;
  HVX_Vector kvacc[NV], otacc[NV], dlv[NV];
  const HVX_Vector vdec = vsplat_f(dec);
  const HVX_Vector vbet = vsplat_f(bet);

  for (int n = 0; n < NV; ++n) kvacc[n] = Q6_V_vzero();
  // 1차 통과
  for (int i = 0; i < dk; ++i) {
    HVX_Vector* Si = reinterpret_cast<HVX_Vector*>(S + static_cast<size_t>(i) * dv);
    const HVX_Vector* source = reinterpret_cast<const HVX_Vector*>(state_src + static_cast<size_t>(i) * dv);
    const HVX_Vector vki = vsplat_f(kt[i]);
    for (int n = 0; n < NV; ++n) {
      const HVX_Vector s = Q6_Vsf_vmpy_VsfVsf(source[n], vdec);
      if constexpr (StoreDecayed) Si[n] = s;
      kvacc[n] = Q6_Vqf32_vadd_Vqf32Vqf32(kvacc[n], Q6_Vqf32_vmpy_VsfVsf(s, vki));
    }
  }
  // d = (v − kv)·beta
  const HVX_Vector* vv = reinterpret_cast<const HVX_Vector*>(vt);
  for (int n = 0; n < NV; ++n) {
    const HVX_Vector kvn = Q6_Vsf_equals_Vqf32(kvacc[n]);
    dlv[n] = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vsub_VsfVsf(vv[n], kvn), vbet);
    otacc[n] = Q6_V_vzero();
  }
  // 2차 통과
  for (int i = 0; i < dk; ++i) {
    HVX_Vector* Si = reinterpret_cast<HVX_Vector*>(S + static_cast<size_t>(i) * dv);
    const HVX_Vector* source = reinterpret_cast<const HVX_Vector*>(state_src + static_cast<size_t>(i) * dv);
    const HVX_Vector vki = vsplat_f(kt[i]);
    const HVX_Vector vqi = vsplat_f(qt[i]);
    for (int n = 0; n < NV; ++n) {
      // T=1 can read the original input twice and write only the final state.
      // Repeat the identical rounded multiply; do not move decay across a sum.
      const HVX_Vector decayed = StoreDecayed ? Si[n] : Q6_Vsf_vmpy_VsfVsf(source[n], vdec);
      const HVX_Vector s = Q6_Vsf_vadd_VsfVsf(decayed, Q6_Vsf_vmpy_VsfVsf(vki, dlv[n]));
      Si[n] = s;
      otacc[n] = Q6_Vqf32_vadd_Vqf32Vqf32(otacc[n], Q6_Vqf32_vmpy_VsfVsf(s, vqi));
    }
  }
  HVX_Vector* vo = reinterpret_cast<HVX_Vector*>(ot);
  for (int n = 0; n < NV; ++n) vo[n] = Q6_Vsf_equals_Vqf32(otacc[n]);
}

// dv 별 디스패치. 128 이 이 모델의 값이라 그 경로만 특수화하면 된다.
template <bool StoreDecayed = true>
inline void step_hvx(int dk, int dv, float* S, const float* qt, const float* kt,
                     const float* vt, float dec, float bet, float* ot,
                     const float* state_src = nullptr) {
  if (!state_src) state_src = S;
  // Columns are independent. Keep at most four live vector accumulators per
  // array; larger specializations spill heavily (NV=8 also failed v79 parity).
  // dv remains the full row stride while pointers select a column tile.
  int offset = 0;
  for (; offset + 128 <= dv; offset += 128)
    step_hvx_n<4, StoreDecayed>(dk, dv, S + offset, qt, kt, vt + offset, dec, bet, ot + offset, state_src + offset);
  S += offset;
  vt += offset;
  ot += offset;
  state_src += offset;
  switch ((dv - offset) / 32) {
    case 1: step_hvx_n<1, StoreDecayed>(dk, dv, S, qt, kt, vt, dec, bet, ot, state_src); return;
    case 2: step_hvx_n<2, StoreDecayed>(dk, dv, S, qt, kt, vt, dec, bet, ot, state_src); return;
    case 3: step_hvx_n<3, StoreDecayed>(dk, dv, S, qt, kt, vt, dec, bet, ot, state_src); return;
  }
}
#endif

// GetQK(b,t,h,i) GetV(b,t,h,j) GetBD(b,t,h) GetS(b,h,i,j) PutOut(b,t,h,j,val) PutS(b,h,i,j,val)
// 인덱스는 전부 논리 좌표다. 물리 주소 계산은 호출자 책임.
// scratch 는 dk*dv float 이상, 128 바이트 정렬이어야 HVX 경로를 탄다.
// LoadState(b,h,dst) / StoreState(b,h,src): 헤드 하나의 상태(dk*dv)를 통째로 옮긴다.
// 원소 단위 접근자로 하면 dtype 분기가 원소마다 실행돼 4바이트 옮기는 데 8.8 사이클이 든다
// (2026-09-10 실측: 이 복사만으로 decode 초과 시간의 96%).
template <typename GetQ, typename GetK, typename GetV, typename GetBeta,
          typename GetDecay, typename LoadState, typename PutOut, typename StoreState>
void run(const Dims& d, GetQ getq, GetK getk, GetV getv, GetBeta getbeta,
         GetDecay getdecay, LoadState load_state, PutOut putout, StoreState store_state,
         float* scratch) {
  const int dk = d.dk, dv = d.dv;
  // 스텝당 작업 벡터. 128 바이트 정렬해 HVX 로드/스토어가 정렬 경로를 타게 한다.
  alignas(128) float qt[256], kt[256], vt[256], ot[256], kv[256], dl[256];

#ifdef DELTA_RULE_HAVE_HVX
  const bool use_hvx =
      (dv > 0) && (dv % 32 == 0) && (dv <= 256) &&
      ((reinterpret_cast<uintptr_t>(scratch) & 127u) == 0) &&
      ((static_cast<size_t>(dv) * sizeof(float)) % 128u == 0);
#else
  const bool use_hvx = false;
#endif

  for (int b = 0; b < d.B; ++b) {
    for (int h = 0; h < d.H; ++h) {
      load_state(b, h, scratch);

      for (int t = 0; t < d.T; ++t) {
        const float dec = getdecay(b, t, h);
        const float bet = getbeta(b, t, h);
        for (int i = 0; i < dk; ++i) { qt[i] = getq(b, t, h, i); kt[i] = getk(b, t, h, i); }
        for (int j = 0; j < dv; ++j) vt[j] = getv(b, t, h, j);

#ifdef DELTA_RULE_HAVE_HVX
        if (use_hvx) step_hvx(dk, dv, scratch, qt, kt, vt, dec, bet, ot);
        else step_scalar(dk, dv, scratch, qt, kt, vt, dec, bet, ot, kv, dl);
#else
        (void)use_hvx;
        step_scalar(dk, dv, scratch, qt, kt, vt, dec, bet, ot, kv, dl);
#endif
        for (int j = 0; j < dv; ++j) putout(b, t, h, j, ot[j]);
      }

      store_state(b, h, scratch);
    }
  }
}

} // namespace deltarule
#endif
