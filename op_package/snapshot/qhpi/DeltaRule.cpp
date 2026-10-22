// QHPI DeltaRule. Each invocation owns disjoint heads and output workspace.
#include "HTP/core/qhpi.h"
#include "../src/ops/delta_rule_core.h"
#include "../src/ops/delta_rule_io.h"
#include "../src/ops/delta_rule_quant_step.h"
#include <cmath>
#include <cstring>
#include <cstdio>
#include "../src/ops/replay_wire.h"
#ifdef __hexagon__
extern "C" void qnndsp_log(int, const char*, ...);
#define SNAPSHOT_LOG(...) qnndsp_log(0, __VA_ARGS__)
#else
#define SNAPSHOT_LOG(...) std::printf(__VA_ARGS__)
#endif

#ifndef DELTA_THREADS
#define DELTA_THREADS 0
#endif
#ifndef DELTA_DECODE_DIRECT_WRITE
#define DELTA_DECODE_DIRECT_WRITE 0
#endif
// DELTA_ROW_REQUANT=1: a uint16-state call with T rows rounds the state to uint16 after
// every row instead of once at the end, so speculative verification (T rows) and one-token
// decoding (T calls) see identical states and produce identical tokens.
#ifndef DELTA_ROW_REQUANT
#define DELTA_ROW_REQUANT 0
#endif
// DELTA_INT16=1: uint16-state T=1 steps run the fixed-point dr16m kernel (no uint16<->float
// state conversion) when the state encoding allows it; otherwise the float path runs.
// DELTA_INT16=2: same, but a T=1 call that cannot use the integer path fails (verification).
// DELTA_INT16=3: same, but only an input/output state encoding mismatch fails (verification).
// DELTA_INT16=4: TIMING PROXY ONLY - runs the integer path even when the output state encoding
//                differs from the input one (writes input-grid codes; model outputs are wrong).
#ifndef DELTA_INT16
#define DELTA_INT16 0
#endif
// DELTA_INT16_PREFILL=1 (with DELTA_INT16): the masked prefill op also runs dr16m, one step per
// valid row, so its state equals sequential T=1 integer steps on the same rows bit for bit
// (the state is rounded to uint16 after every row instead of once per call).
// DELTA_INT16_PREFILL=2: same, but a masked call that cannot use the integer path fails (verification).
#ifndef DELTA_INT16_PREFILL
#define DELTA_INT16_PREFILL 0
#endif
#if DELTA_INT16_PREFILL && !DELTA_INT16
#error "DELTA_INT16_PREFILL needs DELTA_INT16"
#endif
#if DELTA_INT16
#include "../src/ops/dr16m.h"
#endif

namespace {
inline uint64_t cycles() {
#if defined(DELTA_DIAGNOSTICS) && defined(__hexagon__)
  uint64_t value;
  asm volatile ("%0 = C15:14" : "=r"(value));
  return value;
#else
  return 0;
#endif
}

template <bool Quant> struct Access {
  void* data;
  float scale, inv, offset;
  explicit Access(const QHPI_Tensor* t) : data(qhpi_tensor_raw_data(t)) {
    const auto p = qhpi_tensor_quant_parameters(t);
    scale = Quant ? p.stepsize : 1.f;
    inv = 1.f / scale;
    offset = Quant ? static_cast<float>(p.zero_offset) : 0.f;
  }
  float get(size_t i) const {
    if constexpr (Quant) return (static_cast<const uint16_t*>(data)[i] - offset) * scale;
    else return static_cast<const float*>(data)[i];
  }
  void load(size_t base, float* dst, size_t n) const {
    if constexpr (!Quant) std::memcpy(dst, static_cast<const float*>(data) + base, n * sizeof(float));
    else deltarule::load_u16(static_cast<const uint16_t*>(data) + base, dst, n, static_cast<int>(offset), scale);
  }
  void store(size_t base, const float* src, size_t n) const {
    if constexpr (!Quant) std::memcpy(static_cast<float*>(data) + base, src, n * sizeof(float));
    else deltarule::store_u16(src, static_cast<uint16_t*>(data) + base, n, offset, inv);
  }
};

alignas(128) const float kZeroRow[128] = {};

bool shape(const QHPI_Tensor* t, unsigned b, unsigned c, unsigned h, unsigned w) {
  const auto s = qhpi_tensor_shape(t);
  return s.rank == 4 && s.dims[0] == b && s.dims[1] == c && s.dims[2] == h && s.dims[3] == w;
}

// MASK: 7th input valid (B,T,1,1) int32. A row with valid == 0 (prompt padding) leaves the
// state untouched and writes a zero output row, so a padded call equals a call on the valid rows.
template <bool QA, bool QS = false, bool MASK = false>
uint32_t execute(QHPI_RuntimeHandle* handle, uint32_t no, QHPI_Tensor** outputs,
                 uint32_t ni, const QHPI_Tensor* const* inputs) {
  static_assert(!MASK || QS, "the masked op is uint16-state only");
  if (no != (QS ? 3u : 2u) || ni != (MASK ? 7u : 6u)) return 1;
  const auto qshape = qhpi_tensor_shape(inputs[0]);
  const auto vshape = qhpi_tensor_shape(inputs[2]);
  if (qshape.rank != 4 || vshape.rank != 4) return 1;
  const unsigned B=qshape.dims[0], T=qshape.dims[1], H=qshape.dims[2];
  const unsigned dk=qshape.dims[3], dv=vshape.dims[3];
  if (!B || !T || !H || !dk || !dv || dk > 128 || dv > 128) return 1;
  if (!shape(inputs[1], B,T,H,dk) || !shape(inputs[2], B,T,H,dv) ||
      !shape(inputs[3], B,T,H,1) || !shape(inputs[4], B,T,H,1) ||
      !shape(inputs[5], B,H,dk,dv) || !shape(outputs[0], B,T,H,dv) ||
      !shape(outputs[1], B,H,dk,dv)) return 1;
  if constexpr (QS) {
    if (!shape(outputs[2], 1,1,dk,dv) || !qhpi_tensor_raw_data(outputs[2])) return 1;
  }
  const int32_t* valid = nullptr;
  if constexpr (MASK) {
    if (!shape(inputs[6], B,T,1,1)) return 1;
    valid = static_cast<const int32_t*>(qhpi_tensor_raw_data(inputs[6]));
    if (!valid) return 1;
  }
  const unsigned slices = qhpi_num_slices(handle), slice = qhpi_slice_number(handle);
  if (!slices || slice >= slices) return 1;
  // Workspace belongs to this graph op, not a global or a thread stack.
  // Independent head groups may run concurrently, but this op is not self-slicing.
  if constexpr (QS) { if (slices != 1) return 1; }
  Access<QA> q(inputs[0]), k(inputs[1]), v(inputs[2]), beta(inputs[3]), decay(inputs[4]), out(outputs[0]);
  Access<QS> state_in(inputs[5]), state_out(outputs[1]);
  if (!q.data || !k.data || !v.data || !beta.data || !decay.data || !out.data || !state_in.data || !state_out.data) return 1;
  const size_t state_size = static_cast<size_t>(dk) * dv;
  alignas(128) float qt[128], kt[128], vt[128], ot[128], kv[128], dl[128];
  uint64_t load_cycles = 0, step_cycles = 0, store_cycles = 0;
#if DELTA_INT16 && defined(DELTA_RULE_HAVE_HVX) && defined(DR16M_HAVE_HVX)
  // dr16m preconditions: dk = dv = 128 and one shared state encoding (any zero point).
  bool use_int16 = false;
#if DELTA_INT16_PREFILL
  bool int16_rows = false;
#endif
  float int16_lm = 0.f;
  if constexpr (QS) {
    const int zp = static_cast<int>(state_in.offset);
    const bool same_encoding = DELTA_INT16 == 4 ||
        (state_in.offset == state_out.offset && state_in.scale == state_out.scale);
    use_int16 = T == 1 && dk == 128 && dv == 128 && same_encoding && zp >= 0 && zp <= 65535;
#if DELTA_INT16_PREFILL
    if constexpr (MASK) int16_rows = dk == 128 && dv == 128 && same_encoding && zp >= 0 && zp <= 65535;
#if DELTA_INT16_PREFILL == 2
    if constexpr (MASK) { if (!int16_rows) return 1; }
#endif
    if (int16_rows) int16_lm = dr16m::lm_of(zp);
#endif
    if (use_int16) int16_lm = dr16m::lm_of(zp);
#if DELTA_INT16 == 2
    if (T == 1 && !use_int16) return 1;
#elif DELTA_INT16 == 3
    if (T == 1 && (state_in.offset != state_out.offset || state_in.scale != state_out.scale)) return 1;
#endif
  }
#endif
  // Each state/output element has exactly one owner. No locks or shared head state.
  for (unsigned bh = slice; bh < B * H; bh += slices) {
    const unsigned b = bh / H, h = bh % H;
    auto idx = [&](int t) { return (static_cast<size_t>(b) * T + t) * H + h; };
#ifdef DELTA_RULE_HAVE_HVX
    if constexpr (QS && !MASK) {
#if DELTA_INT16 && defined(DR16M_HAVE_HVX)
      if (use_int16) {
        const size_t pos = idx(0);
        q.load(pos * dk, qt, dk); k.load(pos * dk, kt, dk); v.load(pos * dv, vt, dv);
        alignas(128) dr16m::Tables tb;
        dr16m::step_hvx(static_cast<const uint16_t*>(state_in.data) + bh * state_size,
                        static_cast<uint16_t*>(state_out.data) + bh * state_size,
                        static_cast<int>(state_in.offset), state_in.scale, state_in.inv, int16_lm,
                        qt, kt, vt, decay.get(pos), beta.get(pos), static_cast<int>(bh), ot, tb);
        out.store(pos * dv, ot, dv);
        continue;
      }
#endif
      if (T == 1 && (dv == 64 || dv == 128)) {
        const size_t pos = idx(0);
        q.load(pos * dk, qt, dk); k.load(pos * dk, kt, dk); v.load(pos * dv, vt, dv);
        deltarule::step_u16_hvx(dk, dv,
            static_cast<const uint16_t*>(state_in.data) + bh * state_size,
            static_cast<uint16_t*>(state_out.data) + bh * state_size,
            static_cast<int>(state_in.offset), state_in.scale, state_out.offset, state_out.inv,
            qt, kt, vt, decay.get(pos), beta.get(pos), ot,
            bh + 1 == B * H ? static_cast<float*>(qhpi_tensor_raw_data(outputs[2])) : nullptr);
        out.store(pos * dv, ot, dv);
#ifdef DELTA_DIAGNOSTICS
        // 생산 경로(T=1, dv=128)는 아래 진단에 도달하기 전에 continue 하므로
        // 여기서 따로 찍는다. placement: 0=DDR_ONLY 1=TCM_ONLY 2=DDR_OR_TCM.
        if (bh == 0)
          std::printf("DELTA_FASTPATH dk=%d dv=%d in_place=%d out_place=%d ws_place=%d "
                      "in_ptr=%p out_ptr=%p in_align=%u\n",
              dk, dv, int(qhpi_tensor_placement(inputs[5])),
              int(qhpi_tensor_placement(outputs[1])),
              no > 2 ? int(qhpi_tensor_placement(outputs[2])) : -1,
              state_in.data, state_out.data,
              unsigned(reinterpret_cast<uintptr_t>(state_in.data) & 127u));
#endif
        // Only the last head is observable as the workspace output.
        continue;
      }
    }
#endif
#if DELTA_INT16_PREFILL && defined(DELTA_RULE_HAVE_HVX) && defined(DR16M_HAVE_HVX)
    if constexpr (MASK) {
      if (int16_rows) {
        // One dr16m step per valid row, chained on the uint16 output state (dr16m allows in == out).
        // Same head index as the T=1 call, so the decay dither matches as well.
        const uint16_t* src = static_cast<const uint16_t*>(state_in.data) + bh * state_size;
        uint16_t* dst = static_cast<uint16_t*>(state_out.data) + bh * state_size;
        alignas(128) dr16m::Tables tb;
        for (unsigned t = 0; t < T; ++t) {
          const size_t pos = idx(t);
          if (!valid[b * T + t]) { out.store(pos * dv, kZeroRow, dv); continue; }
          q.load(pos * dk, qt, dk); k.load(pos * dk, kt, dk); v.load(pos * dv, vt, dv);
          dr16m::step_hvx(src, dst, static_cast<int>(state_in.offset), state_in.scale, state_in.inv, int16_lm,
                          qt, kt, vt, decay.get(pos), beta.get(pos), static_cast<int>(bh), ot, tb);
          src = dst;
          out.store(pos * dv, ot, dv);
        }
        if (src != dst) std::memcpy(dst, src, state_size * sizeof(uint16_t));  // no valid row
        continue;
      }
    }
#endif
    // The float output is already independently owned scratch. This also avoids
    // a sync block (SDK 2.49 v79 cannot deserialize the generated q::sync_zero).
    float* scratch;
    const float* source;
    if constexpr (QS) {
      scratch = static_cast<float*>(qhpi_tensor_raw_data(outputs[2]));
      state_in.load(bh * state_size, scratch, state_size);
      source = scratch;
    } else {
      scratch = static_cast<float*>(state_out.data) + bh * state_size;
      source = static_cast<const float*>(state_in.data) + bh * state_size;
    }
#ifdef DELTA_DIAGNOSTICS
    if (bh == 0) std::printf("DELTA_QHPI T=%u H=%u slices=%u src=%p dst=%p align=%u placement=%d\n",
        T, H, slices, state_in.data, scratch, unsigned(reinterpret_cast<uintptr_t>(scratch) & 127u),
        int(qhpi_tensor_placement(outputs[1])));
#endif
    unsigned end = T;  // rows after the last valid one never touch the state
    if constexpr (MASK) {
      end = 0;
      for (unsigned t = 0; t < T; ++t) if (valid[b * T + t]) end = t + 1;
    }
    for (unsigned t = 0; t < T; ++t) {
      if constexpr (MASK) {
        if (!valid[b * T + t]) { out.store(idx(t) * dv, kZeroRow, dv); continue; }
      }
      auto stamp = cycles();
      const size_t pos = idx(t);
      q.load(pos * dk, qt, dk);
      k.load(pos * dk, kt, dk);
      v.load(pos * dv, vt, dv);
      const float dec = decay.get(pos), bet = beta.get(pos);
      load_cycles += cycles() - stamp;
      stamp = cycles();
#ifdef DELTA_RULE_HAVE_HVX
      if (dv % 32 == 0 && ((reinterpret_cast<uintptr_t>(scratch) | reinterpret_cast<uintptr_t>(source)) & 127u) == 0) {
        // Experimental: fewer stores did not improve S25 T=1 wall time.
        // Keep the measured original kernel as the default.
        if (DELTA_DECODE_DIRECT_WRITE && T == 1)
          deltarule::step_hvx<false>(dk, dv, scratch, qt, kt, vt, dec, bet, ot, source);
        else
          deltarule::step_hvx(dk, dv, scratch, qt, kt, vt, dec, bet, ot, source);
      }
      else
#endif
        deltarule::step_scalar(dk, dv, scratch, qt, kt, vt, dec, bet, ot, kv, dl, source);
      source = scratch;
      if constexpr (QS) {
        if (DELTA_ROW_REQUANT && t + 1 < end)
          deltarule::roundtrip_u16(scratch, state_size, state_out.offset, state_out.inv,
                                   static_cast<int>(state_in.offset), state_in.scale);
      }
      step_cycles += cycles() - stamp;
      stamp = cycles();
      out.store(pos * dv, ot, dv);
      store_cycles += cycles() - stamp;
    }
    if constexpr (QS) state_out.store(bh * state_size, scratch, state_size);
  }
#ifdef DELTA_DIAGNOSTICS
  std::printf("DELTA_CYCLES load=%llu step=%llu store=%llu\n",
      (unsigned long long)load_cycles, (unsigned long long)step_cycles, (unsigned long long)store_cycles);
#endif
  return 0;
}

// Decode specialization: Q/K retain their native (unique) head count while
// V/beta/decay/state retain value-head count. It has two outputs, so the T=1
// path does not materialize the legacy float workspace just to satisfy an
// unused third output. Q/K head h is shared by value heads h*repeat.. .
uint32_t execute_decode16(QHPI_RuntimeHandle* handle, uint32_t no, QHPI_Tensor** outputs,
                          uint32_t ni, const QHPI_Tensor* const* inputs) {
  if (no != 2 || ni != 6) return 1;
  const auto qshape = qhpi_tensor_shape(inputs[0]);
  const auto vshape = qhpi_tensor_shape(inputs[2]);
  if (qshape.rank != 4 || vshape.rank != 4) return 1;
  const unsigned B=qshape.dims[0], T=qshape.dims[1], Hq=qshape.dims[2], dk=qshape.dims[3];
  const unsigned Hv=vshape.dims[2], dv=vshape.dims[3];
  if (!B || T != 1 || !Hq || !Hv || Hv % Hq || !dk || !dv || dk > 128 || dv > 128) return 1;
  if (!shape(inputs[1], B,T,Hq,dk) || !shape(inputs[2], B,T,Hv,dv) ||
      !shape(inputs[3], B,T,Hv,1) || !shape(inputs[4], B,T,Hv,1) ||
      !shape(inputs[5], B,Hv,dk,dv) || !shape(outputs[0], B,T,Hv,dv) ||
      !shape(outputs[1], B,Hv,dk,dv)) return 1;
  const unsigned slices = qhpi_num_slices(handle), slice = qhpi_slice_number(handle);
  if (slices != 1 || slice != 0) return 1;
  Access<true> q(inputs[0]), k(inputs[1]), v(inputs[2]), beta(inputs[3]), decay(inputs[4]), out(outputs[0]);
  Access<true> state_in(inputs[5]), state_out(outputs[1]);
  if (!q.data || !k.data || !v.data || !beta.data || !decay.data || !out.data || !state_in.data || !state_out.data) return 1;
#ifdef DELTA_RULE_HAVE_HVX
  if (dv != 64 && dv != 128) return 1;
  const size_t state_size = static_cast<size_t>(dk) * dv;
  const unsigned repeat = Hv / Hq;
  alignas(128) float qt[128], kt[128], vt[128], ot[128];
  size_t cached_qpos = static_cast<size_t>(-1);
  for (unsigned bh = slice; bh < B * Hv; bh += slices) {
    const unsigned b = bh / Hv, h = bh % Hv, qh = h / repeat;
    const size_t qpos = (static_cast<size_t>(b) * Hq + qh);
    const size_t vpos = (static_cast<size_t>(b) * Hv + h);
    if (qpos != cached_qpos) {
      q.load(qpos * dk, qt, dk); k.load(qpos * dk, kt, dk);
      cached_qpos = qpos;
    }
    v.load(vpos * dv, vt, dv);
    // Keep the validated 64-column path independent of legacy tile experiments.
    for (unsigned col = 0; col < dv; col += 64)
      deltarule::step_u16_hvx_n<2>(dk, dv,
        static_cast<const uint16_t*>(state_in.data) + bh * state_size + col,
        static_cast<uint16_t*>(state_out.data) + bh * state_size + col,
        static_cast<int>(state_in.offset), state_in.scale, state_out.offset, state_out.inv,
        qt, kt, vt + col, decay.get(vpos), beta.get(vpos), ot + col, nullptr);
    out.store(vpos * dv, ot, dv);
  }
  return 0;
#else
  // This op intentionally has no scalar full-state fallback: it is a v79 HVX
  // decode contract and must fail rather than allocating an unbounded stack
  // workspace on an unsupported target.
  return 1;
#endif
}

QHPI_Tensor_Signature_v1 signature(uint32_t type) {
  QHPI_Tensor_Signature_v1 s{};
  s.element_type = type;
  s.layout = QHPI_LAYOUT_FLAT_4;
  s.storage = QHPI_STORAGE_DIRECT;
  // Request VTCM for the repeatedly accessed recurrent state. Placement alone
  // does not remove boundary conversions or make scalar conversions efficient.
  s.mem_placement = QHPI_MEM_LOC_TCM_ONLY;
  return s;
}

inline void snapshot_copy(void* destination, const void* source, size_t bytes) {
#ifdef DR16M_HAVE_HVX
  auto* dst=static_cast<unsigned char*>(destination);
  const auto* src=static_cast<const unsigned char*>(source);
  size_t i=0;
  for (; i+128<=bytes; i+=128)
    *reinterpret_cast<HVX_UVector*>(dst+i)=*reinterpret_cast<const HVX_UVector*>(src+i);
  if (i<bytes) std::memcpy(dst+i,src+i,bytes-i);
#else
  std::memcpy(destination,source,bytes);
#endif
}

// Preserve the rejected precondition in shared output when DSP printf is unavailable.
uint32_t snapshot_reject(QHPI_Tensor** out, uint32_t no, const QHPI_Tensor* const* in, uint32_t ni, uint32_t code) {
  if (no > 1) {
    auto* d=static_cast<uint32_t*>(qhpi_tensor_raw_data(out[1]));
    if (d) {
      d[0]=0x51a7e001; d[1]=code;
      unsigned j=2;
      if (ni>5) {
        auto a=qhpi_tensor_quant_parameters(in[5]); auto b=qhpi_tensor_quant_parameters(out[1]);
        std::memcpy(d+j,&a.stepsize,4);++j;std::memcpy(d+j,&b.stepsize,4);++j;
        d[j++]=a.zero_offset;d[j++]=b.zero_offset;
      }
      for(unsigned i=0;i<ni && i<7;++i) { auto z=qhpi_tensor_shape(in[i]); d[j++]=z.rank;for(unsigned k=0;k<4;++k)d[j++]=z.dims[k]; }
      for(unsigned i=0;i<no && i<3;++i) { auto z=qhpi_tensor_shape(out[i]); d[j++]=z.rank;for(unsigned k=0;k<4;++k)d[j++]=z.dims[k]; }
    }
  }
  return 1;
}

// Each snapshot is the exact rounded state used as input to the next integer step.
// Layout [B*T,H,128,128]; separate heads write disjoint, contiguous planes.
template <bool QA>
uint32_t execute_snapshots(QHPI_RuntimeHandle* handle, uint32_t no, QHPI_Tensor** outputs,
                          uint32_t ni, const QHPI_Tensor* const* inputs) {
#if !DELTA_INT16
  return 1;
#else
  if (no != 3 || ni != 7 || qhpi_num_slices(handle) != 1) {
    SNAPSHOT_LOG("SNAPSHOT_ERROR arity outputs=%u inputs=%u slices=%u\n",no,ni,qhpi_num_slices(handle)); return snapshot_reject(outputs,no,inputs,ni,1);
  }
  const auto sh = qhpi_tensor_shape(inputs[0]);
  if (sh.rank != 4) { SNAPSHOT_LOG("SNAPSHOT_ERROR rank=%u\n",sh.rank); return snapshot_reject(outputs,no,inputs,ni,2); }
  const unsigned B=sh.dims[0], T=sh.dims[1], H=sh.dims[2];
  if (!B || !T || !H || sh.dims[3] != 128 ||
      !shape(inputs[1], B,T,H,128) || !shape(inputs[2], B,T,H,128) ||
      !shape(inputs[3], B,T,H,1) || !shape(inputs[4], B,T,H,1) ||
      !shape(inputs[5], B,H,128,128) || !shape(inputs[6], B,T,1,1) ||
      !shape(outputs[2], 1,1,128,64) || !shape(outputs[0], B,T,H,128) || !shape(outputs[1], B*T,H,128,128)) {
    SNAPSHOT_LOG("SNAPSHOT_ERROR shapes B=%u T=%u H=%u\n",B,T,H);
    for(unsigned j=0;j<no;++j) { auto z=qhpi_tensor_shape(outputs[j]);SNAPSHOT_LOG("SNAPSHOT_OUT %u rank%u %u %u %u %u\n",j,z.rank,z.dims[0],z.dims[1],z.dims[2],z.dims[3]); }
    for(unsigned j=0;j<ni;++j) { auto z=qhpi_tensor_shape(inputs[j]);SNAPSHOT_LOG("SNAPSHOT_IN %u rank%u %u %u %u %u\n",j,z.rank,z.dims[0],z.dims[1],z.dims[2],z.dims[3]); }
    return snapshot_reject(outputs,no,inputs,ni,3);
  }
  Access<QA> q(inputs[0]), k(inputs[1]), v(inputs[2]), beta(inputs[3]), decay(inputs[4]), out(outputs[0]);
  Access<true> sin(inputs[5]), hist(outputs[1]);
  auto* work=static_cast<uint16_t*>(qhpi_tensor_raw_data(outputs[2]));
  const auto* valid=static_cast<const int32_t*>(qhpi_tensor_raw_data(inputs[6]));
  if (!q.data || !k.data || !v.data || !beta.data || !decay.data || !out.data ||
      !sin.data || !hist.data || !work || !valid || sin.offset != hist.offset || sin.scale != hist.scale ||
      sin.scale <= 0 || sin.offset < 0 || sin.offset > 65535 || sin.data == hist.data) {
    SNAPSHOT_LOG("SNAPSHOT_ERROR data_encoding in_scale=%.12g out_scale=%.12g in_zp=%.0f out_zp=%.0f sin=%p hist=%p work=%p valid=%p\n",sin.scale,hist.scale,sin.offset,hist.offset,sin.data,hist.data,work,valid);
    return snapshot_reject(outputs,no,inputs,ni,4);
  }
  const bool save_recipe = valid[0] == 3;
  if(save_recipe) {
    if constexpr (!QA) return snapshot_reject(outputs,no,inputs,ni,5);
    using namespace replay_wire;
    if(B!=1 || T!=replay_wire::T || H!=replay_wire::H) return snapshot_reject(outputs,no,inputs,ni,6);
    Header header{};header.magic=Magic;header.version=2;header.rows=T;header.heads=H;
    const unsigned ids[5]={1,2,3,4,5};
    for(unsigned i=0;i<5;++i) {auto p=qhpi_tensor_quant_parameters(inputs[ids[i]]);header.scales[i]=p.stepsize;header.zeros[i]=p.zero_offset;}
    for(unsigned t=0;t<T;++t)header.valid[t]=valid[t]!=0;
    auto* raw=static_cast<uint8_t*>(hist.data);std::memcpy(raw,&header,sizeof(header));raw+=sizeof(header);
    for(unsigned i=1;i<=4;++i) {
      const size_t bytes=(i<=2?VectorElements:GateElements)*2;
      snapshot_copy(raw,qhpi_tensor_raw_data(inputs[i]),bytes);raw+=bytes;
    }
  }
  alignas(128) float qt[128], kt[128], vt[128], ot[128];
#ifdef DR16M_HAVE_HVX
  alignas(128) dr16m::Tables tb;
#endif
  constexpr size_t S=128*128;
  for (unsigned b=0; b<B; ++b) for (unsigned h=0; h<H; ++h) {
    snapshot_copy(work,static_cast<const uint16_t*>(sin.data)+(b*H+h)*S,S*2);
    const uint16_t* src=work;
    for (unsigned t=0; t<T; ++t) {
      const size_t row=(static_cast<size_t>(b)*T+t)*H+h;
      auto* dst=static_cast<uint16_t*>(hist.data)+row*S;
      if (valid[b*T+t]) {
        q.load(row*128, qt,128); k.load(row*128,kt,128); v.load(row*128,vt,128);
#ifdef DR16M_HAVE_HVX
        dr16m::step_hvx(src,work,int(sin.offset),sin.scale,sin.inv,dr16m::lm_of(int(sin.offset)),
                       qt,kt,vt,decay.get(row),beta.get(row),int(b*H+h),ot,tb);
#else
        dr16m::step_ref(src,work,int(sin.offset),sin.scale,sin.inv,dr16m::lm_of(int(sin.offset)),
                       qt,kt,vt,decay.get(row),beta.get(row),int(b*H+h),ot);
#endif
        out.store(row*128,ot,128);
      } else {
        // Invalid rows preserve the most recent state.
        out.store(row*128,kZeroRow,128);
      }
      // Save every row regardless of its later acceptance.
      if(!save_recipe) snapshot_copy(dst,work,S*2);
    }
  }
  return 0;
#endif
}
#include "Replay.inc"
QHPI_Tensor_Signature_v1 in_sig[10][7], out_sig[10][3];
QHPI_Kernel_v1 kernels[10];
QHPI_OpInfo_v1 ops[6]{};
}

extern "C" const char* qhpi_init() {
  // Float state throughout the recurrence. These signatures require float state
  // edges; callers must not annotate them as uint16. Activation conversion is
  // vectorized at the boundary, with no per-step state requantization.
  const QHPI_Plugin_Function functions[] = {execute<false>, execute<true>, execute<false, true>, execute<true, true>, execute_decode16};
  const char* names[] = {"DeltaRule_f32_f32", "DeltaRule_u16_f32", "DeltaRuleState16_f32", "DeltaRuleState16_u16", "DeltaRuleDecode16_unique_u16"};
  for (unsigned i = 0; i < 4; ++i) {
    const unsigned act = (i & 1) ? QHPI_QUINT16 : QHPI_FLOAT32;
    const unsigned state = i < 2 ? QHPI_FLOAT32 : QHPI_QUINT16;
    for (unsigned j = 0; j < 6; ++j) in_sig[i][j] = signature(j == 5 ? state : act);
    out_sig[i][0] = signature(act); out_sig[i][1] = signature(state);
    out_sig[i][2] = signature(QHPI_FLOAT32);
    auto& kernel = kernels[i];
    kernel.function_name = names[i]; kernel.function = functions[i];
    kernel.resources = QHPI_RESOURCE_HVX; kernel.multithreaded = DELTA_THREADS;
    kernel.min_inputs = 6; kernel.input_signature = in_sig[i];
    kernel.min_outputs = i < 2 ? 2 : 3; kernel.output_signature = out_sig[i];
    kernel.sync_block_size = 0;
  }
  for (unsigned j = 0; j < 6; ++j) in_sig[4][j] = signature(QHPI_QUINT16);
  out_sig[4][0] = signature(QHPI_QUINT16); out_sig[4][1] = signature(QHPI_QUINT16);
  kernels[4].function_name = names[4]; kernels[4].function = functions[4];
  kernels[4].resources = QHPI_RESOURCE_HVX; kernels[4].multithreaded = DELTA_THREADS;
  kernels[4].min_inputs = 6; kernels[4].input_signature = in_sig[4];
  kernels[4].min_outputs = 2; kernels[4].output_signature = out_sig[4]; kernels[4].sync_block_size = 0;
  ops[0].name = "DeltaRuleOpPackage::DeltaRule";
  ops[1].name = "DeltaRuleOpPackage::DeltaRuleState16";
  ops[2].name = "DeltaRuleOpPackage::DeltaRuleDecode16";
  for (unsigned i = 0; i < 2; ++i) {
    ops[i].num_kernels = 2; ops[i].kernels = kernels + 2 * i;
  }
  ops[2].num_kernels = 1; ops[2].kernels = kernels + 4;
  // Masked prefill op: kernels 5 (f32 activations) and 6 (uint16 activations).
  const QHPI_Plugin_Function masked[] = {execute<false, true, true>, execute<true, true, true>};
  const char* masked_names[] = {"DeltaRuleState16Masked_f32", "DeltaRuleState16Masked_u16"};
  for (unsigned m = 0; m < 2; ++m) {
    const unsigned i = 5 + m, act = m ? QHPI_QUINT16 : QHPI_FLOAT32;
    for (unsigned j = 0; j < 7; ++j) in_sig[i][j] = signature(j == 6 ? QHPI_INT32 : (j == 5 ? QHPI_QUINT16 : act));
    out_sig[i][0] = signature(act); out_sig[i][1] = signature(QHPI_QUINT16); out_sig[i][2] = signature(QHPI_FLOAT32);
    auto& kernel = kernels[i];
    kernel.function_name = masked_names[m]; kernel.function = masked[m];
    kernel.resources = QHPI_RESOURCE_HVX; kernel.multithreaded = DELTA_THREADS;
    kernel.min_inputs = 7; kernel.input_signature = in_sig[i];
    kernel.min_outputs = 3; kernel.output_signature = out_sig[i];
    kernel.sync_block_size = 0;
  }
  ops[3].name = "DeltaRuleOpPackage::DeltaRuleState16Masked";
  ops[3].num_kernels = 2; ops[3].kernels = kernels + 5;
  const QHPI_Plugin_Function snapshots[] = {execute_snapshots<false>, execute_snapshots<true>};
  const char* snapshot_names[] = {"DeltaRuleSnapshots16_f32", "DeltaRuleSnapshots16_u16"};
  for (unsigned m=0; m<2; ++m) {
    const unsigned i=7+m, act=m ? QHPI_QUINT16 : QHPI_FLOAT32;
    for (unsigned j=0; j<7; ++j) in_sig[i][j]=signature(j==6 ? QHPI_INT32 : (j==5 ? QHPI_QUINT16 : act));
    out_sig[i][0]=signature(act);
    out_sig[i][1]=signature(QHPI_QUINT16);
    // Histories exceed VTCM: let QNN place them in external memory.
    out_sig[i][1].mem_placement=QHPI_MEM_LOC_DDR_ONLY;
    out_sig[i][2]=signature(QHPI_FLOAT32); // native 32 KiB VTCM current state
    auto& kernel=kernels[i];
    kernel.function_name=snapshot_names[m]; kernel.function=snapshots[m];
    kernel.resources=QHPI_RESOURCE_HVX; kernel.multithreaded=0;
    kernel.min_inputs=7; kernel.input_signature=in_sig[i];
    kernel.min_outputs=3; kernel.output_signature=out_sig[i];
  }
  ops[4].name="DeltaRuleOpPackage::DeltaRuleSnapshots16";
  ops[4].num_kernels=2; ops[4].kernels=kernels+7;
  in_sig[9][0]=signature(QHPI_INT32);in_sig[9][1]=signature(QHPI_INT32);in_sig[9][2]=signature(QHPI_INT32);
  out_sig[9][0]=signature(QHPI_INT32);
  auto& replay=kernels[9];replay.function_name="DeltaRuleReplay16_raw";replay.function=execute_replay;
  replay.resources=QHPI_RESOURCE_HVX;replay.multithreaded=0;
  replay.min_inputs=3;replay.input_signature=in_sig[9];replay.min_outputs=1;replay.output_signature=out_sig[9];
  ops[5].name="DeltaRuleOpPackage::DeltaRuleReplay16";ops[5].num_kernels=1;ops[5].kernels=kernels+9;
  qhpi_register_ops_v1(6, ops, "DeltaRuleOpPackage");
  return "DeltaRuleOpPackage";
}
