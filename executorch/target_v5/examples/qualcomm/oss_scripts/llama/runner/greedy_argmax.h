#pragma once

#include <cstddef>
#include <cstdint>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace example {

// count must be nonzero. Preserve first-index tie breaking, including tails.
template <typename T>
inline int32_t greedy_argmax(const T* values, size_t count) {
  size_t best = 0;
  for (size_t i = 1; i < count; ++i)
    if (values[i] > values[best]) best = i;
  return static_cast<int32_t>(best);
}

#if defined(__aarch64__)
template <>
inline int32_t greedy_argmax<uint16_t>(const uint16_t* values, size_t count) {
  uint16x8_t maxima = vdupq_n_u16(0);
  size_t i = 0;
  for (; i + 8 <= count; i += 8)
    maxima = vmaxq_u16(maxima, vld1q_u16(values + i));
  uint16_t maximum = vmaxvq_u16(maxima);
  for (; i < count; ++i)
    if (values[i] > maximum) maximum = values[i];
  // Scan eight candidates at a time; inspect lanes only in the first match.
  const uint16x8_t target = vdupq_n_u16(maximum);
  i = 0;
  for (; i + 8 <= count; i += 8) {
    if (vmaxvq_u16(vceqq_u16(vld1q_u16(values + i), target))) break;
  }
  for (; i < count; ++i)
    if (values[i] == maximum) return static_cast<int32_t>(i);
  return 0;
}
#endif

} // namespace example
