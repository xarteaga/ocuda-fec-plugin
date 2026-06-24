/*
 *
 * Copyright 2021-2026 Software Radio Systems Limited
 *
 * By using this file, you agree to the terms and conditions set
 * forth in the LICENSE file which can be found at the top level of
 * the distribution.
 *
 */
#pragma once

namespace ocudu {
namespace cuda {

__device__ static int cuda_device_llr_promotion_sum(int a, int b)
{
  static constexpr int infinity = 127;
  static constexpr int max_val  = 120;

  if (a == -b) {
    return 0;
  }

  if (std::abs(a) == infinity) {
    return a;
  }

  if (std::abs(b) == infinity) {
    return b;
  }

  // When not dealing with special cases, promotion sum: if the sum exceeds/ When not dealing with special cases,
  // promotion sum: if the sum exceeds).
  int tmp = a + b;
  if (std::abs(tmp) > max_val) {
    tmp = ((tmp > 0) ? infinity : -infinity);
  }

  return tmp;
}

__device__ static int cuda_device_llr_sum(int a, int b)
{
  static constexpr int infinity = 127;
  static constexpr int max_val  = 120;

  if (a == -b) {
    return 0;
  }

  if (std::abs(a) == infinity) {
    return a;
  }

  if (std::abs(b) == infinity) {
    return b;
  }

  // When not dealing with special cases, promotion sum: if the sum exceeds LLR_MAX (in absolute value), then return
  // LLR_INFTY (with the proper sign).
  int tmp = a + b;
  tmp     = max(tmp, -max_val);
  tmp     = min(tmp, +max_val);

  return tmp;
}

__device__ static int cuda_copysign(int a, int b)
{
  if (b < 0) {
    return -std::abs(a);
  }
  return std::abs(a);
}

/// Apply the LDPC decoder scaling factor of the min-sum algorithm that is 0.8 and rounding to the nearest integer.
__device__ static int cuda_ldpc_decoder_scale_llr(int value)
{
  static constexpr int infinity = 127;

  if (std::abs(value) != infinity) {
    value = (value * 8 + 5) / 10;
  }

  return value;
}

} // namespace cuda
} // namespace ocudu
