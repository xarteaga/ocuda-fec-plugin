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

__device__ static void cuda_ldpc_rate_dematch(int8_t*       output,
                                              const int8_t* input,
                                              unsigned      nof_cb_llrs,
                                              unsigned      nof_systematic_bits,
                                              unsigned      nof_filler_bits,
                                              unsigned      rm_buffer_size,
                                              unsigned      shift_k0,
                                              unsigned      modulation_order,
                                              bool          new_data)
{
  unsigned i_soft_bit_grid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned i_node_grid     = blockIdx.y * blockDim.y + threadIdx.y;

  unsigned nof_soft_bits_grid = gridDim.x * blockDim.x;
  unsigned nof_nodes_grid     = gridDim.y * blockDim.y;

  unsigned i_input        = i_node_grid * nof_soft_bits_grid + i_soft_bit_grid;
  unsigned nof_input_grid = nof_soft_bits_grid * nof_nodes_grid;

  unsigned nof_info_bits      = nof_systematic_bits - nof_filler_bits;
  unsigned nof_effective_bits = rm_buffer_size - nof_filler_bits;
  unsigned K                  = nof_cb_llrs / modulation_order;

  for (unsigned i = i_input; i < rm_buffer_size; i += nof_input_grid) {
    if (i > nof_info_bits && i < nof_systematic_bits) {
      output[i] = 120;
    } else {
      if (new_data) {
        output[i] = 0;
      }

      unsigned j =
          (i + nof_effective_bits - ((i > nof_info_bits) ? nof_filler_bits : 0) - shift_k0) % nof_effective_bits;

      for (; j < nof_cb_llrs; j += nof_effective_bits) {
        unsigned input_idx = (modulation_order * (j % K)) + j / K;
        output[i]          = cuda_device_llr_sum(output[i], input[input_idx]);
      }
    }
  }

  __syncthreads();
}

} // namespace cuda
} // namespace ocudu
