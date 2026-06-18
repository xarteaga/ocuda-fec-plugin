// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.
#include "ocuda-fec/cuda_helpers/device_vector.h"
#include "ocuda-fec/cuda_helpers/ldpc_decoder.h"
#include "ocuda-fec/cuda_helpers/ldpc_decoder_cuda_helpers.h"

using namespace ocudu;
using namespace cuda;

class ldpc_decoder_stream_impl : public ldpc_decoder
{
public:
  ldpc_decoder_stream_impl(span<const base_graph_description> base_graph_descriptions_) :
    base_graph_descriptions(base_graph_descriptions_)
  {
  }

  void ldpc_decode(span<const ldpc_decoder_cb_arguments> codeblocks, cuda_stream& stream) override;

private:
  device_vector<ldpc_decoder_cb_arguments, max_nof_codeblocks> d_codeblocks;
  span<const base_graph_description>                           base_graph_descriptions;
};

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

  // When not dealing with special cases, promotion sum: if the sum exceeds LLR_MAX (in absolute value), then return
  // LLR_INFTY (with the proper sign).
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

__device__ static void
cuda_sanitize_soft_bits(int8_t* soft_bits, unsigned nof_llrs, unsigned lifting_size, unsigned bg_N_full)
{
  static constexpr int8_t soft_bits_clamp_low  = -64;
  static constexpr int8_t soft_bits_clamp_high = 64;

  unsigned nof_soft_bits_grid = gridDim.x * blockDim.x;
  unsigned nof_nodes_grid     = gridDim.y * blockDim.y;

  unsigned i_soft_bit_grid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned i_node_grid     = blockIdx.y * blockDim.y + threadIdx.y;

  unsigned soft_bits_begin = 2 * lifting_size;
  unsigned soft_bits_end   = soft_bits_begin + nof_llrs;

  for (unsigned i_node = i_node_grid; i_node < bg_N_full; i_node += nof_nodes_grid) {
    for (unsigned i_soft_bit = i_soft_bit_grid; i_soft_bit < lifting_size; i_soft_bit += nof_soft_bits_grid) {
      unsigned absolute_idx = i_node * lifting_size + i_soft_bit;

      if (absolute_idx < soft_bits_begin) {
        soft_bits[absolute_idx] = 0;
      } else if (absolute_idx < soft_bits_end) {
        int8_t this_soft_bit    = soft_bits[absolute_idx];
        soft_bits[absolute_idx] = max(soft_bits_clamp_low, min(soft_bits_clamp_high, this_soft_bit));
      } else {
        soft_bits[absolute_idx] = 0;
      }
    }
  }
  __syncthreads();
}

__device__ static void update_variable_to_check_messages(int*            var_to_check,
                                                         const int8_t*   soft_bits,
                                                         const int8_t*   check_to_var,
                                                         const uint16_t* bg_check_edges,
                                                         unsigned        lifting_size,
                                                         unsigned        nof_bg_check_edges,
                                                         unsigned        bg_N_high_rate,
                                                         unsigned        check_node)
{
  unsigned i_soft_bit_block = threadIdx.x;
  unsigned i_node_block     = threadIdx.y;

  unsigned nof_soft_bits_block = blockDim.x;
  unsigned nof_nodes_block     = blockDim.y;

  for (unsigned i_soft_bit = i_soft_bit_block; i_soft_bit < lifting_size; i_soft_bit += nof_soft_bits_block) {
    for (unsigned i_node = i_node_block; i_node < nof_bg_check_edges; i_node += nof_nodes_block) {
      unsigned bg_check_edge = bg_check_edges[i_node];

      if (bg_check_edge < bg_N_high_rate) {
        int temp = soft_bits[bg_check_edge * lifting_size + i_soft_bit];

        if (check_to_var != nullptr) {
          int this_check_to_var = check_to_var[bg_check_edge * lifting_size + i_soft_bit];
          temp                  = cuda_device_llr_sum(temp, -this_check_to_var);
        }

        var_to_check[bg_check_edge * lifting_size + i_soft_bit] = temp;
      }
    }
  }

  if ((i_node_block == 0) && (check_node >= 4)) {
    for (unsigned i_soft_bit = i_soft_bit_block; i_soft_bit < lifting_size; i_soft_bit += nof_soft_bits_block) {
      int temp = soft_bits[(bg_N_high_rate + check_node - 4) * lifting_size + i_soft_bit];

      if (check_to_var != nullptr) {
        int this_check_to_var = check_to_var[bg_N_high_rate * lifting_size + i_soft_bit];
        temp                  = cuda_device_llr_sum(temp, -this_check_to_var);
      }

      var_to_check[bg_N_high_rate * lifting_size + i_soft_bit] = temp;
    }
  }
}

__device__ static void cuda_check_min_check_node(int&      this_min_var_to_check,
                                                 int&      this_second_min_var_to_check,
                                                 unsigned& this_min_var_to_check_index,
                                                 unsigned& this_sign_prod_var_to_check,
                                                 int       this_var_to_check,
                                                 unsigned  i_node)
{
  int  var_to_check_abs        = std::abs(this_var_to_check);
  bool is_min                  = (var_to_check_abs < this_min_var_to_check);
  int  new_second_min          = is_min ? this_min_var_to_check : var_to_check_abs;
  bool is_best_two             = (var_to_check_abs < this_second_min_var_to_check);
  this_second_min_var_to_check = is_best_two ? new_second_min : this_second_min_var_to_check;
  this_min_var_to_check        = is_min ? var_to_check_abs : this_min_var_to_check;
  this_min_var_to_check_index  = is_min ? i_node : this_min_var_to_check_index;
  this_sign_prod_var_to_check ^= this_var_to_check;
}

__device__ static int cuda_update_check_to_var(unsigned i_node,
                                               unsigned this_min_var_to_check_index,
                                               int      this_min_var_to_check,
                                               int      this_sign_prod_var_to_check,
                                               int      this_second_min_var_to_check,
                                               int      this_var_to_check)
{
  int this_check_to_var =
      (i_node != this_min_var_to_check_index) ? this_min_var_to_check : this_second_min_var_to_check;

  int final_sign    = this_sign_prod_var_to_check ^ this_var_to_check;
  this_check_to_var = cuda_copysign(this_check_to_var, final_sign);

  return this_check_to_var;
}

static __device__ void cuda_ldpc_decoder_process_check_node(int8_t*                           soft_bits,
                                                            int8_t*                           global_check_to_var,
                                                            const base_graph_check_node_info* bg_info,
                                                            unsigned                          check_node,
                                                            unsigned                          lifting_size,
                                                            unsigned                          bg_N_high_rate,
                                                            bool                              check_to_var_is_init)
{
  constexpr unsigned MAX_CHECK_NODE_DEGREE = 27;
  constexpr unsigned MAX_LIFTING_SIZE      = 384;
  constexpr unsigned MAX_BG_CHECK_EDGES    = 20;

  unsigned i_soft_bit_block = threadIdx.x;
  unsigned i_node_block     = threadIdx.y;

  unsigned nof_soft_bits_block = blockDim.x;
  unsigned nof_nodes_block     = blockDim.y;

  const uint16_t nof_bg_check_edges = bg_info[check_node].size;

  int8_t* check_to_var = &global_check_to_var[MAX_LIFTING_SIZE * MAX_CHECK_NODE_DEGREE * check_node];

  // Load BG check edges and shifts in the shared memory.
  __shared__ uint16_t bg_check_edges[MAX_BG_CHECK_EDGES];
  __shared__ uint16_t shifts[MAX_BG_CHECK_EDGES];
  if (i_soft_bit_block == 0) {
    const uint16_t* global_bg_check_edges = bg_info[check_node].bg_check_edges;
    const uint16_t* global_shifts         = bg_info[check_node].shifts;

    for (unsigned i_node = i_node_block; i_node < nof_bg_check_edges; i_node += nof_nodes_block) {
      bg_check_edges[i_node] = global_bg_check_edges[i_node];
      shifts[i_node]         = global_shifts[i_node];
    }
  }
  __syncthreads();

  // Calculate the var_to_check message for the current node.
  __shared__ int var_to_check[MAX_LIFTING_SIZE * MAX_CHECK_NODE_DEGREE];
  update_variable_to_check_messages(var_to_check,
                                    soft_bits,
                                    check_to_var_is_init ? check_to_var : nullptr,
                                    bg_check_edges,
                                    lifting_size,
                                    nof_bg_check_edges,
                                    bg_N_high_rate,
                                    check_node);
  __syncthreads();

  // Compute the minimum var_to_check message for the current node.
  __shared__ int      min_var_to_check[MAX_LIFTING_SIZE];
  __shared__ int      second_min_var_to_check[MAX_LIFTING_SIZE];
  __shared__ unsigned min_var_to_check_index[MAX_LIFTING_SIZE];
  __shared__ unsigned sign_prod_var_to_check[MAX_LIFTING_SIZE];

  if (i_node_block == 0) {
    for (unsigned idx = i_soft_bit_block; idx < lifting_size; idx += nof_soft_bits_block) {
      int      this_min_var_to_check        = 120;
      int      this_second_min_var_to_check = 120;
      unsigned this_min_var_to_check_index  = 0;
      unsigned this_sign_prod_var_to_check  = 0;

      for (unsigned i_node = 0; i_node != nof_bg_check_edges; ++i_node) {
        unsigned tmp_index = (idx + shifts[i_node]) % lifting_size;

        unsigned bg_check_edge      = bg_check_edges[i_node];
        unsigned bg_check_edge_ceil = min(bg_check_edge, bg_N_high_rate);

        int this_var_to_check = var_to_check[bg_check_edge_ceil * lifting_size + tmp_index];

        cuda_check_min_check_node(this_min_var_to_check,
                                  this_second_min_var_to_check,
                                  this_min_var_to_check_index,
                                  this_sign_prod_var_to_check,
                                  this_var_to_check,
                                  i_node);
      }

      this_min_var_to_check        = cuda_ldpc_decoder_scale_llr(this_min_var_to_check);
      this_second_min_var_to_check = cuda_ldpc_decoder_scale_llr(this_second_min_var_to_check);

      min_var_to_check[idx]        = this_min_var_to_check;
      second_min_var_to_check[idx] = this_second_min_var_to_check;
      min_var_to_check_index[idx]  = this_min_var_to_check_index;
      sign_prod_var_to_check[idx]  = this_sign_prod_var_to_check;
    }
  }
  __syncthreads();

  for (unsigned idx = i_soft_bit_block; idx < lifting_size; idx += nof_soft_bits_block) {
    for (unsigned i_node = i_node_block; i_node < nof_bg_check_edges; i_node += nof_nodes_block) {
      unsigned tmp_index                    = (idx + lifting_size - shifts[i_node]) % lifting_size;
      int      this_min_var_to_check        = min_var_to_check[tmp_index];
      int      this_second_min_var_to_check = second_min_var_to_check[tmp_index];
      unsigned this_min_var_to_check_index  = min_var_to_check_index[tmp_index];
      unsigned this_sign_prod_var_to_check  = sign_prod_var_to_check[tmp_index];

      unsigned bg_check_edge      = bg_check_edges[i_node];
      unsigned bg_check_edge_ceil = min(bg_check_edge, bg_N_high_rate);

      int this_var_to_check = var_to_check[bg_check_edge_ceil * lifting_size + idx];
      int this_check_to_var = cuda_update_check_to_var(i_node,
                                                       this_min_var_to_check_index,
                                                       this_min_var_to_check,
                                                       this_sign_prod_var_to_check,
                                                       this_second_min_var_to_check,
                                                       this_var_to_check);
      int this_soft_bits    = cuda_device_llr_promotion_sum(this_check_to_var, this_var_to_check);

      check_to_var[bg_check_edge_ceil * lifting_size + idx] = this_check_to_var;
      soft_bits[bg_check_edge * lifting_size + idx]         = this_soft_bits;
    }
  }
  __syncthreads();
}

static __device__ uint8_t cuda_ldpc_decoder_pack_8bits(const int8_t* input)
{
  // Load unpacked data.
  uint64_t input_u64 = *reinterpret_cast<const uint64_t*>(input);

  // Move Sign to the LSB.
  input_u64 = input_u64 >> 7;

  // Mask the unpacked bits.
  input_u64 &= 0x101010101010101UL;

  // Pack data and select the first eight MSB.
  uint64_t packed = (input_u64 * 0x8040201008040201UL) >> 56UL;

  return packed;
}

static __device__ void cuda_ldpc_decoder_get_hard_bits(uint8_t* output, int8_t* soft_bits, unsigned nof_bytes)
{
  unsigned i_byte_grid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned i_node_grid = blockIdx.y * blockDim.y + threadIdx.y;

  unsigned nof_bytes_grid = gridDim.x * blockDim.x;

  if (i_node_grid == 0) {
    for (unsigned i_byte = i_byte_grid; i_byte < nof_bytes; i_byte += nof_bytes_grid) {
      output[i_byte] = cuda_ldpc_decoder_pack_8bits(&soft_bits[i_byte * 8]);
    }
  }
}

static __global__ void cuda_ldpc_decode(const base_graph_description*    base_graph_descriptions,
                                        const ldpc_decoder_cb_arguments* codeblocks,
                                        unsigned                         nof_codeblocks)
{
  unsigned cb_idx_grid = blockIdx.z * blockDim.z + threadIdx.z;
  unsigned nof_cb_grid = gridDim.z * blockDim.z;

  // Iterate for all codeblocks.
  for (unsigned cb_idx = cb_idx_grid; cb_idx < nof_codeblocks; cb_idx += nof_cb_grid) {
    // Extract codeblock configuration.
    const ldpc_decoder_cb_arguments&     cb_args   = codeblocks[cb_idx];
    const ldpc_decoder_cb_configuration& cb_config = cb_args.cb_configuration;

    const base_graph_check_node_info* bg_info = base_graph_descriptions[cb_config.bg_info_pos].check_nodes_info;

    // Sanitize soft bits.
    cuda_sanitize_soft_bits(cb_config.soft_bits, cb_config.nof_llr, cb_config.lifting_size, cb_config.bg_N_full);

    // Perform decoding iterations.
    for (unsigned iteration = 0; iteration != cb_config.max_nof_iterations; ++iteration) {
      // Process check nodes.
      for (unsigned i_layer = 0; i_layer != cb_config.nof_layers; ++i_layer) {
        cuda_ldpc_decoder_process_check_node(cb_config.soft_bits,
                                             cb_config.check_to_var,
                                             bg_info,
                                             i_layer,
                                             cb_config.lifting_size,
                                             cb_config.bg_N_high_rate,
                                             iteration != 0);
      }
    }

    // Perform hard decision.
    cuda_ldpc_decoder_get_hard_bits(cb_args.output, cb_config.soft_bits, cb_config.message_size_bytes);
  }
}

void ldpc_decoder_stream_impl::ldpc_decode(span<const ocudu::cuda::ldpc_decoder_cb_arguments> codeblocks,
                                           cuda_stream&                                       stream)
{
  unsigned nof_codeblocks = codeblocks.size();

  d_codeblocks.resize(nof_codeblocks);

  copy_host_to_device(d_codeblocks.get(), codeblocks, stream);

  dim3 block_size(256, 4, 1);
  dim3 grid_size(1, 1, nof_codeblocks);

  cuda_ldpc_decode<<<grid_size, block_size, 0, static_cast<cudaStream_t>(stream.get())>>>(
      base_graph_descriptions.data(), d_codeblocks.get().data(), nof_codeblocks);
}

std::unique_ptr<ldpc_decoder> ldpc_decoder::create(span<const base_graph_description> base_graph_descriptions)
{
  return std::make_unique<ldpc_decoder_stream_impl>(base_graph_descriptions);
}
