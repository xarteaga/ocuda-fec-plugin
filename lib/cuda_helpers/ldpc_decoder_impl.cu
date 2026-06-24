// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#include "device_ldpc_decoder.h"
#include "device_ldpc_rate_dematcher.h"
#include "device_math_helpers.h"
#include "ldpc_decoder_impl.h"

using namespace ocudu;
using namespace cuda;

static __global__ void cuda_sch_decode(const base_graph_description*    base_graph_descriptions,
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

    // Revert rate matching. Combine retransmissions.
    cuda_ldpc_rate_dematch(cb_config.rm_buffer,
                           cb_config.input,
                           cb_config.input_size,
                           cb_config.nof_systematic_bits,
                           cb_config.nof_filler_bits,
                           cb_config.rm_buffer_size,
                           cb_config.shift_k0,
                           cb_config.modulation_order,
                           cb_config.new_data);

    // Load soft bits from the rate matching buffer.
    cuda_load_soft_bits(cb_config.soft_bits,
                        cb_config.rm_buffer,
                        cb_config.rm_buffer_size,
                        cb_config.lifting_size,
                        cb_config.bg_N_full);

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

ldpc_decoder_impl::ldpc_decoder_impl(span<const base_graph_description> base_graph_descriptions_) :
  base_graph_descriptions(base_graph_descriptions_)
{
}

void ldpc_decoder_impl::ldpc_decode(span<const ldpc_decoder_cb_arguments> codeblocks, cuda_stream& stream)
{
  unsigned nof_codeblocks = codeblocks.size();

  d_codeblocks.resize(nof_codeblocks);

  copy_host_to_device(d_codeblocks.get(), codeblocks, stream);

  dim3 block_size(256, 4, 1);
  dim3 grid_size(1, 1, nof_codeblocks);

  cuda_sch_decode<<<grid_size, block_size, 0, static_cast<cudaStream_t>(stream.get())>>>(
      base_graph_descriptions.data(), d_codeblocks.get().data(), nof_codeblocks);
}

std::unique_ptr<ldpc_decoder> ldpc_decoder::create(span<const base_graph_description> base_graph_descriptions)
{
  return std::make_unique<ldpc_decoder_impl>(base_graph_descriptions);
}
