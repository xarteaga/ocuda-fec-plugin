// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#include "ocuda-fec/phy/upper/channel_coding/ldpc_decoder_cuda.h"
#include "ldpc_graph_impl.h"
#include "ldpc_luts_impl.h"
#include "ocudu/phy/upper/channel_coding/crc_calculator.h"
#include "ocudu/phy/upper/log_likelihood_ratio.h"
#include "ocudu/support/ocudu_assert.h"
#include "fmt/compile.h"

using namespace ocudu;
using namespace ocudu::ldpc;

static span<const int8_t> cast_llr_to_int8(span<const log_likelihood_ratio> llr)
{
  return {reinterpret_cast<const int8_t*>(llr.data()), llr.size()};
}

ldpc_decoder_cuda::ldpc_decoder_cuda(std::shared_ptr<cuda_ldpc_decoder_backend> backend_) :
  cuda_backend(std::move(backend_))
{
}

void ldpc_decoder_cuda::decode(bit_buffer&                       output,
                               span<const log_likelihood_ratio>  input,
                               const configuration&              cfg,
                               cuda_ldpc_decoder_callback_func&& callback)
{
  unsigned               pos           = get_lifting_size_position(cfg.lifting_size);
  unsigned               skip          = (cfg.base_graph == ldpc_base_graph_type::BG2) ? NOF_LIFTING_SIZES : 0;
  const ldpc_graph_impl& current_graph = graph_array[skip + pos];

  unsigned bg_N_full      = current_graph.get_nof_BG_var_nodes_full();
  unsigned bg_N_short     = current_graph.get_nof_BG_var_nodes_short();
  unsigned bg_M           = current_graph.get_nof_BG_check_nodes();
  unsigned bg_K           = current_graph.get_nof_BG_info_nodes();
  unsigned bg_N_high_rate = bg_K + 4;
  ocudu_assert(bg_K == bg_N_full - bg_M, "Invalid bg_K value '{}'", bg_K);
  unsigned lifting_size = static_cast<uint16_t>(cfg.lifting_size);

  nof_significant_bits = bg_K * lifting_size - cfg.nof_filler_bits;

  units::bits message_length(bg_K * lifting_size);
  uint16_t    max_input_length = bg_N_short * lifting_size;
  ocudu_assert(output.size() == message_length.value(),
               "The output size {} is not equal to the message length {}.",
               output.size(),
               message_length);
  ocudu_assert(input.size() <= max_input_length,
               "The input size {} exceeds the maximum message length {}.",
               input.size(),
               max_input_length);

  // The minimum input length is message_length + two times the lifting size.
  uint16_t min_input_length = message_length.value() + 2 * lifting_size;
  ocudu_assert(input.size() >= min_input_length,
               "The input length {} does not reach minimum {}",
               input.size(),
               min_input_length);

  // Find the last soft bit in the buffer and trim the output.
  const log_likelihood_ratio* last =
      std::find_if(input.rbegin(), input.rend(), [](const log_likelihood_ratio& in) { return in != 0; }).base();

  // Determine input length.
  unsigned input_size = std::distance(input.begin(), last);

  // Recall that the first 2 * lifting_size bits (2 nodes) are not transmitted.
  // Prepare asynchronous input copy the number of data nodes fully occupied by the LLRs.
  d_soft_bits.resize(2 * lifting_size + input.size());
  cuda::host_to_device_promise input_promise(d_soft_bits.get().subspan(2 * lifting_size, input_size),
                                             cast_llr_to_int8(input.first(input_size)));

  // The minimum codeblock length is message_length + four times the lifting size
  // (that is, the length of the high-rate region).
  unsigned min_codeblock_length = message_length.value() + 4 * lifting_size;
  // The decoder works with at least min_codeblock_length bits. Recall that the encoder also shortens
  // the codeblock by 2 * lifting size before returning it as output.
  unsigned codeblock_length = std::max(input_size + 2UL * lifting_size, static_cast<size_t>(min_codeblock_length));
  // The decoder works with a codeblock length that is a multiple of the lifting size.
  if (codeblock_length % lifting_size != 0) {
    codeblock_length = (codeblock_length / lifting_size + 1) * lifting_size;
  }

  unsigned nof_layers = codeblock_length / lifting_size - bg_K;

  cuda::ldpc_decoder_cb_configuration codeblock_config = {.soft_bits    = d_soft_bits.get().data(),
                                                          .check_to_var = d_check_to_var.get().data(),
                                                          .bg_info_pos  = skip + pos,
                                                          .message_size_bytes =
                                                              message_length.round_up_to_bytes().value(),
                                                          .nof_llr            = input_size,
                                                          .nof_layers         = nof_layers,
                                                          .lifting_size       = lifting_size,
                                                          .bg_N_high_rate     = bg_N_high_rate,
                                                          .bg_N_full          = bg_N_full,
                                                          .max_nof_iterations = cfg.max_iterations};

  cuda_backend->decode(output.get_buffer(), input_promise, codeblock_config, std::move(callback));
}
