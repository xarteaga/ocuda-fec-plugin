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
  /// Starting position of different redundancy versions (k0), as per TS38.212 Table 5.4.2.1-2.
  static constexpr std::array<double, 4> shift_factor_bg1 = {0, 17, 33, 56};
  static constexpr std::array<double, 4> shift_factor_bg2 = {0, 13, 25, 43};

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

  // Compute shift_k0 according to TS38.212 Table 5.4.2.1-2.
  span<const double> shift_factor;
  unsigned           BG_K = 0;
  if ((cfg.block_length % BG1_N_SHORT) == 0) {
    // input is a BG1 codeblock
    shift_factor = shift_factor_bg1;
    BG_K         = BG1_N_FULL - BG1_M;
  } else if ((cfg.block_length % BG2_N_SHORT) == 0) {
    // input is a BG2 codeblock
    shift_factor = shift_factor_bg2;
    BG_K         = BG2_N_FULL - BG2_M;
  } else {
    ocudu_assert(false, "LDPC rate dematching: invalid input length.");
  }

  // Recall that 2 * lifting_size systematic bits are shortened out of the codeblock.
  unsigned nof_systematic_bits = (BG_K - 2) * lifting_size;
  ocudu_assert(cfg.nof_filler_bits < nof_systematic_bits, "LDPC rate dematching: invalid number of filler bits.");
  unsigned nof_filler_bits = cfg.nof_filler_bits;

  // If N_ref is given, limit the buffer size. Otherwise, use default.
  unsigned buffer_length = cfg.block_length;
  if (cfg.Nref > 0) {
    buffer_length = std::min(cfg.Nref, cfg.block_length);
  }

  double   tmp      = (shift_factor[cfg.rv] * buffer_length) / cfg.block_length;
  unsigned shift_k0 = static_cast<uint16_t>(std::floor(tmp)) * lifting_size;

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

  unsigned input_size = input.size();
  buffer_length       = std::min(shift_k0 + nof_filler_bits + input_size, buffer_length);

  // The minimum input length is message_length + two times the lifting size.
  ocudu_assert(buffer_length >= message_length.value(),
               "The RM buffer length {} does not reach minimum {}",
               buffer_length,
               message_length.value());

  // Prepare asynchronous input copy the number of data nodes fully occupied by the LLRs.
  d_input.resize(input_size);
  cuda::host_to_device_promise input_promise(d_input.get(), cast_llr_to_int8(input));

  // The minimum codeblock length is message_length + four times the lifting size (that is, the length of the high-rate
  // region).
  unsigned min_codeblock_length = message_length.value() + 4 * lifting_size;

  // The decoder works with at least min_codeblock_length bits. Recall that the encoder also shortens the codeblock by 2
  // * lifting size before returning it as output.
  unsigned codeblock_length = std::max(buffer_length + 2UL * lifting_size, static_cast<size_t>(min_codeblock_length));
  // The decoder works with a codeblock length that is a multiple of the lifting size.
  if (codeblock_length % lifting_size != 0) {
    codeblock_length = (codeblock_length / lifting_size + 1) * lifting_size;
  }

  unsigned nof_layers = codeblock_length / lifting_size - bg_K;

  d_rm_buffer.resize(buffer_length);
  d_soft_bits.resize(lifting_size * nof_layers);

  cuda::ldpc_decoder_cb_configuration codeblock_config = {.input        = d_input.get().data(),
                                                          .input_size   = input_size,
                                                          .soft_bits    = d_soft_bits.get().data(),
                                                          .check_to_var = d_check_to_var.get().data(),
                                                          .rm_buffer    = d_rm_buffer.get().data(),
                                                          .bg_info_pos  = skip + pos,
                                                          .message_size_bytes =
                                                              message_length.round_up_to_bytes().value(),
                                                          .nof_layers          = nof_layers,
                                                          .lifting_size        = lifting_size,
                                                          .bg_N_high_rate      = bg_N_high_rate,
                                                          .bg_N_full           = bg_N_full,
                                                          .max_nof_iterations  = cfg.max_iterations,
                                                          .nof_systematic_bits = nof_systematic_bits,
                                                          .nof_filler_bits     = nof_filler_bits,
                                                          .rm_buffer_size      = buffer_length,
                                                          .shift_k0            = shift_k0,
                                                          .modulation_order    = get_bits_per_symbol(cfg.modulation),
                                                          .new_data            = cfg.new_data};

  cuda_backend->decode(output.get_buffer(), input_promise, codeblock_config, std::move(callback));
}
