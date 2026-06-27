// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#pragma once

#include "ldpc_graph_impl.h"
#include "ocuda-fec/phy/upper/channel_coding/ldpc_decoder_cuda.h"
#include "ocuda-fec/phy/upper/channel_coding/ldpc_decoder_cuda_backend.h"
#include "ocudu/adt/bit_buffer.h"

namespace ocudu {

/// \brief LDPC decoder implementation - CUDA backend.
class ldpc_decoder_cuda_impl : public ldpc_decoder_cuda
{
public:
  /// Maximum number of information bits in a codeblock (before shortening).
  static constexpr unsigned MAX_BG_K = 22;

  /// \brief Maximum degree of a check node.
  ///
  /// In the base graph, each check node is connected, at most, to all variable nodes in the high-rate region
  /// (of maximum length max_BG_K + 4) and an extra variable node in the extension region.
  static constexpr unsigned MAX_CHECK_NODE_DEGREE = MAX_BG_K + 5;

  explicit ldpc_decoder_cuda_impl(std::shared_ptr<cuda_ldpc_decoder_backend> backend_);

  // See the interface ldpc_decoder_cuda for documentation.
  void decode(bit_buffer&                       output,
              span<const log_likelihood_ratio>  input,
              const configuration&              cfg,
              cuda_ldpc_decoder_callback_func&& callback,
              bool                              last_codeblock) override;

private:
  std::shared_ptr<cuda_ldpc_decoder_backend> cuda_backend;
  /// Number of significant (not filler) bits.
  uint16_t nof_significant_bits = 44;

  /// Buffer to store the current value of the soft bits.
  cuda::device_vector<int8_t, static_cast<size_t>(ldpc::MAX_BG_N_FULL* ldpc::MAX_LIFTING_SIZE)> d_input;

  /// Buffer to store the current value of the soft bits.
  cuda::device_vector<int8_t, static_cast<size_t>(ldpc::MAX_BG_N_FULL* ldpc::MAX_LIFTING_SIZE)> d_soft_bits;

  /// Buffer to store the current value of the check-to-variable messages.
  cuda::device_vector<int8_t, static_cast<size_t>(MAX_CHECK_NODE_DEGREE* ldpc::MAX_LIFTING_SIZE* ldpc::MAX_BG_M)>
      d_check_to_var;

  /// Temporal rate matching buffer.
  cuda::device_vector<int8_t, static_cast<size_t>(ldpc::MAX_BG_N_FULL* ldpc::MAX_LIFTING_SIZE)> d_rm_buffer;
};

} // namespace ocudu
