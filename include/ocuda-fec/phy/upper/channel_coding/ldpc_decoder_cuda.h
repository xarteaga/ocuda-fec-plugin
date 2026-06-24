// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

/// \file
/// \brief LDPC decoder interface - CUDA backend.

#pragma once

#include "ocuda-fec/phy/upper/channel_coding/ldpc_decoder_cuda_backend.h"
#include "ocudu/adt/bit_buffer.h"
#include "ocudu/phy/upper/channel_coding/ldpc/ldpc.h"
#include "ocudu/phy/upper/log_likelihood_ratio.h"
#include "ocudu/ran/sch/ldpc_base_graph.h"
#include "ocudu/ran/sch/modulation_scheme.h"
#include <memory>

namespace ocudu {

class cuda_ldpc_decoder_backend;

/// LDPC decoder interface - CUDA backend.
class ldpc_decoder_cuda
{
public:
  virtual ~ldpc_decoder_cuda() = default;

  /// Decoder configuration.
  struct configuration {
    /// Modulation.
    modulation_scheme modulation;
    /// Redundancy version.
    unsigned rv;
    /// Set to true for resetting the rate matching buffer.
    bool new_data;
    /// Code base graph.
    ldpc_base_graph_type base_graph = ldpc_base_graph_type::BG1;
    /// Code lifting size.
    ldpc::lifting_size_t lifting_size = ldpc::LS2;
    /// Number of filler bits in the full codeblock.
    unsigned nof_filler_bits = 0;
    /// Maximum number of iterations.
    unsigned max_iterations = 6;
    ///
    unsigned block_length;
    ///
    unsigned Nref;
  };

  /// \brief Decode a single code block.
  ///
  /// \param[out] output   Bit buffer for the decoded code block.
  /// \param[in]  input    LLRs for the code block.
  /// \param[in]  cfg      LDPC decoder configuration.
  /// \param[in]  callback Completion callback.
  virtual void decode(bit_buffer&                       output,
                      span<const log_likelihood_ratio>  input,
                      const configuration&              cfg,
                      cuda_ldpc_decoder_callback_func&& callback) = 0;
};

/// \brief Create an LDPC decoder for a CUDA backend.
///
/// \param[in] backend  CUDA decoder backend.
std::unique_ptr<ldpc_decoder_cuda> create_ldpc_decoder_cuda(std::shared_ptr<cuda_ldpc_decoder_backend> backend);

} // namespace ocudu
