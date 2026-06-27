// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

/// \file
/// \brief LDPC decoder interface - CUDA backend.
///
/// Combines the functionality of \c ldpc_rate_dematcher (rate dematching and HARQ
/// soft combining) and \c ldpc_decoder (iterative LDPC message-passing decoding and
/// CRC verification) into a single GPU-accelerated per-codeblock decoding step.

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

/// \brief LDPC decoder interface - CUDA backend.
///
/// Encapsulates the full per-codeblock LDPC decoding chain on GPU: rate dematching (reversing the rate matching
/// performed by \c ldpc_rate_matcher and performing HARQ chase combining or incremental redundancy), soft-bit loading,
/// iterative LDPC belief-propagation decoding, hard decision, and CRC verification.
class ldpc_decoder_cuda
{
public:
  /// Default virtual destructor.
  virtual ~ldpc_decoder_cuda() = default;

  /// Decoder configuration.
  struct configuration {
    /// Modulation scheme used for soft-bit loading.
    modulation_scheme modulation;
    /// Redundancy version — selects which rate-matched chunks are retained.
    unsigned rv;
    /// Set to \c true to reset the rate matching buffer (new transmission); set to \c false to accumulate with prior
    /// transmissions (HARQ combining).
    bool new_data;
    /// Code base graph (BG1 or BG2).
    ldpc_base_graph_type base_graph = ldpc_base_graph_type::BG1;
    /// Code lifting size.
    ldpc::lifting_size_t lifting_size = ldpc::LS2;
    /// Number of filler bits in the full codeblock.
    unsigned nof_filler_bits = 0;
    /// Maximum number of LDPC belief-propagation iterations.
    unsigned max_iterations = 6;
    /// Encoded codeblock length in bits (after rate matching).
    unsigned block_length;
    /// Number of bits in the reference codeblock (i.e. number of rows in the parity-check matrix H).
    unsigned Nref;
  };

  /// \brief Decode a single code block on the GPU.
  ///
  /// Executes the full decoding chain asynchronously on the GPU: rate dematching (reversing rate matching and
  /// performing HARQ chase combining or incremental redundancy) → soft-bit loading → iterative LDPC message-passing →
  /// hard decision → CRC verification.
  ///
  /// \param[out] output       Bit buffer for the decoded code block (information bits).
  /// \param[in]  input        LLRs for the rate-matched code block.
  /// \param[in]  cfg          LDPC decoder configuration (modulation, RV, base graph, etc.).
  /// \param[in]  callback     Completion callback invoked when decoding finishes.
  /// \param[in]  last_codeblock Set to \c true when this is the last codeblock of the reception, triggering
  ///                           immediate synchronous dispatch of any pending batch.
  virtual void decode(bit_buffer&                       output,
                      span<const log_likelihood_ratio>  input,
                      const configuration&              cfg,
                      cuda_ldpc_decoder_callback_func&& callback,
                      bool                              last_codeblock = false) = 0;
};

/// \brief Create an LDPC decoder for a CUDA backend.
///
/// \param[in] backend  CUDA decoder backend.
std::unique_ptr<ldpc_decoder_cuda> create_ldpc_decoder_cuda(std::shared_ptr<cuda_ldpc_decoder_backend> backend);

} // namespace ocudu
