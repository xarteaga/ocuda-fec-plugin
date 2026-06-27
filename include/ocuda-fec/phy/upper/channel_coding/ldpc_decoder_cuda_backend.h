// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

/// \file
/// \brief LDPC decoder backend - CUDA interface.

#pragma once

#include "ocuda-fec/cuda_helpers/device_vector.h"
#include "ocuda-fec/cuda_helpers/host_to_device_promise.h"
#include "ocuda-fec/cuda_helpers/ldpc_decoder.h"
#include "ocudu/adt/unique_function.h"
#include "ocudu/ocuduvec/copy.h"
#include "ocudu/phy/upper/channel_coding/ldpc/ldpc.h"
#include "ocudu/support/executors/task_executor.h"

namespace ocudu {

using cuda_ldpc_decoder_callback_func = unique_function<void(), default_unique_function_buffer_size, true>;

/// \brief Base class for CUDA LDPC decoder backends.
///
/// Manages LDPC base graph descriptions stored on the GPU and provides the virtual decode interface. Derived classes
/// implement the actual GPU decoding execution model.
class cuda_ldpc_decoder_backend
{
public:
  /// Number of base graphs (two per lifting size: BG1 and BG2).
  static constexpr unsigned TOTAL_NOF_GRAPHS = ldpc::NOF_LIFTING_SIZES * 2;

  /// Default constructor - load the backend necessary LDPC tables.
  cuda_ldpc_decoder_backend();

  /// Default destructor.
  virtual ~cuda_ldpc_decoder_backend() = default;

  /// \brief Decode a single code block on the GPU.
  ///
  /// Executes the full decoding chain on the GPU: rate dematching, soft-bit loading, iterative LDPC message-passing,
  /// hard decision, and CRC verification.
  ///
  /// \param[out] output            Output bit buffer.
  /// \param[in]  input_promise     Deferred host-to-device transfer for LLR input.
  /// \param[in]  codeblock         Codeblock configuration.
  /// \param[in]  callback          Completion callback invoked when decoding finishes.
  /// \param[in]  last_codeblock    Set to \c true when this is the last codeblock of the reception, triggering
  ///                               immediate synchronous dispatch of any pending batch.
  virtual void decode(span<uint8_t>                              output,
                      cuda::host_to_device_promise<int8_t>       input_promise,
                      const cuda::ldpc_decoder_cb_configuration& codeblock,
                      cuda_ldpc_decoder_callback_func&&          callback,
                      bool                                       last_codeblock = true) = 0;

protected:
  /// Base graph descriptions for each lifting size and base graph.
  cuda::device_vector<cuda::base_graph_description, TOTAL_NOF_GRAPHS> d_bg_info;
};

/// \brief Create an asynchronous LDPC decoder backend.
///
/// Returns a backend backed by a pool of 128 CUDA streams with deferred decode scheduling. Suitable for concurrent
/// processing of multiple codeblocks in a PUSCH decoder.
///
/// \param[in] executor_  Task executor used to invoke completion callbacks on the host thread pool.
std::shared_ptr<cuda_ldpc_decoder_backend> create_asynchronous_backend(task_executor& executor_);

} // namespace ocudu
