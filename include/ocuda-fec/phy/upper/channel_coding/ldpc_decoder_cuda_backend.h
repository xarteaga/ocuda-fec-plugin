// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.
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

class cuda_ldpc_decoder_backend
{
public:
  static constexpr unsigned TOTAL_NOF_GRAPHS = ldpc::NOF_LIFTING_SIZES * 2;

  cuda_ldpc_decoder_backend();

  virtual ~cuda_ldpc_decoder_backend() = default;

  virtual void decode(span<uint8_t>                              output,
                      cuda::host_to_device_promise<int8_t>       input_promise,
                      const cuda::ldpc_decoder_cb_configuration& codeblock,
                      cuda_ldpc_decoder_callback_func&&          callback) = 0;

protected:
  cuda::device_vector<cuda::base_graph_description, TOTAL_NOF_GRAPHS> d_bg_info;
};

std::shared_ptr<cuda_ldpc_decoder_backend> create_asynchronous_backend(task_executor& executor_);

} // namespace ocudu
