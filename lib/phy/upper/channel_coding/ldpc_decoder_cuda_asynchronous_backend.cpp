// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#include "ldpc_decoder_cuda_asynchronous_backend.h"
#include "ocudu/support/executors/task_worker.h"
#include <utility>

using namespace ocudu;

cuda_ldpc_decoder_asynchronous_backend::cuda_ldpc_decoder_asynchronous_backend(task_executor& executor_) :
  decoder_pool(nof_streams, std::reference_wrapper(executor_), d_bg_info.get())
{
}

void cuda_ldpc_decoder_asynchronous_backend::decode(span<uint8_t>                              output,
                                                    cuda::host_to_device_promise<int8_t>       input_promise,
                                                    const cuda::ldpc_decoder_cb_configuration& codeblock,
                                                    cuda_ldpc_decoder_callback_func&&          callback,
                                                    bool                                       last_codeblock)
{
  cuda_ldpc_decoder_batch_pool::ptr local_decoder;
  {
    std::unique_lock lock(backend_mutex);

    // Get a new stream if there is no current stream assigned.
    if (!current_decoder) {
      current_decoder = decoder_pool.get();
      report_fatal_error_if_not(current_decoder, "Failed to obtain LDPC decoder CUDA stream");
    }

    // Push the codeblock to the batch decoder.
    bool batch_is_full = current_decoder->push_back(output, input_promise, codeblock, std::move(callback));

    // Dispatch the batch when it is full or when this is the last codeblock.
    if (batch_is_full || last_codeblock) {
      l1_cuda_tracer << instant_trace_event{"ldpc_dispatch", instant_trace_event::cpu_scope::thread};
      local_decoder = std::exchange(current_decoder, nullptr);
    }
  }

  if (local_decoder) {
    dispatch_batch(std::move(local_decoder));
  }
}

void cuda_ldpc_decoder_asynchronous_backend::dispatch_batch(cuda_ldpc_decoder_batch_pool::ptr decoder_batch)
{
  cuda_ldpc_decoder_batch& batch = *decoder_batch;
  batch.start_asynch_decoding([token = std::move(decoder_batch)]() {});
}
