// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#include "ldpc_decoder_cuda_asynchronous_backend.h"
#include "ldpc_graph_impl.h"
#include <utility>

#include "ocudu/support/executors/task_worker.h"

using namespace ocudu;

cuda_ldpc_decoder_asynchronous_backend::cuda_ldpc_decoder_asynchronous_backend(task_executor& executor_) :
  executor(executor_), decoder_pool(nof_streams, std::reference_wrapper(executor), d_bg_info.get())
{
}

void cuda_ldpc_decoder_asynchronous_backend::decode(span<uint8_t>                              output,
                                                    cuda::host_to_device_promise<int8_t>       input_promise,
                                                    const cuda::ldpc_decoder_cb_configuration& codeblock,
                                                    cuda_ldpc_decoder_callback_func&&          callback)
{
  cuda_ldpc_decoder_batch_pool::ptr local_decoder;
  unsigned                          current_codeblock_count;
  {
    // Protect concurrent access to codeblocks and callbacks.
    std::unique_lock lock(backend_mutex);

    // Increment codeblock count.
    current_codeblock_count = codeblock_count.fetch_add(1) + 1;

    // Get a new stream if there is no current stream assigned.
    if (!current_decoder) {
      current_decoder = decoder_pool.get();
      report_fatal_error_if_not(current_decoder, "Failed to obtain LDPC decoder CUDA stream");
    }

    // Push the codeblock to the batch decoder.
    if (current_decoder->push_back(output, input_promise, codeblock, std::move(callback))) {
      l1_cuda_tracer << instant_trace_event{"ldpc_dispatch", instant_trace_event::cpu_scope::thread};
      local_decoder = std::exchange(current_decoder, nullptr);
    }
  }

  if (local_decoder) {
    cuda_ldpc_decoder_batch& decoder = *local_decoder;
    decoder.start_asynch_decoding([token = std::move(local_decoder)]() {});
  } else {
    timed_decode(current_codeblock_count, std::chrono::system_clock::now() + std::chrono::microseconds(10));
  }
}

void cuda_ldpc_decoder_asynchronous_backend::timed_decode(unsigned                              count,
                                                          std::chrono::system_clock::time_point timeout_time)
{
  cuda_ldpc_decoder_batch_pool::ptr local_decoder;
  {
    std::unique_lock lock(backend_mutex);
    if (count != codeblock_count.load()) {
      return;
    }

    auto now = std::chrono::system_clock::now();
    if (now < timeout_time) {
      bool success = executor.defer([this, count, timeout_time]() { timed_decode(count, timeout_time); });
      report_error_if_not(success,
                          "Error deferring CUDA decoder enqueue timeout task (remaining {:.3f}us).",
                          static_cast<double>((timeout_time - now).count()) / 1e3);
      return;
    }

    l1_cuda_tracer << instant_trace_event{"ldpc_dispatch_timeout", instant_trace_event::cpu_scope::thread};
    local_decoder = std::exchange(current_decoder, nullptr);
  }

  if (local_decoder) {
    cuda_ldpc_decoder_batch& decoder = *local_decoder;
    decoder.start_asynch_decoding([token = std::move(local_decoder)]() {});
  }
}
