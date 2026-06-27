// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.
#pragma once

#include "ocuda-fec/instrumentation/traces/l1_cuda_traces.h"
#include "ocuda-fec/phy/upper/channel_coding/ldpc_decoder_cuda_backend.h"
#include "ocudu/support/executors/task_worker.h"
#include "ocudu/support/memory_pool/bounded_object_pool.h"
#include <mutex>

namespace ocudu {

/// \brief A single batch of codeblocks processed on one CUDA stream.
///
/// Holds up to \c max_nof_codeblocks (32) codeblocks and drives a four-phase callback-chained pipeline on a single
/// CUDA stream. No CPU thread is held waiting for the GPU — each phase defers device-accessing work to the executor
/// thread pool and arms a CUDA stream callback to trigger the next phase when the stream is idle:
///
/// 1. \c start_asynch_decoding — transfers LLR inputs (H2D) via deferred promises, then arms \c launch_decode_kernel.
/// 2. \c launch_decode_kernel — defers the \c ldpc_decode kernel launch, then arms \c unload_output.
/// 3. \c unload_output — defers the D2H output transfer, then arms \c complete.
/// 4. \c complete — defers per-codeblock result copying and callback invocation, then fires the external completion.
///
/// The batch is obtained from and returned to a \c bounded_object_pool via a smart-pointer token captured in the
/// final completion callback (the \c dispatch_batch token pattern).
class cuda_ldpc_decoder_batch
{
public:
  cuda_ldpc_decoder_batch(task_executor& executor_, span<const cuda::base_graph_description> base_graph_descriptions) :
    executor(executor_), d_common_output(0), decoder(cuda::ldpc_decoder::create(base_graph_descriptions))
  {
  }

  bool push_back(span<uint8_t>                              output,
                 cuda::host_to_device_promise<int8_t>       input_promise,
                 const cuda::ldpc_decoder_cb_configuration& codeblock,
                 cuda_ldpc_decoder_callback_func&&          callback_)
  {
    unsigned output_offset = d_common_output.size();
    unsigned output_size   = output.size();

    d_common_output.resize(output_offset + output_size);

    output_buffers.emplace_back(output);
    input_promises.emplace_back(input_promise);
    codeblocks.emplace_back(cuda::ldpc_decoder_cb_arguments{.output = d_common_output.get().last(output_size).data(),
                                                            .cb_configuration = codeblock});
    callbacks.emplace_back(std::move(callback_));

    return codeblocks.full();
  }

  void start_asynch_decoding(unique_task&& callback_)
  {
    const auto input_tp = l1_cuda_tracer.now();
    callback            = std::move(callback_);

    // Transfer input codeblocks from the host to the device.
    for (auto& input_promise : input_promises) {
      input_promise.transfer(stream);
    }

    // Register callback after all transfers so it fires after every piece of work.
    // cudaStreamAddCallback fires synchronously if the stream is idle at registration
    // time, so the callback must be the last thing enqueued on the stream.
    stream.set_callback_next_complete([this, input_tp]() {
      l1_cuda_tracer << trace_event{"ldpc_decode_cuda_input", input_tp};
      launch_decode_kernel();
    });
  }

private:
  void launch_decode_kernel()
  {
    report_error_if_not(executor.defer([this]() {
      const auto ldpc_decode_tp = l1_cuda_tracer.now();

      // Request asynchronous decode.
      decoder->ldpc_decode(codeblocks, stream);

      stream.set_callback_next_complete([this, ldpc_decode_tp]() {
        l1_cuda_tracer << trace_event{"ldpc_decode_cuda_decode", ldpc_decode_tp};
        unload_output();
      });
    }),
                        "Failed to defer LDPC decode kernel.");
  }

  void unload_output()
  {
    report_error_if_not(executor.defer([this]() {
      const auto output_tp = l1_cuda_tracer.now();

      // Transfer output bits from the device to the host.
      h_common_output.resize(d_common_output.size());
      cuda::copy_device_to_host(span<uint8_t>(h_common_output), d_common_output.get_const(), stream);

      stream.set_callback_next_complete([this, output_tp]() {
        l1_cuda_tracer << trace_event{"ldpc_decode_cuda_output", output_tp};
        complete();
      });
    }),
                        "Failed to defer LDPC output unload.");
  }

  void complete()
  {
    report_error_if_not(executor.defer([this]() {
      // Trace completion of the asynchronous task and create trace point for the completion.
      const auto complete_tp = l1_cuda_tracer.now();

      // For each codeblock invoke the callback.
      span<const uint8_t> h_output(h_common_output);
      for (unsigned i_cb = 0, nof_codeblocks = codeblocks.size(); i_cb != nof_codeblocks; ++i_cb) {
        unsigned output_size = output_buffers[i_cb].size();
        ocuduvec::copy(output_buffers[i_cb], h_output.first(output_size));

        h_output = h_output.last(h_output.size() - output_size);

        cuda_ldpc_decoder_callback_func callback_func = std::move(callbacks[i_cb]);
        callback_func();
      }

      // Clear queues.
      d_common_output.resize(0);
      output_buffers.clear();
      input_promises.clear();
      codeblocks.clear();
      callbacks.clear();

      if (!callback.is_empty()) {
        callback();
        callback = {};
      }

      // Trace completion task.
      l1_cuda_tracer << trace_event{"ldpc_decode_cuda_complete", complete_tp};
    }),
                        "Failed to defer LDPC completion task.");
  }

  task_executor&                                                                                  executor;
  cuda::device_vector<uint8_t, cuda::ldpc_decoder::max_nof_codeblocks * ldpc::MAX_CODEBLOCK_SIZE> d_common_output;
  static_vector<uint8_t, cuda::ldpc_decoder::max_nof_codeblocks * ldpc::MAX_CODEBLOCK_SIZE>       h_common_output;
  static_vector<span<uint8_t>, cuda::ldpc_decoder::max_nof_codeblocks>                            output_buffers;
  static_vector<cuda::host_to_device_promise<int8_t>, cuda::ldpc_decoder::max_nof_codeblocks>     input_promises;
  static_vector<cuda::ldpc_decoder_cb_arguments, cuda::ldpc_decoder::max_nof_codeblocks>          codeblocks;
  static_vector<cuda_ldpc_decoder_callback_func, cuda::ldpc_decoder::max_nof_codeblocks>          callbacks;
  cuda::cuda_stream                                                                               stream;
  std::unique_ptr<cuda::ldpc_decoder>                                                             decoder;
  unique_task                                                                                     callback;
};

/// \brief Asynchronous CUDA LDPC decoder backend.
///
/// Manages a pool of 128 CUDA streams, each backed by its own batch decoder. Codeblocks are distributed across the
/// pool: the first call allocates a batch, subsequent calls append to the current batch. When a batch fills (32
/// codeblocks) or the \c last_codeblock flag is set, the batch is immediately dispatched — no timeout fallback.
///
/// Dispatching starts a callback-chained pipeline on the batch's stream (see \c cuda_ldpc_decoder_batch for details).
/// The CPU never polls the GPU; completion is signalled entirely from CUDA stream callbacks that fire from the
/// runtime when each stream is idle.
class cuda_ldpc_decoder_asynchronous_backend : public cuda_ldpc_decoder_backend
{
public:
  explicit cuda_ldpc_decoder_asynchronous_backend(task_executor& executor_);

  // See the interface cuda_ldpc_decoder_backend for documentation.
  void decode(span<uint8_t>                              output,
              cuda::host_to_device_promise<int8_t>       input_promise,
              const cuda::ldpc_decoder_cb_configuration& codeblock,
              cuda_ldpc_decoder_callback_func&&          callback,
              bool                                       last_codeblock = false) override;

private:
  using cuda_ldpc_decoder_batch_pool = bounded_object_pool<cuda_ldpc_decoder_batch>;

  static constexpr unsigned nof_streams = 128;

  /// \brief Dispatch a batch to the GPU and transfer ownership of the decoder.
  ///
  /// Acquires the batch, starts the callback-chained pipeline on its stream, and moves the batch's smart-pointer into
  /// a completion callback token so the batch is returned to the pool on GPU completion.
  ///
  /// \param decoder_batch  Owned pointer to the batch to dispatch.
  static void dispatch_batch(cuda_ldpc_decoder_batch_pool::ptr decoder_batch);

  std::mutex backend_mutex;

  /// \brief Pool of CUDA-stream-backed batch decoders.
  ///
  /// Each batch is associated with one CUDA stream and can accumulate up to 32 codeblocks before dispatch.
  cuda_ldpc_decoder_batch_pool decoder_pool;

  /// Currently assigned stream — \c nullptr when no batch is queued.
  /// Protected by \c backend_mutex.
  cuda_ldpc_decoder_batch_pool::ptr current_decoder;
};

} // namespace ocudu
