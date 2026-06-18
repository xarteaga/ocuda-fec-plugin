// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.
#pragma once

#include "ldpc_graph_impl.h"
#include "ocuda-fec/cuda_helpers/cuda_stream.h"
#include "ocuda-fec/cuda_helpers/device_vector.h"
#include "ocuda-fec/cuda_helpers/host_to_device_promise.h"
#include "ocuda-fec/cuda_helpers/ldpc_decoder.h"
#include "ocuda-fec/cuda_helpers/ldpc_decoder_cuda_helpers.h"
#include "ocudu/ocuduvec/copy.h"
#include "ocudu/phy/upper/channel_coding/ldpc/ldpc.h"
#include "ocudu/support/executors/task_worker.h"
#include "ocudu/support/memory_pool/bounded_object_pool.h"

namespace ocudu {

using cuda_ldpc_decoder_callback_func = unique_function<void(), default_unique_function_buffer_size, true>;

class cuda_ldpc_decoder_batch
{
public:
  cuda_ldpc_decoder_batch(span<const cuda::base_graph_description> base_graph_descriptions) :
    d_common_output(0), decoder(cuda::ldpc_decoder::create(base_graph_descriptions))
  {
  }

  bool push_back(span<uint8_t>                              output,
                 cuda::host_to_device_promise<int8_t>       input_promise,
                 const cuda::ldpc_decoder_cb_configuration& codeblock,
                 cuda_ldpc_decoder_callback_func&&          callback)
  {
    unsigned output_offset = d_common_output.size();
    unsigned output_size   = output.size();

    d_common_output.resize(output_offset + output_size);

    output_buffers.emplace_back(output);
    input_promises.emplace_back(input_promise);
    codeblocks.emplace_back(cuda::ldpc_decoder_cb_arguments{.output = d_common_output.get().last(output_size).data(),
                                                            .cb_configuration = codeblock});
    callbacks.emplace_back(std::move(callback));

    return codeblocks.full();
  }

  void sequential_decode()
  {
    // Transfer input codeblocks from the host to the device.
    for (auto& input_promise : input_promises) {
      input_promise.transfer(stream);
    }

    // Request asynchronous decode.
    decoder->ldpc_decode(codeblocks, stream);

    // Transfer output bits from the device to the host.
    h_common_output.resize(d_common_output.size());
    cuda::copy_device_to_host(span<uint8_t>(h_common_output), d_common_output.get_const(), stream);
  }

  void synchronize() { stream.synchronize(); }

  bool check_and_complete()
  {
    if (!stream.is_idle()) {
      return false;
    }

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

    return true;
  }

private:
  cuda::device_vector<uint8_t, cuda::ldpc_decoder::max_nof_codeblocks * ldpc::MAX_CODEBLOCK_SIZE> d_common_output;
  static_vector<uint8_t, cuda::ldpc_decoder::max_nof_codeblocks * ldpc::MAX_CODEBLOCK_SIZE>       h_common_output;
  static_vector<span<uint8_t>, cuda::ldpc_decoder::max_nof_codeblocks>                            output_buffers;
  static_vector<cuda::host_to_device_promise<int8_t>, cuda::ldpc_decoder::max_nof_codeblocks>     input_promises;
  static_vector<cuda::ldpc_decoder_cb_arguments, cuda::ldpc_decoder::max_nof_codeblocks>          codeblocks;
  static_vector<cuda_ldpc_decoder_callback_func, cuda::ldpc_decoder::max_nof_codeblocks>          callbacks;
  cuda::cuda_stream                                                                               stream;
  std::unique_ptr<cuda::ldpc_decoder>                                                             decoder;
};

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

class cuda_ldpc_decoder_asynchronous_backend : public cuda_ldpc_decoder_backend
{
public:
  cuda_ldpc_decoder_asynchronous_backend(task_executor& executor_) :
    executor(executor_), decoder_pool(nof_streams, d_bg_info.get())
  {
  }

  void decode(span<uint8_t>                              output,
              cuda::host_to_device_promise<int8_t>       input_promise,
              const cuda::ldpc_decoder_cb_configuration& codeblock,
              cuda_ldpc_decoder_callback_func&&          callback) override
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
        local_decoder = std::move(current_decoder);
      }
    }

    if (local_decoder) {
      local_decoder->sequential_decode();
      wait_decode_complete(std::move(local_decoder));
    } else {
      timed_decode(current_codeblock_count, std::chrono::system_clock::now() + std::chrono::microseconds(10));
    }
  }

private:
  using cuda_ldpc_decoder_batch_pool = bounded_object_pool<cuda_ldpc_decoder_batch>;

  void timed_decode(unsigned count, std::chrono::system_clock::time_point timeout_time)
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

      local_decoder = std::move(current_decoder);
    }

    if (local_decoder) {
      local_decoder->sequential_decode();
      wait_decode_complete(std::move(local_decoder));
    }
  }

  void wait_decode_complete(cuda_ldpc_decoder_batch_pool::ptr decoder)
  {
    if (!decoder) {
      return;
    }

    report_error_if_not(executor.defer([this, local_decoder = std::move(decoder)]() mutable {
      if (local_decoder->check_and_complete()) {
        return;
      }

      wait_decode_complete(std::move(local_decoder));
    }),
                        "Error deferring CUDA decoder wait asynchronous task.");
  }

  static constexpr unsigned nof_streams = 128;

  task_executor& executor;
  std::mutex     backend_mutex;

  cuda_ldpc_decoder_batch_pool      decoder_pool;
  cuda_ldpc_decoder_batch_pool::ptr current_decoder;
  std::atomic<uint64_t>             codeblock_count = {0};
};

} // namespace ocudu
