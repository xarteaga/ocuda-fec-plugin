// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.
#pragma once

#include "ocuda-fec/phy/upper/channel_coding/ldpc_decoder_cuda_backend.h"
#include "ocudu/support/executors/task_worker.h"
#include "ocudu/support/memory_pool/bounded_object_pool.h"
#include <atomic>
#include <chrono>
#include <mutex>

namespace ocudu {

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

class cuda_ldpc_decoder_asynchronous_backend : public cuda_ldpc_decoder_backend
{
public:
  explicit cuda_ldpc_decoder_asynchronous_backend(task_executor& executor_);

  void decode(span<uint8_t>                              output,
              cuda::host_to_device_promise<int8_t>       input_promise,
              const cuda::ldpc_decoder_cb_configuration& codeblock,
              cuda_ldpc_decoder_callback_func&&          callback) override;

private:
  using cuda_ldpc_decoder_batch_pool = bounded_object_pool<cuda_ldpc_decoder_batch>;

  void timed_decode(unsigned count, std::chrono::system_clock::time_point timeout_time);
  void wait_decode_complete(cuda_ldpc_decoder_batch_pool::ptr decoder);

  static constexpr unsigned nof_streams = 128;

  task_executor& executor;
  std::mutex     backend_mutex;

  cuda_ldpc_decoder_batch_pool      decoder_pool;
  cuda_ldpc_decoder_batch_pool::ptr current_decoder;
  std::atomic<uint64_t>             codeblock_count = {0};
};

} // namespace ocudu
