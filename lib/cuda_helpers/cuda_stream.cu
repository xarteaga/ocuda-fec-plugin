// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#include "../../include/ocuda-fec/cuda_helpers/cuda_stream.h"
#include "ocudu/support/ocudu_assert.h"
#include <driver_types.h>

using namespace ocudu::cuda;

// Registered with the CUDA runtime via cudaStreamAddCallback. Signature matches cudaStreamCallback_t: (cudaStream_t
// stream, cudaError_t status, void* user_data).
void cuda_stream_complete_callback(void* user_data)
{
  auto* stream_obj = static_cast<cuda_stream*>(user_data);
  if (stream_obj) {
    stream_obj->on_complete();
  }
}

cuda_stream::cuda_stream()
{
  stream_ptr = detail::cuda_stream_create();
}

void cuda_stream::set_callback_next_complete(cuda_stream_callback&& cb)
{
  // Store the callback so on_complete() can invoke it.
  completion_cb = std::move(cb);

  // Register with the CUDA runtime.  The callback fires once per call — the caller must re-arm it by calling
  // set_callback_next_complete() again from within on_complete() after scheduling new work.
  if (stream_ptr && !completion_cb.is_empty()) {
    cudaError_t err =
        cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(stream_ptr), &cuda_stream_complete_callback, this);
    report_error_if_not(err == cudaSuccess, "cudaStreamAddCallback failed: {}.", cudaGetErrorString(err));
  }
}

void* detail::cuda_stream_create()
{
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  return reinterpret_cast<void*>(stream);
}

void detail::cuda_stream_destroy(void* stream)
{
  cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream));
}

void detail::cuda_stream_sync(void* stream)
{
  cudaError_t err = cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
  report_error_if_not(err == cudaSuccess, "CUDA stream synchronize failed: {}", cudaGetErrorString(err));
}

bool detail::cuda_stream_is_busy(void* stream)
{
  if (stream == nullptr) {
    return false;
  }
  return cudaStreamQuery(reinterpret_cast<cudaStream_t>(stream)) != cudaSuccess;
}
