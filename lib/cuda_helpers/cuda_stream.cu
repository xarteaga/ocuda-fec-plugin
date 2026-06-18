// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#include "../../include/ocuda-fec/cuda_helpers/cuda_stream.h"
#include "ocudu/support/ocudu_assert.h"
#include <driver_types.h>

using namespace ocudu::cuda;

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
  ocudu_assert(err == cudaSuccess, "CUDA stream synchronize failed: {}", cudaGetErrorString(err));
}

bool detail::cuda_stream_is_busy(void* stream)
{
  if (stream == nullptr) {
    return false;
  }
  return cudaStreamQuery(reinterpret_cast<cudaStream_t>(stream)) != cudaSuccess;
}
