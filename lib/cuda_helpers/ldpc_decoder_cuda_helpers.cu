// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#include "ocuda-fec/cuda_helpers/ldpc_decoder_cuda_helpers.h"
#include "ocudu/phy/upper/channel_coding/ldpc/ldpc.h"
#include <cuda_runtime_api.h>

using namespace ocudu;
using namespace cuda;

void* ocudu::cuda::detail::cuda_malloc(size_t size)
{
  void*       ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, size);
  ocudu_assert(err == cudaSuccess, "CUDA malloc failed: {}", cudaGetErrorString(err));
  return ptr;
}

void ocudu::cuda::detail::cuda_free(void* ptr)
{
  if (ptr != nullptr) {
    cudaError_t err = cudaFree(ptr);
    ocudu_assert(err == cudaSuccess, "CUDA free failed: {}", cudaGetErrorString(err));
  }
}

void cuda::detail::cuda_host_to_device(void* dst, const void* src, size_t size, void* stream)
{
  if (stream != nullptr) {
    cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream));
    ocudu_assert(err == cudaSuccess, "CUDA memcpy async failed: {}", cudaGetErrorString(err));
  } else {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    ocudu_assert(err == cudaSuccess, "CUDA memcpy failed: {}", cudaGetErrorString(err));
  }
}

void cuda::detail::cuda_device_to_host(void* dst, const void* src, size_t size, void* stream)
{
  cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream));
  ocudu_assert(err == cudaSuccess, "CUDA memcpy failed: {}", cudaGetErrorString(err));
}

void cuda::detail::cuda_device_to_device(void* dst, const void* src, size_t size)
{
  cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
  ocudu_assert(err == cudaSuccess, "CUDA memcpy failed: {}", cudaGetErrorString(err));
}

void cuda::detail::cuda_device_zero(void* ptr, size_t size)
{
  cudaError_t err = cudaMemset(ptr, 0, size);
  ocudu_assert(err == cudaSuccess, "CUDA memset failed: {}", cudaGetErrorString(err));
}

void cuda::detail::cuda_device_memset(void* ptr, int value, size_t size)
{
  cudaError_t err = cudaMemset(ptr, value, size);
  ocudu_assert(err == cudaSuccess, "CUDA memset failed: {}", cudaGetErrorString(err));
}

std::string ocudu::cuda::cuda_get_device_name()
{
  int device_count;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    return "";
  }

  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, 0);
  return std::string(device_prop.name);
}
