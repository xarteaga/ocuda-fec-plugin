// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.
#pragma once

#include "cuda_stream.h"
#include "ocudu/adt/span.h"
#include "ocudu/ocuduvec/types.h"

namespace ocudu {
namespace cuda {

namespace detail {

void* cuda_malloc(size_t size);
void  cuda_free(void* ptr);
void  cuda_host_to_device(void* dst, const void* src, size_t size, void* stream);
void  cuda_device_to_host(void* dst, const void* src, size_t size, void* stream);
void  cuda_device_to_device(void* dst, const void* src, size_t size);
void  cuda_device_zero(void* ptr, size_t size);
void  cuda_device_memset(void* ptr, int value, size_t size);

} // namespace detail

std::string cuda_get_device_name();

template <typename Type>
void copy_device_to_device(span<Type> dst, span<const Type> src)
{
  ocudu_ocuduvec_assert_size(dst, src);

  detail::cuda_device_to_device(dst.data(), reinterpret_cast<const void*>(src.data()), dst.size() * sizeof(Type));
}

template <typename Type>
void copy_device_to_host(span<Type> dst, span<const Type> src, cuda_stream& stream)
{
  ocudu_ocuduvec_assert_size(dst, src);
  detail::cuda_device_to_host(
      dst.data(), reinterpret_cast<const void*>(src.data()), dst.size() * sizeof(Type), stream.get());
}

template <typename Type>
void copy_host_to_device(span<Type> dst, span<const Type> src, cuda_stream& stream)
{
  ocudu_ocuduvec_assert_size(dst, src);
  detail::cuda_host_to_device(
      dst.data(), reinterpret_cast<const void*>(src.data()), dst.size() * sizeof(Type), stream.get());
}

template <typename Type>
void zero(span<Type> data)
{
  detail::cuda_device_zero(data.data(), data.size() * sizeof(Type));
}

template <typename Type>
void fill(span<Type> data, Type value)
{
  static_assert(sizeof(Type) == 1, "Type must be one byte.");
  detail::cuda_device_memset(data.data(), value, data.size() * sizeof(Type));
}

} // namespace cuda
} // namespace ocudu
