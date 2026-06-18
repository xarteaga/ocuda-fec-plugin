// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#pragma once

#include "cuda_stream.h"
#include "ldpc_decoder_cuda_helpers.h"
#include "ocudu/adt/span.h"

namespace ocudu {
namespace cuda {

template <typename Type, size_t Capacity>
class device_vector
{
public:
  explicit device_vector() : device_vector(Capacity) {}

  explicit device_vector(size_t new_size) : current_size(new_size)
  {
    ocudu_assert(current_size <= Capacity, "Device vector capacity exceeded.");
    ptr = static_cast<Type*>(detail::cuda_malloc(Capacity * sizeof(Type)));
  }
  explicit device_vector(span<const Type> host_data) : current_size(host_data.size())
  {
    ocudu_assert(current_size <= Capacity, "Device vector capacity exceeded.");
    ptr = static_cast<Type*>(detail::cuda_malloc(current_size * sizeof(Type)));
    copy_to_device(host_data);
  }

  ~device_vector()
  {
    stream.synchronize();
    detail::cuda_free(ptr);
  }

  void resize(size_t new_size)
  {
    ocudu_assert(new_size <= Capacity,
                 fmt::format("Device vector capacity exceeded: requested size={} but capacity={}", new_size, Capacity));
    current_size = new_size;
  }

  span<Type> get() { return {ptr, current_size}; }

  span<const Type> get() const { return {ptr, current_size}; }

  span<const Type> get_const() const { return {ptr, current_size}; }

  void copy_to_device(span<const Type> src, unsigned offset = 0)
  {
    ocudu_assert(offset + src.size() <= current_size, "Host data source plus the offset exceeds the vector capacity.");
    detail::cuda_host_to_device(
        ptr + offset, reinterpret_cast<const void*>(src.data()), src.size() * sizeof(Type), nullptr);
  }

  stream_sync_token copy_to_device_async(span<const Type> src, unsigned offset = 0)
  {
    ocudu_assert(offset + src.size() <= current_size, "Host data source plus the offset exceeds the vector capacity.");
    detail::cuda_host_to_device(
        ptr + offset, reinterpret_cast<const void*>(src.data()), src.size() * sizeof(Type), stream.get());
    return stream_sync_token(stream);
  }

  void synchronize() { stream.synchronize(); }

  size_t size() { return current_size; }

private:
  cuda_stream stream;
  Type*       ptr;
  size_t      current_size;
};

} // namespace cuda
} // namespace ocudu
