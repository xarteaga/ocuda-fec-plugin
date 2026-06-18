// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.
#pragma once

#include "cuda_stream.h"
#include "ldpc_decoder_cuda_helpers.h"
#include "ocudu/adt/span.h"

namespace ocudu {
namespace cuda {

template <typename Type>
class host_to_device_promise
{
public:
  host_to_device_promise() = default;

  host_to_device_promise(span<Type> device_data_, span<const Type> host_data_) :
    device_data(device_data_), host_data(host_data_)
  {
    ocudu_ocuduvec_assert_size(host_data, device_data);
  }

  host_to_device_promise(const host_to_device_promise&)            = default;
  host_to_device_promise& operator=(const host_to_device_promise&) = default;
  host_to_device_promise(host_to_device_promise&&)                 = default;
  host_to_device_promise& operator=(host_to_device_promise&&)      = default;

  /// Default destructor - will wait for the transfer to complete.
  ~host_to_device_promise() = default;

  void transfer(cuda_stream& stream)
  {
    ocudu_assert(!host_data.empty(), "Host data must not be empty.");
    ocudu_assert(!device_data.empty(), "Device data must not be empty.");

    copy_host_to_device(device_data, host_data, stream);
  }

private:
  span<Type>       device_data;
  span<const Type> host_data;
};

} // namespace cuda
} // namespace ocudu
