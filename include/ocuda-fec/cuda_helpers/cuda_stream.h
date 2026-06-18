// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.
#pragma once
#include "ocudu/support/ocudu_assert.h"

namespace ocudu::cuda {

namespace detail {

void* cuda_stream_create();
void  cuda_stream_sync(void* stream);
void  cuda_stream_destroy(void* stream);
bool  cuda_stream_is_busy(void* stream);

} // namespace detail

class cuda_stream
{
public:
  cuda_stream() : stream_ptr(detail::cuda_stream_create()) {}

  ~cuda_stream()
  {
    if (stream_ptr) {
      synchronize();
      detail::cuda_stream_destroy(stream_ptr);
    }
  }

  cuda_stream(const cuda_stream&)            = delete;
  cuda_stream& operator=(const cuda_stream&) = delete;

  cuda_stream(cuda_stream&& other) noexcept : stream_ptr(other.stream_ptr) { other.stream_ptr = nullptr; }

  cuda_stream& operator=(cuda_stream&& other) noexcept
  {
    if (this != &other) {
      detail::cuda_stream_destroy(stream_ptr);
      stream_ptr       = other.stream_ptr;
      other.stream_ptr = nullptr;
    }
    return *this;
  }

  void synchronize() { detail::cuda_stream_sync(stream_ptr); }

  bool is_idle() const { return !detail::cuda_stream_is_busy(stream_ptr); }

  void* get() const { return stream_ptr; }

private:
  void* stream_ptr = nullptr;
};

class stream_sync_token
{
public:
  stream_sync_token() = default;

  explicit stream_sync_token(cuda_stream& stream) : stream_ptr(&stream) {}

  stream_sync_token(const stream_sync_token&)            = delete;
  stream_sync_token& operator=(const stream_sync_token&) = delete;

  stream_sync_token(stream_sync_token&&)            = default;
  stream_sync_token& operator=(stream_sync_token&&) = default;

  ~stream_sync_token()
  {
    if (is_valid()) {
      synchronize();
    }
  }

  void synchronize()
  {
    if (is_valid()) {
      stream_ptr->synchronize();
      stream_ptr = nullptr;
    }
  }

  cuda_stream& get_device_stream() const
  {
    ocudu_assert(is_valid(), "Stream sync token is not valid.");
    return *stream_ptr;
  }

  bool is_valid() const { return stream_ptr != nullptr; }

  bool is_completed() const
  {
    ocudu_assert(is_valid(), "Stream sync token is not valid.");
    return stream_ptr->is_idle();
  }

private:
  cuda_stream* stream_ptr = nullptr;
};

} // namespace ocudu::cuda
