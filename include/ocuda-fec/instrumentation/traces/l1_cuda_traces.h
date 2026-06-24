// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#pragma once

#include "ocudu/support/tracing/rusage_trace_recorder.h"

namespace ocudu {

/// Set to true for enabling CUDA trace.
#if defined(OCUDU_L1_DL_TRACE) || defined(OCUDU_L1_UL_TRACE)
constexpr bool L1_CUDA_TRACE_ENABLED = true;
#else
constexpr bool L1_CUDA_TRACE_ENABLED = false;
#endif

/// L1 CUDA event tracing. This tracer is used to analyze latencies in the L1 CUDA processing.
extern file_event_tracer<L1_CUDA_TRACE_ENABLED> l1_cuda_tracer;

} // namespace ocudu
