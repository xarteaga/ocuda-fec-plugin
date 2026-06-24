# SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
# SPDX-License-Identifier: BSD-3-Clause-Open-MPI

include_directories(${OCUDA_FEC_SOURCE_DIR}/include)

add_library(ocuda_cuda_traces STATIC
        l1_cuda_traces.cpp)
add_to_exported_libs(ocuda_cuda_traces)
target_link_libraries(ocuda_cuda_traces PRIVATE CUDA::cudart ocudu_support)