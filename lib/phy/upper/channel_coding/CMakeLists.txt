# SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
# SPDX-License-Identifier: BSD-3-Clause-Open-MPI

include_directories(${OCUDA_FEC_SOURCE_DIR}/include)

list(APPEND ldpc_cuda_link_libraries ocuda_ldpc_decoder)

add_library(ocudu_cuda_ldpc STATIC
        ldpc_decoder_cuda.cpp
        ldpc_decoder_cuda_backend.cpp)

target_include_directories(ocudu_cuda_ldpc PRIVATE ${CMAKE_SOURCE_DIR}/lib/phy/upper/channel_coding/ldpc/)

target_link_libraries(ocudu_cuda_ldpc ocuda_ldpc_decoder)
target_compile_definitions(ocudu_cuda_ldpc PUBLIC ${ldpc_definitions})

target_link_libraries(ocudu_ldpc ocudu_cuda_ldpc)
target_compile_definitions(ocudu_ldpc PUBLIC HAVE_LDPC_DECODER_PLUGIN)

add_to_exported_libs(ocudu_cuda_ldpc)
