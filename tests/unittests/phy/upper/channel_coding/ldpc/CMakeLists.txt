# SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
# SPDX-License-Identifier: BSD-3-Clause-Open-MPI

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/tests/test_doubles)
include_directories(${OCUDA_FEC_SOURCE_DIR}/include)

add_executable(ldpc_chain_cuda_test ldpc_chain_cuda_test.cpp)

target_link_libraries(ldpc_chain_cuda_test
        ocudu_cuda_ldpc
        ocudu_channel_coding
        ocudu_support
        GTest::gtest
        GTest::gtest_main)

add_test(ldpc_chain_cuda_test ldpc_chain_cuda_test)