# SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
# SPDX-License-Identifier: BSD-3-Clause-Open-MPI

include_directories(${OCUDA_FEC_SOURCE_DIR}/include)

add_library(ocuda_pusch_decoder STATIC
        factories.cpp
        pusch_codeblock_cuda_decoder.cpp
        pusch_decoder_cuda_impl.cpp)
set(OCUDU_PUSCH_LIBS ocudu_upper_phy_support ocudu_ran)

target_link_libraries(ocuda_pusch_decoder ${OCUDU_PUSCH_LIBS})
target_include_directories(ocuda_pusch_decoder PRIVATE ${CMAKE_SOURCE_DIR}/lib/phy/upper/channel_coding/ldpc/)
add_to_exported_libs(ocuda_pusch_decoder)

target_compile_definitions(ocudu_pusch_processor PRIVATE HAVE_PUSCH_DECODER_PLUGIN)
target_link_libraries(ocudu_pusch_processor ocuda_pusch_decoder)
