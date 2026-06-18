# SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
# SPDX-License-Identifier: BSD-3-Clause-Open-MPI

option(ENABLE_CUDA "Enable CUDA" ON)

set(OCUDA_FEC_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Find CUDA Toolkit
if (ENABLE_CUDA)
    find_package(CUDAToolkit)
    sbom_add(
            PACKAGE "CUDA"
            DOWNLOAD_LOCATION "https://developer.nvidia.com/cuda-toolkit"
            SUPPLIER "NVIDIA Corporation"
            LICENSE "CUDA License"
            VERSION ${CUDAToolkit_VERSION}
    )
endif (ENABLE_CUDA)

if (ENABLE_CUDA AND CUDAToolkit_FOUND)
    add_subdirectory(lib)
endif (ENABLE_CUDA AND CUDAToolkit_FOUND)