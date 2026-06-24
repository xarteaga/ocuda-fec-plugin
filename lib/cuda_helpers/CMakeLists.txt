
project(ocuda_ldpc_decoder LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${OCUDA_FEC_SOURCE_DIR}/include)


# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Set CUDA architecture (adjust for your GPU)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86")  # Adjust based on your GPU
endif ()

add_library(ocuda_ldpc_decoder STATIC
        cuda_stream.cu
        ldpc_decoder_cuda_helpers.cu
        sch_decoder.cu)
add_to_exported_libs(ocuda_ldpc_decoder)
target_link_libraries(ocuda_ldpc_decoder PRIVATE CUDA::cudart)
