cmake_minimum_required(VERSION 3.16)

#add_definitions(-std=c++14)
project(jdqmr16 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 70)
set(MKLROOT /opt/intel/mkl/)
set(CUDADIR /usr/local/cuda-10.2)
#find_package(CUDA CONFIG REQUIRED)
#set(CMAKE_CXX_COMPILER $(CUDADIR)/bin/nvcc)
#target_compile_options(jdqmr16 PRIVATE $(CUDADIR)/bin/nvcc)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples/)

