cmake_minimum_required(VERSION 3.28)
project(CudaPractice LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(BoidNaive
        Naive/boid.cu
        Naive/deviceFunctions.cu
        Naive/deviceFunctions.h
)

add_executable(BoidTiled
        Tiled/main.cu
        Tiled/deviceFunctions.cu
        Tiled/deviceFunctions.cuh
)


add_executable(BoidVisual
        Visual/Source/main.cu
        Visual/Source/deviceFunctions.cu
        Visual/Header/deviceFunctions.cuh
        Visual/Header/glUtils.cuh
        Visual/Source/glUtils.cu
        Visual/Source/glad.c
)

target_include_directories(BoidVisual PRIVATE
        ${CMAKE_SOURCE_DIR}/Visual/Libraries/include
)

target_link_libraries(BoidVisual PRIVATE
        ${CMAKE_SOURCE_DIR}/Visual/Libraries/lib/glfw3.lib
        opengl32
)