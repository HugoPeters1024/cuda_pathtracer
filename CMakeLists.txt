cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(pathtracer CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_FLAGS "-O3")


file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu")
enable_language(CUDA)
add_executable(pathtracer ${cpu_source_files} ${gpu_source_files})

set_target_properties(
        pathtracer 
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(pathtracer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       --gpu-architecture=sm_50
                       -gencode=arch=compute_50,code=sm_50
                       -O3
                       -g
                       >)


# We want to use c++ inline with cuda, so set nvcc as primary compiler
set_source_files_properties(${cpu_source_files} PROPERTIES LANGUAGE CUDA)

add_subdirectory("./glfw")
include_directories("./glfw/include")
target_link_libraries(pathtracer glfw ${GLFW_LIBRARIES})
find_package(OpenGL REQUIRED)

#find_package(glfw3 3.3.2 REQUIRED)
#target_link_libraries(pathtracer glfw)
# These variables have been set by the previous command
target_include_directories(pathtracer PUBLIC ${OPENGL_INCLUDE_DIR})
target_link_libraries(pathtracer ${OPENGL_LIBRARIES})
