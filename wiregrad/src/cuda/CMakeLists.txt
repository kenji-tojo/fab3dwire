cmake_minimum_required(VERSION 3.22...3.28)
project(libwiregrad-cuda)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
add_compile_definitions(COMPILE_WITH_CUDA)


find_package(Eigen3 3.4 REQUIRED NO_MODULE)

set(SRC common.cuh
        line_hierarchy3d.cu
        line_hierarchy2d.cu
        repulsion.cu
        basis_spline.cu
        render.cu
        edge_sampling.cu)

add_library(libwiregrad-cuda ${SRC})
target_include_directories(libwiregrad-cuda PRIVATE ${CMAKE_CURRENT_LIST_DIR}/..)
target_link_libraries(libwiregrad-cuda PRIVATE Eigen3::Eigen)
set_property(TARGET libwiregrad-cuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)



