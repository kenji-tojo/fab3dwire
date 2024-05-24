cmake_minimum_required(VERSION 3.22...3.28)
project(libwiregrad)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CXX CUDA)
    add_subdirectory(cuda)
    target_compile_options(libwiregrad-cuda PRIVATE -fPIC)
else()
    message(STATUS "CUDA is not supported")
endif()


find_package(Eigen3 3.4 REQUIRED NO_MODULE)

add_subdirectory(ext/bvh)

set(SRC common.h
        vector.h
        thread.h
        timer.h
        intersect.h
        antialias.h
        image_tile.h
        bbox.h
        line_hierarchy3d.h
        line_hierarchy3d.cpp
        line_hierarchy2d.h
        line_hierarchy2d.cpp
        repulsion.h
        repulsion.cpp
        basis_spline.h
        basis_spline.cpp
        trimesh.h
        trimesh.cpp
        filled_polygons.h
        polylines.h
        render.h
        render.cpp
        edge_sampling.h
        edge_sampling.cpp
        wiregrad.h
        wiregrad.cpp)

add_library(libwiregrad ${SRC})
target_link_libraries(libwiregrad PRIVATE Eigen3::Eigen bvh)

if(CMAKE_CUDA_COMPILER)
    target_link_libraries(libwiregrad PRIVATE libwiregrad-cuda)
    set_property(TARGET libwiregrad PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()



