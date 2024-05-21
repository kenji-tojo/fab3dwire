cmake_minimum_required(VERSION 3.20)
project(wiregrad-debug)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CXX CUDA)
    add_compile_definitions(WIREGRAD_CUDA)
endif()

set(WIREGRAD_DIR ${CMAKE_CURRENT_LIST_DIR}/../src)

add_subdirectory(${WIREGRAD_DIR} wiregrad)
target_compile_options(libwiregrad PRIVATE -fPIC)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

add_executable(debug debug.cpp)
target_include_directories(debug PUBLIC ${WIREGRAD_DIR})
target_link_libraries(debug PUBLIC libwiregrad Eigen3::Eigen)

