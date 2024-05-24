cmake_minimum_required(VERSION 3.22...3.28)
project(wiregrad)


if (NOT SKBUILD)
  message(WARNING "This CMake file should be executed via scikit-build. "
                      "Please run\n$ pip install .")
endif()

if (SKBUILD)
  # Constrain FindPython to find the Python version used by scikit-build
  set(Python_VERSION "${PYTHON_VERSION_STRING}")
  set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
  set(Python_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}")
  set(Python_LIBRARIES "${PYTHON_LIBRARY}")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CXX CUDA)
    add_compile_definitions(WIREGRAD_CUDA)
endif()

add_subdirectory(src)
target_compile_options(libwiregrad PRIVATE -fPIC)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Run `nanobind.cmake_dir()` from Python to detect install location
execute_process(
  COMMAND
  "${PYTHON_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_VARIABLE _tmp_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")

# Now import nanobind from CMake
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_m NOMINSIZE src/main.cpp)
target_include_directories(_m PUBLIC src ext)
target_link_libraries(_m PUBLIC libwiregrad Eigen3::Eigen)

install(TARGETS _m LIBRARY DESTINATION .)


