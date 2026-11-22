cmake_minimum_required(VERSION 3.26)
project(KernelLens VERSION 0.1.0 LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# Toolchain policy (see docs/hardware.md for the verified versions)
# ---------------------------------------------------------------------------
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # sm_86 = RTX 30-series (GA10x). "86" emits sm_86 SASS plus compute_86 PTX.
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
# Static cudart: no CUDA runtime DLL to ship or find. The DLL still needs
# the GPU driver (nvcuda.dll) and the MSVC runtime (VCRUNTIME140, MSVCP140,
# UCRT), which every system with the driver/Build Tools has.
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

option(KERNELLENS_PTXAS_VERBOSE "Print per-kernel register/shared-memory usage" OFF)

find_package(CUDAToolkit 13.3 REQUIRED)

# ---------------------------------------------------------------------------
# Library: compiled once as objects, linked into the DLL and the C++ tests.
# ---------------------------------------------------------------------------
add_library(kernellens_objs OBJECT
  src/common/error_state.cpp
  src/common/device_buffer.cpp
  src/cuda/saxpy.cu
  src/cuda/softmax.cu
  src/cuda/layernorm.cu
  src/cuda/gemm.cu
  src/kernellens_api.cpp
)
target_include_directories(kernellens_objs
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_definitions(kernellens_objs
  PRIVATE KL_BUILDING_DLL
  # Lets error messages print repo-relative paths (no user directories).
  PUBLIC  KL_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/"
)
target_link_libraries(kernellens_objs PUBLIC CUDA::cudart_static)
target_compile_options(kernellens_objs PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive->
  # -lineinfo: source-line correlation for Compute Sanitizer and Nsight
  # Compute; it does not change generated code.
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -Xcompiler=/W3>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${KERNELLENS_PTXAS_VERBOSE}>>:-Xptxas=-v>
)

add_library(kernellens SHARED)
target_link_libraries(kernellens PRIVATE kernellens_objs)
target_include_directories(kernellens PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()

add_executable(test_smoke tests/cpp/test_smoke.cpp)
target_link_libraries(test_smoke PRIVATE kernellens_objs)
target_include_directories(test_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(test_smoke PRIVATE KL_STATIC)
target_compile_options(test_smoke PRIVATE /W4 /permissive-)
add_test(NAME cpp_smoke COMMAND test_smoke)

add_executable(test_softmax tests/cpp/test_softmax.cpp)
target_link_libraries(test_softmax PRIVATE kernellens_objs)
target_include_directories(test_softmax PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(test_softmax PRIVATE KL_STATIC)
target_compile_options(test_softmax PRIVATE /W4 /permissive-)
add_test(NAME cpp_softmax COMMAND test_softmax)

add_executable(test_layernorm tests/cpp/test_layernorm.cpp)
target_link_libraries(test_layernorm PRIVATE kernellens_objs)
target_include_directories(test_layernorm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(test_layernorm PRIVATE KL_STATIC)
target_compile_options(test_layernorm PRIVATE /W4 /permissive-)
add_test(NAME cpp_layernorm COMMAND test_layernorm)

# GEMM tests also use cuBLAS (an independent reference); its DLLs are in the
# toolkit's bin/x64 folder, prepended to PATH for the test.
add_executable(test_gemm tests/cpp/test_gemm.cpp)
target_link_libraries(test_gemm PRIVATE kernellens_objs CUDA::cublas)
target_include_directories(test_gemm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(test_gemm PRIVATE KL_STATIC)
target_compile_options(test_gemm PRIVATE /W4 /permissive-)
add_test(NAME cpp_gemm COMMAND test_gemm)
set_tests_properties(cpp_gemm PROPERTIES ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${CUDAToolkit_BIN_DIR}/x64")

# ---------------------------------------------------------------------------
# Benchmarks (not part of ctest; run via scripts/run-bench.ps1)
# ---------------------------------------------------------------------------
# Build description recorded in every benchmark row. Written to a generated
# header (flags contain quotes and spaces that a -D definition cannot carry).
string(TOUPPER "${CMAKE_BUILD_TYPE}" KL_BUILD_TYPE_UC)
set(KL_BUILD_INFO_TEXT
  "${CMAKE_BUILD_TYPE} sm_${CMAKE_CUDA_ARCHITECTURES} nvcc ${CMAKE_CUDA_COMPILER_VERSION} [${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_${KL_BUILD_TYPE_UC}} -lineinfo] msvc ${CMAKE_CXX_COMPILER_VERSION} [${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${KL_BUILD_TYPE_UC}} /W4 /permissive-]")
string(REPLACE "\\" "\\\\" KL_BUILD_INFO_TEXT "${KL_BUILD_INFO_TEXT}")
string(REPLACE "\"" "\\\"" KL_BUILD_INFO_TEXT "${KL_BUILD_INFO_TEXT}")
string(REGEX REPLACE "  +" " " KL_BUILD_INFO_TEXT "${KL_BUILD_INFO_TEXT}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/generated/kl_build_info.h"
  "#pragma once\n#define KL_BUILD_INFO \"${KL_BUILD_INFO_TEXT}\"\n")
foreach(bench softmax layernorm)
  add_executable(bench_${bench} benchmarks/bench_${bench}.cpp)
  target_link_libraries(bench_${bench} PRIVATE kernellens_objs CUDA::nvml)
  target_include_directories(bench_${bench} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
                                                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp
                                                    ${CMAKE_CURRENT_BINARY_DIR}/generated)
  target_compile_definitions(bench_${bench} PRIVATE KL_STATIC)
  target_compile_options(bench_${bench} PRIVATE /W4 /permissive-)
endforeach()
add_executable(bench_gemm benchmarks/bench_gemm.cpp)
target_link_libraries(bench_gemm PRIVATE kernellens_objs CUDA::nvml CUDA::cublas)
target_include_directories(bench_gemm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp
                                              ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_compile_definitions(bench_gemm PRIVATE KL_STATIC)
target_compile_options(bench_gemm PRIVATE /W4 /permissive-)

# Profiling driver (docs/profiling.md; run via scripts/run-profile.ps1).
add_executable(profile_kernels benchmarks/profile_kernels.cpp)
target_link_libraries(profile_kernels PRIVATE kernellens_objs CUDA::cublas)
target_include_directories(profile_kernels PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(profile_kernels PRIVATE KL_STATIC)
target_compile_options(profile_kernels PRIVATE /W4 /permissive-)

# Python reference tests run through ctest. Missing them is a configuration
# error unless explicitly opted out, so "100% tests passed" can never quietly
# mean "only the C++ half ran".
set(KERNELLENS_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/.venv/Scripts/python.exe"
    CACHE FILEPATH "Python interpreter with numpy and pytest (project venv)")
option(KERNELLENS_REQUIRE_PYTHON_TESTS "Fail configuration if the Python test venv is missing" ON)
if(EXISTS "${KERNELLENS_PYTHON}")
  add_test(NAME python_reference
    COMMAND "${KERNELLENS_PYTHON}" -m pytest -v -p no:cacheprovider
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/python")
  set_tests_properties(python_reference PROPERTIES
    ENVIRONMENT "KERNELLENS_LIB=$<TARGET_FILE:kernellens>")
elseif(KERNELLENS_REQUIRE_PYTHON_TESTS)
  message(FATAL_ERROR
    "KERNELLENS_PYTHON not found at '${KERNELLENS_PYTHON}'. Create the venv (README 'Build') "
    "or configure with -DKERNELLENS_REQUIRE_PYTHON_TESTS=OFF to build without Python tests.")
else()
  message(WARNING "Python reference tests disabled (KERNELLENS_REQUIRE_PYTHON_TESTS=OFF).")
endif()
