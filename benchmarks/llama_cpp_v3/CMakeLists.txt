cmake_minimum_required(VERSION 3.20)

project(apxinf_llama_cpp_raw_token_runner_v3 LANGUAGES C CXX)

set(LLAMA_CPP_SOURCE_DIR "" CACHE PATH "Path to a pinned llama.cpp source checkout")
set(LLAMA_CPP_SOURCE_ID "" CACHE STRING "Audited llama.cpp source identifier (normally a Git commit)")
set(LLAMA_CPP_SOURCE_ID_PROVENANCE "caller-supplied")

if(NOT LLAMA_CPP_SOURCE_DIR)
    message(FATAL_ERROR "Set LLAMA_CPP_SOURCE_DIR to a pinned llama.cpp source checkout")
endif()

get_filename_component(LLAMA_CPP_SOURCE_DIR "${LLAMA_CPP_SOURCE_DIR}" REALPATH)
if(NOT EXISTS "${LLAMA_CPP_SOURCE_DIR}/CMakeLists.txt" OR
   NOT EXISTS "${LLAMA_CPP_SOURCE_DIR}/include/llama.h")
    message(FATAL_ERROR "LLAMA_CPP_SOURCE_DIR is not a llama.cpp source checkout: ${LLAMA_CPP_SOURCE_DIR}")
endif()

if(NOT LLAMA_CPP_SOURCE_ID)
    execute_process(
        COMMAND git -C "${LLAMA_CPP_SOURCE_DIR}" rev-parse HEAD
        RESULT_VARIABLE LLAMA_CPP_GIT_RESULT
        OUTPUT_VARIABLE LLAMA_CPP_GIT_HEAD
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(LENGTH "${LLAMA_CPP_GIT_HEAD}" LLAMA_CPP_GIT_HEAD_LENGTH)
    if(LLAMA_CPP_GIT_RESULT EQUAL 0 AND
       LLAMA_CPP_GIT_HEAD_LENGTH EQUAL 40 AND
       LLAMA_CPP_GIT_HEAD MATCHES "^[0-9a-fA-F]+$")
        execute_process(
            COMMAND git -C "${LLAMA_CPP_SOURCE_DIR}" status --porcelain --untracked-files=normal
            RESULT_VARIABLE LLAMA_CPP_GIT_STATUS_RESULT
            OUTPUT_VARIABLE LLAMA_CPP_GIT_STATUS
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(LLAMA_CPP_GIT_STATUS_RESULT EQUAL 0 AND NOT LLAMA_CPP_GIT_STATUS)
            set(LLAMA_CPP_SOURCE_ID "${LLAMA_CPP_GIT_HEAD}")
            set(LLAMA_CPP_SOURCE_ID_PROVENANCE "clean-git-head")
        else()
            set(LLAMA_CPP_SOURCE_ID "${LLAMA_CPP_GIT_HEAD}-dirty")
            set(LLAMA_CPP_SOURCE_ID_PROVENANCE "dirty-git-head")
        endif()
    else()
        set(LLAMA_CPP_SOURCE_ID "unavailable")
        set(LLAMA_CPP_SOURCE_ID_PROVENANCE "unavailable")
    endif()
endif()

# This comparison runner has one audited runtime closure: a Release executable
# with llama/ggml, Metal, the embedded metallib, CPU, and Accelerate linked in.
# It must never rely on llama.cpp's runtime backend discovery.
if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(FATAL_ERROR "This runner requires -DCMAKE_BUILD_TYPE=Release")
endif()
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_STATIC OFF CACHE BOOL "" FORCE)
set(GGML_LTO OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)
set(GGML_METAL ON CACHE BOOL "" FORCE)
set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
set(GGML_ACCELERATE ON CACHE BOOL "" FORCE)
set(GGML_CPU ON CACHE BOOL "" FORCE)
set(GGML_CPU_REPACK ON CACHE BOOL "" FORCE)
set(GGML_BLAS ON CACHE BOOL "" FORCE)
set(GGML_BLAS_VENDOR Apple CACHE STRING "" FORCE)
set(GGML_LLAMAFILE ON CACHE BOOL "" FORCE)
set(GGML_NATIVE ON CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_METAL_NDEBUG OFF CACHE BOOL "" FORCE)
set(GGML_METAL_SHADER_DEBUG OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_MUSA OFF CACHE BOOL "" FORCE)
set(GGML_HIP OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_WEBGPU OFF CACHE BOOL "" FORCE)
set(GGML_ZDNN OFF CACHE BOOL "" FORCE)
set(GGML_VIRTGPU OFF CACHE BOOL "" FORCE)
set(GGML_VIRTGPU_BACKEND OFF CACHE BOOL "" FORCE)
set(GGML_RPC OFF CACHE BOOL "" FORCE)
set(GGML_SYCL OFF CACHE BOOL "" FORCE)
set(GGML_OPENVINO OFF CACHE BOOL "" FORCE)
set(GGML_ET OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL OFF CACHE BOOL "" FORCE)
set(GGML_HEXAGON OFF CACHE BOOL "" FORCE)
set(GGML_ZENDNN OFF CACHE BOOL "" FORCE)

# Keep the embedded checkout library-only.
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI OFF CACHE BOOL "" FORCE)

add_subdirectory("${LLAMA_CPP_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/llama.cpp" EXCLUDE_FROM_ALL)

add_executable(apxinf-llama-cpp-raw-token-runner raw_token_runner.cpp)
set_target_properties(apxinf-llama-cpp-raw-token-runner PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_compile_features(apxinf-llama-cpp-raw-token-runner PRIVATE cxx_std_17)
target_include_directories(apxinf-llama-cpp-raw-token-runner PRIVATE
    "${LLAMA_CPP_SOURCE_DIR}/src")
target_link_libraries(apxinf-llama-cpp-raw-token-runner PRIVATE llama)

file(READ "${LLAMA_CPP_SOURCE_DIR}/include/llama.h" LLAMA_CPP_PUBLIC_HEADER)
if(LLAMA_CPP_PUBLIC_HEADER MATCHES "enum llama_load_mode")
    target_compile_definitions(apxinf-llama-cpp-raw-token-runner PRIVATE
        APXINF_LLAMA_CPP_HAS_LOAD_MODE=1)
endif()
if(LLAMA_CPP_PUBLIC_HEADER MATCHES "llama_version\\(void\\)")
    target_compile_definitions(apxinf-llama-cpp-raw-token-runner PRIVATE
        APXINF_LLAMA_CPP_HAS_VERSION_API=1)
endif()

target_compile_definitions(apxinf-llama-cpp-raw-token-runner PRIVATE
    APXINF_LLAMA_CPP_SOURCE_ID="${LLAMA_CPP_SOURCE_ID}"
    APXINF_LLAMA_CPP_SOURCE_ID_PROVENANCE="${LLAMA_CPP_SOURCE_ID_PROVENANCE}"
    APXINF_CMAKE_VERSION="${CMAKE_VERSION}"
    APXINF_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    APXINF_CXX_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
    APXINF_CXX_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}")

if(MSVC)
    target_compile_options(apxinf-llama-cpp-raw-token-runner PRIVATE /W4)
else()
    target_compile_options(apxinf-llama-cpp-raw-token-runner PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)
endif()
