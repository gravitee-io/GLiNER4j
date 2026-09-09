#
# Copyright © 2015 The Gravitee team (http://gravitee.io)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# ggml backend plugin with the fused DeBERTa attention op (see ggml-deberta.h).
#
# Built out of tree against a llama.cpp checkout at the tag llamaj.cpp pins (v0.4.0 for 2.8.0):
#   cmake -S . -B build -DLLAMA_CPP_DIR=~/dev/llama.cpp -DCMAKE_CUDA_ARCHITECTURES=75
#   cmake --build build
# The checkout must have been built with -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON (llamaj's
# configuration): the plugin links libggml-base from its build/bin and includes ggml/src for the
# backend-implementation headers, which ggml does not install.
cmake_minimum_required(VERSION 3.24)
project(ggml-deberta LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")
endif()

set(LLAMA_CPP_DIR "" CACHE PATH "llama.cpp checkout (built) providing ggml headers and libggml-base")
if(NOT LLAMA_CPP_DIR)
    message(FATAL_ERROR "set -DLLAMA_CPP_DIR=<llama.cpp checkout built with BUILD_SHARED_LIBS=ON GGML_BACKEND_DL=ON>")
endif()
set(GGML_INCLUDE_DIR "${LLAMA_CPP_DIR}/ggml/include")
set(GGML_SRC_DIR "${LLAMA_CPP_DIR}/ggml/src")
set(GGML_BIN_DIR "${LLAMA_CPP_DIR}/build/bin" CACHE PATH "directory holding libggml-base")
find_library(GGML_BASE_LIB NAMES ggml-base PATHS "${GGML_BIN_DIR}" NO_DEFAULT_PATH)
find_library(GGML_LIB NAMES ggml PATHS "${GGML_BIN_DIR}" NO_DEFAULT_PATH)   # the backend registry (ggml_backend_reg_by_name)
if(NOT GGML_BASE_LIB OR NOT GGML_LIB)
    message(FATAL_ERROR "libggml-base / libggml not found in ${GGML_BIN_DIR}; build llama.cpp first")
endif()
if(NOT EXISTS "${GGML_SRC_DIR}/ggml-backend-impl.h")
    message(FATAL_ERROR "${GGML_SRC_DIR}/ggml-backend-impl.h missing: LLAMA_CPP_DIR must be a source checkout")
endif()

find_package(CUDAToolkit REQUIRED)

add_library(ggml-deberta MODULE ggml-deberta.cpp deberta-attn.cu lstm.cu)
target_include_directories(ggml-deberta PRIVATE "${GGML_INCLUDE_DIR}" "${GGML_SRC_DIR}")
target_compile_definitions(ggml-deberta PRIVATE GGML_BACKEND_DL GGML_BACKEND_SHARED GGML_BACKEND_BUILD GGML_SHARED)
target_link_libraries(ggml-deberta PRIVATE "${GGML_LIB}" "${GGML_BASE_LIB}" CUDA::cudart)
set_target_properties(ggml-deberta PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "ggml-deberta"       # no soname suffix: llamaj's loader must not System.load it
    CUDA_SEPARABLE_COMPILATION OFF
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden)
if(APPLE)
    set_target_properties(ggml-deberta PROPERTIES SUFFIX ".dylib")
endif()

install(TARGETS ggml-deberta LIBRARY DESTINATION plugins)
