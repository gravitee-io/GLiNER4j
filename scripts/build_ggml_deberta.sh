#!/usr/bin/env bash
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

# Builds the gliner4j "DEBERTA" ggml backend plugin (fused disentangled attention, CUDA) against a
# llama.cpp checkout and installs it as ~/.llama.cpp/plugins/libggml-deberta.so, where the
# gliner4j-llamacpp runtime picks it up (or point -Dgliner4j.ggml.plugin=<path> at it).
#
# Requirements: cmake >= 3.24, the CUDA toolkit (nvcc), and a llama.cpp checkout at the tag
# llamaj.cpp pins (v0.4.0 for llamaj.cpp 2.8.0) already built with
#   cmake -B build -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_CUDA=ON && cmake --build build
# The plugin bakes ggml's backend API version: rebuild it whenever the llama.cpp pin changes.
set -euo pipefail

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/dev/llama.cpp}"
CUDA_ARCHS="${CUDA_ARCHS:-80;86;89;90}"
PLUGIN_DIR="${PLUGIN_DIR:-$HOME/.llama.cpp/plugins}"
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/gliner4j-llamacpp/native/ggml-deberta"
WORKDIR="${WORKDIR:-$ROOT/.ggml-deberta-build}"

require() {
  command -v "$1" >/dev/null 2>&1 || { echo "error: '$1' not found on PATH" >&2; exit 1; }
}
require cmake
require nvcc

if [ ! -f "$LLAMA_CPP_DIR/build/bin/libggml-base.so" ] && [ ! -f "$LLAMA_CPP_DIR/build/bin/libggml-base.dylib" ]; then
  echo "error: $LLAMA_CPP_DIR/build/bin has no libggml-base — build llama.cpp first (BUILD_SHARED_LIBS=ON GGML_BACKEND_DL=ON)" >&2
  exit 1
fi
if [ ! -f "$LLAMA_CPP_DIR/ggml/src/ggml-backend-impl.h" ]; then
  echo "error: $LLAMA_CPP_DIR is not a llama.cpp source checkout" >&2
  exit 1
fi
if command -v git >/dev/null 2>&1; then
  tag="$(git -C "$LLAMA_CPP_DIR" describe --tags --exact-match 2>/dev/null || true)"
  echo "llama.cpp checkout: $LLAMA_CPP_DIR (${tag:-untagged commit $(git -C "$LLAMA_CPP_DIR" rev-parse --short HEAD 2>/dev/null)})"
fi

# Scheduler patch: stream-side ordering between the plugin and the CUDA backend (otherwise every
# layer costs two host round trips, ~2 ms per long call here). Applied once to the checkout and
# libggml-base rebuilt in place; the runtime detects it and switches the plugin's host syncs off.
PATCH="$SRC/patches/0001-ggml-sched-cross-backend-events.patch"
if ! grep -q ggml_backend_sched_cross_backend_events "$LLAMA_CPP_DIR/ggml/src/ggml-backend.cpp"; then
  echo "applying $PATCH to $LLAMA_CPP_DIR (git apply; revert with git checkout ggml/)"
  git -C "$LLAMA_CPP_DIR" apply "$PATCH"
  cmake --build "$LLAMA_CPP_DIR/build" --target ggml-base --parallel "$JOBS"
fi

cmake -S "$SRC" -B "$WORKDIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CPP_DIR="$LLAMA_CPP_DIR" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS"
cmake --build "$WORKDIR" --parallel "$JOBS"

mkdir -p "$PLUGIN_DIR"
lib=""
for candidate in "$WORKDIR/libggml-deberta.so" "$WORKDIR/libggml-deberta.dylib"; do
  [ -f "$candidate" ] && lib="$candidate"
done
[ -n "$lib" ] || { echo "error: build produced no libggml-deberta.{so,dylib} in $WORKDIR" >&2; exit 1; }
cp -f "$lib" "$PLUGIN_DIR/"
echo
echo "installed $(basename "$lib") into $PLUGIN_DIR"
cat <<EOF

Run with the llama.cpp natives the plugin was built against, e.g.

  LLAMA_CPP_LIB_PATH=$LLAMA_CPP_DIR/build/bin mvn -pl gliner4j-llamacpp test

The runtime looks for \$LLAMA_CPP_LIB_PATH/plugins/, then $PLUGIN_DIR/; override with
  -Dgliner4j.ggml.plugin=$PLUGIN_DIR/$(basename "$lib")
Disable it with -Dgliner4j.ggml.plugin=none. GLINER4J_DEBERTA_REF=1 selects the scalar
reference kernel (debugging only); GLINER4J_DEBERTA_KERNEL=turing|ampere forces a tile variant
(default: chosen from the device's shared-memory limit, logged at first use).
EOF
