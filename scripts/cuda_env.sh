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

# --- Resolve the repo root (works when sourced from bash or zsh, any cwd) ------
_cuda_env_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
_cuda_env_dir="${_cuda_env_root}/.cuda-runtime"

# NVIDIA runtime wheels the ORT 1.21 CUDA EP needs (CUDA 12 + cuDNN 9).
_cuda_env_wheels=(
  "nvidia-cuda-runtime-cu12"   # libcudart.so.12
  "nvidia-cublas-cu12"         # libcublas.so.12, libcublasLt.so.12
  "nvidia-cudnn-cu12>=9,<10"   # libcudnn.so.9 (+ sublibs)
  "nvidia-cufft-cu12"          # libcufft.so.11
  "nvidia-curand-cu12"         # libcurand.so.10
)

# --- Fetch the libraries once --------------------------------------------------
if [ ! -d "${_cuda_env_dir}/nvidia" ]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "cuda_env.sh: 'uv' not found on PATH; cannot fetch CUDA runtime wheels." >&2
    return 1 2>/dev/null || exit 1
  fi
  echo "cuda_env.sh: fetching CUDA 12 + cuDNN 9 runtime libraries into ${_cuda_env_dir} ..."
  uv pip install --target "${_cuda_env_dir}" "${_cuda_env_wheels[@]}" || {
    echo "cuda_env.sh: failed to install NVIDIA runtime wheels." >&2
    return 1 2>/dev/null || exit 1
  }
fi

# --- Prepend every nvidia/*/lib dir to LD_LIBRARY_PATH -------------------------
_cuda_env_added=0
for _cuda_env_lib in "${_cuda_env_dir}"/nvidia/*/lib; do
  if [ -d "${_cuda_env_lib}" ]; then
    case ":${LD_LIBRARY_PATH}:" in
      *":${_cuda_env_lib}:"*) ;;                                  # already present
      *) LD_LIBRARY_PATH="${_cuda_env_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
    esac
    _cuda_env_added=$((_cuda_env_added + 1))
  fi
done
export LD_LIBRARY_PATH

echo "cuda_env.sh: added ${_cuda_env_added} nvidia/*/lib dir(s) to LD_LIBRARY_PATH."

unset _cuda_env_root _cuda_env_dir _cuda_env_wheels _cuda_env_lib _cuda_env_added
