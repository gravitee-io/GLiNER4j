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

set -euo pipefail

ORT_VERSION="${ORT_VERSION:-1.21.0}"
OPENVINO_DEVICE="${OPENVINO_DEVICE:-CPU}"
WORKDIR="${WORKDIR:-$(pwd)/.ort-openvino-build}"
JOBS="${JOBS:-$( (nproc 2>/dev/null) || (sysctl -n hw.ncpu 2>/dev/null) || echo 4)}"

GROUP_ID="com.microsoft.onnxruntime"
ARTIFACT_ID="onnxruntime_openvino"
ORT_REPO="https://github.com/microsoft/onnxruntime.git"

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
err() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; }

require() {
  command -v "$1" >/dev/null 2>&1 || {
    err "'$1' is required but not on PATH. $2"
    exit 1
  }
}

# Major Java version of a given java binary (e.g. "21.0.2" -> 21, "25" -> 25).
java_major() { "$1" -version 2>&1 | awk -F'"' '/version/{print $2}' | cut -d. -f1; }

# --- 1. Prerequisite checks ---------------------------------------------------
log "Checking prerequisites"
require git "Install git."
require cmake "Install cmake >= 3.28."
require python3 "Install Python 3."
require curl "Install curl."
require unzip "Install unzip."
require mvn "Install Maven."
require java "Install a JDK (<= 21)."

# --- 1b. Pin OpenVINO to a specific version (optional) ------------------------
# ONNX Runtime is tied to a specific OpenVINO release (ORT 1.21 -> OpenVINO 2025.0). Set
# OPENVINO_VERSION to wipe whatever OpenVINO is installed and apt-install the matched dev package,
# avoiding the option-parsing mismatches you hit with a newer OpenVINO. Leave empty to use whatever
# is already installed. Confirm the exact pin with the cloned ORT source:
#   grep -ri openvino .ort-openvino-build/dockerfiles/Dockerfile.openvino
if [[ -n "${OPENVINO_VERSION:-}" ]]; then
  require apt-get "OPENVINO_VERSION management needs apt (Debian/Ubuntu)."
  require sudo "OPENVINO_VERSION management needs sudo to change apt packages."
  installed="$(dpkg -l 2>/dev/null | awk '/openvino/{print $2}')"
  if [[ -n "$installed" ]]; then
    log "Purging installed OpenVINO packages: $installed"
    # shellcheck disable=SC2086
    sudo apt-get purge -y $installed
    sudo apt-get autoremove -y
  fi
  log "Installing openvino-libraries-dev-$OPENVINO_VERSION"
  sudo apt-get update
  sudo apt-get install -y "openvino-libraries-dev-$OPENVINO_VERSION"
  # New OpenVINO => force a clean ORT reconfigure (drops cached cmake/EP state).
  rm -rf "$WORKDIR/build" 2>/dev/null || true
fi

# --- 2. Locate the OpenVINO toolkit (setupvars.sh or OpenVINOConfig.cmake) -----
export OpenVINO_DIR=$(dirname "$(find /usr -name OpenVINOConfig.cmake 2>/dev/null | head -1)")

if [[ -z "${OpenVINO_DIR:-}" && -z "${INTEL_OPENVINO_DIR:-}" ]]; then
  err "OpenVINO not found. Install the developer package (headers + OpenVINOConfig.cmake), then"
  err "either source setupvars.sh or export OpenVINO_DIR=<dir with OpenVINOConfig.cmake>."
  err "See https://docs.openvino.ai/install"
  exit 1
fi
log "Using OpenVINO (OpenVINO_DIR=${OpenVINO_DIR:-via setupvars}, device=$OPENVINO_DEVICE)"

# --- 3. Pick a JDK <= 21 for the build (Gradle 8.7 rejects Java 25) -----------
# Does not touch the system default — only sets JAVA_HOME for this build.
current_major="$(java_major "$(command -v java)")"
if [[ -n "${JAVA_HOME:-}" && -x "${JAVA_HOME}/bin/java" ]]; then
  current_major="$(java_major "${JAVA_HOME}/bin/java")"
fi
if [[ -z "$current_major" || "$current_major" -gt 21 ]]; then
  log "Active java is major ${current_major:-unknown} (>21); searching /usr/lib/jvm for a JDK <= 21"
  found=""
  for v in 21 17; do
    for d in /usr/lib/jvm/*"$v"*; do
      if [[ -x "$d/bin/java" ]]; then
        m="$(java_major "$d/bin/java")"
        if [[ -n "$m" && "$m" -le 21 ]]; then found="$d"; break 2; fi
      fi
    done
  done
  if [[ -z "$found" ]]; then
    err "No JDK <= 21 found under /usr/lib/jvm. Gradle (ORT's Java build) cannot run on Java > 22."
    err "Install one, e.g.: sudo apt install openjdk-21-jdk"
    exit 1
  fi
  export JAVA_HOME="$found"
  export PATH="$JAVA_HOME/bin:$PATH"
fi
log "Build JDK: ${JAVA_HOME:-$(command -v java)} (major $(java_major "$(command -v java)"))"

# Force the system python3 for codegen (avoid an accidentally-active venv interpreter).
PYTHON_BIN="${PYTHON_BIN:-$([[ -x /usr/bin/python3 ]] && echo /usr/bin/python3 || command -v python3)}"
log "Build Python: $PYTHON_BIN"

# --- 4. Check out ONNX Runtime ------------------------------------------------
if [[ ! -d "$WORKDIR/.git" ]]; then
  log "Cloning ONNX Runtime v$ORT_VERSION into $WORKDIR"
  git clone --recursive --branch "v$ORT_VERSION" --depth 1 "$ORT_REPO" "$WORKDIR"
else
  log "Reusing existing checkout at $WORKDIR"
fi

# --- 5. Pre-fetch Eigen (GitLab archives are flaky and hash-drift) ------------
# ORT's pinned Eigen URL lives in cmake/deps.txt; download it ourselves with retries and hand the
# extracted source to CMake via FETCHCONTENT_SOURCE_DIR_EIGEN, bypassing the download + hash check.
EIGEN_CACHE="$WORKDIR/.eigen-src"
EIGEN_SRC="$(find "$EIGEN_CACHE" -maxdepth 1 -mindepth 1 -type d -name 'eigen-*' 2>/dev/null | head -1 || true)"
if [[ -z "$EIGEN_SRC" ]]; then
  eigen_url="$(grep -E '^eigen;' "$WORKDIR/cmake/deps.txt" | cut -d';' -f2)"
  if [[ -z "$eigen_url" ]]; then
    err "Could not find the Eigen entry in $WORKDIR/cmake/deps.txt"
    exit 1
  fi
  log "Pre-fetching Eigen: $eigen_url"
  curl -L --retry 15 --retry-all-errors -o /tmp/eigen.zip "$eigen_url"
  if ! unzip -tq /tmp/eigen.zip >/dev/null 2>&1; then
    err "Downloaded Eigen archive is not a valid zip (GitLab likely served an error page). Re-run."
    exit 1
  fi
  mkdir -p "$EIGEN_CACHE"
  unzip -qo /tmp/eigen.zip -d "$EIGEN_CACHE"
  EIGEN_SRC="$(find "$EIGEN_CACHE" -maxdepth 1 -mindepth 1 -type d -name 'eigen-*' | head -1)"
fi
log "Using local Eigen source: $EIGEN_SRC"

# --- 6. Build with OpenVINO + Java bindings -----------------------------------
log "Building ONNX Runtime (Release, --use_openvino $OPENVINO_DEVICE, --build_java) with $JOBS jobs"
ROOT_FLAG=""
[[ "$(id -u)" -eq 0 ]] && ROOT_FLAG="--allow_running_as_root"

"$WORKDIR/build.sh" \
  --config Release \
  --parallel "$JOBS" \
  --use_openvino "$OPENVINO_DEVICE" \
  --build_java \
  --skip_tests \
  --compile_no_warning_as_error \
  $ROOT_FLAG \
  --cmake_extra_defines \
  "FETCHCONTENT_SOURCE_DIR_EIGEN=$EIGEN_SRC" \
  "Python_EXECUTABLE=$PYTHON_BIN"

# --- 7. Locate the built jar --------------------------------------------------
log "Locating the built Java artifact"
JAR="$(find "$WORKDIR/build" -path '*/java/build/libs/*' -name 'onnxruntime-*.jar' \
  ! -name '*-sources.jar' ! -name '*-javadoc.jar' | head -1)"
if [[ -z "$JAR" ]]; then
  err "Could not find the built onnxruntime jar under $WORKDIR/build. Did the Java build succeed?"
  exit 1
fi
log "Built jar: $JAR"

# --- 8. Install into the local Maven repository -------------------------------
log "Installing $GROUP_ID:$ARTIFACT_ID:$ORT_VERSION into the local Maven repo"
mvn install:install-file \
  -Dfile="$JAR" \
  -DgroupId="$GROUP_ID" \
  -DartifactId="$ARTIFACT_ID" \
  -Dversion="$ORT_VERSION" \
  -Dpackaging=jar

cat <<EOF

$(log "Done.")

Installed: $GROUP_ID:$ARTIFACT_ID:$ORT_VERSION

Build & run GLiNER4j on the OpenVINO provider:

  mvn clean install -DskipTests -Popenvino
  mvn exec:java -pl gliner4j-demo -Popenvino -Dexec.args="base onnx openvino"
  java -jar gliner4j-benchmark/target/gliner4j-benchmark.jar -p executionProvider=openvino

At run time the OpenVINO runtime libraries must be discoverable (the EP links them dynamically):

  source <openvino>/setupvars.sh    # or: export LD_LIBRARY_PATH=<openvino runtime libs>:\$LD_LIBRARY_PATH
EOF
