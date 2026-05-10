#!/usr/bin/env bash
# Build the GPU GUI (physically-cringe-rendering) and launch it.
#
# First run pulls OIDN + GLFW via CMake FetchContent (~1-2 min on the
# network). Subsequent runs are incremental builds (~5-15 sec) plus
# launch. Pass any args after the script and they get forwarded to the
# binary.
#
# macOS uses the Metal backend; Linux uses the OpenGL 4.3 backend.
# Windows builds through the Visual Studio generator instead, so this
# script doesn't try to handle that case.

set -euo pipefail
cd "$(dirname "$0")"

cmake -S code -B code/Build
cmake --build code/Build -j --target physically-cringe-rendering

./code/Build/physically-cringe-rendering "$@"
