#!/usr/bin/env bash
# Compile shaders that are loaded as pre-built SPIR-V at runtime.
# Search and ray-march shaders are generated/compiled at runtime via Slang
# and don't need offline compilation.
#
# Requires the Vulkan SDK environment to be sourced (VULKAN_SDK set).

set -euo pipefail

if [ -z "${VULKAN_SDK:-}" ]; then
	echo "ERROR: VULKAN_SDK environment variable not set. Source the Vulkan SDK setup script and re-run." >&2
	exit 1
fi

cd "$(dirname "$0")"

"$VULKAN_SDK/bin/glslc" ../shaders/source/encoding/rgb2ycbcr.comp -o ../shaders/compiled/rgb2ycbcr.spv

echo "Shader compilation complete."
