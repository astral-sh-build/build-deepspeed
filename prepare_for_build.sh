#!/bin/bash
# Script to prepare the build environment for DeepSpeed.
#
# Example usage:
#   ./prepare_for_build.sh v0.16.7

set -euxo pipefail

export ROOT=`pwd`

if [ $# -ne 1 ]; then
    echo "Usage: $0 <deepspeed_version>"
    echo "Example: $0 v0.16.7"
    exit 1
fi

DEEPSPEED_VERSION=$1

python --version
uv --version
which python
which uv

# Apply patches.
patch_dir="${ROOT}/build_scripts/patches/${DEEPSPEED_VERSION}"

# Not all DeepSpeed versions need patches.
if [ ! -d "${patch_dir}" ]; then
    echo "Warning: nothing to patch: patches/${DEEPSPEED_VERSION} directory does not exist"
else
    for patch in "${patch_dir}"/*.patch; do
        patch -p1 -d "${ROOT}" -i "${patch}"
    done
fi
