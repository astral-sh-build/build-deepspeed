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

# Ensure that the DeepSped version is supported.
if [ ! -d "${ROOT}/build_scripts/patches/${DEEPSPEED_VERSION}" ]; then
    echo "Error: patches/${DEEPSPEED_VERSION} directory does not exist"
    exit 1
fi

# We want to figure out the CUDA version to download pytorch
# e.g. we can have system CUDA version being 11.7 but if torch==1.12 then we need to download the wheel from cu116
# see https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix
# This code is ugly, maybe there's a better way to do this.
export TORCH_CUDA_VERSION=$(python -c "from os import environ as env; \
    minv = {'1.12': 113, '1.13': 116, '2.0': 117, '2.1': 118, '2.2': 118, '2.3': 118, '2.4': 118, '2.5': 118, '2.6': 118, '2.7': 118}[env['MATRIX_TORCH_VERSION']]; \
    maxv = {'1.12': 116, '1.13': 117, '2.0': 118, '2.1': 121, '2.2': 121, '2.3': 121, '2.4': 124, '2.5': 124, '2.6': 126, '2.7': 128}[env['MATRIX_TORCH_VERSION']]; \
    print(max(min(int(env['MATRIX_CUDA_VERSION']), maxv), minv))" \
)

python --version
uv --version
which python
which uv

# Apply patches.
for patch in "${ROOT}/build_scripts/patches/${DEEPSPEED_VERSION}"/*.patch; do
    patch -p1 -d ${ROOT} -i ${patch}
done

uv pip install hjson ninja numpy packaging psutil py-cpuinfo pydantic pynvml tqdm libaio deepspeed-kernels triton

echo "install torch==${CI_TORCH_VERSION}+cu${TORCH_CUDA_VERSION}"
uv pip install --no-cache-dir torch==${CI_TORCH_VERSION} --index-url https://download.pytorch.org/whl/cu${TORCH_CUDA_VERSION}
