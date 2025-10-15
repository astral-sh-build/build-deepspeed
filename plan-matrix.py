# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "packaging",
# ]
# ///

import json

from packaging.version import Version

# Versions of PyTorch we actually want to include in the matrix.
TORCH_VERSIONS = [
    "2.7.1",
    # "2.8.0",
    "2.9.0",
]

# Versions of Python we actually want to include in the matrix.
PYTHON_VERSIONS = [
    # "3.9",
    # "3.10",
    # "3.11",
    "3.12",
    "3.13",
    # "3.14", # Not supported by DeepSpeed yet.
]

# Supported Python versions for each PyTorch version.
# We use these to filter out the matrix.
TORCH_PYTHON_SUPPORT = {
    "2.7.1": ["3.9", "3.10", "3.11", "3.12", "3.13"],
    "2.8.0": ["3.9", "3.10", "3.11", "3.12", "3.13"],
    "2.9.0": ["3.10", "3.11", "3.12", "3.13", "3.14"],
}

# Minimum and maximum CUDA versions for each PyTorch version.
PYTORCH_CUDA_RANGES: dict[str, tuple[str, str]] = {
    "2.7": ("11.8", "12.8"),
    "2.9": ("12.6", "13.0"),
}

# Actual CUDA versions to build against for each PyTorch version.
PYTORCH_CUDA_VERSIONS: dict[str, list[str]] = {
    "2.7": ["12.6.0", "12.8.0"],
    "2.9": ["12.6.0", "12.8.0", "13.0.0"],
}

# CUDA architectures to build against for each (PyTorch version, CUDA version) pair.
# NOTE(ww): These are probably an overapproximation; I took them from
# the original unrolled matrix.
PYTORCH_CUDA_ARCH_LIST: dict[tuple[str, str], str] = {
    ("2.7", "12.6"): "6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0+PTX",
    ("2.7", "12.8"): "6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX",
    ("2.9", "12.6"): "6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0+PTX",
    ("2.9", "12.8"): "6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX",
    # Torch 2.9 with CUDA 13 requires 7.5+.
    # See: https://github.com/pytorch/pytorch/blob/815d6415996d5b32b569fd2a8206f1e57c75bfe3/RELEASE.md#pytorch-cuda-support-matrix
    ("2.9", "13.0"): "7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX",
}

AUDITWHEEL_BLANKET_EXCLUDES = [
    "libcuda.so",
    "libcuda.so.1",
    "libc10.so",
    "libc10_cuda.so",
    "libtorch.so",
    "libtorch_python.so",
    "libtorch_cpu.so",
    "libtorch_cuda.so",
    "libtorch_cuda_cpp.so",
    "libtorch_cuda_cu.so",
    "libcufile_rdma.so",
    "libcufile_rdma.so.1",
    "libcufile.so.1",
    "libcufile.so.0",
    "libcufile.so",
]

AUDITWHEEL_CUDA_VERSION_EXCLUDES = {
    "10": [
        "libcurand.so.10",
    ],
    "11": [
        "libcudart.so.11",
        "libcudart.so.11.0",
    ],
    "12": [
        "libcudart.so.12",
        "libcudart.so.12.0",
    ],
    "13": [
        "libcudart.so.13",
        "libcudart.so.13.0",
    ],
}

# Matrix exclusions.
EXCLUSIONS = [
    # No exclusions yet.
]


def main() -> None:
    # Every matrix member is a primary 5-tuple of:
    # `torch-version`: the PyTorch version as "X.Y.Z", e.g. "2.7.0"
    # `python-version`: the Python version as "3.X", e.g. "3.10"
    # `cuda-version`: the CUDA version as "X.Y.Z", e.g. "11.8.0"
    # `cuda-arch-list`: the CUDA architectures as a semicolon-separated list
    # `deepcompile`: 1 or 0, whether to build with DeepCompile support

    rows = []
    for python_version in PYTHON_VERSIONS:
        for torch_version in TORCH_VERSIONS:
            if python_version not in TORCH_PYTHON_SUPPORT[torch_version]:
                continue

            torch_version = Version(torch_version)
            torch_x_y = f"{torch_version.major}.{torch_version.minor}"
            cuda_versions = PYTORCH_CUDA_VERSIONS[torch_x_y]
            for cuda_version in cuda_versions:
                cuda_version_parsed = Version(cuda_version)
                cuda_x_y = f"{cuda_version_parsed.major}.{cuda_version_parsed.minor}"
                cuda_arch_list = PYTORCH_CUDA_ARCH_LIST[(torch_x_y, cuda_x_y)]

                row = {
                    "torch-version": str(torch_version),
                    "python-version": python_version,
                    "cuda-version": cuda_version,
                    "cuda-arch-list": cuda_arch_list,
                    # DeepCompile appears to require Torch 2.5 or newer,
                    # but our original matrix only enabled in on 2.6 and newer.
                    # Follow that here.
                    # See: https://github.com/deepspeedai/DeepSpeed/pull/7154
                    "deepcompile": int(torch_version >= Version("2.6")),
                }

                if row not in EXCLUSIONS:
                    rows.append(row)

    # Transform each row to add various nice-to-have representations of fields.
    for row in rows:
        # `CI_*` variables: same as the original ones.
        row["CI_CUDA_VERSION"] = row["cuda-version"]
        row["CI_TORCH_VERSION"] = row["torch-version"]
        row["CI_PYTHON_VERSION"] = row["python-version"]

        # `MATRIX_CUDA_VERSION`: XY instead of X.Y
        cuda_version = Version(row["cuda-version"])
        row["MATRIX_CUDA_VERSION"] = f"{cuda_version.major}{cuda_version.minor}"

        # `MATRIX_TORCH_VERSION`: `torch-version`, but only X.Y, no patch
        torch_version = Version(row["torch-version"])
        row["MATRIX_TORCH_VERSION"] = f"{torch_version.major}.{torch_version.minor}"

        # `MATRIX_PYTHON_VERSION`: same as `python-version`, but with the dot removed
        row["MATRIX_PYTHON_VERSION"] = row["python-version"].replace(".", "")

        # `MANYLINUX_CUDA_VERSION`: X.Y instead of X.Y.Z
        row["MANYLINUX_CUDA_VERSION"] = f"{cuda_version.major}.{cuda_version.minor}"

        # `MANYLINUX_CUDA_COMPAT_VERSION`: X-Y instead of X.Y.Z
        row["MANYLINUX_CUDA_COMPAT_VERSION"] = (
            f"{cuda_version.major}-{cuda_version.minor}"
        )

        # `CI_AUDITWHEEL_EXCLUDES`: `--exclude {lib}` for each lib that should
        # be excluded when running `auditwheel repair`.
        cuda_major = str(cuda_version.major)
        auditwheel_excludes = (
            AUDITWHEEL_BLANKET_EXCLUDES
            + AUDITWHEEL_CUDA_VERSION_EXCLUDES.get(cuda_major, [])
        )
        row["CI_AUDITWHEEL_EXCLUDES"] = " ".join(
            f"--exclude {lib}" for lib in auditwheel_excludes
        )

    print(json.dumps(rows))


if __name__ == "__main__":
    main()
