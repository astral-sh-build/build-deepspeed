# build-deepspeed

Pre-built Linux wheels for [DeepSpeed](https://github.com/deepspeedai/DeepSpeed), across Python,
PyTorch, CUDA, and CPU architectures.

## Installation

Following the PyTorch convention, artifacts are published to a separate index for each CUDA
version. Each wheel has a local version suffix that identifies the CUDA and PyTorch versions it was
built against, such as `deepspeed==0.18.9+cu.12.8.torch.2.10`, and requires the matching PyTorch
release.

Pre-built wheels are available on [Astral's GPU indexes](https://wheels.astral.sh/index.html).
For example, to install a CUDA 12.8 build:

```console
$ uv add deepspeed --index astral-cu128=https://wheels.astral.sh/simple/cu128/
```

This configures the index and uses it as the source for `deepspeed`:

```toml
[tool.uv.sources]
deepspeed = { index = "astral-cu128" }

[[tool.uv.index]]
name = "astral-cu128"
url = "https://wheels.astral.sh/simple/cu128/"
```

Or, with `uv pip`:

```console
$ uv pip install --index https://wheels.astral.sh/simple/cu128/ deepspeed
```

## Supported versions

Wheels are available for the following `deepspeed` versions:

- [`0.19.2`](https://github.com/astral-sh-build/build-deepspeed/releases/tag/v0.19.2)
- [`0.19.1`](https://github.com/astral-sh-build/build-deepspeed/releases/tag/v0.19.1)
- [`0.19.0`](https://github.com/astral-sh-build/build-deepspeed/releases/tag/v0.19.0)
- [`0.18.9`](https://github.com/astral-sh-build/build-deepspeed/releases/tag/v0.18.9)
- [`0.18.8`](https://github.com/astral-sh-build/build-deepspeed/releases/tag/v0.18.8)
- [`0.18.7`](https://github.com/astral-sh-build/build-deepspeed/releases/tag/v0.18.7)
- [`0.18.6`](https://github.com/astral-sh-build/build-deepspeed/releases/tag/v0.18.6)
- [`0.18.5`](https://github.com/astral-sh-build/build-deepspeed/releases/tag/v0.18.5-r1)
- [`0.18.4`](https://github.com/astral-sh-build/build-deepspeed/releases/tag/v0.18.4-r1)
- [`0.18.3`](https://github.com/astral-sh-build/build-deepspeed/releases/tag/v0.18.3)
- [`0.18.2`](https://github.com/astral-sh-build/build-deepspeed/releases/tag/v0.18.2)
- [`0.18.1`](https://github.com/astral-sh-build/build-deepspeed/releases/tag/v0.18.1)
- [`0.18.0`](https://github.com/astral-sh-build/build-deepspeed/releases/tag/v0.18.0)
- [`0.17.6`](https://github.com/astral-sh-build/build-deepspeed/releases/tag/v0.17.6)
- [`0.17.5`](https://github.com/astral-sh-build/build-deepspeed/releases/tag/v0.17.5)

The latest release, DeepSpeed 0.19.2, supports the following combinations:

| PyTorch | Python    | `x86_64` CUDA          |
| ------- | --------- | ---------------------- |
| 2.7.1   | 3.9–3.13  | 12.6, 12.8             |
| 2.8.0   | 3.9–3.13  | 12.6, 12.8, 12.9       |
| 2.9.1   | 3.10–3.13 | 12.6, 12.8, 12.9, 13.0 |
| 2.10.0  | 3.10–3.14 | 12.6, 12.8, 12.9, 13.0 |
| 2.11.0  | 3.10–3.14 | 12.6, 12.8, 12.9, 13.0 |
| 2.12.1  | 3.10–3.14 | 12.6, 13.0, 13.2       |
| 2.13.0  | 3.10–3.15 | 12.6, 13.0, 13.2       |

## License

build-deepspeed is licensed under the [Apache License, Version 2.0](LICENSE).

<div align="center">
  <a target="_blank" href="https://astral.sh" style="background:none">
    <img src="https://raw.githubusercontent.com/astral-sh/ruff/main/assets/svg/Astral.svg" alt="Made by Astral">
  </a>
</div>
