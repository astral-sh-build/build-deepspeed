import importlib
from importlib.metadata import version

import pytest
import torch


@pytest.fixture(scope="module")
def device() -> torch.device:
    assert torch.cuda.is_available(), "The tests must run on a CUDA GPU"
    device = torch.device("cuda")
    return device


def test_published_cuda_wheel(device: torch.device) -> None:
    assert version("deepspeed") == "0.19.2+cu.12.8.torch.2.10"
    assert torch.__version__ == "2.10.0+cu128"
    assert torch.version.cuda == "12.8"
    assert torch.cuda.get_device_name(device)


@pytest.mark.parametrize(
    "module_name", ["deepspeed", "deepspeed.accelerator", "deepspeed.ops"]
)
def test_native_module(device: torch.device, module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


def test_cuda_accelerator(device: torch.device) -> None:
    from deepspeed.accelerator import get_accelerator

    accelerator = get_accelerator()
    assert accelerator.device_name() == "cuda"
    assert accelerator.is_available()


def test_deepspeed_cuda_tensor(device: torch.device) -> None:
    from deepspeed.accelerator import get_accelerator

    accelerator = get_accelerator()
    values = torch.tensor([1.0, 2.0, 3.0], device=accelerator.device_name())
    torch.testing.assert_close(
        values.square(), torch.tensor([1.0, 4.0, 9.0], device=device)
    )
