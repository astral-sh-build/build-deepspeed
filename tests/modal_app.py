import subprocess
from pathlib import Path

import modal

TEST_DIRECTORY = Path(__file__).parent.resolve()

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("libaio-dev")
    .uv_sync(uv_project_dir=str(TEST_DIRECTORY))
    .add_local_file(
        TEST_DIRECTORY / "test_deepspeed.py",
        remote_path="/gpu-tests/test_deepspeed.py",
    )
)

app = modal.App("astral-build-deepspeed-gpu-tests")


@app.function(image=image, gpu="A10G", timeout=900)
def test() -> None:
    subprocess.run(
        ["python", "-m", "pytest", "-v", "/gpu-tests/test_deepspeed.py"],
        check=True,
    )
