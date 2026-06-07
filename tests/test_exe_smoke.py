from pathlib import Path
import os
import subprocess

import numpy as np
import pytest
from pylibCZIrw import czi
from tifffile import TiffFile


@pytest.fixture
def squisher_exe() -> Path:
    exe = os.environ.get("SQUISHER_EXE")
    if exe is None:
        pytest.skip("set SQUISHER_EXE to run executable smoke tests")

    path = Path(exe)
    if not path.exists():
        pytest.fail(f"SQUISHER_EXE does not exist: {path}")
    return path


@pytest.fixture
def smoke_czi(tmp_path: Path) -> Path:
    path = tmp_path / "sample.czi"
    data = np.arange(2 * 3 * 16 * 16, dtype=np.uint16).reshape((2, 3, 16, 16))

    with czi.create_czi(str(path)) as writer:
        for channel_index in range(data.shape[0]):
            for z_index in range(data.shape[1]):
                assert writer.write(
                    data[channel_index, z_index],
                    plane={"C": channel_index, "Z": z_index},
                    scene=0,
                )

    return path


def test_windows_exe_compresses_smoke_czi(squisher_exe: Path, smoke_czi: Path, tmp_path: Path) -> None:
    result = subprocess.run(
        [
            str(squisher_exe),
            "compress",
            str(smoke_czi),
            "--level",
            "90",
            "--tile-size",
            "16",
            "--tiff-maxworkers",
            "1",
            "--czi-tile-workers",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    out = tmp_path / "sample.ome.tif"
    assert out.exists()

    with TiffFile(out) as tif:
        assert tif.pages[0].compression == 22610
        assert tif.pages[0].is_tiled

    verify = subprocess.run(
        [str(squisher_exe), "verify", str(smoke_czi), "--decode-samples"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert verify.returncode == 0, verify.stdout + verify.stderr
