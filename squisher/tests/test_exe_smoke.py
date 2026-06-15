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
def smoke_czis(tmp_path: Path) -> list[Path]:
    data = np.arange(2 * 3 * 16 * 16, dtype=np.uint16).reshape((2, 3, 16, 16))
    paths = [tmp_path / "sample_a.czi", tmp_path / "sample_b.czi"]

    for path in paths:
        with czi.create_czi(str(path)) as writer:
            for channel_index in range(data.shape[0]):
                for z_index in range(data.shape[1]):
                    assert writer.write(
                        data[channel_index, z_index],
                        plane={"C": channel_index, "Z": z_index},
                        scene=0,
                    )

    return paths


def test_windows_exe_compresses_smoke_czi_wildcard(
    squisher_exe: Path,
    smoke_czis: list[Path],
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            str(squisher_exe),
            "compress",
            str(tmp_path / "sample_*.czi"),
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
    for czi_path in smoke_czis:
        out = czi_path.with_suffix(".ome.tif")
        assert out.exists()
        assert czi_path.exists()

        with TiffFile(out) as tif:
            assert len(tif.pages) == 6
            assert all(page.compression == 22610 for page in tif.pages)
            assert all(page.is_tiled for page in tif.pages)

        verify = subprocess.run(
            [str(squisher_exe), "verify", str(czi_path), "--decode-samples"],
            check=False,
            capture_output=True,
            text=True,
        )

        assert verify.returncode == 0, verify.stdout + verify.stderr


def test_windows_exe_delete_removes_source_after_success(
    squisher_exe: Path,
    smoke_czis: list[Path],
) -> None:
    czi_path = smoke_czis[0]
    result = subprocess.run(
        [
            str(squisher_exe),
            "compress",
            str(czi_path),
            "--level",
            "90",
            "--tile-size",
            "16",
            "--tiff-maxworkers",
            "1",
            "--czi-tile-workers",
            "1",
            "--delete",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert czi_path.with_suffix(".ome.tif").exists()
    assert not czi_path.exists()
