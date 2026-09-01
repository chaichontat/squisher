from pathlib import Path

import pytest

from squisher_segment.segment.data_discovery import _discover_training_dirs


def test_discovery_ignores_mask_tiffs(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "sample.tif").touch()

    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    (mask_dir / "sample_masks.tif").touch()

    assert _discover_training_dirs(tmp_path, ["."]) == [Path("images")]


@pytest.mark.parametrize("mask_name", ["sample_masks.tif", "sample_masks.tiff", "sample_MASKS.TIF"])
def test_discovery_rejects_mask_only_path(tmp_path: Path, mask_name: str) -> None:
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    mask_path = mask_dir / mask_name
    mask_path.touch()

    for training_path in ("masks", mask_path.relative_to(tmp_path).as_posix()):
        with pytest.raises(ValueError, match="No training directories discovered"):
            _discover_training_dirs(tmp_path, [training_path])


def test_discovery_keeps_supported_non_tiff_image(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    (sample_dir / "image.nrrd").touch()
    (sample_dir / "image_masks.tif").touch()

    assert _discover_training_dirs(tmp_path, ["sample"]) == [Path("sample")]
