from __future__ import annotations

import numpy as np
import pytest

from squisher_deconv.deconvolution import IdentityDeconvolver, infer_psf_halo_many


def test_identity_deconvolver_preserves_shape_and_values() -> None:
    deconvolver = IdentityDeconvolver()
    volume = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)

    out = deconvolver.deconvolve(volume)

    assert out.shape == volume.shape
    assert out.dtype == np.float32
    assert np.array_equal(out, volume.astype(np.float32))


def test_identity_deconvolver_rejects_non_zyx_volume() -> None:
    deconvolver = IdentityDeconvolver()

    with pytest.raises(ValueError, match="Expected \\(Z, C, Y, X\\) volume"):
        deconvolver.deconvolve(np.zeros((2, 3, 4), dtype=np.uint16))


def test_infer_psf_halo_many_requires_at_least_one_path() -> None:
    with pytest.raises(ValueError, match="At least one PSF path is required"):
        infer_psf_halo_many([])
