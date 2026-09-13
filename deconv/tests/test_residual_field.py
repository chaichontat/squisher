import numpy as np
import pytest

from squisher_deconv.residual_field import cosine_basis, residual_block, residual_plane, validate_coefficients


def test_raw_z_changes_field_and_slab_coordinates_match():
    coefficient = np.zeros((2, 5))
    coefficient[0, 0] = 0.1
    coefficient[1, 0] = 0.4
    first = residual_plane(coefficient, z=0, shape_zyx=(11, 9, 13))
    middle = residual_plane(coefficient, z=5, shape_zyx=(11, 9, 13))
    last = residual_plane(coefficient, z=10, shape_zyx=(11, 9, 13))
    assert first[0, 0] == pytest.approx(np.exp(0.3))
    assert middle[0, 0] == pytest.approx(np.exp(0.1))
    assert last[0, 0] == pytest.approx(np.exp(-0.1))
    crop = residual_plane(coefficient, z=5, shape_zyx=(11, 9, 13), y_slice=slice(2, 6), x_slice=slice(3, 8))
    np.testing.assert_allclose(crop, middle[2:6, 3:8])


def test_residual_block_matches_full_planes_in_original_coordinates():
    coefficient = np.zeros((2, 5))
    coefficient[0, 2] = 0.2
    coefficient[1, 0] = -0.3
    block = residual_block(
        coefficient,
        z_slice=slice(3, 7),
        shape_zyx=(11, 9, 13),
        y_slice=slice(2, 6),
        x_slice=slice(4, 10),
    )
    expected = np.stack(
        [
            residual_plane(coefficient, z=z, shape_zyx=(11, 9, 13))[2:6, 4:10]
            for z in range(3, 7)
        ]
    )
    np.testing.assert_allclose(block, expected)


def test_z_basis_preserves_shared_field_and_rejects_missing_coordinates():
    yx = np.asarray([[0.2, 0.3], [0.4, 0.5]])
    shared = cosine_basis(yx)
    varying = cosine_basis(yx, z=np.asarray([0.0, 1.0]), z_degree=1)
    np.testing.assert_allclose(varying[:, :5], shared)
    np.testing.assert_allclose(varying[:, 5:], shared * np.asarray([0.5, -0.5])[:, None])
    with pytest.raises(ValueError, match="Z coordinates"):
        cosine_basis(yx, z_degree=1)


@pytest.mark.parametrize("coefficient", [np.zeros((2, 4)), np.full((2, 5), np.nan), np.zeros(10)])
def test_invalid_model_is_rejected(coefficient):
    with pytest.raises(ValueError, match="coefficient"):
        validate_coefficients(coefficient)
