from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np

from squisher_deconv.basic_profiles import compose_basic_profile


def test_compose_basic_profile_divides_flatfield_and_preserves_darkfield(tmp_path):
    source = tmp_path / "multi-ch0.pkl"
    output = tmp_path / "corrected.pkl"
    flatfield = np.full((4, 5), 2.0, dtype=np.float32)
    darkfield = np.arange(20, dtype=np.float32).reshape(4, 5)
    payload = {
        "basic": SimpleNamespace(flatfield=flatfield.copy(), darkfield=darkfield.copy()),
        "kept": {"value": 7},
    }
    source.write_bytes(pickle.dumps(payload))
    mask = np.linspace(0.8, 1.2, 20, dtype=np.float32).reshape(4, 5)

    result = compose_basic_profile(
        source,
        output,
        mask=mask,
        provenance={"model": "post-basic"},
    )

    loaded = pickle.loads(output.read_bytes())
    np.testing.assert_allclose(loaded["basic"].flatfield, flatfield / mask)
    np.testing.assert_array_equal(loaded["basic"].darkfield, darkfield)
    assert loaded["kept"] == {"value": 7}
    assert loaded["residual_correction"] == {"model": "post-basic"}
    np.testing.assert_allclose(result.flatfield, flatfield / mask)
    np.testing.assert_array_equal(result.darkfield, darkfield)


def test_compose_basic_profile_rejects_invalid_mask_before_writing(tmp_path):
    source = tmp_path / "multi-ch0.pkl"
    output = tmp_path / "corrected.pkl"
    source.write_bytes(
        pickle.dumps(
            {
                "basic": SimpleNamespace(
                    flatfield=np.ones((3, 4), dtype=np.float32),
                    darkfield=np.zeros((3, 4), dtype=np.float32),
                )
            }
        )
    )

    for mask in (
        np.ones((4, 3), dtype=np.float32),
        np.zeros((3, 4), dtype=np.float32),
        np.full((3, 4), np.nan, dtype=np.float32),
    ):
        try:
            compose_basic_profile(source, output, mask=mask, provenance={})
        except ValueError:
            pass
        else:
            raise AssertionError("invalid mask was accepted")
        assert not output.exists()
