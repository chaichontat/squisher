import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from squisher_deconv.basic_profiles import compose_z_profile
from squisher_lightsheet._legacy.stitch_20x_tl_multiview import load_inverse_flatfield


def test_fusion_requires_z_profiles_to_be_applied_before_fusion(tmp_path):
    base = tmp_path / 'base.pkl'
    base.write_bytes(pickle.dumps({'basic': SimpleNamespace(flatfield=np.ones((4, 5)), darkfield=np.zeros((4, 5)))}))
    profile = tmp_path / 'field-ch0.pkl'
    compose_z_profile(base, profile, coefficient=np.zeros((2, 5)), provenance={})
    tifffile.imwrite(tmp_path / 'field-ch0-flatfield.tif', np.ones((4, 5), dtype=np.float32))
    with pytest.raises(ValueError, match='squisher-deconv before fusion'):
        load_inverse_flatfield(tmp_path, 0, (11, 4, 5))
