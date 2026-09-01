"""Expose the packaged Method8 primitives to historical fused-fixed runners."""

from squisher_lightsheet.cross_register_method8 import (
    README_PRESEED_MATRIX_ZYX,
    _block_mean_downsample_zyx_cupy,
    _corr_gpu,
    _gradient_component_ncc_mean,
    _model_to_fit_downsample,
    _native_pull_from_fit_downsample,
    _native_window_low_content_reason,
    _robust_norm_and_content_stats_cupy,
    _warp_fit_preseed_cupy,
    gradient_component_ncc_3d_gpu,
    output_to_input_to_model,
    preseeded_level0_model,
    register_method8_device,
    zyx_to_xyz_3x4,
)
from squisher_lightsheet.native_reg3dgpu import DEFAULT_LIB_DIR

__all__ = [
    "DEFAULT_LIB_DIR",
    "README_PRESEED_MATRIX_ZYX",
    "_block_mean_downsample_zyx_cupy",
    "_corr_gpu",
    "_gradient_component_ncc_mean",
    "_model_to_fit_downsample",
    "_native_pull_from_fit_downsample",
    "_native_window_low_content_reason",
    "_robust_norm_and_content_stats_cupy",
    "_warp_fit_preseed_cupy",
    "gradient_component_ncc_3d_gpu",
    "output_to_input_to_model",
    "preseeded_level0_model",
    "register_method8_device",
    "zyx_to_xyz_3x4",
]
