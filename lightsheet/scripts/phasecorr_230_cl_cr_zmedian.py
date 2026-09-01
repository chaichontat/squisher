from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.registration import phase_cross_correlation


ROOT = Path("/home/chaichontat/nvme/lightsheet")
INPUT_DIR = ROOT / "20260613/230Tnc-CLR-488514561638/zmedian_level2_dumb_stitch_ch0"
CL_PNG = INPUT_DIR / "230Tnc_CL_ch0_level2_zmedian_dumb_stitch.png"
CR_PNG = INPUT_DIR / "230Tnc_CR_ch0_level2_zmedian_dumb_stitch.png"
MANIFEST = INPUT_DIR / "230Tnc_CLR_ch0_level2_zmedian_dumb_stitch_manifest.json"
OUTPUT_JSON = INPUT_DIR / "230Tnc_CLR_ch0_level2_zmedian_centroid_phasecorr.json"
OUTPUT_OVERLAY = INPUT_DIR / "230Tnc_CLR_ch0_level2_zmedian_centroid_phasecorr_overlay.png"


def _normalize(image: np.ndarray) -> np.ndarray:
    values = image[np.isfinite(image) & (image > 0)]
    if values.size == 0:
        values = image[np.isfinite(image)]
    low, high = np.percentile(values, [1.0, 99.8]) if values.size else (0.0, 1.0)
    if not np.isfinite(high) or high <= low:
        high = low + 1.0
    scaled = np.clip((image - low) / (high - low), 0.0, 1.0)
    background = ndimage.gaussian_filter(scaled, sigma=8.0)
    return (scaled - background).astype(np.float32, copy=False)


def _corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    aa = a[mask].astype(np.float64, copy=False)
    bb = b[mask].astype(np.float64, copy=False)
    if aa.size < 8:
        return float("nan")
    aa -= float(aa.mean())
    bb -= float(bb.mean())
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float("nan") if denom == 0.0 else float(np.dot(aa, bb) / denom)


def _stretch(image: np.ndarray) -> np.ndarray:
    values = image[np.isfinite(image) & (image > 0)]
    if values.size == 0:
        values = image[np.isfinite(image)]
    low, high = np.percentile(values, [0.5, 99.8]) if values.size else (0.0, 1.0)
    if not np.isfinite(high) or high <= low:
        high = low + 1.0
    return np.clip((image - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)


def main() -> None:
    cl = np.asarray(Image.open(CL_PNG), dtype=np.float32)
    cr = np.asarray(Image.open(CR_PNG), dtype=np.float32)
    if cl.shape != cr.shape:
        raise ValueError(f"CL/CR image shapes differ: {cl.shape} vs {cr.shape}")

    cl_mask = cl > 0
    cr_mask = cr > 0
    initial_overlap_mask = cl_mask & cr_mask
    if np.count_nonzero(initial_overlap_mask) < 1024:
        raise ValueError(f"not enough overlap pixels: {np.count_nonzero(initial_overlap_mask)}")

    cl_reg = _normalize(cl)
    cr_reg = _normalize(cr)
    shift_yx, phase_error, phase_diff = phase_cross_correlation(
        cl_reg,
        cr_reg,
        reference_mask=cl_mask,
        moving_mask=cr_mask,
        overlap_ratio=0.1,
    )
    shift_yx = np.asarray(shift_yx, dtype=np.float64)
    shifted_cr = ndimage.shift(cr, shift=shift_yx, order=1, mode="constant", cval=0.0)
    shifted_mask = ndimage.shift(cr_mask.astype(np.float32), shift=shift_yx, order=0, mode="constant", cval=0.0) > 0.5
    after_mask = cl_mask & shifted_mask

    manifest = json.loads(MANIFEST.read_text())
    pixel_um_yx = np.asarray(manifest["pixel_um_yx"], dtype=np.float64)
    shift_um_yx = shift_yx * pixel_um_yx
    metadata_centers = manifest["metadata_centers_um_zyx"]
    initial_cr_minus_cl_um_yx = (
        np.asarray(metadata_centers["CR"], dtype=np.float64)[1:]
        - np.asarray(metadata_centers["CL"], dtype=np.float64)[1:]
    )
    corrected_cr_minus_cl_um_yx = initial_cr_minus_cl_um_yx + shift_um_yx

    overlay = np.zeros((*cl.shape, 3), dtype=np.uint8)
    overlay[..., 1] = _stretch(cl)
    overlay[..., 0] = _stretch(shifted_cr)
    Image.fromarray(overlay).save(OUTPUT_OVERLAY)

    result = {
        "artifact_type": "230Tnc_CLR_ch0_level2_zmedian_centroid_phasecorr.v1",
        "fixed": "CL",
        "moving": "CR",
        "input": {
            "cl_png": str(CL_PNG),
            "cr_png": str(CR_PNG),
            "manifest": str(MANIFEST),
        },
        "image_shape_yx": [int(v) for v in cl.shape],
        "pixel_um_yx": pixel_um_yx.astype(float).tolist(),
        "initial_centroid_cr_minus_cl_um_yx": initial_cr_minus_cl_um_yx.astype(float).tolist(),
        "phase_shift_to_apply_to_cr_yx_px": shift_yx.astype(float).tolist(),
        "phase_shift_to_apply_to_cr_yx_um": shift_um_yx.astype(float).tolist(),
        "corrected_centroid_cr_minus_cl_um_yx": corrected_cr_minus_cl_um_yx.astype(float).tolist(),
        "overlap_pixels_before": int(np.count_nonzero(initial_overlap_mask)),
        "overlap_pixels_after": int(np.count_nonzero(after_mask)),
        "corr_before_on_initial_overlap": _corr(cl, cr, initial_overlap_mask),
        "corr_after_on_shifted_overlap": _corr(cl, shifted_cr, after_mask),
        "phase_error": None if phase_error is None else float(phase_error),
        "phase_diff": None if phase_diff is None else float(phase_diff),
        "output_overlay": str(OUTPUT_OVERLAY),
        "notes": (
            "Shift is residual relative to the centroid-positioned CL/CR z-median dumb-stitch canvas. "
            "Phase correlation uses each side's native nonzero mask, not the initial overlap strip."
        ),
    }
    OUTPUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
