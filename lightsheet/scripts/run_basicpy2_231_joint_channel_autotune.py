from __future__ import annotations

import importlib.util
import json
import pickle
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
from basicpy import BaSiC
from tifffile import imwrite


REPO = Path("/home/chaichontat/nvme/lightsheet")
INPUT_CACHE_DIR = Path(
    "/working/tmpsht/basicpy2_gpu_darkfield_sortIntensity_231Tnc_L_531638_z500_edgeReject_noContent_noAutotune/sample-cache"
)
INPUT_CACHE_LABEL = "231Tnc_L_531638_z500_edgeReject_noContent_basicpy2_gpu_darkfield_sortIntensity_noAutotune"
OUTPUT_DIR = Path(
    "/working/tmpsht/basicpy2_gpu_darkfield_sortIntensity_231Tnc_L_531638_z500_edgeReject_noContent_autotune_jointCh01"
)
LABEL = "231Tnc_L_531638_z500_edgeReject_noContent_basicpy2_gpu_darkfield_sortIntensity_autotune_jointCh01"


def load_fit_module():
    path = REPO / "scripts" / "fit_basic_ome_tiff_tiles.py"
    spec = importlib.util.spec_from_file_location("fit_basic_ome_tiff_tiles", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def copy_to_float32_stack(ch0: np.ndarray, ch1: np.ndarray, output_path: Path) -> np.ndarray:
    if ch0.shape != ch1.shape:
        raise ValueError(f"Channel cache shape mismatch: ch0={ch0.shape}, ch1={ch1.shape}")
    shape = (int(ch0.shape[0] + ch1.shape[0]), int(ch0.shape[1]), int(ch0.shape[2]))
    if output_path.exists():
        existing = np.load(output_path, mmap_mode="r+")
        if tuple(existing.shape) == shape and existing.dtype == np.float32:
            print(f"Loaded existing combined stack {output_path}: shape={existing.shape}", flush=True)
            return existing
        raise ValueError(f"Existing combined stack has unexpected shape/dtype: {existing.shape} {existing.dtype}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stack = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=shape)
    chunk = 25
    for start in range(0, ch0.shape[0], chunk):
        stop = min(start + chunk, ch0.shape[0])
        stack[start:stop] = ch0[start:stop]
        print(f"Copied ch0 slices {start}:{stop}", flush=True)
    offset = int(ch0.shape[0])
    for start in range(0, ch1.shape[0], chunk):
        stop = min(start + chunk, ch1.shape[0])
        stack[offset + start : offset + stop] = ch1[start:stop]
        print(f"Copied ch1 slices {start}:{stop}", flush=True)
    stack.flush()
    print(f"Wrote combined stack {output_path}: shape={stack.shape}", flush=True)
    return stack


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fit_module = load_fit_module()

    ch0_cache = INPUT_CACHE_DIR / f"{INPUT_CACHE_LABEL}-ch0-selected-slices.npy"
    ch1_cache = INPUT_CACHE_DIR / f"{INPUT_CACHE_LABEL}-ch1-selected-slices.npy"
    combined_path = OUTPUT_DIR / "sample-cache" / f"{LABEL}-combined-ch0ch1-selected-slices.float32.npy"
    if combined_path.exists():
        images = np.load(combined_path, mmap_mode="r+")
        if images.dtype != np.float32 or images.ndim != 3:
            raise ValueError(f"Existing combined stack has unexpected shape/dtype: {images.shape} {images.dtype}")
        print(f"Loaded existing combined stack {combined_path}: shape={images.shape}", flush=True)
        ch0_count = int(images.shape[0] // 2)
        ch1_count = int(images.shape[0] - ch0_count)
    else:
        ch0 = np.load(ch0_cache, mmap_mode="r")
        ch1 = np.load(ch1_cache, mmap_mode="r")
        images = copy_to_float32_stack(ch0, ch1, combined_path)
        ch0_count = int(ch0.shape[0])
        ch1_count = int(ch1.shape[0])

    basic = BaSiC(
        max_iterations=1000,
        smoothness_flatfield=1.8,
        fitting_mode="approximate",
        working_size=128,
        sort_intensity=True,
        get_darkfield=True,
        device="cuda",
    )
    print("Running joint-channel BaSiC autotune", flush=True)
    basic.autotune(images, is_timelapse=False)
    print(
        "Autotune selected "
        f"smoothness_flatfield={basic.smoothness_flatfield}, "
        f"smoothness_darkfield={basic.smoothness_darkfield}",
        flush=True,
    )
    basic.fit(images)

    flatfield = np.asarray(basic.flatfield, dtype=np.float32)
    darkfield = np.asarray(basic.darkfield, dtype=np.float32)
    for channel in (0, 1):
        channel_name = f"ch{channel}"
        imwrite(OUTPUT_DIR / f"{LABEL}-{channel_name}-flatfield.tif", flatfield)
        imwrite(OUTPUT_DIR / f"{LABEL}-{channel_name}-darkfield.tif", darkfield)

    basic = fit_module.make_basic_pickle_portable(basic)
    for channel in (0, 1):
        channel_name = f"ch{channel}"
        with (OUTPUT_DIR / f"{LABEL}-{channel_name}.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "basic": basic,
                    "path": str(OUTPUT_DIR.resolve()),
                    "name": LABEL,
                    "channel": channel_name,
                    "shared_profile": True,
                    "training_channels": ["ch0", "ch1"],
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                handle,
            )
        fit_module.plot_basic(basic)
        fit_module.plt.savefig(OUTPUT_DIR / f"{LABEL}-{channel_name}.png", dpi=150, bbox_inches="tight")
        fit_module.plt.close()

    manifest = {
        "label": LABEL,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": "Single shared BaSiC profile trained by concatenating selected ch0 and ch1 planes.",
        "input_cache_dir": str(INPUT_CACHE_DIR),
        "input_cache_label": INPUT_CACHE_LABEL,
        "input_caches": {
            "ch0": str(ch0_cache) if ch0_cache.exists() else None,
            "ch1": str(ch1_cache) if ch1_cache.exists() else None,
        },
        "combined_stack": str(combined_path),
        "combined_shape": [int(value) for value in images.shape],
        "per_channel_samples": {"ch0": ch0_count, "ch1": ch1_count},
        "basic_settings": {
            "get_darkfield": True,
            "autotune": True,
            "autotune_is_timelapse": False,
            "sort_intensity": True,
            "fitting_mode": "approximate",
            "working_size": 128,
            "device": "cuda",
            "smoothness_flatfield": float(basic.smoothness_flatfield),
            "smoothness_darkfield": float(basic.smoothness_darkfield),
        },
        "outputs": {
            "shared_profile_saved_as_channels": [0, 1],
            "flatfield_min": float(np.nanmin(flatfield)),
            "flatfield_max": float(np.nanmax(flatfield)),
            "darkfield_min": float(np.nanmin(darkfield)),
            "darkfield_max": float(np.nanmax(darkfield)),
            "finite_flatfield": bool(np.isfinite(flatfield).all()),
            "finite_darkfield": bool(np.isfinite(darkfield).all()),
        },
    }
    manifest_path = OUTPUT_DIR / f"{LABEL}-sampling.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest_path, flush=True)


if __name__ == "__main__":
    main()
