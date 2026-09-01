from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from squisher_lightsheet.channel_affine import write_global_channel_affine_registration
from squisher_lightsheet.native_reg3dgpu import DEFAULT_LIB_DIR


DEFAULT_SWEEP_RUNNER = Path(
    "/home/chaichontat/nvme/lightsheet/scripts/run_fused_fixed_method8_sweep.py"
)
SUMMARY_NAME = "fused_fixed_method8_summary.json"


def _resolved_path(value: object, *, key: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Method 6 summary has no {key!r} path")
    return Path(value).resolve()


def validate_method6_summary(
    summary_path: Path,
    *,
    moving_position: Path,
    moving_source_position: Path,
    fixed_fused: Path,
    moving_channel: int,
    fixed_mask_threshold: float,
) -> dict[str, Any]:
    """Reject cached output unless it matches the canonical Method 6 contract."""
    payload = json.loads(summary_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{summary_path} does not contain a JSON object")
    cache = payload.get("cache_config")
    if not isinstance(cache, dict):
        raise ValueError(f"{summary_path} has no cache_config object")

    expected_scalars = {
        "native_method": "method6",
        "fit_intensity_transform": "log1p",
        "level0_initializer": "level2-method8",
        "moving_channel": moving_channel,
    }
    for key, expected in expected_scalars.items():
        actual = cache.get(key)
        if actual != expected:
            raise ValueError(
                f"Method 6 summary contract mismatch for {key}: expected {expected!r}, got {actual!r}"
            )
    actual_threshold = cache.get("fixed_mask_threshold")
    if actual_threshold is None or float(actual_threshold) != float(fixed_mask_threshold):
        raise ValueError(
            "Method 6 summary contract mismatch for fixed_mask_threshold: "
            f"expected {fixed_mask_threshold}, got {actual_threshold!r}"
        )

    expected_paths = {
        "moving_position": moving_position,
        "moving_source_position": moving_source_position,
        "fixed_fused": fixed_fused,
    }
    for key, expected in expected_paths.items():
        actual = _resolved_path(cache.get(key), key=key)
        if actual != expected.resolve():
            raise ValueError(
                f"Method 6 summary contract mismatch for {key}: "
                f"expected {expected.resolve()}, got {actual}"
            )
    return payload


def run_fused_fixed_method6(
    *,
    fixed_position: Path,
    moving_position: Path,
    moving_source_position: Path,
    fixed_fused: Path,
    output_dir: Path,
    output_registration: Path,
    moving_channel: int,
    fixed_mask_threshold: float,
    source_label: str,
    target_label: str,
    sweep_runner: Path = DEFAULT_SWEEP_RUNNER,
    window_filter_json: Path | None = None,
    core_shape_zyx: tuple[int, int, int] = (480, 480, 480),
    window_shape_zyx: tuple[int, int, int] = (528, 528, 528),
    fit_downsample_zyx: tuple[int, int, int] = (1, 1, 1),
    native_lib_dir: Path = DEFAULT_LIB_DIR,
    ftol: float = 1e-4,
    max_iterations: int = 300,
    phase_upsample_factor: int = 10,
    min_corr: float = 0.15,
    min_grad_ncc: float = 0.24,
    fixed_mask_level: int = 2,
    fixed_mask_min_voxels: int = 256,
    fixed_mask_max_masked_fraction: float = 0.95,
    workers: int = 1,
    max_tasks_per_worker: int = 10,
    devices: tuple[int, ...] = (0,),
    max_windows: int | None = None,
    resume: bool = True,
) -> tuple[Path, Path]:
    """Run the Prod2-style fused-fixed Method 6 fit and emit a canonical registration."""
    command = [
        sys.executable,
        "-u",
        str(sweep_runner.resolve()),
        "--fixed-position",
        str(fixed_position.resolve()),
        "--moving-position",
        str(moving_position.resolve()),
        "--moving-source-position",
        str(moving_source_position.resolve()),
        "--fixed-fused",
        str(fixed_fused.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--core-shape-zyx",
        ",".join(map(str, core_shape_zyx)),
        "--window-shape-zyx",
        ",".join(map(str, window_shape_zyx)),
        "--fit-downsample-zyx",
        ",".join(map(str, fit_downsample_zyx)),
        "--moving-channel",
        str(moving_channel),
        "--native-lib-dir",
        str(native_lib_dir.resolve()),
        "--native-method",
        "method6",
        "--starting-affine-matrix-zyx",
        "1,0,0,0,1,0,0,0,1",
        "--fit-intensity-transform",
        "log1p",
        "--ftol",
        str(ftol),
        "--max-iterations",
        str(max_iterations),
        "--phase-upsample-factor",
        str(phase_upsample_factor),
        "--min-corr",
        str(min_corr),
        "--min-grad-ncc",
        str(min_grad_ncc),
        "--fixed-mask-threshold",
        str(fixed_mask_threshold),
        "--fixed-mask-level",
        str(fixed_mask_level),
        "--fixed-mask-min-voxels",
        str(fixed_mask_min_voxels),
        "--fixed-mask-max-masked-fraction",
        str(fixed_mask_max_masked_fraction),
        "--workers",
        str(workers),
        "--max-tasks-per-worker",
        str(max_tasks_per_worker),
        "--devices",
        ",".join(map(str, devices)),
        "--level0-initializer",
        "level2-method8",
        "--resume" if resume else "--no-resume",
    ]
    if window_filter_json is not None:
        command.extend(["--window-filter-json", str(window_filter_json.resolve())])
    if max_windows is not None:
        command.extend(["--max-windows", str(max_windows)])

    subprocess.run(command, check=True)
    summary_path = output_dir / SUMMARY_NAME
    if not summary_path.is_file():
        raise RuntimeError(f"Method 6 sweep completed without writing {summary_path}")
    validate_method6_summary(
        summary_path,
        moving_position=moving_position,
        moving_source_position=moving_source_position,
        fixed_fused=fixed_fused,
        moving_channel=moving_channel,
        fixed_mask_threshold=fixed_mask_threshold,
    )
    registration = write_global_channel_affine_registration(
        window_dir=output_dir / "window_json",
        reference_registration_input=moving_source_position,
        output_registration=output_registration,
        expected_moving_channel=moving_channel,
        expected_fixed_fused=fixed_fused,
        source_label=source_label,
        target_label=target_label,
    )
    return summary_path.resolve(), registration
