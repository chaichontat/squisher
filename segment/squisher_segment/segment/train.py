from collections.abc import Iterable
from hashlib import md5
from pathlib import Path
from typing import Any, Literal, Pattern, Sequence, cast

import numpy as np
import torch
from cellpose.contrib.cellposetrt import trt_build
from cellpose.models import CellposeModel
from cellpose.train import train_seg as train_seg_transformer
from cellpose.train_unet import train_seg as train_seg_unet
from cellpose.unet import CellposeUNetModel
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from squisher_segment.segment.augment import PhotometricConfig, build_batch_augmenter
from squisher_segment.segment.data_discovery import _compile_patterns, _discover_training_dirs, _matches_any
from squisher_segment.segment.model_artifacts import plan_path_for_device


def _cleanup_model_artifacts(model_path: Path) -> None:
    """Remove stale TensorRT artifacts for a trained model.

    Deletes device-specific `.plan` files first, followed by any ONNX files
    derived from the same model name. This ensures we rebuild engines from a
    clean slate after training.
    """

    model_dir = model_path.parent
    base_name = model_path.name

    plan_files = sorted(model_dir.glob(f"{base_name}-*.plan"))
    onnx_files = sorted(model_dir.glob(f"{base_name}*.onnx"))

    if plan_files or onnx_files:
        logger.info(
            f"Removing existing TensorRT artifacts for {base_name}: "
            f"plans={[p.name for p in plan_files]}, onnx={[o.name for o in onnx_files]}"
        )

    for plan_file in plan_files:
        plan_file.unlink(missing_ok=True)

    for onnx_file in onnx_files:
        onnx_file.unlink(missing_ok=True)


def _compute_model_md5(model_path: Path) -> str:
    hasher = md5(usedforsecurity=False)
    with model_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_model: str | None
    # Backend selector: 'sam' (v4 transformer) or 'unet' (legacy UNet).
    backend: Literal["sam", "unet"] = "sam"
    channels: tuple[int, ...]
    training_paths: list[str]
    diameter: float | None = 60  # target rescale diameter (px) for SAM training; if None, infer per-image
    # Optional explicit test set roots. Accepts a single string or list of strings
    # using the same semantics as `training_paths` (relative to training root,
    # may point to files or directories; image directories are discovered by
    # scanning for supported image/mask files one level below as needed).
    test_folder: str | list[str] | None = None
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    n_epochs: int = 200

    learning_rate: float = 0.008
    batch_size: int = 16
    weight_decay: float = 1e-5

    normalization_percs: tuple[float, float] = (1, 99.5)
    train_losses: list[float] = Field(default_factory=list)
    packed: bool = False
    pack_k: int = 3
    pack_guard: int = 16
    pack_stripe_height: int | None = 68
    skip_trt: bool = False
    model_md5: str | None = None


def _filter_images_by_patterns(
    *,
    images: list,
    labels: list,
    image_names: Sequence[str | Path] | None,
    include_regexes: list[Pattern[str]] | None,
    exclude_regexes: list[Pattern[str]] | None,
    training_root: Path,
    resolved_sample_map: dict[Path, Path],
) -> tuple[list, list, list | None, int, int, int]:
    """Filter image/label pairs by include/exclude regexes.

    Preserves existing behavior and logs while isolating complexity.

    Returns
    -------
    filtered_images, filtered_labels, filtered_image_names, total_images, kept_images, excluded_images
    """
    if not (include_regexes or exclude_regexes):
        total = len(images)
        return (
            images,
            labels,
            list(image_names) if image_names is not None else None,
            total,
            total,
            0,
        )

    if image_names is None:
        raise ValueError("Cellpose did not provide image names; include/exclude filters require them.")

    # Normalize raw names and compute alias/resolved candidates for robust matching
    original_strings = [
        (name.as_posix() if isinstance(name, Path) else str(name)).replace("\\", "/") for name in image_names
    ]

    def _normalize_image_name(raw: Path | str) -> tuple[str, str]:
        raw_string = raw.as_posix() if isinstance(raw, Path) else str(raw)
        normalized_string = raw_string.replace("\\", "/")
        normalized_path = Path(normalized_string)

        if normalized_path.is_absolute():
            resolved_path = normalized_path.resolve(strict=False)
        else:
            resolved_path = (training_root / normalized_path).resolve(strict=False)

        alias_candidate: str | None = None
        for resolved_root, alias in resolved_sample_map.items():
            if resolved_path.is_relative_to(resolved_root):
                rebased = training_root / alias / resolved_path.relative_to(resolved_root)
                alias_candidate = rebased.as_posix()
                break

        return alias_candidate or normalized_string, resolved_path.as_posix()

    normalized_infos = [_normalize_image_name(name) for name in image_names]

    # records: (img, lbl, original_name, original_string, alias_string, resolved_string)
    records = [
        (img, lbl, original_name, original_string, alias_res[0], alias_res[1])
        for img, lbl, original_name, original_string, alias_res in zip(
            images, labels, image_names, original_strings, normalized_infos
        )
    ]

    def _record_candidates(record: tuple) -> tuple[str, str, str]:
        return record[3], record[4], record[5]

    def _patterns_match(patterns: list[Pattern[str]] | None, record: tuple) -> bool:
        if not patterns:
            return False
        candidates = _record_candidates(record)
        return any(regex.search(candidate) for regex in patterns for candidate in candidates)

    dropped_by_include = [
        record[4] for record in records if include_regexes and not _patterns_match(include_regexes, record)
    ]
    candidate_records = [
        record for record in records if not include_regexes or _patterns_match(include_regexes, record)
    ]
    dropped_by_exclude = [
        record[4] for record in candidate_records if _patterns_match(exclude_regexes, record)
    ]

    filtered_records = [
        record
        for record in candidate_records
        if not exclude_regexes or not _patterns_match(exclude_regexes, record)
    ]

    if dropped_by_include:
        logger.info(
            f"Skipped {len(dropped_by_include)} image paths due to include filters: {dropped_by_include}"
        )

    if dropped_by_exclude:
        logger.info(
            f"Skipped {len(dropped_by_exclude)} image paths due to exclude filters: {dropped_by_exclude}"
        )

    if not filtered_records:
        raise ValueError(
            "No training images remain after applying include/exclude filters."
            f" include={[r.pattern for r in include_regexes] if include_regexes else []},"
            f" exclude={[r.pattern for r in exclude_regexes] if exclude_regexes else []}"
        )

    filtered_images = [record[0] for record in filtered_records]
    filtered_labels = [record[1] for record in filtered_records]
    filtered_image_names = [record[2] for record in filtered_records]

    total_images = len(images)
    kept_images = len(filtered_images)
    excluded_images = total_images - kept_images

    return (
        filtered_images,
        filtered_labels,
        filtered_image_names,
        total_images,
        kept_images,
        excluded_images,
    )


def _prepare_2d_training_images(images: Iterable[Any]) -> tuple[list[Any], int, int]:
    """Return images with channel-first arrays normalized for Cellpose 2D training.

    Cellpose expects channel-first data when `channel_axis=0`. We trim any extra
    channels up front to avoid accidental use of higher indices that might exist
    in the raw stacks. Singleton channel-first arrays are squeezed to 2D because
    Cellpose expands `(1, Y, X)` to a 4D array in its 2D training path.
    """

    prepared: list[Any] = []
    trimmed = 0
    squeezed = 0
    for image in images:
        if isinstance(image, np.ndarray) and image.ndim >= 3:
            if image.shape[0] == 1:
                prepared.append(image[0].copy())
                squeezed += 1
                continue
            if image.shape[0] > 2:
                prepared.append(image[:2].copy())
                trimmed += 1
                continue
        prepared.append(image)
    return prepared, trimmed, squeezed


# models = sorted(
#     (file for file in (path / "models").glob("*") if "train_losses" not in file.name),
#     key=lambda x: x.stat().st_mtime,
#     reverse=True,
# )


def concat_output(
    path: Path,
    samples: list[str],
    mask_filter: list[str] | str = "_seg.npy",
    look_one_level_down: bool = False,
) -> tuple[list, ...]:
    from .cp_io import load_train_test_data

    if isinstance(mask_filter, list) and len(samples) != len(mask_filter):
        raise ValueError("Number of samples must match number of mask filters.")

    first: tuple = load_train_test_data(
        (path / samples[0]).as_posix(),
        mask_filter=mask_filter if isinstance(mask_filter, str) else mask_filter[0],
        look_one_level_down=look_one_level_down,
    )
    for index, sample in enumerate(samples[1:], 1):
        _out = load_train_test_data(
            (path / sample).as_posix(),
            # test_dir=(path.parent / "pi-wgatest").as_posix(),
            mask_filter=mask_filter if isinstance(mask_filter, str) else mask_filter[index],
            look_one_level_down=look_one_level_down,
        )

        for lis, new in zip(first, _out):
            if lis is not None and new is not None:
                lis.extend(new)
    if first[0] is not None and first[2] is not None:
        assert len(first[0]) == len(first[2])

    return first


def _concat_images_only(
    path: Path,
    samples: list[str],
    *,
    mask_filter: str = "_seg.npy",
    look_one_level_down: bool = False,
) -> tuple[list, list, list]:
    """Load and concatenate only images/labels/names from multiple sample roots.

    Mirrors the behavior of `concat_output` for training data but does not
    attempt to load per-sample test data; instead, it simply concatenates the
    loaded triplets for the provided `samples`.
    """
    from .cp_io import load_images_labels

    images_all: list = []
    labels_all: list = []
    names_all: list = []

    for sample in samples:
        sample_dir = (path / sample).as_posix()
        images, labels, names = load_images_labels(
            sample_dir, mask_filter=mask_filter, image_filter=None, look_one_level_down=look_one_level_down
        )
        if images is not None:
            images_all.extend(images)
        if labels is not None:
            labels_all.extend(labels)
        if names is not None:
            names_all.extend(names)

    if images_all is not None:
        assert len(images_all) == len(labels_all)

    return images_all, labels_all, names_all


def build_trt_engine(
    *,
    model_path: Path,
    bsize: int,
    device: torch.device,
    batch_size: int = 1,
    backend: Literal["sam", "unet"] = "sam",
    opset: int = 22,
) -> Path:
    device_name_raw = torch.cuda.get_device_name(device)
    base_name = model_path.name
    plan_path = plan_path_for_device(model_path, device_name_raw)
    onnx_path = model_path.with_name(f"{base_name}.onnx")
    device_props = torch.cuda.get_device_properties(device)
    vram_mb = max(1024, int(device_props.total_memory / (1024 * 1024)))
    logger.info(
        f"Building TensorRT engine {plan_path.name} (batch_size={batch_size}, bsize={bsize}, device={device_name_raw}, vram={vram_mb} MB)"
    )

    # Skip ONNX export if ONNX file exists and is newer than the model file
    if onnx_path.exists() and onnx_path.stat().st_mtime > model_path.stat().st_mtime:
        logger.info(f"Skipping ONNX export: {onnx_path.name} is up-to-date")
    else:
        trt_build.export_onnx(
            str(model_path),
            str(onnx_path),
            batch_size=batch_size,
            bsize=bsize,
            opset=opset,
            backend=backend,
        )
    trt_build.build_engine(
        str(onnx_path),
        str(plan_path),
        batch_size=batch_size,
        bsize=bsize,
        vram=vram_mb,
    )
    return plan_path


def _train(out: tuple[Any, ...], path: Path, name: str, train_config: TrainConfig):
    images, labels, image_names, test_images, test_labels, image_names_test = out

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for training but was not detected.")

    device = torch.device("cuda")

    if not train_config.base_model:
        if train_config.backend == "sam":
            logger.info("No base_model specified; defaulting to 'cpsam' for SAM backend.")
            train_config = train_config.model_copy(update={"base_model": "cpsam"})
        else:
            logger.info("No base_model specified; defaulting to 'cyto3' for UNet backend.")
            train_config = train_config.model_copy(update={"base_model": "cyto3"})

    if train_config.backend == "unet":
        model = CellposeUNetModel(
            gpu=True,
            pretrained_model=train_config.base_model,
            device=device,
        )
        # Always train in bf16 for UNet as well
        model.net.to(torch.bfloat16)
        setattr(model.net, "dtype", torch.bfloat16)
        train_seg_fn = train_seg_unet
    else:
        model = CellposeModel(
            gpu=True,
            pretrained_model=train_config.base_model,
            device=device,
            use_bfloat16=True,
        )
        train_seg_fn = train_seg_transformer

    # Photometric augmenter: pick 1–2 ops per image, applied before packing
    phot_aug = build_batch_augmenter(
        cfg=PhotometricConfig(),
        seed=0,
        min_ops_per_image=1,
        max_ops_per_image=2,
    )

    pack_kwargs: dict[str, Any] = {
        "pack_to_single_tile": train_config.packed,
        "pack_k": train_config.pack_k,
        "pack_guard": train_config.pack_guard,
        "pack_stripe_height": train_config.pack_stripe_height,
        "pack_stripe_border": 0,
    }
    if train_config.backend == "unet":
        pack_kwargs["channels"] = train_config.channels

    extra_train_kwargs: dict[str, Any] = {}
    if train_config.backend == "sam" and train_config.diameter is not None:
        extra_train_kwargs["diameter_override"] = train_config.diameter
        model_diam = getattr(model.net, "diam_mean", None)
        model_diam_val = float(model_diam.item()) if model_diam is not None else None
        if model_diam_val is not None:
            logger.info(
                f"Using diameter_override={train_config.diameter:.1f}px; model diam_mean={model_diam_val:.1f}px"
            )
        else:
            logger.info(
                f"Using diameter_override={train_config.diameter:.1f}px (model diam_mean unavailable)"
            )

    model_path, train_losses, test_losses = train_seg_fn(
        model.net,
        train_data=images,
        train_labels=labels,
        save_path=path,  # /models automatically appended
        batch_size=train_config.batch_size,
        channel_axis=0,
        test_data=test_images,
        test_labels=test_labels,
        weight_decay=train_config.weight_decay,
        SGD=False,
        learning_rate=train_config.learning_rate,
        rescale=True,
        n_epochs=train_config.n_epochs,
        model_name=name,
        normalize=cast(bool, {"percentile": train_config.normalization_percs}),
        min_train_masks=4,
        batch_photom_augment=phot_aug,
        **pack_kwargs,
        **extra_train_kwargs,
    )

    model_path = Path(model_path)
    if train_config.skip_trt:
        logger.info("Skipping TensorRT engine generation after training.")
    else:
        _cleanup_model_artifacts(model_path)
        build_trt_engine(
            model_path=model_path,
            bsize=256,
            device=device,
            batch_size=1,
            backend=train_config.backend,
        )

    return model_path, train_losses, test_losses


def run_train(name: str, path: Path, train_config: TrainConfig) -> TrainConfig:
    logger.info(f"Started training {name} with paths: {train_config.training_paths}")

    include_regexes = _compile_patterns(train_config.include) if train_config.include else []
    exclude_regexes = _compile_patterns(train_config.exclude) if train_config.exclude else []

    discovered_samples = _discover_training_dirs(path, train_config.training_paths)
    if exclude_regexes:
        kept_samples: list[Path] = []
        dropped_samples: list[Path] = []

        for sample in discovered_samples:
            alias_path = path / sample
            resolved_path = alias_path.resolve(strict=False)
            sample_values = (
                sample.as_posix(),
                alias_path.as_posix(),
                resolved_path.as_posix(),
            )
            if any(_matches_any(exclude_regexes, value) for value in sample_values):
                dropped_samples.append(sample)
            else:
                kept_samples.append(sample)

        if dropped_samples:
            logger.info(
                f"Skipped {len(dropped_samples)} training directories due to exclude filters: "
                f"{[sample.as_posix() for sample in dropped_samples]}"
            )

        discovered_samples = kept_samples

    if not discovered_samples:
        raise ValueError(
            "No training images remain after applying include/exclude filters."
            f" include={train_config.include}, exclude={train_config.exclude}"
        )

    resolved_sample_map: dict[Path, Path] = {}
    for sample in discovered_samples:
        alias_path = path / sample
        try:
            resolved_target = alias_path.resolve(strict=True)
        except FileNotFoundError:
            resolved_target = alias_path.resolve(strict=False)
        resolved_sample_map.setdefault(resolved_target, sample)

    logger.info(
        f"Loading training data from {len(discovered_samples)} discovered directories: "
        f"{[sample.as_posix() for sample in discovered_samples]}"
    )

    out2 = concat_output(
        path,
        [sample.as_posix() for sample in discovered_samples],
        mask_filter="_seg.npy",
        look_one_level_down=False,
    )

    logger.info(f"Pre-filter training discovered {len(out2[0])} images")
    images, labels, image_names, test_images, test_labels, image_names_test = out2

    (
        filtered_images,
        filtered_labels,
        filtered_image_names,
        total_images,
        kept_images,
        excluded_images,
    ) = _filter_images_by_patterns(
        images=images,
        labels=labels,
        image_names=image_names,
        include_regexes=include_regexes,
        exclude_regexes=exclude_regexes,
        training_root=path,
        resolved_sample_map=resolved_sample_map,
    )

    logger.info(f"Filter summary: kept {kept_images}/{total_images} images; excluded {excluded_images}.")

    limited_train_images, trimmed_train, squeezed_train = _prepare_2d_training_images(filtered_images)

    # Optional explicit test set discovery/loading using the same semantics as training
    final_test_images: list[Any] | None = None
    final_test_labels: list[Any] | None = None
    final_test_names: list[Any] | None = None
    trimmed_test = 0
    squeezed_test = 0

    if train_config.test_folder is not None:
        # Normalize to list[str]
        test_entries: list[str] = (
            [train_config.test_folder]
            if isinstance(train_config.test_folder, str)
            else list(train_config.test_folder)
        )

        discovered_test_samples = _discover_training_dirs(path, test_entries)

        if exclude_regexes:
            kept_test_samples: list[Path] = []
            dropped_test_samples: list[Path] = []
            for sample in discovered_test_samples:
                alias_path = path / sample
                resolved_path = alias_path.resolve(strict=False)
                sample_values = (
                    sample.as_posix(),
                    alias_path.as_posix(),
                    resolved_path.as_posix(),
                )
                if any(_matches_any(exclude_regexes, value) for value in sample_values):
                    dropped_test_samples.append(sample)
                else:
                    kept_test_samples.append(sample)

            if dropped_test_samples:
                logger.info(
                    f"Skipped {len(dropped_test_samples)} test directories due to exclude filters: {[s.as_posix() for s in dropped_test_samples]}"
                )

            discovered_test_samples = kept_test_samples

        if not discovered_test_samples:
            raise ValueError(
                "No test images remain after applying include/exclude filters."
                f" include={train_config.include}, exclude={train_config.exclude}"
            )

        resolved_test_sample_map: dict[Path, Path] = {}
        for sample in discovered_test_samples:
            alias_path = path / sample
            try:
                resolved_target = alias_path.resolve(strict=True)
            except FileNotFoundError:
                resolved_target = alias_path.resolve(strict=False)
            resolved_test_sample_map.setdefault(resolved_target, sample)

        logger.info(
            f"Loading test data from {len(discovered_test_samples)} discovered directories: "
            f"{[sample.as_posix() for sample in discovered_test_samples]}"
        )

        test_images_raw, test_labels_raw, test_names_raw = _concat_images_only(
            path,
            [sample.as_posix() for sample in discovered_test_samples],
            mask_filter="_seg.npy",
            look_one_level_down=False,
        )

        (
            filtered_test_images,
            filtered_test_labels,
            filtered_test_names,
            total_test_images,
            kept_test_images,
            excluded_test_images,
        ) = _filter_images_by_patterns(
            images=test_images_raw,
            labels=test_labels_raw,
            image_names=test_names_raw,
            include_regexes=include_regexes,
            exclude_regexes=exclude_regexes,
            training_root=path,
            resolved_sample_map=resolved_test_sample_map,
        )

        logger.info(
            f"Test filter summary: kept {kept_test_images}/{total_test_images} images; excluded {excluded_test_images}."
        )

        if filtered_test_images:
            final_test_images, trimmed_test, squeezed_test = _prepare_2d_training_images(
                filtered_test_images
            )
            final_test_labels = filtered_test_labels
            final_test_names = filtered_test_names
        elif include_regexes or exclude_regexes:
            raise ValueError(
                "No test images remain after applying include/exclude filters."
                f" include={train_config.include}, exclude={train_config.exclude}"
            )
        else:
            logger.info("No test images discovered; proceeding without a test set.")
    else:
        # Fall back to whatever came back from the original loader (usually None)
        if test_images is not None:
            final_test_images, trimmed_test, squeezed_test = _prepare_2d_training_images(test_images)
            final_test_labels = test_labels
            final_test_names = image_names_test

    if trimmed_train or trimmed_test:
        logger.info(
            f"Restricted image channels to first two for {trimmed_train} training and {trimmed_test} test samples."
        )
    if squeezed_train or squeezed_test:
        logger.info(
            f"Squeezed singleton channel axis for {squeezed_train} training and {squeezed_test} test samples."
        )

    filtered: tuple[list, ...] = (
        limited_train_images,
        filtered_labels,
        filtered_image_names if filtered_image_names is not None else None,
        final_test_images if final_test_images is not None else None,
        final_test_labels,
        final_test_names,
    )

    if not filtered[0]:
        raise ValueError(
            "No training images remain after applying include/exclude filters."
            f" include={train_config.include}, exclude={train_config.exclude}"
        )

    model_path, train_losses, _test_losses = _train(filtered, path, train_config=train_config, name=name)

    model_md5 = _compute_model_md5(model_path)
    logger.info(f"Model MD5 checksum (md5sum {model_path.name}) = {model_md5}")

    normalized_train_losses = train_losses.tolist() if hasattr(train_losses, "tolist") else list(train_losses)

    return train_config.model_copy(update={"train_losses": normalized_train_losses, "model_md5": model_md5})
