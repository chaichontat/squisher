from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


ARTIFACT_TYPE = "squisher_lightsheet.fusion_provenance.v1"
MANIFEST_RELATIVE_PATH = Path("provenance/manifest.json")
_PACKAGE_DISTRIBUTIONS = {
    "squisher-lightsheet": "squisher-lightsheet",
    "squisher": "squisher",
    "multiview-stitcher": "multiview-stitcher",
    "numpy": "numpy",
    "scipy": "scipy",
    "cupy": "cupy-cuda12x",
    "zarr": "zarr",
    "tifffile": "tifffile",
}


@dataclass
class _ArtifactCollector:
    files: dict[Path, str] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    unresolved: list[dict[str, str]] = field(default_factory=list)
    payloads: dict[Path, dict[str, Any]] = field(default_factory=dict)
    _artifact_by_source: dict[Path, dict[str, Any]] = field(default_factory=dict)
    _failed_sources: set[Path] = field(default_factory=set)

    def add_json(self, source: Path, *, role: str, required: bool = False) -> dict[str, Any] | None:
        resolved = source.expanduser().resolve()
        existing = self._artifact_by_source.get(resolved)
        if existing is not None:
            roles = existing.setdefault("roles", [existing["role"]])
            if role not in roles:
                roles.append(role)
            return self.payloads[resolved]
        if resolved in self._failed_sources:
            if required:
                raise ValueError(f"Required provenance JSON {resolved} could not be read")
            return None

        try:
            text = resolved.read_text()
            payload = _loads_source_json_object(text, context=resolved)
        except (OSError, TypeError, ValueError) as error:
            self._failed_sources.add(resolved)
            if required:
                raise ValueError(f"Required provenance JSON {resolved} could not be read: {error}") from error
            self.unresolved.append(
                {"stage": _stage_for_role(role), "role": role, "path": str(resolved), "reason": str(error)}
            )
            return None

        bundled_path = Path("provenance/artifacts") / f"{len(self.artifacts):06d}-{resolved.name}"
        artifact = {
            "role": role,
            "source_path": str(resolved),
            "bundled_path": bundled_path.as_posix(),
            "size_bytes": len(text.encode()),
            "artifact_type": payload.get("artifact_type"),
        }
        self.files[bundled_path] = text
        self.artifacts.append(artifact)
        self._artifact_by_source[resolved] = artifact
        self.payloads[resolved] = payload

        for reference_role, reference in _json_references(payload):
            referenced_path = Path(reference).expanduser()
            if not referenced_path.is_absolute():
                referenced_path = resolved.parent / referenced_path
            self.add_json(referenced_path, role=reference_role)
        return payload

    def artifact_for(self, source: Path) -> dict[str, Any] | None:
        return self._artifact_by_source.get(source.expanduser().resolve())


def write_fusion_provenance(
    *,
    output: Path,
    input_dir: Path,
    position_input: Path,
    registration_input: Path,
    channel: int,
    requested_settings: Mapping[str, Any],
    resolved_settings: Mapping[str, Any],
    output_grid_template: Path | None = None,
    output_grid_template_level: int = 0,
    flatfield_dirs: Sequence[Path] = (),
    additional_json_inputs: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Record the structured lineage for one fused channel without reading pixel chunks.

    Missing transitive artifacts make the record partial. The direct position and
    registration inputs and the destination Zarr metadata are required because they
    define the pixels and transforms claimed by the manifest.
    """
    output = output.expanduser().resolve()
    if not (output / "zarr.json").is_file():
        raise FileNotFoundError(f"Fusion output does not contain root Zarr metadata: {output}")

    collector = _ArtifactCollector()
    position_payload = collector.add_json(position_input, role="position_input", required=True)
    registration_payload = collector.add_json(registration_input, role="registration_input", required=True)
    assert position_payload is not None
    assert registration_payload is not None

    _add_materialization_summary(collector, position_input)
    _add_materialization_summary(collector, registration_input)
    for role, path in sorted((additional_json_inputs or {}).items()):
        collector.add_json(path, role=role)
    for flatfield_dir in sorted((Path(path) for path in flatfield_dirs), key=str):
        manifests = sorted(flatfield_dir.glob("*-sampling.json"))
        if len(manifests) == 1:
            collector.add_json(manifests[0], role="fusion_basic_fit_manifest")
        else:
            collector.unresolved.append(
                {
                    "stage": "basic",
                    "role": "fusion_basic_fit_manifest",
                    "path": str(flatfield_dir.resolve()),
                    "reason": f"expected exactly one *-sampling.json; found {len(manifests)}",
                }
            )

    source_paths = _deconvolution_source_paths(position_payload, registration_payload, input_dir)
    deconvolution, external_artifacts = _collect_deconvolution(collector, source_paths)
    if output_grid_template is not None:
        external_artifacts.append(
            _external_artifact_record(
                output_grid_template,
                role="output_grid_template",
                selected_level=output_grid_template_level,
            )
        )

    registration = _registration_summary(collector, registration_payload)
    materialization = _materialization_summary(position_payload, registration_payload)
    output_layout = _output_layout(output)
    status = "partial" if collector.unresolved else "complete"
    manifest = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "recording_mode": "inline",
        "status": status,
        "workflow": _workflow_type(registration_payload),
        "channel": int(channel),
        "invocation": [str(value) for value in sys.argv],
        "software": _software_versions(),
        "inputs": {
            "input_dir": str(input_dir.expanduser().resolve()),
            "position_input": str(position_input.expanduser().resolve()),
            "registration_input": str(registration_input.expanduser().resolve()),
            "output_grid_template": None
            if output_grid_template is None
            else {
                "path": str(output_grid_template.expanduser().resolve()),
                "level": int(output_grid_template_level),
            },
        },
        "settings": {
            "requested": _jsonable(dict(requested_settings)),
            "resolved": _jsonable(dict(resolved_settings)),
        },
        "preprocessing": {"deconvolution": deconvolution},
        "registration": registration,
        "materialization": materialization,
        "fusion": {
            "channel": int(channel),
            "requested": _jsonable(dict(requested_settings)),
            "resolved": _jsonable(dict(resolved_settings)),
        },
        "output": output_layout,
        "artifacts": collector.artifacts,
        "external_artifacts": _deduplicate_external_artifacts(external_artifacts),
        "coverage": {
            "status": status,
            "unresolved": sorted(
                collector.unresolved,
                key=lambda item: (item["stage"], item["role"], item["path"]),
            ),
        },
    }
    manifest_text = _dumps_json(manifest, context="fusion provenance manifest")
    collector.files[MANIFEST_RELATIVE_PATH] = manifest_text
    root_index = _root_index(manifest)

    staging = Path(tempfile.mkdtemp(prefix=".provenance-", dir=output))
    destination = output / MANIFEST_RELATIVE_PATH.parent
    promoted = False
    try:
        for relative_path, text in collector.files.items():
            staged_path = staging / relative_path.relative_to(MANIFEST_RELATIVE_PATH.parent)
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            staged_path.write_text(text)
        _validate_staged_bundle(staging, manifest)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite existing provenance bundle: {destination}")
        staging.replace(destination)
        promoted = True
        _publish_root_index(output, root_index)
    except BaseException:
        if promoted:
            shutil.rmtree(destination, ignore_errors=True)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return root_index


def _loads_json_object(text: str, *, context: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    payload = json.loads(text, parse_constant=reject_constant)
    if not isinstance(payload, dict):
        raise TypeError(f"{context} must contain a JSON object")
    return payload


def _loads_source_json_object(text: str, *, context: Path) -> dict[str, Any]:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise TypeError(f"{context} must contain a JSON object")
    return payload


def _dumps_json(payload: object, *, context: str) -> str:
    try:
        return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} is not strict JSON: {error}") from error


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite provenance value {value}")
        return value
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return _jsonable(value.item())
    raise TypeError(f"Unsupported provenance value of type {type(value).__name__}")


def _json_references(payload: object, path: tuple[str, ...] = ()) -> list[tuple[str, str]]:
    references: list[tuple[str, str]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            references.extend(_json_references(value, (*path, str(key))))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            references.extend(_json_references(value, (*path, str(index))))
    elif isinstance(payload, str) and payload.lower().endswith(".json"):
        role = next((part for part in reversed(path) if not part.isdigit()), "referenced_json")
        references.append((role, payload))
    return references


def _stage_for_role(role: str) -> str:
    lowered = role.lower()
    if "basic" in lowered:
        return "basic"
    if "scal" in lowered or "sample" in lowered:
        return "deconvolution"
    if "material" in lowered:
        return "materialization"
    if "fusion" in lowered or "candidate" in lowered:
        return "fusion"
    return "registration"


def _add_materialization_summary(collector: _ArtifactCollector, path: Path) -> None:
    name = path.name
    for suffix in (".positions.json", ".registration.json"):
        if name.endswith(suffix):
            summary = path.with_name(f"{name.removesuffix(suffix)}.summary.json")
            if summary.exists():
                collector.add_json(summary, role="materialization_summary")


def _workflow_type(registration: Mapping[str, Any]) -> str:
    artifact_type = str(registration.get("artifact_type") or "")
    if "fused_fixed_overlapping_materialized" in artifact_type:
        return "fused_fixed_cross_channel"
    metrics = registration.get("metrics")
    if isinstance(metrics, dict) and (
        "registration_run" in metrics or "phase_recovery" in json.dumps(metrics, sort_keys=True)
    ):
        return "standard_affine"
    return "registered_tiles"


def _registration_summary(
    collector: _ArtifactCollector, registration_payload: Mapping[str, Any]
) -> dict[str, Any]:
    preferred = []
    for key in ("source_summary_input", "source_registration_summary", "input_summary"):
        value = registration_payload.get(key)
        if isinstance(value, str) and value.endswith(".json"):
            preferred.append(Path(value).expanduser().resolve())
    candidates = preferred + [path for path, payload in collector.payloads.items() if isinstance(payload.get("windows"), list)]
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        payload = collector.payloads.get(path)
        windows = None if payload is None else payload.get("windows")
        if not isinstance(windows, list):
            continue
        status_counts = Counter(
            str(window.get("status")) for window in windows if isinstance(window, dict) and window.get("status")
        )
        if not status_counts:
            continue
        selected_attempt_counts = Counter(
            str(window.get("selected_attempt"))
            for window in windows
            if isinstance(window, dict) and window.get("selected_attempt")
        )
        rejection_counts = Counter(
            str(window.get("rejection_reason"))
            for window in windows
            if isinstance(window, dict) and window.get("rejection_reason")
        )
        artifact = collector.artifact_for(path)
        return {
            "summary_artifact": None if artifact is None else artifact["bundled_path"],
            "counts": {
                "accepted": int(status_counts.get("accepted", 0)),
                "rejected": int(status_counts.get("rejected", 0)),
            },
            "status_counts": dict(sorted(status_counts.items())),
            "selected_attempt_counts": dict(sorted(selected_attempt_counts.items())),
            "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        }

    metrics = registration_payload.get("metrics")
    return {
        "summary_artifact": None,
        "counts": {},
        "registration_run": metrics.get("registration_run")
        if isinstance(metrics, dict) and isinstance(metrics.get("registration_run"), dict)
        else None,
    }


def _materialization_summary(
    position_payload: Mapping[str, Any], registration_payload: Mapping[str, Any]
) -> dict[str, Any] | None:
    grid = registration_payload.get("materialization_grid") or position_payload.get("materialization_grid")
    tiles = registration_payload.get("tiles")
    if grid is None and not isinstance(tiles, list):
        return None
    accepted = 0
    rejected = 0
    if isinstance(tiles, list):
        for tile in tiles:
            if not isinstance(tile, dict):
                continue
            if tile.get("status") == "rejected":
                rejected += 1
            else:
                accepted += 1
    return {"grid": grid, "accepted_tiles": accepted, "rejected_tiles": rejected}


def _deconvolution_source_paths(
    position_payload: Mapping[str, Any],
    registration_payload: Mapping[str, Any],
    input_dir: Path,
) -> list[Path]:
    candidates: set[Path] = set()
    for payload in (position_payload, registration_payload):
        tiles = payload.get("tiles")
        if not isinstance(tiles, list):
            continue
        for tile in tiles:
            if not isinstance(tile, dict):
                continue
            value = tile.get("materialized_source_path") or tile.get("path")
            if isinstance(value, str) and value.endswith((".zarr", ".ome.zarr")):
                path = Path(value).expanduser()
                if not path.is_absolute():
                    path = input_dir / path.name
                candidates.add(path.resolve())
    return sorted(candidates, key=str)


def _collect_deconvolution(
    collector: _ArtifactCollector, source_paths: Sequence[Path]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sources: list[dict[str, Any]] = []
    common_payloads: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    external: list[dict[str, Any]] = []
    for source in source_paths:
        metadata_path = source / "zarr.json"
        try:
            root = _loads_json_object(metadata_path.read_text(), context=metadata_path)
            attrs = root.get("attributes")
            deconv = attrs.get("squisher_deconv") if isinstance(attrs, dict) else None
            if not isinstance(deconv, dict):
                continue
            provenance = deconv.get("provenance")
            if not isinstance(provenance, dict):
                raise ValueError("squisher_deconv.provenance is missing")
            common = {
                "output_mode": deconv.get("output_mode"),
                "run_settings": provenance.get("run_settings"),
                "psfs": provenance.get("psfs"),
                "basic_profiles": provenance.get("basic_profiles"),
                "scaling": provenance.get("scaling"),
                "versions": provenance.get("versions"),
                "storage": deconv.get("storage"),
            }
            specific = {
                "path": str(source),
                "created_utc": provenance.get("created_utc"),
                "source": provenance.get("source"),
                "source_file": provenance.get("source_file"),
                "source_metadata_hash": deconv.get("source_metadata_hash"),
                "source_metadata_summary": deconv.get("source_metadata_summary"),
                "source_ome": deconv.get("source_ome"),
                "root_metadata": _file_record(metadata_path),
                "level0_metadata": _file_record(source / "0" / "zarr.json"),
            }
            common_payloads.append((source, common, specific))
            external.append(_external_artifact_record(source, role="deconvolution_store", selected_level=0))
        except (OSError, TypeError, ValueError) as error:
            collector.unresolved.append(
                {
                    "stage": "deconvolution",
                    "role": "deconvolution_store",
                    "path": str(source),
                    "reason": str(error),
                }
            )

    cohort_ids: dict[str, str] = {}
    cohorts: list[dict[str, Any]] = []
    for source, common, specific in common_payloads:
        key = json.dumps(_jsonable(common), sort_keys=True, allow_nan=False)
        cohort_id = cohort_ids.get(key)
        if cohort_id is None:
            cohort_id = f"cohort-{len(cohorts) + 1:04d}"
            cohort_ids[key] = cohort_id
            cohorts.append({"id": cohort_id, "settings": common})
            _collect_deconvolution_manifests(collector, common)
            for role in ("psfs", "basic_profiles"):
                records = common.get(role)
                if isinstance(records, list):
                    external.extend(
                        _external_artifact_record(Path(record["path"]), role=role.removesuffix("s"))
                        for record in records
                        if isinstance(record, dict) and isinstance(record.get("path"), str)
                    )
        specific["cohort_id"] = cohort_id
        sources.append(specific)

    status = "present" if sources else "absent"
    return {"status": status, "cohorts": cohorts, "sources": sources}, external


def _collect_deconvolution_manifests(
    collector: _ArtifactCollector, common: Mapping[str, Any]
) -> None:
    profiles = common.get("basic_profiles")
    profile_paths = [
        Path(record["path"]).expanduser().resolve()
        for record in profiles
        if isinstance(record, dict) and isinstance(record.get("path"), str)
    ] if isinstance(profiles, list) else []
    for parent in sorted({path.parent for path in profile_paths}, key=str):
        manifests = sorted(parent.glob("*-sampling.json"))
        if len(manifests) == 1:
            manifest = collector.add_json(manifests[0], role="basic_fit_manifest")
            outputs = None if manifest is None else manifest.get("outputs")
            recorded_profiles = outputs.get("profiles") if isinstance(outputs, dict) else None
            recorded = {
                str(Path(path).expanduser().resolve())
                for path in recorded_profiles
                if isinstance(path, str)
            } if isinstance(recorded_profiles, list) else set()
            expected = {str(path) for path in profile_paths if path.parent == parent}
            if not expected.issubset(recorded):
                collector.unresolved.append(
                    {
                        "stage": "basic",
                        "role": "basic_fit_manifest",
                        "path": str(manifests[0].resolve()),
                        "reason": "manifest outputs.profiles do not contain every applied BaSiC profile",
                    }
                )
        elif profile_paths:
            collector.unresolved.append(
                {
                    "stage": "basic",
                    "role": "basic_fit_manifest",
                    "path": str(parent),
                    "reason": f"expected exactly one *-sampling.json; found {len(manifests)}",
                }
            )

    scaling = common.get("scaling")
    scaling_path = (
        Path(scaling["path"]).expanduser().resolve()
        if isinstance(scaling, dict) and isinstance(scaling.get("path"), str)
        else None
    )
    if scaling_path is not None:
        scaling_payload = collector.add_json(scaling_path, role="scaling_settings")
        sample_manifest = scaling_path.parent / "sample-manifest.json"
        sample_payload = collector.add_json(sample_manifest, role="deconvolution_sample_manifest")
        if (
            scaling_payload is not None
            and sample_payload is not None
            and sample_payload.get("scaling") != scaling_payload
        ):
            collector.unresolved.append(
                {
                    "stage": "deconvolution",
                    "role": "deconvolution_sample_manifest",
                    "path": str(sample_manifest.resolve()),
                    "reason": "sample manifest scaling does not match scaling.json",
                }
            )


def _external_artifact_record(
    path: Path, *, role: str, selected_level: int | None = None
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    record: dict[str, Any] = {"role": role, "path": str(resolved), "exists": resolved.exists()}
    if resolved.exists():
        stat = resolved.stat()
        record.update({"size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
    if selected_level is not None:
        record["selected_level"] = int(selected_level)
        array_metadata = resolved / str(selected_level) / "zarr.json"
        if array_metadata.is_file():
            try:
                record["selected_array"] = _array_layout(
                    _loads_json_object(array_metadata.read_text(), context=array_metadata)
                )
            except (OSError, TypeError, ValueError):
                pass
    return record


def _deduplicate_external_artifacts(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        unique[(str(record.get("role")), str(record.get("path")))] = record
    return [unique[key] for key in sorted(unique)]


def _file_record(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _output_layout(output: Path) -> dict[str, Any]:
    metadata_path = output / "zarr.json"
    root = _loads_json_object(metadata_path.read_text(), context=metadata_path)
    attrs = root.get("attributes")
    if not isinstance(attrs, dict):
        raise ValueError(f"{metadata_path} has no attributes object")
    ome = attrs.get("ome")
    if not isinstance(ome, dict):
        raise ValueError(f"{metadata_path} has no OME metadata")
    multiscales = ome.get("multiscales")
    if not isinstance(multiscales, list) or not multiscales or not isinstance(multiscales[0], dict):
        raise ValueError(f"{metadata_path} has no OME multiscales entry")
    datasets = multiscales[0].get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"{metadata_path} has no OME datasets")
    levels = []
    for dataset in datasets:
        if not isinstance(dataset, dict) or not isinstance(dataset.get("path"), str):
            raise ValueError(f"{metadata_path} contains an invalid OME dataset")
        array_path = output / dataset["path"] / "zarr.json"
        array = _loads_json_object(array_path.read_text(), context=array_path)
        levels.append(
            {
                "path": dataset["path"],
                "coordinate_transformations": dataset.get("coordinateTransformations", []),
                **_array_layout(array),
            }
        )
    return {
        "path": str(output),
        "zarr_format": root.get("zarr_format"),
        "ome": ome,
        "levels": levels,
    }


def _array_layout(metadata: Mapping[str, Any]) -> dict[str, Any]:
    chunk_grid = metadata.get("chunk_grid")
    chunk_shape = None
    if isinstance(chunk_grid, dict):
        configuration = chunk_grid.get("configuration")
        if isinstance(configuration, dict):
            chunk_shape = configuration.get("chunk_shape")
    return {
        "shape": metadata.get("shape"),
        "data_type": metadata.get("data_type"),
        "dimension_names": metadata.get("dimension_names"),
        "chunk_shape": chunk_shape,
        "codecs": metadata.get("codecs"),
    }


def _software_versions() -> dict[str, Any]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for label, distribution in _PACKAGE_DISTRIBUTIONS.items():
        try:
            versions[label] = version(distribution)
        except PackageNotFoundError:
            versions[label] = "not-installed"
    repo_root = Path(__file__).resolve().parents[3]
    git: dict[str, Any] = {"available": False}
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        git = {"available": True, "revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        pass
    return {"versions": versions, "git": git}


def _root_index(manifest: Mapping[str, Any]) -> dict[str, Any]:
    registration = manifest.get("registration")
    counts = registration.get("counts", {}) if isinstance(registration, dict) else {}
    output = manifest["output"]
    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "manifest": MANIFEST_RELATIVE_PATH.as_posix(),
        "recorded_utc": manifest["recorded_utc"],
        "recording_mode": manifest["recording_mode"],
        "status": manifest["status"],
        "unresolved_count": len(manifest["coverage"]["unresolved"]),
        "workflow": manifest["workflow"],
        "channel": manifest["channel"],
        "registration_counts": counts,
        "output": {
            "level_count": len(output["levels"]),
            "level0": output["levels"][0],
        },
    }


def _validate_staged_bundle(staging: Path, manifest: Mapping[str, Any]) -> None:
    staged_manifest = staging / "manifest.json"
    loaded = _loads_json_object(staged_manifest.read_text(), context=staged_manifest)
    if loaded != manifest:
        raise ValueError("Staged provenance manifest differs from the validated payload")
    for artifact in manifest["artifacts"]:
        bundled_path = Path(str(artifact["bundled_path"]))
        path = staging / bundled_path.relative_to(MANIFEST_RELATIVE_PATH.parent)
        if not path.is_file():
            raise FileNotFoundError(f"Staged provenance artifact is missing: {path}")
        _loads_source_json_object(path.read_text(), context=path)


def _publish_root_index(output: Path, index: Mapping[str, Any]) -> None:
    metadata_path = output / "zarr.json"
    payload = _loads_json_object(metadata_path.read_text(), context=metadata_path)
    attrs = payload.setdefault("attributes", {})
    if not isinstance(attrs, dict):
        raise ValueError(f"{metadata_path} attributes must be an object")
    attrs["squisher_fusion"] = dict(index)
    attrs["squisher_complete"] = True
    temporary = metadata_path.with_name(f".{metadata_path.name}.squisher-provenance.tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite stale root metadata temporary file: {temporary}")
    temporary.write_text(_dumps_json(payload, context=str(metadata_path)))
    temporary.replace(metadata_path)
