from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import zarr

from squisher_lightsheet import hcr_workflow, hcr_acquisition, hcr_channels, hcr_reference, workflow_runner
from squisher_lightsheet.workflow_runner import ReviewRequired, StageRunner


def _acquisition(
    root: Path, name: str, side: str, labels: list[str], tracks: list[list[int]] | None = None
) -> dict:
    directory = root / "raw" / name
    directory.mkdir(parents=True)
    tiles = []
    for index in range(2):
        tile = directory / f"{name}.{index:03d}.ome.tif"
        tile.write_bytes(b"tile")
        tiles.append({"name": tile.name})
    return {
        "side": side,
        "directory": str(directory),
        "channels": [{"index": index, "label": label} for index, label in enumerate(labels)],
        "tiles": tiles,
        "tracks": tracks if tracks is not None else [list(range(len(labels)))],
    }


@pytest.fixture
def manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(hcr_workflow, "verify_manifest", lambda _manifest: None)
    monkeypatch.setattr(hcr_workflow, "gpu_uuids", lambda: ("GPU-a", "GPU-b"))
    payload = {
        "dataset": "d",
        "layout": "CL/CR",
        "fixed_side": "CL",
        "moving_side": "CR",
        "acquisitions": [
            _acquisition(tmp_path, "CL-main", "CL", ["561", "638", "514", "488"], [[0, 1], [2], [3]]),
            _acquisition(tmp_path, "CR-561638", "CR", ["561", "638"]),
            _acquisition(tmp_path, "CR-514", "CR", ["514"]),
            _acquisition(tmp_path, "CR-488", "CR", ["488"]),
        ],
        "approved": False,
    }
    path = tmp_path / "work" / "manifest.json"
    path.parent.mkdir()
    path.write_text(json.dumps(payload))
    return path


def test_unapproved_manifest_stops_before_any_stage(manifest: Path) -> None:
    with pytest.raises(ReviewRequired, match="set approved=true"):
        hcr_workflow.run_hcr_workflow(manifest, dry_run=True)

    assert not (manifest.parent / ".stages").exists()


def test_dry_run_preprocesses_each_acquisition_then_gates_registration_thresholds(
    manifest: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = json.loads(manifest.read_text())
    manifest.write_text(json.dumps(payload | {"approved": True}))

    with pytest.raises(ReviewRequired, match="registration threshold"):
        hcr_workflow.run_hcr_workflow(manifest, dry_run=True)

    commands = [
        line for line in capsys.readouterr().out.splitlines() if line.startswith("[hcr] squisher-deconv run")
    ]
    assert len(commands) == 4
    main = next(command for command in commands if "CL-main" in command)
    assert main.count("zstride6_xy9") == 4 and "zcenter2" not in main
    assert main.split(" --psf ")[2].startswith(str(hcr_acquisition.PSF_648))
    assert "--slab-depth 25" in main
    assert "--slab-depth 50" in next(command for command in commands if "CR-514" in command)
    thresholds = json.loads((manifest.parent / "native-thresholds.json").read_text())["thresholds"]
    assert set(thresholds) == {"CL/561", "CL/514", "CL/488", "CR/561", "CR/514", "CR/488"}


def test_638_reuses_the_561_registration_of_its_own_track(manifest: Path) -> None:
    sources = hcr_acquisition.channel_sources(json.loads(manifest.read_text()))

    by_key = {(source.side, source.label): source for source in sources}
    assert by_key[("CL", "638")].registration_label == "561"
    assert by_key[("CL", "638")].acquisition == "CL-main"
    assert by_key[("CL", "514")].registration_label == "514"
    assert by_key[("CR", "638")].index == 1


def test_638_outside_the_561_track_has_no_shared_geometry(manifest: Path) -> None:
    payload = json.loads(manifest.read_text())
    payload["acquisitions"][0]["tracks"] = [[0], [1], [2], [3]]

    with pytest.raises(ValueError, match="638 does not share a track with 561"):
        hcr_acquisition.channel_sources(payload)


class _RecordingRunner:
    """Records each stage's argv and creates its required outputs, as a completed stage would; reuses finished ones."""

    def __init__(self, *, dry_run: bool = False) -> None:
        self.commands: dict[str, list[str]] = {}
        self.actions: list[str] = []
        self.dry_run = dry_run

    def command(self, name: str, argv: list, *, required: list[Path], **_options: object) -> None:
        if name in self.commands:
            return
        self.commands[name] = [str(value) for value in argv]
        for path in required:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"stage": name}))

    def action(self, name: str, _identity: object, action, *, required: list[Path]) -> None:
        if name in self.actions:
            return
        self.actions.append(name)
        action()


def _approve(path: Path) -> None:
    path.write_text(json.dumps(json.loads(path.read_text()) | {"approved": True}))


def _place(runner: _RecordingRunner, manifest: Path) -> dict[str, Path]:
    payload = json.loads(manifest.read_text())
    return hcr_reference.place_clcr(
        runner,
        hcr_acquisition.HcrPaths(root=manifest.parent),
        payload,
        hcr_acquisition.channel_sources(payload),
        {"CL/561": 900.0, "CR/561": 150.0},
        all_gpu_env={"CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b"},
    )


def test_placement_reviews_the_coarse_alignment_before_method8_and_publishing(manifest: Path) -> None:
    root = manifest.parent
    runner = _RecordingRunner()
    with pytest.raises(ReviewRequired, match="phase-overlay"):
        _place(runner, manifest)
    assert "placement-method8" not in runner.commands

    # The author rejects the phase shift and supplies a manual one, which aligns into its own folder.
    request = root / "placement-coarse.json"
    request.write_text(json.dumps(json.loads(request.read_text()) | {"manual_cr_shift_zyx_um": [1, -2, 3.5]}))
    (root / "placement-coarse-approval.json").unlink()
    with pytest.raises(ReviewRequired, match="phase-overlay"):
        _place(runner, manifest)
    manual = runner.commands["placement-align-coarse-manual_+1.000_-2.000_+3.500"]
    assert "--manual-cr-shift-zyx-um=1.0,-2.0,3.5" in manual
    assert manual[manual.index("--output-dir") + 1].endswith("placement/coarse-manual_+1.000_-2.000_+3.500")
    assert manual[manual.index("--left-fusion") + 1].endswith("sides/CL/fusion-561/fused.ch0.ome.zarr")

    _approve(root / "placement-coarse-approval.json")
    with pytest.raises(ReviewRequired, match="CL/CR placement overlay"):
        _place(runner, manifest)
    method8 = runner.commands["placement-method8"]
    assert method8[method8.index("--fixed-position") + 1].endswith(
        "coarse-manual_+1.000_-2.000_+3.500/cl.positions.json"
    )
    assert method8[method8.index("--fixed-mask-threshold") + 1] == "900.0"
    assert "--preseed-matrix-zyx" in method8
    assert method8[method8.index("--preseed-matrix-zyx") + 1] == "1,0,0,0,1,0,0,0,1"
    assert method8[method8.index("--pair-mode") + 1] == "spatial-overlap"
    assert (root / "placement" / "deconv-561").is_dir()

    _approve(root / "placement-approval.json")
    assert _place(runner, manifest)["positions"] == root / "placement" / "final" / "positions.json"


def test_joint_fusion_requires_one_561_channel_index_across_sides(manifest: Path) -> None:
    payload = json.loads(manifest.read_text())
    payload["acquisitions"][1]["channels"] = [{"index": 0, "label": "638"}, {"index": 1, "label": "561"}]
    payload["acquisitions"][1]["tracks"] = [[0, 1]]

    with pytest.raises(ValueError, match="561 is channel 0 on CL but 1 on CR"):
        hcr_workflow.supported_sources(payload)


@pytest.mark.parametrize(
    ("change", "error"),
    [
        (lambda payload: payload.update(layout="TL/TR"), "must be acquired once on each of TL and TR"),
        (lambda payload: payload["acquisitions"].pop(), "488 must be acquired once on each of CL and CR"),
        (
            lambda payload: payload["acquisitions"].append(
                payload["acquisitions"][-1]
                | {"directory": "CR-405", "channels": [{"index": 0, "label": "405"}]}
            ),
            "HCR samples skip 405",
        ),
        (
            lambda payload: [payload["acquisitions"].pop(2), payload["acquisitions"][0]["channels"].pop(2)],
            "488 registers to fused 514, which is not acquired",
        ),
    ],
)
def test_unsupported_manifests_are_refused_before_preprocessing(manifest: Path, change, error: str) -> None:
    payload = json.loads(manifest.read_text())
    change(payload)
    manifest.write_text(json.dumps(payload | {"approved": True}))

    with pytest.raises((ValueError, NotImplementedError), match=error):
        hcr_workflow.run_hcr_workflow(manifest, dry_run=True)
    assert not list((manifest.parent / ".stages").glob("*"))


def _grid(path: Path, *, planes: int, scale_z: float) -> Path:
    group = zarr.open_group(str(path), mode="w")
    group.create_array("0", shape=(planes, 4, 4), dtype="uint16")
    group.attrs["multiscales"] = [
        {
            "axes": ["z", "y", "x"],
            "datasets": [
                {"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [scale_z, 1, 1]}]}
            ],
        }
    ]
    return path


def _positions(path: Path, z_ranges: dict[str, tuple[float, int]]) -> Path:
    tiles = [
        {
            "source_side": side,
            "translation_um": {"z": start, "y": 0.0, "x": 0.0},
            "scale_um": {"z": 2.0, "y": 1.0, "x": 1.0},
            "shape": [2, planes, 4, 4],
        }
        for side, (start, planes) in z_ranges.items()
    ]
    path.write_text(json.dumps({"tiles": tiles}))
    return path


def test_shared_slab_samples_only_planes_both_sides_cover_inside_the_grid(tmp_path: Path) -> None:
    grid = _grid(tmp_path / "grid.zarr", planes=101, scale_z=2.0)
    # CL covers 0-120 um; CR covers 80-260 um, past the 200 um grid end.
    positions = _positions(tmp_path / "positions.json", {"CL": (0.0, 60), "CR": (80.0, 90)})

    percentiles = hcr_reference.shared_slab_percentiles(grid, positions, sides=("CL", "CR"))

    # 90-110 um after the 10 um margin: planes 45-55.
    assert percentiles[0] == pytest.approx(45.5) and percentiles[-1] == pytest.approx(55.5)
    assert len(percentiles) == 11


def test_shared_slab_requires_overlapping_sides(tmp_path: Path) -> None:
    grid = _grid(tmp_path / "grid.zarr", planes=101, scale_z=2.0)
    positions = _positions(tmp_path / "positions.json", {"CL": (0.0, 40), "CR": (90.0, 40)})

    with pytest.raises(ValueError, match="share no Z slab"):
        hcr_reference.shared_slab_percentiles(grid, positions, sides=("CL", "CR"))


@pytest.mark.parametrize("native_only", [False, True])
def test_canonical_638_uses_its_own_side_corrections_on_the_561_grid(
    manifest: Path, monkeypatch: pytest.MonkeyPatch, native_only: bool
) -> None:
    payload = json.loads(manifest.read_text())
    paths = hcr_acquisition.HcrPaths(root=manifest.parent)
    runner = _RecordingRunner(dry_run=True)
    final = manifest.parent / "placement" / "final"
    placement = {
        "positions": final / "positions.json",
        "registration": final / "registration.json",
        "deconv": manifest.parent / "placement" / "deconv-561",
    }

    native = manifest.parent / "canonical/native-corrections-561/correction.json"
    if native_only:
        monkeypatch.setattr(hcr_reference, "reviewed_native_561_corrections", lambda *args, **kwargs: native)
    fused = hcr_reference.fuse_canonical(
        runner,
        paths,
        hcr_acquisition.channel_sources(payload),
        placement,
        sides=("CL", "CR"),
        all_gpu_env={},
        primary_gpu_env={},
    )

    affine = runner.commands["canonical-638-affine"]
    assert affine[affine.index("--expected-moving-channel") + 1] == "1"
    match = runner.commands["canonical-match-638"]
    corrections = [match[i + 1] for i, value in enumerate(match) if value == "--correction-by-source-view"]
    assert corrections == [
        f"L={paths.residual('CL-main', '638') / 'correction.json'}",
        f"R={paths.residual('CR-561638', '638') / 'correction.json'}",
    ]
    assert match[match.index("--registration") + 1] == str(final / "registration-638.json")
    fuse = runner.commands["canonical-fuse-638"]
    assert fuse[fuse.index("--output-grid-template") + 1] == str(fused["561"])
    assert "--output-grid-template" not in runner.commands["canonical-fuse-561"]
    fused_561 = runner.commands["canonical-fuse-561"]
    assert ("--registered-view-correction" in fused_561) == (not native_only)
    assert ("canonical-match-561" in runner.commands) == (not native_only)
    if native_only:
        assert fused_561[fused_561.index("--residual-correction") + 1] == str(native)
    assert "--registered-view-correction" in fuse


def test_cr_initializer_flips_z_and_x_couplings_as_229tnc_did() -> None:
    # 229Tnc run-xreg.sh flipped these terms of the EDU 514 initializer by hand for CR.
    cr = [
        float(value)
        for value in hcr_channels.side_starting_matrix(hcr_channels.STARTING_MATRIX_514, "CR").split(",")
    ]
    expected = [
        0.9999090828,
        0.0083501824,
        -0.0029757332,
        -0.0141554887,
        0.9999284579,
        0.0096124636,
        0.0026881526,
        -0.0099598448,
        0.9999495898,
    ]
    assert cr == pytest.approx(expected)
    cl = hcr_channels.side_starting_matrix(hcr_channels.STARTING_MATRIX_514, "CL").split(",")
    assert [float(value) for value in cl] == [
        float(value) for value in hcr_channels.STARTING_MATRIX_514.split(",")
    ]


def _complete_summary(path: Path, windows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"windows": windows}))
    return path


def _seed_master(paths: hcr_acquisition.HcrPaths, channel: str, entries: dict) -> None:
    workflow_runner.atomic_json(
        paths.manual_initializations,
        {"artifact_type": hcr_channels.MANUAL_INITIALIZATIONS_ARTIFACT, "channels": {channel: entries}},
    )


def _rounds(
    manifest: Path,
    monkeypatch: pytest.MonkeyPatch,
    proposals: list[dict],
    counts: list[int],
    runner: _RecordingRunner | None = None,
):
    """Run the round controller with propose-seeds writing ``proposals[n]`` and round n leaving ``counts[n]``."""
    paths = hcr_acquisition.HcrPaths(root=manifest.parent)
    source = hcr_acquisition.ChannelSource("CL", "514", "CL-main", 2, "514")
    runner = runner or _RecordingRunner()

    def propose(*, complete_summary: Path, master: Path, channel: str) -> None:
        _seed_master(paths, channel, proposals[int(complete_summary.parent.name.split("-")[1])])

    monkeypatch.setattr(hcr_channels, "propose_manual_initializations", propose)
    monkeypatch.setattr(
        hcr_channels,
        "seedable_window_count",
        lambda summary: counts[int(summary.parent.name.split("-")[1])],
    )
    monkeypatch.setattr(hcr_channels, "_recovery_inputs", lambda summary: str(summary))
    summary = hcr_channels._recover_in_rounds(
        runner, paths, source, manifest.parent / "m6.json", primary_gpu_env={}
    )
    return runner, summary, paths


def test_recovery_rounds_stop_when_measured_seeds_stop_reducing_rejected_windows(
    manifest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = {"w1": {"seed_origin": "measured"}}
    reseed = {"w1": {"seed_origin": "measured", "local_translation_zyx": [1, 0, 0]}}
    runner, summary, paths = _rounds(manifest, monkeypatch, [seed, reseed, reseed], [5, 5, 5])

    # Round 0 proposes a seed, round 1 recovers with it but leaves as many windows, so its new proposal is not tried.
    assert summary.parent.name == "round-1"
    recover = [name for name in runner.commands if "recover-native" in name]
    assert recover == ["514-CL-recover-native-round0", "514-CL-recover-native-round1"]
    assert "--manual-initializations" not in runner.commands["514-CL-recover-native-round0"]
    seeds = runner.commands["514-CL-recover-native-round1"]
    snapshot = Path(seeds[seeds.index("--manual-initializations") + 1])
    assert json.loads(snapshot.read_text())["channels"]["514-CL"] == seed


def test_recovery_rounds_continue_while_seeds_reduce_rejected_windows(
    manifest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = {"w1": {"n": 1}}, {"w1": {"n": 1}, "w2": {"n": 2}}
    _runner, summary, _paths = _rounds(manifest, monkeypatch, [first, second, second], [5, 3, 3])

    # Round 2's proposals equal its seeds, so nothing new remains to try.
    assert summary.parent.name == "round-2"


def test_author_seeds_preserved_by_proposal_start_a_round_without_count_improvement(
    manifest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    measured = {"w1": {"seed_origin": "measured"}}
    edited = measured | {"w9": {"seed_origin": "manual"}}
    # An author adds a seed while round 1 runs; propose-seeds preserves it in its output.
    runner, summary, _paths = _rounds(manifest, monkeypatch, [measured, edited, edited], [5, 5, 5])
    assert summary.parent.name == "round-2"
    assert "514-CL-recover-native-round2" in runner.commands


def test_an_author_seed_edit_starts_a_new_round(manifest: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proposals = [{}, {"w9": {"seed_origin": "manual"}}]
    runner, summary, paths = _rounds(manifest, monkeypatch, proposals, [5, 5])
    assert summary.parent.name == "round-0"

    # Rerunning the workflow reuses the finished round 0; the edited master differs from its seeds.
    _seed_master(paths, "514-CL", {"w9": {"seed_origin": "manual"}})
    _runner, summary, _paths = _rounds(manifest, monkeypatch, proposals, [5, 5], runner)
    assert summary.parent.name == "round-1"


def _window(path: Path, *, z: tuple[int, int], translation_x: float = 0.0, **record: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    window = {
        "status": "accepted",
        "moving_tile": "d.001.ome.zarr",
        "moving_start_l0_zyx": [z[0], 0, 0],
        "fixed_start_zyx": [z[0], 0, 0],
        "fixed_stop_zyx": [z[1], 8, 8],
        "selected_local_matrix_zyx": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "selected_local_translation_zyx": [0.0, 0.0, translation_x],
    }
    path.write_text(json.dumps(window | record))
    return str(path)


def test_screen_lists_each_defect_with_the_qc_plane_through_it(tmp_path: Path) -> None:
    from squisher_lightsheet.registration_screen import screen_recovery

    fused = tmp_path / "fused.zarr"
    group = zarr.open_group(str(fused), mode="w")
    for level, factor in enumerate((1, 2, 4)):
        group.create_array(str(level), shape=(400 // factor, 8, 8), dtype="uint16")
    group.attrs["multiscales"] = [
        {
            "axes": ["z", "y", "x"],
            "datasets": [
                {
                    "path": str(level),
                    "coordinateTransformations": [{"type": "scale", "scale": [factor, 1, 1]}],
                }
                for level, factor in enumerate((1, 2, 4))
            ],
        }
    ]
    windows = tmp_path / "windows"
    rerun = {"selected_candidate": "native_rerun", "residual": {"verdict": "unverified"}}
    withheld = {
        "selected_candidate": "interpolated",
        "residual": {"verdict": "unverified"},
        "native_rerun_withheld_for_departure": True,
    }
    repaired = {"outlier_recovery": {}}
    rows = [
        {"status": "accepted", "tile": "t", "level0_json": _window(windows / "ok.json", z=(0, 80))},
        {
            "status": "accepted",
            "tile": "t",
            "level0_json": _window(windows / "seeded.json", z=(40, 120), **repaired),
        },
        {
            "status": "accepted",
            "tile": "t",
            "level0_json": _window(
                windows / "rerun.json", z=(200, 280), interpolated_fallback_promotion=rerun, **repaired
            ),
        },
        {
            "status": "accepted",
            "tile": "t",
            "level0_json": _window(
                windows / "withheld.json", z=(220, 300), interpolated_fallback_promotion=withheld, **repaired
            ),
        },
        {
            "status": "accepted",
            "tile": "t",
            "level0_json": _window(
                windows / "linear.json",
                z=(300, 380),
                linear_mapping_recovery={"selected_candidate": "fitted", "fitted_image_ncc_planes": [0.2]},
                **repaired,
            ),
        },
        {
            "status": "accepted",
            "tile": "t",
            "level0_json": _window(windows / "smoothed.json", z=(160, 240), final_linear_interpolation={}),
        },
        {
            "status": "rejected",
            "rejection_reason": "spatial_fallback_contradicted_by_image",
            "fixed_masked_fraction": 0.1,
            "tile": "t",
            "level0_json": _window(windows / "hole.json", z=(100, 180)),
        },
        {
            "status": "rejected",
            "rejection_reason": "fixed_threshold_mask_too_masked",
            "fixed_masked_fraction": 0.99,
            "tile": "t",
            "level0_json": str(windows / "absent.json"),
        },
    ]
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"windows": rows, "window_shape_zyx": [80, 8, 8], "core_shape_zyx": [60, 8, 8]})
    )

    report = screen_recovery({"514-CL": summary}, fixed_fused=fused)

    kinds = {(Path(defect["window"]).stem, defect["kind"]) for defect in report["defects"]}
    assert kinds == {
        ("rerun", "unverified_native_rerun"),
        ("withheld", "withheld_native_rerun"),
        ("linear", "linear_fit_without_image_evidence"),
        ("hole", "uncovered"),
    }
    hole = next(defect for defect in report["defects"] if defect["kind"] == "uncovered")
    # Level 2 halves Z twice: the window centred at level-0 plane 140 is level-2 plane 35.
    assert hole["qc_plane"] == 35
    assert report["repaired_count"] == 5
    # Level-2 window cores (level-0 margins of 10): seeded 12-27, smoothed 42-57, rerun 52-67, withheld 57-72,
    # linear 77-92; plane 57 passes through the smoothing-only repair and the rerun.
    assert {27, 57, 92} <= set(report["qc_planes"])
    assert min(report["qc_planes"]) == 10 and max(report["qc_planes"]) == 92


def test_stabbing_planes_pass_through_every_interval_with_the_fewest_planes() -> None:
    from squisher_lightsheet.registration_screen import stabbing_planes

    assert stabbing_planes([(0, 10), (5, 15), (12, 20), (30, 31)]) == [9, 19, 30]


@pytest.mark.parametrize("reviewed_no_shift", [False, True])
def test_cross_registration_places_each_side_by_its_acquisition(
    manifest: Path, monkeypatch: pytest.MonkeyPatch, reviewed_no_shift: bool
) -> None:
    root = manifest.parent
    payload = json.loads(manifest.read_text())
    # 229Ptprz1: CL 514/488 share the 561 acquisition; CR 514 and 488 are separate acquisitions.
    sources = hcr_acquisition.channel_sources(payload)
    paths = hcr_acquisition.HcrPaths(root=root)
    positions = root / "placement" / "final" / "positions.json"
    positions.parent.mkdir(parents=True)
    positions.write_text(
        json.dumps(
            {
                "alignment": {"net_cr_shift_after_centroid_um_zyx": [-1.5, 2.0, 3.25]},
                "tiles": [{"tile": "a", "source_side": "CL"}, {"tile": "b", "source_side": "CR"}],
            }
        )
    )
    placement = {
        "positions": positions,
        "registration": root / "r.json",
        "deconv": root / "placement" / "deconv",
    }
    monkeypatch.setattr(
        hcr_channels,
        "_recover_in_rounds",
        lambda runner, paths, source, method6, **_: (
            paths.channel(source.label) / source.side / "round-0.json"
        ),
    )
    monkeypatch.setattr(hcr_channels, "_screen_registration", lambda *args, **kwargs: None)
    monkeypatch.setattr(hcr_channels, "shared_slab_percentiles", lambda *args, **kwargs: [50.0])
    monkeypatch.setattr(hcr_channels, "registration_input_fingerprint", lambda *args: None)
    monkeypatch.setattr(hcr_channels, "_recovery_inputs", lambda summary: None)
    if reviewed_no_shift:

        def retain(_runner, *, label, side, moving_placed, output, inputs):
            if (label, side) != ("514", "CL"):
                return False
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(moving_placed.read_text())
            return True

        monkeypatch.setattr(hcr_channels, "_reviewed_no_shift_rough", retain)
    offsets: list[dict] = []

    def measure_offset(**kwargs: object) -> None:
        offsets.append(kwargs)
        directory = Path(kwargs["output_dir"])
        directory.mkdir(parents=True)
        check = {
            "passed": True,
            "peak_margin": 0.2,
            "shift_yx_px": [1.0, 2.0],
            "expected_shift_yx_px": [1.0, 2.5],
        }
        summary = {
            "validation": {"passed": True, "held_out_tiles": ["009"], "checks": [check]},
            "median_correction_zyx_um": [50.0, 1.0, 2.0],
            "correction_mad_zyx_um": [1.0, 0.5, 0.5],
        }
        (directory / "acquisition-offset.json").write_text(json.dumps(summary))
        Path(kwargs["output_position"]).write_text(
            json.dumps({"reference_channel": kwargs["reference_channel"]})
        )

    monkeypatch.setattr(hcr_channels, "run_acquisition_offset", measure_offset)
    runner = _RecordingRunner()
    fused = {"561": root / "fused-561"}
    thresholds = {"561": 150.0}

    def register(label: str) -> None:
        fused[label], _review = hcr_channels.cross_register_label(
            runner,
            paths,
            payload,
            sources,
            label,
            placement=placement,
            fused=fused,
            fused_thresholds=thresholds,
            preview=root / "preview",
            all_gpu_env={},
            primary_gpu_env={},
        )
        thresholds[label] = 700.0

    # CR-514 can be offset against either channel of the CR 561/638 track: both are measured, and the author chooses.
    with pytest.raises(ReviewRequired, match="Compare the reference channels measured for 514-CR"):
        register("514")
    choice = paths.channel("514") / "CR-offset-reference.json"
    assert set(json.loads(choice.read_text())["options"]) == {"561", "638"}
    assert {offset["reference_channel"] for offset in offsets} == {0, 1}
    choice.write_text(json.dumps(json.loads(choice.read_text()) | {"selected": "638"}))
    register("514")
    register("488")

    commands = runner.commands
    assert "--manual-cr-shift-zyx-um=-1.500000,2.000000,3.250000" in commands["514-placement"]
    assert ("514-CL-global-phase" in commands) == (not reviewed_no_shift)
    if not reviewed_no_shift:
        cl_phase = commands["514-CL-global-phase"]
        assert cl_phase[cl_phase.index("--fixed-tile-dir") + 1] == str(placement["deconv"])
        assert cl_phase[cl_phase.index("--moving-channel") + 1] == "2"
    method6_cl = commands["514-CL-method6"]
    assert method6_cl[method6_cl.index("--moving-position") + 1] == str(
        paths.channel("514") / "CL" / "rough" / "rough.positions.json"
    )
    assert offsets[0]["reference_position"] == root / "canonical" / "positions-CR.json"
    assert json.loads((paths.channel("514") / "CR" / "rough" / "rough.positions.json").read_text()) == {
        "reference_channel": 1
    }
    assert "514-CR-global-phase" not in commands
    method6 = commands["514-CR-method6"]
    matrix = method6[method6.index("--starting-affine-matrix-zyx") + 1]
    assert matrix == hcr_channels.side_starting_matrix(hcr_channels.STARTING_MATRIX_514, "CR")
    assert method6[method6.index("--fixed-mask-threshold") + 1] == "150.0"
    assert json.loads((root / "canonical" / "positions-CR.json").read_text())["tiles"] == [
        {"tile": "b", "source_side": "CR"}
    ]
    # 488 reaches 561 through fused 514: CR-488 is offset against the CR-514 tiles, CL-488 phase-matched to CL 514.
    # CR-514 has one channel, so its 488 offset needs no choice.
    offset_488 = offsets[-1]
    assert offset_488["reference_tile_dir"] == paths.deconv("CR-514")
    assert offset_488["reference_position"] == paths.channel("514") / "CR" / "rough" / "rough.positions.json"
    assert len(offsets) == 3
    assert (
        commands["488-CL-global-phase"][commands["488-CL-global-phase"].index("--fixed-channel") + 1] == "2"
    )
    method6_488 = commands["488-CL-method6"]
    assert method6_488[method6_488.index("--fixed-fused") + 1] == str(fused["514"])
    assert method6_488[method6_488.index("--fixed-mask-threshold") + 1] == "700.0"
    materialize = commands["514-CR-materialize"]
    assert materialize[materialize.index("--output-channel-label") + 1] == "514"
    assert materialize[materialize.index("--source-summary") + 1] == str(
        paths.channel("514") / "CR" / "round-0.json"
    )
    assert materialize[materialize.index("--residual-correction") + 1] == str(
        paths.residual("CR-514", "514") / "correction.json"
    )
    fuse = commands["514-fuse"]
    assert fuse[fuse.index("--output-grid-template") + 1] == str(fused["561"])


FAKE_LIGHTSHEET = """#!/usr/bin/env python3
import json, sys
from pathlib import Path

args = sys.argv[1:]
option = lambda name: Path(args[args.index(name) + 1])
output = option("--output-dir")
if args[1] == "recover-native":
    # The resume cache changes on every run, as reruns of seeded windows rewrite their window records.
    window = output / "window_json" / "w.json"
    window.parent.mkdir(parents=True, exist_ok=True)
    runs = json.loads(window.read_text())["runs"] + 1 if window.exists() else 1
    window.write_text(json.dumps({"runs": runs}))
    (output / "fused_fixed_method8_summary.json").write_text(json.dumps({"windows": [{"level0_json": str(window)}]}))
elif args[1] == "complete-recovery":
    native = json.loads(option("--native-summary").read_text())
    output.mkdir(parents=True)
    (output / "fused_fixed_method8_summary.json").write_text(json.dumps(native | {"input_summary": "x"}))
    (output / "manifest.json").write_text("{}")
"""


def test_recovery_rounds_resume_after_the_cache_moved_on(
    manifest: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "lightsheet").write_text(FAKE_LIGHTSHEET)
    (bin_dir / "lightsheet").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    paths = hcr_acquisition.HcrPaths(root=manifest.parent)
    source = hcr_acquisition.ChannelSource("CL", "514", "CL-main", 2, "514")
    method6 = manifest.parent / "method6.json"
    method6.write_text(json.dumps({"windows": []}))
    adjacency = paths.registration_dir("CL-main", "514") / "registration.measurements.json"
    adjacency.parent.mkdir(parents=True)
    adjacency.write_text("{}")
    proposals = [{"w": {"n": 1}}, {"w": {"n": 2}}]

    def propose(*, complete_summary: Path, master: Path, channel: str) -> None:
        _seed_master(paths, channel, proposals[int(complete_summary.parent.name.split("-")[1])])

    monkeypatch.setattr(hcr_channels, "propose_manual_initializations", propose)
    monkeypatch.setattr(hcr_channels, "seedable_window_count", lambda summary: 5)

    def run() -> Path:
        runner = StageRunner(
            logs=paths.logs, stages=paths.stages, prefix="t", artifact_type="t.v1", dry_run=False
        )
        return hcr_channels._recover_in_rounds(runner, paths, source, method6, primary_gpu_env={})

    first = run()
    assert first.parent.name == "round-1"
    # A second invocation replays round 0 from its own snapshot although the shared cache now holds round 1.
    assert run() == first
    snapshot = json.loads(
        (paths.channel("514") / "CL" / "native-recovery" / "round-0" / "window_json" / "w.json").read_text()
    )
    assert snapshot == {"runs": 1}
    round_1 = json.loads(first.read_text())["windows"][0]["level0_json"]
    assert "native-recovery/round-1/window_json" in round_1


def test_final_transforms_are_scored_against_their_tile_s_other_windows(tmp_path: Path) -> None:
    from squisher_lightsheet.fused_fixed_recovery import final_transform_departures

    rows = []
    for index, translation_x in enumerate((0.0, 0.0, 0.0, 0.0, 100.0)):
        path = Path(
            _window(tmp_path / f"w{index}.json", z=(index * 80, index * 80 + 80), translation_x=translation_x)
        )
        rows.append((path, json.loads(path.read_text())))
    lone = Path(_window(tmp_path / "lone.json", z=(0, 80), moving_tile="d.002.ome.zarr"))
    rows.append((lone, json.loads(lone.read_text())))

    departures = final_transform_departures(rows, maximum_px=60.0)

    assert departures[rows[4][0]]["departs"] is True and departures[rows[4][0]]["score_px"] > 60.0
    assert not any(departures[path]["departs"] for path, _row in rows[:4])
    assert departures[lone] == {
        "status": "insufficient_support",
        "support_count": 0,
        "score_px": None,
        "threshold_px": 60.0,
        "departs": None,
    }


def test_finalization_reviews_the_subtraction_fit_and_final_qc_before_exporting(
    manifest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = manifest.parent
    paths = hcr_acquisition.HcrPaths(root=root)
    fused = {label: root / f"fused-{label}" for label in ("561", "638", "514", "488")}
    runner = _RecordingRunner()

    on_grid: list[Path] = []
    monkeypatch.setattr(
        hcr_workflow, "require_same_multiscale_grid", lambda path, template: on_grid.append(path)
    )

    def finalize() -> dict[str, Path]:
        return hcr_workflow._finalize(runner, paths, fused)

    with pytest.raises(ReviewRequired, match="fused_subtraction_fit_qc.png"):
        finalize()
    assert "subtraction" not in runner.commands
    _approve(root / "subtraction-approval.json")
    with pytest.raises(ReviewRequired, match="same-grid middle-Z and XYZ overlay QC"):
        finalize()
    middle = runner.commands["final-qc-middle-z"]
    assert f"514-488={root / 'final' / 'fusion-514-minus-488' / 'fused.ch0.ome.zarr'}" in middle
    assert {name for name in runner.commands if name.startswith("final-qc-overlay-")} == {
        f"final-qc-overlay-{label}-on-561" for label in ("638", "514", "488", "514-488")
    }
    assert "export-nrrd-561" not in runner.commands
    assert set(on_grid) == set(fused.values()) | {
        root / "final" / "fusion-514-minus-488" / "fused.ch0.ome.zarr"
    }
    _approve(root / "final-qc-approval.json")

    finals = finalize()
    export = runner.commands["export-nrrd-561"]
    assert export[export.index("--level") + 1] == "3" and export[export.index("--channel") + 1] == "561"
    assert finals["nrrd_561"] == root / "final" / "export" / "561.level3.nrrd"


def test_an_export_is_accepted_only_as_the_expected_validated_file(tmp_path: Path) -> None:
    source, output = tmp_path / "fused-561", tmp_path / "561.level3.nrrd"
    receipt = tmp_path / "561.level3.json"
    expected = {"source": str(source), "source_level": 3, "channel": "561", "output": str(output)}
    output.write_bytes(b"nrrd")

    receipt.write_text(json.dumps(expected | {"all_voxels_equal": False}))
    with pytest.raises(ValueError, match="readback did not reproduce"):
        hcr_workflow._require_validated_export(receipt, source=source, output=output)
    receipt.write_text(json.dumps(expected | {"source_level": 2, "all_voxels_equal": True}))
    with pytest.raises(ValueError, match="does not describe the expected export"):
        hcr_workflow._require_validated_export(receipt, source=source, output=output)
    receipt.write_text(json.dumps(expected | {"all_voxels_equal": True}))
    hcr_workflow._require_validated_export(receipt, source=source, output=output)


def test_after_cleanup_a_rerun_only_verifies_the_retained_finals(
    manifest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = json.loads(manifest.read_text())
    manifest.write_text(json.dumps(payload | {"approved": True}))
    record = manifest.parent / "cleanup" / "record.json"
    record.parent.mkdir()
    record.write_text("{}")
    verified = []
    monkeypatch.setattr(
        hcr_workflow, "_cleanup", lambda paths, finals, **_: verified.append(finals) or record
    )

    hcr_workflow.run_hcr_workflow(manifest)

    assert verified == [{}]
    assert not list((manifest.parent / ".stages").glob("*"))


@pytest.fixture
def tltr_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(hcr_workflow, "verify_manifest", lambda _manifest: None)
    monkeypatch.setattr(hcr_workflow, "gpu_uuids", lambda: ("GPU-a", "GPU-b"))
    payload = {
        "dataset": "d",
        "layout": "TL/TR",
        "fixed_side": "TL",
        "moving_side": "TR",
        "acquisitions": [
            _acquisition(tmp_path, "TL-main", "TL", ["561", "638", "514"], [[0, 1], [2]]),
            _acquisition(tmp_path, "TR-main", "TR", ["561", "638", "514"], [[0, 1], [2]]),
        ],
        "approved": True,
    }
    path = tmp_path / "work" / "manifest.json"
    path.parent.mkdir()
    path.write_text(json.dumps(payload))
    paths = hcr_acquisition.HcrPaths(root=path.parent)
    for acquisition in ("TL-main", "TR-main"):
        for label in ("561", "514"):
            registered = paths.registered_positions(acquisition, label)
            registered.parent.mkdir(parents=True)
            registered.write_text(json.dumps({"tiles": [], "acquisition": acquisition}))
    return path


@pytest.mark.parametrize("reviewed_affine", [False, True])
def test_tltr_placement_phase_correlates_tiles_and_lets_the_author_refine_it(
    tltr_manifest: Path, monkeypatch: pytest.MonkeyPatch, reviewed_affine: bool
) -> None:
    root = tltr_manifest.parent
    payload = json.loads(tltr_manifest.read_text())
    sources = hcr_workflow.supported_sources(payload)
    joins: list[dict] = []

    def join(**kwargs: object) -> None:
        joins.append(kwargs)
        output = Path(kwargs["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("{}")

    def split(coarse: Path, output: Path) -> None:
        output.mkdir(parents=True, exist_ok=True)
        for side in ("tl", "tr"):
            (output / f"{side}.positions.json").write_text("{}")

    monkeypatch.setattr(hcr_reference, "write_joined_sides", join)
    monkeypatch.setattr(hcr_reference, "split_sides", split)
    published: list[dict] = []

    def publish(**kwargs: object) -> None:
        published.append(kwargs)
        final = Path(kwargs["output_dir"])
        final.mkdir(parents=True)
        (final / "positions.json").write_text("{}")
        (final / "registration.json").write_text("{}")

    monkeypatch.setattr(hcr_reference, "publish_tltr_placement", publish)
    overlays: list[Path] = []

    def overlay(coarse: Path, output_dir: Path, **_: object) -> None:
        overlays.append(coarse)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "zmedian_overlay.json").write_text("{}")

    monkeypatch.setattr(hcr_reference, "render_coarse_overlay", overlay)
    runner = _RecordingRunner()

    def place() -> dict[str, Path]:
        return hcr_reference.place_tltr(
            runner, hcr_acquisition.HcrPaths(root=root), payload, sources, {"TL/561": 900.0}, all_gpu_env={}
        )

    with pytest.raises(ReviewRequired, match="TL/TR coarse placement"):
        place()
    assert "--mode" in runner.commands["placement-metadata"]
    assert runner.commands["placement-metadata"][-1] == "tltr_x_join_center_z_phase"
    phase = runner.commands["placement-phase-coarse"]
    assert phase[1] == "rough-phase" and phase[phase.index("--channel") + 1] == "0"
    assert joins[0]["right_shift_zyx_um"].tolist() == [0.0, 0.0, 0.0]

    # The author replaces the phase shift with a manual one, which places TR into its own folder without phase.
    request = root / "placement-coarse.json"
    request.write_text(json.dumps(json.loads(request.read_text()) | {"manual_tr_shift_zyx_um": [0, -2, 30]}))
    (root / "placement-coarse-approval.json").unlink()
    with pytest.raises(ReviewRequired, match="TL/TR coarse placement"):
        place()
    assert joins[-1]["right_shift_zyx_um"].tolist() == [0.0, -2.0, 30.0]
    # The manual placement has no phase overlays, so its own overlay is rendered for the review.
    assert overlays[-1].name == "joined.positions.json"
    assert not any(name.startswith("placement-phase-coarse-manual") for name in runner.commands)

    _approve(root / "placement-coarse-approval.json")
    if reviewed_affine:
        candidate = root / "reviewed-affine.json"
        candidate.write_text("{}")
        request.write_text(
            json.dumps(
                json.loads(request.read_text())
                | {
                    "reviewed_global_affine": {
                        "path": str(candidate),
                        "sha256": hcr_reference.sha256_file(candidate),
                    }
                }
            )
        )
    with pytest.raises(ReviewRequired, match="TL/TR placement overlay"):
        place()
    method8 = runner.commands["placement-method8"]
    assert method8[method8.index("--fixed-position") + 1].endswith(
        "coarse-manual_+0.000_-2.000_+30.000/sides/tl.positions.json"
    )
    assert method8[method8.index("--fixed-mask-threshold") + 1] == "900.0"
    assert "--preseed-matrix-zyx" in method8
    assert method8[method8.index("--preseed-matrix-zyx") + 1] == "1,0,0,0,1,0,0,0,1"
    _approve(root / "placement-approval.json")
    assert place()["positions"] == root / "placement" / "final" / "positions.json"
    approval = json.loads((root / "placement-approval.json").read_text())
    assert ("registration" in approval["identity"]) == reviewed_affine
    if reviewed_affine:
        assert published[-1]["reviewed_global_affine"] == candidate
        candidate.write_text("changed")
        with pytest.raises(ValueError, match="changed since selection"):
            place()


def test_tltr_labels_are_placed_with_the_net_tr_shift_and_no_flip(
    tltr_manifest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tltr_manifest.parent
    payload = json.loads(tltr_manifest.read_text())
    sources = hcr_workflow.supported_sources(payload)
    positions = root / "placement" / "final" / "positions.json"
    positions.parent.mkdir(parents=True)
    positions.write_text(
        json.dumps({"alignment": {"net_right_shift_after_centroid_um_zyx": [1.0, 2.0, 3.0]}})
    )
    joins: list[dict] = []

    def join(**kwargs: object) -> None:
        joins.append(kwargs)

    def split(coarse: Path, output: Path) -> None:
        for side in ("tl", "tr"):
            (output / f"{side}.positions.json").write_text("{}")

    monkeypatch.setattr(hcr_channels, "write_joined_sides", join)
    monkeypatch.setattr(hcr_channels, "split_sides", split)
    placed = root / "channels" / "514" / "placed"
    by_side = {source.side: source for source in sources if source.label == "514"}
    runner = _RecordingRunner()

    hcr_channels._place_label(
        runner,
        hcr_acquisition.HcrPaths(root=root),
        payload,
        sources,
        "514",
        by_side,
        placement={"positions": positions},
        placed=placed,
    )

    assert runner.commands["514-placement-metadata"][-1] == "tltr_x_join_center_z_phase"
    assert joins[0]["right_shift_zyx_um"].tolist() == [1.0, 2.0, 3.0]
    assert "514-placement" not in runner.commands
    matrix = hcr_channels.side_starting_matrix(hcr_channels.STARTING_MATRIX_514, "TR")
    assert [float(value) for value in matrix.split(",")] == [
        float(value) for value in hcr_channels.STARTING_MATRIX_514.split(",")
    ]


def test_windows_left_when_exclusion_exhausts_support_are_not_scored(tmp_path: Path) -> None:
    from squisher_lightsheet.fused_fixed_recovery import final_transform_departures

    rows = []
    # The middle window departs from its two neighbours; excluding it leaves two windows, too few to compare.
    for index, translation_x in enumerate((0.0, 2000.0, 0.0)):
        path = Path(
            _window(tmp_path / f"w{index}.json", z=(index * 80, index * 80 + 80), translation_x=translation_x)
        )
        rows.append((path, json.loads(path.read_text())))

    departures = final_transform_departures(rows, maximum_px=60.0)

    statuses = sorted((value["status"], value["departs"]) for value in departures.values())
    assert statuses == [("insufficient_support", None), ("insufficient_support", None), ("scored", True)]


def test_native_only_561_requires_exact_review_and_preserves_native_corrections(tmp_path: Path) -> None:
    import numpy as np

    canonical = tmp_path / "canonical"
    match = canonical / "match-561"
    match.mkdir(parents=True)
    (match / "match.json").write_text("{}")
    (match / "registered-view-correction.json").write_text("{}")
    registration = tmp_path / "registration.json"
    registration.write_text(
        json.dumps(
            {
                "tiles": [
                    {"path": str(tmp_path / "a"), "source_view": "L"},
                    {"path": str(tmp_path / "b"), "source_view": "R"},
                ]
            }
        )
    )
    corrections = {}
    for view, name, gain in (("L", "a", 0.6), ("R", "b", 0.8)):
        corrections[view] = tmp_path / f"{view}.json"
        corrections[view].write_text(
            json.dumps(
                {
                    "schema_version": 3,
                    "kind": "squisher.residual-correction",
                    "channel": 0,
                    "coordinate": "normalized-original-raw-zyx",
                    "basis": "cosine-xy2-z-scaled",
                    "coefficient": [[0, 0, 0, 0, 0]],
                    "global_scale": 1,
                    "sources": [
                        {
                            "path": str(tmp_path / name),
                            "source_sha256": name,
                            "shape_zyx": [4, 5, 6],
                            "gain": gain,
                            "coefficient": [[0, 0, 0, 0, 0]],
                        }
                    ],
                }
            )
        )
    runner = _RecordingRunner()
    call = lambda: hcr_reference.reviewed_native_561_corrections(
        runner, canonical=canonical, registration=registration, corrections=corrections, channel=0
    )
    assert call() is None
    approval = canonical / "intensity-561-native-only-approval.json"
    identity = {
        "registration": hcr_reference.sha256_file(registration),
        "rejected_match": hcr_reference.sha256_file(match / "match.json"),
        "rejected_field": hcr_reference.sha256_file(match / "registered-view-correction.json"),
        **{f"native_{view}": hcr_reference.sha256_file(path) for view, path in corrections.items()},
    }
    approval.write_text(
        json.dumps(
            {
                "artifact_type": "squisher_lightsheet.hcr_native_only_561_approval.v1",
                "approved": False,
                "identity": identity,
            }
        )
    )
    with pytest.raises(ReviewRequired):
        call()
    payload = json.loads(approval.read_text())
    payload["approved"] = True
    approval.write_text(json.dumps(payload))
    output = call()
    correction = json.loads(output.read_text())
    np.testing.assert_allclose([row["gain"] for row in correction["sources"]], [0.6, 0.8])
    assert correction["global_scale"] == 1
    corrections["R"].write_text("changed")
    with pytest.raises(ValueError, match="identity no longer matches"):
        call()


@pytest.mark.parametrize("changed", ["inputs", "before", "phase_log"])
def test_reviewed_no_shift_rough_preserves_positions_and_refuses_stale_review(
    tmp_path: Path,
    changed: str,
) -> None:
    runner = StageRunner(
        logs=tmp_path / "logs", stages=tmp_path / "stages", prefix="hcr", artifact_type="stage", dry_run=False
    )
    moving = tmp_path / "moving.json"
    original = {
        "tiles": [{"tile": "a", "translation_um": {"z": 4, "y": 5, "x": 6}, "source_side": "TL"}],
        "alignment": {"native": True},
    }
    moving.write_text(json.dumps(original))
    output = tmp_path / "rough" / "rough.positions.json"
    output.parent.mkdir()
    inputs = {"moving": {"sha256": "native"}, "reference": {"sha256": "fixed"}}
    attempt = runner.stages / "514-TL-global-phase.started.json"
    attempt.write_text(json.dumps({"inputs": inputs}))
    log = runner.logs / "514-TL-global-phase.log"
    log.write_text("Bounded global phase did not improve correlation")
    for name in ("before.png", "after.png"):
        (output.parent / name).write_bytes(name.encode())
    call = lambda: hcr_channels._reviewed_no_shift_rough(
        runner, label="514", side="TL", moving_placed=moving, output=output, inputs=inputs
    )
    assert call() is False
    assert not output.exists()
    identity = {
        "inputs": workflow_runner.json_sha256(inputs),
        "phase_attempt": workflow_runner.sha256_file(attempt),
        "phase_log": workflow_runner.sha256_file(log),
        **{name: workflow_runner.sha256_file(output.parent / f"{name}.png") for name in ("before", "after")},
    }
    approval = output.parent / "no-shift-approval.json"
    approval.write_text(
        json.dumps(
            {
                "artifact_type": "squisher_lightsheet.hcr_rough_no_shift_approval.v1",
                "approved": False,
                "identity": identity,
            }
        )
    )
    with pytest.raises(ReviewRequired):
        call()
    assert not output.exists()
    _approve(approval)
    assert call() is True
    retained = json.loads(output.read_text())
    assert retained["tiles"] == original["tiles"]
    assert retained["alignment"] == original["alignment"]
    assert retained["rough_placement_review"]["total_shift_to_apply_moving_zyx_um"] == [0, 0, 0]
    if changed == "inputs":
        inputs["moving"] = {"sha256": "changed"}
    elif changed == "before":
        (output.parent / "before.png").write_bytes(b"changed")
    else:
        log.write_text("changed")
    with pytest.raises(ValueError, match="different inputs|identity no longer matches"):
        call()


@pytest.mark.parametrize("changed", ["measurement", "inputs", "qc"])
def test_failed_acquisition_offset_requires_exact_visual_approval_before_method6(
    manifest: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed: str,
) -> None:
    payload = json.loads(manifest.read_text())
    sources = hcr_acquisition.channel_sources(payload)
    by_key = {(s.side, s.label): s for s in sources}
    paths = hcr_acquisition.HcrPaths(root=manifest.parent)
    moving = paths.root / "moving.json"
    moving_payload = {
        "tiles": [
            {
                "tile": "CR-514.000.ome.zarr",
                "source_side": "CR",
                "translation_um": {"z": 2, "y": 3, "x": 4},
                "scale_um": {"z": -1.8, "y": 0.3, "x": -0.3},
            }
        ]
    }
    moving.write_text(json.dumps(moving_payload))
    reference = paths.root / "reference.json"
    reference.write_text("{}")
    output = paths.channel("514") / "CR/rough/rough.positions.json"
    inputs = {"moving": "native", "reference": "fixed"}

    def measure(**kwargs):
        directory = kwargs["output_dir"]
        directory.mkdir(parents=True)
        check = {"passed": False, "peak_margin": 0.1, "shift_yx_px": [2, 1], "expected_shift_yx_px": [0, 0]}
        summary = {
            "artifact_type": "squisher_lightsheet.acquisition_offset.v2",
            "moving_position": str(moving),
            "reference_position": str(reference),
            "moving_channel": 0,
            "reference_channel": kwargs["reference_channel"],
            "median_correction_zyx_um": [-4, -2, 3],
            "correction_mad_zyx_um": [0.3, 0.8, 0.9],
            "validation": {"passed": False, "held_out_tiles": ["000"], "checks": [check]},
        }
        (directory / "acquisition-offset.json").write_text(json.dumps(summary))

    monkeypatch.setattr(hcr_channels, "run_acquisition_offset", measure)
    runner = _RecordingRunner()
    call = lambda: hcr_channels._offset_placement(
        runner,
        paths,
        payload,
        sources,
        by_key[("CR", "514")],
        by_key[("CR", "561")],
        moving_placed=moving,
        reference_positions=reference,
        reference_tiles=paths.root / "reference-tiles",
        output=output,
        inputs=inputs,
    )
    with pytest.raises(ReviewRequired):
        call()
    choice = paths.channel("514") / "CR-offset-reference.json"
    choice.write_text(json.dumps(json.loads(choice.read_text()) | {"selected": "638"}))
    with pytest.raises(ValueError, match="fails held-out validation"):
        call()
    assert not output.exists()
    review = output.parent / "offset-review"
    review.mkdir()
    qc = review / "held-out-638-contact-sheet.png"
    qc.write_bytes(b"reviewed cells")
    render = review / "render.json"
    render.write_text("{}")
    measurement = output.parent / "offset-638/acquisition-offset.json"
    approval = measurement.parent / "rough-initializer-approval.json"
    identity = {
        "inputs": workflow_runner.json_sha256(inputs),
        "moving": workflow_runner.sha256_file(moving),
        "measurement": workflow_runner.sha256_file(measurement),
        "qc": workflow_runner.sha256_file(qc),
        "render": workflow_runner.sha256_file(render),
    }
    approval.write_text(
        json.dumps(
            {
                "artifact_type": "squisher_lightsheet.hcr_offset_initializer_approval.v1",
                "approved": False,
                "identity": identity,
            }
        )
    )
    with pytest.raises(ReviewRequired):
        call()
    assert not output.exists()
    _approve(approval)
    call()
    placed = json.loads(output.read_text())
    assert placed["tiles"][0]["translation_um"] == {"z": -2.0, "y": 1.0, "x": 7.0}
    assert placed["tiles"][0]["scale_um"] == moving_payload["tiles"][0]["scale_um"]
    assert placed["rough_placement_review"]["approval"] == str(approval)
    assert placed["rough_placement_review"]["held_out_validation_passed"] is False
    assert json.loads(measurement.read_text())["validation"]["passed"] is False
    if changed == "measurement":
        measurement.write_text(json.dumps(json.loads(measurement.read_text()) | {"changed": True}))
        # Selection identity is also bound to these measurements.
        with pytest.raises(ValueError, match="different measurements"):
            call()
    else:
        if changed == "inputs":
            inputs["moving"] = "changed"
        else:
            qc.write_bytes(b"changed")
        with pytest.raises(ValueError, match="identity no longer matches"):
            call()


def test_hcr_native_registration_uses_edge_retention_policy(manifest, monkeypatch):
    monkeypatch.setattr(hcr_acquisition, "write_metadata_grid", lambda *_args: None)
    runner = _RecordingRunner()
    payload = json.loads(manifest.read_text())
    thresholds = {key: 250 for key in hcr_acquisition.registration_threshold_keys(payload)}
    hcr_acquisition.register_native_geometry(
        runner, hcr_acquisition.HcrPaths(manifest.parent), payload, thresholds, primary_gpu_env={}
    )
    registrations = [
        argv for argv in runner.commands.values() if argv[:2] == ["lightsheet-stitch", "register"]
    ]
    assert registrations
    assert all(
        "--exclude-disconnected" not in argv and "--allow-disconnected" not in argv for argv in registrations
    )
