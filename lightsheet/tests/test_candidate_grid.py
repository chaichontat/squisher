from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from squisher_lightsheet import candidate_grid


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def test_render_candidate_grid_writes_variant_payloads_and_sheets(tmp_path, monkeypatch) -> None:
    position = tmp_path / "positions.json"
    registration = tmp_path / "registration.json"
    candidates = tmp_path / "candidates.json"
    output_dir = tmp_path / "candidate-grid"

    _write_json(
        position,
        {
            "tiles": [
                {
                    "tile": "230Tnc-CL-405.001.ome.tif",
                    "side": "L",
                    "path": "/sample/230Tnc-CL-405.001.ome.tif",
                    "translation_um": {"z": 10.0, "y": 20.0, "x": 30.0},
                    "scale_um": {"z": 2.0, "y": 0.5, "x": 0.25},
                },
                {
                    "tile": "230Tnc-CL-405.005.ome.tif",
                    "side": "L",
                    "path": "/sample/230Tnc-CL-405.005.ome.tif",
                    "translation_um": {"z": 100.0, "y": 200.0, "x": 300.0},
                    "scale_um": {"z": 2.0, "y": 0.5, "x": 0.25},
                },
            ]
        },
    )
    _write_json(
        registration,
        {
            "tiles": [
                {
                    "tile": "230Tnc-CL-405.001.ome.tif",
                    "path": "/sample/230Tnc-CL-405.001.ome.tif",
                    "source_view": "L",
                    "stage_translation_um": {"z": 10.0, "y": 20.0, "x": 30.0},
                    "stage_scale_um": {"z": 2.0, "y": 0.5, "x": 0.25},
                },
                {
                    "tile": "230Tnc-CL-405.005.ome.tif",
                    "path": "/sample/230Tnc-CL-405.005.ome.tif",
                    "source_view": "L",
                    "stage_translation_um": {"z": 100.0, "y": 200.0, "x": 300.0},
                    "stage_scale_um": {"z": 2.0, "y": 0.5, "x": 0.25},
                },
            ]
        },
    )
    _write_json(
        candidates,
        {
            "tiles": [
                {
                    "tile": "230Tnc-CL-405.001.ome.tif",
                    "label": "L001",
                    "candidates": {"smallY": [0.2, 9.0, -0.1]},
                },
                {
                    "tile": "230Tnc-CL-405.005.ome.tif",
                    "label": "L005",
                    "candidates": {"zero": [0.0, 0.0, 0.0], "minusY58": [5.0, -58.0, 22.0]},
                },
            ]
        },
    )

    def fake_render_registration_qc(
        *,
        position_input: Path,
        registration_input: Path,
        output_dir: Path,
        channel: int,
        level: int,
        center_y_xz: bool,
    ) -> None:
        assert position_input.exists()
        assert registration_input.exists()
        assert channel == 0
        assert level == 4
        assert center_y_xz is False
        Image.new("RGB", (100, 80), "black").save(output_dir / "level4_registered_lr_xy_isoZ_yellowOverlay_ch0.png")
        _write_json(
            output_dir / "level4_registered_lr_isoZ_yellowOverlay_ch0.json",
            {
                "tiles": [
                    {
                        "tile": "230Tnc-CL-405.001.ome.tif",
                        "side": "L",
                        "level_start_zyx": [0, 5, 7],
                        "sampled_shape_zyx": [10, 20, 30],
                    }
                ]
            },
        )

    monkeypatch.setattr(candidate_grid, "render_registration_qc", fake_render_registration_qc)

    summary_path = candidate_grid.render_candidate_grid(
        position_input=position,
        registration_input=registration,
        candidate_json=candidates,
        output_dir=output_dir,
        render_jobs=1,
    )

    records = json.loads(summary_path.read_text())
    assert [record["name"] for record in records] == ["L001_smallY__L005_minusY58", "L001_smallY__L005_zero"]
    selected_position = json.loads(Path(records[0]["position"]).read_text())
    tile001 = selected_position["tiles"][0]["translation_um"]
    tile005 = selected_position["tiles"][1]["translation_um"]
    assert tile001 == {"z": 10.4, "y": 24.5, "x": 29.975}
    assert tile005 == {"z": 110.0, "y": 171.0, "x": 305.5}
    assert (output_dir / "candidate_grid_sheet.png").exists()
    assert (output_dir / "candidate_grid_boundaries_sheet.png").exists()
