import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from squisher_lightsheet.artifact_io import sha256_file
from squisher_lightsheet.cli import app
from squisher_lightsheet.surface_layers import assign_surface_layers


def inputs(tmp_path: Path):
    surface = tmp_path / 'surface.npz'
    vertices = np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    geometry = {}
    for prefix in ('image', 'original_image'):
        geometry[f'{prefix}_origin_xyz_um'] = np.zeros(3)
        geometry[f'{prefix}_spacing_xyz_um'] = np.ones(3)
        geometry[f'{prefix}_direction'] = np.eye(3)
    np.savez(surface, vertices_world_xyz_um=vertices, vertices_intrinsic_xyz_um=vertices,
             faces=[[0, 1, 2]], cavity_direction_world_xyz=[1., 0., 0.], **geometry)
    coord = tmp_path / 'coords'
    coord.mkdir()
    rows = 7
    table = pa.table({
        'cell': [f'cell{i}' for i in range(rows)], 'dataset': ['test']*rows,
        'roi': ['test']*rows, 'label': np.arange(rows), 'u': np.zeros(rows), 'v': np.zeros(rows),
        'r_vz_um': [5., 10., 15., 20., 25., None, 5.], 'face_id': np.zeros(rows, dtype=int),
        'barycentric_0': np.ones(rows), 'barycentric_1': np.zeros(rows), 'barycentric_2': np.zeros(rows),
        'on_surface_boundary': [False]*rows, 'on_tissue_side': [True]*6+[False],
    })
    pq.write_table(table, coord / 'cells.parquet')
    (coord / 'metadata.json').write_text(json.dumps({
        'output_sha256': sha256_file(coord/'cells.parquet'), 'surface_sha256': sha256_file(surface),
        'rows': rows,
    }))
    bounds = tmp_path / 'boundaries'
    bounds.mkdir()
    np.savez(bounds/'boundaries.npz', inner_um=[10.]*3, outer_um=[20.]*3,
             estimated_inner=[True]*3, estimated_outer=[True]*3, wall_support=[True]*3)
    meta = {'surface_sha256': sha256_file(surface),
            'outputs': {'boundaries.npz': sha256_file(bounds/'boundaries.npz')},
            'inner': {'peak_fraction': 1., 'smooth_um': 80.},
            'outer': {'peak_fraction': .5, 'smooth_um': 80.}}
    (bounds/'metadata.json').write_text(json.dumps(meta))
    return coord, surface, bounds/'boundaries.npz'


def test_cli_keeps_cutoff_endpoints_validity_and_actual_fractions(tmp_path):
    coord, surface, bounds = inputs(tmp_path)
    out = tmp_path/'layers'
    result = CliRunner().invoke(app, ['eomes-layers', '--coordinates-dir', str(coord),
        '--surface', str(surface), '--boundaries', str(bounds), '--output-dir', str(out)])
    assert result.exit_code == 0, result.output
    table = pq.read_table(out/'cells.parquet').to_pydict()
    assert table['layer'] == ['VZ', 'VZ', 'SVZ', 'SVZ', 'CP', 'CP', 'CP']
    assert table['boundary_defined'] == [True]*5+[False, False]
    assert table['inner_boundary_um'] == [10.]*5+[None, None]
    meta = json.loads((out/'metadata.json').read_text())
    assert meta['counts'] == {'VZ': 2, 'SVZ': 2, 'CP': 3}
    assert meta['inner']['peak_fraction'] == 1.
    assert meta['outer']['peak_fraction'] == .5


def test_rejects_different_surface_before_any_output(tmp_path):
    coord, surface, bounds = inputs(tmp_path)
    metadata = json.loads((bounds.parent/'metadata.json').read_text())
    metadata['surface_sha256'] = 'different-surface'
    (bounds.parent/'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='identities differ'):
        assign_surface_layers(coordinates_dir=coord, surface_path=surface,
                              boundaries_path=bounds, output_dir=tmp_path/'layers')
    assert not (tmp_path/'layers').exists()
