from pathlib import Path
import sys

from squisher_lightsheet.fused_fixed_recovery import _load_sweep_module


def test_custom_sweep_loads_selected_path_despite_equal_basename(tmp_path: Path) -> None:
    paths = [tmp_path / name / 'selected_sweep.py' for name in ('first', 'second')]
    for path, marker in zip(paths, ('first', 'second'), strict=True):
        path.parent.mkdir()
        path.write_text(f'MARKER = {marker!r}\n')
    original_path = list(sys.path)
    try:
        assert _load_sweep_module(paths[0]).MARKER == 'first'
        assert _load_sweep_module(paths[1]).MARKER == 'second'
        assert sys.path == original_path
    finally:
        sys.path[:] = original_path
        sys.modules.pop('selected_sweep', None)


def test_default_backend_chain_uses_packaged_modules() -> None:
    from squisher_lightsheet import fused_fixed
    from squisher_lightsheet._backends import (
        fused_fixed_contact_sheet,
        fused_fixed_sweep,
        fused_fixed_trial,
    )

    assert _load_sweep_module() is fused_fixed_sweep
    assert fused_fixed.DEFAULT_SWEEP_RUNNER == Path(fused_fixed_sweep.__file__)
    for parser in (fused_fixed_sweep.build_parser(), fused_fixed_trial.build_parser()):
        assert parser.get_default("moving_source_position") == fused_fixed_trial.DEFAULT_MOVING_POSITION
    assert callable(fused_fixed_sweep.run)
    assert callable(fused_fixed_trial.run)
    assert callable(fused_fixed_contact_sheet.run)


def test_backend_executable_selects_requested_interpreter_environment(tmp_path: Path) -> None:
    from squisher_lightsheet._backends import backend_executable

    interpreter = tmp_path / 'bin' / 'python'
    interpreter.parent.mkdir()
    interpreter.symlink_to(sys.executable)
    executable = interpreter.parent / 'lightsheet-fused-fixed-sweep'
    executable.touch()

    assert backend_executable(executable.name, python=interpreter) == executable


def test_backend_executable_requires_package_installation(tmp_path: Path) -> None:
    import pytest
    from squisher_lightsheet._backends import backend_executable

    with pytest.raises(FileNotFoundError, match='install squisher-lightsheet'):
        backend_executable('lightsheet-fused-fixed-sweep', python=tmp_path / 'python')


def test_custom_sweep_supports_module_aware_declarations(tmp_path: Path) -> None:
    runner = tmp_path / 'sweep.py'
    runner.write_text(
        'from __future__ import annotations\n'
        'from dataclasses import dataclass\n'
        '@dataclass\n'
        'class Settings:\n'
        '    workers: int = 1\n'
    )

    assert _load_sweep_module(runner).Settings().workers == 1
