from __future__ import annotations

from pathlib import Path
import os
import shlex
import subprocess
import sys


LEGACY_DIR = Path(__file__).resolve().parent / "_legacy"


def command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_legacy_script(script_name: str, args: list[str], *, dry_run: bool = False) -> str:
    command = [sys.executable, str(LEGACY_DIR / script_name), *args]
    text = command_text(command)
    if dry_run:
        print(f"DRY RUN not executed: {text}", flush=True)
        return text
    print(text, flush=True)
    env = os.environ.copy()
    extra_pythonpath = env.pop("SQUISHER_LEGACY_EXTRA_PYTHONPATH", "")
    if extra_pythonpath:
        current_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            extra_pythonpath
            if not current_pythonpath
            else f"{extra_pythonpath}{os.pathsep}{current_pythonpath}"
        )
    subprocess.run(command, check=True, env=env)
    return text
