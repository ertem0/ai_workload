from __future__ import annotations

from pathlib import Path
import re
import shutil


def make_experiment_output_dir(experiment_name: str) -> Path:
    safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", experiment_name).strip("_").lower()
    output_dir = Path("results") / safe_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def reset_experiment_output_dir(output_dir: Path) -> None:
    """
    Remove any previous artifacts so each run starts with a clean results directory.
    """

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        return

    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
