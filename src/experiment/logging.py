from __future__ import annotations

import re
from datetime import datetime
from typing import Any


_enabled_phases: set[str] | None = None
_disabled_phases: set[str] = set()
_show_phase = True
_progress_interval = 10
_progress_throttled_phases = {"prompt", "generation"}
_progress_pattern = re.compile(r"\b(\d+)/(\d+)\b")


def configure_logging(logging_config: dict[str, Any] | None) -> None:
    global _enabled_phases, _disabled_phases, _show_phase, _progress_interval

    logging_config = logging_config or {}
    enabled_phases = logging_config.get("enabled_phases")
    disabled_phases = logging_config.get("disabled_phases", [])

    _enabled_phases = (
        {str(phase) for phase in enabled_phases}
        if enabled_phases is not None
        else None
    )
    _disabled_phases = {str(phase) for phase in disabled_phases}
    _show_phase = bool(logging_config.get("show_phase", True))
    _progress_interval = int(logging_config.get("progress_interval", 10))


def _should_throttle_progress(message: str, phase: str) -> bool:
    if phase not in _progress_throttled_phases or _progress_interval <= 1:
        return False

    match = _progress_pattern.search(message)
    if match is None:
        return False

    current = int(match.group(1))
    total = int(match.group(2))
    return current % _progress_interval != 0 and current != total


def log_step(message: str, *, phase: str = "general") -> None:
    if _enabled_phases is not None and phase not in _enabled_phases:
        return
    if phase in _disabled_phases:
        return
    if _should_throttle_progress(message, phase):
        return

    ts = datetime.now().strftime("%H:%M:%S")
    prefix = f"[{ts}][MAIN][{phase}]" if _show_phase else f"[{ts}][MAIN]"
    print(f"{prefix} {message}", flush=True)


def print_inference_outputs(outputs: list[dict[str, str]]) -> None:
    if not outputs:
        return

    print("=" * 96)
    print("Inference Outputs")
    print("=" * 96)
    for item in outputs:
        print(f"Prompt {item['prompt_index']}")
        print(f"Input : {item['prompt']}")
        print(f"Output: {item['continuation'] or item['decoded_text']}")
        print("-" * 96)
