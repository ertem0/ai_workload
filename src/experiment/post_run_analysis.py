from __future__ import annotations

from pathlib import Path
from typing import Any

from src.analysis.expert_routing_report import run_expert_routing_analysis
from src.experiment.logging import log_step


def expert_routing_enabled(config_dict: dict[str, Any]) -> bool:
    metrics_cfg = config_dict["metrics"]
    return bool(metrics_cfg.get("calculate_expert_routing", False))


def run_enabled_post_analysis(
    config_dict: dict[str, Any],
    run_summary: dict[str, Any],
    output_dir: Path,
) -> None:
    if not expert_routing_enabled(config_dict):
        return

    raw_trace_path = run_summary.get("raw_trace_path")
    if raw_trace_path is None or not Path(raw_trace_path).exists():
        raise RuntimeError(
            "Phase 1 completed without producing expert_traces_raw.pkl, "
            "so expert routing analysis cannot begin."
        )

    log_step("Phase 2: starting expert routing analysis from saved trace file.", phase="analysis")
    run_expert_routing_analysis(
        Path(raw_trace_path),
        output_dir=output_dir,
    )
