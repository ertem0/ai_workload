from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

from src.experiment.config import load_dataset_prompts, load_yaml_config
from src.experiment.logging import configure_logging, log_step
from src.experiment.output import make_experiment_output_dir, reset_experiment_output_dir
from src.experiment.post_run_analysis import run_enabled_post_analysis
from src.experiment.runner import run_experiment


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the benchmarking entrypoint.
    """

    parser = argparse.ArgumentParser(
        description="Run YAML-driven AIMC benchmarks for MoE models."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the experiment YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    log_step(f"Reading config from {config_path}.")
    config_dict = load_yaml_config(config_path)
    configure_logging(config_dict.get("logging"))
    config_dict["dataset"] = load_dataset_prompts(config_dict["dataset"], config_path)
    output_dir = make_experiment_output_dir(config_dict["experiment_name"])
    reset_experiment_output_dir(output_dir)
    log_path = output_dir / "run.log"

    with log_path.open("w", encoding="utf-8") as log_handle:
        with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
            log_step(f"Writing combined logs to {log_path.resolve()}", phase="setup")
            run_summary = run_experiment(config_dict, output_dir)
            run_enabled_post_analysis(config_dict, run_summary, output_dir)


if __name__ == "__main__":
    main()
