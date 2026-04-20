from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml

from src.metrics.aimc import AIMCMetricTracker
from src.metrics.cws import CWSTracker
from src.metrics.static_analyzer import calculate_tiling_efficiency, print_tiling_report
from src.models.loader import load_model_and_tokenizer


def log_step(message: str) -> None:
    print(f"[MAIN] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the benchmarking entrypoint.
    """

    parser = argparse.ArgumentParser(
        description="Run YAML-driven AIMC benchmarks for MoE models."
    )
    parser.add_argument(
        "--config",
        default="configs/experiment_cws.yaml",
        help="Path to the experiment YAML file.",
    )
    return parser.parse_args()


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """
    Load a YAML experiment configuration into a Python dictionary.
    """

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    log_step(f"Reading config from {config_path}.")
    config_dict = load_yaml_config(config_path)

    log_step(f"Experiment: {config_dict['experiment_name']}")
    log_step(f"Loading model: {config_dict['model']['id']}")

    model, tokenizer, model_info = load_model_and_tokenizer(config_dict)
    log_step("Model and tokenizer ready.")

    cws_tracker: CWSTracker | None = None
    if config_dict["metrics"].get("calculate_cws", False):
        log_step("Initializing CWS tracker.")
        cws_tracker = CWSTracker(model=model, top_k=int(config_dict["model"]["top_k"]))
        cws_tracker.register_hooks()
        log_step(f"CWS hooks registered on {len(cws_tracker.handles)} gate modules.")

    aimc_runtime_tracker: AIMCMetricTracker | None = None
    aimc_metric_flags = (
        "calculate_system_arithmetic_intensity",
        "calculate_crossbar_arithmetic_intensity",
        "calculate_linear_vs_nonlinear_ratio",
        "calculate_static_vs_dynamic_ratio",
        "calculate_activation_sparsity",
    )
    if any(config_dict["metrics"].get(flag, False) for flag in aimc_metric_flags):
        log_step("Initializing dynamic AIMC metric tracker.")
        aimc_runtime_tracker = AIMCMetricTracker(model=model, metrics_cfg=config_dict["metrics"])
        aimc_runtime_tracker.register_hooks()
        log_step(f"AIMC runtime hooks registered: {len(aimc_runtime_tracker.handles)}.")

    if config_dict["metrics"].get("calculate_tiling_efficiency", False):
        crossbar_size = tuple(config_dict["metrics"]["crossbar_dimensions"])
        log_step(
            f"Running static tiling analysis for crossbar "
            f"{crossbar_size[0]}x{crossbar_size[1]}."
        )
        tiling_metrics = calculate_tiling_efficiency(model, crossbar_size)
        print_tiling_report(tiling_metrics)

    prompts = config_dict["dataset"]
    total_input_tokens = 0
    log_step(f"Running {len(prompts)} prompts.")

    try:
        for prompt_index, prompt in enumerate(prompts, start=1):
            log_step(f"Tokenizing prompt {prompt_index}/{len(prompts)}.")
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=False,
            )

            # The loader forces the model onto CPU, so inputs are also placed on CPU.
            encoded = {name: tensor.to("cpu") for name, tensor in encoded.items()}
            total_input_tokens += int(encoded["input_ids"].numel())

            log_step(
                f"Starting forward pass {prompt_index}/{len(prompts)} "
                f"(sequence length {encoded['input_ids'].shape[-1]})."
            )

            # For CWS, a plain forward pass is sufficient and cheaper than generation.
            if aimc_runtime_tracker is not None:
                aimc_runtime_tracker.start_prompt()
            with torch.inference_mode():
                model(**encoded)
            log_step(f"Completed forward pass {prompt_index}/{len(prompts)}.")
            if aimc_runtime_tracker is not None:
                aimc_runtime_tracker.finish_prompt()

                # Disable forward-hook accounting during the auxiliary fvcore trace
                # so routing counts and runtime hook metrics are not double counted.
                log_step(f"Starting fvcore MAC analysis for prompt {prompt_index}/{len(prompts)}.")
                aimc_runtime_tracker.set_enabled(False)
                if cws_tracker is not None:
                    cws_tracker.set_enabled(False)
                aimc_runtime_tracker.analyze_flops_for_prompt(encoded)
                aimc_runtime_tracker.set_enabled(True)
                if cws_tracker is not None:
                    cws_tracker.set_enabled(True)
                log_step(f"Completed fvcore MAC analysis for prompt {prompt_index}/{len(prompts)}.")
    finally:
        if cws_tracker is not None:
            cws_tracker.remove_hooks()
            log_step("Removed CWS hooks.")
        if aimc_runtime_tracker is not None:
            aimc_runtime_tracker.remove_hooks()
            log_step("Removed AIMC runtime hooks.")

    if cws_tracker is not None:
        log_step("Printing CWS report.")
        cws_tracker.print_report(
            model_id=model_info["model_id"],
            configured_top_k=model_info["configured_top_k"],
            original_top_k=model_info["original_top_k"],
            execution_device=model_info["execution_device"],
            prompts_processed=len(prompts),
            input_token_count=total_input_tokens,
        )
    if aimc_runtime_tracker is not None:
        log_step("Printing AIMC runtime report.")
        aimc_runtime_tracker.print_report()
    log_step("Benchmark run complete.")


if __name__ == "__main__":
    main()
