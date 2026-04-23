from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import re
import shutil
from typing import Any

import torch
import yaml

from src.metrics.aimc import AIMCMetricTracker
from src.metrics.cws import CWSTracker
from src.metrics.static_analyzer import calculate_tiling_efficiency, print_tiling_report
from src.models.loader import load_model_and_tokenizer


def log_step(message: str) -> None:
    print(f"[MAIN] {message}", flush=True)


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


def load_dataset_prompts(dataset_config: Any, config_path: Path) -> list[str]:
    """
    Resolve the dataset section from either an inline prompt list or a JSON file path.
    """

    if isinstance(dataset_config, list):
        prompts = dataset_config
    elif isinstance(dataset_config, str):
        dataset_path = Path(dataset_config)
        if not dataset_path.is_absolute():
            dataset_path = (config_path.parent / dataset_path).resolve()

        with dataset_path.open("r", encoding="utf-8") as handle:
            prompts = json.load(handle)
    else:
        raise TypeError(
            "Config field 'dataset' must be either a list of prompt strings or a JSON file path."
        )

    if not isinstance(prompts, list) or not all(isinstance(prompt, str) for prompt in prompts):
        raise ValueError(
            "Resolved dataset must be a JSON/YAML list containing only prompt strings."
        )

    return prompts


def set_trackers_enabled(
    cws_tracker: CWSTracker | None,
    aimc_runtime_tracker: AIMCMetricTracker | None,
    enabled: bool,
) -> None:
    if cws_tracker is not None:
        cws_tracker.set_enabled(enabled)
    if aimc_runtime_tracker is not None:
        aimc_runtime_tracker.set_enabled(enabled)


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


def run_experiment(config_dict: dict[str, Any], output_dir: Path) -> None:
    log_step(f"Experiment: {config_dict['experiment_name']}")
    log_step(f"Results directory: {output_dir.resolve()}")
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
    inference_cfg = config_dict.get("inference", {})
    max_new_tokens = int(inference_cfg.get("max_new_tokens", 64))
    total_input_tokens = 0
    total_output_tokens = 0
    inference_outputs: list[dict[str, str]] = []
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
            if cws_tracker is not None:
                cws_tracker.start_prompt_trace(
                    prompt_index=prompt_index,
                    prompt_token_ids=encoded["input_ids"][0],
                )

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
                set_trackers_enabled(cws_tracker, aimc_runtime_tracker, False)
                aimc_runtime_tracker.analyze_flops_for_prompt(encoded)
                set_trackers_enabled(cws_tracker, aimc_runtime_tracker, True)
                log_step(f"Completed fvcore MAC analysis for prompt {prompt_index}/{len(prompts)}.")

            log_step(
                f"Generating decoded output for prompt {prompt_index}/{len(prompts)} "
                f"(max_new_tokens={max_new_tokens})."
            )
            if cws_tracker is not None:
                cws_tracker.start_generation_trace(
                    prompt_index=prompt_index,
                    prompt_token_ids=encoded["input_ids"][0],
                )
            try:
                if aimc_runtime_tracker is not None:
                    aimc_runtime_tracker.set_enabled(False)
                with torch.inference_mode():
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
            finally:
                if aimc_runtime_tracker is not None:
                    aimc_runtime_tracker.set_enabled(True)
            if cws_tracker is not None:
                cws_tracker.finalize_generation_trace(
                    prompt_index=prompt_index,
                    generated_token_ids=generated[0],
                )

            input_length = int(encoded["input_ids"].shape[-1])
            total_output_tokens += int(generated[0][input_length:].shape[-1])
            decoded_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            continuation = tokenizer.decode(
                generated[0][input_length:],
                skip_special_tokens=True,
            ).strip()
            inference_outputs.append(
                {
                    "prompt_index": str(prompt_index),
                    "prompt": prompt,
                    "decoded_text": decoded_text.strip(),
                    "continuation": continuation,
                }
            )
            log_step(f"Captured generated output for prompt {prompt_index}/{len(prompts)}.")
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
            output_token_count=total_output_tokens,
            output_dir=output_dir,
        )
    if aimc_runtime_tracker is not None:
        log_step("Printing AIMC runtime report.")
        aimc_runtime_tracker.print_report()
    log_step("Printing inference outputs.")
    print_inference_outputs(inference_outputs)
    log_step("Benchmark run complete.")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    log_step(f"Reading config from {config_path}.")
    config_dict = load_yaml_config(config_path)
    config_dict["dataset"] = load_dataset_prompts(config_dict["dataset"], config_path)
    output_dir = make_experiment_output_dir(config_dict["experiment_name"])
    reset_experiment_output_dir(output_dir)
    log_path = output_dir / "run.log"

    with log_path.open("w", encoding="utf-8") as log_handle:
        with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
            log_step(f"Writing combined logs to {log_path.resolve()}")
            run_experiment(config_dict, output_dir)


if __name__ == "__main__":
    main()
