from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import torch

from src.experiment.logging import log_step, print_inference_outputs
from src.experiment.post_run_analysis import expert_routing_enabled
from src.metrics.crossbar_tiling_analyzer import (
    calculate_tiling_efficiency,
    print_tiling_report,
)
from src.metrics.expert_routing_tracker import ExpertRoutingTracker
from src.metrics.runtime_aimc_tracker import RuntimeAIMCTracker
from src.models.loader import load_model_and_tokenizer
from src.tracing.workload_trace import export_workload_trace


def synchronize_device(device_name: str) -> None:
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def set_trackers_enabled(
    expert_routing_tracker: ExpertRoutingTracker | None,
    runtime_aimc_tracker: RuntimeAIMCTracker | None,
    enabled: bool,
) -> None:
    if expert_routing_tracker is not None:
        expert_routing_tracker.set_enabled(enabled)
    if runtime_aimc_tracker is not None:
        runtime_aimc_tracker.set_enabled(enabled)


def should_use_chat_template(
    model_info: dict[str, Any],
    tokenizer: Any,
    inference_cfg: dict[str, Any],
) -> bool:
    """Use chat formatting automatically for instruction models that need it."""

    configured = inference_cfg.get("use_chat_template")
    if configured is not None:
        return bool(configured)

    model_id = str(model_info.get("model_id", "")).lower()
    chat_template_model = "gemma" in model_id or "olmoe" in model_id
    return chat_template_model and bool(getattr(tokenizer, "chat_template", None))


def encode_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    max_input_tokens: int | None,
    use_chat_template: bool,
) -> dict[str, torch.Tensor]:
    if use_chat_template:
        rendered_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return tokenizer(
            rendered_prompt,
            return_tensors="pt",
            truncation=max_input_tokens is not None,
            max_length=max_input_tokens,
            padding=False,
            add_special_tokens=False,
        )

    return tokenizer(
        prompt,
        return_tensors="pt",
        truncation=max_input_tokens is not None,
        max_length=max_input_tokens,
        padding=False,
    )


def run_experiment(config_dict: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    log_step(f"Experiment: {config_dict['experiment_name']}", phase="setup")
    log_step(f"Results directory: {output_dir.resolve()}", phase="setup")
    log_step(f"Loading model: {config_dict['model']['id']}", phase="model")

    model, tokenizer, model_info = load_model_and_tokenizer(config_dict)
    log_step("Model and tokenizer ready.", phase="model")

    expert_routing_tracker: ExpertRoutingTracker | None = None
    if expert_routing_enabled(config_dict):
        if not model_info.get("is_moe", False):
            log_step("Skipping expert routing tracker — model is not MoE.", phase="tracker")
        else:
            log_step("Initializing expert routing tracker.", phase="tracker")
            expert_routing_tracker = ExpertRoutingTracker(
                model=model,
                top_k=int(model_info["configured_top_k"]),
            )
            expert_routing_tracker.register_hooks()
            log_step(
                f"Expert routing hooks registered on "
                f"{len(expert_routing_tracker.handles)} gate modules.",
                phase="tracker",
            )

    runtime_aimc_tracker: RuntimeAIMCTracker | None = None
    aimc_metric_flags = (
        "calculate_system_arithmetic_intensity",
        "calculate_crossbar_arithmetic_intensity",
        "calculate_linear_vs_nonlinear_ratio",
        "calculate_static_vs_dynamic_ratio",
        "calculate_activation_sparsity",
    )
    aimc_report_enabled = any(
        config_dict["metrics"].get(flag, False) for flag in aimc_metric_flags
    )
    workload_trace_enabled = bool(config_dict.get("trace", {}).get("enabled", True))
    if aimc_report_enabled or workload_trace_enabled:
        log_step("Initializing dynamic AIMC metric tracker.", phase="tracker")
        runtime_aimc_tracker = RuntimeAIMCTracker(
            model=model,
            metrics_cfg=config_dict["metrics"],
        )
        runtime_aimc_tracker.register_hooks()
        log_step(
            f"AIMC runtime hooks registered: {len(runtime_aimc_tracker.handles)}.",
            phase="tracker",
        )

    if config_dict["metrics"].get("calculate_tiling_efficiency", False):
        crossbar_size = tuple(config_dict["metrics"]["crossbar_dimensions"])
        log_step(
            f"Running static tiling analysis for crossbar "
            f"{crossbar_size[0]}x{crossbar_size[1]}.",
            phase="analysis",
        )
        tiling_metrics = calculate_tiling_efficiency(model, crossbar_size)
        print_tiling_report(tiling_metrics)

    prompts = config_dict["dataset"]
    inference_cfg = config_dict.get("inference", {})
    _raw_max_new = inference_cfg.get("max_new_tokens", 64)
    max_new_tokens = None if _raw_max_new is None else int(_raw_max_new)
    _raw_max_in = inference_cfg.get("max_input_tokens", 256)
    max_input_tokens = None if _raw_max_in is None else int(_raw_max_in)
    use_chat_template = should_use_chat_template(model_info, tokenizer, inference_cfg)
    generation_eos_token_id = getattr(model.generation_config, "eos_token_id", None)
    if generation_eos_token_id is None:
        generation_eos_token_id = tokenizer.eos_token_id
    if use_chat_template:
        log_step("Using tokenizer chat template for prompt formatting.", phase="inference")
    total_input_tokens = 0
    total_output_tokens = 0
    inference_outputs: list[dict[str, str]] = []
    raw_trace_path: Path | None = None
    workload_trace_path: Path | None = None
    log_step(f"Running {len(prompts)} prompts.", phase="inference")

    try:
        synchronize_device(model_info["input_device"])
        inference_start_time = time.perf_counter()
        for prompt_index, prompt in enumerate(prompts, start=1):
            log_step(f"Tokenizing prompt {prompt_index}/{len(prompts)}.", phase="prompt")
            encoded = encode_prompt(
                tokenizer,
                prompt,
                max_input_tokens=max_input_tokens,
                use_chat_template=use_chat_template,
            )

            encoded = {
                name: tensor.to(model_info["input_device"])
                for name, tensor in encoded.items()
            }
            total_input_tokens += int(encoded["input_ids"].numel())
            if expert_routing_tracker is not None:
                expert_routing_tracker.start_prompt_trace(
                    prompt_index=prompt_index,
                    prompt_token_ids=encoded["input_ids"][0],
                )

            log_step(
                f"Starting forward pass {prompt_index}/{len(prompts)} "
                f"(sequence length {encoded['input_ids'].shape[-1]}).",
                phase="prompt",
            )

            # For expert routing, a plain forward pass is sufficient and cheaper than generation.
            if runtime_aimc_tracker is not None:
                runtime_aimc_tracker.start_prompt(
                    prompt_index=prompt_index,
                    batch=encoded,
                    phase="prefill",
                )
            with torch.inference_mode():
                model(**encoded)
            log_step(f"Completed forward pass {prompt_index}/{len(prompts)}.", phase="prompt")
            if runtime_aimc_tracker is not None:
                runtime_aimc_tracker.finish_prompt()

            if runtime_aimc_tracker is not None and aimc_report_enabled:
                # Disable forward-hook accounting during the auxiliary fvcore trace
                # so routing counts and runtime hook metrics are not double counted.
                log_step(
                    f"Starting fvcore MAC analysis for prompt "
                    f"{prompt_index}/{len(prompts)}.",
                    phase="analysis",
                )
                set_trackers_enabled(expert_routing_tracker, runtime_aimc_tracker, False)
                try:
                    runtime_aimc_tracker.analyze_flops_for_prompt(encoded)
                finally:
                    set_trackers_enabled(expert_routing_tracker, runtime_aimc_tracker, True)
                log_step(
                    f"Completed fvcore MAC analysis for prompt "
                    f"{prompt_index}/{len(prompts)}.",
                    phase="analysis",
                )

            log_step(
                f"Generating decoded output for prompt {prompt_index}/{len(prompts)} "
                f"(max_new_tokens={max_new_tokens}).",
                phase="generation",
            )
            if expert_routing_tracker is not None:
                expert_routing_tracker.start_generation_trace(
                    prompt_index=prompt_index,
                    prompt_token_ids=encoded["input_ids"][0],
                )
            if runtime_aimc_tracker is not None:
                runtime_aimc_tracker.start_prompt(
                    prompt_index=prompt_index,
                    batch=encoded,
                    phase="decode",
                )
            generate_kwargs: dict[str, Any] = dict(
                input_ids=encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
            if generation_eos_token_id is not None:
                generate_kwargs["eos_token_id"] = generation_eos_token_id
            try:
                with torch.inference_mode():
                    generated = model.generate(**generate_kwargs)
            finally:
                if runtime_aimc_tracker is not None:
                    runtime_aimc_tracker.finish_prompt()
            if expert_routing_tracker is not None:
                expert_routing_tracker.finalize_generation_trace(
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
            log_step(
                f"Captured generated output for prompt {prompt_index}/{len(prompts)}.",
                phase="generation",
            )
        synchronize_device(model_info["input_device"])
        inference_elapsed_seconds = time.perf_counter() - inference_start_time
        average_prompt_seconds = (
            inference_elapsed_seconds / len(prompts) if prompts else 0.0
        )
        log_step(
            "Completed inference for all prompts in "
            f"{inference_elapsed_seconds:.2f}s "
            f"({average_prompt_seconds:.2f}s/prompt).",
            phase="inference",
        )
    finally:
        inference_output_path = output_dir / "inference_outputs.json"
        inference_output_path.write_text(json.dumps(inference_outputs, indent=2, ensure_ascii=False))
        log_step(f"Saved inference outputs to {inference_output_path.resolve()}.", phase="trace")
        if workload_trace_enabled:
            workload_trace_path = export_workload_trace(
                output_path=output_dir / "workload_trace.pkl",
                model=model,
                model_info=model_info,
                config_dict=config_dict,
                runtime_aimc_tracker=runtime_aimc_tracker,
                expert_routing_tracker=expert_routing_tracker,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
            )
            log_step(f"Saved workload trace to {workload_trace_path.resolve()}.", phase="trace")
        if expert_routing_tracker is not None:
            raw_trace_metadata = {
                "model_id": model_info["model_id"],
                "configured_top_k": model_info["configured_top_k"],
                "original_top_k": model_info["original_top_k"],
                "execution_device": model_info["execution_device"],
                "prompts_processed": len(prompts),
                "input_token_count": total_input_tokens,
                "output_token_count": total_output_tokens,
            }
            raw_trace_path = expert_routing_tracker.export_routing_trace(
                output_dir / "expert_traces_raw.pkl",
                metadata=raw_trace_metadata,
            )
            log_step(f"Saved raw expert traces to {raw_trace_path.resolve()}.", phase="trace")
            raw_trace_json_path = expert_routing_tracker.export_routing_trace_json(
                output_dir / "expert_traces_raw.json",
                metadata=raw_trace_metadata,
            )
            log_step(
                f"Saved raw expert traces JSON to {raw_trace_json_path.resolve()}.",
                phase="trace",
            )
            expert_routing_tracker.remove_hooks()
            log_step("Removed expert routing hooks.", phase="cleanup")
        if runtime_aimc_tracker is not None:
            runtime_aimc_tracker.remove_hooks()
            log_step("Removed AIMC runtime hooks.", phase="cleanup")

    if runtime_aimc_tracker is not None and aimc_report_enabled:
        log_step("Printing AIMC runtime report.", phase="report")
        runtime_aimc_tracker.print_report()
    log_step("Benchmark run complete.", phase="summary")
    return {
        "model_info": model_info,
        "prompts_processed": len(prompts),
        "input_token_count": total_input_tokens,
        "output_token_count": total_output_tokens,
        "raw_trace_path": raw_trace_path,
        "workload_trace_path": workload_trace_path,
    }
