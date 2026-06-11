from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import re
import time
from typing import Any

import torch
import torch.nn as nn

from src.experiment.logging import log_step, print_inference_outputs
from src.experiment.post_run_analysis import expert_routing_enabled, small_expert_routing_enabled
from src.metrics.crossbar_tiling_analyzer import (
    calculate_tiling_efficiency,
    print_tiling_report,
)
from src.metrics.expert_routing_tracker import ExpertRoutingTracker
from src.metrics.runtime_aimc_tracker import RuntimeAIMCTracker
from src.metrics.small_expert_routing_tracker import SmallExpertRoutingTracker
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
    small_expert_routing_tracker: SmallExpertRoutingTracker | None = None,
) -> None:
    if expert_routing_tracker is not None:
        expert_routing_tracker.set_enabled(enabled)
    if runtime_aimc_tracker is not None:
        runtime_aimc_tracker.set_enabled(enabled)
    if small_expert_routing_tracker is not None:
        small_expert_routing_tracker.set_enabled(enabled)


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


def extract_profiler_layer_id(module_name: str) -> int | None:
    match = re.search(r"(?:^|\.)(?:decoder\.)?layers\.(\d+)(?:\.|$)", module_name)
    if match is None:
        match = re.search(r"(?:^|\.)blocks\.(\d+)(?:\.|$)", module_name)
    if match is None:
        match = re.search(r"(?:^|\.)h\.(\d+)(?:\.|$)", module_name)
    return int(match.group(1)) if match else None


def is_full_layer_module(module_name: str, layer_id: int) -> bool:
    return bool(
        re.search(rf"(?:^|\.)(?:decoder\.)?layers\.{layer_id}$", module_name)
        or re.search(rf"(?:^|\.)blocks\.{layer_id}$", module_name)
        or re.search(rf"(?:^|\.)h\.{layer_id}$", module_name)
    )


def profiler_module_label(module_name: str, module: nn.Module) -> str:
    module_type = module.__class__.__name__
    leaf_name = module_name.rsplit(".", 1)[-1].lower() if module_name else ""
    layer_id = extract_profiler_layer_id(module_name)

    if not module_name:
        return f"model ({module_type})"

    if module_name == "lm_head" or leaf_name == "lm_head":
        return f"lm_head ({module_type})"

    if isinstance(module, nn.Embedding) or leaf_name in {"embed_tokens", "wte", "embedding"}:
        return f"embedding ({module_type})"

    if layer_id is not None and is_full_layer_module(module_name, layer_id):
        return f"layer{layer_id} ({module_type})"

    expert_match = re.search(r"(?:^|\.)(?:experts?|mlp)\.(\d+)(?:\.|$)", module_name)
    if layer_id is not None and expert_match is not None:
        return f"layer{layer_id}.expert{int(expert_match.group(1))} ({module_type})"

    if layer_id is not None and leaf_name in {"experts", "expert"}:
        return f"layer{layer_id}.experts ({module_type})"

    if layer_id is not None and (
        leaf_name in {"self_attn", "attention", "attn"} or "attention" in module_type.lower()
    ):
        return f"layer{layer_id}.attention ({module_type})"

    if layer_id is not None and (
        leaf_name in {"router", "gate"} or "router" in module_type.lower()
    ):
        return f"layer{layer_id}.router ({module_type})"

    if layer_id is not None and (
        "norm" in leaf_name or "rmsnorm" in module_type.lower() or "layernorm" in module_type.lower()
    ):
        norm_suffix = leaf_name if leaf_name and leaf_name != "norm" else "norm"
        return f"layer{layer_id}.norm.{norm_suffix} ({module_type})"

    return f"{module_name} ({module_type})"


class ModuleProfilerHooks:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.handles: list[Any] = []
        self.active_contexts: dict[int, list[Any]] = {}

    def register(self) -> None:
        for module_name, module in self.model.named_modules():
            label = profiler_module_label(module_name, module)
            module_id = id(module)

            def pre_hook(
                current_module: nn.Module,
                _inputs: tuple[Any, ...],
                *,
                current_label: str = label,
                current_module_id: int = module_id,
            ) -> None:
                context = torch.profiler.record_function(current_label)
                context.__enter__()
                self.active_contexts.setdefault(current_module_id, []).append(context)

            def post_hook(
                current_module: nn.Module,
                _inputs: tuple[Any, ...],
                _output: Any,
                *,
                current_module_id: int = module_id,
            ) -> None:
                contexts = self.active_contexts.get(current_module_id)
                if not contexts:
                    return
                context = contexts.pop()
                context.__exit__(None, None, None)

            self.handles.append(module.register_forward_pre_hook(pre_hook))
            self.handles.append(module.register_forward_hook(post_hook))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for contexts in self.active_contexts.values():
            while contexts:
                contexts.pop().__exit__(None, None, None)
        self.active_contexts.clear()


def build_profiler_context(enabled: bool) -> Any:
    if not enabled:
        return nullcontext(None)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    return torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=True,
        acc_events=True,
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
                f"{expert_routing_tracker.router_hook_count} gate modules.",
                phase="tracker",
            )

    small_expert_routing_tracker: SmallExpertRoutingTracker | None = None
    if small_expert_routing_enabled(config_dict):
        if not model_info.get("is_moe", False):
            log_step(
                "Skipping small expert routing tracker — model is not MoE.",
                phase="tracker",
            )
        else:
            log_step("Initializing small expert routing tracker.", phase="tracker")
            small_expert_routing_tracker = SmallExpertRoutingTracker(
                model=model,
                top_k=int(model_info["configured_top_k"]),
            )
            small_expert_routing_tracker.register_hooks()
            log_step(
                f"Small expert routing hooks registered on "
                f"{small_expert_routing_tracker.router_hook_count} gate modules.",
                phase="tracker",
            )
            small_expert_trace_path = output_dir / "expert_routing_compact.jsonl"
            small_expert_routing_tracker.open_output(
                small_expert_trace_path,
                metadata={
                    "model_id": model_info["model_id"],
                    "configured_top_k": model_info["configured_top_k"],
                    "original_top_k": model_info["original_top_k"],
                    "num_routed_experts": small_expert_routing_tracker.num_routed_experts,
                    "execution_device": model_info["execution_device"],
                    "prompts_total": len(config_dict["dataset"]),
                },
            )
            log_step(
                f"Small expert trace output opened at {small_expert_trace_path.resolve()}.",
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
    trace_cfg = config_dict.get("trace", {})
    workload_trace_enabled = bool(trace_cfg.get("enabled", True))
    inference_outputs_enabled = bool(trace_cfg.get("save_inference_outputs", True))
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
    profiler_cfg = config_dict.get("profiler", {})
    # Full PyTorch profiling is expensive, so it is opt-in via `profiler.enabled: true`.
    profiler_enabled = bool(profiler_cfg.get("enabled", False))
    total_input_tokens = 0
    total_output_tokens = 0
    inference_outputs: list[dict[str, str]] = []
    raw_trace_path: Path | None = None
    workload_trace_path: Path | None = None
    profiler_trace_path: Path | None = None
    small_expert_trace_path: Path | None = None
    profiler: Any | None = None
    profiler_hooks: ModuleProfilerHooks | None = None
    log_step(f"Running {len(prompts)} prompts.", phase="inference")
    if profiler_enabled:
        if prompts:
            log_step(
                "Running unprofiled warmup on the first prompt "
                "(max_new_tokens=1) to reduce cold-start effects.",
                phase="profiler",
            )
            warmup_start = time.perf_counter()
            set_trackers_enabled(expert_routing_tracker, runtime_aimc_tracker, False, small_expert_routing_tracker)
            try:
                warmup_encoded = encode_prompt(
                    tokenizer,
                    prompts[0],
                    max_input_tokens=max_input_tokens,
                    use_chat_template=use_chat_template,
                )
                warmup_encoded = {
                    name: tensor.to(model_info["input_device"])
                    for name, tensor in warmup_encoded.items()
                }
                warmup_generate_kwargs: dict[str, Any] = dict(
                    input_ids=warmup_encoded["input_ids"],
                    attention_mask=warmup_encoded.get("attention_mask"),
                    max_new_tokens=1,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
                if generation_eos_token_id is not None:
                    warmup_generate_kwargs["eos_token_id"] = generation_eos_token_id

                synchronize_device(model_info["input_device"])
                with torch.inference_mode():
                    model(**warmup_encoded)
                    model.generate(**warmup_generate_kwargs)
                synchronize_device(model_info["input_device"])
            finally:
                set_trackers_enabled(expert_routing_tracker, runtime_aimc_tracker, True, small_expert_routing_tracker)
                if "warmup_encoded" in locals():
                    del warmup_encoded
                if "warmup_generate_kwargs" in locals():
                    del warmup_generate_kwargs
            log_step(
                f"Completed profiler warmup in {time.perf_counter() - warmup_start:.2f}s.",
                phase="profiler",
            )
        profiler_hooks = ModuleProfilerHooks(model)
        profiler_hooks.register()
        log_step(
            f"PyTorch profiler enabled with module hooks: "
            f"{len(profiler_hooks.handles) // 2} modules labeled.",
            phase="profiler",
        )
        log_step(
            "Residual additions are inline tensor ops in the model forward code, "
            "so they appear as aten::add/aten::add_ unless the installed model "
            "implementation is patched directly.",
            phase="profiler",
        )
        log_step(
            "Batched expert implementations without per-expert submodules are labeled "
            "at the experts module level; models exposing individual expert modules "
            "are labeled as layer{n}.expert{id}.",
            phase="profiler",
        )

    try:
        with build_profiler_context(profiler_enabled) as active_profiler:
            profiler = active_profiler
            synchronize_device(model_info["input_device"])
            inference_start_time = time.perf_counter()
            for prompt_index, prompt in enumerate(prompts, start=1):
                log_step(f"Tokenizing prompt {prompt_index}/{len(prompts)}.", phase="prompt")
                with (
                    torch.profiler.record_function("tokenize_prompt")
                    if profiler_enabled
                    else nullcontext()
                ):
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
                if small_expert_routing_tracker is not None:
                    small_expert_routing_tracker.start_generation_trace(
                        prompt_index=prompt_index,
                    )
                if runtime_aimc_tracker is not None:
                    runtime_aimc_tracker.start_prompt(
                        prompt_index=prompt_index,
                        batch=encoded,
                        phase="generate",
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
                    with (
                        torch.profiler.record_function("generation")
                        if profiler_enabled
                        else nullcontext()
                    ):
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
                if small_expert_routing_tracker is not None:
                    small_expert_routing_tracker.finalize_generation_trace(
                        prompt_index=prompt_index,
                    )
                    small_expert_routing_tracker.flush_prompt(
                        output_dir / "expert_routing_compact.jsonl"
                    )

                input_length = int(encoded["input_ids"].shape[-1])
                total_output_tokens += int(generated[0][input_length:].shape[-1])
                if inference_outputs_enabled:
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

        if runtime_aimc_tracker is not None and aimc_report_enabled:
            for prompt_index, prompt in enumerate(prompts, start=1):
                log_step(
                    f"Starting fvcore MAC analysis for prompt "
                    f"{prompt_index}/{len(prompts)}.",
                    phase="analysis",
                )
                set_trackers_enabled(expert_routing_tracker, runtime_aimc_tracker, False, small_expert_routing_tracker)
                try:
                    analysis_encoded = encode_prompt(
                        tokenizer,
                        prompt,
                        max_input_tokens=max_input_tokens,
                        use_chat_template=use_chat_template,
                    )
                    analysis_encoded = {
                        name: tensor.to(model_info["input_device"])
                        for name, tensor in analysis_encoded.items()
                    }
                    runtime_aimc_tracker.analyze_flops_for_prompt(analysis_encoded)
                finally:
                    set_trackers_enabled(expert_routing_tracker, runtime_aimc_tracker, True, small_expert_routing_tracker)
                    if "analysis_encoded" in locals():
                        del analysis_encoded
                log_step(
                    f"Completed fvcore MAC analysis for prompt "
                    f"{prompt_index}/{len(prompts)}.",
                    phase="analysis",
                )
    finally:
        if profiler_hooks is not None:
            profiler_hooks.remove()
            log_step("Removed PyTorch profiler module hooks.", phase="cleanup")
        if profiler is not None:
            profiler_trace_path = output_dir / "inference_trace.json"
            log_step(
                f"Exporting PyTorch profiler Chrome trace to {profiler_trace_path.resolve()}.",
                phase="profiler",
            )
            profiler_export_start = time.perf_counter()
            profiler.export_chrome_trace(str(profiler_trace_path))
            log_step(
                "Saved PyTorch profiler Chrome trace to "
                f"{profiler_trace_path.resolve()} "
                f"in {time.perf_counter() - profiler_export_start:.2f}s.",
                phase="profiler",
            )
            sort_key = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
            log_step(
                f"Aggregating PyTorch profiler key averages by {sort_key}.",
                phase="profiler",
            )
            profiler_summary_start = time.perf_counter()
            profiler_summary_table = profiler.key_averages().table(
                sort_by=sort_key,
                row_limit=30,
            )
            log_step(
                "Finished profiler key averages aggregation in "
                f"{time.perf_counter() - profiler_summary_start:.2f}s. "
                f"Top operations by {sort_key}:",
                phase="profiler",
            )
            print(profiler_summary_table)
        if inference_outputs_enabled:
            inference_output_path = output_dir / "inference_outputs.json"
            log_step(f"Writing inference outputs to {inference_output_path.resolve()}.", phase="trace")
            inference_output_start = time.perf_counter()
            inference_output_path.write_text(json.dumps(inference_outputs, indent=2, ensure_ascii=False))
            log_step(
                f"Saved inference outputs to {inference_output_path.resolve()} "
                f"in {time.perf_counter() - inference_output_start:.2f}s.",
                phase="trace",
            )
        if workload_trace_enabled:
            workload_trace_start = time.perf_counter()
            log_step(
                f"Exporting workload trace to {(output_dir / 'workload_trace.pkl').resolve()}.",
                phase="trace",
            )
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
            log_step(
                f"Saved workload trace to {workload_trace_path.resolve()} "
                f"in {time.perf_counter() - workload_trace_start:.2f}s.",
                phase="trace",
            )
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
            raw_trace_start = time.perf_counter()
            log_step(
                f"Exporting raw expert trace pickle to "
                f"{(output_dir / 'expert_traces_raw.pkl').resolve()}.",
                phase="trace",
            )
            raw_trace_path = expert_routing_tracker.export_routing_trace(
                output_dir / "expert_traces_raw.pkl",
                metadata=raw_trace_metadata,
            )
            log_step(
                f"Saved raw expert traces to {raw_trace_path.resolve()} "
                f"in {time.perf_counter() - raw_trace_start:.2f}s.",
                phase="trace",
            )
            raw_trace_json_start = time.perf_counter()
            log_step(
                f"Exporting raw expert trace JSON to "
                f"{(output_dir / 'expert_traces_raw.json').resolve()}.",
                phase="trace",
            )
            raw_trace_json_path = expert_routing_tracker.export_routing_trace_json(
                output_dir / "expert_traces_raw.json",
                metadata=raw_trace_metadata,
            )
            log_step(
                f"Saved raw expert traces JSON to {raw_trace_json_path.resolve()} "
                f"in {time.perf_counter() - raw_trace_json_start:.2f}s.",
                phase="trace",
            )
            expert_routing_tracker.remove_hooks()
            log_step("Removed expert routing hooks.", phase="cleanup")
        if small_expert_routing_tracker is not None:
            small_expert_routing_tracker.remove_hooks()
            small_expert_routing_tracker.close()
            log_step(
                f"Removed small expert routing hooks "
                f"({small_expert_routing_tracker.prompts_flushed} prompts flushed).",
                phase="cleanup",
            )
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
        "profiler_trace_path": profiler_trace_path,
        "small_expert_trace_path": small_expert_trace_path,
    }
