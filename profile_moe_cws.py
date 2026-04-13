#!/usr/bin/env python3
"""
Profile Crossbar Workload Skewness (CWS) for Qwen/Qwen1.5-MoE-A2.7B.

This script is written for a local macOS setup and avoids CUDA-only quantization
paths such as bitsandbytes. It loads the model in float16, attaches forward
hooks to the Qwen MoE router modules, tallies expert selections during one
forward pass, and reports the coefficient of variation of expert accesses.

Important architectural note:
- The official Qwen1.5-MoE-A2.7B config exposes `num_experts=60`, not 64.
- The model also contains a separate shared expert, which is not part of the
  routed top-k expert pool and is therefore not counted in the CWS tally.
"""

from __future__ import annotations

import logging
import platform
import re
from typing import Any

import torch
import torch.nn as nn

try:
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover - import guard for local runtime
    raise SystemExit(
        "Missing dependency: transformers\n"
        "Install with: python -m pip install transformers accelerate"
    ) from exc


# -----------------------------
# User-tunable experiment knobs
# -----------------------------
MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B"
TOP_K = 2
MAX_INPUT_LENGTH = 256

PROMPT_SUITE = [
    (
        "A water tank initially contains 320 liters. A maintenance team drains 15 percent of the "
        "water, then adds 48 liters, and finally splits the remaining water equally into 8 storage "
        "containers. How many liters end up in each container, and what is the full reasoning?"
    ),
    (
        "Write a Python function that takes a list of dictionaries representing student records and "
        "returns the top three students by average grade. Include type hints, edge case handling, "
        "and a short unit test example."
    ),
    (
        "La Revolution francaise a transforme durablement la vie politique europeenne. Expliquez en "
        "quelques phrases les causes principales, le role de la prise de la Bastille et les "
        "consequences institutionnelles les plus importantes."
    ),
    (
        "Explain how the Roman Empire managed frontier defense during the second century CE, including "
        "the role of roads, forts, and provincial administration in maintaining military control."
    ),
    (
        "Hey there, I just got back from work and I am exhausted. Can you chat with me for a minute "
        "and suggest a simple relaxing evening routine that does not require much effort?"
    ),
]


def configure_logging() -> None:
    logging.getLogger("transformers").setLevel(logging.ERROR)


def log_step(message: str) -> None:
    print(f"[CWS] {message}", flush=True)


def is_qwen_router_module(module_name: str, module: nn.Module, num_experts: int) -> bool:
    """
    Detect likely Qwen MoE router modules without hardcoding an exact class name.

    We look for modules whose leaf name is exactly `gate`, because that is the
    conventional router name inside Qwen sparse MoE blocks. We allow either:
    - a dedicated router module that owns `top_k` / `num_experts`
    - a plain Linear layer that emits expert logits
    """

    leaf_name = module_name.rsplit(".", maxsplit=1)[-1]
    if leaf_name != "gate":
        return False
    if isinstance(module, nn.Linear) and module.out_features == num_experts:
        return True
    required_attrs = ("top_k", "num_experts", "weight")
    return all(hasattr(module, attr) for attr in required_attrs)


def find_qwen_router_modules(
    model: nn.Module, num_experts: int
) -> list[tuple[str, nn.Module]]:
    """
    Recursively scan the model and return every likely Qwen router `gate`.
    """

    routers: list[tuple[str, nn.Module]] = []
    for module_name, module in model.named_modules():
        if is_qwen_router_module(module_name, module, num_experts):
            routers.append((module_name, module))
    return routers


def extract_layer_identifier(module_name: str) -> str:
    """
    Normalize a gate module name into a stable layer identifier.

    Expected Qwen paths typically look like `model.layers.5.mlp.gate`. If the
    exact naming differs, fall back to the parent path around the gate.
    """

    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", module_name)
    if match is not None:
        return f"layer {int(match.group(1))}"

    parent_name = module_name.rsplit(".gate", maxsplit=1)[0]
    return parent_name


def list_gate_like_modules(model: nn.Module) -> list[str]:
    return [
        f"{module_name}: {module.__class__.__name__}"
        for module_name, module in model.named_modules()
        if "gate" in module_name.lower()
    ]


def get_first_transformer_block(model: nn.Module) -> tuple[str, nn.Module] | None:
    """
    Find the first major transformer block using common container conventions.
    """

    candidate_paths = (
        "model.layers.0",
        "layers.0",
        "transformer.h.0",
        "h.0",
        "encoder.layer.0",
        "decoder.layers.0",
        "blocks.0",
    )
    modules = dict(model.named_modules())
    for path in candidate_paths:
        if path in modules:
            return path, modules[path]
    return None


def describe_module_tree(module: nn.Module, root_name: str, max_depth: int = 3) -> str:
    """
    Produce a compact text tree for debugging the first transformer block.
    """

    lines: list[str] = [f"{root_name}: {module.__class__.__name__}"]

    def visit(current_name: str, current_module: nn.Module, depth: int) -> None:
        if depth >= max_depth:
            return
        for child_name, child in current_module.named_children():
            full_name = f"{current_name}.{child_name}"
            indent = "  " * (depth + 1)
            lines.append(f"{indent}{full_name}: {child.__class__.__name__}")
            visit(full_name, child, depth + 1)

    visit(root_name, module, depth=0)
    return "\n".join(lines)


def extract_selected_experts(
    output: Any,
    top_k: int,
    num_experts: int,
) -> torch.Tensor | None:
    """
    Extract selected expert ids from either:
    - a router module returning (..., router_indices)
    - a Linear gate returning raw expert logits
    """

    if isinstance(output, tuple):
        for item in reversed(output):
            if not isinstance(item, torch.Tensor):
                continue
            if item.dtype in (torch.int32, torch.int64) and item.shape[-1] == top_k:
                return item
        for item in output:
            if not isinstance(item, torch.Tensor):
                continue
            if item.shape[-1] == num_experts:
                probabilities = torch.softmax(item.float(), dim=-1)
                return torch.topk(probabilities, k=top_k, dim=-1).indices
        return None

    if isinstance(output, torch.Tensor):
        if output.dtype in (torch.int32, torch.int64) and output.shape[-1] == top_k:
            return output
        if output.shape[-1] == num_experts:
            probabilities = torch.softmax(output.float(), dim=-1)
            return torch.topk(probabilities, k=top_k, dim=-1).indices

    return None


def format_expert_usage(counts: torch.Tensor) -> str:
    usage = {expert_id: int(count) for expert_id, count in enumerate(counts.tolist())}
    return str(usage)


def top_k_experts(counts: torch.Tensor, k: int = 3) -> list[tuple[int, int]]:
    """
    Return the top-k most used experts as (expert_id, count) pairs.
    """

    if counts.numel() == 0:
        return []
    top_count = min(k, counts.numel())
    values, indices = torch.topk(counts, k=top_count)
    return [(int(index), int(value)) for value, index in zip(values.tolist(), indices.tolist())]


def choose_runtime_device() -> tuple[str, dict[str, str] | str]:
    """
    Choose a safe execution/load strategy for the local machine.

    On Apple Silicon, `device_map="auto"` can trigger Accelerate offload paths
    that try to materialize checkpoint tensors with their original bfloat16 dtype
    on MPS, which fails because MPS does not support bfloat16 robustly in that
    loader path. For this profiling script, reliability matters more than peak
    speed, so we force a full CPU placement on macOS.
    """

    is_macos = platform.system() == "Darwin"
    if is_macos:
        return "cpu", {"": "cpu"}
    if torch.cuda.is_available():
        return "cuda", "auto"
    return "cpu", {"": "cpu"}


def main() -> None:
    configure_logging()
    log_step("Starting Qwen MoE CWS profiling run.")

    # Load the config first so the routed top-k can be overridden before any
    # model weights are instantiated.
    log_step(f"Loading config for {MODEL_ID}.")
    try:
        config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    except Exception as exc:
        raise SystemExit(
            "Failed to load the Qwen MoE config.\n"
            f"Installed transformers version: {transformers.__version__}\n"
            "This checkpoint uses model type `qwen2_moe`, which requires a newer "
            "Transformers release with Qwen2-MoE support.\n"
            "Upgrade with: python -m pip install -U 'transformers>=4.40,<5' accelerate\n"
            f"Original error: {exc}"
        ) from exc
    if not hasattr(config, "num_experts_per_tok"):
        raise SystemExit(
            f"{MODEL_ID} config does not expose `num_experts_per_tok`; cannot override top-k."
        )

    original_top_k = int(config.num_experts_per_tok)
    config.num_experts_per_tok = TOP_K
    config.use_cache = False
    log_step(
        f"Config loaded. Overriding num_experts_per_tok from {original_top_k} to {TOP_K}."
    )

    # Qwen1.5-MoE-A2.7B officially exposes 60 routed experts. The shared expert is
    # separate and not part of the routed expert histogram.
    num_routed_experts = int(getattr(config, "num_experts", 0))
    if num_routed_experts <= 0:
        raise SystemExit(f"{MODEL_ID} did not report a valid `num_experts` value in config.")
    log_step(f"Detected {num_routed_experts} routed experts in config.")

    log_step("Loading tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    runtime_device, device_map = choose_runtime_device()
    log_step(f"Runtime placement selected: {runtime_device} with device_map={device_map}.")

    log_step("Loading model weights. This can take a while.")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()
    log_step("Model loaded and switched to eval mode.")

    routers = find_qwen_router_modules(model, num_routed_experts)
    if not routers:
        block = get_first_transformer_block(model)
        if block is not None:
            block_name, block_module = block
            print("No Qwen MoE router modules were found.")
            print("First transformer block architecture:")
            print(describe_module_tree(block_module, block_name))
        gate_like_modules = list_gate_like_modules(model)
        if gate_like_modules:
            print("Gate-like modules found:")
            for entry in gate_like_modules:
                print(f"  {entry}")
        raise SystemExit(
            "No Qwen MoE router modules were found. The model implementation may have changed."
        )
    log_step(f"Discovered {len(routers)} router gate modules.")

    layer_names_by_router = {
        router_name: extract_layer_identifier(router_name) for router_name, _ in routers
    }
    ordered_layer_names = sorted(
        set(layer_names_by_router.values()),
        key=lambda name: int(name.split()[-1]) if name.startswith("layer ") else name,
    )

    # Each MoE layer gets its own routed-expert histogram. This is the data
    # structure used for layer-local CWS rather than a single global tally.
    per_layer_counts: dict[str, torch.Tensor] = {
        layer_name: torch.zeros(num_routed_experts, dtype=torch.long)
        for layer_name in ordered_layer_names
    }

    hook_handles: list[Any] = []

    def make_router_hook(router_name: str, layer_name: str, router_module: nn.Module):
        def hook(_: nn.Module, __: tuple[Any, ...], output: Any) -> None:
            router_top_k = int(getattr(router_module, "top_k", TOP_K))
            selected_experts = extract_selected_experts(
                output=output,
                top_k=router_top_k,
                num_experts=num_routed_experts,
            )
            if selected_experts is None:
                return

            flat_indices = selected_experts.reshape(-1)
            router_counts = torch.bincount(
                flat_indices,
                minlength=num_routed_experts,
            ).cpu()

            per_layer_counts[layer_name].add_(router_counts)

        return hook

    for router_name, router_module in routers:
        hook_handles.append(
            router_module.register_forward_hook(
                make_router_hook(
                    router_name,
                    layer_names_by_router[router_name],
                    router_module,
                )
            )
        )
    log_step("Registered forward hooks on all discovered router gates.")

    total_tokens = 0
    log_step(f"Processing workload suite with {len(PROMPT_SUITE)} prompts.")
    for prompt_index, prompt in enumerate(PROMPT_SUITE, start=1):
        log_step(f"Tokenizing prompt {prompt_index}/{len(PROMPT_SUITE)}.")
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
            padding=False,
        )

        # Inputs should start on the embedding device. On macOS we intentionally keep
        # the full model on CPU to avoid MPS + bfloat16 offload failures.
        if runtime_device == "cpu":
            encoded = {name: tensor.to("cpu") for name, tensor in encoded.items()}
        else:
            input_device = model.get_input_embeddings().weight.device
            encoded = {name: tensor.to(input_device) for name, tensor in encoded.items()}

        prompt_tokens = int(encoded["input_ids"].shape[-1])
        total_tokens += prompt_tokens
        log_step(
            f"Starting forward pass for prompt {prompt_index}/{len(PROMPT_SUITE)} "
            f"(sequence length {prompt_tokens})."
        )
        with torch.inference_mode():
            _ = model(**encoded)
        log_step(f"Completed prompt {prompt_index}/{len(PROMPT_SUITE)}.")

    for handle in hook_handles:
        handle.remove()
    log_step("Removed all router hooks.")

    layer_results: list[dict[str, Any]] = []
    for layer_name in ordered_layer_names:
        counts = per_layer_counts[layer_name].to(torch.float64)
        mean_count = float(counts.mean().item())
        std_count = float(counts.std(unbiased=False).item())
        cws = float(std_count / mean_count) if mean_count > 0 else float("inf")
        layer_results.append(
            {
                "layer_name": layer_name,
                "counts": per_layer_counts[layer_name],
                "cws": cws,
                "top_experts": top_k_experts(per_layer_counts[layer_name], k=3),
                "total_assignments": int(per_layer_counts[layer_name].sum().item()),
            }
        )
    log_step("Computed per-layer expert histograms and CWS metrics.")

    total_assignments = sum(result["total_assignments"] for result in layer_results)
    average_cws = (
        sum(result["cws"] for result in layer_results) / len(layer_results)
        if layer_results
        else float("nan")
    )

    print("=" * 96)
    print("Qwen1.5-MoE Per-Layer Crossbar Workload Skewness (CWS) Report")
    print("=" * 96)
    print(f"Model                         : {MODEL_ID}")
    print(f"Configured top-k             : {TOP_K}")
    print(f"Original config top-k        : {original_top_k}")
    print(f"Routed experts counted       : {num_routed_experts}")
    print(f"MoE routers hooked           : {len(routers)}")
    print(f"Execution device             : {runtime_device}")
    print(f"Prompts processed            : {len(PROMPT_SUITE)}")
    print(f"Input token count            : {total_tokens}")
    print(f"Total expert assignments     : {total_assignments}")
    print("-" * 96)
    print("Per-layer results")
    for result in layer_results:
        top_summary = ", ".join(
            f"expert {expert_id}: {count}" for expert_id, count in result["top_experts"]
        )
        print(
            f"{result['layer_name']:<30} "
            f"top-3=[{top_summary}] "
            f"CWS={result['cws']:.6f}"
        )
    print("-" * 96)
    print(f"Average layer CWS           : {average_cws:.6f}")
    print("=" * 96)


if __name__ == "__main__":
    main()
