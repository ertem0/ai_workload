from __future__ import annotations

from typing import Any

import torch

try:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "Missing dependency: transformers\n"
        "Install with: python -m pip install transformers accelerate"
    ) from exc


PRECISION_MAP = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def _log_step(message: str) -> None:
    print(f"[LOADER] {message}", flush=True)


def _resolve_precision(precision_name: str) -> torch.dtype:
    """Map a YAML precision string to a PyTorch dtype."""

    normalized = precision_name.lower()
    if normalized not in PRECISION_MAP:
        raise ValueError(
            f"Unsupported precision '{precision_name}'. Supported values: {sorted(PRECISION_MAP)}"
        )
    return PRECISION_MAP[normalized]


def load_model_and_tokenizer(config_dict: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    """
    Load the Qwen MoE model and tokenizer from a parsed YAML configuration.

    Responsibilities:
    - parse model settings from the YAML dictionary
    - load the Hugging Face config first
    - override `num_experts_per_tok` before weights are materialized
    - force CPU placement in float16 to avoid Apple Silicon MPS bfloat16 issues
    """

    model_cfg = config_dict["model"]
    model_id = model_cfg["id"]
    top_k = int(model_cfg["top_k"])
    model_dtype = _resolve_precision(model_cfg["precision"])
    prefer_safetensors = bool(model_cfg.get("prefer_safetensors", True))

    _log_step(f"Loading config for {model_id}.")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if not hasattr(config, "num_experts_per_tok"):
        raise ValueError(
            f"Model config for {model_id} does not expose 'num_experts_per_tok'."
        )

    # Preserve the checkpoint default so reports can display both the original
    # and the experiment-overridden top-k values.
    config.original_num_experts_per_tok = int(config.num_experts_per_tok)

    # Override the MoE router fan-out before the checkpoint weights are loaded.
    config.num_experts_per_tok = top_k
    config.use_cache = False
    _log_step(f"Config loaded. Overriding num_experts_per_tok to {top_k}.")

    _log_step("Loading tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Force a full CPU placement. This is slower than MPS, but avoids the
    # bfloat16/offload instability that commonly appears on Apple Silicon.
    _log_step(
        f"Loading model weights on CPU with dtype={model_dtype} "
        f"(prefer_safetensors={prefer_safetensors})."
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            trust_remote_code=True,
            dtype=model_dtype,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True,
            use_safetensors=prefer_safetensors,
        )
    except Exception as exc:
        message = str(exc)
        if "safetensors" in message.lower():
            raise RuntimeError(
                f"Failed to load {model_id} with safetensors.\n"
                "This benchmark loader defaults to `use_safetensors=True` so model "
                "loading remains compatible with modern Transformers security checks.\n"
                "If the checkpoint only publishes `.bin` weights, you have two viable options:\n"
                "1. upgrade PyTorch to >= 2.6 so Transformers can load legacy pickle checkpoints safely\n"
                "2. switch to a safetensors-backed model for this benchmark\n"
                "You can override this behavior in YAML with `model.prefer_safetensors: false`, "
                "but that still requires torch>=2.6 for `.bin` checkpoints."
            ) from exc

        if "torch.load" in message and "at least v2.6" in message:
            raise RuntimeError(
                f"Failed to load {model_id} because the checkpoint requires legacy "
                "PyTorch `.bin` loading, but your installed torch is older than 2.6.\n"
                "Recent Transformers releases block this path due to CVE-2025-32434.\n"
                "Fix options:\n"
                "1. upgrade torch to >= 2.6 in this environment\n"
                "2. use a model repo that publishes `.safetensors` weights\n"
                "3. keep `prefer_safetensors: true` so unsupported repos fail fast before a large download"
            ) from exc

        raise
    model.eval()
    _log_step("Model loaded and switched to eval mode.")

    model_info = {
        "model_id": model_id,
        "configured_top_k": top_k,
        "original_top_k": int(getattr(config, "original_num_experts_per_tok", top_k)),
        "routed_experts": int(getattr(config, "num_experts", 0)),
        "execution_device": "cpu",
    }

    return model, tokenizer, model_info
