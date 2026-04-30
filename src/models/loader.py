from __future__ import annotations

from typing import Any

import torch

try:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import Cache, DynamicCache
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

DEVICE_ALIASES = {
    "gpu": "cuda",
    "cuda": "cuda:0",
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


def _resolve_device(device_name: str | None) -> str:
    """Map a YAML device string to a concrete PyTorch device name."""

    requested = (device_name or "cpu").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "auto"
        raise RuntimeError(
            "Config requested device 'auto', but CUDA is not available in this environment."
        )

    resolved = DEVICE_ALIASES.get(requested, requested)
    device = torch.device(resolved)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Config requested device '{device_name}', but CUDA is not available "
            "in this environment."
        )
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(
            f"Unsupported device '{device_name}'. Use 'cpu', 'cuda', 'cuda:N', or 'gpu'."
        )
    return str(device)


def _normalize_max_memory(max_memory: Any) -> dict[Any, str] | None:
    if max_memory is None:
        return None
    if not isinstance(max_memory, dict):
        raise TypeError("model.max_memory must be a mapping, for example {0: '20GiB', cpu: '110GiB'}.")

    normalized: dict[Any, str] = {}
    for key, value in max_memory.items():
        if isinstance(key, int):
            normalized[key] = str(value)
            continue

        key_text = str(key).lower()
        if key_text in {"cpu", "disk"}:
            normalized[key_text] = str(value)
        elif key_text in {"gpu", "cuda", "cuda:0"}:
            normalized[0] = str(value)
        else:
            normalized[key] = str(value)
    return normalized


def _first_parameter_device(model: Any) -> str:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return str(parameter.device)
    return "cpu"


def _patch_transformers_cache_compatibility() -> None:
    """
    Bridge older remote-code cache expectations onto newer Transformers cache APIs.

    Some trusted-remote-code models still expect `Cache.seen_tokens` and
    `Cache.get_max_length()`, while newer Transformers releases expose
    `get_seq_length()` and `get_max_cache_shape()` instead.
    """

    if not hasattr(Cache, "seen_tokens"):
        Cache.seen_tokens = property(lambda self: self.get_seq_length())  # type: ignore[attr-defined]

    if not hasattr(Cache, "get_max_length") and hasattr(Cache, "get_max_cache_shape"):
        def _get_max_length(self: Cache) -> int | None:
            try:
                max_length = self.get_max_cache_shape()
            except ValueError:
                return None
            return None if max_length is not None and max_length < 0 else max_length

        Cache.get_max_length = _get_max_length  # type: ignore[attr-defined]

    if not hasattr(Cache, "get_usable_length"):
        def _get_usable_length(
            self: Cache,
            new_seq_length: int,
            layer_idx: int = 0,
        ) -> int:
            max_length = self.get_max_length()
            previous_seq_length = self.get_seq_length(layer_idx)
            if max_length is not None and previous_seq_length + new_seq_length > max_length:
                return max_length - new_seq_length
            return previous_seq_length

        Cache.get_usable_length = _get_usable_length  # type: ignore[attr-defined]

    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(  # type: ignore[attr-defined]
            lambda self: self.get_seq_length()
        )

    if not hasattr(DynamicCache, "get_max_length") and hasattr(
        DynamicCache, "get_max_cache_shape"
    ):
        def _dynamic_get_max_length(self: DynamicCache) -> int | None:
            try:
                max_length = self.get_max_cache_shape()
            except ValueError:
                return None
            return None if max_length is not None and max_length < 0 else max_length

        DynamicCache.get_max_length = _dynamic_get_max_length  # type: ignore[attr-defined]

    if not hasattr(DynamicCache, "get_usable_length"):
        def _dynamic_get_usable_length(
            self: DynamicCache,
            new_seq_length: int,
            layer_idx: int = 0,
        ) -> int:
            max_length = self.get_max_length()
            previous_seq_length = self.get_seq_length(layer_idx)
            if max_length is not None and previous_seq_length + new_seq_length > max_length:
                return max_length - new_seq_length
            return previous_seq_length

        DynamicCache.get_usable_length = _dynamic_get_usable_length  # type: ignore[attr-defined]


def load_model_and_tokenizer(config_dict: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    """
    Load a Hugging Face causal language model and tokenizer for an experiment.

    The loader applies experiment-level model settings before weights are
    materialized: precision, device placement, optional max-memory/offload
    settings, safetensors preference, and MoE router top-k override. It also
    patches known Transformers cache API compatibility gaps for remote-code
    models, fills a missing tokenizer pad token from EOS when possible, switches
    the model to eval mode, and returns metadata used by tracing/reporting.
    """

    model_cfg = config_dict["model"]
    model_id = model_cfg["id"]
    top_k = int(model_cfg["top_k"])
    model_dtype = _resolve_precision(model_cfg["precision"])
    execution_device = _resolve_device(model_cfg.get("device", "cpu"))
    max_memory = _normalize_max_memory(model_cfg.get("max_memory"))
    offload_folder = model_cfg.get("offload_folder")
    prefer_safetensors = bool(model_cfg.get("prefer_safetensors", True))

    _patch_transformers_cache_compatibility()

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

    _log_step(
        f"Loading model weights with device_map={execution_device} and dtype={model_dtype} "
        f"(prefer_safetensors={prefer_safetensors})."
    )
    load_kwargs: dict[str, Any] = {
        "config": config,
        "trust_remote_code": True,
        "dtype": model_dtype,
        "device_map": execution_device if execution_device == "auto" else {"": execution_device},
        "low_cpu_mem_usage": True,
        "use_safetensors": prefer_safetensors,
    }
    if max_memory is not None:
        load_kwargs["max_memory"] = max_memory
    if offload_folder is not None:
        load_kwargs["offload_folder"] = offload_folder
        load_kwargs["offload_state_dict"] = True

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
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
        "execution_device": execution_device,
        "input_device": _first_parameter_device(model),
    }

    return model, tokenizer, model_info
