#!/usr/bin/env python3
"""
Inject AIMC-style analog noise into Qwen/Qwen1.5-MoE-A2.7B linear layers.

This script simulates Matrix-Vector Multiplication (MVM) noise by attaching a
forward hook to every `nn.Linear` module. Each hook perturbs the layer output
with zero-mean Gaussian noise whose standard deviation is proportional to the
observed output standard deviation:

    sigma_noise = noise_factor * std(output)

The same model is evaluated multiple times in one process:
- clean baseline (`noise_factor = 0.0`)
- increasingly noisy analog settings (for example 5%, 10%, 20%)

The goal is to visually inspect when generated text starts to degrade under
crossbar-like compute noise.
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass, field
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
PROMPT = "Explain the theory of relativity in one sentence."
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20]
MAX_INPUT_LENGTH = 128
MAX_NEW_TOKENS = 48
TEMPERATURE = 0.0
SEED = 7


def configure_logging() -> None:
    logging.getLogger("transformers").setLevel(logging.ERROR)


def log_step(message: str) -> None:
    print(f"[AIMC] {message}", flush=True)


def choose_runtime_device() -> tuple[str, dict[str, str] | str]:
    """
    On macOS, keep the full model on CPU.

    This avoids MPS + offload interactions that are brittle for large checkpoints
    and is consistent with the user's requirement to run on CPU in float16.
    """

    if platform.system() == "Darwin":
        return "cpu", {"": "cpu"}
    if torch.cuda.is_available():
        return "cuda", "auto"
    return "cpu", {"": "cpu"}


@dataclass
class NoiseController:
    """
    Mutable state shared by all forward hooks.

    The hooks remain attached for the entire process lifetime. We simply toggle
    `enabled` and update `noise_factor` before each generation run.
    """

    enabled: bool = False
    noise_factor: float = 0.0
    touched_layers: int = 0
    noise_events: int = 0
    total_noise_std: float = 0.0
    layer_names: set[str] = field(default_factory=set)

    def set_noise_factor(self, value: float) -> None:
        self.noise_factor = float(value)
        self.enabled = self.noise_factor > 0.0
        self.reset_counters()

    def reset_counters(self) -> None:
        self.touched_layers = 0
        self.noise_events = 0
        self.total_noise_std = 0.0
        self.layer_names.clear()


def create_noise_hook(controller: NoiseController, layer_name: str):
    """
    Create a forward hook for one linear layer.

    The hook intercepts the layer output and returns a perturbed tensor. In
    PyTorch, returning a tensor from a forward hook replaces the original output,
    which is exactly what we want for simulating analog MVM corruption.
    """

    def hook(_: nn.Module, __: tuple[Any, ...], output: Any) -> Any:
        if not controller.enabled or controller.noise_factor <= 0.0:
            return output
        if not isinstance(output, torch.Tensor):
            return output
        if not output.is_floating_point():
            return output

        output_fp32 = output.float()
        output_std = float(output_fp32.std(unbiased=False).item())
        noise_std = controller.noise_factor * output_std
        if noise_std == 0.0:
            return output

        # Generate additive Gaussian noise with the same shape as the linear
        # layer output. This approximates crossbar conductance/readout errors
        # that perturb the analog MVM result before subsequent digital stages.
        noise = torch.randn_like(output_fp32) * noise_std
        corrupted = output_fp32 + noise

        controller.touched_layers += 1
        controller.noise_events += 1
        controller.total_noise_std += noise_std
        controller.layer_names.add(layer_name)

        return corrupted.to(dtype=output.dtype)

    return hook


def attach_linear_noise_hooks(
    model: nn.Module, controller: NoiseController
) -> list[torch.utils.hooks.RemovableHandle]:
    """
    Attach a forward hook to every `nn.Linear` module in the model.
    """

    handles: list[torch.utils.hooks.RemovableHandle] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        handles.append(module.register_forward_hook(create_noise_hook(controller, module_name)))
    return handles


def load_model_and_tokenizer() -> tuple[Any, Any, str]:
    """
    Load Qwen on CPU in float16 to reduce memory pressure.
    """

    runtime_device, device_map = choose_runtime_device()
    log_step(f"Loading config for {MODEL_ID}.")
    try:
        config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    except Exception as exc:
        raise SystemExit(
            "Failed to load the Qwen model config.\n"
            f"Installed transformers version: {transformers.__version__}\n"
            "Upgrade with: python -m pip install -U 'transformers>=4.40,<5' accelerate\n"
            f"Original error: {exc}"
        ) from exc

    log_step("Loading tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    log_step(f"Loading model weights on {runtime_device}. This can take a while.")
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
    return model, tokenizer, runtime_device


def prepare_inputs(model: nn.Module, tokenizer: Any, prompt: str, runtime_device: str) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        padding=False,
    )

    if runtime_device == "cpu":
        return {name: tensor.to("cpu") for name, tensor in encoded.items()}

    input_device = model.get_input_embeddings().weight.device
    return {name: tensor.to(input_device) for name, tensor in encoded.items()}


def generate_with_noise(
    model: nn.Module,
    tokenizer: Any,
    encoded: dict[str, torch.Tensor],
    controller: NoiseController,
    noise_factor: float,
) -> str:
    controller.set_noise_factor(noise_factor)
    label = "clean baseline" if noise_factor == 0.0 else f"noise factor {noise_factor:.2f}"
    log_step(f"Running generation for {label}.")

    with torch.inference_mode():
        generated_ids = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False if TEMPERATURE == 0.0 else True,
            temperature=None if TEMPERATURE == 0.0 else TEMPERATURE,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    prompt_length = encoded["input_ids"].shape[-1]
    new_tokens = generated_ids[0, prompt_length:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return decoded


def main() -> None:
    configure_logging()
    torch.manual_seed(SEED)
    log_step("Starting AIMC noise injection benchmark.")

    model, tokenizer, runtime_device = load_model_and_tokenizer()
    encoded = prepare_inputs(model, tokenizer, PROMPT, runtime_device)
    log_step(f"Prepared prompt with sequence length {encoded['input_ids'].shape[-1]} tokens.")

    controller = NoiseController()
    handles = attach_linear_noise_hooks(model, controller)
    log_step(f"Attached noise hooks to {len(handles)} linear layers.")

    print("=" * 96)
    print("Qwen1.5-MoE AIMC Noise Injection Report")
    print("=" * 96)
    print(f"Model                         : {MODEL_ID}")
    print(f"Execution device              : {runtime_device}")
    print(f"Prompt                        : {PROMPT}")
    print(f"Noise levels tested           : {NOISE_LEVELS}")
    print(f"Hooked linear layers          : {len(handles)}")
    print("-" * 96)

    try:
        for noise_factor in NOISE_LEVELS:
            generated_text = generate_with_noise(
                model=model,
                tokenizer=tokenizer,
                encoded=encoded,
                controller=controller,
                noise_factor=noise_factor,
            )
            avg_noise_std = (
                controller.total_noise_std / controller.noise_events
                if controller.noise_events > 0
                else 0.0
            )
            print(f"Noise Factor                  : {noise_factor:.2f}")
            print(f"Hook noise enabled            : {controller.enabled}")
            print(f"Linear layers perturbed       : {len(controller.layer_names)}")
            print(f"Noise injection events        : {controller.noise_events}")
            print(f"Average injected sigma        : {avg_noise_std:.6f}")
            print(f"Generated text                : {generated_text}")
            print("-" * 96)
    finally:
        for handle in handles:
            handle.remove()
        log_step("Removed all linear-layer noise hooks.")

    print("=" * 96)


if __name__ == "__main__":
    main()
