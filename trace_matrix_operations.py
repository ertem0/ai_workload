#!/usr/bin/env python3
"""
Profile MobileBERT with Analog In-Memory Computing (AIMC) proxy metrics.

This script is CPU-only and macOS-friendly. It loads a Hugging Face model,
executes a dummy forward pass, and reports six hardware-oriented metrics that
approximate how well the network maps to a memristor crossbar accelerator.
"""

from __future__ import annotations

import argparse
import logging
import math
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

try:
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn.jit_handles import get_shape
except ImportError as exc:  # pragma: no cover - import guard for local runtime
    raise SystemExit(
        "Missing dependency: fvcore\n"
        "Install with: python -m pip install fvcore"
    ) from exc

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError as exc:  # pragma: no cover - import guard for local runtime
    raise SystemExit(
        "Missing dependency: transformers\n"
        "Install with: python -m pip install transformers"
    ) from exc


DEFAULT_MODEL = "google/mobilebert-uncased"
DEFAULT_SENTENCE = (
    "Analog in-memory computing accelerators benefit from dense matrix "
    "operations, but they still need digital support for control-heavy steps."
)
DYNAMIC_MATMUL_OPS = {"matmul", "bmm", "baddbmm", "einsum"}


class HookableActivation(nn.Module):
    """Wrap a functional activation so PyTorch hooks can inspect its outputs."""

    def __init__(self, fn: Any, label: str) -> None:
        super().__init__()
        self.fn = fn
        self.label = label

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile MobileBERT for AIMC deployment on a CPU."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id.")
    parser.add_argument("--sentence", default=DEFAULT_SENTENCE, help="Dummy input sentence.")
    parser.add_argument(
        "--tile-size",
        type=int,
        default=128,
        help="Physical crossbar tile dimension (default: 128 for 128x128).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=64,
        help="Tokenizer truncation length used for the dummy input.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.getLogger("fvcore").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning, module="fvcore")
    warnings.filterwarnings(
        "ignore", category=FutureWarning, module="huggingface_hub.file_download"
    )


def iter_tensors(obj: Any) -> list[torch.Tensor]:
    """Flatten arbitrarily nested outputs into a list of tensors."""
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, Mapping):
        tensors: list[torch.Tensor] = []
        for value in obj.values():
            tensors.extend(iter_tensors(value))
        return tensors
    if hasattr(obj, "to_tuple"):
        return iter_tensors(obj.to_tuple())
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        tensors = []
        for item in obj:
            tensors.extend(iter_tensors(item))
        return tensors
    return []


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def tensor_storage_key(tensor: torch.Tensor) -> tuple[int, int, int]:
    storage = tensor.untyped_storage()
    return (storage.data_ptr(), tensor.storage_offset(), tensor_nbytes(tensor))


def format_int(value: float) -> str:
    return f"{value:,.0f}"


def format_float(value: float) -> str:
    return f"{value:,.4f}"


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    unit = units[0]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            break
        value /= 1024.0
    return f"{value:.2f} {unit}"


def product(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def classify_static_matrix(module_name: str) -> str:
    lowered = module_name.lower()
    if lowered.endswith(".query"):
        return "attention query projection"
    if lowered.endswith(".key"):
        return "attention key projection"
    if lowered.endswith(".value"):
        return "attention value projection"
    if ".attention.output.dense" in lowered:
        return "attention output projection"
    if ".intermediate." in lowered:
        return "feed-forward expansion"
    if ".output." in lowered:
        return "feed-forward contraction"
    if ".bottleneck." in lowered:
        return "bottleneck projection"
    if "pooler" in lowered:
        return "pooler projection"
    if "classifier" in lowered:
        return "task head projection"
    return "dense projection"


def is_attention_matrix_module(module: nn.Module) -> bool:
    required = ("num_attention_heads", "attention_head_size", "query", "key", "value")
    return all(hasattr(module, attr) for attr in required)


def collect_static_matrix_inventory(model: nn.Module, tile_size: int) -> list[dict[str, Any]]:
    matrices: list[dict[str, Any]] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        rows, cols = module.weight.shape
        row_tiles = math.ceil(rows / tile_size)
        col_tiles = math.ceil(cols / tile_size)
        tile_count = row_tiles * col_tiles
        provisioned_cells = tile_count * tile_size * tile_size
        used_cells = rows * cols
        matrices.append(
            {
                "name": module_name,
                "role": classify_static_matrix(module_name),
                "shape": tuple(module.weight.shape),
                "weight_bytes": tensor_nbytes(module.weight),
                "tiles": tile_count,
                "tiling_efficiency": used_cells / provisioned_cells if provisioned_cells else 0.0,
            }
        )
    return matrices


def wrap_functional_activations(model: nn.Module) -> list[str]:
    """
    Transformers often store activations as plain callables (e.g. ACT2FN["relu"]).
    Replacing them with modules makes register_forward_hook usable for sparsity.
    """

    wrapped: list[str] = []
    for module_name, module in model.named_modules():
        for attr_name, attr_value in vars(module).items():
            if not attr_name.endswith("_act_fn"):
                continue
            if isinstance(attr_value, nn.Module) or not callable(attr_value):
                continue
            label = f"{module_name}.{attr_name}" if module_name else attr_name
            setattr(module, attr_name, HookableActivation(attr_value, label))
            wrapped.append(label)
    return wrapped


def build_fvcore_inputs(batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    ordered_keys = ("input_ids", "attention_mask", "token_type_ids")
    return tuple(batch[key] for key in ordered_keys if key in batch and batch[key] is not None)


def matmul_mac_handle(inputs: Any, outputs: Any) -> Counter[str]:
    """Count multiply-accumulates for aten::matmul-like nodes."""
    input_shape = get_shape(inputs[0])
    output_shape = get_shape(outputs[0])
    if input_shape is None or output_shape is None or len(input_shape) == 0:
        return Counter()
    # For GEMM-like kernels, each output element performs one dot product of
    # length K, so MACs = output_elements * K.
    reduction_dim = int(input_shape[-1])
    macs = product(output_shape) * reduction_dim
    return Counter({"matmul": macs})


def bmm_mac_handle(inputs: Any, outputs: Any) -> Counter[str]:
    input_shape = get_shape(inputs[0])
    output_shape = get_shape(outputs[0])
    if input_shape is None or output_shape is None or len(input_shape) < 3:
        return Counter()
    reduction_dim = int(input_shape[-1])
    macs = product(output_shape) * reduction_dim
    return Counter({"bmm": macs})


def collect_hook_metrics(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    activation_bytes = 0
    seen_storages: set[tuple[int, int, int]] = set()
    dense_layers: list[dict[str, Any]] = []
    activation_zero_count = 0
    activation_value_count = 0
    activation_layer_names: set[str] = set()
    dynamic_ops: list[dict[str, Any]] = []
    handles: list[Any] = []

    def activation_memory_hook(module_name: str):
        def hook(_: nn.Module, __: tuple[Any, ...], output: Any) -> None:
            nonlocal activation_bytes
            for tensor in iter_tensors(output):
                key = tensor_storage_key(tensor)
                if key in seen_storages:
                    continue
                seen_storages.add(key)
                activation_bytes += tensor_nbytes(tensor)

        return hook

    def linear_hook(module_name: str, module: nn.Linear):
        def hook(_: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            input_tensors = iter_tensors(inputs)
            output_tensors = iter_tensors(output)
            if not input_tensors or not output_tensors:
                return

            input_tensor = input_tensors[0]
            output_tensor = output_tensors[0]
            if input_tensor.shape[-1] != module.in_features:
                return

            # A Linear layer maps every input vector of length K onto N outputs,
            # so MACs = number_of_vectors * K * N.
            vectors = input_tensor.numel() // module.in_features
            layer_macs = vectors * module.in_features * module.out_features
            input_bytes = tensor_nbytes(input_tensor)
            output_bytes = tensor_nbytes(output_tensor)
            weight_bytes = tensor_nbytes(module.weight)
            # This denominator approximates the data traffic a crossbar tile must
            # source/sink around the analog MAC itself.
            denominator = input_bytes + output_bytes + weight_bytes
            arithmetic_intensity = layer_macs / denominator if denominator else 0.0

            dense_layers.append(
                {
                    "name": module_name,
                    "shape": tuple(module.weight.shape),
                    "macs": layer_macs,
                    "input_bytes": input_bytes,
                    "output_bytes": output_bytes,
                    "weight_bytes": weight_bytes,
                    "arithmetic_intensity": arithmetic_intensity,
                }
            )

        return hook

    def sparsity_hook(module_name: str):
        def hook(_: nn.Module, __: tuple[Any, ...], output: Any) -> None:
            nonlocal activation_zero_count, activation_value_count
            for tensor in iter_tensors(output):
                activation_layer_names.add(module_name)
                activation_zero_count += int(tensor.eq(0).sum().item())
                activation_value_count += tensor.numel()

        return hook

    def attention_hook(module_name: str, module: nn.Module):
        def hook(_: nn.Module, inputs: tuple[Any, ...], __: Any) -> None:
            input_tensors = [tensor for tensor in iter_tensors(inputs) if tensor.dim() == 3]
            if not input_tensors:
                return

            hidden_states = input_tensors[0]
            batch_size = int(hidden_states.shape[0])
            query_len = int(hidden_states.shape[1])
            key_len = query_len
            num_heads = int(getattr(module, "num_attention_heads"))
            head_dim = int(getattr(module, "attention_head_size"))

            # These two tensor products are dynamic because both operands are
            # runtime activations rather than fixed programmed conductance maps.
            score_macs = batch_size * num_heads * query_len * key_len * head_dim
            dynamic_ops.append(
                {
                    "module": module_name,
                    "kind": "Q x K^T",
                    "role": "attention score matrix",
                    "lhs_shape": (batch_size, num_heads, query_len, head_dim),
                    "rhs_shape": (batch_size, num_heads, head_dim, key_len),
                    "output_shape": (batch_size, num_heads, query_len, key_len),
                    "macs": score_macs,
                }
            )
            dynamic_ops.append(
                {
                    "module": module_name,
                    "kind": "Attention x V",
                    "role": "attention context mixing",
                    "lhs_shape": (batch_size, num_heads, query_len, key_len),
                    "rhs_shape": (batch_size, num_heads, key_len, head_dim),
                    "output_shape": (batch_size, num_heads, query_len, head_dim),
                    "macs": score_macs,
                }
            )

        return hook

    for module_name, module in model.named_modules():
        if module_name and len(list(module.children())) == 0:
            handles.append(module.register_forward_hook(activation_memory_hook(module_name)))
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook(module_name, module)))
        if isinstance(module, (nn.ReLU, nn.GELU, HookableActivation)):
            handles.append(module.register_forward_hook(sparsity_hook(module_name)))
        if is_attention_matrix_module(module):
            handles.append(module.register_forward_hook(attention_hook(module_name, module)))

    with torch.inference_mode():
        model(**batch)

    for handle in handles:
        handle.remove()

    activation_sparsity = (
        activation_zero_count / activation_value_count if activation_value_count else 0.0
    )

    return {
        "activation_bytes": activation_bytes,
        "dense_layers": dense_layers,
        "dynamic_ops": dynamic_ops,
        "activation_sparsity": activation_sparsity,
        "activation_layer_count": len(activation_layer_names),
    }


def analyze_flops(model: nn.Module, fvcore_inputs: tuple[torch.Tensor, ...]) -> dict[str, Any]:
    flops = FlopCountAnalysis(model, fvcore_inputs)
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    flops.tracer_warnings("none")
    flops.set_op_handle("aten::matmul", matmul_mac_handle)
    flops.set_op_handle("aten::bmm", bmm_mac_handle)
    flops.set_op_handle("aten::baddbmm", bmm_mac_handle)

    total_macs = float(flops.total())
    by_operator = flops.by_operator()
    by_module_and_operator = flops.by_module_and_operator()

    linear_module_names = [
        module_name
        for module_name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    linear_macs = float(
        sum(sum(by_module_and_operator.get(name, Counter()).values()) for name in linear_module_names)
    )
    nonlinear_macs = max(total_macs - linear_macs, 0.0)

    dynamic_macs = float(sum(by_operator.get(op, 0.0) for op in DYNAMIC_MATMUL_OPS))
    static_weight_macs = linear_macs

    return {
        "total_macs": total_macs,
        "linear_macs": linear_macs,
        "nonlinear_macs": nonlinear_macs,
        "dynamic_macs": dynamic_macs,
        "static_weight_macs": static_weight_macs,
        "by_operator": by_operator,
        "unsupported_ops": flops.unsupported_ops(),
    }


def compute_tiling_metrics(model: nn.Module, tile_size: int) -> dict[str, Any]:
    total_tiles = 0
    used_cells = 0
    provisioned_cells = 0
    layer_count = 0

    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue
        rows, cols = module.weight.shape
        # A weight matrix is physically partitioned across square crossbar tiles.
        # Any partial edge tile still consumes a full 128x128 (or user-chosen)
        # array, which creates zero-padding waste.
        row_tiles = math.ceil(rows / tile_size)
        col_tiles = math.ceil(cols / tile_size)
        tiles = row_tiles * col_tiles
        used = rows * cols
        provisioned = tiles * tile_size * tile_size

        total_tiles += tiles
        used_cells += used
        provisioned_cells += provisioned
        layer_count += 1

    wasted_cells = provisioned_cells - used_cells
    efficiency = used_cells / provisioned_cells if provisioned_cells else 0.0

    return {
        "layer_count": layer_count,
        "total_tiles": total_tiles,
        "used_cells": used_cells,
        "provisioned_cells": provisioned_cells,
        "wasted_cells": wasted_cells,
        "efficiency": efficiency,
    }


def print_report(
    args: argparse.Namespace,
    batch: Mapping[str, torch.Tensor],
    param_bytes: int,
    static_matrices: Sequence[Mapping[str, Any]],
    hook_metrics: Mapping[str, Any],
    flop_metrics: Mapping[str, Any],
    tiling_metrics: Mapping[str, Any],
    wrapped_activations: Sequence[str],
) -> None:
    total_memory_bytes = param_bytes + int(hook_metrics["activation_bytes"])
    system_ai = (
        flop_metrics["total_macs"] / total_memory_bytes if total_memory_bytes else 0.0
    )

    dense_layers = hook_metrics["dense_layers"]
    avg_crossbar_ai = (
        sum(layer["arithmetic_intensity"] for layer in dense_layers) / len(dense_layers)
        if dense_layers
        else 0.0
    )

    static_plus_dynamic = flop_metrics["static_weight_macs"] + flop_metrics["dynamic_macs"]
    dynamic_share = (
        flop_metrics["dynamic_macs"] / static_plus_dynamic if static_plus_dynamic else 0.0
    )
    static_share = (
        flop_metrics["static_weight_macs"] / static_plus_dynamic if static_plus_dynamic else 0.0
    )

    linear_share = (
        flop_metrics["linear_macs"] / flop_metrics["total_macs"]
        if flop_metrics["total_macs"]
        else 0.0
    )
    nonlinear_share = (
        flop_metrics["nonlinear_macs"] / flop_metrics["total_macs"]
        if flop_metrics["total_macs"]
        else 0.0
    )

    sequence_length = int(batch["input_ids"].shape[-1])
    dynamic_ops = hook_metrics["dynamic_ops"]

    print("=" * 92)
    print("AIMC Hardware Profiling Report")
    print("=" * 92)
    print(f"Model                         : {args.model}")
    print("Device                        : CPU")
    print(f"Dummy sentence                : {args.sentence}")
    print(f"Tokenized sequence length     : {sequence_length}")
    print(f"Dense layers profiled         : {len(dense_layers)}")
    print(f"Activation layers probed      : {hook_metrics['activation_layer_count']}")
    print(f"Wrapped functional activations: {len(wrapped_activations)}")
    print("-" * 92)
    print("1. System-Level Arithmetic Intensity")
    print(
        f"   Total MACs / Total memory footprint = "
        f"{format_int(flop_metrics['total_macs'])} / {format_bytes(total_memory_bytes)}"
    )
    print(f"   Parameter bytes            : {format_bytes(param_bytes)}")
    print(
        f"   Intermediate activation bytes: "
        f"{format_bytes(int(hook_metrics['activation_bytes']))}"
    )
    print(f"   System arithmetic intensity: {format_float(system_ai)} MACs/byte")
    print("-" * 92)
    print("2. Crossbar-Level Arithmetic Intensity (Predicted)")
    print(
        "   Mean over all Linear/Dense layers of "
        "MACs / (input bytes + output bytes + weight bytes)"
    )
    print(f"   Average dense-layer AI     : {format_float(avg_crossbar_ai)} MACs/byte")
    print("-" * 92)
    print("3. Crossbar Tiling Efficiency & Footprint")
    print(f"   Tile size                  : {args.tile_size} x {args.tile_size}")
    print(f"   Total crossbar tiles       : {format_int(tiling_metrics['total_tiles'])}")
    print(f"   Provisioned cells          : {format_int(tiling_metrics['provisioned_cells'])}")
    print(f"   Used weight cells          : {format_int(tiling_metrics['used_cells'])}")
    print(f"   Wasted padded cells        : {format_int(tiling_metrics['wasted_cells'])}")
    print(f"   Tiling efficiency          : {format_percent(tiling_metrics['efficiency'])}")
    print("-" * 92)
    print("4. Linear vs. Non-Linear Operation Ratio")
    print(f"   Linear/Dense MACs          : {format_int(flop_metrics['linear_macs'])}")
    print(f"   Non-linear/digital MACs    : {format_int(flop_metrics['nonlinear_macs'])}")
    print(f"   Linear share               : {format_percent(linear_share)}")
    print(f"   Non-linear share           : {format_percent(nonlinear_share)}")
    print("-" * 92)
    print("5. Static vs. Dynamic Tensor Operations Ratio")
    print(
        "   Static MACs use fixed trained weights. "
        "Dynamic MACs come from tensor-to-tensor products such as attention."
    )
    print(f"   Static-weight MACs         : {format_int(flop_metrics['static_weight_macs'])}")
    print(f"   Dynamic tensor MACs        : {format_int(flop_metrics['dynamic_macs'])}")
    print(f"   Static share               : {format_percent(static_share)}")
    print(f"   Dynamic share              : {format_percent(dynamic_share)}")
    print("-" * 92)
    print("Identified Static Weight Matrices")
    print("   These matrices are fixed after training and are natural AIMC crossbar candidates.")
    for matrix in static_matrices:
        rows, cols = matrix["shape"]
        print(
            "   "
            f"[STATIC] {matrix['name']} | role={matrix['role']} | "
            f"shape=({rows}, {cols}) | bytes={format_bytes(matrix['weight_bytes'])} | "
            f"tiles={format_int(matrix['tiles'])} | "
            f"tile_eff={format_percent(matrix['tiling_efficiency'])}"
        )
    print("-" * 92)
    print("Identified Dynamic Tensor Products")
    print("   These matrices are formed at runtime from activations, so they require digital support.")
    for op in dynamic_ops:
        print(
            "   "
            f"[DYNAMIC] {op['module']} | op={op['kind']} | role={op['role']} | "
            f"lhs={op['lhs_shape']} | rhs={op['rhs_shape']} | "
            f"out={op['output_shape']} | MACs={format_int(op['macs'])}"
        )
    print("-" * 92)
    print("6. Dynamic Activation Sparsity")
    print(
        "   Exact fraction of activation outputs that are numerically equal to 0.0 "
        "after hookable activation layers."
    )
    print(
        f"   Exact zero sparsity        : "
        f"{format_percent(hook_metrics['activation_sparsity'])}"
    )
    print("-" * 92)
    print("Operator MAC breakdown from fvcore")
    for op_name, op_macs in sorted(
        flop_metrics["by_operator"].items(), key=lambda item: item[1], reverse=True
    ):
        print(f"   {op_name:<28} {format_int(op_macs)}")
    unsupported_ops = flop_metrics["unsupported_ops"]
    if unsupported_ops:
        ignored = ", ".join(f"{name} ({count})" for name, count in unsupported_ops.items())
        print("-" * 92)
        print(f"Unsupported ops suppressed    : {ignored}")
    print("=" * 92)


def main() -> None:
    args = parse_args()
    configure_logging()

    torch.set_num_threads(max(torch.get_num_threads(), 1))
    device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    model.eval()

    wrapped_activations = wrap_functional_activations(model)

    batch = tokenizer(
        args.sentence,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    )
    batch = {name: tensor.to(device) for name, tensor in batch.items()}
    fvcore_inputs = build_fvcore_inputs(batch)

    param_bytes = sum(tensor_nbytes(parameter) for parameter in model.parameters())
    static_matrices = collect_static_matrix_inventory(model, args.tile_size)
    hook_metrics = collect_hook_metrics(model, batch)
    flop_metrics = analyze_flops(model, fvcore_inputs)
    tiling_metrics = compute_tiling_metrics(model, args.tile_size)

    print_report(
        args=args,
        batch=batch,
        param_bytes=param_bytes,
        static_matrices=static_matrices,
        hook_metrics=hook_metrics,
        flop_metrics=flop_metrics,
        tiling_metrics=tiling_metrics,
        wrapped_activations=wrapped_activations,
    )


if __name__ == "__main__":
    main()
