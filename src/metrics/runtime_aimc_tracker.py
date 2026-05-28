from __future__ import annotations

import logging
import re
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from src.metrics.crossbar_tiling_analyzer import classify_static_matrix

try:
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn.jit_handles import get_shape
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "Missing dependency: fvcore\n"
        "Install with: python -m pip install fvcore"
    ) from exc


DYNAMIC_MATMUL_OPS = {"matmul", "bmm", "baddbmm", "einsum"}


class HookableActivation(nn.Module):
    """Wrap callable activation functions so forward hooks can inspect outputs."""

    def __init__(self, fn: Any, label: str) -> None:
        super().__init__()
        self.fn = fn
        self.label = label

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


def configure_fvcore_logging() -> None:
    logging.getLogger("fvcore").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning, module="fvcore")


def iter_tensors(obj: Any) -> list[torch.Tensor]:
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
        tensors: list[torch.Tensor] = []
        for item in obj:
            tensors.extend(iter_tensors(item))
        return tensors
    return []


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def tensor_storage_key(tensor: torch.Tensor) -> tuple[int, int, int]:
    storage = tensor.untyped_storage()
    return (storage.data_ptr(), tensor.storage_offset(), tensor_nbytes(tensor))


def product(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def build_fvcore_inputs(batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    ordered_keys = ("input_ids", "attention_mask", "token_type_ids", "position_ids")
    return tuple(batch[key] for key in ordered_keys if key in batch and batch[key] is not None)


def matmul_mac_handle(inputs: Any, outputs: Any) -> Counter[str]:
    input_shape = get_shape(inputs[0])
    output_shape = get_shape(outputs[0])
    if input_shape is None or output_shape is None or len(input_shape) == 0:
        return Counter()
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


def compact_module_name(module_name: str) -> str:
    """Compact repeated layer indices so multi-layer inventories stay readable."""

    return re.sub(r"(\.(?:layers|h|block|blocks))\.\d+(\.)", r"\1.*\2", module_name)


def is_rope_module(module: nn.Module) -> bool:
    cls = module.__class__.__name__.lower()
    return (
        "rotary" in cls
        or hasattr(module, "inv_freq")
        or (hasattr(module, "cos_cached") and hasattr(module, "sin_cached"))
    )


def is_rms_norm_module(module: nn.Module) -> bool:
    cls = module.__class__.__name__.lower()
    return "rmsnorm" in cls or "rms_norm" in cls


def is_activation_module(module: nn.Module) -> bool:
    if isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU)):
        return True
    cls = module.__class__.__name__.lower()
    return any(kw in cls for kw in ("activation", "silu", "gelu", "relu", "swish", "mish", "newgelu", "quickgelu"))


def is_gated_mlp_module(module: nn.Module) -> bool:
    # Standard naming: LLaMA / Mistral / Qwen dense MLP
    if hasattr(module, "gate_proj") and hasattr(module, "up_proj") and hasattr(module, "down_proj"):
        return True
    # Alternative naming: MiniCPM / some MoE expert FFNs use w1/w2/w3
    if hasattr(module, "w1") and hasattr(module, "w2") and hasattr(module, "w3"):
        return True
    return False


def is_transformer_block_module(module: nn.Module) -> bool:
    has_attn = hasattr(module, "self_attn") or hasattr(module, "attention")
    has_mlp = hasattr(module, "mlp") or hasattr(module, "feed_forward")
    return has_attn and has_mlp


def is_moe_block_module(module: nn.Module) -> bool:
    gate = getattr(module, "gate", None)
    has_experts = hasattr(module, "experts") or hasattr(module, "num_experts")
    if gate is None or not has_experts:
        return False
    if isinstance(gate, nn.Linear):
        return True
    # OLMoE-style: gate is a routing Module with a weight parameter
    if isinstance(gate, nn.Module) and isinstance(getattr(gate, "weight", None), torch.nn.Parameter):
        return True
    return False


def is_olmoe_experts_module(module: nn.Module) -> bool:
    """Detect OlmoeExperts-style batched expert blocks (3D Parameter tensors, not nn.Linear)."""
    gate_up = getattr(module, "gate_up_proj", None)
    down = getattr(module, "down_proj", None)
    return (
        isinstance(gate_up, torch.nn.Parameter) and gate_up.dim() == 3
        and isinstance(down, torch.nn.Parameter) and down.dim() == 3
    )


def is_olmoe_router_module(module: nn.Module) -> bool:
    """Detect OlmoeTopKRouter-style modules that apply F.linear on a raw 2D Parameter."""
    if isinstance(module, nn.Linear):
        return False
    weight = getattr(module, "weight", None)
    return (
        isinstance(weight, torch.nn.Parameter)
        and weight.dim() == 2
        and hasattr(module, "top_k")
        and hasattr(module, "num_experts")
    )


def is_attention_matrix_module(module: nn.Module) -> bool:
    """
    Identify attention modules that produce dynamic tensor products.

    Supports three families:
    - MobileBERT: num_attention_heads + attention_head_size + query/key/value
    - Qwen / dense: num_heads + head_dim + q_proj/k_proj/v_proj
    - LLaMA / MiniCPM SDPA: q_proj/k_proj/v_proj + any head-count attr
      (head_dim may be absent when computed on the fly)
    """
    mobilebert_attrs = ("num_attention_heads", "attention_head_size", "query", "key", "value")
    if all(hasattr(module, a) for a in mobilebert_attrs):
        return True

    has_qkv = all(hasattr(module, a) for a in ("q_proj", "k_proj", "v_proj"))
    if not has_qkv:
        return False

    has_heads = any(hasattr(module, a) for a in ("num_heads", "num_attention_heads", "num_key_value_heads"))
    has_dim   = any(hasattr(module, a) for a in ("head_dim", "attention_head_size"))
    return has_heads and has_dim


class RuntimeAIMCTracker:
    """
    Collect runtime AIMC metrics originally prototyped in trace_matrix_operations.py.

    Implemented metrics:
    - system-level arithmetic intensity
    - crossbar-level arithmetic intensity
    - linear vs non-linear operation ratio
    - static vs dynamic tensor operation ratio
    - activation sparsity
    """

    def __init__(self, model: nn.Module, metrics_cfg: Mapping[str, Any]) -> None:
        self.model = model
        self.metrics_cfg = metrics_cfg
        self.handles: list[Any] = []
        self.enabled = True
        self.prompt_active = False
        self.prompt_count = 0
        self.current_phase = "prefill"
        self._decode_pass_count = 0
        self._suppress_current_pass = False

        self.param_numels = sum(p.numel() for p in model.parameters())
        self.wrapped_activations = self._wrap_functional_activations()

        self.current_activation_numels = 0
        self.current_seen_storages: set[tuple[int, int, int]] = set()
        self.current_prompt_index: int | None = None
        self.current_input_shapes: dict[str, tuple[int, ...]] = {}
        self.current_input_dtypes: dict[str, str] = {}
        self.current_operations: list[dict[str, Any]] = []
        self.inference_traces: list[dict[str, Any]] = []

        self.total_activation_numels = 0
        self.dense_layers: list[dict[str, Any]] = []
        self.activation_zero_count = 0
        self.activation_value_count = 0
        self.activation_layer_names: set[str] = set()
        self.dynamic_ops: list[dict[str, Any]] = []

        self.total_macs = 0.0
        self.linear_macs = 0.0
        self.nonlinear_macs = 0.0
        self.dynamic_macs = 0.0
        self.static_weight_macs = 0.0
        self.by_operator_accumulator: Counter[str] = Counter()

    def _wrap_functional_activations(self) -> list[str]:
        wrapped: list[str] = []
        for module_name, module in self.model.named_modules():
            for attr_name, attr_value in vars(module).items():
                if not attr_name.endswith("_act_fn"):
                    continue
                if isinstance(attr_value, nn.Module) or not callable(attr_value):
                    continue
                label = f"{module_name}.{attr_name}" if module_name else attr_name
                setattr(module, attr_name, HookableActivation(attr_value, label))
                wrapped.append(label)
        return wrapped

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def register_hooks(self) -> None:
        def activation_memory_hook(_: nn.Module, __: tuple[Any, ...], output: Any) -> None:
            if not self.enabled or not self.prompt_active or self._suppress_current_pass:
                return
            for tensor in iter_tensors(output):
                key = tensor_storage_key(tensor)
                if key in self.current_seen_storages:
                    continue
                self.current_seen_storages.add(key)
                self.current_activation_numels += tensor.numel()

        def linear_hook(module_name: str, module: nn.Linear):
            def hook(_: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
                if not self.enabled or not self.prompt_active or self._suppress_current_pass:
                    return
                input_tensors = iter_tensors(inputs)
                output_tensors = iter_tensors(output)
                if not input_tensors or not output_tensors:
                    return

                input_tensor = input_tensors[0]
                output_tensor = output_tensors[0]
                if input_tensor.shape[-1] != module.in_features:
                    return

                vectors = input_tensor.numel() // module.in_features
                layer_macs = vectors * module.in_features * module.out_features
                input_numels = input_tensor.numel()
                output_numels = output_tensor.numel()
                weight_numels = module.weight.numel()
                denominator = input_numels + output_numels + weight_numels
                arithmetic_intensity = layer_macs / denominator if denominator else 0.0

                self.dense_layers.append(
                    {
                        "name": module_name,
                        "shape": tuple(module.weight.shape),
                        "macs": layer_macs,
                        "input_numels": input_numels,
                        "output_numels": output_numels,
                        "weight_numels": weight_numels,
                        "arithmetic_intensity": arithmetic_intensity,
                    }
                )
                self.current_operations.append(
                    {
                        "op_type": "linear",
                        "op_family": "static_weight_matmul",
                        "module": module_name,
                        "role": classify_static_matrix(module_name),
                        "inputs": [
                            {
                                "name": "input",
                                "shape": tuple(input_tensor.shape),
                                "dtype": str(input_tensor.dtype),
                                "kind": "activation",
                                "static": False,
                                "numels": input_numels,
                            }
                        ],
                        "weights": [
                            {
                                "parameter_ref": f"{module_name}.weight",
                                "shape": tuple(module.weight.shape),
                                "dtype": str(module.weight.dtype),
                                "kind": "parameter",
                                "static": True,
                                "numels": weight_numels,
                            }
                        ],
                        "outputs": [
                            {
                                "name": "output",
                                "shape": tuple(output_tensor.shape),
                                "dtype": str(output_tensor.dtype),
                                "kind": "activation",
                                "static": False,
                                "numels": output_numels,
                            }
                        ],
                        "math": {
                            "macs": layer_macs,
                            "flops_estimate": layer_macs * 2,
                            "arithmetic_intensity": arithmetic_intensity,
                        },
                    }
                )

            return hook

        def sparsity_hook(module_name: str):
            def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
                if not self.enabled or not self.prompt_active or self._suppress_current_pass:
                    return
                input_tensors = iter_tensors(inputs)
                for tensor in iter_tensors(output):
                    self.activation_layer_names.add(module_name)
                    self.activation_zero_count += int(tensor.eq(0).sum().item())
                    self.activation_value_count += tensor.numel()
                    input_shape = (
                        tuple(input_tensors[0].shape)
                        if input_tensors
                        else tuple(tensor.shape)
                    )
                    input_numels = input_tensors[0].numel() if input_tensors else tensor.numel()
                    output_numels = tensor.numel()
                    total_numels = input_numels + output_numels
                    n = tensor.numel()
                    ai = n / total_numels if total_numels else 0.0
                    self.current_operations.append(
                        {
                            "op_type": "activation",
                            "op_family": "elementwise",
                            "module": module_name,
                            "role": "nonlinear activation",
                            "activation": getattr(module, "label", module.__class__.__name__),
                            "inputs": [{"name": "input", "shape": input_shape, "dtype": str(tensor.dtype), "numels": input_numels}],
                            "outputs": [{"name": "output", "shape": tuple(tensor.shape), "dtype": str(tensor.dtype), "numels": output_numels}],
                            "input_shape": input_shape,
                            "output_shape": tuple(tensor.shape),
                            "dtype": str(tensor.dtype),
                            "element_count": n,
                            "static": False,
                            "math": {"macs": n // 2, "flops_estimate": n, "arithmetic_intensity": ai},
                        }
                    )

            return hook

        def attention_hook(module_name: str, module: nn.Module):
            def hook(_: nn.Module, inputs: tuple[Any, ...], __: Any) -> None:
                if not self.enabled or not self.prompt_active or self._suppress_current_pass:
                    return

                input_tensors = [tensor for tensor in iter_tensors(inputs) if tensor.dim() == 3]
                if not input_tensors:
                    return

                hidden_states = input_tensors[0]
                batch_size = int(hidden_states.shape[0])
                query_len = int(hidden_states.shape[1])
                key_len = query_len
                num_heads = int(getattr(module, "num_heads",
                               getattr(module, "num_attention_heads",
                               getattr(module, "num_key_value_heads", 0))))
                head_dim = int(getattr(module, "head_dim",
                              getattr(module, "attention_head_size", 0)))
                if head_dim <= 0 and num_heads > 0:
                    head_dim = int(hidden_states.shape[-1]) // num_heads
                if num_heads <= 0 or head_dim <= 0:
                    return

                score_macs = batch_size * num_heads * query_len * key_len * head_dim
                compact_name = compact_module_name(module_name)

                # These two tensor products are formed from runtime activations,
                # not fixed programmed weights, so they represent digital-side
                # attention work rather than static crossbar mappings.
                attention_ops = [
                    {
                        "module": compact_name,
                        "kind": "Q x K^T",
                        "role": "attention score matrix",
                        "lhs_shape": (batch_size, num_heads, query_len, head_dim),
                        "rhs_shape": (batch_size, num_heads, head_dim, key_len),
                        "output_shape": (batch_size, num_heads, query_len, key_len),
                        "macs": score_macs,
                    },
                    {
                        "module": compact_name,
                        "kind": "Attention x V",
                        "role": "attention context mixing",
                        "lhs_shape": (batch_size, num_heads, query_len, key_len),
                        "rhs_shape": (batch_size, num_heads, key_len, head_dim),
                        "output_shape": (batch_size, num_heads, query_len, head_dim),
                        "macs": score_macs,
                    },
                ]
                self.dynamic_ops.extend(attention_ops)
                for op in attention_ops:
                    self.current_operations.append(
                        {
                            "op_type": "matmul",
                            "op_family": "dynamic_activation_matmul",
                            "module": op["module"],
                            "role": op["role"],
                            "kind": op["kind"],
                            "inputs": [
                                {
                                    "name": "lhs",
                                    "shape": op["lhs_shape"],
                                    "dtype": str(hidden_states.dtype),
                                    "kind": "activation",
                                    "static": False,
                                },
                                {
                                    "name": "rhs",
                                    "shape": op["rhs_shape"],
                                    "dtype": str(hidden_states.dtype),
                                    "kind": "activation",
                                    "static": False,
                                },
                            ],
                            "outputs": [
                                {
                                    "name": "output",
                                    "shape": op["output_shape"],
                                    "dtype": str(hidden_states.dtype),
                                    "kind": "activation",
                                    "static": False,
                                }
                            ],
                            "math": {
                                "macs": op["macs"],
                                "flops_estimate": op["macs"] * 2,
                            },
                        }
                    )

            return hook

        def layernorm_hook(module_name: str):
            def hook(_: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
                if not self.enabled or not self.prompt_active or self._suppress_current_pass:
                    return
                input_tensors = iter_tensors(inputs)
                if not input_tensors or not isinstance(output, torch.Tensor):
                    return
                x = input_tensors[0]
                n = x.numel()
                input_numels = x.numel()
                output_numels = output.numel()
                total_numels = input_numels + output_numels
                flops = 5 * n
                ai = flops / total_numels if total_numels else 0.0
                self.current_operations.append({
                    "op_type": "layernorm",
                    "op_family": "reduction",
                    "module": module_name,
                    "role": "layer normalization",
                    "inputs": [{"name": "input", "shape": tuple(x.shape), "dtype": str(x.dtype), "numels": input_numels}],
                    "outputs": [{"name": "output", "shape": tuple(output.shape), "dtype": str(output.dtype), "numels": output_numels}],
                    "dtype": str(x.dtype),
                    "element_count": n,
                    "math": {"macs": flops // 2, "flops_estimate": flops, "arithmetic_intensity": ai},
                })
            return hook

        def softmax_hook(module_name: str):
            def hook(_: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
                if not self.enabled or not self.prompt_active or self._suppress_current_pass:
                    return
                input_tensors = iter_tensors(inputs)
                if not input_tensors or not isinstance(output, torch.Tensor):
                    return
                x = input_tensors[0]
                n = x.numel()
                input_numels = x.numel()
                output_numels = output.numel()
                total_numels = input_numels + output_numels
                flops = 3 * n
                ai = flops / total_numels if total_numels else 0.0
                self.current_operations.append({
                    "op_type": "softmax",
                    "op_family": "reduction",
                    "module": module_name,
                    "role": "softmax normalization",
                    "inputs": [{"name": "input", "shape": tuple(x.shape), "dtype": str(x.dtype), "numels": input_numels}],
                    "outputs": [{"name": "output", "shape": tuple(output.shape), "dtype": str(output.dtype), "numels": output_numels}],
                    "dtype": str(x.dtype),
                    "element_count": n,
                    "math": {"macs": flops // 2, "flops_estimate": flops, "arithmetic_intensity": ai},
                })
            return hook

        def rope_hook(module_name: str):
            def hook(_: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
                if not self.enabled or not self.prompt_active or self._suppress_current_pass:
                    return
                output_tensors = [t for t in iter_tensors(output) if t.is_floating_point()]
                if not output_tensors:
                    return
                n = sum(t.numel() for t in output_tensors)
                total_numels = sum(t.numel() for t in output_tensors)
                flops = 6 * n
                ai = flops / (2 * total_numels) if total_numels else 0.0
                self.current_operations.append({
                    "op_type": "rope_embed",
                    "op_family": "rope",
                    "module": module_name,
                    "role": "rotary position embedding",
                    "outputs": [
                        {"name": f"output_{i}", "shape": tuple(t.shape), "dtype": str(t.dtype), "numels": t.numel()}
                        for i, t in enumerate(output_tensors)
                    ],
                    "element_count": n,
                    "math": {"macs": flops // 2, "flops_estimate": flops, "arithmetic_intensity": ai},
                })
            return hook

        def gated_mlp_hook(module_name: str, module: nn.Module):
            def hook(_: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
                if not self.enabled or not self.prompt_active or self._suppress_current_pass:
                    return
                input_tensors = [t for t in iter_tensors(inputs) if t.is_floating_point()]
                if not input_tensors:
                    return
                x = input_tensors[0]
                gate_linear = getattr(module, "gate_proj", None) or getattr(module, "w1", None)
                if gate_linear is None:
                    return
                intermediate_size = gate_linear.out_features
                n_tokens = x.numel() // x.shape[-1]
                if n_tokens == 0:
                    return
                n = n_tokens * intermediate_size
                total_numels = 3 * n
                ai = n / total_numels if total_numels else 0.0
                self.current_operations.append({
                    "op_type": "elementwise_multiply",
                    "op_family": "elementwise",
                    "module": module_name,
                    "role": "gated mlp activation multiply",
                    "inputs": [
                        {"name": "gate", "shape": (n_tokens, intermediate_size), "dtype": str(x.dtype), "numels": n},
                        {"name": "up", "shape": (n_tokens, intermediate_size), "dtype": str(x.dtype), "numels": n},
                    ],
                    "outputs": [
                        {"name": "output", "shape": (n_tokens, intermediate_size), "dtype": str(x.dtype), "numels": n},
                    ],
                    "element_count": n,
                    "math": {"macs": n, "flops_estimate": n, "arithmetic_intensity": ai},
                })
            return hook

        def transformer_block_hook(module_name: str):
            def hook(_: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
                if not self.enabled or not self.prompt_active or self._suppress_current_pass:
                    return
                hidden_states = next(
                    (t for t in iter_tensors(inputs) if t.is_floating_point() and t.dim() >= 2),
                    None,
                )
                if hidden_states is None:
                    return
                n = hidden_states.numel()
                shape = tuple(hidden_states.shape)
                dtype = str(hidden_states.dtype)
                for role in ("attention", "mlp"):
                    total_numels = 2 * n
                    ai = n / total_numels if total_numels else 0.0
                    self.current_operations.append({
                        "op_type": "residual_add",
                        "op_family": "elementwise",
                        "module": module_name,
                        "role": f"residual add after {role}",
                        "inputs": [
                            {"name": "residual", "shape": shape, "dtype": dtype, "numels": n},
                            {"name": "sublayer_output", "shape": shape, "dtype": dtype, "numels": n},
                        ],
                        "outputs": [{"name": "output", "shape": shape, "dtype": dtype, "numels": n}],
                        "element_count": n,
                        "math": {"macs": n, "flops_estimate": n, "arithmetic_intensity": ai},
                    })
            return hook

        def olmoe_router_hook(module_name: str, module: nn.Module):
            def hook(_: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
                if not self.enabled or not self.prompt_active or self._suppress_current_pass:
                    return
                input_tensors = [t for t in iter_tensors(inputs) if t.is_floating_point()]
                if not input_tensors:
                    return
                x = input_tensors[0]
                weight = module.weight  # (num_experts, hidden_dim)
                num_experts = int(weight.shape[0])
                hidden_dim = int(weight.shape[1])
                n_tokens = x.numel() // x.shape[-1]
                if n_tokens == 0:
                    return
                in_n  = n_tokens * hidden_dim
                w_n   = weight.numel()
                out_n = n_tokens * num_experts
                macs  = n_tokens * hidden_dim * num_experts
                denom = in_n + w_n + out_n
                ai    = macs / denom if denom else 0.0
                dtype   = str(x.dtype)
                w_dtype = str(weight.dtype)

                self.dense_layers.append({
                    "name":                 module_name,
                    "shape":                (num_experts, hidden_dim),
                    "macs":                 macs,
                    "input_numels":         in_n,
                    "output_numels":        out_n,
                    "weight_numels":        w_n,
                    "arithmetic_intensity": ai,
                })
                self.current_operations.append({
                    "op_type":   "linear",
                    "op_family": "static_weight_matmul",
                    "module":    module_name,
                    "role":      classify_static_matrix(module_name),
                    "inputs":  [{"name": "input",  "shape": (n_tokens, hidden_dim),   "dtype": dtype,   "kind": "activation", "static": False, "numels": in_n}],
                    "weights": [{"parameter_ref": f"{module_name}.weight", "shape": (num_experts, hidden_dim), "dtype": w_dtype, "kind": "parameter", "static": True, "numels": w_n}],
                    "outputs": [{"name": "output", "shape": (n_tokens, num_experts), "dtype": dtype,   "kind": "activation", "static": False, "numels": out_n}],
                    "math": {"macs": macs, "flops_estimate": macs * 2, "arithmetic_intensity": ai},
                })
            return hook

        def olmoe_experts_hook(module_name: str, module: nn.Module):
            def hook(_: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
                if not self.enabled or not self.prompt_active or self._suppress_current_pass:
                    return
                # OlmoeExperts.forward(hidden_states, top_k_index, top_k_weights)
                if len(inputs) < 2:
                    return
                hidden_states = inputs[0]
                top_k_index = inputs[1]
                if not isinstance(hidden_states, torch.Tensor) or not isinstance(top_k_index, torch.Tensor):
                    return

                gate_up_proj = module.gate_up_proj  # (num_experts, 2*inter, hidden)
                down_proj = module.down_proj        # (num_experts, hidden, inter)
                num_experts = int(gate_up_proj.shape[0])
                intermediate_size = int(gate_up_proj.shape[1]) // 2
                hidden_size = int(gate_up_proj.shape[2])
                dtype = str(hidden_states.dtype)
                w_dtype = str(gate_up_proj.dtype)

                with torch.no_grad():
                    flat_indices = top_k_index.reshape(-1).cpu().to(torch.int64)
                    counts = torch.bincount(flat_indices, minlength=num_experts)

                for expert_id, n_tokens in enumerate(counts.tolist()):
                    n_tokens = int(n_tokens)
                    if n_tokens == 0:
                        continue
                    base = f"{module_name}.{expert_id}"

                    for proj, in_size, out_size in (
                        ("gate_proj", hidden_size, intermediate_size),
                        ("up_proj",   hidden_size, intermediate_size),
                        ("down_proj", intermediate_size, hidden_size),
                    ):
                        proj_module = f"{base}.{proj}"
                        in_n  = n_tokens * in_size
                        w_n   = out_size * in_size
                        out_n = n_tokens * out_size
                        macs  = n_tokens * in_size * out_size
                        denom = in_n + w_n + out_n
                        ai    = macs / denom if denom else 0.0

                        self.dense_layers.append({
                            "name":                 proj_module,
                            "shape":                (out_size, in_size),
                            "macs":                 macs,
                            "input_numels":         in_n,
                            "output_numels":        out_n,
                            "weight_numels":        w_n,
                            "arithmetic_intensity": ai,
                        })
                        self.current_operations.append({
                            "op_type":   "linear",
                            "op_family": "static_weight_matmul",
                            "module":    proj_module,
                            "role":      classify_static_matrix(proj_module),
                            "inputs": [{"name": "input",  "shape": (n_tokens, in_size),  "dtype": dtype,   "kind": "activation", "static": False, "numels": in_n}],
                            "weights":[{"parameter_ref":  f"{proj_module}.weight", "shape": (out_size, in_size), "dtype": w_dtype, "kind": "parameter", "static": True,  "numels": w_n}],
                            "outputs":[{"name": "output", "shape": (n_tokens, out_size), "dtype": dtype,   "kind": "activation", "static": False, "numels": out_n}],
                            "math": {"macs": macs, "flops_estimate": macs * 2, "arithmetic_intensity": ai},
                        })
            return hook

        def moe_block_hook(module_name: str, module: nn.Module):
            def hook(_: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
                if not self.enabled or not self.prompt_active or self._suppress_current_pass:
                    return
                hidden_states = next(
                    (t for t in iter_tensors(inputs) if t.is_floating_point() and t.dim() >= 2),
                    None,
                )
                if hidden_states is None:
                    return
                n_tokens = hidden_states.numel() // hidden_states.shape[-1]
                hidden_size = int(hidden_states.shape[-1])
                experts_sub = getattr(module, "experts", None)
                gate_sub = getattr(module, "gate", None)
                num_experts = int(
                    getattr(module, "num_experts", None)
                    or getattr(experts_sub, "num_experts", None)
                    or getattr(gate_sub, "num_experts", None)
                    or (len(experts_sub) if isinstance(experts_sub, (list, nn.ModuleList)) else 0)
                    or 0
                )
                top_k = int(
                    getattr(module, "top_k", None)
                    or getattr(gate_sub, "top_k", None)
                    or getattr(module, "num_experts_per_tok", None)
                    or 2
                )
                if num_experts == 0:
                    return
                dtype = str(hidden_states.dtype)

                gate_n = n_tokens * num_experts
                gate_flops = 3 * gate_n
                gate_numels = gate_n * 2
                gate_ai = gate_flops / gate_numels if gate_numels else 0.0

                topk_flops = n_tokens * num_experts
                topk_numels = gate_n
                topk_ai = topk_flops / topk_numels if topk_numels else 0.0

                scatter_n = n_tokens * top_k * hidden_size
                scatter_numels = scatter_n * 2

                self.current_operations.extend([
                    {
                        "op_type": "gate_softmax",
                        "op_family": "moe_routing",
                        "module": module_name,
                        "role": "moe gate softmax",
                        "inputs": [{"name": "logits", "shape": (n_tokens, num_experts), "dtype": dtype, "numels": gate_n}],
                        "outputs": [{"name": "weights", "shape": (n_tokens, num_experts), "dtype": dtype, "numels": gate_n}],
                        "element_count": gate_n,
                        "math": {"macs": gate_flops // 2, "flops_estimate": gate_flops, "arithmetic_intensity": gate_ai},
                    },
                    {
                        "op_type": "top_k",
                        "op_family": "moe_routing",
                        "module": module_name,
                        "role": "moe top-k expert selection",
                        "inputs": [{"name": "weights", "shape": (n_tokens, num_experts), "dtype": dtype, "numels": gate_n}],
                        "outputs": [{"name": "selected", "shape": (n_tokens, top_k), "dtype": dtype, "numels": n_tokens * top_k}],
                        "element_count": gate_n,
                        "math": {"macs": topk_flops // 2, "flops_estimate": topk_flops, "arithmetic_intensity": topk_ai},
                    },
                    {
                        "op_type": "token_dispatch",
                        "op_family": "moe_routing",
                        "module": module_name,
                        "role": "moe token scatter/gather",
                        "inputs": [{"name": "tokens", "shape": (n_tokens, hidden_size), "dtype": dtype, "numels": n_tokens * hidden_size}],
                        "outputs": [{"name": "dispatched", "shape": (n_tokens, top_k, hidden_size), "dtype": dtype, "numels": scatter_n}],
                        "element_count": scatter_n,
                        "math": {"macs": scatter_n, "flops_estimate": scatter_n, "arithmetic_intensity": 1.0},
                    },
                ])
            return hook

        def decode_pass_pre_hook(_module: nn.Module, _args: tuple) -> None:
            if not self.enabled or not self.prompt_active or self.current_phase != "decode":
                return
            self._decode_pass_count += 1
            self._suppress_current_pass = self._decode_pass_count == 1

        self.handles.append(self.model.register_forward_pre_hook(decode_pass_pre_hook))

        for module_name, module in self.model.named_modules():
            if module_name and len(list(module.children())) == 0:
                self.handles.append(module.register_forward_hook(activation_memory_hook))
            if isinstance(module, nn.Linear):
                self.handles.append(module.register_forward_hook(linear_hook(module_name, module)))
            if is_activation_module(module) or isinstance(module, HookableActivation):
                self.handles.append(module.register_forward_hook(sparsity_hook(module_name)))
            if isinstance(module, nn.LayerNorm) or is_rms_norm_module(module):
                self.handles.append(module.register_forward_hook(layernorm_hook(module_name)))
            if isinstance(module, nn.Softmax):
                self.handles.append(module.register_forward_hook(softmax_hook(module_name)))
            if is_attention_matrix_module(module):
                self.handles.append(module.register_forward_hook(attention_hook(module_name, module)))
            if is_rope_module(module):
                self.handles.append(module.register_forward_hook(rope_hook(module_name)))
            if is_gated_mlp_module(module):
                self.handles.append(module.register_forward_hook(gated_mlp_hook(module_name, module)))
            if is_transformer_block_module(module):
                self.handles.append(module.register_forward_hook(transformer_block_hook(module_name)))
            if is_moe_block_module(module):
                self.handles.append(module.register_forward_hook(moe_block_hook(module_name, module)))
            if is_olmoe_router_module(module):
                self.handles.append(module.register_forward_hook(olmoe_router_hook(module_name, module)))
            if is_olmoe_experts_module(module):
                self.handles.append(module.register_forward_hook(olmoe_experts_hook(module_name, module)))

    def remove_hooks(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def start_prompt(
        self,
        prompt_index: int | None = None,
        batch: Mapping[str, torch.Tensor] | None = None,
        phase: str = "prefill",
    ) -> None:
        self.prompt_active = True
        self.current_phase = phase
        self.current_prompt_index = prompt_index
        self._decode_pass_count = 0
        self._suppress_current_pass = False
        self.current_activation_numels = 0
        self.current_seen_storages.clear()
        self.current_operations = []
        self.current_input_shapes = {}
        self.current_input_dtypes = {}
        if batch is not None:
            self.current_input_shapes = {
                name: tuple(tensor.shape)
                for name, tensor in batch.items()
                if isinstance(tensor, torch.Tensor)
            }
            self.current_input_dtypes = {
                name: str(tensor.dtype)
                for name, tensor in batch.items()
                if isinstance(tensor, torch.Tensor)
            }

    def finish_prompt(self) -> None:
        if self.prompt_active:
            self.total_activation_numels += self.current_activation_numels
            self.prompt_count += 1
            operation_counts = Counter(
                str(operation.get("op_type", "unknown"))
                for operation in self.current_operations
            )
            static_weight_macs = sum(
                int(operation.get("math", {}).get("macs", 0))
                for operation in self.current_operations
                if operation.get("op_family") == "static_weight_matmul"
            )
            dynamic_activation_macs = sum(
                int(operation.get("math", {}).get("macs", 0))
                for operation in self.current_operations
                if operation.get("op_family") == "dynamic_activation_matmul"
            )
            nonlinear_element_ops = sum(
                int(operation.get("element_count", 0))
                for operation in self.current_operations
                if operation.get("op_family") == "elementwise"
            )
            reduction_ops = sum(
                1 for operation in self.current_operations
                if operation.get("op_family") == "reduction"
            )
            moe_routing_ops = sum(
                1 for operation in self.current_operations
                if operation.get("op_family") == "moe_routing"
            )
            rope_ops = sum(
                1 for operation in self.current_operations
                if operation.get("op_family") == "rope"
            )
            operations = [
                {"event_id": event_id, **operation}
                for event_id, operation in enumerate(self.current_operations)
            ]
            self.inference_traces.append(
                {
                    "inference_id": len(self.inference_traces),
                    "prompt_index": self.current_prompt_index,
                    "phase": self.current_phase,
                    "input_shape": self.current_input_shapes,
                    "input_dtypes": self.current_input_dtypes,
                    "operations": operations,
                    "summary": {
                        "total_ops": len(operations),
                        "operation_counts": dict(operation_counts),
                        "linear_ops": operation_counts.get("linear", 0),
                        "dynamic_matmul_ops": operation_counts.get("matmul", 0),
                        "activation_ops": operation_counts.get("activation", 0),
                        "static_weight_macs": static_weight_macs,
                        "dynamic_activation_macs": dynamic_activation_macs,
                        "nonlinear_element_ops": nonlinear_element_ops,
                        "reduction_ops": reduction_ops,
                        "moe_routing_ops": moe_routing_ops,
                        "rope_ops": rope_ops,
                        "activation_numels": self.current_activation_numels,
                    },
                }
            )
        self.prompt_active = False

    def analyze_flops_for_prompt(self, batch: Mapping[str, torch.Tensor]) -> None:
        configure_fvcore_logging()
        fvcore_inputs = build_fvcore_inputs(batch)
        flops = FlopCountAnalysis(self.model, fvcore_inputs)
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
            for module_name, module in self.model.named_modules()
            if isinstance(module, nn.Linear)
        ]
        linear_macs = float(
            sum(
                sum(by_module_and_operator.get(name, Counter()).values())
                for name in linear_module_names
            )
        )
        nonlinear_macs = max(total_macs - linear_macs, 0.0)
        dynamic_macs = float(sum(by_operator.get(op, 0.0) for op in DYNAMIC_MATMUL_OPS))
        static_weight_macs = linear_macs

        self.total_macs += total_macs
        self.linear_macs += linear_macs
        self.nonlinear_macs += nonlinear_macs
        self.dynamic_macs += dynamic_macs
        self.static_weight_macs += static_weight_macs
        self.by_operator_accumulator.update(by_operator)

    def build_report(self) -> dict[str, Any]:
        average_activation_numels = (
            self.total_activation_numels / self.prompt_count if self.prompt_count else 0.0
        )
        average_total_macs = self.total_macs / self.prompt_count if self.prompt_count else 0.0
        system_ai = (
            average_total_macs / (self.param_numels + average_activation_numels)
            if (self.param_numels + average_activation_numels) > 0
            else 0.0
        )
        average_crossbar_ai = (
            sum(layer["arithmetic_intensity"] for layer in self.dense_layers) / len(self.dense_layers)
            if self.dense_layers
            else 0.0
        )
        linear_share = self.linear_macs / self.total_macs if self.total_macs else 0.0
        nonlinear_share = self.nonlinear_macs / self.total_macs if self.total_macs else 0.0
        static_plus_dynamic = self.static_weight_macs + self.dynamic_macs
        static_share = self.static_weight_macs / static_plus_dynamic if static_plus_dynamic else 0.0
        dynamic_share = self.dynamic_macs / static_plus_dynamic if static_plus_dynamic else 0.0
        activation_sparsity = (
            self.activation_zero_count / self.activation_value_count
            if self.activation_value_count
            else 0.0
        )

        return {
            "prompt_count": self.prompt_count,
            "param_numels": self.param_numels,
            "average_activation_numels": average_activation_numels,
            "average_total_macs": average_total_macs,
            "system_arithmetic_intensity": system_ai,
            "average_crossbar_arithmetic_intensity": average_crossbar_ai,
            "linear_macs": self.linear_macs,
            "nonlinear_macs": self.nonlinear_macs,
            "linear_share": linear_share,
            "nonlinear_share": nonlinear_share,
            "static_weight_macs": self.static_weight_macs,
            "dynamic_macs": self.dynamic_macs,
            "static_share": static_share,
            "dynamic_share": dynamic_share,
            "activation_sparsity": activation_sparsity,
            "activation_layer_count": len(self.activation_layer_names),
            "wrapped_activation_count": len(self.wrapped_activations),
            "dynamic_matrix_summary": self._summarize_dynamic_matrices(),
            "operator_breakdown": dict(self.by_operator_accumulator),
        }

    def _summarize_dynamic_matrices(self) -> list[dict[str, Any]]:
        """Group repeated dynamic tensor products across layers and prompts."""

        grouped_ops: dict[
            tuple[str, str, str, tuple[int, ...], tuple[int, ...], tuple[int, ...]],
            dict[str, Any],
        ] = {}

        for op in self.dynamic_ops:
            key = (
                op["module"],
                op["kind"],
                op["role"],
                op["lhs_shape"],
                op["rhs_shape"],
                op["output_shape"],
            )
            if key not in grouped_ops:
                grouped_ops[key] = {
                    "module": op["module"],
                    "kind": op["kind"],
                    "role": op["role"],
                    "lhs_shape": op["lhs_shape"],
                    "rhs_shape": op["rhs_shape"],
                    "output_shape": op["output_shape"],
                    "macs": op["macs"],
                    "instances": 1,
                }
                continue
            grouped_ops[key]["instances"] += 1

        return sorted(
            grouped_ops.values(),
            key=lambda item: (item["module"], item["kind"], item["lhs_shape"]),
        )

    def print_report(self) -> None:
        metrics = self.build_report()
        print("=" * 96)
        print("Dynamic AIMC Runtime Report")
        print("=" * 96)
        print(f"Prompts profiled              : {metrics['prompt_count']}")
        print(f"Parameter numels             : {metrics['param_numels']:,}")
        print(f"Average activation numels    : {metrics['average_activation_numels']:.0f}")
        print(
            f"System arithmetic intensity  : "
            f"{metrics['system_arithmetic_intensity']:.6f} MACs/numel"
        )
        print(
            f"Crossbar arithmetic intensity: "
            f"{metrics['average_crossbar_arithmetic_intensity']:.6f} MACs/numel"
        )
        print(f"Linear MAC share             : {metrics['linear_share'] * 100:.2f}%")
        print(f"Non-linear MAC share         : {metrics['nonlinear_share'] * 100:.2f}%")
        print(f"Static MAC share             : {metrics['static_share'] * 100:.2f}%")
        print(f"Dynamic MAC share            : {metrics['dynamic_share'] * 100:.2f}%")
        print(f"Activation sparsity          : {metrics['activation_sparsity'] * 100:.2f}%")
        print(f"Activation layers probed     : {metrics['activation_layer_count']}")
        print(f"Wrapped activations          : {metrics['wrapped_activation_count']}")
        print("-" * 96)
        print("Dynamic matrices")
        for op in metrics["dynamic_matrix_summary"]:
            print(
                f"[DYNAMIC] {op['module']} | op={op['kind']} | role={op['role']} | "
                f"lhs={op['lhs_shape']} | rhs={op['rhs_shape']} | "
                f"out={op['output_shape']} | MACs={int(op['macs']):,} | "
                f"instances={op['instances']}"
            )
        print("-" * 96)
        print("Operator MAC breakdown")
        for op_name, op_macs in sorted(
            metrics["operator_breakdown"].items(), key=lambda item: item[1], reverse=True
        ):
            print(f"{op_name:<32} {int(op_macs):,}")
        print("=" * 96)
