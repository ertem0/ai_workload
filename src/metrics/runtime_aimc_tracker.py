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


def is_attention_matrix_module(module: nn.Module) -> bool:
    """
    Identify attention modules that produce dynamic tensor products.

    MobileBERT-style modules typically expose:
    - num_attention_heads
    - attention_head_size
    - query, key, value

    Qwen-style modules typically expose:
    - num_heads
    - head_dim
    - q_proj, k_proj, v_proj
    """

    mobilebert_attrs = ("num_attention_heads", "attention_head_size", "query", "key", "value")
    qwen_attrs = ("num_heads", "head_dim", "q_proj", "k_proj", "v_proj")
    return all(hasattr(module, attr) for attr in mobilebert_attrs) or all(
        hasattr(module, attr) for attr in qwen_attrs)


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

        self.param_bytes = sum(tensor_nbytes(parameter) for parameter in model.parameters())
        self.wrapped_activations = self._wrap_functional_activations()

        self.current_activation_bytes = 0
        self.current_seen_storages: set[tuple[int, int, int]] = set()
        self.current_prompt_index: int | None = None
        self.current_input_shapes: dict[str, tuple[int, ...]] = {}
        self.current_input_dtypes: dict[str, str] = {}
        self.current_operations: list[dict[str, Any]] = []
        self.inference_traces: list[dict[str, Any]] = []

        self.total_activation_bytes = 0
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
            if not self.enabled or not self.prompt_active:
                return
            for tensor in iter_tensors(output):
                key = tensor_storage_key(tensor)
                if key in self.current_seen_storages:
                    continue
                self.current_seen_storages.add(key)
                self.current_activation_bytes += tensor_nbytes(tensor)

        def linear_hook(module_name: str, module: nn.Linear):
            def hook(_: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
                if not self.enabled or not self.prompt_active:
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
                input_bytes = tensor_nbytes(input_tensor)
                output_bytes = tensor_nbytes(output_tensor)
                weight_bytes = tensor_nbytes(module.weight)
                denominator = input_bytes + output_bytes + weight_bytes
                arithmetic_intensity = layer_macs / denominator if denominator else 0.0

                self.dense_layers.append(
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
                                "bytes": input_bytes,
                            }
                        ],
                        "weights": [
                            {
                                "parameter_ref": f"{module_name}.weight",
                                "shape": tuple(module.weight.shape),
                                "dtype": str(module.weight.dtype),
                                "kind": "parameter",
                                "static": True,
                                "bytes": weight_bytes,
                            }
                        ],
                        "outputs": [
                            {
                                "name": "output",
                                "shape": tuple(output_tensor.shape),
                                "dtype": str(output_tensor.dtype),
                                "kind": "activation",
                                "static": False,
                                "bytes": output_bytes,
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
                if not self.enabled or not self.prompt_active:
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
                    self.current_operations.append(
                        {
                            "op_type": "activation",
                            "op_family": "elementwise",
                            "module": module_name,
                            "role": "nonlinear activation",
                            "activation": getattr(module, "label", module.__class__.__name__),
                            "input_shape": input_shape,
                            "output_shape": tuple(tensor.shape),
                            "dtype": str(tensor.dtype),
                            "element_count": tensor.numel(),
                            "static": False,
                        }
                    )

            return hook

        def attention_hook(module_name: str, module: nn.Module):
            def hook(_: nn.Module, inputs: tuple[Any, ...], __: Any) -> None:
                if not self.enabled or not self.prompt_active:
                    return

                input_tensors = [tensor for tensor in iter_tensors(inputs) if tensor.dim() == 3]
                if not input_tensors:
                    return

                hidden_states = input_tensors[0]
                batch_size = int(hidden_states.shape[0])
                query_len = int(hidden_states.shape[1])
                key_len = query_len
                num_heads = int(
                    getattr(module, "num_heads", getattr(module, "num_attention_heads", 0))
                )
                head_dim = int(
                    getattr(module, "head_dim", getattr(module, "attention_head_size", 0))
                )
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

        activation_types = (nn.ReLU, nn.GELU, nn.SiLU, HookableActivation)

        for module_name, module in self.model.named_modules():
            if module_name and len(list(module.children())) == 0:
                self.handles.append(module.register_forward_hook(activation_memory_hook))
            if isinstance(module, nn.Linear):
                self.handles.append(module.register_forward_hook(linear_hook(module_name, module)))
            if isinstance(module, activation_types):
                self.handles.append(module.register_forward_hook(sparsity_hook(module_name)))
            if is_attention_matrix_module(module):
                self.handles.append(module.register_forward_hook(attention_hook(module_name, module)))

    def remove_hooks(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def start_prompt(
        self,
        prompt_index: int | None = None,
        batch: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        self.prompt_active = True
        self.current_prompt_index = prompt_index
        self.current_activation_bytes = 0
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
            self.total_activation_bytes += self.current_activation_bytes
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
            operations = [
                {"event_id": event_id, **operation}
                for event_id, operation in enumerate(self.current_operations)
            ]
            self.inference_traces.append(
                {
                    "inference_id": len(self.inference_traces),
                    "prompt_index": self.current_prompt_index,
                    "phase": "prefill",
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
                        "activation_bytes": self.current_activation_bytes,
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
        average_activation_bytes = (
            self.total_activation_bytes / self.prompt_count if self.prompt_count else 0.0
        )
        average_total_macs = self.total_macs / self.prompt_count if self.prompt_count else 0.0
        system_ai = (
            average_total_macs / (self.param_bytes + average_activation_bytes)
            if (self.param_bytes + average_activation_bytes) > 0
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
            "param_bytes": self.param_bytes,
            "average_activation_bytes": average_activation_bytes,
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
        print(f"Parameter bytes              : {metrics['param_bytes']:,}")
        print(f"Average activation bytes     : {metrics['average_activation_bytes']:.0f}")
        print(
            f"System arithmetic intensity  : "
            f"{metrics['system_arithmetic_intensity']:.6f} MACs/byte"
        )
        print(
            f"Crossbar arithmetic intensity: "
            f"{metrics['average_crossbar_arithmetic_intensity']:.6f} MACs/byte"
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
