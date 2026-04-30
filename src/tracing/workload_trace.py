from __future__ import annotations

import pickle
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import transformers

from src.metrics.crossbar_tiling_analyzer import (
    calculate_tiling_efficiency,
    classify_static_matrix,
)
from src.metrics.expert_routing_tracker import ExpertRoutingTracker
from src.metrics.runtime_aimc_tracker import RuntimeAIMCTracker


TRACE_SCHEMA_VERSION = 1


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def module_class_inventory(model: nn.Module) -> dict[str, dict[str, Any]]:
    modules: dict[str, dict[str, Any]] = {}
    module_parameter_refs: dict[str, list[str]] = {}

    for parameter_name, _ in model.named_parameters():
        module_name, _, _ = parameter_name.rpartition(".")
        module_parameter_refs.setdefault(module_name, []).append(parameter_name)

    for module_name, module in model.named_modules():
        if module_name == "":
            continue
        modules[module_name] = {
            "class": module.__class__.__name__,
            "role": classify_static_matrix(module_name)
            if isinstance(module, nn.Linear)
            else "module",
            "parameter_refs": module_parameter_refs.get(module_name, []),
        }

    return modules


def collect_parameter_inventory(model: nn.Module) -> dict[str, dict[str, Any]]:
    parameters: dict[str, dict[str, Any]] = {}
    module_by_parameter = {
        parameter_name: parameter_name.rpartition(".")[0]
        for parameter_name, _ in model.named_parameters()
    }

    for parameter_name, parameter in model.named_parameters():
        module_name = module_by_parameter[parameter_name]
        parameters[parameter_name] = {
            "kind": "parameter",
            "module": module_name,
            "tensor_name": parameter_name.rpartition(".")[2],
            "shape": tuple(parameter.shape),
            "dtype": str(parameter.dtype),
            "numel": parameter.numel(),
            "bytes": tensor_nbytes(parameter),
            "requires_grad": bool(parameter.requires_grad),
            "trainable": bool(parameter.requires_grad),
            "static_during_inference": True,
            "role": classify_static_matrix(module_name),
        }

    return parameters


def collect_buffer_inventory(model: nn.Module) -> dict[str, dict[str, Any]]:
    buffers: dict[str, dict[str, Any]] = {}

    for module_name, module in model.named_modules():
        for local_name, buffer in module.named_buffers(recurse=False):
            buffer_name = f"{module_name}.{local_name}" if module_name else local_name
            buffers[buffer_name] = {
                "kind": "buffer",
                "module": module_name,
                "tensor_name": local_name,
                "shape": tuple(buffer.shape),
                "dtype": str(buffer.dtype),
                "numel": buffer.numel(),
                "bytes": tensor_nbytes(buffer),
                "persistent": local_name not in module._non_persistent_buffers_set,
                "static_during_inference": True,
            }

    return buffers


def build_static_matrix_inventory(
    model: nn.Module,
    crossbar_size: tuple[int, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tiling_metrics = calculate_tiling_efficiency(model, crossbar_size)
    matrices: list[dict[str, Any]] = []

    module_map = dict(model.named_modules())
    for matrix in tiling_metrics["matrices"]:
        module_name = matrix["name"]
        module = module_map[module_name]
        weight = module.weight
        rows, cols = matrix["shape"]
        matrices.append(
            {
                "matrix_id": f"{module_name}.weight",
                "module": module_name,
                "role": matrix["role"],
                "shape": matrix["shape"],
                "dtype": str(weight.dtype),
                "rows": rows,
                "cols": cols,
                "static_during_inference": True,
                "crossbar": {
                    "tile_shape": crossbar_size,
                    "tiles": matrix["tiles"],
                    "used_cells": matrix["used_cells"],
                    "provisioned_cells": matrix["provisioned_cells"],
                    "tiling_efficiency": matrix["tiling_efficiency"],
                },
            }
        )

    return matrices, tiling_metrics


def serialize_routing_trace(
    expert_routing_tracker: ExpertRoutingTracker | None,
) -> dict[int, list[dict[str, Any]]]:
    if expert_routing_tracker is None:
        return {}

    serialized: dict[int, list[dict[str, Any]]] = {}
    for prompt_index, records in expert_routing_tracker.routing_trace.items():
        serialized[prompt_index] = [
            asdict(record) if is_dataclass(record) else dict(record)
            for record in records
        ]
    return serialized


def attach_routing_to_inferences(
    inferences: list[dict[str, Any]],
    routing_trace: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for inference in inferences:
        copied = dict(inference)
        prompt_index = copied.get("prompt_index")
        copied["routing"] = routing_trace.get(prompt_index, [])
        enriched.append(copied)
    return enriched


def build_trace_summary(
    parameters: dict[str, dict[str, Any]],
    static_matrices: list[dict[str, Any]],
    inferences: list[dict[str, Any]],
    tiling_metrics: dict[str, Any],
) -> dict[str, Any]:
    operator_breakdown: dict[str, int] = {}
    total_runtime_ops = 0
    total_static_weight_macs = 0
    total_dynamic_activation_macs = 0
    total_nonlinear_element_ops = 0

    for inference in inferences:
        summary = inference.get("summary", {})
        total_runtime_ops += int(summary.get("total_ops", 0))
        total_static_weight_macs += int(summary.get("static_weight_macs", 0))
        total_dynamic_activation_macs += int(summary.get("dynamic_activation_macs", 0))
        total_nonlinear_element_ops += int(summary.get("nonlinear_element_ops", 0))
        for op_name, count in summary.get("operation_counts", {}).items():
            operator_breakdown[op_name] = operator_breakdown.get(op_name, 0) + int(count)

    return {
        "num_inferences": len(inferences),
        "total_parameters": len(parameters),
        "total_parameter_bytes": sum(item["bytes"] for item in parameters.values()),
        "total_static_matrices": len(static_matrices),
        "total_runtime_ops": total_runtime_ops,
        "total_static_weight_macs": total_static_weight_macs,
        "total_dynamic_activation_macs": total_dynamic_activation_macs,
        "total_nonlinear_element_ops": total_nonlinear_element_ops,
        "operator_breakdown": operator_breakdown,
        "crossbar": {
            "tile_shape": tiling_metrics["crossbar_size"],
            "total_tiles": tiling_metrics["total_tiles"],
            "used_cells": tiling_metrics["used_cells"],
            "provisioned_cells": tiling_metrics["provisioned_cells"],
            "wasted_cells": tiling_metrics["wasted_cells"],
            "tiling_efficiency": tiling_metrics["tiling_efficiency"],
        },
    }


def export_workload_trace(
    *,
    output_path: Path,
    model: nn.Module,
    model_info: dict[str, Any],
    config_dict: dict[str, Any],
    runtime_aimc_tracker: RuntimeAIMCTracker | None,
    expert_routing_tracker: ExpertRoutingTracker | None,
    total_input_tokens: int,
    total_output_tokens: int,
) -> Path:
    metrics_cfg = config_dict.get("metrics", {})
    crossbar_size = tuple(metrics_cfg.get("crossbar_dimensions", (128, 128)))
    parameters = collect_parameter_inventory(model)
    buffers = collect_buffer_inventory(model)
    static_matrices, tiling_metrics = build_static_matrix_inventory(model, crossbar_size)
    routing_trace = serialize_routing_trace(expert_routing_tracker)
    inferences = (
        list(runtime_aimc_tracker.inference_traces)
        if runtime_aimc_tracker is not None
        else []
    )
    enriched_inferences = attach_routing_to_inferences(inferences, routing_trace)

    payload = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "metadata": {
            "model_id": model_info["model_id"],
            "model_class": model.__class__.__name__,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "device": model_info.get("execution_device", "unknown"),
            "requested_precision": config_dict.get("model", {}).get("precision"),
            "eval_mode": not model.training,
            "use_cache": bool(getattr(getattr(model, "config", None), "use_cache", False)),
            "configured_top_k": model_info.get("configured_top_k"),
            "original_top_k": model_info.get("original_top_k"),
            "routed_experts": model_info.get("routed_experts"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "prompts_processed": len(config_dict.get("dataset", [])),
            "input_token_count": total_input_tokens,
            "output_token_count": total_output_tokens,
        },
        "model": {
            "parameters": parameters,
            "buffers": buffers,
            "modules": module_class_inventory(model),
            "static_matrix_inventory": static_matrices,
        },
        "inferences": enriched_inferences,
        "routing_trace": routing_trace,
        "summary": build_trace_summary(
            parameters,
            static_matrices,
            enriched_inferences,
            tiling_metrics,
        ),
    }

    with output_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return output_path
