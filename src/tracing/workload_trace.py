from __future__ import annotations

import pickle
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


TRACE_SCHEMA_VERSION = 2


def _is_olmoe_experts_module(module: nn.Module) -> bool:
    """Detect OlmoeExperts-style batched expert blocks (3D Parameter tensors, not nn.Linear)."""
    gate_up = getattr(module, "gate_up_proj", None)
    down = getattr(module, "down_proj", None)
    return (
        isinstance(gate_up, torch.nn.Parameter) and gate_up.dim() == 3
        and isinstance(down, torch.nn.Parameter) and down.dim() == 3
    )


def _is_olmoe_router_module(module: nn.Module) -> bool:
    """Detect OlmoeTopKRouter-style modules (2D Parameter weight, not nn.Linear)."""
    if isinstance(module, nn.Linear):
        return False
    weight = getattr(module, "weight", None)
    return (
        isinstance(weight, torch.nn.Parameter)
        and weight.dim() == 2
        and hasattr(module, "top_k")
        and hasattr(module, "num_experts")
    )


# ─── Static inventory ────────────────────────────────────────────────────────

def build_weights_inventory(
    model: nn.Module,
    crossbar_size: tuple[int, int],
) -> dict[str, dict[str, Any]]:
    """One entry per weight matrix (nn.Linear or OLMoE batched expert) with shape, role and tiling."""
    tiling_metrics = calculate_tiling_efficiency(model, crossbar_size)
    tile_map = {m["name"]: m for m in tiling_metrics["matrices"]}

    weights: dict[str, dict[str, Any]] = {}
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            w = module.weight
            t = tile_map.get(module_name, {})
            weights[module_name] = {
                "shape":             list(w.shape),
                "dtype":             str(w.dtype),
                "numels":            w.numel(),
                "role":              classify_static_matrix(module_name),
                "tiles":             t.get("tiles"),
                "used_cells":        t.get("used_cells"),
                "provisioned_cells": t.get("provisioned_cells"),
                "tile_efficiency":   t.get("tiling_efficiency"),
            }
        elif _is_olmoe_router_module(module):
            w = module.weight  # (num_experts, hidden_dim)
            weights[module_name] = {
                "shape":             list(w.shape),
                "dtype":             str(w.dtype),
                "numels":            w.numel(),
                "role":              classify_static_matrix(module_name),
                "tiles":             None,
                "used_cells":        None,
                "provisioned_cells": None,
                "tile_efficiency":   None,
            }
        elif _is_olmoe_experts_module(module):
            # OlmoeExperts stores weights as 3D Parameters; emit one virtual entry
            # per expert per projection so the trace matches how ops are recorded.
            gate_up = module.gate_up_proj  # (num_experts, 2*inter, hidden)
            down = module.down_proj        # (num_experts, hidden, inter)
            num_experts = int(gate_up.shape[0])
            intermediate_size = int(gate_up.shape[1]) // 2
            hidden_size = int(gate_up.shape[2])
            dtype = str(gate_up.dtype)
            for expert_id in range(num_experts):
                for proj, shape in (
                    ("gate_proj", (intermediate_size, hidden_size)),
                    ("up_proj",   (intermediate_size, hidden_size)),
                    ("down_proj", (hidden_size, intermediate_size)),
                ):
                    vname = f"{module_name}.{expert_id}.{proj}"
                    weights[vname] = {
                        "shape":             list(shape),
                        "dtype":             dtype,
                        "numels":            shape[0] * shape[1],
                        "role":              "expert_ffn",
                        "tiles":             None,
                        "used_cells":        None,
                        "provisioned_cells": None,
                        "tile_efficiency":   None,
                    }
    return weights


# ─── Op flattening ───────────────────────────────────────────────────────────

def flatten_ops(inferences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert nested inference_traces into a flat op list.

    Each op gets:
      seq_id        — which input sequence (prompt_index or inference_id)
      phase         — "prefill" or "decode"
      input_numels  — activation input element count (weight numels are in weights{})
      weight_numels — weight element count for this call (fixed per module)
      output_numels — output activation element count
    """
    ops: list[dict[str, Any]] = []
    for inference in inferences:
        phase  = inference.get("phase", "prefill")
        seq_id = inference.get("prompt_index", inference.get("inference_id", 0))
        for op in inference.get("operations", []):
            if not isinstance(op, dict):
                continue
            math_info = op.get("math", {})
            raw_inputs  = op.get("inputs",  [])
            raw_weights = op.get("weights", [])
            raw_outputs = op.get("outputs", [])

            input_shape  = list(raw_inputs[0]["shape"])  if raw_inputs  and raw_inputs[0].get("shape")  else None
            output_shape = list(raw_outputs[0]["shape"]) if raw_outputs and raw_outputs[0].get("shape") else None

            input_numels  = sum(int(i.get("numels", 0)) for i in raw_inputs)
            weight_numels = sum(int(w.get("numels", 0)) for w in raw_weights)
            output_numels = sum(int(o.get("numels", 0)) for o in raw_outputs)

            if input_numels == 0 and output_numels == 0:
                continue

            ops.append({
                "seq_id":       seq_id,
                "phase":        phase,
                "module":       op.get("module", ""),
                "op_type":      op.get("op_type", "unknown"),
                "op_family":    op.get("op_family", ""),
                "role":         op.get("role", ""),
                "input_shape":  input_shape,
                "output_shape": output_shape,
                "flops":        int(math_info.get("flops_estimate", 0)),
                "input_numels":  input_numels,
                "weight_numels": weight_numels,
                "output_numels": output_numels,
            })
    return ops


# ─── Export ──────────────────────────────────────────────────────────────────

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
    metrics_cfg   = config_dict.get("metrics", {})
    crossbar_size = tuple(metrics_cfg.get("crossbar_dimensions", (128, 128)))

    inferences = (
        list(runtime_aimc_tracker.inference_traces)
        if runtime_aimc_tracker is not None
        else []
    )

    payload = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "metadata": {
            "model_id":      model_info["model_id"],
            "model_class":   model.__class__.__name__,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "device":        model_info.get("execution_device", "unknown"),
            "precision":     config_dict.get("model", {}).get("precision"),
            "created_at":    datetime.now(timezone.utc).isoformat(),
            "n_sequences":   len(config_dict.get("dataset", [])),
            "input_tokens":  total_input_tokens,
            "output_tokens": total_output_tokens,
            "top_k":         model_info.get("configured_top_k"),
            "n_experts":     model_info.get("routed_experts"),
        },
        "weights": build_weights_inventory(model, crossbar_size),
        "ops":     flatten_ops(inferences),
    }

    with output_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return output_path
