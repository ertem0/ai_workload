from __future__ import annotations

import math
import re
from typing import Any

import torch.nn as nn


def classify_static_matrix(module_name: str) -> str:
    lowered = module_name.lower()
    if lowered.endswith(".q_proj"):
        return "attention query projection"
    if lowered.endswith(".k_proj"):
        return "attention key projection"
    if lowered.endswith(".v_proj"):
        return "attention value projection"
    if lowered.endswith(".o_proj"):
        return "attention output projection"
    if lowered.endswith(".gate"):
        return "router projection"
    if ".shared_expert" in lowered:
        return "shared expert projection"
    if ".experts." in lowered:
        return "expert feed-forward projection"
    if ".mlp." in lowered:
        return "feed-forward projection"
    if "lm_head" in lowered:
        return "language modeling head"
    return "dense projection"


def collect_static_matrix_inventory(
    model: nn.Module, crossbar_size: tuple[int, int]
) -> list[dict[str, Any]]:
    tile_rows, tile_cols = crossbar_size
    matrices: list[dict[str, Any]] = []

    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        rows, cols = module.weight.shape
        row_tiles = math.ceil(rows / tile_rows)
        col_tiles = math.ceil(cols / tile_cols)
        tile_count = row_tiles * col_tiles
        provisioned_cells = tile_count * tile_rows * tile_cols
        used_cells = rows * cols

        matrices.append(
            {
                "name": module_name,
                "role": classify_static_matrix(module_name),
                "shape": tuple(module.weight.shape),
                "tiles": tile_count,
                "used_cells": used_cells,
                "provisioned_cells": provisioned_cells,
                "tiling_efficiency": used_cells / provisioned_cells if provisioned_cells else 0.0,
            }
        )

    return matrices


def calculate_tiling_efficiency(
    model: nn.Module, crossbar_size: tuple[int, int]
) -> dict[str, Any]:
    matrices = collect_static_matrix_inventory(model, crossbar_size)
    total_tiles = sum(matrix["tiles"] for matrix in matrices)
    used_cells = sum(matrix["used_cells"] for matrix in matrices)
    provisioned_cells = sum(matrix["provisioned_cells"] for matrix in matrices)
    wasted_cells = provisioned_cells - used_cells
    efficiency = used_cells / provisioned_cells if provisioned_cells else 0.0

    return {
        "crossbar_size": crossbar_size,
        "matrix_count": len(matrices),
        "total_tiles": total_tiles,
        "used_cells": used_cells,
        "provisioned_cells": provisioned_cells,
        "wasted_cells": wasted_cells,
        "tiling_efficiency": efficiency,
        "matrices": matrices,
    }


def compact_matrix_name(module_name: str) -> str:
    """
    Collapse repeated per-layer matrix paths into a wildcarded representation.

    Example:
    - `model.layers.0.mlp.gate` -> `model.layers.*.mlp.gate`
    """

    compacted = re.sub(r"(\.layers\.)\d+(\.|$)", r"\1*\2", module_name)
    compacted = re.sub(r"(\.h\.)\d+(\.|$)", r"\1*\2", compacted)
    compacted = re.sub(r"(\.block\.)\d+(\.|$)", r"\1*\2", compacted)
    compacted = re.sub(r"(\.blocks\.)\d+(\.|$)", r"\1*\2", compacted)
    compacted = re.sub(r"(\.decoder\.layers\.)\d+(\.|$)", r"\1*\2", compacted)
    compacted = re.sub(r"(\.encoder\.layer\.)\d+(\.|$)", r"\1*\2", compacted)
    return compacted


def summarize_static_matrices(
    matrices: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries: dict[tuple[str, tuple[int, int], str], dict[str, Any]] = {}

    for matrix in matrices:
        compact_name = compact_matrix_name(matrix["name"])
        key = (compact_name, matrix["shape"], matrix["role"])
        if key not in summaries:
            summaries[key] = {
                "name": compact_name,
                "shape": matrix["shape"],
                "role": matrix["role"],
                "instances": 0,
            }
        summaries[key]["instances"] += 1

    return sorted(
        summaries.values(),
        key=lambda item: (item["name"], item["shape"][0], item["shape"][1]),
    )


def print_tiling_report(metrics: dict[str, Any]) -> None:
    rows, cols = metrics["crossbar_size"]
    matrix_summaries = summarize_static_matrices(metrics["matrices"])
    print("=" * 96)
    print("Static AIMC Mapping Report")
    print("=" * 96)
    print(f"Crossbar size                : {rows} x {cols}")
    print(f"Static linear matrices       : {metrics['matrix_count']}")
    print(f"Total crossbar tiles         : {metrics['total_tiles']:,}")
    print(f"Provisioned cells            : {metrics['provisioned_cells']:,}")
    print(f"Used weight cells            : {metrics['used_cells']:,}")
    print(f"Wasted padded cells          : {metrics['wasted_cells']:,}")
    print(f"Tiling efficiency            : {metrics['tiling_efficiency'] * 100:.2f}%")
    print("-" * 96)
    print("Static matrices")
    for summary in matrix_summaries:
        rows_, cols_ = summary["shape"]
        instance_suffix = (
            f" | instances={summary['instances']}"
            if summary["instances"] > 1
            else ""
        )
        print(
            f"[STATIC] {summary['name']} | role={summary['role']} | "
            f"shape=({rows_}, {cols_}){instance_suffix}"
        )
    print("=" * 96)
