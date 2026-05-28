"""
MoE Workload Metrics Pipeline for AIMC Analysis.

Extracts 9 metrics from workload trace files to assess suitability of MoE
models for Analog In-Memory Computing on memristor crossbar tiles.

Usage (CLI):
    python -m src.metrics.workload_metrics trace.pkl [trace2.pkl ...] \\
        --output-dir metrics/ --tile-size 128 --peak-compute 100 --peak-bandwidth 2

Usage (API):
    from src.metrics.workload_metrics import run_all_metrics, DEFAULT_CONFIG
    results = run_all_metrics([Path("trace.pkl")], Path("metrics/"), config)

Metric 9 (precision sensitivity) requires a live model and must be called
separately via run_precision_sensitivity(), then saved with
write_precision_sensitivity().
"""
from __future__ import annotations

import json
import math
import pickle
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# ─── Config ──────────────────────────────────────────────────────────────────

DEFAULT_CONFIG: dict[str, Any] = {
    "tile_size": 128,
    "aimc_roofline_threshold": 10.0,
    "peak_compute_tops": 1.3,
    "peak_bandwidth_tbs": 0.0597,
    "precisions_to_test": ["fp16", "int8", "int4"],
}

# ─── Parsing helpers ─────────────────────────────────────────────────────────

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_EXPERT_RE = re.compile(r"(?:^|\.)experts\.(\d+)(?:\.|$)")


def parse_layer_id(module_name: str) -> int | None:
    """Extract transformer layer index from a module path (e.g. 'model.layers.3...' → 3)."""
    m = _LAYER_RE.search(module_name)
    return int(m.group(1)) if m else None


def parse_expert_id(module_name: str) -> int | None:
    """Extract expert index from a module path (e.g. '...experts.7.w1' → 7)."""
    m = _EXPERT_RE.search(module_name)
    return int(m.group(1)) if m else None


def parse_op_role(module_name: str) -> str:
    """Classify a module's functional role: 'attention' | 'expert_ffn' | 'gate' | 'lm_head' | 'other'."""
    lower = module_name.lower()
    if any(lower.endswith(s) for s in (".q_proj", ".k_proj", ".v_proj", ".o_proj")):
        return "attention"
    if ".experts." in lower:
        return "expert_ffn"
    if lower.endswith(".gate") or "router" in lower:
        return "gate"
    if "lm_head" in lower:
        return "lm_head"
    return "other"


def is_aimc_suitable(op: dict[str, Any]) -> bool:
    """Return True if the op multiplies a static weight matrix → eligible for AIMC crossbar."""
    return op.get("op_family") == "static_weight_matmul"


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_trace(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return pickle.load(fh)


def _collect_all_ops(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the flat ops list from a trace (v2) or flatten from nested inferences (v1)."""
    if "ops" in trace:
        return trace["ops"]
    # v1 fallback
    ops: list[dict[str, Any]] = []
    for inference in trace.get("inferences", []):
        phase  = inference.get("phase", "prefill")
        inf_id = inference.get("inference_id")
        for op in inference.get("operations", []):
            if isinstance(op, dict):
                ops.append({**op, "_phase": phase, "_inference_id": inf_id})
    return ops


# ─── Op-level accessors ───────────────────────────────────────────────────────

def _op_flops(op: dict[str, Any]) -> int:
    if "flops" in op:
        return int(op["flops"])
    return int(op.get("math", {}).get("flops_estimate", 0))


def _op_total_numels(op: dict[str, Any]) -> int:
    # v2 flat fields
    if "input_numels" in op:
        return (int(op.get("input_numels", 0))
                + int(op.get("weight_numels", 0))
                + int(op.get("output_numels", 0)))
    # nested lists
    total = 0
    for lst in (op.get("inputs", []), op.get("weights", []), op.get("outputs", [])):
        if isinstance(lst, list):
            for item in lst:
                total += int(item.get("numels", 0))
    return total


def _op_arithmetic_intensity(op: dict[str, Any]) -> float:
    """Return arithmetic intensity in FLOP/numel."""
    flops = _op_flops(op)
    total_numels = _op_total_numels(op)
    if total_numels > 0:
        return flops / total_numels
    ai = op.get("math", {}).get("arithmetic_intensity")
    return float(ai) if ai is not None else 0.0


def _is_expert_op(op: dict[str, Any]) -> bool:
    return (
        op.get("op_family") == "static_weight_matmul"
        and ".experts." in (op.get("module") or "")
    )


def _op_input_tokens(op: dict[str, Any]) -> int:
    """Return the number of tokens (leading non-feature dims) from an op's first input."""
    shape = op.get("input_shape")           # v2
    if shape is None:
        inputs = op.get("inputs", [])       # v1
        if not inputs:
            return 0
        shape = inputs[0].get("shape", ())
    if not shape:
        return 0
    return math.prod(shape[:-1]) if len(shape) > 1 else int(shape[0])


# ─── Statistics helpers ───────────────────────────────────────────────────────

def _percentile(data: list[float], p: float) -> float:
    """Compute the p-th percentile (0–100) via linear interpolation."""
    if not data:
        return 0.0
    s = sorted(data)
    idx = (p / 100.0) * (len(s) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] * (1 - (idx - lo)) + s[hi] * (idx - lo)


def _histogram(data: list[float], n_bins: int = 20) -> dict[str, Any]:
    if not data:
        return {"bin_centers": [], "counts": [], "bin_edges": []}
    mn, mx = min(data), max(data)
    if mn == mx:
        return {"bin_centers": [mn], "counts": [len(data)], "bin_edges": [mn, mx]}
    width = (mx - mn) / n_bins
    counts = [0] * n_bins
    for v in data:
        counts[min(int((v - mn) / width), n_bins - 1)] += 1
    edges = [mn + i * width for i in range(n_bins + 1)]
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(n_bins)]
    return {"bin_centers": centers, "counts": counts, "bin_edges": edges}


def _stats(data: list[float]) -> dict[str, float | int]:
    if not data:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(data),
        "mean": statistics.mean(data),
        "p50": _percentile(data, 50),
        "p95": _percentile(data, 95),
        "min": min(data),
        "max": max(data),
    }


# ─── Metric 1: Arithmetic Intensity Distribution ──────────────────────────────

def compute_arithmetic_intensity(
    ops: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Histogram arithmetic intensity per op category with below-threshold flagging.

    Categories: attention_matmul, expert_matmul, gate, other_matmul, other.
    """
    threshold = float(config.get("aimc_roofline_threshold", DEFAULT_CONFIG["aimc_roofline_threshold"]))
    by_category: dict[str, list[float]] = defaultdict(list)

    for op in ops:
        ai = _op_arithmetic_intensity(op)
        if ai <= 0:
            continue
        family = op.get("op_family", "")
        module = op.get("module") or ""

        if family == "dynamic_activation_matmul":
            cat = "attention_matmul"
        elif family == "static_weight_matmul":
            role = parse_op_role(module)
            cat = {
                "attention": "attention_matmul",
                "expert_ffn": "expert_matmul",
                "gate": "gate",
            }.get(role, "other_matmul")
        elif family == "moe_routing":
            cat = "gate"
        else:
            cat = "other"

        by_category[cat].append(ai)

    categories: dict[str, Any] = {}
    for cat, values in by_category.items():
        below = sum(1 for v in values if v < threshold)
        categories[cat] = {
            "values": values,
            "histogram": _histogram(values),
            "stats": _stats(values),
            "below_threshold_count": below,
            "below_threshold_fraction": below / len(values) if values else 0.0,
        }

    return {
        "config": {"aimc_roofline_threshold": threshold},
        "categories": categories,
    }


# ─── Metric 2: Tile Mapping per Weight Matrix ─────────────────────────────────

def compute_tile_mapping(
    static_matrices: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute tiles_needed and tile_utilization for each static weight matrix.

    tiles_needed = ceil(rows/tile_size) * ceil(cols/tile_size)
    tile_utilization = (rows*cols) / (tiles_needed * tile_size^2)
    """
    tile_size = int(config.get("tile_size", DEFAULT_CONFIG["tile_size"]))
    per_op: list[dict[str, Any]] = []
    per_layer: dict[int, dict[str, Any]] = {}
    total_tiles = 0

    for mat in static_matrices:
        module = mat.get("module") or mat.get("matrix_id", "")
        rows = int(mat.get("rows", 0))
        cols = int(mat.get("cols", 0))
        if rows == 0 or cols == 0:
            shape = mat.get("shape", (0, 0))
            rows, cols = int(shape[0]), int(shape[1])
        if rows == 0 or cols == 0:
            continue

        row_tiles = math.ceil(rows / tile_size)
        col_tiles = math.ceil(cols / tile_size)
        tiles_needed = row_tiles * col_tiles
        provisioned = tiles_needed * tile_size * tile_size
        tile_utilization = (rows * cols) / provisioned if provisioned > 0 else 0.0
        layer_id = parse_layer_id(module)
        total_tiles += tiles_needed

        per_op.append({
            "module": module,
            "role": mat.get("role", parse_op_role(module)),
            "layer_id": layer_id,
            "shape": [rows, cols],
            "tiles_needed": tiles_needed,
            "tile_utilization": tile_utilization,
        })

        if layer_id is not None:
            if layer_id not in per_layer:
                per_layer[layer_id] = {"total_tiles": 0, "_util_sum": 0.0, "_count": 0, "modules": []}
            per_layer[layer_id]["total_tiles"] += tiles_needed
            per_layer[layer_id]["_util_sum"] += tile_utilization
            per_layer[layer_id]["_count"] += 1
            per_layer[layer_id]["modules"].append({
                "module": module,
                "tiles": tiles_needed,
                "utilization": tile_utilization,
            })

    per_layer_out: dict[str, Any] = {
        str(lid): {
            "total_tiles": v["total_tiles"],
            "avg_utilization": v["_util_sum"] / v["_count"] if v["_count"] else 0.0,
            "modules": v["modules"],
        }
        for lid, v in sorted(per_layer.items())
    }

    return {
        "config": {"tile_size": tile_size},
        "ops": per_op,
        "per_layer_budget": per_layer_out,
        "total_tiles": total_tiles,
    }


# ─── Metric 3: Load Imbalance per Layer ──────────────────────────────────────

def compute_load_imbalance(
    ops: list[dict[str, Any]],
    config: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Per-expert activation rate per layer.

    activation_rate = n_expert_activated / n_routings_layer
    where n_routings_layer = sum(all expert activations in layer) / top_k.

    Each expert op (w1/w2/w3) counts as one activation; duplicates across
    projections are deduplicated by tracking seen (seq_id, phase, expert_id)
    tuples so each routing decision is counted once.

    load_imbalance_ratio = max_rate / mean_rate across experts in a layer.
    """
    top_k = int((metadata or {}).get("top_k") or 1) or 1

    # (layer_id, seq_id, phase) → set of expert_ids seen — deduplicate w1/w2/w3
    seen: dict[tuple[int, int, str], set[int]] = defaultdict(set)

    for op in ops:
        if not _is_expert_op(op):
            continue
        module = op.get("module") or ""
        layer_id = parse_layer_id(module)
        expert_id = parse_expert_id(module)
        if layer_id is None or expert_id is None:
            continue
        seq_id = op.get("seq_id", 0)
        phase  = op.get("phase", "")
        seen[(layer_id, seq_id, phase)].add(expert_id)

    # Accumulate activation counts per (layer, expert)
    layer_expert_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for (layer_id, _seq, _phase), expert_ids in seen.items():
        for eid in expert_ids:
            layer_expert_counts[layer_id][eid] += 1

    per_layer: dict[str, Any] = {}
    for layer_id in sorted(layer_expert_counts):
        expert_map = layer_expert_counts[layer_id]
        counts = list(expert_map.values())
        if not counts:
            continue
        total_activations = sum(counts)
        n_routings = max(total_activations / top_k, 1)
        rates = {eid: c / n_routings for eid, c in expert_map.items()}
        rate_values = list(rates.values())
        mean_rate = statistics.mean(rate_values)
        max_rate  = max(rate_values)
        min_rate  = min(rate_values)
        imbalance = max_rate / mean_rate if mean_rate > 0 else 0.0
        per_layer[str(layer_id)] = {
            "expert_activation_rate": {str(eid): r for eid, r in sorted(rates.items())},
            "n_routings": n_routings,
            "mean_rate": mean_rate,
            "max_rate":  max_rate,
            "min_rate":  min_rate,
            "load_imbalance_ratio": imbalance,
        }

    return {"per_layer": per_layer, "top_k": top_k}


# ─── Metric 4: Digital vs Analog Operation Split ─────────────────────────────

def compute_digital_analog_split(
    ops: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Classify FLOPs as analog (static-weight matmuls) or digital (everything else).

    Produces overall ratio and per-layer stacked breakdown.
    """
    overall_analog = 0
    overall_digital = 0
    layer_data: dict[int, dict[str, int]] = defaultdict(lambda: {"analog": 0, "digital": 0})

    for op in ops:
        flops = _op_flops(op)
        module = op.get("module") or ""
        layer_id = parse_layer_id(module)
        if is_aimc_suitable(op):
            overall_analog += flops
            if layer_id is not None:
                layer_data[layer_id]["analog"] += flops
        else:
            overall_digital += flops
            if layer_id is not None:
                layer_data[layer_id]["digital"] += flops

    total = overall_analog + overall_digital
    per_layer: dict[str, Any] = {}
    for lid in sorted(layer_data):
        a = layer_data[lid]["analog"]
        d = layer_data[lid]["digital"]
        t = a + d
        per_layer[str(lid)] = {
            "analog_flops": a,
            "digital_flops": d,
            "total_flops": t,
            "analog_ratio": a / t if t > 0 else 0.0,
        }

    return {
        "overall": {
            "analog_flops": overall_analog,
            "digital_flops": overall_digital,
            "total_flops": total,
            "analog_ratio": overall_analog / total if total > 0 else 0.0,
        },
        "per_layer": per_layer,
    }


# ─── Metric 5: Effective Batch Size Distribution ─────────────────────────────

def compute_batch_size_distribution(
    ops: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Collect tokens_received per expert-op activation and compute CDF.

    Each data point is the number of tokens dispatched to one expert in one op.
    """
    token_counts: list[int] = []
    for op in ops:
        if not _is_expert_op(op):
            continue
        tokens = _op_input_tokens(op)
        if tokens > 0:
            token_counts.append(tokens)

    if not token_counts:
        return {"values": [], "sorted_values": [], "cdf": [], "percentiles": {}}

    sorted_vals = sorted(token_counts)
    n = len(sorted_vals)
    cdf = [[sorted_vals[i], (i + 1) / n] for i in range(n)]
    fvals = [float(v) for v in sorted_vals]

    return {
        "values": token_counts,
        "sorted_values": sorted_vals,
        "cdf": cdf,
        "percentiles": {
            "p10": _percentile(fvals, 10),
            "p25": _percentile(fvals, 25),
            "p50": _percentile(fvals, 50),
            "p75": _percentile(fvals, 75),
            "p90": _percentile(fvals, 90),
        },
    }


# ─── Metric 6: Expert Activation Frequency ───────────────────────────────────

def compute_expert_activation_frequency(
    traces: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Count how often each (layer, expert) pair is activated across multiple trace files.

    frequency[layer][expert] = total_activations / total_requests
    Uses the routing_trace embedded in each workload trace pickle.
    """
    total_requests = len(traces)
    if total_requests == 0:
        return {
            "n_layers": 0, "n_experts": 0, "layer_names": [],
            "heatmap": [], "frequency": [], "total_requests": 0,
        }

    layer_expert_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    layer_id_to_name: dict[int, str] = {}

    for trace in traces:
        routing_trace = trace.get("routing_trace", {})
        for prompt_records in routing_trace.values():
            for record in prompt_records:
                if isinstance(record, dict):
                    layer_id = record.get("layer_id")
                    layer_name = record.get("layer_name", str(layer_id))
                    selected = record.get("selected_experts", [])
                elif hasattr(record, "layer_id"):
                    layer_id = record.layer_id
                    layer_name = record.layer_name
                    selected = record.selected_experts
                else:
                    continue
                if layer_id is None:
                    continue
                layer_id_to_name[int(layer_id)] = str(layer_name)
                for expert_id in selected:
                    layer_expert_counts[int(layer_id)][int(expert_id)] += 1

    if not layer_expert_counts:
        return {
            "n_layers": 0, "n_experts": 0, "layer_names": [],
            "heatmap": [], "frequency": [], "total_requests": total_requests,
        }

    sorted_layer_ids = sorted(layer_expert_counts)
    all_eids = sorted({eid for counts in layer_expert_counts.values() for eid in counts})
    n_experts = (max(all_eids) + 1) if all_eids else 0
    layer_names = [layer_id_to_name.get(lid, f"layer {lid}") for lid in sorted_layer_ids]

    heatmap: list[list[int]] = []
    frequency: list[list[float]] = []
    for lid in sorted_layer_ids:
        row = [layer_expert_counts[lid].get(eid, 0) for eid in range(n_experts)]
        heatmap.append(row)
        frequency.append([c / total_requests for c in row])

    return {
        "n_layers": len(sorted_layer_ids),
        "n_experts": n_experts,
        "layer_names": layer_names,
        "heatmap": heatmap,
        "frequency": frequency,
        "total_requests": total_requests,
    }


# ─── Metric 7: Roofline Positioning ──────────────────────────────────────────

def compute_roofline(
    ops: list[dict[str, Any]],
    config: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Position each op on the roofline model.

    roofline_bound = min(peak_compute, intensity * peak_bandwidth)
    Ridge point = peak_compute / peak_bandwidth; ops below are memory-bound.
    """
    peak_compute = float(config.get("peak_compute_tops", DEFAULT_CONFIG["peak_compute_tops"]))
    peak_bw = float(config.get("peak_bandwidth_tbs", DEFAULT_CONFIG["peak_bandwidth_tbs"]))
    ridge_point = peak_compute / peak_bw if peak_bw > 0 else float("inf")

    # Aggregate all invocations of the same weight matrix across all inferences.
    # One dot per unique module path (= one dot per physical weight matrix).
    weight_groups: dict[str, dict[str, Any]] = {}

    for op in ops:
        if op.get("op_type") != "linear":
            continue
        module = op.get("module") or ""
        family = op.get("op_family", "")

        if family == "dynamic_activation_matmul":
            category = "attention"
        elif family == "moe_routing":
            category = "gate"
        else:
            category = parse_op_role(module)

        if module not in weight_groups:
            # Weight numels are a fixed property of the matrix — recorded once.
            # v2: op carries weight_numels directly; v1: sum from nested weights list.
            w_numels = int(op.get("weight_numels", 0)) or sum(
                int(w.get("numels", 0)) for w in op.get("weights", [])
            )
            weight_groups[module] = {
                "module":              module,
                "op_family":           family,
                "role_category":       category,
                "flops":               0,
                "numels":              w_numels,
                "n_calls":             0,
                "total_input_numels":  0,
                "total_output_numels": 0,
            }
        weight_groups[module]["flops"]              += _op_flops(op)
        weight_groups[module]["n_calls"]            += 1
        weight_groups[module]["total_input_numels"] += int(op.get("input_numels", 0))
        weight_groups[module]["total_output_numels"]+= int(op.get("output_numels", 0))

    op_data: list[dict[str, Any]] = []
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"compute_bound": 0, "memory_bound": 0})

    for group in weight_groups.values():
        if group["numels"] <= 0:
            continue
        n_calls = group["n_calls"]
        w_numels = group["numels"]
        total_flops = group["flops"]
        # Classic roofline intensity: FLOPs vs weight numels only.
        intensity = total_flops / w_numels
        # Full intensity: FLOPs vs all tensor numels (input + weight-loaded + output).
        weight_loaded  = n_calls * w_numels
        full_numels    = group["total_input_numels"] + weight_loaded + group["total_output_numels"]
        arithmetic_intensity = total_flops / full_numels if full_numels > 0 else 0.0
        bound = "compute_bound" if intensity >= ridge_point else "memory_bound"
        by_category[group["role_category"]][bound] += 1
        op_data.append({
            "module":               group["module"],
            "op_type":              "linear",
            "op_family":            group["op_family"],
            "role_category":        group["role_category"],
            "intensity":            intensity,              # FLOPs / weight_numels
            "arithmetic_intensity": arithmetic_intensity,   # FLOPs / all_numels
            "flops":                total_flops,
            "n_calls":              n_calls,
            "weight_numels":        w_numels,
            "flops_per_call":       total_flops / n_calls  if n_calls else 0.0,
            "numels_per_call":      full_numels / n_calls  if n_calls else 0.0,
            "bound":                bound,
        })

    n_sequences = int((metadata or {}).get("n_sequences", 1))

    return {
        "config": {
            "peak_compute_tops": peak_compute,
            "peak_bandwidth_tbs": peak_bw,
        },
        "ridge_point": ridge_point,
        "n_sequences": n_sequences,
        "ops": op_data,
        "by_category": {k: dict(v) for k, v in by_category.items()},
    }


# ─── Metric 8a: Numels vs FLOPs per Forward Step ─────────────────────────────

def compute_numels_flops_per_step(
    ops: list[dict[str, Any]],
    metadata: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Per-weight AIMC candidate profile.

    For each linear weight matrix, aggregates across every call in the trace:
      n_calls              — times the weight was used
      total_flops          — accumulated FLOPs (y-axis of bubble chart)
      arithmetic_intensity — total_flops / (input_numels + weight_numels_loaded + output_numels)
                             i.e. flops_per_call / numels_per_call  (x-axis)
      weight_numels_loaded = n_calls × weight_numels  (loaded fresh each call)

    n_forward_steps = n_sequences (prefill) + output_tokens (decode steps)
    """
    n_sequences   = int(metadata.get("n_sequences", 1))
    output_tokens = int(metadata.get("output_tokens", 0))
    n_steps = max(n_sequences + output_tokens, 1)

    weight_groups: dict[str, dict[str, Any]] = {}
    for op in ops:
        if op.get("op_type") != "linear":
            continue
        module = op.get("module") or ""
        family = op.get("op_family", "")
        if family == "dynamic_activation_matmul":
            category = "attention"
        elif family == "moe_routing":
            category = "gate"
        else:
            category = parse_op_role(module)

        if module not in weight_groups:
            w_numels = int(op.get("weight_numels", 0)) or sum(
                int(w.get("numels", 0)) for w in op.get("weights", [])
            )
            weight_groups[module] = {
                "module":              module,
                "role_category":       category,
                "n_calls":             0,
                "total_flops":         0,
                "weight_numels":       w_numels,
                "total_input_numels":  0,
                "total_output_numels": 0,
            }
        weight_groups[module]["n_calls"]             += 1
        weight_groups[module]["total_flops"]         += _op_flops(op)
        weight_groups[module]["total_input_numels"]  += int(op.get("input_numels", 0))
        weight_groups[module]["total_output_numels"] += int(op.get("output_numels", 0))

    result_ops: list[dict[str, Any]] = []
    for group in weight_groups.values():
        if group["weight_numels"] <= 0:
            continue
        n_calls         = group["n_calls"]
        w_numels        = group["weight_numels"]
        total_flops     = group["total_flops"]
        weight_loaded   = n_calls * w_numels
        total_all_numels = (
            group["total_input_numels"] + weight_loaded + group["total_output_numels"]
        )
        ai = total_flops / total_all_numels if total_all_numels > 0 else 0.0
        result_ops.append({
            "module":               group["module"],
            "role_category":        group["role_category"],
            "n_calls":              n_calls,
            "weight_numels":        w_numels,
            "total_flops":          total_flops,
            "total_numels":         total_all_numels,
            "arithmetic_intensity": ai,
            "flops_per_call":       total_flops / n_calls    if n_calls else 0.0,
            "numels_per_call":      total_all_numels / n_calls if n_calls else 0.0,
            "numels_per_step":      weight_loaded / n_steps,
            "flops_per_step":       total_flops   / n_steps,
        })

    return {
        "n_steps":       n_steps,
        "n_sequences":   n_sequences,
        "output_tokens": output_tokens,
        "ops":           result_ops,
    }


# ─── Metric 8: Tile Activation Timeline ──────────────────────────────────────

def compute_tile_timeline(
    ops: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Estimate relative execution timeline per expert per layer.

    Duration is proportional to total FLOPs, normalized so the busiest expert
    in each layer ends at relative_end=1.0. All experts start at 0 (parallel).
    """
    layer_expert_flops: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    layer_expert_tokens: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for op in ops:
        if not _is_expert_op(op):
            continue
        module = op.get("module") or ""
        layer_id = parse_layer_id(module)
        expert_id = parse_expert_id(module)
        if layer_id is None or expert_id is None:
            continue
        layer_expert_flops[layer_id][expert_id] += _op_flops(op)
        tokens = _op_input_tokens(op)
        if tokens > 0:
            layer_expert_tokens[layer_id][expert_id] = max(
                layer_expert_tokens[layer_id][expert_id], tokens
            )

    per_layer: dict[str, Any] = {}
    for layer_id in sorted(layer_expert_flops):
        expert_flops = layer_expert_flops[layer_id]
        expert_tokens = layer_expert_tokens[layer_id]
        busiest = max(expert_flops.values()) if expert_flops else 1

        experts = [
            {
                "expert_id": eid,
                "tokens": expert_tokens.get(eid, 0),
                "total_flops": expert_flops[eid],
                "relative_start": 0.0,
                "relative_end": expert_flops[eid] / busiest if busiest > 0 else 0.0,
            }
            for eid in sorted(expert_flops)
        ]
        per_layer[str(layer_id)] = {
            "experts": experts,
            "busiest_expert_flops": busiest,
        }

    return {"per_layer": per_layer}


# ─── Metric 9: Precision Sensitivity (separate experiment runner) ─────────────

def run_precision_sensitivity(
    model: Any,
    input_ids: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Run inference at multiple precisions and measure routing + output quality
    relative to the fp16 baseline.

    Returns a dict suitable for write_precision_sensitivity(). Requires torch;
    bitsandbytes is used for int4 when available, with fake-quantization fallback.

    Args:
        model:     A loaded nn.Module (e.g. from AutoModelForCausalLM).
        input_ids: torch.Tensor with input token ids.
        config:    Metrics config dict (reads 'precisions_to_test').
    """
    import copy

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return {"error": "torch not available"}

    precisions = list(config.get("precisions_to_test", DEFAULT_CONFIG["precisions_to_test"]))
    device = next(model.parameters()).device

    def _capture(mdl: nn.Module, ids: Any) -> tuple[list[list[int]], Any]:
        """Run a forward pass and capture gate routing decisions + output logits."""
        decisions: list[list[int]] = []

        def _hook(_: nn.Module, __: Any, output: Any) -> None:
            if isinstance(output, tuple):
                for item in output:
                    if isinstance(item, torch.Tensor) and item.dtype in (torch.int32, torch.int64):
                        decisions.append(item.reshape(-1).tolist())
                        return
            elif isinstance(output, torch.Tensor) and output.dtype in (torch.int32, torch.int64):
                decisions.append(output.reshape(-1).tolist())

        handles = [
            mod.register_forward_hook(_hook)
            for name, mod in mdl.named_modules()
            if name.rsplit(".", 1)[-1] == "gate"
        ]
        try:
            with torch.no_grad():
                out = mdl(ids)
        finally:
            for h in handles:
                h.remove()
        return decisions, getattr(out, "logits", None)

    def _to_int8(mdl: nn.Module) -> nn.Module:
        import torch.quantization
        return torch.quantization.quantize_dynamic(mdl, {nn.Linear}, dtype=torch.qint8)

    def _fake_q4(t: torch.Tensor) -> torch.Tensor:
        mn, mx = t.min(), t.max()
        scale = (mx - mn) / 15.0
        return (torch.round((t - mn) / scale).clamp(0, 15) * scale + mn) if scale != 0 else t

    def _to_int4(mdl: nn.Module) -> nn.Module:
        try:
            import bitsandbytes as bnb  # noqa: F401
            from bitsandbytes.nn import Linear4bit
            m = copy.deepcopy(mdl)
            for name, mod in list(m.named_modules()):
                if isinstance(mod, nn.Linear):
                    parent_name, _, child_name = name.rpartition(".")
                    parent = m.get_submodule(parent_name) if parent_name else m
                    setattr(parent, child_name,
                            Linear4bit(mod.in_features, mod.out_features, bias=mod.bias is not None))
            return m
        except ImportError:
            m = copy.deepcopy(mdl)
            with torch.no_grad():
                for mod in m.modules():
                    if isinstance(mod, nn.Linear):
                        mod.weight.data = _fake_q4(mod.weight.data.float()).to(mod.weight.dtype)
            return m

    def _selective_int8(mdl: nn.Module, weight_type: str) -> nn.Module:
        """Quantize only modules of weight_type to int8; keep all others at current dtype."""
        import torch.quantization
        m = copy.deepcopy(mdl)
        target_role = {"gate": "gate", "expert": "expert_ffn", "attention": "attention"}[weight_type]
        for name, mod in list(m.named_modules()):
            if not isinstance(mod, nn.Linear) or parse_op_role(name) != target_role:
                continue
            parent_name, _, child_name = name.rpartition(".")
            parent = m.get_submodule(parent_name) if parent_name else m
            q = torch.quantization.quantize_dynamic(
                nn.Sequential(mod), {nn.Linear}, dtype=torch.qint8
            )[0]
            setattr(parent, child_name, q)
        return m

    def _routing_agreement(base: list[list[int]], test: list[list[int]]) -> float:
        n = min(len(base), len(test))
        scores = [
            len(set(base[i]) & set(test[i])) / len(set(base[i]))
            for i in range(n)
            if base[i]
        ]
        return statistics.mean(scores) if scores else 0.0

    def _mse(a: Any, b: Any) -> float | None:
        if a is None or b is None or a.shape != b.shape:
            return None
        return float(((a.float() - b.float()) ** 2).mean().item())

    # fp16 baseline
    try:
        base_model = copy.deepcopy(model).half().eval().to(device)
        ids = input_ids.to(device) if hasattr(input_ids, "to") else input_ids
        base_routing, base_logits = _capture(base_model, ids)
    except Exception as exc:
        return {"error": f"fp16 baseline failed: {exc}"}

    results: dict[str, Any] = {}
    for prec in precisions:
        if prec == "fp16":
            results["fp16"] = {"routing_agreement": 1.0, "output_mse": 0.0}
            continue
        try:
            src = copy.deepcopy(model).float().eval()
            test_model = (_to_int8(src) if prec == "int8" else _to_int4(src)).to(device)
            test_routing, test_logits = _capture(test_model, ids)
            results[prec] = {
                "routing_agreement": _routing_agreement(base_routing, test_routing),
                "output_mse": _mse(base_logits, test_logits),
            }
        except Exception as exc:
            results[prec] = {"error": str(exc)}

    by_weight_type: dict[str, Any] = {}
    for wtype in ("gate", "expert", "attention"):
        try:
            sel = _selective_int8(copy.deepcopy(model).half().eval(), wtype).to(device)
            sel_routing, sel_logits = _capture(sel, ids)
            by_weight_type[f"{wtype}_only"] = {
                "quantization": "int8",
                "routing_agreement": _routing_agreement(base_routing, sel_routing),
                "output_mse": _mse(base_logits, sel_logits),
            }
        except Exception as exc:
            by_weight_type[f"{wtype}_only"] = {"error": str(exc)}

    return {
        "baseline_precision": "fp16",
        "precisions_tested": precisions,
        "results": results,
        "by_weight_type": by_weight_type,
    }


def write_precision_sensitivity(results: dict[str, Any], output_dir: Path) -> Path:
    """Write the output of run_precision_sensitivity() to the standard metrics path."""
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / "precision_sensitivity.json"
    _write_json(results, out)
    return out


# ─── Report generation ────────────────────────────────────────────────────────

def _fmt(v: float | None, n: int = 4) -> str:
    return "N/A" if v is None else f"{v:.{n}f}"


def generate_report(all_results: dict[str, Any]) -> str:
    """Render a human-readable Markdown report from the collected metric results."""
    lines: list[str] = ["# MoE AIMC Workload Metrics Report", ""]

    # 1. Arithmetic Intensity
    ai = all_results.get("arithmetic_intensity")
    if ai:
        lines += ["## 1. Arithmetic Intensity Distribution", ""]
        thresh = ai.get("config", {}).get("aimc_roofline_threshold", "?")
        lines.append(f"AIMC roofline threshold: **{thresh} FLOP/numel**")
        lines.append("")
        lines.append("| Category | Count | Mean | P50 | P95 | Below Threshold |")
        lines.append("|----------|------:|-----:|----:|----:|----------------:|")
        for cat, d in ai.get("categories", {}).items():
            s = d.get("stats", {})
            below_pct = d.get("below_threshold_fraction", 0) * 100
            lines.append(
                f"| {cat} | {s.get('count', 0)} | {_fmt(s.get('mean'))} | "
                f"{_fmt(s.get('p50'))} | {_fmt(s.get('p95'))} | {below_pct:.1f}% |"
            )
        lines.append("")

    # 2. Tile Mapping
    tm = all_results.get("tile_mapping")
    if tm:
        lines += ["## 2. Tile Mapping per Weight Matrix", ""]
        ts = tm.get("config", {}).get("tile_size", "?")
        lines.append(f"Tile size: **{ts}×{ts}** — Total tiles: **{tm.get('total_tiles', 0):,}**")
        lines.append("")
        lines.append("| Layer | Total Tiles | Avg Utilization |")
        lines.append("|------:|------------:|----------------:|")
        for lid, d in tm.get("per_layer_budget", {}).items():
            util_pct = d.get("avg_utilization", 0) * 100
            lines.append(f"| {lid} | {d.get('total_tiles', 0):,} | {util_pct:.1f}% |")
        lines.append("")

    # 3. Load Imbalance
    li = all_results.get("load_imbalance")
    if li:
        lines += ["## 3. Load Imbalance per Layer", ""]
        lines.append("| Layer | Routings | Mean Rate | Max Rate | Min Rate | Imbalance Ratio |")
        lines.append("|------:|---------:|----------:|---------:|---------:|----------------:|")
        for lid, d in li.get("per_layer", {}).items():
            lines.append(
                f"| {lid} | {d['n_routings']:.0f} | {d['mean_rate']:.4f} | "
                f"{d['max_rate']:.4f} | {d['min_rate']:.4f} | **{d['load_imbalance_ratio']:.3f}** |"
            )
        lines.append("")

    # 4. Digital vs Analog Split
    da = all_results.get("digital_analog_split")
    if da:
        lines += ["## 4. Digital vs Analog Operation Split", ""]
        ov = da.get("overall", {})
        analog_pct = ov.get("analog_ratio", 0) * 100
        lines.append(f"**Overall analog FLOP ratio: {analog_pct:.2f}%**")
        lines.append("")
        lines.append(f"- Analog FLOPs : {ov.get('analog_flops', 0):,}")
        lines.append(f"- Digital FLOPs: {ov.get('digital_flops', 0):,}")
        lines.append(f"- Total FLOPs  : {ov.get('total_flops', 0):,}")
        lines.append("")
        lines.append("| Layer | Analog FLOPs | Digital FLOPs | Analog Ratio |")
        lines.append("|------:|-------------:|--------------:|-------------:|")
        for lid, d in da.get("per_layer", {}).items():
            lines.append(
                f"| {lid} | {d['analog_flops']:,} | {d['digital_flops']:,} | "
                f"{d['analog_ratio'] * 100:.1f}% |"
            )
        lines.append("")

    # 5. Batch Size Distribution
    bs = all_results.get("batch_size_distribution")
    if bs:
        lines += ["## 5. Effective Batch Size Distribution (Tokens per Expert)", ""]
        pct = bs.get("percentiles", {})
        lines.append("| Percentile | Tokens |")
        lines.append("|:----------:|-------:|")
        for p in ("p10", "p25", "p50", "p75", "p90"):
            lines.append(f"| {p.upper()} | {_fmt(pct.get(p), 1)} |")
        lines.append("")

    # 6. Expert Activation Frequency
    ef = all_results.get("expert_activation_frequency")
    if ef:
        lines += ["## 6. Expert Activation Frequency", ""]
        lines.append(
            f"Layers: **{ef.get('n_layers', 0)}** — "
            f"Experts: **{ef.get('n_experts', 0)}** — "
            f"Requests: **{ef.get('total_requests', 0)}**"
        )
        lines.append("")
        lines.append("*(Full heatmap in `raw/expert_activation_frequency.json`)*")
        lines.append("")

    # 7. Roofline
    rf = all_results.get("roofline")
    if rf:
        lines += ["## 7. Roofline Positioning", ""]
        cfg = rf.get("config", {})
        ridge = rf.get("ridge_point", 0)
        lines.append(
            f"Peak compute: **{cfg.get('peak_compute_tops')} TOPS** — "
            f"Peak BW: **{cfg.get('peak_bandwidth_tbs')} TB/s** — "
            f"Ridge point: **{ridge:.2f} FLOP/numel**"
        )
        lines.append("")
        lines.append("| Category | Compute-Bound | Memory-Bound |")
        lines.append("|:---------|-------------:|--------------:|")
        for cat, counts in rf.get("by_category", {}).items():
            lines.append(
                f"| {cat} | {counts.get('compute_bound', 0)} | {counts.get('memory_bound', 0)} |"
            )
        lines.append("")

    # 8. Tile Timeline
    tt = all_results.get("tile_timeline")
    if tt:
        lines += ["## 8. Tile Activation Timeline", ""]
        n = len(tt.get("per_layer", {}))
        lines.append(
            f"Timeline data for **{n} layers**. "
            "*(Full Gantt data in `raw/tile_timeline.json`)*"
        )
        lines.append("")

    # 9. Precision Sensitivity
    ps = all_results.get("precision_sensitivity")
    if ps:
        lines += ["## 9. Precision Sensitivity", ""]
        if "error" in ps:
            lines.append(f"> Error: {ps['error']}")
        else:
            lines.append("**Full-model quantization vs fp16 baseline:**")
            lines.append("")
            lines.append("| Precision | Routing Agreement | Output MSE |")
            lines.append("|:----------|------------------:|-----------:|")
            for prec, d in ps.get("results", {}).items():
                if "error" in d:
                    lines.append(f"| {prec} | *(error)* | *(error)* |")
                else:
                    ra = (d.get("routing_agreement") or 0) * 100
                    lines.append(f"| {prec} | {ra:.2f}% | {_fmt(d.get('output_mse'), 6)} |")
            lines.append("")
            lines.append("**Selective int8 quantization (one weight type at a time):**")
            lines.append("")
            lines.append("| Weight Type | Routing Agreement | Output MSE |")
            lines.append("|:------------|------------------:|-----------:|")
            for wt, d in ps.get("by_weight_type", {}).items():
                if "error" in d:
                    lines.append(f"| {wt} | *(error)* | *(error)* |")
                else:
                    ra = (d.get("routing_agreement") or 0) * 100
                    lines.append(f"| {wt} | {ra:.2f}% | {_fmt(d.get('output_mse'), 6)} |")
        lines.append("")

    return "\n".join(lines)


# ─── JSON serialization ───────────────────────────────────────────────────────

def _to_json_safe(obj: Any) -> Any:
    """Recursively convert to JSON-serializable types."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if hasattr(obj, "item"):
        return _to_json_safe(obj.item())
    if hasattr(obj, "tolist"):
        return _to_json_safe(obj.tolist())
    return str(obj)


def _write_json(data: Any, path: Path) -> None:
    path.write_text(json.dumps(_to_json_safe(data), indent=2), encoding="utf-8")


# ─── Orchestration ────────────────────────────────────────────────────────────

def run_all_metrics(
    trace_paths: list[Path],
    output_dir: Path,
    config: dict[str, Any] | None = None,
    precision_sensitivity_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run metrics 1–8 from trace file(s) and write all outputs under output_dir.

    Metric 9 (precision sensitivity) requires a live model. Pass pre-computed
    results via precision_sensitivity_results, or call run_precision_sensitivity()
    and write_precision_sensitivity() separately.

    Args:
        trace_paths:                   One or more workload_trace.pkl paths.
                                       The first is the primary trace; all are
                                       used for Metric 6 (activation frequency).
        output_dir:                    Root for metrics/ output tree.
        config:                        Override any DEFAULT_CONFIG keys.
        precision_sensitivity_results: Optional pre-computed Metric 9 output.

    Returns:
        Dict mapping metric name → raw result dict.
    """
    if not trace_paths:
        raise ValueError("At least one trace path is required.")

    cfg: dict[str, Any] = {**DEFAULT_CONFIG, **(config or {})}
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"[metrics] Loading {len(trace_paths)} trace file(s)...")
    primary = load_trace(trace_paths[0])
    all_traces = [primary] + [load_trace(p) for p in trace_paths[1:]]

    ops = _collect_all_ops(primary)

    # v2: weights dict; v1: static_matrix_inventory list
    if "weights" in primary:
        weights_dict = primary["weights"]
        static_matrices = [
            {
                "module":  name,
                "shape":   w["shape"],
                "role":    w["role"],
                "crossbar": {
                    "tile_shape":         (128, 128),
                    "tiles":              w.get("tiles") or 0,
                    "used_cells":         w.get("used_cells") or 0,
                    "provisioned_cells":  w.get("provisioned_cells") or 0,
                    "tiling_efficiency":  w.get("tile_efficiency") or 0.0,
                },
            }
            for name, w in weights_dict.items()
        ]
        # Enrich ops: fill weight_numels from the inventory when the op
        # itself doesn't carry it (e.g. traces recorded before this field existed).
        for op in ops:
            if not op.get("weight_numels") and op.get("op_type") == "linear":
                numel = weights_dict.get(op.get("module", ""), {}).get("numels", 0)
                if numel:
                    op["weight_numels"] = numel
    else:
        weights_dict = {}
        static_matrices = primary.get("model", {}).get("static_matrix_inventory", [])

    linear_ops    = [op for op in ops if op.get("op_type") == "linear"]
    non_linear_ops = [op for op in ops if op.get("op_type") != "linear"]
    print(f"[metrics] {len(ops)} ops from primary trace ({len(linear_ops)} linear, {len(non_linear_ops)} non-linear), {len(static_matrices)} static matrices.")

    if non_linear_ops:
        counts = Counter(
            (op.get("op_type", "unknown"), op.get("role", ""))
            for op in non_linear_ops
        )
        print("[metrics] Non-linear ops excluded from metrics:")
        for (op_type, role), count in sorted(counts.items(), key=lambda x: -x[1]):
            role_str = f"  ({role})" if role else ""
            print(f"[metrics]   {op_type:<30} {count:>6}{role_str}")

    all_results: dict[str, Any] = {}

    metadata = primary.get("metadata", {})

    _METRICS = [
        ("roofline",                  lambda: compute_roofline(ops, cfg, metadata)),
    ]

    for i, (name, fn) in enumerate(_METRICS, 1):
        print(f"[metrics] {i}/{len(_METRICS)} {name}...")
        all_results[name] = fn()
        _write_json(all_results[name], raw_dir / f"{name}.json")

    if precision_sensitivity_results is not None:
        all_results["precision_sensitivity"] = precision_sensitivity_results
        _write_json(precision_sensitivity_results, raw_dir / "precision_sensitivity.json")
    else:
        print("[metrics] Skipping metric 9 (precision_sensitivity) — no model provided.")
        print("[metrics]   Call run_precision_sensitivity(model, input_ids, config) separately.")

    report = generate_report(all_results)
    report_path = output_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"[metrics] Done. Report: {report_path}")

    return all_results


# ─── CLI entry point ──────────────────────────────────────────────────────────

def _build_cli() -> "argparse.ArgumentParser":
    import argparse

    p = argparse.ArgumentParser(
        description="Run the MoE AIMC workload metrics pipeline on trace file(s).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("traces", nargs="+", metavar="TRACE",
                   help="workload_trace.pkl file(s). First is primary.")
    p.add_argument("--output-dir", default="metrics",
                   help="Root output directory.")
    p.add_argument("--tile-size", type=int, default=DEFAULT_CONFIG["tile_size"],
                   metavar="N")
    p.add_argument("--aimc-threshold", type=float,
                   default=DEFAULT_CONFIG["aimc_roofline_threshold"],
                   metavar="FLOP/numel")
    p.add_argument("--peak-compute", type=float,
                   default=DEFAULT_CONFIG["peak_compute_tops"],
                   metavar="TOPS")
    p.add_argument("--peak-bandwidth", type=float,
                   default=DEFAULT_CONFIG["peak_bandwidth_tbs"],
                   metavar="TB/s")
    return p


if __name__ == "__main__":
    import argparse

    args = _build_cli().parse_args()
    run_all_metrics(
        trace_paths=[Path(t).resolve() for t in args.traces],
        output_dir=Path(args.output_dir).resolve(),
        config={
            "tile_size": args.tile_size,
            "aimc_roofline_threshold": args.aimc_threshold,
            "peak_compute_tops": args.peak_compute,
            "peak_bandwidth_tbs": args.peak_bandwidth,
        },
    )
