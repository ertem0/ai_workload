from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from src.plotting.expert_heatmaps import (
    generate_individual_spatial_heatmaps,
    generate_layer_transition_heatmaps,
    plot_transition_umap,
    plot_expert_load_and_entropy,
    save_expert_heatmap_plot,
)
from src.tracing.routing_trace import RoutingTraceRecord, load_routing_trace


def _layer_sort_key(name: str) -> tuple[int, str]:
    if name.startswith("layer "):
        return int(name.split()[-1]), name
    return 10**9, name


def _infer_num_routed_experts(routing_trace: dict[int, list[RoutingTraceRecord]]) -> int:
    max_expert_id = -1
    for records in routing_trace.values():
        for record in records:
            if record.selected_experts:
                max_expert_id = max(max_expert_id, max(record.selected_experts))
    return max_expert_id + 1 if max_expert_id >= 0 else 0


def _build_layer_transition_statistics(
    routing_trace: dict[int, list[RoutingTraceRecord]],
    *,
    layer_names: list[str],
    layer_name_to_id: dict[str, int],
    num_routed_experts: int,
) -> list[dict[str, Any]]:
    if num_routed_experts <= 0:
        return []

    layer_id_to_name = {
        layer_id: layer_name
        for layer_name, layer_id in layer_name_to_id.items()
        if layer_id >= 0
    }
    layer_ids = sorted(layer_id_to_name)
    transition_counts = {
        layer_id: torch.zeros(
            (num_routed_experts, num_routed_experts),
            dtype=torch.long,
        )
        for layer_id in layer_ids
        if layer_id + 1 in layer_id_to_name
    }
    source_counts = {
        layer_id: torch.zeros(num_routed_experts, dtype=torch.long)
        for layer_id in transition_counts
    }

    grouped_records: dict[tuple[int, str, int], dict[int, RoutingTraceRecord]] = {}
    for records in routing_trace.values():
        for record in records:
            if record.layer_id < 0:
                continue
            key = (
                int(record.prompt_index),
                str(record.phase),
                int(record.token_position),
            )
            grouped_records.setdefault(key, {})[int(record.layer_id)] = record

    for records_by_layer in grouped_records.values():
        for source_layer_id in transition_counts:
            source_record = records_by_layer.get(source_layer_id)
            target_record = records_by_layer.get(source_layer_id + 1)
            if source_record is None or target_record is None:
                continue

            source_experts = sorted(
                {
                    int(expert_id)
                    for expert_id in source_record.selected_experts
                    if 0 <= int(expert_id) < num_routed_experts
                }
            )
            target_experts = sorted(
                {
                    int(expert_id)
                    for expert_id in target_record.selected_experts
                    if 0 <= int(expert_id) < num_routed_experts
                }
            )
            if not source_experts or not target_experts:
                continue

            for source_expert in source_experts:
                source_counts[source_layer_id][source_expert] += 1
                for target_expert in target_experts:
                    transition_counts[source_layer_id][source_expert, target_expert] += 1

    transitions: list[dict[str, Any]] = []
    for source_layer_id in sorted(transition_counts):
        counts = transition_counts[source_layer_id]
        probabilities = torch.zeros_like(counts, dtype=torch.float64)
        active_sources = source_counts[source_layer_id] > 0
        probabilities[active_sources] = (
            counts[active_sources].to(torch.float64)
            / source_counts[source_layer_id][active_sources].to(torch.float64).unsqueeze(1)
        )
        transitions.append(
            {
                "source_layer_id": source_layer_id,
                "target_layer_id": source_layer_id + 1,
                "source_layer_name": layer_id_to_name.get(
                    source_layer_id,
                    layer_names[source_layer_id] if source_layer_id < len(layer_names) else "",
                ),
                "target_layer_name": layer_id_to_name.get(
                    source_layer_id + 1,
                    layer_names[source_layer_id + 1]
                    if source_layer_id + 1 < len(layer_names)
                    else "",
                ),
                "source_counts": source_counts[source_layer_id],
                "source_event_count": int(source_counts[source_layer_id].sum().item()),
                "counts": counts,
                "probabilities": probabilities,
            }
        )

    return transitions


def _build_expert_routing_statistics(routing_trace: dict[int, list[RoutingTraceRecord]]) -> dict[str, Any]:
    num_routed_experts = _infer_num_routed_experts(routing_trace)
    layer_names = sorted(
        {
            record.layer_name
            for records in routing_trace.values()
            for record in records
        },
        key=_layer_sort_key,
    )
    layer_counts = {
        layer_name: torch.zeros(num_routed_experts, dtype=torch.long)
        for layer_name in layer_names
    }
    layer_name_to_id = {
        layer_name: next(
            record.layer_id
            for records in routing_trace.values()
            for record in records
            if record.layer_name == layer_name
        )
        for layer_name in layer_names
    }
    expert_pair_counts = torch.zeros(
        (num_routed_experts, num_routed_experts),
        dtype=torch.long,
    )
    layer_event_counts = {layer_name: 0 for layer_name in layer_names}
    pair_layer_counts: dict[tuple[int, int], dict[str, int]] = {}
    total_routing_events = 0

    for records in routing_trace.values():
        for record in records:
            if record.layer_name not in layer_counts:
                continue
            selected = [
                int(expert_id)
                for expert_id in record.selected_experts
                if 0 <= int(expert_id) < num_routed_experts
            ]
            if not selected:
                continue

            total_routing_events += 1
            layer_event_counts[record.layer_name] += 1
            counts = torch.bincount(
                torch.tensor(selected, dtype=torch.long),
                minlength=num_routed_experts,
            )
            layer_counts[record.layer_name].add_(counts)

            unique_experts = sorted(set(selected))
            for expert_id in unique_experts:
                expert_pair_counts[expert_id, expert_id] += 1
            for left_index, expert_i in enumerate(unique_experts):
                for expert_j in unique_experts[left_index + 1 :]:
                    expert_pair_counts[expert_i, expert_j] += 1
                    expert_pair_counts[expert_j, expert_i] += 1
                    pair_key = (expert_i, expert_j)
                    layer_counter = pair_layer_counts.setdefault(pair_key, {})
                    layer_counter[record.layer_name] = (
                        layer_counter.get(record.layer_name, 0) + 1
                    )

    results: list[dict[str, Any]] = []
    for layer_name in layer_names:
        counts = layer_counts[layer_name].to(torch.float64)
        mean_count = float(counts.mean().item()) if counts.numel() > 0 else 0.0
        std_count = float(counts.std(unbiased=False).item()) if counts.numel() > 0 else 0.0
        workload_skewness = float(std_count / mean_count) if mean_count > 0 else float("inf")
        results.append(
            {
                "layer_name": layer_name,
                "counts": layer_counts[layer_name],
                "workload_skewness": workload_skewness,
            }
        )

    layer_expert_matrix = (
        torch.stack([layer_counts[layer_name] for layer_name in layer_names], dim=0)
        if layer_names
        else torch.zeros((0, num_routed_experts), dtype=torch.long)
    )

    top_pairs: list[dict[str, Any]] = []
    for expert_i in range(num_routed_experts):
        for expert_j in range(expert_i + 1, num_routed_experts):
            pair_count = int(expert_pair_counts[expert_i, expert_j].item())
            if pair_count == 0:
                continue
            pair_key = (expert_i, expert_j)
            layer_counter = pair_layer_counts.get(pair_key, {})
            dominant_layer_name = ""
            dominant_layer_count = 0
            dominant_layer_id = -1
            dominant_layer_events = 0
            if layer_counter:
                dominant_layer_name, dominant_layer_count = max(
                    layer_counter.items(),
                    key=lambda item: (item[1], -layer_name_to_id.get(item[0], 10**9), item[0]),
                )
                dominant_layer_id = int(layer_name_to_id.get(dominant_layer_name, -1))
                dominant_layer_events = int(layer_event_counts.get(dominant_layer_name, 0))
            top_pairs.append(
                {
                    "expert_i": expert_i,
                    "expert_j": expert_j,
                    "count": pair_count,
                    "probability": (
                        float(dominant_layer_count / dominant_layer_events)
                        if dominant_layer_events > 0
                        else 0.0
                    ),
                    "dominant_layer_name": dominant_layer_name,
                    "dominant_layer_id": dominant_layer_id,
                    "dominant_layer_count": dominant_layer_count,
                    "dominant_layer_events": dominant_layer_events,
                }
            )

    top_pairs.sort(
        key=lambda item: (
            -item["probability"],
            -item["dominant_layer_count"],
            item["dominant_layer_id"],
            item["expert_i"],
            item["expert_j"],
        )
    )

    top_experts: list[dict[str, Any]] = []
    for expert_id in range(num_routed_experts):
        dominant_layer_name = ""
        dominant_layer_id = -1
        dominant_layer_count = 0
        dominant_layer_events = 0
        dominant_layer_total_assignments = 0

        for layer_name in layer_names:
            expert_count = int(layer_counts[layer_name][expert_id].item())
            if expert_count <= 0:
                continue
            layer_id = int(layer_name_to_id.get(layer_name, -1))
            if (
                expert_count > dominant_layer_count
                or (
                    expert_count == dominant_layer_count
                    and dominant_layer_id >= 0
                    and layer_id < dominant_layer_id
                )
                or (
                    expert_count == dominant_layer_count
                    and dominant_layer_id < 0
                )
            ):
                dominant_layer_name = layer_name
                dominant_layer_id = layer_id
                dominant_layer_count = expert_count
                dominant_layer_events = int(layer_event_counts.get(layer_name, 0))
                dominant_layer_total_assignments = int(
                    layer_counts[layer_name].sum().item()
                )

        if dominant_layer_count == 0:
            continue

        top_experts.append(
            {
                "expert_id": expert_id,
                "probability": (
                    float(dominant_layer_count / dominant_layer_total_assignments)
                    if dominant_layer_total_assignments > 0
                    else 0.0
                ),
                "dominant_layer_name": dominant_layer_name,
                "dominant_layer_id": dominant_layer_id,
                "dominant_layer_count": dominant_layer_count,
                "dominant_layer_events": dominant_layer_events,
                "dominant_layer_total_assignments": dominant_layer_total_assignments,
            }
        )

    top_experts.sort(
        key=lambda item: (
            -item["probability"],
            -item["dominant_layer_count"],
            item["dominant_layer_id"],
            item["expert_id"],
        )
    )
    layer_transition_stats = _build_layer_transition_statistics(
        routing_trace,
        layer_names=layer_names,
        layer_name_to_id=layer_name_to_id,
        num_routed_experts=num_routed_experts,
    )

    return {
        "num_routed_experts": num_routed_experts,
        "results": results,
        "layer_names": layer_names,
        "layer_expert_matrix": layer_expert_matrix,
        "top_experts": top_experts,
        "top_pairs": top_pairs,
        "layer_transition_stats": layer_transition_stats,
        "total_routing_events": total_routing_events,
    }


def _print_pair_correlation_summary(
    *,
    top_pairs: list[dict[str, Any]],
    total_routing_events: int,
) -> None:
    print("Spatial correlation")
    print(
        "   P(Ei, Ej | layer) = pair co-activation count divided by the number of "
        "routing events in that layer."
    )
    print(f"   Routing events traced      : {total_routing_events}")
    if not top_pairs:
        print("   No expert pairs were co-selected in the traced workload.")
        return
    for pair in top_pairs[:15]:
        calculation = (
            f"{pair['dominant_layer_count']}/{pair['dominant_layer_events']}"
            if pair.get("dominant_layer_events", 0) > 0
            else "0/0"
        )
        layer_suffix = (
            f" layer=L{pair['dominant_layer_id']:02d}"
            if pair.get("dominant_layer_id", -1) >= 0
            else ""
        )
        print(
            f"   (E{pair['expert_i']:02d}, E{pair['expert_j']:02d}) -> "
            f"{pair['probability']:.6f} ({calculation}){layer_suffix}"
        )


def _print_single_expert_summary(
    *,
    top_experts: list[dict[str, Any]],
) -> None:
    print("Single expert usage")
    print(
        "   P(Ei | layer) = expert activation count divided by the total number "
        "of expert assignments in that layer."
    )
    if not top_experts:
        print("   No expert activations were collected in the traced workload.")
        return
    for expert in top_experts[:15]:
        calculation = (
            f"{expert['dominant_layer_count']}/{expert['dominant_layer_total_assignments']}"
            if expert.get("dominant_layer_total_assignments", 0) > 0
            else "0/0"
        )
        layer_suffix = (
            f" layer=L{expert['dominant_layer_id']:02d}"
            if expert.get("dominant_layer_id", -1) >= 0
            else ""
        )
        print(
            f"   E{expert['expert_id']:02d} -> "
            f"{expert['probability']:.6f} ({calculation}){layer_suffix}"
        )


def _print_layer_transition_summary(
    *,
    transitions: list[dict[str, Any]],
) -> None:
    print("Layer transition routing")
    print(
        "   P(Ln+1 expert x | Ln expert y) = count(y selected in layer n and x "
        "selected in layer n+1 for the same token) divided by count(y selected "
        "in layer n)."
    )
    if not transitions:
        print("   No adjacent-layer transitions were collected in the traced workload.")
        return

    for transition in transitions[:10]:
        source_layer_id = int(transition["source_layer_id"])
        target_layer_id = int(transition["target_layer_id"])
        source_event_count = int(transition.get("source_event_count", 0))
        probabilities = transition.get("probabilities")
        active_sources = 0
        if isinstance(probabilities, torch.Tensor) and probabilities.numel() > 0:
            source_counts = transition.get("source_counts")
            if isinstance(source_counts, torch.Tensor):
                active_sources = int((source_counts > 0).sum().item())
        print(
            f"   L{source_layer_id:02d} -> L{target_layer_id:02d}: "
            f"{source_event_count} source expert activations across "
            f"{active_sources} source experts"
        )


def run_expert_routing_analysis(
    trace_path: Path,
    *,
    model_id: str | None = None,
    configured_top_k: int | None = None,
    original_top_k: int | None = None,
    execution_device: str | None = None,
    prompts_processed: int | None = None,
    input_token_count: int | None = None,
    output_token_count: int | None = None,
    output_dir: Path | None = None,
) -> None:
    trace_path = trace_path.resolve()
    analysis_output_dir = output_dir.resolve() if output_dir is not None else trace_path.parent.resolve()
    payload = load_routing_trace(trace_path)
    routing_trace = payload["routing_trace"]
    metadata = payload["metadata"]
    model_id = model_id or metadata.get("model_id", "unknown")
    configured_top_k = (
        configured_top_k
        if configured_top_k is not None
        else int(metadata.get("configured_top_k", 0))
    )
    original_top_k = (
        original_top_k
        if original_top_k is not None
        else int(metadata.get("original_top_k", configured_top_k))
    )
    execution_device = execution_device or metadata.get("execution_device", "unknown")
    prompts_processed = (
        prompts_processed
        if prompts_processed is not None
        else int(metadata.get("prompts_processed", len(routing_trace)))
    )
    input_token_count = (
        input_token_count
        if input_token_count is not None
        else int(metadata.get("input_token_count", 0))
    )
    output_token_count = (
        output_token_count
        if output_token_count is not None
        else int(metadata.get("output_token_count", 0))
    )
    stats = _build_expert_routing_statistics(routing_trace)
    results = stats["results"]

    if not results:
        print("[EXPERT_ROUTING] No routing data was collected.")
        return

    total_assignments = sum(int(result["counts"].sum().item()) for result in results)

    print("=" * 96)
    print("Per-Layer Crossbar Workload Skewness (CWS) Report")
    print("=" * 96)
    print(f"Model                         : {model_id}")
    print(f"Configured top-k             : {configured_top_k}")
    print(f"Original config top-k        : {original_top_k}")
    print(f"Routed experts counted       : {stats['num_routed_experts']}")
    print(f"Execution device             : {execution_device}")
    print(f"Prompts processed            : {prompts_processed}")
    print(f"Input token count            : {input_token_count}")
    print(f"Output token count           : {output_token_count}")
    print(f"Total expert assignments     : {total_assignments}")
    print("-" * 96)
    for result in results:
        print(f"{result['layer_name']:<24} CWS={result['workload_skewness']:.6f}")
    print("=" * 96)
    print("Expert-Centric Routing Report")
    print("=" * 96)
    print(f"Loaded raw routing trace      : {trace_path}")
    flat_records = [
        record
        for prompt_records in routing_trace.values()
        for record in prompt_records
    ]
    heatmap_path = save_expert_heatmap_plot(
        layer_expert_matrix=stats["layer_expert_matrix"],
        layer_names=stats["layer_names"],
        num_routed_experts=stats["num_routed_experts"],
        model_id=model_id,
        configured_top_k=configured_top_k,
        output_dir=analysis_output_dir,
    )
    if heatmap_path is not None:
        print(f"Saved expert heatmap plot     : {heatmap_path}")
        print("Heatmap axes                  : y=layer, x=expert id")
    spatial_heatmap_paths = generate_individual_spatial_heatmaps(
        flat_records,
        analysis_output_dir,
        stats["num_routed_experts"],
    )
    if spatial_heatmap_paths:
        print(
            f"Saved spatial heatmaps        : {len(spatial_heatmap_paths)} files in "
            f"{(analysis_output_dir / 'spatial_correlation_heatmaps').resolve()}"
        )
    transition_heatmap_paths = generate_layer_transition_heatmaps(
        stats["layer_transition_stats"],
        analysis_output_dir,
        stats["num_routed_experts"],
    )
    if transition_heatmap_paths:
        print(
            f"Saved transition heatmaps     : {len(transition_heatmap_paths)} files in "
            f"{(analysis_output_dir / 'layer_transition_heatmaps').resolve()}"
        )
        print("Transition heatmap axes       : y=layer n expert id, x=layer n+1 expert id")
    transition_umap_paths = plot_transition_umap(
        stats["layer_transition_stats"],
        analysis_output_dir,
        stats["num_routed_experts"],
    )
    if transition_umap_paths:
        print(
            f"Saved transition UMAP plots   : {len(transition_umap_paths)} files in "
            f"{(analysis_output_dir / 'layer_transition_umap').resolve()}"
        )
    load_entropy_path = plot_expert_load_and_entropy(
        flat_records,
        analysis_output_dir,
        stats["num_routed_experts"],
    )
    if load_entropy_path is not None:
        print(f"Saved load/entropy plot       : {load_entropy_path}")
    print("-" * 96)
    _print_single_expert_summary(
        top_experts=stats["top_experts"],
    )
    print("-" * 96)
    _print_pair_correlation_summary(
        top_pairs=stats["top_pairs"],
        total_routing_events=stats["total_routing_events"],
    )
    print("-" * 96)
    _print_layer_transition_summary(
        transitions=stats["layer_transition_stats"],
    )
    print("=" * 96)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate expert routing metrics from a saved expert_traces_raw.pkl file."
    )
    parser.add_argument("trace_path", help="Path to expert_traces_raw.pkl.")
    parser.add_argument("--model-id", help="Model identifier for reporting.")
    parser.add_argument("--configured-top-k", type=int)
    parser.add_argument("--original-top-k", type=int)
    parser.add_argument("--execution-device")
    parser.add_argument("--prompts-processed", type=int)
    parser.add_argument("--input-token-count", type=int)
    parser.add_argument("--output-token-count", type=int)
    parser.add_argument("--output-dir", help="Directory where analysis artifacts should be written.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_expert_routing_analysis(
        Path(args.trace_path),
        model_id=args.model_id,
        configured_top_k=args.configured_top_k,
        original_top_k=args.original_top_k,
        execution_device=args.execution_device,
        prompts_processed=args.prompts_processed,
        input_token_count=args.input_token_count,
        output_token_count=args.output_token_count,
        output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
