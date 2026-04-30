from __future__ import annotations

import math
import os
import re
from pathlib import Path

import torch

from src.tracing.routing_trace import RoutingTraceRecord


def save_expert_heatmap_plot(
    *,
    layer_expert_matrix: torch.Tensor,
    layer_names: list[str],
    num_routed_experts: int,
    model_id: str,
    configured_top_k: int,
    output_dir: Path,
) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[EXPERT_ROUTING] matplotlib not available. Skipping saved expert heatmap plot.")
        return None

    if layer_expert_matrix.numel() == 0:
        return None

    safe_model_name = re.sub(r"[^a-zA-Z0-9]+", "_", model_id).strip("_").lower()
    output_path = output_dir / (
        f"{safe_model_name}_topk_{configured_top_k}_layer_expert_usage_heatmap.png"
    )

    heatmap = layer_expert_matrix.to(torch.float64).numpy()
    figure_width = max(14.0, 0.26 * num_routed_experts)
    figure_height = max(7.0, 0.42 * len(layer_names))
    figure, axis = plt.subplots(figsize=(figure_width, figure_height))
    image = axis.imshow(heatmap, cmap="inferno", interpolation="nearest", aspect="auto")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Expert usage count", rotation=90)

    axis.set_title("Layer-By-Expert Routing Heatmap", pad=14)
    axis.set_xlabel("Expert id")
    axis.set_ylabel("Layer")
    axis.set_xticks(range(num_routed_experts))
    axis.set_xticklabels([str(expert_id) for expert_id in range(num_routed_experts)])
    axis.set_yticks(range(len(layer_names)))
    axis.set_yticklabels(layer_names)
    axis.tick_params(axis="x", labelrotation=90, labelsize=7)
    axis.tick_params(axis="y", labelsize=8)

    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def generate_individual_spatial_heatmaps(
    records: list[RoutingTraceRecord],
    base_output_dir: Path | str,
    num_experts: int,
) -> list[Path]:
    if num_experts <= 0:
        print("[EXPERT_ROUTING] Invalid expert count. Skipping spatial correlation heatmaps.")
        return []

    if not records:
        print("[EXPERT_ROUTING] No routing records available. Skipping spatial correlation heatmaps.")
        return []

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
    except ImportError:
        print("[EXPERT_ROUTING] matplotlib/numpy/seaborn not available. Skipping spatial correlation heatmaps.")
        return []

    heatmap_dir = Path(base_output_dir) / "spatial_correlation_heatmaps"
    os.makedirs(heatmap_dir, exist_ok=True)

    unique_layer_ids = sorted({record.layer_id for record in records if record.layer_id >= 0})
    if not unique_layer_ids:
        print("[EXPERT_ROUTING] No valid layer ids found. Skipping spatial correlation heatmaps.")
        return []

    filename_width = max(2, len(str(max(unique_layer_ids))))
    saved_paths: list[Path] = []
    layer_matrices: dict[int, torch.Tensor] = {}
    global_max = 0.0

    for layer_id in unique_layer_ids:
        layer_records = [record for record in records if record.layer_id == layer_id]
        if not layer_records:
            continue

        coactivation_counts = torch.zeros((num_experts, num_experts), dtype=torch.float64)
        valid_event_count = 0

        for record in layer_records:
            selected_experts = sorted(
                {
                    int(expert_id)
                    for expert_id in record.selected_experts
                    if 0 <= int(expert_id) < num_experts
                }
            )
            if not selected_experts:
                continue

            valid_event_count += 1
            for left_index, expert_i in enumerate(selected_experts):
                for expert_j in selected_experts[left_index + 1 :]:
                    coactivation_counts[expert_i, expert_j] += 1.0
                    coactivation_counts[expert_j, expert_i] += 1.0

        if valid_event_count == 0:
            print(f"[EXPERT_ROUTING] Layer {layer_id} has no valid routing events. Skipping heatmap.")
            continue

        coactivation_probabilities = coactivation_counts / float(valid_event_count)
        layer_matrices[layer_id] = coactivation_probabilities.clone()
        global_max = max(global_max, float(coactivation_probabilities.max().item()))

    if not layer_matrices:
        print("[EXPERT_ROUTING] No layer matrices were generated. Skipping spatial correlation heatmaps.")
        return []

    for layer_id in unique_layer_ids:
        if layer_id not in layer_matrices:
            continue

        figure = plt.figure(figsize=(12, 10))
        axis = figure.add_subplot(111)
        matrix = layer_matrices[layer_id].numpy()
        mask = np.triu(np.ones_like(matrix, dtype=bool))
        sns.heatmap(
            matrix,
            ax=axis,
            cmap="magma",
            mask=mask,
            square=True,
            vmin=0.0,
            vmax=global_max,
            cbar_kws={"label": "Co-activation probability"},
        )
        axis.set_title(f"Layer {layer_id} - Expert Co-Activation Matrix", pad=14)
        axis.set_xlabel("Expert id")
        axis.set_ylabel("Expert id")

        output_path = heatmap_dir / f"layer_{layer_id:0{filename_width}d}_spatial.png"
        figure.tight_layout()
        figure.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(figure)
        saved_paths.append(output_path)

    return saved_paths


def generate_layer_transition_heatmaps(
    transitions: list[dict[str, object]],
    base_output_dir: Path | str,
    num_experts: int,
) -> list[Path]:
    if num_experts <= 0:
        print("[EXPERT_ROUTING] Invalid expert count. Skipping transition heatmaps.")
        return []

    if not transitions:
        print("[EXPERT_ROUTING] No layer transitions available. Skipping transition heatmaps.")
        return []

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("[EXPERT_ROUTING] matplotlib/seaborn not available. Skipping transition heatmaps.")
        return []

    heatmap_dir = Path(base_output_dir) / "layer_transition_heatmaps"
    os.makedirs(heatmap_dir, exist_ok=True)

    max_layer_id = max(
        int(transition["target_layer_id"])
        for transition in transitions
        if int(transition["target_layer_id"]) >= 0
    )
    filename_width = max(2, len(str(max_layer_id)))
    saved_paths: list[Path] = []

    for transition in transitions:
        source_layer_id = int(transition["source_layer_id"])
        target_layer_id = int(transition["target_layer_id"])
        probabilities = transition["probabilities"]
        if not isinstance(probabilities, torch.Tensor) or probabilities.numel() == 0:
            continue

        figure = plt.figure(figsize=(12, 10))
        axis = figure.add_subplot(111)
        sns.heatmap(
            probabilities.numpy(),
            ax=axis,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            square=True,
            cbar_kws={"label": "P(next-layer expert | source expert)"},
        )
        axis.set_title(
            f"Layer {source_layer_id} -> Layer {target_layer_id} Expert Transition",
            pad=14,
        )
        axis.set_xlabel(f"Layer {target_layer_id} expert id")
        axis.set_ylabel(f"Layer {source_layer_id} expert id")

        output_path = heatmap_dir / (
            f"layer_{source_layer_id:0{filename_width}d}_to_"
            f"{target_layer_id:0{filename_width}d}_transition.png"
        )
        figure.tight_layout()
        figure.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(figure)
        saved_paths.append(output_path)

    return saved_paths


def plot_transition_umap(
    transitions: list[dict[str, object]],
    base_output_dir: Path | str,
    num_experts: int,
) -> list[Path]:
    if num_experts <= 0:
        print("[EXPERT_ROUTING] Invalid expert count. Skipping transition UMAP plots.")
        return []

    if not transitions:
        print("[EXPERT_ROUTING] No layer transitions available. Skipping transition UMAP plots.")
        return []

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[EXPERT_ROUTING] matplotlib/numpy not available. Skipping transition UMAP plots.")
        return []

    try:
        import umap
    except ImportError:
        print(
            "[EXPERT_ROUTING] umap-learn not installed. Skipping transition UMAP "
            "plots. Install with: pip install umap-learn"
        )
        return []

    umap_dir = Path(base_output_dir) / "layer_transition_umap"
    os.makedirs(umap_dir, exist_ok=True)

    max_layer_id = max(
        int(transition["target_layer_id"])
        for transition in transitions
        if int(transition["target_layer_id"]) >= 0
    )
    filename_width = max(2, len(str(max_layer_id)))
    saved_paths: list[Path] = []

    for transition in transitions:
        source_layer_id = int(transition["source_layer_id"])
        target_layer_id = int(transition["target_layer_id"])
        probabilities = transition.get("probabilities")
        if not isinstance(probabilities, torch.Tensor) or probabilities.numel() == 0:
            continue

        matrix = probabilities.to(torch.float64).cpu().numpy()
        row_sums = matrix.sum(axis=1)
        active_mask = row_sums > 0.0
        if not np.any(active_mask):
            print(
                f"[EXPERT_ROUTING] Layer {source_layer_id}->{target_layer_id} "
                "has no active source experts. Skipping UMAP."
            )
            continue

        normalized_rows = np.divide(
            matrix[active_mask],
            row_sums[active_mask, None],
            out=np.zeros_like(matrix[active_mask]),
            where=row_sums[active_mask, None] > 0.0,
        )
        active_expert_ids = np.flatnonzero(active_mask)
        nonzero_probabilities = np.where(normalized_rows > 0.0, normalized_rows, 1.0)
        entropy_values = -np.sum(
            np.where(
                normalized_rows > 0.0,
                normalized_rows * np.log(nonzero_probabilities),
                0.0,
            ),
            axis=1,
        )

        if normalized_rows.shape[0] == 1:
            embedding = np.zeros((1, 2), dtype=np.float64)
        else:
            requested_neighbors = min(15, num_experts - 1)
            n_neighbors = min(requested_neighbors, normalized_rows.shape[0] - 1)
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=0.1,
                metric="cosine",
                random_state=0,
            )
            embedding = reducer.fit_transform(normalized_rows)

        source_counts = transition.get("source_counts")
        color_label = "Source expert load"
        if isinstance(source_counts, torch.Tensor) and source_counts.numel() >= num_experts:
            colors = source_counts.to(torch.float64).cpu().numpy()[active_expert_ids]
        else:
            colors = active_expert_ids
            color_label = "Source expert id"

        figure, axis = plt.subplots(figsize=(10, 8))
        scatter = axis.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=colors,
            cmap="viridis",
            s=70,
            alpha=0.9,
            edgecolors="black",
            linewidths=0.35,
        )
        colorbar = figure.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label(color_label, rotation=90)

        if num_experts <= 64:
            for expert_id, (x_position, y_position) in zip(active_expert_ids, embedding):
                axis.annotate(
                    str(int(expert_id)),
                    (x_position, y_position),
                    xytext=(4, 3),
                    textcoords="offset points",
                    fontsize=8,
                )

        entropy_mean = float(entropy_values.mean()) if entropy_values.size > 0 else 0.0
        entropy_max = float(entropy_values.max()) if entropy_values.size > 0 else 0.0
        axis.set_title(
            f"Layer {source_layer_id} -> Layer {target_layer_id} "
            f"Transition UMAP (entropy mean={entropy_mean:.3f}, max={entropy_max:.3f})",
            pad=14,
        )
        axis.set_xlabel("UMAP 1")
        axis.set_ylabel("UMAP 2")
        axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.35)

        output_path = umap_dir / (
            f"layer_{source_layer_id:0{filename_width}d}_to_"
            f"{target_layer_id:0{filename_width}d}_transition_umap.png"
        )
        figure.tight_layout()
        figure.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(figure)
        saved_paths.append(output_path)

    return saved_paths


def plot_expert_load_and_entropy(
    records: list[RoutingTraceRecord],
    output_dir: Path | str,
    num_experts: int,
) -> Path | None:
    if num_experts <= 0:
        print("[EXPERT_ROUTING] Invalid expert count. Skipping expert load and entropy plot.")
        return None

    if not records:
        print("[EXPERT_ROUTING] No routing records available. Skipping expert load and entropy plot.")
        return None

    valid_layer_ids = [record.layer_id for record in records if record.layer_id >= 0]
    if not valid_layer_ids:
        print("[EXPERT_ROUTING] No valid layer ids found. Skipping expert load and entropy plot.")
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
    except ImportError:
        print("[EXPERT_ROUTING] matplotlib/numpy/seaborn not available. Skipping expert load and entropy plot.")
        return None

    max_layer_id = max(valid_layer_ids)
    num_layers = max_layer_id + 1
    load_matrix = np.zeros((num_layers, num_experts), dtype=np.float64)

    for record in records:
        if record.layer_id < 0 or record.layer_id >= num_layers:
            continue
        for expert_id in record.selected_experts:
            expert_index = int(expert_id)
            if 0 <= expert_index < num_experts:
                load_matrix[record.layer_id, expert_index] += 1.0

    row_sums = load_matrix.sum(axis=1, keepdims=True)
    normalized_matrix = np.divide(
        load_matrix,
        row_sums,
        out=np.zeros_like(load_matrix),
        where=row_sums > 0,
    )

    entropy_values = np.zeros(num_layers, dtype=np.float64)
    for layer_id in range(num_layers):
        probabilities = normalized_matrix[layer_id]
        nonzero_probabilities = probabilities[probabilities > 0.0]
        if nonzero_probabilities.size == 0:
            continue
        entropy_values[layer_id] = -np.sum(
            nonzero_probabilities * np.log2(nonzero_probabilities)
        )

    figure, (ax_heat, ax_ent) = plt.subplots(
        1,
        2,
        figsize=(16, 10),
        gridspec_kw={"width_ratios": [5, 1]},
        sharey=True,
    )
    heatmap_max = float(normalized_matrix.max()) if normalized_matrix.size > 0 else 0.0

    sns.heatmap(
        normalized_matrix,
        ax=ax_heat,
        cmap="magma",
        vmin=0.0,
        vmax=heatmap_max if heatmap_max > 0.0 else 1.0,
        cbar_kws={"label": "Activation probability"},
    )
    ax_heat.set_title("Expert Load Heatmap", pad=14)
    ax_heat.set_xlabel("Expert ID")
    ax_heat.set_ylabel("Layer ID")

    y_positions = np.arange(num_layers) + 0.5
    ax_ent.plot(entropy_values, y_positions, color="red", marker="o")
    ax_ent.axvline(
        x=math.log2(num_experts),
        color="black",
        linestyle="--",
        linewidth=1.2,
    )
    ax_ent.set_title("Routing Entropy", pad=14)
    ax_ent.set_xlabel("Entropy (bits)")
    ax_ent.invert_yaxis()
    ax_ent.grid(axis="x", linestyle=":", alpha=0.35)

    figure.tight_layout()
    output_path = Path(output_dir) / "expert_load_and_entropy.png"
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path
