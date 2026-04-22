from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class CWSTracker:
    """
    Track Crossbar Workload Skewness (CWS) for Qwen MoE router layers.

    Design goals:
    - discover router `gate` modules dynamically
    - accumulate expert usage counts across many forward passes
    - maintain counts per layer rather than collapsing all layers together
    - report layer-local skew using coefficient of variation: std / mean
    """

    def __init__(self, model: nn.Module, top_k: int) -> None:
        self.model = model
        self.top_k = top_k
        self.handles: list[Any] = []
        self.enabled = True
        self.num_routed_experts = self._infer_num_routed_experts()
        self.layer_counts: dict[str, torch.Tensor] = {}
        self.global_expert_counts = torch.zeros(
            self.num_routed_experts,
            dtype=torch.long,
        )
        self.expert_pair_counts = torch.zeros(
            (self.num_routed_experts, self.num_routed_experts),
            dtype=torch.long,
        )
        self.total_routing_events = 0

    def _infer_num_routed_experts(self) -> int:
        """
        Infer the number of routed experts from the loaded model config.

        For Qwen/Qwen1.5-MoE-A2.7B, this is 60 routed experts. Shared experts
        are not part of the routed top-k histogram and are intentionally ignored.
        """

        config = getattr(self.model, "config", None)
        num_experts = int(getattr(config, "num_experts", 0))
        if num_experts <= 0:
            raise ValueError(
                "Unable to determine the number of routed experts from model.config."
            )
        return num_experts

    def _is_router_gate(self, module_name: str, module: nn.Module) -> bool:
        """
        Identify Qwen MoE gate modules robustly.

        We accept either:
        - a plain `nn.Linear` named `gate` that outputs logits over routed experts
        - a dedicated router module named `gate` that stores router metadata
        """

        leaf_name = module_name.rsplit(".", maxsplit=1)[-1]
        if leaf_name != "gate":
            return False

        if (
            isinstance(module, nn.Linear)
            and module.out_features == self.num_routed_experts
        ):
            return True

        required_attrs = ("top_k", "num_experts", "weight")
        return all(hasattr(module, attr) for attr in required_attrs)

    def _extract_layer_name(self, module_name: str) -> str:
        """
        Convert a gate module path into a stable human-readable layer label.

        Typical Qwen paths look like `model.layers.5.mlp.gate`.
        """

        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", module_name)
        if match is not None:
            return f"layer {int(match.group(1))}"
        return module_name.rsplit(".gate", maxsplit=1)[0]

    def _extract_selected_experts(
        self, output: Any, router_top_k: int
    ) -> torch.Tensor | None:
        """
        Recover routed expert ids from a gate output.

        Depending on the exact router implementation, the hook may observe:
        - a tensor of raw logits with shape [..., num_routed_experts]
        - a tuple containing selected expert indices directly

        In both cases, we return a tensor of selected expert ids shaped
        [..., router_top_k], containing only the 60 routed experts.
        """

        if isinstance(output, tuple):
            for item in reversed(output):
                if not isinstance(item, torch.Tensor):
                    continue
                if (
                    item.dtype in (torch.int32, torch.int64)
                    and item.shape[-1] == router_top_k
                ):
                    return item

            for item in output:
                if not isinstance(item, torch.Tensor):
                    continue
                if item.shape[-1] == self.num_routed_experts:
                    probabilities = torch.softmax(item.float(), dim=-1)
                    return torch.topk(probabilities, k=router_top_k, dim=-1).indices
            return None

        if isinstance(output, torch.Tensor):
            if (
                output.dtype in (torch.int32, torch.int64)
                and output.shape[-1] == router_top_k
            ):
                return output
            if output.shape[-1] == self.num_routed_experts:
                probabilities = torch.softmax(output.float(), dim=-1)
                return torch.topk(probabilities, k=router_top_k, dim=-1).indices

        return None

    def register_hooks(self) -> None:
        """
        Attach forward hooks to every discovered router `gate` module.
        """

        for module_name, module in self.model.named_modules():
            if not self._is_router_gate(module_name, module):
                continue

            layer_name = self._extract_layer_name(module_name)
            if layer_name not in self.layer_counts:
                self.layer_counts[layer_name] = torch.zeros(
                    self.num_routed_experts,
                    dtype=torch.long,
                )

            def make_hook(current_layer_name: str, current_module: nn.Module):
                def hook(_: nn.Module, __: tuple[Any, ...], output: Any) -> None:
                    if not self.enabled:
                        return
                    router_top_k = int(getattr(current_module, "top_k", self.top_k))
                    selected_experts = self._extract_selected_experts(
                        output, router_top_k
                    )
                    if selected_experts is None:
                        return

                    event_indices = selected_experts.reshape(-1, router_top_k).cpu()
                    self.total_routing_events += int(event_indices.shape[0])

                    # Flatten all token-level assignments from this layer and
                    # tally only the routed expert ids 0..59.
                    flat_indices = event_indices.reshape(-1)
                    counts = torch.bincount(
                        flat_indices,
                        minlength=self.num_routed_experts,
                    ).cpu()
                    self.layer_counts[current_layer_name].add_(counts)
                    self.global_expert_counts.add_(counts)

                    # Treat each (token, layer) selection as one routing event.
                    # Off-diagonal entries count joint usage of expert pairs in
                    # the exact same event; diagonal entries count marginal usage.
                    for event in event_indices.tolist():
                        unique_experts = sorted(
                            {
                                int(expert_id)
                                for expert_id in event
                                if 0 <= int(expert_id) < self.num_routed_experts
                            }
                        )
                        for expert_id in unique_experts:
                            self.expert_pair_counts[expert_id, expert_id] += 1
                        for left_index, expert_i in enumerate(unique_experts):
                            for expert_j in unique_experts[left_index + 1 :]:
                                self.expert_pair_counts[expert_i, expert_j] += 1
                                self.expert_pair_counts[expert_j, expert_i] += 1

                return hook

            self.handles.append(
                module.register_forward_hook(make_hook(layer_name, module))
            )

        if not self.handles:
            raise RuntimeError("No Qwen MoE gate layers were found for CWS tracking.")

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def remove_hooks(self) -> None:
        """Detach all registered hooks."""

        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    @staticmethod
    def _usage_frequency_summary(counts: torch.Tensor) -> list[tuple[int, int]]:
        """
        Summarize how many experts were used exactly n times in a layer.

        Example:
        - if 10 experts were selected 0 times and 5 experts were selected
          3 times, the summary contains `(0, 10)` and `(3, 5)`.
        """

        usage_frequencies: dict[int, int] = {}
        for usage_count in counts.tolist():
            usage_value = int(usage_count)
            usage_frequencies[usage_value] = usage_frequencies.get(usage_value, 0) + 1
        return sorted(usage_frequencies.items(), key=lambda item: item[0])

    def compute_layer_results(self) -> list[dict[str, Any]]:
        """
        Compute per-layer CWS values from the accumulated expert histograms.
        """

        def sort_key(name: str) -> tuple[int, str]:
            if name.startswith("layer "):
                return int(name.split()[-1]), name
            return 10**9, name

        results: list[dict[str, Any]] = []
        for layer_name in sorted(self.layer_counts.keys(), key=sort_key):
            counts = self.layer_counts[layer_name].to(torch.float64)
            mean_count = float(counts.mean().item())
            std_count = float(counts.std(unbiased=False).item())
            cws = float(std_count / mean_count) if mean_count > 0 else float("inf")

            results.append(
                {
                    "layer_name": layer_name,
                    "counts": self.layer_counts[layer_name],
                    "usage_frequency_summary": self._usage_frequency_summary(
                        self.layer_counts[layer_name]
                    ),
                    "cws": cws,
                }
            )
        return results

    def compute_expert_centric_results(self) -> dict[str, Any]:
        """
        Compute global expert-centric statistics over the flattened token-layer trace.

        The trace order is conceptually:
        token0-layer0, token0-layer1, ..., token0-layerN,
        token1-layer0, ..., last-token-last-layer

        Each trace element is one routing event containing the top-k selected experts.
        """

        grid_side = int(math.ceil(math.sqrt(self.num_routed_experts)))
        pair_probabilities = self.expert_pair_counts.to(torch.float64)
        if self.total_routing_events > 0:
            pair_probabilities /= float(self.total_routing_events)

        top_pairs: list[dict[str, Any]] = []
        for expert_i in range(self.num_routed_experts):
            for expert_j in range(expert_i + 1, self.num_routed_experts):
                pair_count = int(self.expert_pair_counts[expert_i, expert_j].item())
                if pair_count == 0:
                    continue
                top_pairs.append(
                    {
                        "expert_i": expert_i,
                        "expert_j": expert_j,
                        "count": pair_count,
                        "probability": float(pair_probabilities[expert_i, expert_j].item()),
                    }
                )

        top_pairs.sort(
            key=lambda item: (-item["probability"], -item["count"], item["expert_i"], item["expert_j"])
        )

        return {
            "grid_side": grid_side,
            "global_expert_counts": self.global_expert_counts.clone(),
            "pair_probability_matrix": pair_probabilities,
            "top_pairs": top_pairs,
            "total_routing_events": self.total_routing_events,
        }

    @staticmethod
    def _format_heatmap_row(cells: list[str]) -> str:
        return " | ".join(cells)

    def _print_expert_heatmap(self, counts: torch.Tensor, grid_side: int) -> None:
        print("Spatial heatmap (absolute expert usage counts)")
        cell_width = 12
        cells: list[str] = []
        for expert_id in range(self.num_routed_experts):
            cells.append(f"E{expert_id:02d}:{int(counts[expert_id].item()):>6}")
        while len(cells) < grid_side * grid_side:
            cells.append(" " * cell_width)
        for row_start in range(0, len(cells), grid_side):
            row = cells[row_start : row_start + grid_side]
            print(f"   {self._format_heatmap_row(row)}")

    def _save_expert_heatmap_plot(
        self,
        counts: torch.Tensor,
        grid_side: int,
        model_id: str,
    ) -> Path | None:
        """
        Save a matplotlib heatmap of absolute expert usage counts.

        Experts are laid out on an MxM grid, where M = ceil(sqrt(num_experts)).
        Empty trailing cells are masked so the color scale only reflects real
        expert tiles.
        """

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print(
                "[CWS] matplotlib/numpy not available. Skipping saved expert heatmap plot."
            )
            return None

        safe_model_name = re.sub(r"[^a-zA-Z0-9]+", "_", model_id).strip("_").lower()
        output_path = Path(f"{safe_model_name}_expert_usage_heatmap.png")

        heatmap = np.full((grid_side, grid_side), np.nan, dtype=float)
        for expert_id, usage_count in enumerate(counts.tolist()):
            row_index = expert_id // grid_side
            col_index = expert_id % grid_side
            heatmap[row_index, col_index] = float(usage_count)

        masked_heatmap = np.ma.masked_invalid(heatmap)
        cmap = plt.cm.inferno.copy()
        cmap.set_bad(color="#f4f4f4")

        figure, axis = plt.subplots(figsize=(1.45 * grid_side, 1.35 * grid_side))
        image = axis.imshow(masked_heatmap, cmap=cmap, interpolation="nearest")
        colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label("Absolute expert usage count", rotation=90)

        axis.set_title("Expert-Centric Spatial Heatmap", pad=14)
        axis.set_xlabel("Grid column")
        axis.set_ylabel("Grid row")
        axis.set_xticks(range(grid_side))
        axis.set_yticks(range(grid_side))

        finite_values = masked_heatmap.compressed()
        text_threshold = float(finite_values.max() * 0.55) if finite_values.size else 0.0

        for expert_id in range(self.num_routed_experts):
            row_index = expert_id // grid_side
            col_index = expert_id % grid_side
            usage_count = int(counts[expert_id].item())
            text_color = "white" if usage_count >= text_threshold else "black"
            axis.text(
                col_index,
                row_index,
                f"E{expert_id:02d}\n{usage_count}",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
                fontweight="semibold",
            )

        figure.tight_layout()
        figure.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(figure)
        return output_path

    def _print_pair_correlation_summary(
        self,
        top_pairs: list[dict[str, Any]],
        total_routing_events: int,
    ) -> None:
        print("Spatial correlation")
        print(
            "   P(Ei, Ej) = probability that experts Ei and Ej are selected in the same "
            "token-layer routing event."
        )
        print(f"   Routing events traced      : {total_routing_events}")
        if not top_pairs:
            print("   No expert pairs were co-selected in the traced workload.")
            return
        for pair in top_pairs[:15]:
            print(
                f"   (E{pair['expert_i']:02d}, E{pair['expert_j']:02d}) -> "
                f"{pair['probability']:.6f} ({pair['count']}/{total_routing_events})"
            )

    def print_report(
        self,
        *,
        model_id: str,
        configured_top_k: int,
        original_top_k: int,
        execution_device: str,
        prompts_processed: int,
        input_token_count: int,
    ) -> None:
        """
        Print a formatted per-layer CWS report.
        """

        results = self.compute_layer_results()
        if not results:
            print("[CWS] No routing data was collected.")
            return

        total_assignments = sum(
            int(result["counts"].sum().item()) for result in results
        )
        expert_centric_results = self.compute_expert_centric_results()

        print("=" * 96)
        print("Per-Layer Crossbar Workload Skewness (CWS) Report")
        print("=" * 96)
        print(f"Model                         : {model_id}")
        print(f"Configured top-k             : {configured_top_k}")
        print(f"Original config top-k        : {original_top_k}")
        print(f"Routed experts counted       : {self.num_routed_experts}")
        print(f"MoE routers hooked           : {len(self.handles)}")
        print(f"Execution device             : {execution_device}")
        print(f"Prompts processed            : {prompts_processed}")
        print(f"Input token count            : {input_token_count}")
        print(f"Total expert assignments     : {total_assignments}")
        print("-" * 96)
        usage_histograms = [
            ", ".join(
                f"{usage_count}-{expert_count}exp "
                for usage_count, expert_count in result["usage_frequency_summary"]
            )
            for result in results
        ]
        usage_column_width = max(len(histogram) for histogram in usage_histograms)
        for result, usage_summary in zip(results, usage_histograms):
            print(
                f"{result['layer_name']:<24} "
                f"usage_hist=[{usage_summary:<{usage_column_width}}] "
                f"CWS={result['cws']:.6f}"
            )
        print("=" * 96)
        print("Expert-Centric Routing Report")
        print("=" * 96)
        heatmap_path = self._save_expert_heatmap_plot(
            counts=expert_centric_results["global_expert_counts"],
            grid_side=expert_centric_results["grid_side"],
            model_id=model_id,
        )
        self._print_expert_heatmap(
            counts=expert_centric_results["global_expert_counts"],
            grid_side=expert_centric_results["grid_side"],
        )
        if heatmap_path is not None:
            print(f"Saved expert heatmap plot     : {heatmap_path}")
        print("-" * 96)
        self._print_pair_correlation_summary(
            top_pairs=expert_centric_results["top_pairs"],
            total_routing_events=expert_centric_results["total_routing_events"],
        )
        print("=" * 96)
