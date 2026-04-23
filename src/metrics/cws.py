from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass
class RoutingTraceRecord:
    prompt_index: int
    phase: str
    token_position: int
    token_id: int | None
    layer_id: int
    layer_name: str
    selected_experts: list[int]


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
        self.layer_order: list[str] = []
        self.layer_name_to_id: dict[str, int] = {}
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
        self.routing_trace: dict[int, list[RoutingTraceRecord]] = {}
        self.current_prompt_index: int | None = None
        self.current_prompt_token_ids: list[int] = []
        self.current_phase = "idle"
        self.generate_prompt_length = 0
        self.current_decode_step = -1

    @staticmethod
    def _normalize_token_ids(token_ids: torch.Tensor | list[int]) -> list[int]:
        if isinstance(token_ids, torch.Tensor):
            return [int(token_id) for token_id in token_ids.reshape(-1).tolist()]
        return [int(token_id) for token_id in token_ids]

    @staticmethod
    def _extract_layer_id(layer_name: str) -> int:
        if layer_name.startswith("layer "):
            return int(layer_name.split()[-1])
        return -1

    def start_prompt_trace(
        self,
        *,
        prompt_index: int,
        prompt_token_ids: torch.Tensor | list[int],
    ) -> None:
        self.current_prompt_index = prompt_index
        self.current_prompt_token_ids = self._normalize_token_ids(prompt_token_ids)
        self.current_phase = "prefill"
        self.generate_prompt_length = len(self.current_prompt_token_ids)
        self.current_decode_step = -1
        self.routing_trace.setdefault(prompt_index, [])

    def start_generation_trace(
        self,
        *,
        prompt_index: int,
        prompt_token_ids: torch.Tensor | list[int],
    ) -> None:
        self.current_prompt_index = prompt_index
        self.current_prompt_token_ids = self._normalize_token_ids(prompt_token_ids)
        self.current_phase = "generate"
        self.generate_prompt_length = len(self.current_prompt_token_ids)
        self.current_decode_step = -1
        self.routing_trace.setdefault(prompt_index, [])

    def finalize_generation_trace(
        self,
        *,
        prompt_index: int,
        generated_token_ids: torch.Tensor | list[int],
    ) -> None:
        full_sequence = self._normalize_token_ids(generated_token_ids)
        generated_suffix = full_sequence[self.generate_prompt_length :]
        for record in self.routing_trace.get(prompt_index, []):
            if record.phase != "decode" or record.token_id is not None:
                continue
            decode_index = record.token_position - self.generate_prompt_length
            if 0 <= decode_index < len(generated_suffix):
                record.token_id = generated_suffix[decode_index]
        self.current_phase = "idle"

    def _record_trace_event(
        self,
        *,
        prompt_index: int,
        phase: str,
        token_position: int,
        token_id: int | None,
        layer_name: str,
        selected_experts: list[int],
    ) -> None:
        self.routing_trace.setdefault(prompt_index, []).append(
            RoutingTraceRecord(
                prompt_index=prompt_index,
                phase=phase,
                token_position=token_position,
                token_id=token_id,
                layer_id=self.layer_name_to_id[layer_name],
                layer_name=layer_name,
                selected_experts=selected_experts,
            )
        )

    def _accumulate_layer_event(
        self,
        *,
        current_layer_name: str,
        event_indices: torch.Tensor,
    ) -> None:
        self.total_routing_events += int(event_indices.shape[0])
        flat_indices = event_indices.reshape(-1)
        counts = torch.bincount(
            flat_indices,
            minlength=self.num_routed_experts,
        ).cpu()
        self.layer_counts[current_layer_name].add_(counts)
        self.global_expert_counts.add_(counts)

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

    def _capture_prefill_trace(
        self,
        *,
        current_layer_name: str,
        event_indices: torch.Tensor,
    ) -> None:
        prompt_index = self.current_prompt_index
        if prompt_index is None:
            return

        self._accumulate_layer_event(
            current_layer_name=current_layer_name,
            event_indices=event_indices,
        )
        for token_offset, event in enumerate(event_indices.tolist()):
            token_id = None
            if token_offset < len(self.current_prompt_token_ids):
                token_id = self.current_prompt_token_ids[token_offset]
            self._record_trace_event(
                prompt_index=prompt_index,
                phase="prefill",
                token_position=token_offset,
                token_id=token_id,
                layer_name=current_layer_name,
                selected_experts=[int(expert_id) for expert_id in event],
            )

    def _capture_decode_trace(
        self,
        *,
        current_layer_name: str,
        event_indices: torch.Tensor,
    ) -> None:
        prompt_index = self.current_prompt_index
        if prompt_index is None:
            return

        if event_indices.shape[0] != 1:
            return

        if self.layer_order and current_layer_name == self.layer_order[0]:
            self.current_decode_step += 1

        token_position = self.generate_prompt_length + max(self.current_decode_step, 0)
        self._accumulate_layer_event(
            current_layer_name=current_layer_name,
            event_indices=event_indices,
        )
        self._record_trace_event(
            prompt_index=prompt_index,
            phase="decode",
            token_position=token_position,
            token_id=None,
            layer_name=current_layer_name,
            selected_experts=[int(expert_id) for expert_id in event_indices[0].tolist()],
        )

    def export_routing_trace(self, output_path: Path) -> Path:
        with output_path.open("wb") as handle:
            pickle.dump(self.routing_trace, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return output_path

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
                self.layer_order.append(layer_name)
                self.layer_name_to_id[layer_name] = self._extract_layer_id(layer_name)
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
                    if self.current_phase == "generate":
                        self._capture_decode_trace(
                            current_layer_name=current_layer_name,
                            event_indices=event_indices,
                        )
                        return

                    self._capture_prefill_trace(
                        current_layer_name=current_layer_name,
                        event_indices=event_indices,
                    )

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
    def _layer_sort_key(name: str) -> tuple[int, str]:
        if name.startswith("layer "):
            return int(name.split()[-1]), name
        return 10**9, name

    def compute_layer_results(self) -> list[dict[str, Any]]:
        """
        Compute per-layer CWS values from the accumulated expert histograms.
        """

        results: list[dict[str, Any]] = []
        for layer_name in sorted(self.layer_counts.keys(), key=self._layer_sort_key):
            counts = self.layer_counts[layer_name].to(torch.float64)
            mean_count = float(counts.mean().item())
            std_count = float(counts.std(unbiased=False).item())
            cws = float(std_count / mean_count) if mean_count > 0 else float("inf")

            results.append(
                {
                    "layer_name": layer_name,
                    "counts": self.layer_counts[layer_name],
                    "cws": cws,
                }
            )
        return results

    def compute_expert_centric_results(self) -> dict[str, Any]:
        """
        Compute expert-centric statistics over the flattened token-layer trace.

        The trace order is conceptually:
        token0-layer0, token0-layer1, ..., token0-layerN,
        token1-layer0, ..., last-token-last-layer

        Each trace element is one routing event containing the top-k selected experts.
        """

        ordered_layer_names = sorted(self.layer_counts.keys(), key=self._layer_sort_key)
        if ordered_layer_names:
            layer_expert_matrix = torch.stack(
                [self.layer_counts[layer_name] for layer_name in ordered_layer_names],
                dim=0,
            )
        else:
            layer_expert_matrix = torch.zeros(
                (0, self.num_routed_experts),
                dtype=torch.long,
            )

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
            "layer_names": ordered_layer_names,
            "layer_expert_matrix": layer_expert_matrix.clone(),
            "pair_probability_matrix": pair_probabilities,
            "top_pairs": top_pairs,
            "total_routing_events": self.total_routing_events,
            "routing_trace": self.routing_trace,
        }

    def _save_expert_heatmap_plot(
        self,
        layer_expert_matrix: torch.Tensor,
        layer_names: list[str],
        model_id: str,
        configured_top_k: int,
        output_dir: Path,
    ) -> Path | None:
        """
        Save a matplotlib heatmap of absolute expert usage counts per layer.

        The heatmap uses:
        - x-axis: routed expert id
        - y-axis: transformer layer
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

        if layer_expert_matrix.numel() == 0:
            return None

        safe_model_name = re.sub(r"[^a-zA-Z0-9]+", "_", model_id).strip("_").lower()
        output_path = output_dir / (
            f"{safe_model_name}_topk_{configured_top_k}_layer_expert_usage_heatmap.png"
        )

        heatmap = layer_expert_matrix.to(torch.float64).numpy()
        figure_width = max(14.0, 0.26 * self.num_routed_experts)
        figure_height = max(7.0, 0.42 * len(layer_names))
        figure, axis = plt.subplots(figsize=(figure_width, figure_height))
        image = axis.imshow(heatmap, cmap="inferno", interpolation="nearest", aspect="auto")
        colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label("Expert usage count", rotation=90)

        axis.set_title("Layer-By-Expert Routing Heatmap", pad=14)
        axis.set_xlabel("Expert id")
        axis.set_ylabel("Layer")
        axis.set_xticks(range(self.num_routed_experts))
        axis.set_xticklabels([str(expert_id) for expert_id in range(self.num_routed_experts)])
        axis.set_yticks(range(len(layer_names)))
        axis.set_yticklabels(layer_names)
        axis.tick_params(axis="x", labelrotation=90, labelsize=7)
        axis.tick_params(axis="y", labelsize=8)

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
        output_token_count: int,
        output_dir: Path,
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
        print(f"Output token count           : {output_token_count}")
        print(f"Total expert assignments     : {total_assignments}")
        print("-" * 96)
        for result in results:
            print(
                f"{result['layer_name']:<24} "
                f"CWS={result['cws']:.6f}"
            )
        print("=" * 96)
        print("Expert-Centric Routing Report")
        print("=" * 96)
        trace_path = self.export_routing_trace(output_dir / "routing_trace.pkl")
        print(f"Saved routing trace           : {trace_path}")
        heatmap_path = self._save_expert_heatmap_plot(
            layer_expert_matrix=expert_centric_results["layer_expert_matrix"],
            layer_names=expert_centric_results["layer_names"],
            model_id=model_id,
            configured_top_k=configured_top_k,
            output_dir=output_dir,
        )
        if heatmap_path is not None:
            print(f"Saved expert heatmap plot     : {heatmap_path}")
            print("Heatmap axes                  : y=layer, x=expert id")
        print("-" * 96)
        self._print_pair_correlation_summary(
            top_pairs=expert_centric_results["top_pairs"],
            total_routing_events=expert_centric_results["total_routing_events"],
        )
        print("=" * 96)
