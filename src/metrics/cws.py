from __future__ import annotations

import re
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

                    # Flatten all token-level assignments from this layer and
                    # tally only the routed expert ids 0..59.
                    flat_indices = selected_experts.reshape(-1)
                    counts = torch.bincount(
                        flat_indices,
                        minlength=self.num_routed_experts,
                    ).cpu()
                    self.layer_counts[current_layer_name].add_(counts)

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
