from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class SmallExpertRoutingTracker:
    """
    Memory-efficient expert routing tracker.

    Instead of storing one record per token per layer (which grows unboundedly),
    accumulates fixed-size per-layer histograms for the current prompt only.
    After each prompt, the compact object is flushed to a JSONL file and the
    accumulators are zeroed — memory stays constant regardless of prompt count.

    Output format per line of the JSONL file:
    {
      "prompt": <int>,
      "total_tokens": <int>,
      "prefill": {
        "tokens": <int>,
        "layer 0": {
          "experts":      {"<id>": <count>, ...},   # non-zero only
          "expert_zeros": <int>,                    # experts with 0 activations
          "expert_pairs": {"<i>,<j>": <count>, ...}, # non-zero upper-triangle only
          "pair_zeros":   <int>                     # pairs with 0 co-activations
        },
        ...
      },
      "decode": { ... same structure ... }
    }
    """

    def __init__(self, model: nn.Module, top_k: int) -> None:
        self.model = model
        self.top_k = top_k
        self.handles: list[Any] = []
        self.enabled = True
        self.router_hook_count = 0
        self.num_routed_experts = self._infer_num_routed_experts()
        self.layer_order: list[str] = []
        self.layer_name_to_id: dict[str, int] = {}

        # per-prompt accumulators — shape (num_routed_experts,) and (n, n)
        self._prefill_expert: dict[str, torch.Tensor] = {}
        self._prefill_pairs: dict[str, torch.Tensor] = {}
        self._decode_expert: dict[str, torch.Tensor] = {}
        self._decode_pairs: dict[str, torch.Tensor] = {}
        self._prefill_tokens = 0
        self._decode_tokens = 0

        # current prompt state
        self._current_prompt_index: int | None = None
        self._phase = "idle"
        self._generation_forward_index = -1

        # lazily-opened JSONL output file
        self._output_handle: Any = None
        self.prompts_flushed = 0

    # ── Expert / layer discovery ──────────────────────────────────────────────

    def _infer_num_routed_experts(self) -> int:
        config = getattr(self.model, "config", None)
        for attr in (
            "num_experts",
            "num_local_experts",
            "n_routed_experts",
            "num_routed_experts",
            "moe_num_experts",
        ):
            val = int(getattr(config, attr, 0))
            if val > 0:
                return val
        raise ValueError(
            "Unable to determine the number of routed experts from model.config. "
            "Tried: num_experts, num_local_experts, n_routed_experts, "
            "num_routed_experts, moe_num_experts."
        )

    def _is_router_gate(self, module_name: str, module: nn.Module) -> bool:
        leaf_name = module_name.rsplit(".", maxsplit=1)[-1]
        if leaf_name == "router":
            router_layer = getattr(module, "layer", None)
            if (
                hasattr(module, "top_k")
                and int(getattr(module, "num_experts", 0)) == self.num_routed_experts
                and isinstance(router_layer, nn.Linear)
                and router_layer.out_features == self.num_routed_experts
            ):
                return True

        if leaf_name not in ("gate", "router", "gate_proj") or "experts" in module_name:
            return False

        if (
            isinstance(module, nn.Linear)
            and module.out_features == self.num_routed_experts
        ):
            return True

        if not (hasattr(module, "top_k") and hasattr(module, "weight")):
            return False

        for expert_count_attr in (
            "num_experts",
            "num_routed_experts",
            "n_routed_experts",
            "num_local_experts",
        ):
            if int(getattr(module, expert_count_attr, 0)) == self.num_routed_experts:
                return True

        return False

    def _extract_layer_name(self, module_name: str) -> str:
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", module_name)
        if match is not None:
            return f"layer {int(match.group(1))}"
        return module_name.rsplit(".gate", maxsplit=1)[0]

    @staticmethod
    def _extract_layer_id(layer_name: str) -> int:
        if layer_name.startswith("layer "):
            return int(layer_name.split()[-1])
        return -1

    def _extract_selected_experts(
        self, output: Any, router_top_k: int
    ) -> torch.Tensor | None:
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

    # ── Accumulation ─────────────────────────────────────────────────────────

    def _accumulate(
        self,
        expert_counts: torch.Tensor,
        pair_counts: torch.Tensor,
        event_indices: torch.Tensor,
    ) -> None:
        """
        event_indices: (T, top_k) CPU long tensor.
        Updates expert_counts (n,) and pair_counts (n, n) in-place.
        pair_counts[i,j] counts tokens where both expert i and j were selected.
        """
        n = self.num_routed_experts
        T = event_indices.shape[0]

        flat = event_indices.reshape(-1).long()
        expert_counts.add_(torch.bincount(flat, minlength=n))

        # Build (T, n) binary indicator and compute co-occurrence matrix
        indicator = torch.zeros(T, n, dtype=torch.long)
        indicator.scatter_(1, event_indices.long().clamp(0, n - 1), 1)
        pair_matrix = indicator.T.mm(indicator)
        pair_matrix.fill_diagonal_(0)
        pair_counts.add_(pair_matrix)

    # ── Prompt lifecycle ──────────────────────────────────────────────────────

    def start_generation_trace(
        self,
        *,
        prompt_index: int,
        prompt_token_ids: Any = None,
    ) -> None:
        self._current_prompt_index = prompt_index
        self._phase = "generate"
        self._generation_forward_index = -1
        self._prefill_tokens = 0
        self._decode_tokens = 0
        for layer_name in self.layer_order:
            self._prefill_expert[layer_name].zero_()
            self._prefill_pairs[layer_name].zero_()
            self._decode_expert[layer_name].zero_()
            self._decode_pairs[layer_name].zero_()

    def finalize_generation_trace(self, *, prompt_index: int) -> None:
        self._phase = "idle"

    def open_output(self, output_path: Path, metadata: dict[str, Any]) -> None:
        """
        Open the output JSONL file and write a metadata header as the first line.
        Must be called before the first flush_prompt.
        """
        self._output_handle = output_path.open("w", encoding="utf-8")
        header = {"_type": "metadata", **metadata}
        self._output_handle.write(json.dumps(header) + "\n")
        self._output_handle.flush()

    def flush_prompt(self, output_path: Path) -> None:
        """Serialize the current prompt's compact object to JSONL, then reset."""
        if self._current_prompt_index is None:
            return

        n = self.num_routed_experts
        total_pairs = n * (n - 1) // 2

        def build_phase(
            expert_counts: dict[str, torch.Tensor],
            pair_counts: dict[str, torch.Tensor],
            token_count: int,
        ) -> dict[str, Any]:
            phase: dict[str, Any] = {"tokens": token_count}
            for layer_name in self.layer_order:
                ec = expert_counts[layer_name]
                pc = pair_counts[layer_name]

                experts = {
                    str(i): int(ec[i].item())
                    for i in range(n)
                    if ec[i].item() > 0
                }
                pairs = {
                    f"{i},{j}": int(pc[i, j].item())
                    for i in range(n)
                    for j in range(i + 1, n)
                    if pc[i, j].item() > 0
                }
                phase[layer_name] = {
                    "experts": experts,
                    "expert_zeros": n - len(experts),
                    "expert_pairs": pairs,
                    "pair_zeros": total_pairs - len(pairs),
                }
            return phase

        record = {
            "prompt": self._current_prompt_index,
            "total_tokens": self._prefill_tokens + self._decode_tokens,
            "prefill": build_phase(
                self._prefill_expert, self._prefill_pairs, self._prefill_tokens
            ),
            "decode": build_phase(
                self._decode_expert, self._decode_pairs, self._decode_tokens
            ),
        }

        if self._output_handle is None:
            self._output_handle = output_path.open("a", encoding="utf-8")
        self._output_handle.write(json.dumps(record) + "\n")
        self._output_handle.flush()
        self.prompts_flushed += 1
        self._current_prompt_index = None

    # ── Hook registration ─────────────────────────────────────────────────────

    def register_hooks(self) -> None:
        def generation_forward_pre_hook(
            _module: nn.Module, _args: tuple[Any, ...]
        ) -> None:
            if not self.enabled or self._phase != "generate":
                return
            self._generation_forward_index += 1

        self.handles.append(
            self.model.register_forward_pre_hook(generation_forward_pre_hook)
        )
        self.router_hook_count = 0

        for module_name, module in self.model.named_modules():
            if not self._is_router_gate(module_name, module):
                continue

            layer_name = self._extract_layer_name(module_name)
            if layer_name not in self._prefill_expert:
                n = self.num_routed_experts
                self.layer_order.append(layer_name)
                self.layer_name_to_id[layer_name] = self._extract_layer_id(layer_name)
                self._prefill_expert[layer_name] = torch.zeros(n, dtype=torch.long)
                self._prefill_pairs[layer_name] = torch.zeros((n, n), dtype=torch.long)
                self._decode_expert[layer_name] = torch.zeros(n, dtype=torch.long)
                self._decode_pairs[layer_name] = torch.zeros((n, n), dtype=torch.long)

            def make_hook(
                current_layer_name: str, current_module: nn.Module
            ) -> Any:
                def hook(
                    _: nn.Module, __: tuple[Any, ...], output: Any
                ) -> None:
                    if not self.enabled or self._phase != "generate":
                        return
                    router_top_k = int(getattr(current_module, "top_k", self.top_k))
                    selected = self._extract_selected_experts(output, router_top_k)
                    if selected is None:
                        return
                    event_indices = selected.reshape(-1, router_top_k).cpu()
                    is_prefill = self._generation_forward_index <= 0
                    if is_prefill:
                        if current_layer_name == self.layer_order[0]:
                            self._prefill_tokens += event_indices.shape[0]
                        self._accumulate(
                            self._prefill_expert[current_layer_name],
                            self._prefill_pairs[current_layer_name],
                            event_indices,
                        )
                    else:
                        if current_layer_name == self.layer_order[0]:
                            self._decode_tokens += 1
                        self._accumulate(
                            self._decode_expert[current_layer_name],
                            self._decode_pairs[current_layer_name],
                            event_indices,
                        )

                return hook

            self.handles.append(
                module.register_forward_hook(make_hook(layer_name, module))
            )
            self.router_hook_count += 1

        if self.router_hook_count == 0:
            self.remove_hooks()
            raise RuntimeError(
                "No MoE router gate layers were found. "
                "Expected Linear modules named 'gate' or 'router', or "
                "JetMoE-style 'router.layer' modules, with "
                f"out_features={self.num_routed_experts}."
            )

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def remove_hooks(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def close(self) -> None:
        if self._output_handle is not None:
            self._output_handle.close()
            self._output_handle = None
