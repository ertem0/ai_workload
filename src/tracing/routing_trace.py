from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class RoutingTraceRecord:
    prompt_index: int
    phase: str
    token_position: int
    token_id: int | None
    layer_id: int
    layer_name: str
    selected_experts: list[int]


def _normalize_prompt_index(prompt_index: Any) -> int:
    return int(prompt_index)


def _record_to_dict(record: RoutingTraceRecord | dict[str, Any]) -> dict[str, Any]:
    if is_dataclass(record):
        return asdict(record)
    return dict(record)


def _record_from_dict(record: dict[str, Any]) -> RoutingTraceRecord:
    return RoutingTraceRecord(
        prompt_index=int(record["prompt_index"]),
        phase=str(record["phase"]),
        token_position=int(record["token_position"]),
        token_id=(
            int(record["token_id"])
            if record.get("token_id") is not None
            else None
        ),
        layer_id=int(record["layer_id"]),
        layer_name=str(record["layer_name"]),
        selected_experts=[int(expert_id) for expert_id in record["selected_experts"]],
    )


def normalize_routing_trace(
    routing_trace: dict[Any, list[RoutingTraceRecord | dict[str, Any]]],
) -> dict[int, list[RoutingTraceRecord]]:
    return {
        _normalize_prompt_index(prompt_index): [
            record
            if isinstance(record, RoutingTraceRecord)
            else _record_from_dict(record)
            for record in records
        ]
        for prompt_index, records in routing_trace.items()
    }


def routing_trace_to_json_payload(
    routing_trace: dict[Any, list[RoutingTraceRecord | dict[str, Any]]],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "metadata": metadata or {},
        "routing_trace": {
            str(prompt_index): [_record_to_dict(record) for record in records]
            for prompt_index, records in routing_trace.items()
        },
    }


def routing_trace_from_json_payload(payload: dict[str, Any]) -> dict[str, Any]:
    routing_trace = payload.get("routing_trace", payload)
    return {
        "metadata": payload.get("metadata", {}) if isinstance(payload, dict) else {},
        "routing_trace": normalize_routing_trace(routing_trace),
    }


def export_routing_trace(
    routing_trace: dict[int, list[RoutingTraceRecord]],
    output_path: Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    payload = {
        "metadata": metadata or {},
        "routing_trace": routing_trace,
    }
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return output_path


def export_routing_trace_json(
    routing_trace: dict[int, list[RoutingTraceRecord]],
    output_path: Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    payload = routing_trace_to_json_payload(routing_trace, metadata)
    output_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return output_path


def load_routing_trace(trace_path: Path) -> dict[str, Any]:
    if trace_path.suffix.lower() == ".json":
        return load_routing_trace_json(trace_path)

    with trace_path.open("rb") as handle:
        payload = pickle.load(handle)

    if isinstance(payload, dict) and "routing_trace" in payload:
        return {
            "routing_trace": normalize_routing_trace(payload["routing_trace"]),
            "metadata": payload.get("metadata", {}),
        }

    return {
        "routing_trace": normalize_routing_trace(payload),
        "metadata": {},
    }


def load_routing_trace_json(trace_path: Path) -> dict[str, Any]:
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    return routing_trace_from_json_payload(payload)
