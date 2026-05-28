from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert workload_trace.pkl into a readable JSON file."
    )
    parser.add_argument("input_pickle", help="Path to workload_trace.pkl.")
    parser.add_argument(
        "--output",
        help="Optional output JSON path. Defaults to the input path with a .json suffix.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Write compact JSON instead of pretty-printed JSON.",
    )
    return parser.parse_args()


def _product(shape: list[int]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def transform_trace(trace: Any) -> Any:
    """
    Normalize a workload trace so weights use numels (not bytes) and ops
    carry weight_numels.

    - weights[name]["bytes"] → weights[name]["numels"] = product(shape)
    - ops[i]["weight_numels"] filled from inventory when missing
    """
    if not isinstance(trace, dict):
        return trace

    weights: dict[str, Any] = trace.get("weights", {})

    for w in weights.values():
        if not isinstance(w, dict):
            continue
        if "numels" not in w and "shape" in w:
            w["numels"] = _product(w["shape"])
        w.pop("bytes", None)

    for op in trace.get("ops", []):
        if not isinstance(op, dict):
            continue
        if not op.get("weight_numels") and op.get("op_type") == "linear":
            numel = weights.get(op.get("module", ""), {}).get("numels", 0)
            if numel:
                op["weight_numels"] = numel

    return trace


def normalize_for_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(normalize_for_json(k)): normalize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [normalize_for_json(v) for v in value]
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        tensor = value.detach().cpu()
        return normalize_for_json(tensor.item() if tensor.ndim == 0 else tensor.tolist())
    if hasattr(value, "tolist"):
        return normalize_for_json(value.tolist())
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_pickle).resolve()
    output_path = (
        Path(args.output).resolve() if args.output else input_path.with_suffix(".json")
    )

    with input_path.open("rb") as handle:
        payload = pickle.load(handle)

    payload = transform_trace(payload)
    serializable = normalize_for_json(payload)
    json_kwargs: dict[str, Any] = {"ensure_ascii": False}
    if not args.compact:
        json_kwargs["indent"] = 2

    output_path.write_text(json.dumps(serializable, **json_kwargs), encoding="utf-8")
    print(f"Wrote JSON workload trace to {output_path}")


if __name__ == "__main__":
    main()
