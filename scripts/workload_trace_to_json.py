from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert workload_trace.pkl into a readable JSON file."
    )
    parser.add_argument(
        "input_pickle",
        help="Path to workload_trace.pkl.",
    )
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


def normalize_for_json(value: Any) -> Any:
    if is_dataclass(value):
        return normalize_for_json(asdict(value))

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(normalize_for_json(key)): normalize_for_json(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [normalize_for_json(item) for item in value]

    if hasattr(value, "detach") and hasattr(value, "cpu"):
        tensor = value.detach().cpu()
        if tensor.ndim == 0:
            return normalize_for_json(tensor.item())
        return normalize_for_json(tensor.tolist())

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
        Path(args.output).resolve()
        if args.output
        else input_path.with_suffix(".json")
    )

    with input_path.open("rb") as handle:
        payload = pickle.load(handle)

    serializable_payload = normalize_for_json(payload)
    json_kwargs: dict[str, Any] = {"ensure_ascii": False}
    if not args.compact:
        json_kwargs["indent"] = 2

    output_path.write_text(
        json.dumps(serializable_payload, **json_kwargs),
        encoding="utf-8",
    )
    print(f"Wrote JSON workload trace to {output_path}")


if __name__ == "__main__":
    main()
