from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tracing.routing_trace import export_routing_trace, load_routing_trace_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert expert_traces_raw.json into expert_traces_raw.pkl."
    )
    parser.add_argument(
        "input_json",
        help="Path to expert_traces_raw.json.",
    )
    parser.add_argument(
        "--output",
        help="Optional output pickle path. Defaults to the input path with a .pkl suffix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_json).resolve()
    output_path = (
        Path(args.output).resolve()
        if args.output
        else input_path.with_suffix(".pkl")
    )

    payload = load_routing_trace_json(input_path)
    export_routing_trace(
        payload["routing_trace"],
        output_path,
        payload["metadata"],
    )
    print(f"Wrote pickle trace to {output_path}")


if __name__ == "__main__":
    main()
