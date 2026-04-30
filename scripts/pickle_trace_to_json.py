from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tracing.routing_trace import export_routing_trace_json, load_routing_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a pickled routing trace into the previous JSON format."
    )
    parser.add_argument(
        "input_pickle",
        help="Path to the routing_trace.pkl file.",
    )
    parser.add_argument(
        "--output",
        help="Optional output JSON path. Defaults to the input path with a .json suffix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_pickle).resolve()
    output_path = (
        Path(args.output).resolve()
        if args.output
        else input_path.with_suffix(".json")
    )

    payload = load_routing_trace(input_path)
    export_routing_trace_json(
        payload["routing_trace"],
        output_path,
        payload["metadata"],
    )
    print(f"Wrote JSON trace to {output_path}")


if __name__ == "__main__":
    main()
