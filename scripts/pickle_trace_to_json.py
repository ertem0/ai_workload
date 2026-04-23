from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path

from src.metrics.cws import RoutingTraceRecord


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

    with input_path.open("rb") as handle:
        routing_trace: dict[int, list[RoutingTraceRecord]] = pickle.load(handle)

    serializable_trace = {
        str(prompt_index): [asdict(record) for record in records]
        for prompt_index, records in routing_trace.items()
    }

    output_path.write_text(
        json.dumps(serializable_trace, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote JSON trace to {output_path}")


if __name__ == "__main__":
    main()
