from __future__ import annotations

import argparse
import collections
import itertools
import os
import pickle
from pathlib import Path
from typing import Any

from tqdm import tqdm


def load_trace_payload(trace_file: Path) -> dict[str, Any]:
    with trace_file.open("rb") as handle:
        payload = pickle.load(handle)

    if isinstance(payload, dict) and "routing_trace" in payload:
        return payload

    return {
        "metadata": {},
        "routing_trace": payload,
    }


def build_token_journeys(
    routing_trace: dict[int, list[Any]],
) -> list[list[list[int]]]:
    grouped_records: dict[tuple[int, int], list[Any]] = collections.defaultdict(list)

    for prompt_records in routing_trace.values():
        for record in prompt_records:
            grouped_records[(int(record.prompt_index), int(record.token_position))].append(record)

    token_journeys: list[list[list[int]]] = []
    for records in grouped_records.values():
        ordered_records = sorted(records, key=lambda record: int(record.layer_id))
        journey = [list(record.selected_experts) for record in ordered_records if record.selected_experts]
        if journey:
            token_journeys.append(journey)

    return token_journeys


def extract_branching_trajectories(
    token_journeys: list[list[list[int]]],
    max_n: int = 10,
) -> dict[tuple[int, ...], int]:
    ngram_counts: dict[tuple[int, ...], int] = collections.defaultdict(int)

    for journey in tqdm(token_journeys, desc="Mining Trajectories"):
        if len(journey) < 2:
            continue

        upper_n = min(max_n, len(journey))
        for n in range(2, upper_n + 1):
            for start in range(0, len(journey) - n + 1):
                window = journey[start : start + n]
                for path in itertools.product(*window):
                    ngram_counts[tuple(int(expert_id) for expert_id in path)] += 1

    return ngram_counts


def summarize_max_frequencies(
    ngram_counts: dict[tuple[int, ...], int],
) -> dict[int, int]:
    max_frequency_by_length: dict[int, int] = {}

    for sequence, count in ngram_counts.items():
        if count < 2:
            continue
        length = len(sequence)
        current_max = max_frequency_by_length.get(length, 0)
        if count > current_max:
            max_frequency_by_length[length] = count

    return max_frequency_by_length


def plot_max_frequencies(
    max_frequency_by_length: dict[int, int],
    output_dir: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    output_path = output_dir / "branching_trajectory_frequencies.png"
    figure, axis = plt.subplots(figsize=(10, 6))

    if max_frequency_by_length:
        lengths = sorted(max_frequency_by_length.keys())
        frequencies = [max_frequency_by_length[length] for length in lengths]
        axis.plot(lengths, frequencies, marker="o")
    else:
        axis.plot([], [])

    axis.set_yscale("log")
    axis.grid(True, which="both", linestyle=":", alpha=0.4)
    axis.set_xlabel("Sequence Length (Consecutive Layers)")
    axis.set_ylabel("Maximum Frequency (Log Scale)")
    axis.set_title("Maximum Frequency of Branching Token Trajectories")

    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze frequent branching token trajectories from a saved routing trace."
    )
    parser.add_argument(
        "--trace_file",
        required=True,
        help="Path to the pickled routing trace file.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where branching_trajectory_frequencies.png will be written.",
    )
    parser.add_argument(
        "--max_n",
        type=int,
        default=10,
        help="Maximum sequence length to analyze.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_file = Path(args.trace_file).resolve()
    output_dir = Path(args.output_dir).resolve()

    payload = load_trace_payload(trace_file)
    routing_trace = payload["routing_trace"]
    token_journeys = build_token_journeys(routing_trace)
    ngram_counts = extract_branching_trajectories(token_journeys, max_n=args.max_n)
    max_frequency_by_length = summarize_max_frequencies(ngram_counts)
    output_path = plot_max_frequencies(max_frequency_by_length, output_dir)

    print(f"Loaded token journeys          : {len(token_journeys)}")
    print(f"Unique branching sequences     : {len(ngram_counts)}")
    print(f"Saved trajectory frequency plot: {output_path}")


if __name__ == "__main__":
    main()
