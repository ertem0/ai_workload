from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """
    Load a YAML experiment configuration into a Python dictionary.
    """

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_dataset_prompts(dataset_config: Any, config_path: Path) -> list[str]:
    """
    Resolve the dataset section from either an inline prompt list or a JSON file path.
    """

    if isinstance(dataset_config, list):
        prompts = dataset_config
    elif isinstance(dataset_config, str):
        dataset_path = Path(dataset_config)
        if not dataset_path.is_absolute():
            dataset_path = (config_path.parent / dataset_path).resolve()

        with dataset_path.open("r", encoding="utf-8") as handle:
            prompts = json.load(handle)
    else:
        raise TypeError(
            "Config field 'dataset' must be either a list of prompt strings or a JSON file path."
        )

    if not isinstance(prompts, list) or not all(isinstance(prompt, str) for prompt in prompts):
        raise ValueError(
            "Resolved dataset must be a JSON/YAML list containing only prompt strings."
        )

    return prompts
