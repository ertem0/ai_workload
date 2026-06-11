from datasets import load_dataset
import random
import json
from pathlib import Path

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
DATASETS_DIR.mkdir(exist_ok=True)


def sample_prompts(prompts, n=500, seed=42):
    random.seed(seed)
    return random.sample(list(prompts), min(n, len(prompts)))


def _get_conversations(row):
    return row.get("conversations") or row.get("conversation") or []


def _is_human(turn):
    role = turn.get("from") or turn.get("user") or turn.get("role") or turn.get("speaker") or ""
    return role.lower() in ("human", "user")


def _get_value(turn):
    return turn.get("value") or turn.get("text") or turn.get("content") or ""


# ShareGPT
print("Downloading ShareGPT...")
sharegpt = load_dataset("theblackcat102/sharegpt-english", split="train")
sharegpt_raw = []
for row in sharegpt:
    conversations = _get_conversations(row)
    first_human = next(
        (_get_value(turn) for turn in conversations if _is_human(turn)),
        None,
    )
    if first_human and first_human.strip():
        sharegpt_raw.append(first_human.strip())
sharegpt_prompts = sample_prompts(sharegpt_raw, n=10000)

# GSM8K
print("Downloading GSM8K...")
gsm8k = load_dataset("gsm8k", "main", split="test")
gsm8k_prompts = sample_prompts(gsm8k["question"], n=10000)

# CodeSearchNet
print("Downloading CodeSearchNet...")
code = load_dataset("code_search_net", "python", split="train")
codesearchnet_prompts = sample_prompts(
    [s for s in code["func_documentation_string"] if s.strip()],
    n=10000,
)

# Save
for name, prompts in [
    ("sharegpt", sharegpt_prompts),
    ("gsm8k", gsm8k_prompts),
    ("codesearchnet", codesearchnet_prompts),
]:
    path = DATASETS_DIR / f"{name}_prompts.json"
    with path.open("w") as f:
        json.dump(prompts, f, indent=2)
    print(f"Saved {len(prompts)} prompts to {path}")
