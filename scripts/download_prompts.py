# download_prompts.py
import json
import random
import numpy as np
from datasets import load_dataset

def save_prompts(prompts, filename):
    with open(filename, "w") as f:
        json.dump(prompts, f, indent=4)
    print(f"Saved {len(prompts)} prompts to {filename}")



# ── ShareGPT English ───────────────────────────────────────────────────────
print("Downloading ShareGPT English...")
sharegpt = load_dataset("theblackcat102/sharegpt-english", split="train")
shuffled_sharegpt = sharegpt.shuffle(seed=42)

# Inspect first row to detect structure
_first = shuffled_sharegpt[0]
print(f"  columns: {list(_first.keys())}")
_convs = _first.get("conversations") or _first.get("conversation") or []
if _convs:
    print(f"  first turn keys: {list(_convs[0].keys())}")
    print(f"  first turn sample: {_convs[0]}")

def _get_conversations(row):
    return row.get("conversations") or row.get("conversation") or []

def _is_human(turn):
    role = turn.get("from") or turn.get("user") or turn.get("role") or turn.get("speaker") or ""
    return role.lower() in ("human", "user")

def _get_value(turn):
    return turn.get("value") or turn.get("text") or turn.get("content") or ""

prompts_sharegpt = []
for row in shuffled_sharegpt:
    conversations = _get_conversations(row)
    first_human = next(
        (_get_value(turn) for turn in conversations if _is_human(turn)),
        None,
    )
    if first_human and first_human.strip():
        prompts_sharegpt.append(first_human.strip())
    if len(prompts_sharegpt) >= 500:
        break

save_prompts(prompts_sharegpt, "sharegpt_prompts.json")

print("\nDone! Files saved:")
print("  dolly_A_prompts.json       — 500 prompts (split A)")
print("  dolly_B_prompts.json       — 500 prompts (split B)")
print("  gsm8k_prompts.json         — 500 math reasoning prompts")
print("  codesearchnet_prompts.json — 500 code docstrings")
print("  random_prompts.json        — 500 random character sequences")
print("  sharegpt_prompts.json      — 500 first human turns from ShareGPT")