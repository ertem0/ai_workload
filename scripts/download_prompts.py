# download_prompts.py
import json
import random
import numpy as np
from datasets import load_dataset

def save_prompts(prompts, filename):
    with open(filename, "w") as f:
        json.dump(prompts, f, indent=4)
    print(f"Saved {len(prompts)} prompts to {filename}")

# ── Dolly 15k ──────────────────────────────────────────────────────────────
print("Downloading Dolly 15k...")
dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
shuffled_dolly = dolly.shuffle(seed=42)

prompts_dolly_A = []
for row in shuffled_dolly.select(range(500)):
    if row["context"]:
        prompts_dolly_A.append(f"{row['instruction']}\n\nContext: {row['context']}")
    else:
        prompts_dolly_A.append(row["instruction"])

prompts_dolly_B = []
for row in shuffled_dolly.select(range(500, 1000)):
    if row["context"]:
        prompts_dolly_B.append(f"{row['instruction']}\n\nContext: {row['context']}")
    else:
        prompts_dolly_B.append(row["instruction"])

save_prompts(prompts_dolly_A, "dolly_A_prompts.json")
save_prompts(prompts_dolly_B, "dolly_B_prompts.json")

# ── GSM8K ──────────────────────────────────────────────────────────────────
print("Downloading GSM8K...")
gsm8k = load_dataset("gsm8k", "main", split="train")
shuffled_gsm8k = gsm8k.shuffle(seed=42).select(range(500))

prompts_gsm8k = [row["question"] for row in shuffled_gsm8k]

save_prompts(prompts_gsm8k, "gsm8k_prompts.json")

# ── CodeSearchNet ──────────────────────────────────────────────────────────
print("Downloading CodeSearchNet (Python)...")
code = load_dataset("code_search_net", "python", split="train")
shuffled_code = code.shuffle(seed=42)

# filter out empty docstrings first
valid_code = [
    row["func_documentation_string"]
    for row in shuffled_code
    if row["func_documentation_string"].strip()
]
prompts_code = valid_code[:500]

save_prompts(prompts_code, "codesearchnet_prompts.json")

# ── Random tokens ──────────────────────────────────────────────────────────
print("Generating random token prompts...")
VOCAB = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:"
random.seed(42)

prompts_random = [
    "".join(random.choices(VOCAB, k=random.randint(40, 80)))
    for _ in range(500)
]

save_prompts(prompts_random, "random_prompts.json")

print("\nDone! Files saved:")
print("  dolly_A_prompts.json       — 500 prompts (split A)")
print("  dolly_B_prompts.json       — 500 prompts (split B)")
print("  gsm8k_prompts.json         — 500 math reasoning prompts")
print("  codesearchnet_prompts.json — 500 code docstrings")
print("  random_prompts.json        — 500 random character sequences")