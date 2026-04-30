# download_prompts.py
import json
from datasets import load_dataset

print("Downloading dataset from Hugging Face...")
dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

# Shuffle and pick exactly 500
shuffled_dataset = dataset.shuffle(seed=21).select(range(500))

# Clean them up into a simple list of text strings
prompts = []
for row in shuffled_dataset:
    if row["context"]:
        full_prompt = f"{row['instruction']}\n\nContext: {row['context']}"
    else:
        full_prompt = row["instruction"]
    prompts.append(full_prompt)

# Save to a local, offline file
with open("thesis_1000_prompts.json", "w") as f:
    json.dump(prompts, f, indent=4)

print("Success! Saved 500 prompts to thesis_500_prompts.json")