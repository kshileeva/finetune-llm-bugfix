import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

INPUT_PATH = "bugfix_prompts.jsonl"

tokenizer = AutoTokenizer.from_pretrained("WizardLMTeam/WizardMath-7B-V1.1")
tokenizer.model_max_length = 4096
MAX_TOKENS = 4096

raw = []
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        prompt = f"{ex['instruction']}\n{ex['input']}"
        tokens = tokenizer(prompt, truncation=True, max_length=MAX_TOKENS)
        if len(tokens.input_ids) <= MAX_TOKENS:
            raw.append({
                "instruction": ex["instruction"],
                "input": ex["input"],
                "output": ex["output"]
            })

print(f"Filtered down to {len(raw)} examples.")

ds = Dataset.from_list(raw)
splits = ds.train_test_split(test_size=0.2, seed=42)
val_test = splits["test"].train_test_split(test_size=0.5, seed=42)

final = DatasetDict({
    "train": splits["train"],
    "validation": val_test["train"],
    "test": val_test["test"]
})
OUTPUT_DIR = 'splits'
final["train"].to_json(f"{OUTPUT_DIR}/train.json")
final["validation"].to_json(f"{OUTPUT_DIR}/validation.json")
final["test"].to_json(f"{OUTPUT_DIR}/test.json")
