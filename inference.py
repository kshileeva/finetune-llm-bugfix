from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
import torch
import matplotlib.pyplot as plt
import evaluate
from tqdm import tqdm
import json
import os

base_model = "WizardLMTeam/WizardMath-7B-V1.1"
adapter_model = "shhhks/bugfix-wizardmath"
test_file = "test.json"
batch_size = 1
max_new_tokens = 128

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading base + adapter model...")
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(model, adapter_model)
model.eval()

print("Loading test set...")
dataset = load_dataset(
    "json",
    data_files={"test": f"hf://datasets/shhhks/bugfix/{test_file}"},
    split="test"
)
print(f"Loaded {len(dataset)} test examples.")

def format_prompt(example):
    return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
"""

def generate(example):
    prompt = format_prompt(example)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return decoded.split("### Response:")[-1].strip()

if os.path.exists("predictions.json") and os.path.exists("references.json"):
    print("Found existing prediction files. Loading...")
    with open("predictions.json") as f:
        predictions = json.load(f)
    with open("references.json") as f:
        references = json.load(f)
else:
    print("Running inference...")
    predictions = []
    references = []
    for example in tqdm(dataset):
        try:
            pred = generate(example)
        except Exception as e:
            print(f"Error generating prediction: {e}")
            pred = ""
        predictions.append(pred)
        references.append(example["output"])

    with open("predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)
    with open("references.json", "w") as f:
        json.dump(references, f, indent=2)
    print("Saved predictions and references.")

print("Evaluating...")
accuracy = evaluate.load("accuracy")
results = accuracy.compute(predictions=predictions, references=references)
print("Exact match accuracy:", results["accuracy"])

plt.figure(figsize=(6, 4))
plt.bar(["Accuracy"], [results["accuracy"]], color="teal")
plt.ylim(0, 1)
plt.ylabel("Exact Match Accuracy")
plt.title("Evaluation on Buggy Line Prediction")
plt.tight_layout()
plt.savefig("eval_accuracy.png")
print("Saved plot to eval_accuracy.png")
plt.show()