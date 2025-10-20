import json
import re
import matplotlib.pyplot as plt

with open("predictions.json") as f:
    predictions = json.load(f)
with open("references.json") as f:
    references = json.load(f)

print(f"Loaded {len(predictions)} predictions and {len(references)} references")

if len(predictions) != len(references):
    print("Mismatch in lengths, truncating")
    n = min(len(predictions), len(references))
    predictions = predictions[:n]
    references = references[:n]


def extract_line_numbers(text):
    """
    Extracts integers following 'Line:' or 'Line(s):' patterns.
    Returns a list of ints (could be empty if no number found).
    """
    if not isinstance(text, str):
        return []
    matches = re.findall(r"\d+", text)
    return [int(m) for m in matches]


parsed_preds = [extract_line_numbers(p) for p in predictions]
parsed_refs = [extract_line_numbers(r) for r in references]

exact_matches = 0
for pred, ref in zip(parsed_preds, parsed_refs):
    if set(pred) == set(ref):  # same buggy lines
        exact_matches += 1

accuracy = exact_matches / len(parsed_preds)
print(f"Exact line-number match accuracy: {accuracy:.3f}")

results = {
    "total": len(parsed_preds),
    "correct": exact_matches,
    "accuracy": accuracy,
}
with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("💾 Saved evaluation_results.json")

plt.figure(figsize=(6,4))
plt.bar(["Accuracy"], [accuracy], color="teal")
plt.ylim(0, 1)
plt.ylabel("Exact Match Accuracy")
plt.title("Evaluation on Buggy Line Prediction")
plt.tight_layout()
plt.savefig("eval_accuracy.png")
print("Saved plot to eval_accuracy.png")
plt.show()