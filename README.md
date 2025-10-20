# Fine-tuning an LLM for Bug Location Prediction

This project fine-tunes a small language model to predict the location of bugs in Python code.

---

## Task definition
**Given**: A Python file containing a bug (pre-fix version)

**Instruction**: Find the buggy line(s) in this code.

**Model Output**: One or more line numbers where the bug occurs.

This task is framed as instruction-following to allow use of instruction-tuned models (e.g., CodeGen, WizardMath).

---

## Dataset construction
- **Source**: popular python frameworks ([jinja](https://github.com/pallets/jinja.git),
  [httpx](https://github.com/encode/httpx.git),
    [mypy](https://github.com/python/mypy.git),
    [black](https://github.com/psf/black.git),
    [django](https://github.com/django/django.git),
    [fastapi](https://github.com/tiangolo/fastapi.git),
    [pytest](https://github.com/pytest-dev/pytest.git),
    [pip](https://github.com/pypa/pip.git)) Git histories
- **Process**:
  1. Extract `.py` files before/after each fix commit
  2. Compute diffs to find lines with bugs
  3. Create JSONL prompt with input code and expected buggy line(s)

> See `create_dataset/build_bugfix_prompts.py` for extraction logic

---

## Fine-tuning setup
- **Base Model**: [`WizardLMTeam/WizardMath-7B-V1.1`](https://huggingface.co/WizardLMTeam/WizardMath-7B-V1.1)
- **Adapter**: LoRA
- **Frameworks**: Hugging Face `transformers`, `peft`, `bitsandbytes`
- **Quantization**: 4-bit (NF4) using `bitsandbytes`
- **Platform**: Trained on RunPod A40 GPU instance
- **Training Details**:
  - LoRA fine-tuning on ~4320 examples
  - Sequence length: 4096
  - Epochs: 3
  - Micro batch size: 2
  - Gradient Accumulation: 4
  - Optimizer: `adamw_bnb_8bit`
- **Output**:
  - Final adapter saved as `bugfix-wizardmath`
  - Published to Hugging Face Hub: [`shhhks/bugfix-wizardmath`](https://huggingface.co/shhhks/bugfix-wizardmath/tree/main)

> See `train.py` for full training pipeline

---

## Evaluation
The model was evaluated on a held-out test set of 432 examples.

### Scripts:
- `inference.py`: generates predictions on test set
- `evaluate_results.py`: compares predictions to ground truth (line numbers)

### Limitations:
- Local inference fails if bitsandbytes lacks GPU support
- Dataset is small and limited to 8 python repos