"""
train_tool_sft.py
=================
Phase 4: Tool-Use SFT Trainer

Fine-tunes the Phase 2 Mamba-2.8B checkpoint on the ReAct/BASH tool-use
dataset. Trains the model to emit structured agentic sequences:
  [AGENT] task → loops → <TOOL: BASH> cmd </TOOL> → <RESULT> → loops → answer

Uses the same FP16+manual freeze approach proven stable in Phase 2,
but with a lower LR (1e-4) appropriate for fine-tune over SFT.
"""

import torch
import json
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)

CHECKPOINT_DIR = "checkpoints/mamba-2.8b-phase2"
DATA_FILE      = "data/tool_use_dataset.jsonl"
OUTPUT_DIR     = "checkpoints/mamba-2.8b-phase4-tool"
MAX_LEN        = 384

print("=" * 62)
print("  PHASE 4: TOOL-USE SFT (REACT/BASH ALIGNMENT)")
print("=" * 62)

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Add special tool tokens so the model can reliably parse them
special_tokens = ["<TOOL: BASH>", "</TOOL>", "<RESULT>", "</RESULT>", "[AGENT]"]
tokenizer.add_tokens(special_tokens)
print(f"[INIT] Added {len(special_tokens)} special tool tokens to vocabulary.")

print(f"[INIT] Loading Phase 2 checkpoint from {CHECKPOINT_DIR}...")
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model.resize_token_embeddings(len(tokenizer))

# Freeze backbone — train only x_proj, dt_proj, and the new token embeddings
trainable, total = 0, 0
for name, param in model.named_parameters():
    total += param.numel()
    if any(k in name for k in ["x_proj", "dt_proj", "embed_tokens"]):
        param.requires_grad = True
        trainable += param.numel()
    else:
        param.requires_grad = False

pct = 100 * trainable / total
print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {pct:.4f}")

# Dataset
with open(DATA_FILE) as f:
    rows = [json.loads(l) for l in f]

def tokenize(row):
    """Tokenize full agentic tool-use sequence."""
    tokens = tokenizer(
        row["full_text"],
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length"
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

from datasets import Dataset as HFDataset
raw = HFDataset.from_list(rows)
tokenized = raw.map(tokenize, remove_columns=raw.column_names)
tokenized.set_format("torch")

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,       # Lower LR for fine-tune over SFT
    logging_steps=5,
    max_steps=200,            # Dataset is small, 200 steps is sufficient
    save_steps=100,
    fp16=False,
    bf16=True,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_steps=10,
    report_to="none"
)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

print("\n[SYSTEM] Igniting Phase 4 Tool-Use SFT Forge.")
trainer = Trainer(
    model=model,
    train_dataset=tokenized,
    args=args,
    data_collator=collator,
)

try:
    trainer.train()
    print("\n[SYSTEM] Phase 4 SFT Complete. Tool routing geometry acquired.")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[SYSTEM] Model saved to {OUTPUT_DIR}")
except Exception as e:
    import traceback
    print(f"\n[CRITICAL FAILURE] {e}")
    traceback.print_exc()
