"""
train_qlora_7b.py
=================
Phase 2: Universal Latent SFT for Mamba-2.8B

APPROACH: FP16 half-precision with manual parameter freezing.
BitsAndBytes 4-bit quantization is INCOMPATIBLE with Mamba's fused CUDA scan
kernels (which call `dt_proj.weight @ x` directly, bypassing BNB's dequant proxy).

Instead, we load in FP16 and manually freeze all backbone parameters,
only leaving `x_proj` and `dt_proj` trainable. At FP16, the 2.8B model
uses ~5.6GB VRAM which fits cleanly in a 12GB GPU.
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)

print("===============================================================")
print("  PHASE 2: MAMBA-2.8B LATENT FORGE (FP16 + MANUAL FREEZE)")
print("===============================================================")

MODEL_ID = "state-spaces/mamba-2.8b-hf"
DATA_FILE = "data/universal_7b_latent.jsonl"
OUTPUT_DIR = "checkpoints/mamba-2.8b-phase2"
MAX_LEN = 512

print(f"[INIT] Loading Tokenizer for {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("[INIT] Loading Mamba-2.8B in FP16 (BNB-free for CUDA kernel compatibility)...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
except Exception as e:
    print(f"[FATAL ERROR] Model instantiation failed: {e}")
    exit(1)

print("[INIT] Freezing backbone — leaving only x_proj and dt_proj trainable...")
trainable_params = 0
all_params = 0
for name, param in model.named_parameters():
    all_params += param.numel()
    # Only train the discrete-time projection and input-projection matrices
    if "x_proj" in name or "dt_proj" in name:
        param.requires_grad = True
        trainable_params += param.numel()
    else:
        param.requires_grad = False

pct = 100 * trainable_params / all_params
print(f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {pct:.4f}")

print("[INIT] Loading Universal Latent Dataset...")
raw_dataset = load_dataset("json", data_files=DATA_FILE, split="train")

def tokenize(example):
    """Tokenize the full formatted prompt+loop+answer sequence."""
    prompt  = example['instruction']
    loops   = example['dark_loops']
    answer  = example['response']
    full_text = f"{prompt}{loops}\n{answer}{tokenizer.eos_token}"
    tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length"
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

print("[INIT] Tokenizing 10,164 rows...")
tokenized = raw_dataset.map(tokenize, remove_columns=raw_dataset.column_names)
tokenized.set_format("torch")

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=500,
    save_steps=250,
    fp16=False,
    bf16=True,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_steps=25,
    report_to="none"
)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

print("\n[SYSTEM] Igniting the Mamba Latent Forge (FP16 stable).")
trainer = Trainer(
    model=model,
    train_dataset=tokenized,
    args=args,
    data_collator=collator,
)

try:
    trainer.train()
    print("\n[SYSTEM] Phase 2 Complete. Latent Geometry acquired.")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[SYSTEM] Model saved to {OUTPUT_DIR}")
except Exception as e:
    import traceback
    print(f"\n[CRITICAL FAILURE] Forge collapsed: {e}")
    traceback.print_exc()
