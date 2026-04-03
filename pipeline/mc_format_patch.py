"""
mc_format_patch.py
==================
50-step MC Format Patch

Teaches the model to output clean "A", "B", "C", or "D" after
its chain-of-thought reasoning. Does NOT overwrite existing
reasoning behaviour — just adds the final letter anchoring.

Uses ARC-C style, HellaSwag style, and Winogrande style examples
with the exact [LOGIC]/[CHAT] domain tags the model already knows.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset as HFDataset
import shutil, os

ENGINE_DIR = "checkpoints/mamba-2.8b-latent"
OUT_DIR    = "checkpoints/mamba-2.8b-latent-mc"
MAX_LEN    = 300

print("=" * 58)
print("  MC FORMAT PATCH — 50-STEP SFT")
print("=" * 58)

tok = AutoTokenizer.from_pretrained(ENGINE_DIR, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

mdl = AutoModelForCausalLM.from_pretrained(
    ENGINE_DIR, torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True
)

# Only train the token embedding output projection — lightest possible touch
for name, param in mdl.named_parameters():
    param.requires_grad = any(k in name for k in ["x_proj", "dt_proj", "embeddings"])

trainable = sum(p.numel() for p in mdl.parameters() if p.requires_grad)
print(f"[DATA] Trainable params: {trainable:,} (lm_head + embed_tokens only)")

# ── patch dataset ─────────────────────────────────────────────────
# Template: chain-of-thought reasoning → final letter on new line
# Deliberately vary loop depth (=, ==, ===) and all 4 domains

ROWS = []

# ARC-style: science reasoning
ARC_EXAMPLES = [
    ("[LOGIC] Which material is a good conductor of electricity?\nA: Wood\nB: Rubber\nC: Copper\nD: Glass\nThink step by step. The answer is letter:", "=====", "Metals like copper allow electrons to flow freely, making them conductors.\nAnswer: C"),
    ("[LOGIC] What happens to water when it is heated to 100°C at sea level?\nA: It freezes\nB: It evaporates slowly\nC: It boils\nD: It becomes denser\nThink step by step. The answer is letter:", "======", "At 100°C and standard pressure, water reaches its boiling point and turns to steam.\nAnswer: C"),
    ("[LOGIC] Which planet is closest to the Sun?\nA: Earth\nB: Venus\nC: Mars\nD: Mercury\nThink step by step. The answer is letter:", "=====", "Mercury orbits the Sun at an average distance of 0.39 AU, the shortest of any planet.\nAnswer: D"),
    ("[LOGIC] What force keeps planets in orbit around the Sun?\nA: Magnetism\nB: Friction\nC: Gravity\nD: Centrifugal force\nThink step by step. The answer is letter:", "======", "Gravity is the attractive force between masses. The Sun's gravity keeps planets in orbit.\nAnswer: C"),
    ("[LOGIC] Which of the following is a renewable energy source?\nA: Coal\nB: Natural gas\nC: Oil\nD: Solar power\nThink step by step. The answer is letter:", "=====", "Solar power is generated from sunlight, which is continuously available and renewable.\nAnswer: D"),
    ("[LOGIC] A magnet will attract which of the following?\nA: Aluminum\nB: Steel\nC: Plastic\nD: Glass\nThink step by step. The answer is letter:", "======", "Steel contains iron, which is ferromagnetic and strongly attracted to magnets.\nAnswer: B"),
    ("[LOGIC] What is the chemical symbol for gold?\nA: Go\nB: Gd\nC: Au\nD: Ag\nThink step by step. The answer is letter:", "=====", "Gold's symbol Au comes from the Latin word 'Aurum'.\nAnswer: C"),
    ("[LOGIC] Which process do plants use to make food from sunlight?\nA: Respiration\nB: Fermentation\nC: Photosynthesis\nD: Digestion\nThink step by step. The answer is letter:", "======", "Plants convert sunlight, CO2 and water into glucose via photosynthesis.\nAnswer: C"),
]

# HellaSwag-style: sentence completion
HELLA_EXAMPLES = [
    ("[CHAT] Complete: A person is shown mixing ingredients in a bowl.\nA: They then pour the batter into a pan and bake it.\nB: They suddenly start running outside.\nC: They sit down and read a book.\nD: They drive to the store.\nBest completion:", "==", "Mixing batter logically leads to baking.\nAnswer: A"),
    ("[CHAT] Complete: Someone is shown tying their shoes before going out.\nA: They take a nap immediately.\nB: They walk out the door.\nC: They start cooking dinner.\nD: They paint the walls.\nBest completion:", "==", "After tying shoes, the natural next step is leaving.\nAnswer: B"),
    ("[CHAT] Complete: A dog is shown chasing a ball across a field.\nA: The dog falls asleep mid-run.\nB: The dog retrieves the ball and brings it back.\nC: The dog starts digging a hole.\nD: The dog ignores the ball completely.\nBest completion:", "==", "Dogs typically retrieve balls when chasing them.\nAnswer: B"),
    ("[CHAT] Complete: A chef is shown carefully plating a dish in a restaurant kitchen.\nA: They throw the food away.\nB: They hand the plate to a waiter to serve.\nC: They eat it themselves immediately.\nD: They refrigerate it for a week.\nBest completion:", "==", "A plated dish in a restaurant is sent out to a customer via a waiter.\nAnswer: B"),
]

# Winogrande-style: fill in the blank
WINO_EXAMPLES = [
    ("[CHAT] Fill in the blank.\nSentence: Sarah carried the groceries while Emma opened the door because ___ had her hands full.\nA: Emma\nB: Sarah\nAnswer A or B:", "==", "Sarah was carrying groceries, so her hands were full.\nAnswer: B"),
    ("[CHAT] Fill in the blank.\nSentence: The trophy didn't fit in the brown case because it was too big.\nA: The trophy\nB: The case\nAnswer A or B:", "==", "The 'it' refers to the thing that is too big to fit — the trophy.\nAnswer: A"),
    ("[CHAT] Fill in the blank.\nSentence: Mark told Jake he needed to study more because ___ failed the test.\nA: Jake\nB: Mark\nAnswer A or B:", "==", "Mark is giving advice to Jake, implying Jake is the one who failed.\nAnswer: A"),
    ("[CHAT] Fill in the blank.\nSentence: The dog chased the cat until ___ was exhausted.\nA: the dog\nB: the cat\nAnswer A or B:", "==", "The dog was doing the chasing and would tire first.\nAnswer: A"),
]

for prompt, loops, answer in ARC_EXAMPLES:
    for _ in range(10):  # repeat for weight
        ROWS.append({"full_text": f"{prompt}{loops}\n{answer}"})

for prompt, loops, answer in HELLA_EXAMPLES:
    for _ in range(8):
        ROWS.append({"full_text": f"{prompt}{loops}\n{answer}"})

for prompt, loops, answer in WINO_EXAMPLES:
    for _ in range(8):
        ROWS.append({"full_text": f"{prompt}{loops}\n{answer}"})

print(f"[DATA] {len(ROWS)} patch rows  ({len(ARC_EXAMPLES)*10} ARC + {len(HELLA_EXAMPLES)*8} Hella + {len(WINO_EXAMPLES)*8} Wino)")

def tokenize(row):
    t = tok(row["full_text"], truncation=True, max_length=MAX_LEN, padding="max_length")
    t["labels"] = t["input_ids"].copy()
    return t

ds = HFDataset.from_list(ROWS)
ds = ds.map(tokenize, remove_columns=ds.column_names)
ds.set_format("torch")

args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=3e-5,       # Very conservative — only lm_head
    max_steps=75,
    logging_steps=10,
    save_steps=75,
    bf16=True, fp16=False,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_steps=5,
    report_to="none"
)

collator = DataCollatorForLanguageModeling(tok, mlm=False)

print(f"[TRAIN] Running 75-step MC format patch (LR=3e-5, lm_head only)...")
trainer = Trainer(model=mdl, train_dataset=ds, args=args, data_collator=collator)
trainer.train()

print("\n[SAVE] Saving patched model...")
trainer.save_model(OUT_DIR)
tok.save_pretrained(OUT_DIR)
shutil.copy2(f"{ENGINE_DIR}/halting_head.pt", f"{OUT_DIR}/halting_head.pt")
shutil.copy2(f"{ENGINE_DIR}/engine_manifest.json", f"{OUT_DIR}/engine_manifest.json")
print(f"[SAVE] Done → {OUT_DIR}")
