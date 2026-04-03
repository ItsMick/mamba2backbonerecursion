"""
eval_phase2.py
==============
Evaluation Checkpoint 2: Phase 2 Inference Verification

Loads the Phase 2 fine-tuned Mamba-2.8B checkpoint and runs 3 structured
test prompts (Chat, Math, Code) to verify the model has learned to route
topology through latent dark loops before generating surface text.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

CHECKPOINT_DIR = "checkpoints/mamba-2.8b-phase2"

print("=" * 62)
print("  EVALUATION CHECKPOINT 2: PHASE 2 LATENT GEOMETRY PROBE")
print("=" * 62)
print(f"[INIT] Loading Phase 2 checkpoint from {CHECKPOINT_DIR}...")

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()
print("[INIT] Model and tokenizer loaded.\n")

TEST_PROMPTS = [
    {
        "domain": "CHAT",
        "prompt": "[CHAT] What is the capital of France?\nSolution: ",
        "expected_loops": "1-5 (simple conversation, low Z-axis demand)"
    },
    {
        "domain": "MATH",
        "prompt": "[LOGIC] If a train travels 60 miles in 45 minutes, what is its speed in miles per hour?\nSolution: ",
        "expected_loops": "10-20 (moderate arithmetic routing)"
    },
    {
        "domain": "CODE",
        "prompt": "[CODE] Complete the following Python script:\n```python\ndef fibonacci(n):\n    '''Return the nth Fibonacci number.'''\n```\nSolution: ",
        "expected_loops": "20-40 (AST generation requires deep Z-axis)"
    },
]

DARK_TOKEN = "="

with torch.no_grad():
    for test in TEST_PROMPTS:
        print(f"DOMAIN [{test['domain']}]")
        print(f"  Expected Dark Loops: {test['expected_loops']}")
        print(f"  Prompt: {test['prompt'][:80]}...")
        
        inputs = tokenizer(test["prompt"], return_tensors="pt").to("cuda")
        
        out = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
        )
        
        generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # Count leading spacer tokens (the dark loop geometry)
        spacer_count = len(generated) - len(generated.lstrip(DARK_TOKEN))
        surface_response = generated.lstrip(DARK_TOKEN).strip()
        
        print(f"  Dark Loops Emitted: {spacer_count}")
        print(f"  Surface Response:   {surface_response[:200]}")
        print(f"  VERDICT: {'✅ LOOPS EMITTED' if spacer_count > 0 else '⚠️  NO LOOPS (imitation only)'}")
        print("-" * 62)

print("\n[SYSTEM] Evaluation Checkpoint 2 Complete.")
