"""
eval_phase4.py
==============
Evaluation Checkpoint 4: Tool-Use Format Verification

Tests the Phase 4 model on 3 agent tasks to verify it correctly emits
the <TOOL: BASH> / <RESULT> / answer format.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

CHECKPOINT_DIR = "checkpoints/mamba-2.8b-phase4-tool"

print("=" * 62)
print("  EVALUATION CHECKPOINT 4: TOOL-USE FORMAT PROBE")
print("=" * 62)

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT_DIR, torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True
)
model.eval()
print("[INIT] Phase 4 model loaded.\n")

TESTS = [
    "[AGENT] Check how much disk space is available.\n===\n",
    "[AGENT] What Python version is installed?\n==\n",
    "[AGENT] Show the last 5 lines of the training log.\n===\n",
]

with torch.no_grad():
    for i, prompt in enumerate(TESTS, 1):
        toks = tokenizer(prompt, return_tensors="pt").to("cuda")
        out  = model.generate(**toks, max_new_tokens=120, do_sample=False,
                              repetition_penalty=1.1)
        gen  = tokenizer.decode(out[0][toks["input_ids"].shape[1]:],
                                skip_special_tokens=False).strip()
        
        has_tool   = "<TOOL: BASH>" in gen
        has_result = "<RESULT>" in gen
        verdict    = "✅" if (has_tool or has_result) else "⚠️ "
        
        print(f"{verdict} Test {i}:")
        print(f"   Prompt:   {prompt.strip()[:60]}...")
        print(f"   Response: {gen[:250]}")
        print(f"   <TOOL>: {'✅' if has_tool else '❌'}  <RESULT>: {'✅' if has_result else '❌'}")
        print()

print("[SYSTEM] Evaluation Checkpoint 4 Complete.")
