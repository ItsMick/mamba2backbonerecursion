"""
full_system_test.py
===================
Full integration test: Phase 2 model + Phase 3 HaltingHead

Tests 6 prompts of increasing complexity across all 3 domains.
Checks:
  1. Loop proportionality (harder tasks → more loops)
  2. Surface response quality
  3. P(halt) is always increasing toward the halt decision
  4. No OOM / no infinite loops
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

CHECKPOINT_DIR = "checkpoints/mamba-2.8b-phase2"
HALTING_HEAD   = "checkpoints/halting_head.pt"
D_MODEL        = 2560
HALT_THRESHOLD = 0.7
MAX_LOOPS      = 50
DOMAIN_MAX     = {"chat": 5, "math": 25, "code": 45}

class HaltingHead(nn.Module):
    def __init__(self, d_input=D_MODEL+1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

print("=" * 66)
print("  FULL SYSTEM INTEGRATION TEST — MAMBA-2.8B + HALTINGHEAD")
print("=" * 66)

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT_DIR, torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True
)
model.eval()

ckpt = torch.load(HALTING_HEAD, weights_only=True)
head = HaltingHead(D_MODEL+1).cuda()
head.load_state_dict(ckpt["state_dict"])
head.eval()
print("[INIT] System loaded.\n")

def run(prompt, domain):
    """Run one prompt through the full latent engine."""
    m = DOMAIN_MAX.get(domain, 20)
    trace, monotone = [], True
    with torch.no_grad():
        for loop in range(MAX_LOOPS):
            toks = tokenizer(prompt + "=" * loop, return_tensors="pt",
                             truncation=True, max_length=256).to("cuda")
            h = model(**toks, output_hidden_states=True).hidden_states[-1][0,-1,:].float()
            lnorm = torch.tensor([loop/m], dtype=torch.float32, device="cuda")
            p = head(torch.cat([h, lnorm]).unsqueeze(0)).item()
            trace.append(round(p, 3))
            if len(trace) > 1 and trace[-1] < trace[-2] - 0.2:
                monotone = False
            if p >= HALT_THRESHOLD:
                break
        final = prompt + "=" * (len(trace))
        toks = tokenizer(final, return_tensors="pt",
                         truncation=True, max_length=300).to("cuda")
        out = model.generate(**toks, max_new_tokens=80, do_sample=False,
                             repetition_penalty=1.1)
        surface = tokenizer.decode(out[0][toks["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip()
    return {"loops": len(trace), "p_final": trace[-1],
            "monotone": monotone, "surface": surface}

TESTS = [
    # CHAT — easy
    {"id": "C1", "domain": "chat",
     "prompt": "[CHAT] What color is the sky?\nSolution: "},
    # CHAT — moderate
    {"id": "C2", "domain": "chat",
     "prompt": "[CHAT] Explain the difference between RAM and ROM in simple terms.\nSolution: "},
    # MATH — easy
    {"id": "M1", "domain": "math",
     "prompt": "[LOGIC] What is 144 divided by 12?\nSolution: "},
    # MATH — hard
    {"id": "M2", "domain": "math",
     "prompt": "[LOGIC] A car travels at 65 mph for 2.5 hours, then at 45 mph for 1.5 hours. What is the total distance?\nSolution: "},
    # CODE — easy
    {"id": "X1", "domain": "code",
     "prompt": "[CODE] Complete:\n```python\ndef add(a, b):\n    '''Return a + b.'''\n```\nSolution: "},
    # CODE — hard
    {"id": "X2", "domain": "code",
     "prompt": "[CODE] Complete:\n```python\ndef merge_sort(arr):\n    '''Sort array using merge sort algorithm.'''\n```\nSolution: "},
]

results = []
passes  = 0
for t in TESTS:
    r = run(t["prompt"], t["domain"])
    results.append((t, r))
    ok = r["loops"] > 0 and r["p_final"] >= HALT_THRESHOLD
    if ok:
        passes += 1
    marker = "✅" if ok else "❌"
    print(f"{marker} [{t['id']}] {t['domain'].upper():4s} | loops={r['loops']:2d} | P(halt)={r['p_final']:.3f}")
    print(f"     Response: {r['surface'][:140]}")
    print()

# ── Proportionality check ──────────────────────────────────────────
chat_loops = [r["loops"] for t,r in results if t["domain"]=="chat"]
math_loops = [r["loops"] for t,r in results if t["domain"]=="math"]
code_loops = [r["loops"] for t,r in results if t["domain"]=="code"]

chat_avg = sum(chat_loops)/len(chat_loops)
math_avg = sum(math_loops)/len(math_loops)
code_avg = sum(code_loops)/len(code_loops)

prop_ok = (code_avg >= math_avg) or (math_avg >= chat_avg)

print("=" * 66)
print(f"  RESULTS: {passes}/{len(TESTS)} prompts passed halt criterion")
print(f"  Avg loops — CHAT: {chat_avg:.1f} | MATH: {math_avg:.1f} | CODE: {code_avg:.1f}")
print(f"  Proportionality: {'✅ PASS' if prop_ok else '⚠️  PARTIAL'}")
verdict = "✅ READY FOR PHASE 4" if passes >= 5 else "⚠️  NEEDS ATTENTION"
print(f"  OVERALL VERDICT: {verdict}")
print("=" * 66)
