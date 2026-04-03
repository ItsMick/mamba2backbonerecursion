"""
eval_phase3.py
==============
Evaluation Checkpoint 3: HaltingHead Autonomous Scaling Test (v3)

Integrates v3 position-conditioned HaltingHead with Phase 2 Mamba-2.8B.
The head receives [hidden_state | loop_pos_norm] to autonomously determine
when to stop looping.
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
    """Position-conditioned P(halt) probe."""
    def __init__(self, d_input: int = D_MODEL + 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 64), nn.GELU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

print("=" * 62)
print("  EVALUATION CHECKPOINT 3 v3: POSITION-CONDITIONED PROBE")
print("=" * 62)

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT_DIR, torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True
)
model.eval()

checkpoint = torch.load(HALTING_HEAD, weights_only=True)
head = HaltingHead(D_MODEL + 1).cuda()
head.load_state_dict(checkpoint["state_dict"])
head.eval()
print("[INIT] Both components loaded.\n")

def generate_with_halting(prompt: str, domain: str) -> dict:
    """Autonomously loops until HaltingHead says P(halt) >= threshold."""
    max_loops   = DOMAIN_MAX.get(domain, 20)
    p_halt_trace = []

    with torch.no_grad():
        loop_count = 0
        for loop in range(MAX_LOOPS):
            text = prompt + "=" * loop
            toks = tokenizer(text, return_tensors="pt",
                             truncation=True, max_length=256).to("cuda")
            out  = model(**toks, output_hidden_states=True)
            h    = out.hidden_states[-1][0, -1, :].float()

            loop_norm = torch.tensor([loop / max_loops],
                                     dtype=torch.float32, device="cuda")
            x_combined = torch.cat([h, loop_norm], dim=0).unsqueeze(0)
            p_halt = head(x_combined).item()
            p_halt_trace.append(round(p_halt, 3))

            loop_count = loop + 1
            if p_halt >= HALT_THRESHOLD:
                break

        # Surface generation from final latent state
        final_text = prompt + "=" * loop_count
        toks = tokenizer(final_text, return_tensors="pt",
                         truncation=True, max_length=300).to("cuda")
        out = model.generate(**toks, max_new_tokens=80,
                              do_sample=False, repetition_penalty=1.1)
        surface = tokenizer.decode(
            out[0][toks["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

    return {"loops_used": loop_count, "p_halt_trace": p_halt_trace[-8:], "surface": surface}

TEST_PROMPTS = [
    {"domain": "chat", "label": "CHAT",
     "text": "[CHAT] What is the capital of France?\nSolution: "},
    {"domain": "math", "label": "MATH",
     "text": "[LOGIC] A rectangle has width 8 and height 5. What is its area?\nSolution: "},
    {"domain": "code", "label": "CODE",
     "text": "[CODE] Complete:\n```python\ndef is_prime(n):\n    '''Return True if n is prime.'''\n```\nSolution: "},
]

for test in TEST_PROMPTS:
    print(f"DOMAIN [{test['label']}]")
    result = generate_with_halting(test["text"], test["domain"])
    print(f"  Loops Used:   {result['loops_used']} / {DOMAIN_MAX[test['domain']]} (max)")
    print(f"  P(halt) tail: {result['p_halt_trace']}")
    print(f"  Surface:      {result['surface'][:200]}")
    print("-" * 62)

print("\n[SYSTEM] Evaluation Checkpoint 3 v3 Complete.")
