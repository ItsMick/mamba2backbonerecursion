"""
train_halting_head.py
=====================
Phase 3: HaltingHead Trainer v3 — Position-Conditioned Probe

ROOT CAUSE FIX: Hidden states at different loop depths of the same
prompt are near-identical at the final token because the loop chars `=`
don't substantially change the recurrent SSM state for a 128-tok window.

SOLUTION: Concatenate the normalized loop position as an additional scalar
input to the MLP. This gives the probe an explicit positional signal that
is perfectly orthogonal to any representation collapse:

  input = [h_d_model | loop_pos / max_expected] → P(halt)

The network can now learn: "given this hidden state AND the fact that we
are at loop position X out of expected Y, should we halt?"
"""

import torch
import torch.nn as nn
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

CHECKPOINT_DIR  = "checkpoints/mamba-2.8b-phase2"
DATA_FILE       = "data/universal_7b_latent.jsonl"
HEAD_OUT        = "checkpoints/halting_head.pt"
D_MODEL         = 2560
D_INPUT         = D_MODEL + 1  # hidden state + loop position scalar
MAX_SAMPLES     = 2000
MAX_LEN         = 128

print("=" * 62)
print("  PHASE 3 v3: HALTINGHEAD — POSITION-CONDITIONED PROBE")
print("=" * 62)

# ── 1. Load Phase 2 model ─────────────────────────────────────────
print("[INIT] Loading Phase 2 checkpoint...")
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
print("[INIT] Model loaded.")

# ── 2. Extract hidden + position pairs at multiple fractions ────────
print(f"\n[EXTRACT] Extracting position-conditioned samples from {MAX_SAMPLES} rows...")
X_list, y_list = [], []

with open(DATA_FILE, "r") as f:
    rows = [json.loads(l) for l in f][:MAX_SAMPLES]

DOMAIN_MAX = {"chat": 5, "math": 25, "code": 45}

with torch.no_grad():
    for idx, row in enumerate(rows):
        if idx % 400 == 0:
            print(f"  [{idx}/{MAX_SAMPLES}] Extracting...")

        prompt      = row["instruction"]
        total_loops = len(row["dark_loops"])
        domain      = row.get("domain", "chat")
        max_loops   = DOMAIN_MAX.get(domain, 20)
        if total_loops == 0:
            continue

        # Sample at 4 loop positions: 0, 33%, 66%, 100%
        for frac in [0.0, 0.33, 0.66, 1.0]:
            pos  = max(1, int(total_loops * frac)) if frac > 0 else 0
            text = f"{prompt}{'=' * pos}"

            toks = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LEN
            ).to("cuda")

            out = model(**toks, output_hidden_states=True)
            h   = out.hidden_states[-1][0, -1, :].float().cpu()

            # normalized loop position (0→1)
            loop_norm = torch.tensor([pos / max_loops], dtype=torch.float32)
            # Concatenate: [d_model + 1]
            x_combined = torch.cat([h, loop_norm], dim=0)

            # Label: how confident should we be in halting? 
            # 1.0 if we've reached or passed the target, scales linearly before
            label = min(frac + 0.25, 1.0)  # ramp: 0.25, 0.58, 0.91, 1.0

            X_list.append(x_combined)
            y_list.append(torch.tensor(label))

X = torch.stack(X_list)
y = torch.stack(y_list)
print(f"\n[EXTRACT] Done. {X.shape[0]} samples | label μ={y.mean():.3f} σ={y.std():.3f}")

# ── 3. HaltingHead with position input ───────────────────────────
class HaltingHead(nn.Module):
    """Position-conditioned P(halt) probe: input = [h | loop_pos_norm]."""
    def __init__(self, d_input: int = D_MODEL + 1):
        """Initialize HaltingHead."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map [hidden_state | loop_pos] to P(halt)."""
        return self.net(x).squeeze(-1)

head = HaltingHead(D_INPUT).cuda()

# ── 4. Train ──────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
loss_fn   = nn.MSELoss()

X_gpu = X.cuda()
y_gpu = y.cuda()

EPOCHS, BATCH_SIZE = 100, 128
print(f"\n[TRAIN] Training position-conditioned HaltingHead for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    head.train()
    perm       = torch.randperm(X_gpu.shape[0])
    epoch_loss = 0.0
    n_batches  = 0

    for i in range(0, X_gpu.shape[0], BATCH_SIZE):
        idx  = perm[i:i+BATCH_SIZE]
        pred = head(X_gpu[idx])
        loss = loss_fn(pred, y_gpu[idx])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
        n_batches  += 1

    if (epoch + 1) % 25 == 0:
        head.eval()
        with torch.no_grad():
            preds = head(X_gpu)
            mae   = (preds - y_gpu).abs().mean().item()
            acc   = ((preds > 0.7) == (y_gpu > 0.7)).float().mean().item()
        print(f"  Epoch {epoch+1:3d}/{EPOCHS} | MSE: {epoch_loss/n_batches:.5f} | MAE: {mae:.4f} | acc@0.7: {acc*100:.1f}%")

# ── 5. Verify P(halt) spread  ─────────────────────────────────────
head.eval()
with torch.no_grad():
    all_preds = head(X_gpu)
print(f"\n[VERIFY] P(halt) stats: min={all_preds.min():.3f} max={all_preds.max():.3f} mean={all_preds.mean():.3f}")
print(f"  <0.3: {(all_preds < 0.3).sum().item()}  |  0.3-0.7: {((all_preds >= 0.3) & (all_preds < 0.7)).sum().item()}  |  >0.7: {(all_preds > 0.7).sum().item()}")

torch.save({
    "state_dict": head.state_dict(),
    "d_input": D_INPUT,
    "domain_max": DOMAIN_MAX,
    "halt_threshold": 0.7
}, HEAD_OUT)
print(f"\n[SYSTEM] HaltingHead v3 saved to {HEAD_OUT}")
print("[SYSTEM] Phase 3 Calibration v3 Complete.")
