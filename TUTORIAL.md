# Latent Forcing — Reproduction Tutorial

Step-by-step guide to reproduce every result in this repo, from scratch to proof.

---

## Project Map

```
mamba2backbonerecursion/
│
├── TUTORIAL.md                      ← you are here
├── README.md                        ← quick start + results tables
├── requirements.txt                 ← pip install this first
│
├── ── TRAINING SCRIPTS ─────────────────────────────────────
│
├── finetune_mamba_130m_v28.py       ← v28: Latent Forcing (Mamba1, 130m)
├── finetune_mamba_130m_v28_nolf.py  ← NoLF: same arch, final-answer only (ablation control)
├── finetune_mamba_130m_v29.py       ← v29: Latent Forcing + <HALT> token (Mamba1)
├── finetune_mamba2_130m_v30.py      ← v30: Latent Forcing + <HALT> (Mamba2 backbone)
│
├── ── DATA ─────────────────────────────────────────────────
│
├── system2_logic_v1.json            ← multi-hop chain training data
├── mmlu_format_v17.json             ← factual QA training data
│
├── ── PROOF & EVALUATION ───────────────────────────────────
│
├── baseline_vs_v28.py               ← 2-way: BASE vs v28 (the original proof)
├── ablation_comparison.py           ← 3-way: BASE vs NoLF vs v28 (rigorous)
├── fixed_loop_probe.py              ← 5-loop no-halt raw trace
├── diagnostic_suite_v28.py         ← compact 4-phase diagnostic
└── diagnostic_big_v28.py           ← full 5-phase diagnostic
```

---

## Prerequisites

```bash
# Python 3.10+, CUDA GPU required (tested on 12GB VRAM)
pip install -r requirements.txt

# requirements.txt:
#   torch>=2.1.0
#   transformers>=4.40.0
#   mamba-ssm>=1.2.0
```

The `mamba-ssm` package requires CUDA and a compatible GPU (Ampere/Ada recommended).
It will compile Triton kernels on the first run — takes 2-3 minutes.

---

## Step 1: Understand the Core Idea

**The problem:** Recursive neural models loop their hidden state N times before answering.
Standard training supervises only the final answer. All intermediate loops are blind.

**Latent Forcing:** Supervise *every* loop with an explicit per-step target.

For `A = red. B = A. What is B?`:
```
Training targets:
  Loop 0 → predict "A"    (the anchor variable)
  Loop 1 → predict "red"  (the resolved value)
```

For `X = Apple. Y = X. Z = Y. What is Z?`:
```
Training targets:
  Loop 0 → predict "X"      (start of chain)
  Loop 1 → predict "Y"      (hop forward)
  Loop 2 → predict "Apple"  (resolved answer)
```

**v29/v30 extension:** Add one more loop target — `<HALT>`. The model learns to
autonomously predict when it is done, rather than relying on a Python heuristic.

```
  Loop 3 → predict "<HALT>" (I am done — return previous token)
```

---

## Step 2: Train the v28 Baseline (Mamba1, ~1 hour)

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
python -u finetune_mamba_130m_v28.py 2>&1 | tee train_v28.log
```

**What to watch for:**
```
Step    50 | Loss: 2.231 | AllLoopAcc:  65.9% | FinalAcc:  68.8%
Step   100 | Loss: 0.313 | AllLoopAcc:  87.7% | FinalAcc:  92.1%
Step   500 | Loss: 0.011 | AllLoopAcc:  99.9% | FinalAcc:  99.9%
Step  1500 | ✅ EARLY STOP  Val AllLoopAcc: 100.0%  gap: +0.0pp
```

**Saves:** `mamba_130m_v28_latent_forcing_best.pt` (257 MB)

`AllLoopAcc` = accuracy at *every* loop, not just the final answer.
`100.0%` with `0.0pp` train-val gap = no memorization, learned the algorithm.

---

## Step 3: Run the Scientific Proof (3-way Ablation)

### 3a. Train the NoLF control (same arch, final-answer only, ~20 min)

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
python -u finetune_mamba_130m_v28_nolf.py 2>&1 | tee train_nolf.log
```

**Saves:** `mamba_130m_v28_nolf_best.pt`

### 3b. Run the 3-way comparison

```bash
python ablation_comparison.py 2>&1 | tee ablation_results.log
```

**What you should see:**

```
Chain: 3-hop: X→Y→Z=Apple  |  Answer: 'Apple'

Loop   BASE (no train)      NoLF (final-only)    v28 (Latent Forcing)
──────────────────────────────────────────────────────────────────────
L1     'Z'   p=0.412        'Apple' p=1.000 ✅   'X'     p=0.993
L2     'Z'   p=0.412        'Apple' p=0.992 ✅   'Y'     p=1.000
L3     'Z'   p=0.412        halted             →  'Y'     p=0.574
L4     'Z'   p=0.412                              'Apple' p=0.999 ✅

Chain: 4-hop: A→B→C→D=moon  |  Answer: 'moon'
L1-3   (empty, frozen)       'B' stuck p=0.93     B → B → B
L4     (frozen)              'B' stuck ❌          'moon' p=0.996 ✅
```

**Interpretation guide:**
- `BASE`: identical token at every loop (deterministic function, zero state update)
- `NoLF`: gets 1-hop right at L1 but **fails** 4-hop (stuck on `'B'` forever — learned a shortcut, not the algorithm)
- `v28`: traverses the chain step-by-step, resolves at loop N = hop count ✅

### 3c. Run the raw no-halt trace

```bash
python fixed_loop_probe.py 2>&1 | tee fixed_loop_results.log
```

Runs exactly 5 loops with zero halt logic. Cleanest view of what each model computes per pass.

---

## Step 4: Train v29 — `<HALT>` Self-Halting (Mamba1, ~1 hour)

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
python -u finetune_mamba_130m_v29.py 2>&1 | tee train_v29.log
```

v29 warm-starts from `mamba_130m_v28_latent_forcing_best.pt` automatically.

**What to watch — three new metrics:**
```
Step    50 | AllLoopAcc: 85.5% | AnswerAcc: 87.1% | HaltAcc: 82.1%
```

- `AllLoopAcc` = avg across all loops including the HALT loop
- `AnswerAcc`  = accuracy on chain-traversal loops specifically
- `HaltAcc`    = how often the model correctly predicts `<HALT>` on the last loop

**Saves:** `mamba_130m_v29_halt_best.pt`

**At inference:** model runs loops until it predicts `<HALT>`, then returns the previous token. No Python heuristic — the model decides when it's done.

---

## Step 5: Train v30 — Full Mamba2 End-to-End (~1-2 hours)

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
python -u finetune_mamba2_130m_v30.py 2>&1 | tee train_v30.log
```

**Differences from v28/v29:**
- Backbone: `state-spaces/mamba2-130m` (Mamba2 SSD blocks throughout)
- Loop engine: `Mamba2(headdim=64)` — 12 independent state heads at d=768
- Same `<HALT>` token mechanism as v29
- No warm-start (different architecture — trains from pretrained mamba2 weights)

**Saves:** `mamba2_130m_v30_halt_best.pt`

TPS will be ~750 (vs ~4,500 for Mamba1) due to richer SSD scan. Still converges in ~1,500 steps.

---

## Step 6: Full Diagnostic Suite

```bash
python diagnostic_big_v28.py 2>&1 | tee diagnostics.log
```

5 phases:
| Phase | What it tests |
|-------|--------------|
| 1 | OOD extrapolation — 4-10 hop chains (trained on 1-3) |
| 2 | Per-loop latent probe — token trace at each loop iteration |
| 3 | Dynamic halt — does compute scale with problem depth? |
| 4 | Reality override — can context beat pretrained world knowledge? |
| 5 | In-distribution accuracy (without pointer mask harness) |

---

## Architecture At a Glance

```
                      Input prompt
                           │
                    Embedding (d=768)
                           │
               ┌──── Layers 0-5 ─────┐
               │  FROZEN Mamba2/1    │  ← stable pretrained features
               └─────────────────────┘
                           │
         ┌─────────────────────────────────────────────┐
         │              LOOP (× N times)               │
         │                                             │
         │  + step_emb[loop_i]  ← "which tick am I?"  │
         │          │                                  │
         │  Layers 6-23 (LoRA rank=8 on in/out_proj)  │
         │          │                                  │
         │  + mamba_core(x)  ← stateful loop scan      │
         │          │         (Mamba1 v28/v29)          │
         │          │         (Mamba2 v30)              │
         │  loop_norm (RMSNorm)                        │
         │          │                                  │
         │  lm_head → loss vs chain_targets[loop_i]    │
         │      or → predict <HALT> to stop (v29/v30)  │
         └─────────────────────────────────────────────┘
                           │
                     Final answer
```

---

## Expected Results Summary

| Test | BASE | NoLF | v28 (Mamba1 LF) |
|------|------|------|-----------------|
| 1-hop correct | ❌ | ✅ L1 | ✅ L2 |
| 3-hop correct | ❌ | ✅ L1 (no traversal) | ✅ L4 (X→Y→Apple) |
| 4-hop correct | ❌ | ❌ stuck on 'B' | ✅ L4 |
| 5-hop correct | ❌ | ❌ stuck on 'B' | ✅ L4 |
| Reality override (novel) | 0/4 | 0/4 | 1/4 |
| Pointer traversal visible | ❌ | ❌ | ✅ |

**The proof is in the 4-hop result.** NoLF fails completely (learned a 1-hop shortcut). v28 succeeds by genuinely traversing the chain.

---

## Common Issues

**`ImportError: mamba_ssm not found`**
```bash
pip install mamba-ssm  # requires CUDA
```

**First run is slow (2-5 minutes at the start)**
Normal — Triton JIT is compiling the Mamba kernels. Subsequent runs are fast.

**OOM during training**
Reduce `BATCH_SIZE = 4` and `ACCUM = 8` at the top of the training script.
The effective batch size (`BATCH_SIZE * ACCUM = 32`) should stay the same.

**`mamba_130m_v28_latent_forcing_best.pt` not found when running v29**
Run Step 2 (v28 training) first. v29 warm-starts from that checkpoint.

**Tokenizer warning about unauthenticated HF requests**
Safe to ignore, or set `HF_TOKEN` in your environment for faster downloads.

---

## Reproducing the Paper Tables

The numbers in the paper (`latent_forcing_paper.md`) map to these commands:

| Paper section | Command |
|---------------|---------|
| Table 1 (training curve) | `grep "Step" train_v28.log` |
| Table 2 (3-hop trace) | `python ablation_comparison.py` |
| Table 3 (4-hop trace) | `python ablation_comparison.py` |
| Table 4 (reality override) | `python ablation_comparison.py` |
| Phase 2 latent probe | `python fixed_loop_probe.py` |
| Phase 5 accuracy | `python diagnostic_big_v28.py` |
