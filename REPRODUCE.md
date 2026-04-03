# Reproducing the Mamba-2.8B Latent Reasoning Engine

Complete step-by-step pipeline. Entire run was done on a single **RTX 3060 12GB** in `bfloat16`. No QLoRA/bitsandbytes — manual layer freezing only.

---

## Hardware & Environment

```bash
# Tested on:
#   GPU:    NVIDIA RTX 3060 12GB
#   VRAM:   12 GB (11.6 GB usable)
#   OS:     Ubuntu 22.04 / Linux
#   Python: 3.10+
#   CUDA:   12.x

pip install -r requirements.txt
```

> **Base model:** `state-spaces/mamba-2.8b-hf` (~5.5 GB download from HuggingFace)

---

## Phase 1 — Build the Latent Dataset

Pulls from UltraChat (chat), GSM8K (math), HumanEval (code) and reformats to:
`[LOGIC] question ==== answer` / `[CHAT] ...` / `[CODE] ...`

```bash
python pipeline/phase1_build_dataset.py
# Output: data/latent_dataset.jsonl  (~10,164 rows)
```

**Verify:**
```bash
python eval/test_phase1.py
```

---

## Phase 2 — Latent SFT (Carve the Loop Pathways)

Trains only `x_proj`, `dt_proj`, `embed_tokens` layers (frozen core). BF16.
Loss 17.3 → 10.5 over 500 steps.

```bash
python pipeline/phase2_sft_trainer.py
# Checkpoint: checkpoints/mamba-2.8b-sft/
# ~45 min on RTX 3060
```

**Verify:**
```bash
python eval/eval_phase2_sft.py
```

---

## Phase 3 — Train the HaltingHead (Adaptive Computation Time)

Extracts hidden states from Phase 2 checkpoint and trains a 3-layer MLP probe
on fractional ramp targets conditioned on loop depth.

```bash
python pipeline/phase3_train_halting_head.py
# Output: checkpoints/mamba-2.8b-sft/halting_head.pt
# Target: MAE < 0.06  (achieved: 0.052)
```

**Verify:**
```bash
python eval/eval_phase3_halting_head.py
# Expect: ~88% halt-step accuracy on 2000-sample probe
```

---

## Phase 4 — Tool-Use SFT (ReAct / Bash Execution)

72-row dataset with `<TOOL: BASH>` and `<RESULT>` tags.
Loss 13.7 → 0.9 in 200 steps.

```bash
python pipeline/phase4_build_tool_dataset.py
# Output: data/tool_dataset.jsonl

python pipeline/phase4_tool_sft_trainer.py
# Checkpoint: checkpoints/mamba-2.8b-tool/
```

**Verify:**
```bash
python eval/eval_phase4_tool_use.py
# Expect: 3/3 tool emission tests pass
```

---

## Phase 5 — Merge & Export Production Engine

Merges Phase 4 checkpoint + HaltingHead probe into a single deployable directory.

```bash
python pipeline/phase5_merge_and_export.py
# Output: checkpoints/mamba-2.8b-latent/
#   ├── config.json / tokenizer.json / model.safetensors
#   ├── halting_head.pt
#   └── engine_manifest.json
```

---

## Phase 6 — Session Memory (32 KB Persistent State)

No training required — purely inference-time. Test it:

```bash
python session_memory.py --new-session demo
# /save  /load demo  /history  /clear
```

---

## Phase 7 — Live Agent Loop

```bash
python agent_loop.py "How much disk space is available?"
python agent_loop.py "What Python version is installed?"
```

---

## Scientific Validation — The Crucible

Run the 4-proof scientific test harness:

```bash
python eval/the_crucible.py
# Proof 1: ACT loop proportionality
# Proof 2: O(1) VRAM flatline  
# Proof 3: Lobotomy ablation (W=8 vs W=4)
# Proof 4: Tool execution

# Expected ACT results:
#   ARC-Challenge: ~5.9 loops avg
#   HellaSwag:     ~2.0 loops avg
#   Ratio:          ~3x  ← emergent scaling
```

---

## Full Benchmark Suite

```bash
# Generative eval — all 4 tasks with reasoning loops active
python eval/generative_benchmark.py

# Format-free content eval — answer text matching, no A/B/C/D
python eval/content_benchmark.py

# ARC-Challenge only, fastest (200 samples, ~7 min)
python eval/eval_latent_arc.py

# Log-likelihood lobotomy baseline (for comparison)
lm_eval --model hf \
  --model_args pretrained=checkpoints/mamba-2.8b-latent,trust_remote_code=True,dtype=bfloat16 \
  --tasks arc_challenge,hellaswag,piqa,winogrande \
  --num_fewshot 0 \
  --output_path benchmark_results/
```

---

## MC Format Patch (optional)

If generative scores show "Verbose Genius" failure (correct reasoning, wrong letter format):

```bash
python pipeline/mc_format_patch.py
# Trains embeddings only, 300 steps, LR=5e-4
# Output: checkpoints/mamba-2.8b-latent-mc/
```

---

## Checkpoint Sizes

| Checkpoint | Size | Note |
|---|---|---|
| `mamba-2.8b-sft/` | ~5.55 GB | After Phase 2 |
| `mamba-2.8b-tool/` | ~5.55 GB | After Phase 4 |
| `mamba-2.8b-latent/` | ~5.55 GB | **Final engine** |
| `halting_head.pt` | ~5 MB | Co-located with engine |
| `sessions/demo.pt` | ~32 KB | Per-session state |

> Intermediate checkpoints can be deleted after merging. Only `mamba-2.8b-latent/` is needed for inference.
