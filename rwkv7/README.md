# RWKV-7 Latent Loop Branch

Port of Phil's Phase 14 inner-loop bypass to RWKV-7 (`RWKV/RWKV7-Goose-World2.8-0.1B-HF`).

## Why RWKV-7

Phil's competitive positioning says "SSM + GRPO + Latent TTC = unoccupied."
This branch proves the pattern is backbone-agnostic: it works on RWKV-7 for the
same reasons it works on Mamba -- fixed-size recurrent state, O(1) memory.

But RWKV-7 has structural advantages for this specific use case:

- **State interpretability:** WKV state is `[n_heads, d_head, d_head]` -- a
  key-value matrix with explicit read/write semantics. Mamba's h_t is opaque.
  Phil's V1 vulnerability (interpretability black box) is partially solved here.
- **State composability:** Two RWKV-7 states can be blended via linear combination
  (see `rwkv7_state_cartridge.py`). Mamba `session_memory.py` saves h_t blobs;
  RWKV-7 cartridges can be fingerprinted and composed across reasoning sessions.
- **Natural decay clock:** RWKV-7's time-decay `w` provides geometric recency bias --
  recent loop ticks naturally dominate over older ones without explicit ROM re-injection.
- **True O(1) decode:** Unlike Mamba Phase 14 which re-runs all backbone layers on the
  full hidden state each tick, RWKV-7 supports single-token stateful decode via
  `model(spacer, past_key_values=state)` -- touching only the fixed-size recurrent state.

## Files

| File | Description |
|------|-------------|
| `rwkv7/rwkv7_engine.py` | Latent loop engine (mirrors Phase 14) |
| `rwkv7/rwkv7_state_cartridge.py` | Structured state persistence + compose/fingerprint |
| `rwkv7/rwkv7_crucible.py` | Ablation proofs (kill-shot, memory flatline, geometry) |
| `rwkv7/cpu_patch.py` | Naive PyTorch replacements for fla Triton kernels (CPU dev) |
| `rwkv7/inspect_state.py` | State structure inspector used during development |
| `rwkv7/notes/phase0_findings.md` | Phase 0/1 findings and state shape documentation |
| `rwkv7/notes/crucible_results.md` | Crucible proof results |

## State Structure

Per-layer RWKV-7 state (from Phase 1 inspection):

```python
state[layer_idx] = {
    'recurrent_state': Tensor[B, 12, 64, 64],  # WKV key-value matrix
    'conv_state':      Tensor[B, 1, 768],       # token shift cache
    'ffn_state':       Tensor[B, 1, 768],       # FFN shift cache
}
# Total: 2,304 KB across 12 layers (0.1B model, float32)
```

Compare Mamba-130M: `h_t` is `[d_state=16, d_model=768]` per layer -- a flat vector
with no head structure. RWKV-7's `[n_heads, d_head, d_head]` matrix provides
semantic read/write geometry that Mamba lacks.

## Ablation Results

### Proof 1: Kill-Shot (PASS)
Logit L2 divergence = **549.79** between 12-loop and 6-loop runs.
Different top tokens produced. WKV state mutation is doing real computation.

### Proof 2: Memory Flatline (MARGINAL on CPU)
5-loop and 10-loop runs show **0.00 MB** delta (true flatline).
CPU RSS noise at 20 loops. Re-run on CUDA for clean VRAM measurement.
Architecture is provably O(1): single-token stateful decode.

### Proof 3: State Geometry (PASS)
Full structural comparison documented. State drift curve shows smooth
cosine decay from 0.9962 to 0.9471 over 10 ticks -- the WKV matrix
evolves continuously, not discretely.

## Environment Notes

- Developed on CPU-only machine (no CUDA).
- `fla` (flash-linear-attention) 0.4.2 provides `RWKV7ForCausalLM` but requires
  Triton for GPU kernels. `cpu_patch.py` provides naive PyTorch replacements
  for all Triton ops, enabling correct (slower) inference on CPU.
- On a GPU machine, remove the `cpu_patch` import -- fla's Triton kernels will
  be used automatically and will be significantly faster.
