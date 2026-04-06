# Phase 0/1 Findings: RWKV-7 State Inspection

## Phase 14 Inner Loop Shape (Mamba reference)

```
embed once (model.backbone.embedding) 
→ first full pass through all MambaBlock layers
→ inner loop N times:
    - re-run all backbone layers on hidden_states
    - ROM re-injection every ROMI_PERIOD=5 ticks (add pooled prompt embedding)
    - HaltingHead reads hidden_states, outputs P(halt)
    - inference mode: break if P(halt) > 0.70
→ final norm + LM head fires ONCE
```

### HaltingHead (Mamba)
- Input: hidden_state (B, seq_len, d_model=768)
- Pools across sequence dim → (B, d_model)
- 2-layer MLP: Linear(768, 192) → GELU → Linear(192, 1) → Sigmoid
- Output: P(halt) scalar per batch

### session_memory.py
- Serializes Mamba `conv_states` and `ssm_states` from `MambaCache`
- ~5 MB per session (Mamba-2.8B)
- Saves conversation history alongside state
- O(1) resume: state loaded directly, no re-prefill

## Environment

- Python 3.14.3 on Linux (no CUDA/GPU)
- `flash-linear-attention` (fla) 0.4.2 installed — provides `RWKV7ForCausalLM`
- `rwkv` 0.8.32 also installed (fallback)
- fla requires Triton for its kernels; wrote `rwkv7/cpu_patch.py` with naive PyTorch 
  replacements for all Triton ops (token_shift, chunk_rwkv7, fused_addcmul, etc.)
- **CPU patches work correctly** — model loads and runs on CPU

## RWKV-7 Model

- Model: `RWKV/RWKV7-Goose-World2.8-0.1B-HF` (0.1B parameters)
- Model class: `RWKV7ForCausalLM` (via fla)
- Tokenizer vocab: 65,531 tokens

### Config
| Parameter | Value |
|-----------|-------|
| hidden_size | 768 |
| num_hidden_layers | 12 |
| num_heads | 12 |
| head_dim | 64 |

## State Structure (past_key_values)

The HF `past_key_values` is an `fla.models.utils.Cache` object.
Indexed by layer, each layer returns a dict:

```python
state[layer_idx] = {
    'recurrent_state': Tensor[B, n_heads, d_head, d_head],  # [1, 12, 64, 64]
    'attn_state': None,
    'conv_state': Tensor[B, 1, hidden_size],                 # [1, 1, 768]
    'ffn_state': Tensor[B, 1, hidden_size],                  # [1, 1, 768]
}
```

### Per-layer state sizes (0.1B model, float32)
| Component | Shape | Bytes |
|-----------|-------|-------|
| recurrent_state | [1, 12, 64, 64] | 196,608 |
| conv_state | [1, 1, 768] | 3,072 |
| ffn_state | [1, 1, 768] | 3,072 |

**Total: 2,376 KB (2.3 MB) across 12 layers**

Compare: Mamba-2.8B session_memory.py reports ~5 MB.

## Key Difference from Mamba

The `recurrent_state` shape `[B, n_heads, d_head, d_head]` is a **structured key-value 
matrix** per layer. This is NOT an opaque blob like Mamba's h_t:
- Each head has a 64x64 matrix = explicit KV memory
- The matrix is updated via: `state = state * exp(w) + k ⊗ v + bonus_term`
- This geometric structure makes the state interpretable and composable

## Stateful Single-Token Decode

**CONFIRMED WORKING:**
```python
out = model(single_token_id, past_key_values=state, use_cache=True, output_hidden_states=True)
# Returns: logits [1, 1, 65536], new state, hidden_states [1, 1, 768]
```

State evolves across ticks:
- Tick 1 drift (L1 mean): 0.003477
- Tick 2 drift (L1 mean): 0.003219
- State changes are consistent and non-trivial

**This means true O(1) per-tick compute in the latent loop is possible.**
Each spacer token updates only the recurrent state, not a growing sequence.

## Spacer Token
- Token `=` maps to ID 62

## Phase 1 Gate: PASS
Single-token stateful decode works through the HuggingFace interface via fla.
The latent loop engine can use `model(spacer, past_key_values=state, use_cache=True)`
for O(1) per-tick computation.
