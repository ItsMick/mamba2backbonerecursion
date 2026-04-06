"""
RWKV-7 Latent Loop Engine
=========================
Port of Phil's Phase 14 inner-loop bypass to RWKV-7.

Architecture difference vs Mamba:
- Mamba h_t: opaque [d_state, d_model] per layer - hard to interpret
- RWKV-7 WKV state: [n_heads, d_head, d_head] per layer - explicit KV matrix
  This means latent computation is structured: each tick writes to a
  key-value memory that subsequent ticks can read with full geometric clarity.

The latent loop:
1. Prefill: encode prompt, obtain initial WKV state per layer
2. Inner loop: feed spacer token through WKV update only (no LM head)
   - State evolves: state = state * decay + k (x) v
   - ROM re-injection: add pooled prompt hidden into residual every 5 ticks
3. Halt: HaltingHead reads final layer WKV state, signals stop
4. Decode: LM head fires ONCE on final hidden state

State structure per layer (from Phase 1 inspection):
  state[i] = {
      'recurrent_state': [B, n_heads=12, d_head=64, d_head=64],  # WKV matrix
      'conv_state':      [B, 1, hidden_size=768],                  # token shift
      'ffn_state':       [B, 1, hidden_size=768],                  # FFN shift
  }
  Total: ~2.3 MB across 12 layers (0.1B model, float32)
"""

import torch
import torch.nn as nn
import json
import random
import os
import time
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HALT_THRESHOLD = 0.70
MIN_LOOPS = 1
MAX_LOOPS = 20
ROMI_PERIOD = 5
SPACER_TOKEN = "="

# Model config for RWKV7-Goose-World2.8-0.1B-HF
DEFAULT_MODEL_ID = "RWKV/RWKV7-Goose-World2.8-0.1B-HF"
D_MODEL = 768
N_HEADS = 12
D_HEAD = 64
N_LAYERS = 12


class HaltingHead(nn.Module):
    """
    Same architecture as Phil's HaltingHead but reads from RWKV-7 WKV state.

    Input: state embedding projected from the final-layer WKV recurrent_state
           [n_heads, d_head, d_head] -> d_model via StateProjector.
    Output: P(halt) in [0, 1].
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.probe = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, state_embedding: torch.Tensor) -> torch.Tensor:
        # state_embedding: (B, d_model)
        return self.probe(state_embedding).squeeze(-1)


class StateProjector(nn.Module):
    """
    Projects RWKV-7 WKV recurrent_state [B, n_heads, d_head, d_head] -> [B, d_model]
    so HaltingHead can read from it.

    This is architecturally meaningful: the WKV matrix is the model's
    structured memory, and the projector learns to read it.
    """

    def __init__(self, n_heads: int, d_head: int, d_model: int):
        super().__init__()
        state_dim = n_heads * d_head * d_head
        self.proj = nn.Linear(state_dim, d_model)

    def forward(self, wkv_state: torch.Tensor) -> torch.Tensor:
        # wkv_state: (B, n_heads, d_head, d_head)
        B = wkv_state.shape[0]
        flat = wkv_state.reshape(B, -1).float()
        return self.proj(flat)


def extract_recurrent_state(cache, layer_idx: int) -> torch.Tensor:
    """Extract the WKV recurrent_state from an fla Cache at a given layer."""
    layer_state = cache[layer_idx]
    if isinstance(layer_state, dict):
        return layer_state['recurrent_state']
    raise ValueError(f"Unexpected cache layer format: {type(layer_state)}")


def load_model(model_id: str = DEFAULT_MODEL_ID):
    """Load RWKV-7 model, tokenizer, and configure for latent loop."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Apply CPU patches if needed
    if DEVICE == "cpu":
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from rwkv7.cpu_patch import apply_cpu_patches
        apply_cpu_patches()

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    dtype = torch.float32 if DEVICE == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True,
    )
    if DEVICE == "cpu":
        model = model.to(DEVICE)
    model.config.use_cache = True
    model.eval()

    spacer_id = tok.encode(SPACER_TOKEN, add_special_tokens=False)[0]

    return model, tok, spacer_id


def run_rwkv7_inner_loop(
    model,
    halting_head: HaltingHead,
    state_projector: StateProjector,
    prompt_ids: torch.Tensor,
    spacer_id: int,
    training_mode: bool = True,
    n_true: int = 8,
) -> tuple:
    """
    RWKV-7 latent loop forward pass.

    Unlike Mamba Phase 14 which re-runs all MambaBlock layers on the full
    hidden state each tick, RWKV-7 does true O(1) per-tick compute:
    feed one spacer token through the stateful WKV update, which touches
    only the fixed-size recurrent state -- not a growing sequence.

    Args:
        model: RWKV7ForCausalLM with pretrained weights
        halting_head: HaltingHead binary classifier
        state_projector: StateProjector for WKV state -> d_model
        prompt_ids: (1, seq_len) input token IDs
        spacer_id: token ID for the spacer (=)
        training_mode: if True, loop exactly n_true times (teacher forcing)
        n_true: oracle loop count for teacher forcing

    Returns:
        logits: (1, 1, vocab_size) - LM head output from final tick
        n_loops: number of inner loop ticks executed
        halt_probs: list of P(halt) at each tick
        state: final WKV state (fla Cache object)
    """
    n_layers = model.config.num_hidden_layers

    # Phase 1: Prefill -- encode full prompt, obtain initial WKV state
    with torch.no_grad():
        prefill_out = model(
            prompt_ids,
            use_cache=True,
            output_hidden_states=True,
        )
        state = prefill_out.past_key_values
        last_hidden = prefill_out.hidden_states[-1]  # (1, seq_len, d_model)

    # ROM: pool prompt embedding for re-injection (frozen, no gradient)
    rom = last_hidden.mean(dim=1, keepdim=True).detach()  # (1, 1, d_model)

    spacer = torch.tensor([[spacer_id]], device=prompt_ids.device)
    halt_probs = []
    n_loops = 0
    loop_limit = n_true if training_mode else MAX_LOOPS
    hidden = last_hidden[:, -1:, :]  # (1, 1, d_model) last position

    while n_loops < loop_limit:
        n_loops += 1

        # O(1) single-token stateful decode: feed spacer through WKV update
        # This is THE key difference from Mamba Phase 14's full layer re-scan
        tick_out = model(
            spacer,
            past_key_values=state,
            use_cache=True,
            output_hidden_states=True,
        )
        state = tick_out.past_key_values
        hidden = tick_out.hidden_states[-1]  # (1, 1, d_model)

        # ROM re-injection every ROMI_PERIOD ticks
        # Add pooled prompt embedding to hidden state as residual.
        # In RWKV-7, we inject into the token representation rather than
        # into the WKV state directly -- this preserves WKV geometry
        # while still anchoring the computation to the original prompt.
        if n_loops % ROMI_PERIOD == 0:
            hidden = hidden + rom.to(hidden.dtype)

        # HaltingHead reads from final-layer WKV recurrent_state
        final_recurrent = extract_recurrent_state(state, n_layers - 1)
        state_emb = state_projector(final_recurrent.float())  # (1, d_model)
        p_halt = halting_head(state_emb).item()
        halt_probs.append(p_halt)

        if not training_mode and p_halt > HALT_THRESHOLD and n_loops >= MIN_LOOPS:
            break

    # LM head fires ONCE on final hidden state
    if hasattr(model, 'lm_head'):
        logits = model.lm_head(hidden.to(next(model.lm_head.parameters()).dtype))
    else:
        logits = tick_out.logits

    return logits, n_loops, halt_probs, state


def demo_inference(prompt: str = "[LOGIC] What is 7 + 5?", verbose: bool = True):
    """
    Demonstrate the RWKV-7 latent loop on a single prompt.

    This is the RWKV-7 equivalent of running Phil's Phase 14 in inference mode.
    """
    if verbose:
        print("=" * 62)
        print("  RWKV-7 LATENT LOOP ENGINE -- INFERENCE DEMO")
        print("=" * 62)

    model, tok, spacer_id = load_model()

    # Initialize halting components (untrained -- random weights)
    halting_head = HaltingHead(d_model=D_MODEL).to(DEVICE)
    state_projector = StateProjector(N_HEADS, D_HEAD, D_MODEL).to(DEVICE)

    prompt_ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    if verbose:
        print(f"\nPrompt: '{prompt}' ({prompt_ids.shape[1]} tokens)")

    t0 = time.time()
    logits, n_loops, halt_probs, state = run_rwkv7_inner_loop(
        model, halting_head, state_projector,
        prompt_ids, spacer_id,
        training_mode=False,
    )
    elapsed = time.time() - t0

    # Decode top prediction
    top_token = logits[0, -1].argmax().item()
    answer = tok.decode([top_token])

    if verbose:
        print(f"Loops: {n_loops}")
        print(f"Halt probs: {[f'{p:.3f}' for p in halt_probs]}")
        print(f"Top token: '{answer}' (id={top_token})")
        print(f"Elapsed: {elapsed:.2f}s")

        # State summary
        final_recurrent = extract_recurrent_state(state, N_LAYERS - 1)
        print(f"\nFinal state (layer {N_LAYERS-1}):")
        print(f"  recurrent_state: {final_recurrent.shape}")
        print(f"  L2 norm: {final_recurrent.float().norm().item():.2f}")
        print(f"  Mean: {final_recurrent.float().mean().item():.6f}")

    return logits, n_loops, halt_probs, state


if __name__ == "__main__":
    demo_inference()
