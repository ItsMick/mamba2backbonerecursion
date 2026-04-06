"""
RWKV-7 State Inspector
======================
Load a small RWKV-7 model via fla/HF and inspect the shape of
past_key_values (WKV state) to determine how stateful decode works.
"""

import torch
import sys
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Only apply CPU patches when there's no GPU — on CUDA, fla's native
# Triton kernels are faster and correct.
if DEVICE == "cpu":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rwkv7.cpu_patch import apply_cpu_patches
    apply_cpu_patches()
MODEL_ID = "RWKV/RWKV7-Goose-World2.8-0.1B-HF"
FALLBACK_ID = "RWKV/RWKV7-Goose-World3-1.5B-HF"


def inspect_model(model_id: str):
    print(f"\n{'='*60}")
    print(f"  RWKV-7 State Inspector - {model_id}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}\n")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[1/5] Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print(f"  Vocab size: {tok.vocab_size}")

    print("[2/5] Loading model...")
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True,
    )
    if DEVICE == "cpu":
        model = model.to(DEVICE)
    model.config.use_cache = True
    model.eval()
    print(f"  Model class: {type(model).__name__}")

    config = model.config
    for attr in ['hidden_size', 'num_hidden_layers', 'head_dim', 'num_heads']:
        if hasattr(config, attr):
            print(f"  config.{attr} = {getattr(config, attr)}")

    print("\n[3/5] Forward pass with use_cache=True, output_hidden_states=True...")
    prompt = "The capital of France is"
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    print(f"  Prompt: '{prompt}' -> {ids.shape[1]} tokens")

    with torch.no_grad():
        out = model(ids, use_cache=True, output_hidden_states=True)

    print("\n[4/5] State (past_key_values) inspection:")
    state = out.past_key_values
    print(f"  Type: {type(state).__name__}")
    print(f"  Module: {type(state).__module__}")

    if state is None:
        print("  WARNING: past_key_values is None!")
        return

    total_bytes = 0

    # Enumerate all tensor-like attributes on the Cache object
    for attr_name in sorted(dir(state)):
        if attr_name.startswith('_'):
            continue
        try:
            val = getattr(state, attr_name)
        except Exception:
            continue
        if isinstance(val, torch.Tensor):
            print(f"  .{attr_name}: shape={val.shape} dtype={val.dtype}")
            total_bytes += val.numel() * val.element_size()
        elif isinstance(val, (list, tuple)) and len(val) > 0:
            if isinstance(val[0], torch.Tensor):
                print(f"  .{attr_name}: list of {len(val)} tensors")
                for i, t in enumerate(val):
                    print(f"    [{i}]: shape={t.shape} dtype={t.dtype}")
                    total_bytes += t.numel() * t.element_size()
                    if i >= 2:
                        remaining = sum(t2.numel() * t2.element_size() for t2 in val[3:])
                        total_bytes += remaining
                        print(f"    ... ({len(val) - 3} more layers)")
                        break
            elif isinstance(val[0], dict):
                print(f"  .{attr_name}: list of {len(val)} dicts")
                for i, d in enumerate(val):
                    for dk, dv in d.items():
                        if isinstance(dv, torch.Tensor):
                            print(f"    [{i}].{dk}: shape={dv.shape} dtype={dv.dtype}")
                            total_bytes += dv.numel() * dv.element_size()
                    if i >= 2:
                        for d2 in val[3:]:
                            for dv in d2.values():
                                if isinstance(dv, torch.Tensor):
                                    total_bytes += dv.numel() * dv.element_size()
                        print(f"    ... ({len(val) - 3} more layers)")
                        break

    print(f"\n  Total state size: {total_bytes} bytes ({total_bytes/1024:.1f} KB)")

    # Hidden states
    print(f"\n  Hidden states: {len(out.hidden_states)} outputs")
    print(f"  Final hidden shape: {out.hidden_states[-1].shape}")

    # Test single-token stateful decode
    print("\n[5/5] Testing single-token stateful decode...")
    single_id = ids[:, -1:].clone()
    print(f"  Single token: {single_id}")

    try:
        with torch.no_grad():
            out2 = model(single_id, past_key_values=state, use_cache=True,
                         output_hidden_states=True)
        print("  SUCCESS: Single-token stateful decode works!")
        print(f"  Output logits shape: {out2.logits.shape}")
        state2 = out2.past_key_values
        print(f"  New state type: {type(state2).__name__}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*60}")
    print("  Inspection complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    try:
        inspect_model(MODEL_ID)
    except Exception as e:
        print(f"\nFailed with {MODEL_ID}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nTrying fallback: {FALLBACK_ID}")
        try:
            inspect_model(FALLBACK_ID)
        except Exception as e2:
            print(f"\nFallback also failed: {type(e2).__name__}: {e2}")
            traceback.print_exc()
