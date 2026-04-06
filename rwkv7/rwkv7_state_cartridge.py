"""
RWKV-7 State Cartridge
======================
Structured state persistence for RWKV-7 latent loop sessions.

Unlike Mamba session_memory.py (which saves h_t as opaque blobs),
RWKV-7 WKV state is a per-layer [n_heads, d_head, d_head] matrix --
a key-value memory whose geometry carries semantic content.

This means cartridges can be:
- Inspected: what information is encoded in each layer's KV matrix
- Fingerprinted: cosine similarity between cartridges identifies semantic drift
- Composed: weighted sum of two cartridges blends two reasoning contexts

State structure per layer (from Phase 1 inspection):
  {
      'recurrent_state': [B, n_heads, d_head, d_head],  # WKV KV matrix
      'conv_state':      [B, 1, hidden_size],             # token shift cache
      'ffn_state':       [B, 1, hidden_size],             # FFN shift cache
  }
"""

import torch
import time
from pathlib import Path


def _extract_state_list(cache) -> list[dict]:
    """Extract serializable state dicts from an fla Cache object."""
    states = []
    for i in range(len(cache)):
        layer = cache[i]
        if isinstance(layer, dict):
            states.append({
                k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
                for k, v in layer.items()
            })
        else:
            states.append(layer)
    return states


def save_cartridge(state, metadata: dict, path: str) -> float:
    """
    Serialize RWKV-7 WKV state to disk with metadata.

    Args:
        state: fla Cache object or list of per-layer state dicts
        metadata: dict with context info (prompt, n_loops, etc.)
        path: output file path

    Returns:
        size in KB
    """
    if hasattr(state, '__len__') and hasattr(state, '__getitem__'):
        # Could be a Cache object or already a list
        try:
            _ = state[0]
            if isinstance(state[0], dict):
                state_list = [
                    {k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
                     for k, v in layer.items()}
                    for layer in state
                ]
            else:
                state_list = _extract_state_list(state)
        except (TypeError, KeyError):
            state_list = _extract_state_list(state)
    else:
        state_list = _extract_state_list(state)

    cartridge = {
        "state": state_list,
        "metadata": metadata,
        "saved_at": time.time(),
        "architecture": "rwkv7",
        "n_layers": len(state_list),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cartridge, path)
    size_kb = Path(path).stat().st_size / 1024
    print(f"[cartridge] saved {size_kb:.1f} KB -> {path}")
    return size_kb


def load_cartridge(path: str, device: str = "cpu") -> tuple:
    """
    Load RWKV-7 state cartridge and return (state_list, metadata).

    Args:
        path: path to .pt cartridge file
        device: target device for tensors

    Returns:
        (state_list, metadata) where state_list is a list of per-layer dicts
    """
    cartridge = torch.load(path, map_location=device, weights_only=False)
    age_h = (time.time() - cartridge["saved_at"]) / 3600
    n_layers = cartridge.get("n_layers", len(cartridge["state"]))
    print(f"[cartridge] loaded {cartridge['architecture']} state "
          f"from {age_h:.1f}h ago -- {n_layers} layers")

    state_list = []
    for layer in cartridge["state"]:
        if isinstance(layer, dict):
            state_list.append({
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in layer.items()
            })
        else:
            state_list.append(layer)

    return state_list, cartridge["metadata"]


def fingerprint(state) -> torch.Tensor:
    """
    Compute a semantic fingerprint of the WKV state.

    Concatenates the mean of each layer's recurrent_state matrix into a
    single vector. Use cosine similarity between fingerprints to detect
    state drift across loop iterations or between sessions.

    Args:
        state: fla Cache object or list of per-layer state dicts

    Returns:
        1-D tensor fingerprint
    """
    layer_means = []
    for i in range(len(state)):
        layer = state[i]
        if isinstance(layer, dict):
            rs = layer.get('recurrent_state')
            if rs is not None and isinstance(rs, torch.Tensor):
                # Mean over heads and spatial dims -> scalar per layer
                layer_means.append(rs.float().mean(dim=[-2, -1]).flatten())
                continue
        # Fallback: treat as tensor directly
        if isinstance(layer, torch.Tensor):
            layer_means.append(layer.float().mean(dim=[-2, -1]).flatten())
        elif isinstance(layer, (list, tuple)):
            for t in layer:
                if isinstance(t, torch.Tensor):
                    layer_means.append(t.float().mean(dim=[-2, -1]).flatten())
                    break
    return torch.cat(layer_means)


def compose(state_a, state_b, alpha: float = 0.5) -> list:
    """
    Blend two RWKV-7 states: state_c = alpha * state_a + (1-alpha) * state_b.

    Enables context composition -- mix two reasoning traces before continuing.
    This is structurally valid because the WKV state is a linear key-value matrix:
    blending two states is equivalent to blending two sets of accumulated KV pairs.

    Args:
        state_a, state_b: lists of per-layer state dicts
        alpha: blending weight (0=all B, 1=all A)

    Returns:
        Composed state as list of per-layer dicts
    """
    composed = []
    for sa, sb in zip(state_a, state_b):
        if isinstance(sa, dict) and isinstance(sb, dict):
            layer = {}
            for key in sa:
                va, vb = sa[key], sb[key]
                if isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor):
                    layer[key] = alpha * va.float() + (1 - alpha) * vb.float()
                    layer[key] = layer[key].to(va.dtype)
                else:
                    layer[key] = va  # non-tensor, keep from A
            composed.append(layer)
        elif isinstance(sa, torch.Tensor) and isinstance(sb, torch.Tensor):
            composed.append(alpha * sa + (1 - alpha) * sb)
        else:
            composed.append(sa)
    return composed


def state_summary(state, label: str = "state") -> str:
    """Print a human-readable summary of the state."""
    lines = [f"[{label}]"]
    total_bytes = 0
    for i in range(len(state)):
        layer = state[i]
        if isinstance(layer, dict):
            rs = layer.get('recurrent_state')
            if rs is not None and isinstance(rs, torch.Tensor):
                norm = rs.float().norm().item()
                mean = rs.float().mean().item()
                lines.append(f"  L{i}: recurrent norm={norm:.2f} mean={mean:.6f}")
                total_bytes += rs.numel() * rs.element_size()
    lines.append(f"  Total: {total_bytes / 1024:.1f} KB across {len(state)} layers")
    result = "\n".join(lines)
    print(result)
    return result


if __name__ == "__main__":
    # Smoke test with dummy state
    print("=== State Cartridge Smoke Test ===\n")

    n_layers = 12
    n_heads = 12
    d_head = 64
    hidden_size = 768

    # Create dummy state matching real RWKV-7 structure
    dummy_state = []
    for _ in range(n_layers):
        dummy_state.append({
            'recurrent_state': torch.randn(1, n_heads, d_head, d_head),
            'attn_state': None,
            'conv_state': torch.randn(1, 1, hidden_size),
            'ffn_state': torch.randn(1, 1, hidden_size),
        })

    # Test save
    path = "/tmp/test_rwkv7_cartridge.pt"
    size = save_cartridge(dummy_state, {'prompt': 'test', 'n_loops': 10}, path)
    print(f"  Save: {size:.1f} KB")

    # Test load
    loaded_state, meta = load_cartridge(path, device='cpu')
    print(f"  Load: {len(loaded_state)} layers, meta={meta}")

    # Verify round-trip
    for i in range(n_layers):
        diff = (dummy_state[i]['recurrent_state'] - loaded_state[i]['recurrent_state']).abs().max()
        assert diff < 1e-6, f"Layer {i} round-trip error: {diff}"
    print("  Round-trip: PASS")

    # Test fingerprint
    fp = fingerprint(dummy_state)
    print(f"  Fingerprint: shape={fp.shape}")

    # Test compose
    state_b = []
    for _ in range(n_layers):
        state_b.append({
            'recurrent_state': torch.randn(1, n_heads, d_head, d_head),
            'attn_state': None,
            'conv_state': torch.randn(1, 1, hidden_size),
            'ffn_state': torch.randn(1, 1, hidden_size),
        })

    composed = compose(dummy_state, state_b, alpha=0.7)
    # Verify composition is correct
    expected = 0.7 * dummy_state[0]['recurrent_state'].float() + 0.3 * state_b[0]['recurrent_state'].float()
    diff = (composed[0]['recurrent_state'].float() - expected).abs().max()
    assert diff < 1e-5, f"Compose error: {diff}"
    print("  Compose: PASS")

    # Test fingerprint similarity
    fp_a = fingerprint(dummy_state)
    fp_b = fingerprint(state_b)
    cos_sim = torch.nn.functional.cosine_similarity(fp_a.unsqueeze(0), fp_b.unsqueeze(0)).item()
    print(f"  Cosine similarity (random vs random): {cos_sim:.4f}")

    fp_self = fingerprint(dummy_state)
    cos_self = torch.nn.functional.cosine_similarity(fp_a.unsqueeze(0), fp_self.unsqueeze(0)).item()
    print(f"  Cosine similarity (self vs self): {cos_self:.4f}")

    # Test state summary
    print()
    state_summary(dummy_state, "dummy")

    print("\n=== ALL SMOKE TESTS PASSED ===")
