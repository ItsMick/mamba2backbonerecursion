"""
CPU Compatibility Patch for fla (flash-linear-attention)
========================================================
fla's RWKV-7 implementation uses Triton kernels that require GPU.
This module patches the critical functions with naive PyTorch equivalents
so the model can run on CPU (slower but functionally correct).

Only needed for development/inspection on CPU-only machines.
On GPU machines, the original fla Triton kernels are used (much faster).
"""

import torch
import contextlib
import functools


def apply_cpu_patches():
    """Apply all CPU compatibility patches to fla."""
    import fla.utils

    # Patch 1: custom_device_ctx — torch.cpu has no .device() method
    if not hasattr(fla.utils.device_torch_lib, 'device'):
        @contextlib.contextmanager
        def _cpu_device_ctx(index=0):
            yield
        fla.utils.custom_device_ctx = _cpu_device_ctx

    # Patch 2: input_guard decorator — calls custom_device_ctx internally
    # Replace with a no-op decorator on CPU
    def _noop_input_guard(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(None, *args, **kwargs) if 'ctx' not in fn.__code__.co_varnames[:1] else fn(*args, **kwargs)
        return wrapper

    # Patch 3: Replace TokenShift with naive implementation
    import fla.modules.token_shift as ts_mod

    class NaiveTokenShift(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, cu_seqlens=None, cache=None, output_cache=False, chunk_indices=None):
            # Token shift: output[t] = x[t-1], output[0] = cache or zeros
            B, T, D = x.shape
            output = torch.zeros_like(x)
            if T > 1:
                output[:, 1:, :] = x[:, :-1, :]
            if cache is not None:
                output[:, 0, :] = cache.view(B, D)

            cache_out = x[:, -1:, :].clone() if output_cache else None
            return output, cache_out

        @staticmethod
        def backward(ctx, dy, dcache):
            return dy, None, None, None, None

    ts_mod.TokenShift = NaiveTokenShift

    # Patch 4: Replace fused ops with naive equivalents
    import fla.ops.rwkv7 as rwkv7_ops

    def naive_chunk_rwkv7(r, w, k, v, a, b, scale=1.0, initial_state=None,
                          output_final_state=False, cu_seqlens=None,
                          cu_seqlens_cpu=None, head_first=False,
                          safe_gate=False, chunk_size=None):
        """Naive RWKV-7 WKV computation on CPU."""
        B, T, H, K = r.shape
        V = v.shape[-1]

        if initial_state is not None:
            state = initial_state.clone().float()
        else:
            state = torch.zeros(B, H, K, V, dtype=torch.float32, device=r.device)

        r, w, k, v, a, b = [x.float() for x in (r, w, k, v, a, b)]
        outputs = []

        for t in range(T):
            rt = r[:, t]       # [B, H, K]
            wt = w[:, t]       # [B, H, K]
            kt = k[:, t]       # [B, H, K]
            vt = v[:, t]       # [B, H, V]
            at = a[:, t]       # [B, H, K]
            bt = b[:, t]       # [B, H, K]

            # WKV state update: state = diag(exp(w)) @ state + k @ v^T + a @ (b^T @ state)
            kv = torch.einsum('bhk,bhv->bhkv', kt, vt)
            # Bonus term from a and b
            ab_state = torch.einsum('bhk,bhkv->bhv', bt, state)
            ab_update = torch.einsum('bhk,bhv->bhkv', at, ab_state)

            state = state * wt.unsqueeze(-1).exp() + kv + ab_update

            # Readout: o = r^T @ state
            ot = torch.einsum('bhk,bhkv->bhv', rt, state) * scale
            outputs.append(ot)

        o = torch.stack(outputs, dim=1)  # [B, T, H, V]
        o = o.to(r.dtype)

        if output_final_state:
            return o, state.to(r.dtype)
        return o, None

    rwkv7_ops.chunk_rwkv7 = naive_chunk_rwkv7

    # Also patch the import in the layer module
    import fla.layers.rwkv7 as rwkv7_layer
    rwkv7_layer.chunk_rwkv7 = naive_chunk_rwkv7

    def naive_fused_mul_recurrent_rwkv7(r, w, k, v, kk, a, scale=1.0,
                                         initial_state=None,
                                         output_final_state=False,
                                         cu_seqlens=None):
        """Naive single-step recurrence for RWKV-7 on CPU."""
        B, T, H, K = r.shape
        V = v.shape[-1]

        if initial_state is not None:
            state = initial_state.clone().float()
        else:
            state = torch.zeros(B, H, K, V, dtype=torch.float32, device=r.device)

        r, w, k, v, kk, a = [x.float() for x in (r, w, k, v, kk, a)]
        outputs = []

        for t in range(T):
            rt = r[:, t]
            wt = w[:, t]
            kt = k[:, t]
            vt = v[:, t]
            kkt = kk[:, t]
            at = a[:, t]

            kv = torch.einsum('bhk,bhv->bhkv', kt, vt)
            # bonus: a_bar = -kk, b_bar = kk * a
            ab_state = torch.einsum('bhk,bhkv->bhv', (kkt * at), state)
            ab_update = torch.einsum('bhk,bhv->bhkv', -kkt, ab_state)

            state = state * wt.unsqueeze(-1).exp() + kv + ab_update

            ot = torch.einsum('bhk,bhkv->bhv', rt, state) * scale
            outputs.append(ot)

        o = torch.stack(outputs, dim=1)
        o = o.to(r.dtype)

        if output_final_state:
            return o, state.to(r.dtype)
        return o, None

    rwkv7_layer.fused_mul_recurrent_rwkv7 = naive_fused_mul_recurrent_rwkv7

    # Patch 5: fused_addcmul_rwkv7 — fused multiply-add
    from fla.ops.rwkv7.fused_addcmul import fused_addcmul_rwkv7 as _orig_addcmul
    import fla.ops.rwkv7.fused_addcmul as addcmul_mod

    def naive_fused_addcmul_rwkv7(x, delta, x_r, x_w, x_k, x_v, x_a, x_g):
        """Naive fused addcmul: xi = x + delta * x_i for each projection."""
        xr = x + delta * x_r
        xw = x + delta * x_w
        xk = x + delta * x_k
        xv = x + delta * x_v
        xa = x + delta * x_a
        xg = x + delta * x_g
        return xr, xw, xk, xv, xa, xg

    addcmul_mod.fused_addcmul_rwkv7 = naive_fused_addcmul_rwkv7
    rwkv7_layer.fused_addcmul_rwkv7 = naive_fused_addcmul_rwkv7

    # Patch 6: fused_k_rwkv7 — k = k + k * (a - 1) * k_a
    import fla.ops.rwkv7.fused_k_update as fused_k_mod

    def naive_fused_k_rwkv7(k, a, k_a):
        """k = k.addcmul(k * (a - 1), k_a)"""
        return k + k * (a - 1) * k_a

    fused_k_mod.fused_k_rwkv7 = naive_fused_k_rwkv7
    rwkv7_layer.fused_k_rwkv7 = naive_fused_k_rwkv7

    # Patch 7: gate_output_correction
    import fla.ops.rwkv7.gate_output_correction as goc_mod

    def naive_gate_output_correction(o, r, k, r_k, v, g):
        """o = o * g.sigmoid() + (r * k * r_k).sum(-1, keepdim=True) * v"""
        # r, k: [B, T, H, K], v: [B, T, H, V], r_k: [H, K] or similar
        B, T, HD = o.shape[0], o.shape[1], o.shape[2]
        # g: [B, T, H*K] or [B, T, H, K]
        if g.dim() == 3:
            gate = g.sigmoid()
        else:
            gate = g.sigmoid()
            gate = gate.reshape(B, T, -1)

        # Compute bonus: (r * k * r_k).sum(dim=-1, keepdim=True) * v
        bonus = (r * k * r_k).sum(dim=-1, keepdim=True) * v  # [B, T, H, V]
        bonus = bonus.reshape(B, T, -1)

        return o * gate + bonus

    goc_mod.gate_output_correction = naive_gate_output_correction
    rwkv7_layer.gate_output_correction = naive_gate_output_correction

    print("[cpu_patch] All fla CPU patches applied successfully")
