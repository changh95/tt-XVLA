# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TT-NN port of DaViT's PreNorm + WindowAttention block.

WindowAttention on the upstream side:

    1. View x: [B, T, C] -> [B, H, W, C], pad H/W to multiples of window
    2. window_partition: 6D reshape+permute -> [num_w * B, ws, ws, C]
    3. Flatten windows -> [num_w * B, ws*ws, C]
    4. Standard multi-head self-attention inside each window
    5. window_reverse: 6D reshape+permute -> [B, H_padded, W_padded, C]
    6. Unpad to [B, H, W, C] -> view [B, T, C]

The window partition / reverse uses 6D ops which we keep on torch. The
matmul-heavy interior (LayerNorm + qkv linear + attention + proj linear)
moves to the device. PreNorm's residual add stays on torch (cheap, and
the post-attention output is already a torch tensor at that point).

Two implementations: ``TTNNDaViTPreNormWindowAttn`` (legacy, default, untouched: 9 launches
on the host-partitioned windows, host tilize/untilize) and ``TTNNDaViTPreNormWindowAttnFused``
(``TT_FUSED=1``): ROW_MAJOR upload + device tilize, LN, qkv, split(k untransposed) +
``scaled_dot_product_attention`` on ``[BW, heads, 144, 32]`` (q/k chunk 160; the kernel masks
the 16 padded keys) + concat, proj, ``untilize_with_unpadding``, ROW_MAJOR readback = 8
launches; pad / window_partition / window_reverse / unpad / residual stay on torch unless
``TT_FUSED_WINDOWS_ON_DEVICE=1`` moves them on device as rank-4 ``reshape``/``permute``/
``pad``/``slice`` on ROW_MAJOR tensors (the 6-D permute of ``window_partition`` is exactly one
4-D ``permute(0, 2, 1, 3)`` of the free view ``[B*nh, ws, nw, ws*C]``, see
``fused.window_partition_4d``), so the per-block upload is the ``[B, T, C]`` tokens (1.2 MB
at stage 2) instead of the ``[BW, 144, C]`` windows (3.5 MB).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


def _fused_module():
    """Import ``tt/fused.py`` as a package sibling or by path (see ``tt/policy.py``)."""
    import importlib
    import importlib.util
    import sys

    pkg = __name__.rpartition(".")[0]
    if pkg:
        try:
            return importlib.import_module(pkg + ".fused")
        except ImportError:
            pass
    name = "xvla_ttnn_fused"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(Path(__file__).resolve().with_name("fused.py")))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _bf16_tile(ttnn_mod, t: torch.Tensor, device):
    return ttnn_mod.from_torch(
        t.to(torch.bfloat16).contiguous(),
        dtype=ttnn_mod.bfloat16, layout=ttnn_mod.TILE_LAYOUT, device=device,
    )


def _bfp8_tile(ttnn_mod, t: torch.Tensor, device):
    return ttnn_mod.from_torch(
        t.to(torch.bfloat16).contiguous(),
        dtype=ttnn_mod.bfloat8_b, layout=ttnn_mod.TILE_LAYOUT, device=device,
    )


class TTNNDaViTPreNormWindowAttn(nn.Module):
    """`PreNorm(LN, WindowAttention)` with the LN+attention+proj on chip."""

    def __init__(self, prenorm_torch: nn.Module, device) -> None:
        super().__init__()
        import ttnn

        self._ttnn = ttnn
        self.device = device

        norm = prenorm_torch.norm
        attn = prenorm_torch.fn  # WindowAttention
        self.window_size = int(attn.window_size)
        self.num_heads = int(attn.num_heads)
        self.dim = int(attn.qkv.in_features)
        self.head_dim = self.dim // self.num_heads
        self.head_dim_inv_sqrt = float(self.head_dim) ** -0.5

        self.ln_w = _bf16_tile(ttnn, norm.weight.detach(), device)
        self.ln_b = _bf16_tile(ttnn, norm.bias.detach(), device)
        self.qkv_w = _bf16_tile(ttnn, attn.qkv.weight.detach().t().contiguous(), device)
        self.qkv_b = _bf16_tile(
            ttnn,
            (attn.qkv.bias.detach() if attn.qkv.bias is not None
             else torch.zeros(3 * self.dim)),
            device,
        )
        self.proj_w = _bf16_tile(ttnn, attn.proj.weight.detach().t().contiguous(), device)
        self.proj_b = _bf16_tile(
            ttnn,
            (attn.proj.bias.detach() if attn.proj.bias is not None
             else torch.zeros(self.dim)),
            device,
        )

    # --- on-device interior: attention over windows -----------------------

    def _on_chip_attn(self, windows_torch: torch.Tensor) -> torch.Tensor:
        """`windows_torch` is [BW, ws*ws, C]. Returns same shape."""
        ttnn = self._ttnn
        BW, T, C = windows_torch.shape

        x_tt = _bf16_tile(ttnn, windows_torch, self.device)
        h = ttnn.layer_norm(x_tt, weight=self.ln_w, bias=self.ln_b)
        ttnn.deallocate(x_tt)
        qkv = ttnn.linear(h, self.qkv_w, bias=self.qkv_b)
        ttnn.deallocate(h)

        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, num_heads=self.num_heads
        )
        ttnn.deallocate(qkv)
        # k pre-transposed to [BW, H, head_dim, T]
        scores = ttnn.matmul(q, k)  # [BW, H, T, T]
        ttnn.deallocate(q); ttnn.deallocate(k)
        scores = ttnn.multiply(scores, self.head_dim_inv_sqrt)
        probs = ttnn.softmax(scores, dim=-1)
        ttnn.deallocate(scores)
        attn_out = ttnn.matmul(probs, v)
        ttnn.deallocate(probs); ttnn.deallocate(v)
        attn_out = ttnn.transformer.concatenate_heads(attn_out)  # [BW, T, C]

        out = ttnn.linear(attn_out, self.proj_w, bias=self.proj_b)
        ttnn.deallocate(attn_out)
        result = ttnn.to_torch(out).to(windows_torch.dtype)
        ttnn.deallocate(out)
        return result

    # --- forward -----------------------------------------------------------

    def forward(self, x: torch.Tensor, size):
        """`x`: [B, T, C], T = H*W. Returns (x_with_residual, size)."""
        H, W = size
        B, T, C = x.shape
        assert T == H * W
        ws = self.window_size

        residual = x

        # Pad to multiples of window_size
        x4 = x.view(B, H, W, C)
        pad_r = (ws - W % ws) % ws
        pad_b = (ws - H % ws) % ws
        x4p = F.pad(x4, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = x4p.shape[1], x4p.shape[2]

        # window_partition: [B, Hp/ws, ws, Wp/ws, ws, C] -> [B*nh*nw, ws, ws, C]
        nh, nw = Hp // ws, Wp // ws
        x6 = x4p.view(B, nh, ws, nw, ws, C).permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = x6.view(-1, ws * ws, C)  # [B*nh*nw, ws*ws, C]

        # On-chip self-attention over windows
        attn = self._on_chip_attn(windows)

        # window_reverse: [B*nh*nw, ws*ws, C] -> [B, Hp, Wp, C]
        attn = attn.view(B, nh, nw, ws, ws, C)
        attn = attn.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, C)
        # Unpad
        if pad_r or pad_b:
            attn = attn[:, :H, :W, :].contiguous()
        attn = attn.view(B, T, C)

        return residual + attn, size


class TTNNDaViTPreNormWindowAttnFused(nn.Module):
    """``TT_FUSED=1`` variant of ``TTNNDaViTPreNormWindowAttn`` (same ``forward(x, size)`` ->
    ``(x + attn, size)`` contract). Window shapes per stage (3 views, ws=12, 144 tokens/window):
    [75, 144, 256] h8 / [27, 144, 512] h16 / [12, 144, 1024] h32 / [3, 144, 2048] h64;
    head_dim 32 everywhere (SDPA/concat need it tile-aligned: ok)."""

    def __init__(self, prenorm_torch: nn.Module, device, kit) -> None:
        super().__init__()
        self._fused = _fused_module()
        self.kit = kit
        self.cfg = kit.cfg
        self._ttnn = kit.ttnn
        self.device = device
        norm = prenorm_torch.norm
        attn = prenorm_torch.fn  # WindowAttention
        self.window_size = int(attn.window_size)
        self.num_heads = int(attn.num_heads)
        self.dim = int(attn.qkv.in_features)
        self.head_dim = self.dim // self.num_heads
        assert self.head_dim % 32 == 0, "SDPA / concatenate_heads need a tile-aligned head_dim"
        self.head_dim_inv_sqrt = float(self.head_dim) ** -0.5

        def bias_or_zeros(lin):
            return lin.bias.detach() if lin.bias is not None else torch.zeros(lin.out_features)

        self.ln_w = kit.vec(norm.weight)
        self.ln_b = kit.vec(norm.bias)
        self.ln_eps = float(norm.eps)
        self.qkv_w = kit.weight(attn.qkv.weight.detach().t())
        self.qkv_b = kit.bias_row(bias_or_zeros(attn.qkv))
        self.proj_w = kit.weight(attn.proj.weight.detach().t())
        self.proj_b = kit.bias_row(bias_or_zeros(attn.proj))

    # --- on-device interior: LN + attention + proj on TILE windows ---------

    def _attn_tile(self, win_t, seq_len: int):
        """TILE ``[BW, T, C]`` -> TILE ``[BW, T, C]`` (proj output, no residual)."""
        kit = self.kit
        h = kit.layer_norm(win_t, self.ln_w, self.ln_b, self.ln_eps)
        qkv = kit.linear(h, self.qkv_w, self.qkv_b)
        kit.free(h)
        ctx = kit.attention(qkv, self.num_heads, seq_len, self.head_dim_inv_sqrt)
        kit.free(qkv)
        out = kit.linear(ctx, self.proj_w, self.proj_b)
        kit.free(ctx)
        return out

    def _on_chip_attn(self, windows_torch: torch.Tensor) -> torch.Tensor:
        """Host-partitioned windows ``[BW, ws*ws, C]`` -> attention output, same shape."""
        kit = self.kit
        BW, T, C = windows_torch.shape
        x_dev = kit.upload(windows_torch)
        x_t = kit.ingest(x_dev)
        out_t = self._attn_tile(x_t, T)
        kit.free(x_t, x_dev)
        out_dev = kit.egress(out_t, (BW, T, C))
        if out_dev is not out_t:
            kit.free(out_t)
        result = kit.readback(out_dev).to(windows_torch.dtype)
        kit.free(out_dev)
        return result

    # --- forward -----------------------------------------------------------

    def forward(self, x: torch.Tensor, size):
        if self.cfg.windows_on_device:
            return self._forward_windows_on_device(x, size)
        H, W = size
        B, T, C = x.shape
        assert T == H * W
        ws = self.window_size
        residual = x
        x4 = x.view(B, H, W, C)
        pad_b, pad_r, Hp, Wp = self._fused.window_pad(H, W, ws)
        x4p = F.pad(x4, (0, 0, 0, pad_r, 0, pad_b))
        nh, nw = Hp // ws, Wp // ws
        x6 = x4p.view(B, nh, ws, nw, ws, C).permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = x6.view(-1, ws * ws, C)
        attn = self._on_chip_attn(windows)
        attn = attn.view(B, nh, nw, ws, ws, C)
        attn = attn.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, C)
        if pad_r or pad_b:
            attn = attn[:, :H, :W, :].contiguous()
        attn = attn.view(B, T, C)
        return residual + attn, size

    def _forward_windows_on_device(self, x: torch.Tensor, size):
        """``TT_FUSED_WINDOWS_ON_DEVICE=1``: upload ``[B, T, C]`` ROW_MAJOR once; pad, partition
        (reshape/permute/reshape), tilize, attention, untilize, reverse, unpad (slice), residual
        add -- all on device; one ROW_MAJOR readback. The pad / partition / reverse / unpad are
        exact data movement (``fused.window_*_4d`` proven against the reference in
        tests/test_fused_host.py; every RM op accepted on the p150a 2026-09-13), but the residual
        add now happens on device in bf16 instead of torch: measured NOT bit-identical to the
        host path (max abs diff 3.9e-3 on the action chunk, PCC vs fp32 unchanged) -- a
        bf16-rounding lever, kept because it passed the gates and saved 8.6 ms."""
        ttnn = self._ttnn
        kit = self.kit
        H, W = size
        B, T, C = x.shape
        assert T == H * W
        ws = self.window_size
        pad_b, pad_r, Hp, Wp = self._fused.window_pad(H, W, ws)
        nh, nw = Hp // ws, Wp // ws
        BW = B * nh * nw
        tmp = []

        x_rm = kit.upload_rm(x)  # [B, T, C] ROW_MAJOR, kept for the residual
        x4 = ttnn.reshape(x_rm, (B, H, W, C))
        tmp.append(x4)
        if pad_b or pad_r:
            x4 = ttnn.pad(x4, padding=[(0, 0), (0, pad_b), (0, pad_r), (0, 0)], value=0.0)
            tmp.append(x4)
        v = ttnn.reshape(x4, (B * nh, ws, nw, ws * C))
        p = ttnn.permute(v, (0, 2, 1, 3))
        win = ttnn.reshape(p, (BW, ws * ws, C))
        tmp += [v, p, win]
        win_t = ttnn.tilize_with_zero_padding(win, use_multicore=True)
        out_t = self._attn_tile(win_t, ws * ws)
        kit.free(win_t)
        out_rm = ttnn.untilize_with_unpadding(out_t, output_tensor_end=[BW - 1, ws * ws - 1, C - 1])
        kit.free(out_t)
        r = ttnn.reshape(out_rm, (B * nh, nw, ws, ws * C))
        r = ttnn.permute(r, (0, 2, 1, 3))
        r4 = ttnn.reshape(r, (B, Hp, Wp, C))
        tmp += [out_rm, r, r4]
        if pad_b or pad_r:
            r4 = ttnn.slice(r4, [0, 0, 0, 0], [B, H, W, C])
            tmp.append(r4)
        r3 = ttnn.reshape(r4, (B, T, C))
        tmp.append(r3)
        y = ttnn.add(x_rm, r3)  # ROW_MAJOR + ROW_MAJOR -> ROW_MAJOR (binary_ng)
        out = kit.readback(y).to(x.dtype)
        kit.free(y, x_rm, *tmp)
        return out, size


def swap_davit_window_attns(vision_tower: nn.Module, device, kit=None) -> int:
    """``kit`` (a ``fused.DeviceKit``, TT_FUSED=1) selects the fused class; None = legacy."""
    swapped = 0
    for stage_seq in vision_tower.blocks:
        for dual_block in stage_seq:
            sb = getattr(dual_block, "spatial_block", None)
            if sb is None or not hasattr(sb, "window_attn"):
                continue
            if kit is None:
                sb.window_attn = TTNNDaViTPreNormWindowAttn(sb.window_attn, device).to(torch.bfloat16)
            else:
                sb.window_attn = TTNNDaViTPreNormWindowAttnFused(sb.window_attn, device, kit).to(torch.bfloat16)
            swapped += 1
    return swapped
