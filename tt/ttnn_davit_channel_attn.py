# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TT-NN port of DaViT's PreNorm + ChannelAttention block, on-device only.

ChannelAttention computes attention over CHANNELS rather than tokens:

    qkv = Linear(x)                                    # [B, T, 3C]
    q, k, v reshape to [B, groups, T, C/groups]
    q *= 1/sqrt(T)
    scores = q.T @ k                                   # [B, g, C/g, C/g]
    probs  = softmax(scores)
    out    = (probs @ v.T).T                           # [B, g, T, C/g]
    out reshape to [B, T, C]
    out = Linear_proj(out)

This implementation keeps everything on chip — no intermediate
torch round-trips. The 5D split-and-permute is decomposed into
4D ttnn ops: ttnn.split along the channel axis, ttnn.reshape to
introduce the groups dim, and ttnn.permute(0,2,1,3) to make
groups precede tokens.

Two implementations: ``TTNNDaViTPreNormChannelAttn`` (legacy, default, untouched: 21
launches plus the hidden untilize/tilize pairs inside ``ttnn.reshape`` of padded TILE
tensors) and ``TTNNDaViTPreNormChannelAttnFused`` (``TT_FUSED=1``): the group split is ONE
``split_query_key_value_and_split_heads(num_heads=groups, transpose_key=False)`` (32 channels
per group = one tile wide), the padded token rows of k (T = 784/196/49 -> 800/224/64; they hold
LN(0)=beta @ W + b, NOT zeros, and the q^T k contraction runs over them) are zeroed with
``fill_implicit_tile_padding(k, 0)`` (exact; in place -- the result views k's buffer, so k is
freed once, after the contraction), then ``transpose(q) @ k``,
``scale_mask_softmax_in_place(scale=T^-0.5, mask=zeros[B, 1, 32, 32])``, ``v @ transpose(probs)`` (==
``(probs @ v^T)^T``), ``concatenate_heads``, and proj + residual as one
``dit_minimal_matmul_addcmul_fused``: 12-13 launches incl. device tilize/untilize.
"""

from __future__ import annotations

from pathlib import Path

import torch
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


class TTNNDaViTPreNormChannelAttn(nn.Module):
    def __init__(self, prenorm_torch: nn.Module, device) -> None:
        super().__init__()
        import ttnn

        self._ttnn = ttnn
        self.device = device

        norm = prenorm_torch.norm
        attn = prenorm_torch.fn
        self.groups = int(attn.groups)
        self.dim = int(attn.qkv.in_features)
        assert self.dim % self.groups == 0
        self.ch_per_group = self.dim // self.groups

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

    def forward(self, x: torch.Tensor, size):
        ttnn = self._ttnn
        residual_dtype = x.dtype
        B, T, C = x.shape
        g = self.groups
        cpg = self.ch_per_group
        scale = float(T) ** -0.5

        x_tt = _bf16_tile(ttnn, x, self.device)
        residual = x_tt

        # PreNorm + qkv linear
        h = ttnn.layer_norm(x_tt, weight=self.ln_w, bias=self.ln_b)
        qkv = ttnn.linear(h, self.qkv_w, bias=self.qkv_b)  # [B, T, 3C]
        ttnn.deallocate(h)

        # Split into Q, K, V each [B, T, C] along channel dim. ttnn.split's
        # second arg is chunk SIZE (like torch.split), not num chunks.
        q, k, v = ttnn.split(qkv, C, dim=-1)
        ttnn.deallocate(qkv)

        # Reshape [B, T, C] -> [B, T, g, cpg], then permute to [B, g, T, cpg]
        def _to_groups(t):
            t = ttnn.reshape(t, (B, T, g, cpg))
            return ttnn.permute(t, (0, 2, 1, 3))

        q = _to_groups(q)
        k = _to_groups(k)
        v = _to_groups(v)

        # scores = q.T @ k  ->  [B, g, cpg, cpg]
        q_T = ttnn.transpose(q, -2, -1)        # [B, g, cpg, T]
        ttnn.deallocate(q)
        scores = ttnn.matmul(q_T, k)
        ttnn.deallocate(q_T); ttnn.deallocate(k)
        scores = ttnn.multiply(scores, scale)
        probs = ttnn.softmax(scores, dim=-1)
        ttnn.deallocate(scores)

        # (probs @ v.T).T  ->  [B, g, T, cpg]
        v_T = ttnn.transpose(v, -2, -1)        # [B, g, cpg, T]
        ttnn.deallocate(v)
        attn_out = ttnn.matmul(probs, v_T)     # [B, g, cpg, T]
        ttnn.deallocate(probs); ttnn.deallocate(v_T)
        attn_out = ttnn.transpose(attn_out, -2, -1)  # [B, g, T, cpg]

        # Merge groups back to channels: [B, g, T, cpg] -> [B, T, g, cpg] -> [B, T, C]
        attn_out = ttnn.permute(attn_out, (0, 2, 1, 3))
        attn_out = ttnn.reshape(attn_out, (B, T, C))

        # proj linear + residual
        out = ttnn.linear(attn_out, self.proj_w, bias=self.proj_b)
        ttnn.deallocate(attn_out)
        x_tt = ttnn.add(residual, out)
        ttnn.deallocate(out)

        out_torch = ttnn.to_torch(x_tt).to(residual_dtype)
        ttnn.deallocate(x_tt)
        return out_torch, size


class TTNNDaViTPreNormChannelAttnFused(nn.Module):
    """``TT_FUSED=1`` variant of ``TTNNDaViTPreNormChannelAttn`` (same ``forward(x, size)`` ->
    ``(x + attn, size)`` contract). Per stage (3 views): [3, 3136, 256] g8 / [3, 784, 512] g16 /
    [3, 196, 1024] g32 / [3, 49, 2048] g64; 32 channels per group everywhere.

    Device graph (``fused.channel_attention_restructured`` is its torch model, proven exact
    against the reference ``ChannelAttention`` in tests/test_fused_host.py):

        x_t   = tilize_with_zero_padding(upload_rm(x))         1
        h     = LN(x_t); qkv = linear(h)  [B, T, 3C]           2
        q,k,v = split_heads(qkv, groups, transpose_key=False)  1   [B, g, T, 32]
        k     = fill_implicit_tile_padding(k, 0.0)             1   (only when T % 32 != 0;
                                                                   in place -> a view of k)
        s     = transpose(q, -2, -1) @ k                       2   [B, g, 32, 32]
        p     = scale_mask_softmax_in_place(s, T^-0.5, 0-mask) 1   (a constant [B, 1, 32, 32]
                                                                   zero mask: the validate refuses
                                                                   a scale without a mask)
        o     = v @ transpose(p, -2, -1)                       2   [B, g, T, 32]
        ctx   = concatenate_heads(o)                           1   [B, T, C]
        y     = dit(ctx, W_proj, 1.0, x_t, ones) + b           1   (proj + residual)
        out   = untilize_with_unpadding(y) -> ROW_MAJOR        1
    ``TT_FUSED_CATTN=0`` keeps the legacy 4-D split/reshape/permute chain (A/B)."""

    def __init__(self, prenorm_torch: nn.Module, device, kit) -> None:
        super().__init__()
        self._fused = _fused_module()
        self.kit = kit
        self.cfg = kit.cfg
        self._ttnn = kit.ttnn
        self.device = device
        norm = prenorm_torch.norm
        attn = prenorm_torch.fn
        self.groups = int(attn.groups)
        self.dim = int(attn.qkv.in_features)
        assert self.dim % self.groups == 0
        self.ch_per_group = self.dim // self.groups
        if self.cfg.cattn:
            assert self.ch_per_group % 32 == 0, "head ops need 32-aligned channels per group"

        def bias_or_zeros(lin):
            return lin.bias.detach() if lin.bias is not None else torch.zeros(lin.out_features)

        self.ln_w = kit.vec(norm.weight)
        self.ln_b = kit.vec(norm.bias)
        self.ln_eps = float(norm.eps)
        self.qkv_w = kit.weight(attn.qkv.weight.detach().t())
        self.qkv_b = kit.bias_row(bias_or_zeros(attn.qkv))
        self.proj_w = kit.weight(attn.proj.weight.detach().t())
        self.proj_b = kit.bias_row(bias_or_zeros(attn.proj))

    def _context_head_ops(self, qkv, T: int):
        """qkv TILE ``[B, T, 3C]`` -> context ``[B, T, C]`` with the head ops (see class doc)."""
        ttnn = self._ttnn
        kit = self.kit
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, num_heads=self.groups, transpose_key=False
        )
        if T % 32 != 0:
            # In-place op: FillPadDeviceOperation::create_output_tensors returns its input, and
            # for rank > 3 fill_pad.cpp wraps it in two ttnn::reshape VIEWS of the same buffer, so
            # the returned Python object is new but owns no memory of its own. Never free the
            # input here (ttnn.deallocate defaults to force=True and would free the buffer the
            # matmul below reads); the view is freed once, after the contraction.
            k = ttnn.fill_implicit_tile_padding(k, 0.0)
        q_T = ttnn.transpose(q, -2, -1)  # [B, g, cpg, T]
        kit.free(q)
        scores = ttnn.matmul(q_T, k)  # [B, g, cpg, cpg]
        kit.free(q_T, k)
        probs = kit.scale_softmax_(scores, float(T) ** -0.5)  # + zero mask: the validate needs one
        p_T = ttnn.transpose(probs, -2, -1)
        kit.free(probs)
        out = ttnn.matmul(v, p_T)  # [B, g, T, cpg] == (probs @ v^T)^T
        kit.free(v, p_T)
        ctx = ttnn.transformer.concatenate_heads(out)  # [B, T, C]
        kit.free(out)
        return ctx

    def _context_legacy(self, qkv, B: int, T: int, C: int):
        """The legacy 4-D chain (``TT_FUSED_CATTN=0``), verbatim from the legacy class."""
        ttnn = self._ttnn
        g, cpg = self.groups, self.ch_per_group
        q, k, v = ttnn.split(qkv, C, dim=-1)

        def _to_groups(t):
            t = ttnn.reshape(t, (B, T, g, cpg))
            return ttnn.permute(t, (0, 2, 1, 3))

        q = _to_groups(q); k = _to_groups(k); v = _to_groups(v)
        q_T = ttnn.transpose(q, -2, -1)
        ttnn.deallocate(q)
        scores = ttnn.matmul(q_T, k)
        ttnn.deallocate(q_T); ttnn.deallocate(k)
        scores = ttnn.multiply(scores, float(T) ** -0.5)
        probs = ttnn.softmax(scores, dim=-1)
        ttnn.deallocate(scores)
        v_T = ttnn.transpose(v, -2, -1)
        ttnn.deallocate(v)
        attn_out = ttnn.matmul(probs, v_T)
        ttnn.deallocate(probs); ttnn.deallocate(v_T)
        attn_out = ttnn.transpose(attn_out, -2, -1)
        attn_out = ttnn.permute(attn_out, (0, 2, 1, 3))
        return ttnn.reshape(attn_out, (B, T, C))

    def forward(self, x: torch.Tensor, size):
        kit = self.kit
        B, T, C = x.shape
        x_dev = kit.upload(x)
        x_t = kit.ingest(x_dev)
        h = kit.layer_norm(x_t, self.ln_w, self.ln_b, self.ln_eps)
        qkv = kit.linear(h, self.qkv_w, self.qkv_b)  # [B, T, 3C]
        kit.free(h)
        ctx = self._context_head_ops(qkv, T) if self.cfg.cattn else self._context_legacy(qkv, B, T, C)
        kit.free(qkv)
        y = kit.linear_residual(ctx, self.proj_w, self.proj_b, x_t)
        kit.free(ctx, x_t, x_dev)
        y_out = kit.egress(y, (B, T, C))
        if y_out is not y:
            kit.free(y)
        out = kit.readback(y_out).to(x.dtype)
        kit.free(y_out)
        return out, size


def swap_davit_channel_attns(vision_tower: nn.Module, device, kit=None) -> int:
    """``kit`` (a ``fused.DeviceKit``, TT_FUSED=1) selects the fused class; None = legacy."""
    swapped = 0
    for stage_seq in vision_tower.blocks:
        for dual_block in stage_seq:
            cb = getattr(dual_block, "channel_block", None)
            if cb is None or not hasattr(cb, "channel_attn"):
                continue
            if kit is None:
                cb.channel_attn = TTNNDaViTPreNormChannelAttn(cb.channel_attn, device).to(torch.bfloat16)
            else:
                cb.channel_attn = TTNNDaViTPreNormChannelAttnFused(cb.channel_attn, device, kit).to(torch.bfloat16)
            swapped += 1
    return swapped
