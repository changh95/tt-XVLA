# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TT-NN port of the FFN module inside every DaViT block.

DaViT's `SpatialBlock` and `ChannelBlock` each wrap their FFN as

    PreNorm(LayerNorm, Mlp(linear -> GELU(tanh) -> linear), drop_path)

which evaluates as `x + drop_path(Mlp(LN(x)))`. In eval mode drop_path is
identity, so it's just `x + Mlp(LN(x))`.

The Mlp's two linear weights are by far the heaviest tensors in each
DaViT block (dim x 4*dim each way). Across the 12 dual-blocks (24 FFN
modules) this dominates DaViT's parameter footprint.

Each ported FFN takes a torch tensor in and returns a torch tensor out
(plus the unchanged `size` tuple), matching the upstream signature so
the `MySequential` wrapper still works.

Two implementations: ``TTNNDaViTPreNormFFN`` (legacy, default, untouched: host tilize,
LN, fc1+gelu, fc2, add, host untilize) and ``TTNNDaViTPreNormFFNFused`` (``TT_FUSED=1``:
ROW_MAJOR upload, ``tilize_with_zero_padding``, LN, fc1+gelu, fc2 + residual -- one
``dit_minimal_matmul_addcmul_fused`` when ``TT_FUSED_FC2_BF16=1``, else fc2 + add --,
``untilize_with_unpadding``, ROW_MAJOR readback: 6 launches, no host layout work).
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


def _bfp8_tile(ttnn_mod, t: torch.Tensor, device):
    return ttnn_mod.from_torch(
        t.to(torch.bfloat16).contiguous(),
        dtype=ttnn_mod.bfloat8_b, layout=ttnn_mod.TILE_LAYOUT, device=device,
    )


class TTNNDaViTPreNormFFN(nn.Module):
    """`PreNorm(LN, Mlp)` evaluated end-to-end on the device.

    Replaces a `PreNorm` whose `fn` is an `Mlp` and whose `norm` is a
    `LayerNorm`. `drop_path` is treated as identity (eval-mode behavior).
    """

    def __init__(self, prenorm_torch: nn.Module, device) -> None:
        super().__init__()
        import ttnn

        self._ttnn = ttnn
        self.device = device

        norm = prenorm_torch.norm
        mlp = prenorm_torch.fn

        # Mlp.net is an nn.Sequential of OrderedDict([fc1, act, drop1, norm, fc2, drop2]).
        net = mlp.net
        fc1 = net.fc1
        fc2 = net.fc2

        self.ln_w = _bf16_tile(ttnn, norm.weight.detach(), device)
        self.ln_b = _bf16_tile(ttnn, norm.bias.detach(), device)
        # bfp8_b for the big matmul weights
        self.fc1_w = _bfp8_tile(ttnn, fc1.weight.detach().t().contiguous(), device)
        self.fc1_b = _bf16_tile(
            ttnn,
            (fc1.bias.detach() if fc1.bias is not None
             else torch.zeros(fc1.out_features)),
            device,
        )
        self.fc2_w = _bfp8_tile(ttnn, fc2.weight.detach().t().contiguous(), device)
        self.fc2_b = _bf16_tile(
            ttnn,
            (fc2.bias.detach() if fc2.bias is not None
             else torch.zeros(fc2.out_features)),
            device,
        )

    def forward(self, x: torch.Tensor, size):
        ttnn = self._ttnn
        residual_dtype = x.dtype
        x_tt = _bf16_tile(ttnn, x, self.device)
        h = ttnn.layer_norm(x_tt, weight=self.ln_w, bias=self.ln_b)
        h = ttnn.linear(h, self.fc1_w, bias=self.fc1_b, activation="gelu")
        h = ttnn.linear(h, self.fc2_w, bias=self.fc2_b)
        x_tt = ttnn.add(x_tt, h)
        ttnn.deallocate(h)
        out = ttnn.to_torch(x_tt).to(residual_dtype)
        ttnn.deallocate(x_tt)
        return out, size


class TTNNDaViTPreNormFFNFused(nn.Module):
    """``TT_FUSED=1`` variant of ``TTNNDaViTPreNormFFN``: ``x + fc2(gelu(fc1(LN(x))))`` with
    ROW_MAJOR I/O and device tilize/untilize (exact data movement), fc1 bfp8 + exact erf GELU
    (as legacy), fc2 bfp8 + ``add`` by default or bf16 + fused residual with
    ``TT_FUSED_FC2_BF16=1`` (dtype rule of the fused kernel). Shapes per stage (3 views):
    [3, 3136, 256] / [3, 784, 512] / [3, 196, 1024] / [3, 49, 2048] (tile-padded rows
    3136/800/224/64; padded rows are LN(0)=beta garbage, row-wise ops only, dropped by the unpad)."""

    def __init__(self, prenorm_torch: nn.Module, device, kit) -> None:
        super().__init__()
        self._fused = _fused_module()
        self.kit = kit
        self.cfg = kit.cfg
        self._ttnn = kit.ttnn
        self.device = device
        norm = prenorm_torch.norm
        net = prenorm_torch.fn.net
        fc1, fc2 = net.fc1, net.fc2

        def bias_or_zeros(lin):
            return lin.bias.detach() if lin.bias is not None else torch.zeros(lin.out_features)

        self.ln_w = kit.vec(norm.weight)
        self.ln_b = kit.vec(norm.bias)
        self.ln_eps = float(norm.eps)
        self.fc1_w = kit.weight(fc1.weight.detach().t(), bf8=True)
        self.fc1_b = kit.bias_row(bias_or_zeros(fc1))
        self.fc2_w = kit.weight(fc2.weight.detach().t(), bf8=not self.cfg.fc2_bf16)
        self.fc2_b = kit.bias_row(bias_or_zeros(fc2))

    def forward(self, x: torch.Tensor, size):
        kit = self.kit
        B, T, C = x.shape
        x_dev = kit.upload(x)
        x_t = kit.ingest(x_dev)
        h = kit.layer_norm(x_t, self.ln_w, self.ln_b, self.ln_eps)
        h2 = kit.linear(h, self.fc1_w, self.fc1_b, gelu=True)
        kit.free(h)
        y = kit.linear_residual(h2, self.fc2_w, self.fc2_b, x_t)
        kit.free(h2, x_t, x_dev)
        y_out = kit.egress(y, (B, T, C))
        if y_out is not y:
            kit.free(y)
        out = kit.readback(y_out).to(x.dtype)
        kit.free(y_out)
        return out, size


def swap_davit_ffns(vision_tower: nn.Module, device, kit=None) -> int:
    """Walk the DaViT and replace each block's `ffn` PreNorm with the
    on-device version. Returns the number of swaps made. ``kit`` (a
    ``fused.DeviceKit``, TT_FUSED=1) selects the fused class; None = legacy.

    DaViT's structure is:
        vision_tower.blocks: ModuleList[stage_seq]
        stage_seq: MySequential of dual_blocks
        dual_block: MySequential with .spatial_block and .channel_block
        spatial_block / channel_block: SpatialBlock / ChannelBlock
        each has .ffn = PreNorm(LayerNorm, Mlp(...))
    """
    swapped = 0
    for stage_seq in vision_tower.blocks:
        for dual_block in stage_seq:
            for sb_name in ("spatial_block", "channel_block"):
                stage_block = getattr(dual_block, sb_name, None)
                if stage_block is None or not hasattr(stage_block, "ffn"):
                    continue
                cls = TTNNDaViTPreNormFFN if kit is None else TTNNDaViTPreNormFFNFused
                args = (stage_block.ffn, device) if kit is None else (stage_block.ffn, device, kit)
                stage_block.ffn = cls(*args).to(torch.bfloat16)
                swapped += 1
    return swapped
