# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TT-NN port of the 24-layer TransformerBlock stack inside
`SoftPromptedTransformer` for X-VLA.

Each block is pre-LN:

    x = x + attn(layer_norm(x))
    x = x + mlp (layer_norm(x))

where:
    attn  : Q/K/V linear (fused), SDPA (non-causal), output proj, each with bias.
    mlp   : fc1 (H -> 4H), GELU-tanh, fc2 (4H -> H), both with bias.

The stack is what runs in the hot path. Moving 24 blocks on-chip amortizes
the single torch->ttnn and ttnn->torch transfer per inference.

Two implementations live here:

* ``TTNNTransformerBlockStack`` -- the legacy port (default): 14 eager launches per block
  (LN, qkv, split, q@kT, scale, softmax, p@v, concat, proj, add, LN, fc1+gelu, fc2, add),
  host tilize on upload and host untilize on readback. Untouched.
* ``TTNNTransformerBlockStackFused`` -- selected only with ``TT_FUSED=1`` (``tt/fused.py``):
  10 launches per block (LN, qkv, split(k untransposed), scaled_dot_product_attention,
  concat, proj+residual as one ``dit_minimal_matmul_addcmul_fused``, LN, fc1+gelu, fc2, add;
  9 with ``TT_FUSED_FC2_BF16=1`` which fuses fc2+residual too), ROW_MAJOR upload +
  ``tilize_with_zero_padding`` / ``untilize_with_unpadding`` + ROW_MAJOR readback of only the
  ``ceil32(chunk_size)`` action rows the transformer consumes, and the WHOLE 24-block graph
  captured as one metal trace replayed per denoising step (persistent input buffer,
  ``copy_host_to_device_tensor`` per step; eager fallback).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import torch
from torch import nn


def _fused_module():
    """Import ``tt/fused.py`` both as a package sibling (``import tt.ttnn_block_stack``) and
    when this file is loaded by path (``tt/policy.py:_load_module_file``)."""
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


def _to_tile_tensor(ttnn_mod, t: torch.Tensor, device):
    """Upload a 2-D weight/1-D bias to device DRAM as bfloat16 tiled tensor."""
    return ttnn_mod.from_torch(
        t.to(torch.bfloat16).contiguous(),
        dtype=ttnn_mod.bfloat16,
        layout=ttnn_mod.TILE_LAYOUT,
        device=device,
    )


class TTNNTransformerBlockStack(nn.Module):
    """Drop-in replacement for `SoftPromptedTransformer.blocks` — runs the
    entire 24-block residual stack on device in one shot.

    The outer `SoftPromptedTransformer.forward` uses `for block in self.blocks:
    x = block(x)`. We preserve that by exposing this object as a
    `nn.ModuleList`-like iterable of length 1, where the single element is a
    callable that runs all 24 blocks in sequence.
    """

    def __init__(self, torch_blocks: nn.ModuleList, device, num_heads: int) -> None:
        super().__init__()
        import ttnn

        self._ttnn = ttnn
        self.device = device
        self.num_heads = num_heads
        self.num_layers = len(torch_blocks)
        head_dim = torch_blocks[0].attn.head_dim
        self.head_dim_inv_sqrt = float(head_dim) ** -0.5

        # Per-layer weight bundles, each kept on device DRAM.
        self._layers: List[Any] = []
        for blk in torch_blocks:
            self._layers.append(self._bundle_block(blk))

    # -- weight loading ------------------------------------------------------

    def _bundle_block(self, blk: nn.Module) -> dict:
        ttnn = self._ttnn
        dev = self.device

        def bf16(t):
            return _to_tile_tensor(ttnn, t, dev)

        def bfp8(t):
            # Per-block 8-bit exponent-sharing float — half the bytes of bf16.
            # Fine for over-parameterized transformer MLP weights (GPT-3-class
            # accuracy studies show <0.5% drop).
            return ttnn.from_torch(
                t.to(torch.bfloat16).contiguous(),
                dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev,
            )

        # LayerNorm 1
        ln1_w = bf16(blk.norm1.weight.detach())
        ln1_b = bf16(blk.norm1.bias.detach())
        # Attention — weights stay bf16 (small, accuracy-sensitive for scores)
        qkv_w = bf16(blk.attn.qkv.weight.detach().t().contiguous())
        qkv_b = bf16(blk.attn.qkv.bias.detach())
        proj_w = bf16(blk.attn.proj.weight.detach().t().contiguous())
        proj_b = bf16(blk.attn.proj.bias.detach())
        # LayerNorm 2
        ln2_w = bf16(blk.norm2.weight.detach())
        ln2_b = bf16(blk.norm2.bias.detach())
        # MLP — by far the biggest weights (H*4H each way). bfp8 here halves
        # their DRAM footprint (for 24 layers: 24 * 2 * 1024*4096 * 2B =
        # 384MB -> 192MB), and matmul FLOPs are typically memory-bound on
        # this shape/batch so it should translate to throughput.
        fc1_w = bfp8(blk.mlp.fc1.weight.detach().t().contiguous())
        fc1_b = bf16(blk.mlp.fc1.bias.detach())
        fc2_w = bfp8(blk.mlp.fc2.weight.detach().t().contiguous())
        fc2_b = bf16(blk.mlp.fc2.bias.detach())

        return dict(
            ln1_w=ln1_w, ln1_b=ln1_b,
            qkv_w=qkv_w, qkv_b=qkv_b,
            proj_w=proj_w, proj_b=proj_b,
            ln2_w=ln2_w, ln2_b=ln2_b,
            fc1_w=fc1_w, fc1_b=fc1_b,
            fc2_w=fc2_w, fc2_b=fc2_b,
        )

    # -- forward -------------------------------------------------------------

    def _block_forward(self, x_tt, wb: dict, head_dim_inv_sqrt: float):
        """Run one pre-LN transformer block on-device.

        Attention is implemented manually (matmul+softmax+matmul) rather than
        calling `ttnn.transformer.scaled_dot_product_attention`, because
        Flash-2 on Blackhole currently requires Q/K/V seq lengths to be
        padded to the tile size and matching, and our seq length is not a
        multiple of 32. Manual SDPA works at any length.
        """
        ttnn = self._ttnn

        # --- Self-attention ------------------------------------------------
        h = ttnn.layer_norm(x_tt, weight=wb["ln1_w"], bias=wb["ln1_b"])
        qkv = ttnn.linear(h, wb["qkv_w"], bias=wb["qkv_b"])
        ttnn.deallocate(h)
        # split heads: [B, S, 3H] -> q,k,v each [B, num_heads, S, head_dim]
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, num_heads=self.num_heads
        )
        ttnn.deallocate(qkv)
        # split_query_key_value_and_split_heads already returns K pre-transposed
        # to shape [B, H, head_dim, S], so q @ k directly gives attention scores.
        scores = ttnn.matmul(q, k)
        ttnn.deallocate(q); ttnn.deallocate(k)
        scores = ttnn.multiply(scores, head_dim_inv_sqrt)
        probs = ttnn.softmax(scores, dim=-1)
        ttnn.deallocate(scores)
        attn_out = ttnn.matmul(probs, v)
        ttnn.deallocate(probs); ttnn.deallocate(v)
        # merge heads: [B, H, S, Dh] -> [B, S, H*Dh]
        attn_out = ttnn.transformer.concatenate_heads(attn_out)
        proj = ttnn.linear(attn_out, wb["proj_w"], bias=wb["proj_b"])
        ttnn.deallocate(attn_out)
        x_tt = ttnn.add(x_tt, proj)
        ttnn.deallocate(proj)

        # --- MLP -----------------------------------------------------------
        h = ttnn.layer_norm(x_tt, weight=wb["ln2_w"], bias=wb["ln2_b"])
        h = ttnn.linear(h, wb["fc1_w"], bias=wb["fc1_b"], activation="gelu")
        h = ttnn.linear(h, wb["fc2_w"], bias=wb["fc2_b"])
        x_tt = ttnn.add(x_tt, h)
        ttnn.deallocate(h)
        return x_tt

    def __iter__(self):
        # Support the `for block in self.blocks: x = block(x)` idiom — we
        # represent the whole stack as a single "block".
        yield self._run_all

    def __len__(self):
        return 1

    def _run_all(self, x_torch: torch.Tensor) -> torch.Tensor:
        """Torch tensor in, torch tensor out; the 24-block stack runs on chip."""
        ttnn = self._ttnn
        x_bf16 = x_torch.to(torch.bfloat16).contiguous()
        x_tt = ttnn.from_torch(
            x_bf16, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        for wb in self._layers:
            x_tt = self._block_forward(x_tt, wb, self.head_dim_inv_sqrt)
        out = ttnn.to_torch(x_tt).to(x_torch.dtype)
        ttnn.deallocate(x_tt)
        return out


class TTNNTransformerBlockStackFused(nn.Module):
    """``TT_FUSED=1`` variant of ``TTNNTransformerBlockStack`` (same iterable-of-one-callable
    contract: ``for block in blocks: x = block(x)``; torch ``[1, S, 1024]`` in, torch out).

    Per block (``_block_forward``), S = 244 (tile-padded 256), 16 heads x 64:

        h    = layer_norm(x)                                   1
        qkv  = linear(h)  [1, S, 3072]                         1   (minimal_matmul with TT_FUSED_MINIMAL_MM)
        ctx  = split_heads(k untransposed) -> SDPA -> concat   3   (SDPA: is_causal=False, scale=1/8,
                                                                    q/k chunk 256, no attn_mask -- the
                                                                    kernel masks the 12 padded keys)
        x    = dit_minimal_matmul_addcmul_fused(ctx, W_proj, 1.0, x, ones) + b   1  (residual fused;
                                                                    bf16 weight == bf16 residual)
        h    = layer_norm(x)                                   1
        h    = linear(h, fc1, activation=gelu)                 1   (exact erf GELU, as legacy)
        x    = fc2 + residual                                  2   (1 when TT_FUSED_FC2_BF16=1 -> dit;
                                                                    bfp8 fc2 cannot fuse: dtype rule)
    = 10 launches (legacy 14). Whole call: tilize_with_zero_padding(RM input) + 24 blocks +
    untilize_with_unpadding to ``[1, out_rows, 1024]`` ROW_MAJOR = 242 launches, ONE trace.

    ``out_rows``: the transformer only reads ``x[:, :chunk_size]`` (30 rows); by default the
    graph unpads to 32 rows (``fused.stack_out_rows``), so the readback is 64 KB instead of
    512 KB. ``None`` returns the full sequence (legacy contract).
    """

    def __init__(self, torch_blocks: nn.ModuleList, device, num_heads: int, kit,
                 out_rows: Optional[int] = None, log=print) -> None:
        super().__init__()
        fused = _fused_module()
        self._fused = fused
        self.kit = kit
        self.cfg = kit.cfg
        self._ttnn = kit.ttnn
        self.device = device
        self.num_heads = int(num_heads)
        self.num_layers = len(torch_blocks)
        head_dim = torch_blocks[0].attn.head_dim
        self.head_dim_inv_sqrt = float(head_dim) ** -0.5
        self.out_rows = out_rows
        self._layers: List[dict] = [self._bundle_block(blk) for blk in torch_blocks]
        self.trace = fused.TracedGraph(kit.ttnn, device, self._graph, "block_stack", enabled=self.cfg.trace, log=log)

    # -- weight loading ------------------------------------------------------

    def _bundle_block(self, blk: nn.Module) -> dict:
        kit = self.kit
        return dict(
            ln1_w=kit.vec(blk.norm1.weight), ln1_b=kit.vec(blk.norm1.bias), eps1=float(blk.norm1.eps),
            qkv_w=kit.weight(blk.attn.qkv.weight.detach().t()), qkv_b=kit.bias_row(blk.attn.qkv.bias),
            proj_w=kit.weight(blk.attn.proj.weight.detach().t()), proj_b=kit.bias_row(blk.attn.proj.bias),
            ln2_w=kit.vec(blk.norm2.weight), ln2_b=kit.vec(blk.norm2.bias), eps2=float(blk.norm2.eps),
            # fc1 stays bfp8 (as legacy); fc2 is bf16 only when its residual fusion is requested.
            fc1_w=kit.weight(blk.mlp.fc1.weight.detach().t(), bf8=True), fc1_b=kit.bias_row(blk.mlp.fc1.bias),
            fc2_w=kit.weight(blk.mlp.fc2.weight.detach().t(), bf8=not self.cfg.fc2_bf16),
            fc2_b=kit.bias_row(blk.mlp.fc2.bias),
        )

    # -- forward -------------------------------------------------------------

    def _block_forward(self, x, wb: dict, keep_input: bool = False):
        """One pre-LN block. ``keep_input``: do not free ``x`` (it is the graph's input)."""
        ttnn = self._ttnn
        kit = self.kit
        seq_len = int(x.shape[1])

        h = kit.layer_norm(x, wb["ln1_w"], wb["ln1_b"], wb["eps1"])
        qkv = kit.linear(h, wb["qkv_w"], wb["qkv_b"])
        ttnn.deallocate(h)
        ctx = kit.attention(qkv, self.num_heads, seq_len, self.head_dim_inv_sqrt)
        ttnn.deallocate(qkv)
        x2 = kit.linear_residual(ctx, wb["proj_w"], wb["proj_b"], x)
        ttnn.deallocate(ctx)
        if not keep_input:
            ttnn.deallocate(x)

        h = kit.layer_norm(x2, wb["ln2_w"], wb["ln2_b"], wb["eps2"])
        h2 = kit.linear(h, wb["fc1_w"], wb["fc1_b"], gelu=True)
        ttnn.deallocate(h)
        x3 = kit.linear_residual(h2, wb["fc2_w"], wb["fc2_b"], x2)
        ttnn.deallocate(h2)
        ttnn.deallocate(x2)
        return x3

    def _graph(self, x_in):
        """Device input (RM ``[1, S, C]`` when rm_io, else TILE) -> device output (RM
        ``[1, out_rows, C]`` when rm_io, else TILE). Traced as a whole."""
        kit = self.kit
        x = kit.ingest(x_in)
        for i, wb in enumerate(self._layers):
            x = self._block_forward(x, wb, keep_input=(i == 0 and x is x_in))
        b, s, c = (int(d) for d in x.shape)
        rows = s if self.out_rows is None else min(int(self.out_rows), s)
        if not self.cfg.rm_io and rows < s:
            y = self._ttnn.slice(x, [0, 0, 0], [b, rows, c])
            self._ttnn.deallocate(x)
            return y
        y = kit.egress(x, (b, rows, c))
        if y is not x:
            self._ttnn.deallocate(x)
        return y

    def __iter__(self):
        yield self._run_all

    def __len__(self):
        return 1

    def _run_all(self, x_torch: torch.Tensor) -> torch.Tensor:
        """Torch ``[1, S, C]`` in; torch ``[1, out_rows or S, C]`` out. First call: eager warm run
        + trace capture (+ replay); later calls: copy into the persistent input + replay."""
        host = self.kit.host_tensor(x_torch)
        out = self.trace.run(host)
        return out.to(x_torch.dtype)

    def status(self) -> dict:
        st = self.trace.status()
        st.update(module="block_stack", layers=self.num_layers, out_rows=self.out_rows,
                  fc2_dtype="bf16" if self.cfg.fc2_bf16 else "bfp8_b")
        return st
