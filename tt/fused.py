# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in fused / traced device path for the X-VLA port (``TT_FUSED=1``).

Everything in the ``*Fused`` classes of ``ttnn_block_stack.py``, ``ttnn_bart_encoder.py``
and the three ``ttnn_davit_*.py`` modules is selected when ``TT_FUSED`` is unset or 1 (the
DEFAULT since the p150a validation of 2026-09-13, DEVICE_VALIDATION.md "Results"); read once
at model build time (``tt/policy.py``). ``TT_FUSED=0`` selects the legacy classes, bit-for-bit
the published 2026-09-12 behaviour. This module holds what those classes share:

1. ``FusedConfig`` -- the knobs, read ONCE from the environment (see ``FusedConfig.from_env``).
2. Torch-only reformulations used both by the device code (to build constants) and by
   ``tests/test_fused_host.py`` (to prove them against the reference math without a device):
   tile arithmetic, the ttnn head split/merge as torch, the DaViT channel-attention
   restructure (head ops + zero-filled tile padding + ``v @ probs^T``), the 4-D permute form
   of ``window_partition`` / ``window_reverse``, the BART all-ones-mask shortcut.
3. ``DeviceKit`` -- thin wrappers over the fused ttnn ops with the constraints the evaluation
   verified on this tree (v0.71): ROW_MAJOR upload + ``tilize_with_zero_padding`` /
   ``untilize_with_unpadding`` + ROW_MAJOR readback, ``scaled_dot_product_attention`` (no
   explicit padding mask; the kernel masks padded keys), ``dit_minimal_matmul_addcmul_fused``
   for matmul + residual (ones scale vector in the residual's buffer type, residual dtype ==
   weight dtype), ``minimal_matmul`` as a separate knob, ``layer_norm(residual_input_tensor=)``.
4. ``TracedGraph`` -- metal-trace capture of a whole device graph with a persistent
   ROW_MAJOR input buffer (``copy_host_to_device_tensor`` per call), eager warm run before
   capture (program cache), eager fallback when tracing is off, fails, or the input shape
   changes.

``import ttnn`` never happens at import time here: the tests run torch-only, and the device
helpers receive the ``ttnn`` module from their caller.

Knobs (all read once by ``FusedConfig.from_env``; the defaults are the set validated on the
p150a on 2026-09-13 -- every lever below that is on by default passed the model's own gates
and was measured faster; see DEVICE_VALIDATION.md "Results (device, 2026-09-13)"):

    TT_FUSED=1                  master switch (default 1; TT_FUSED=0 = legacy path, untouched)
    TT_FUSED_TRACE=1            capture the block stack and the BART encoder as metal traces
    TT_FUSED_TRACE_REGION_MB=64 trace_region_size passed to ttnn.open_device (default 0 in v0.71!)
    TT_FUSED_RM_IO=1            ROW_MAJOR upload/readback + device tilize/untilize (exact)
    TT_FUSED_SDPA=1             fused scaled_dot_product_attention (precision-affecting, hist. up)
    TT_FUSED_DIT=1              dit_minimal_matmul_addcmul_fused for proj/fc2 + residual
    TT_FUSED_LN_RESIDUAL=1      BART post-LN residual adds fused into layer_norm(residual_input_tensor=)
    TT_FUSED_FC2_BF16=1         upload fc2 weights as bf16 so fc2+residual fuses into the dit kernel
                                (validated: -5.6 ms at 1 step, -22.6 ms at 10 steps, +230 MB DRAM)
    TT_FUSED_MINIMAL_MM=1       minimal_matmul kernel for qkv/fc1/out-proj (validated: -16.5 ms, PCC up)
    TT_FUSED_MM_FIDELITY=       LoFi|HiFi2|HiFi3|HiFi4 for minimal/dit kernels ("" = op default)
    TT_FUSED_MM_FP32ACC=        0|1 fp32 accumulation for minimal/dit kernels ("" = op default)
    TT_FUSED_CATTN=1            DaViT channel attention with head ops (21 -> 12/13 launches)
    TT_FUSED_WINDOWS_ON_DEVICE=1  DaViT window pad/partition/reverse/unpad + residual on device
                                (validated: every RM data-movement op ran, -8.6 ms; NOT bit-identical
                                to the host path -- the residual add is a device bf16 add: max abs
                                diff 3.9e-3 on the action chunk, PCC vs fp32 unchanged -> bf16-rounding)
    TT_FUSED_LN_EPS_REF=0       pass the module's LayerNorm eps (1e-5) instead of ttnn's 1e-12
    TT_FUSED_STACK_OUT_ROWS=-1  rows read back from the block stack (-1 = ceil32(chunk), 0 = all)
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import torch

TILE = 32

# DaViT (Florence-2-large vision tower) stage geometry for 224x224 inputs, 3 views.
# (tokens T, channels C, heads/groups, windows per view at ws=12, padded grid)
DAVIT_STAGES = (
    dict(stage=0, T=3136, H=56, W=56, C=256, heads=8, groups=8, windows_per_view=25),
    dict(stage=1, T=784, H=28, W=28, C=512, heads=16, groups=16, windows_per_view=9),
    dict(stage=2, T=196, H=14, W=14, C=1024, heads=32, groups=32, windows_per_view=4),
    dict(stage=3, T=49, H=7, W=7, C=2048, heads=64, groups=64, windows_per_view=1),
)
DAVIT_WINDOW = 12


# ----------------------------------------------------------------------------- knobs
def _flag(env: Mapping[str, str], name: str, default: bool) -> bool:
    v = env.get(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() not in ("0", "false", "no", "off")


def _int(env: Mapping[str, str], name: str, default: int) -> int:
    v = env.get(name)
    if v is None or v.strip() == "":
        return default
    return int(v)


@dataclass(frozen=True)
class FusedConfig:
    """All ``TT_FUSED*`` knobs. Immutable; built once per process by ``from_env``."""

    enabled: bool = True
    trace: bool = True
    trace_region_mb: int = 64
    rm_io: bool = True
    sdpa: bool = True
    dit: bool = True
    ln_residual: bool = True
    fc2_bf16: bool = True
    minimal_mm: bool = True
    mm_fidelity: str = ""
    mm_fp32acc: str = ""
    cattn: bool = True
    windows_on_device: bool = True
    ln_eps_ref: bool = False
    stack_out_rows: int = -1

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "FusedConfig":
        env = os.environ if env is None else env
        enabled = _flag(env, "TT_FUSED", True)  # default ON since the 2026-09-13 device validation
        if not enabled:
            return cls(enabled=False)
        fid = env.get("TT_FUSED_MM_FIDELITY", "").strip()
        if fid and fid not in ("LoFi", "HiFi2", "HiFi3", "HiFi4"):
            raise RuntimeError(f"TT_FUSED_MM_FIDELITY={fid!r} must be LoFi|HiFi2|HiFi3|HiFi4")
        fp32 = env.get("TT_FUSED_MM_FP32ACC", "").strip()
        if fp32 and fp32 not in ("0", "1"):
            raise RuntimeError(f"TT_FUSED_MM_FP32ACC={fp32!r} must be 0 or 1")
        return cls(
            enabled=True,
            trace=_flag(env, "TT_FUSED_TRACE", True),
            trace_region_mb=_int(env, "TT_FUSED_TRACE_REGION_MB", 64),
            rm_io=_flag(env, "TT_FUSED_RM_IO", True),
            sdpa=_flag(env, "TT_FUSED_SDPA", True),
            dit=_flag(env, "TT_FUSED_DIT", True),
            ln_residual=_flag(env, "TT_FUSED_LN_RESIDUAL", True),
            fc2_bf16=_flag(env, "TT_FUSED_FC2_BF16", True),
            minimal_mm=_flag(env, "TT_FUSED_MINIMAL_MM", True),
            mm_fidelity=fid,
            mm_fp32acc=fp32,
            cattn=_flag(env, "TT_FUSED_CATTN", True),
            windows_on_device=_flag(env, "TT_FUSED_WINDOWS_ON_DEVICE", True),
            ln_eps_ref=_flag(env, "TT_FUSED_LN_EPS_REF", False),
            stack_out_rows=_int(env, "TT_FUSED_STACK_OUT_ROWS", -1),
        )

    def describe(self) -> Dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------ tile arithmetic
def tile_pad(n: int, tile: int = TILE) -> int:
    """Smallest multiple of ``tile`` >= n (the tile-padded extent of a TILE_LAYOUT dim)."""
    return -(-int(n) // tile) * tile


def sdpa_chunk(seq_len: int) -> int:
    """SDPA q/k chunk that covers the whole (tile-padded) sequence in ONE chunk: 244 -> 256,
    82 -> 96, 144 -> 160. Chunks must be multiples of 32 (validate); one chunk per sequence
    means a single-pass softmax (no running rescale) and b*heads work items."""
    return tile_pad(seq_len)


def stack_out_rows(chunk_size: int, knob: int) -> Optional[int]:
    """Rows of the block-stack output the host reads back. The transformer consumes only
    ``x[:, :chunk_size]`` (``SoftPromptedTransformer.forward``), so the traced graph unpads to
    ``ceil32(chunk_size)`` rows by default (knob -1: 30 -> 32); knob 0 = full sequence (None);
    knob n = n rows. The graph clamps the value to the sequence length at run time."""
    if knob == 0:
        return None
    rows = tile_pad(chunk_size) if knob < 0 else int(knob)
    if rows < chunk_size:
        raise ValueError(f"TT_FUSED_STACK_OUT_ROWS={rows} < chunk_size {chunk_size}")
    return rows


# ------------------------------------------------- torch models of the ttnn head ops
def split_heads_torch(qkv: torch.Tensor, num_heads: int, transpose_key: bool) -> Tuple[torch.Tensor, ...]:
    """Torch model of ``ttnn.transformer.split_query_key_value_and_split_heads`` (docstring of
    this tree): ``[B, S, 3*H*dh]`` -> q, k, v ``[B, H, S, dh]`` (k ``[B, H, dh, S]`` when
    ``transpose_key``). Heads are laid out q-heads, then k-heads, then v-heads along the last
    dim -- the same order ``nn.Linear(dim, 3*dim)`` + ``reshape(B, S, 3, H, dh)`` produces in
    the reference ``Attention`` / ``WindowAttention`` / ``ChannelAttention``."""
    b, s, hidden3 = qkv.shape
    head = hidden3 // (3 * num_heads)
    t = qkv.reshape(b, s, 3 * num_heads, head)
    q = t[..., :num_heads, :].permute(0, 2, 1, 3)
    k = t[..., num_heads : 2 * num_heads, :].permute(0, 2, 1, 3)
    v = t[..., 2 * num_heads :, :].permute(0, 2, 1, 3)
    if transpose_key:
        k = k.transpose(-1, -2)
    return q.contiguous(), k.contiguous(), v.contiguous()


def concat_heads_torch(x: torch.Tensor) -> torch.Tensor:
    """Torch model of ``ttnn.transformer.concatenate_heads``: ``[B, H, S, dh]`` -> ``[B, S, H*dh]``."""
    b, h, s, d = x.shape
    return x.permute(0, 2, 1, 3).reshape(b, s, h * d)


def reference_qkv_split(qkv: torch.Tensor, num_heads: int) -> Tuple[torch.Tensor, ...]:
    """The reference modules' own split: ``reshape(B, S, 3, H, dh).permute(2, 0, 3, 1, 4)``."""
    b, s, hidden3 = qkv.shape
    t = qkv.reshape(b, s, 3, num_heads, hidden3 // (3 * num_heads)).permute(2, 0, 3, 1, 4)
    return t[0], t[1], t[2]


# ------------------------------------------------------ DaViT channel attention
def channel_attention_reference(
    x: torch.Tensor, qkv_w: torch.Tensor, qkv_b: torch.Tensor, proj_w: torch.Tensor,
    proj_b: torch.Tensor, groups: int,
) -> torch.Tensor:
    """lerobot ``ChannelAttention.forward`` (modeling_florence2.py) verbatim, torch."""
    batch_size, num_tokens, channels = x.shape
    qkv = (
        torch.nn.functional.linear(x, qkv_w, qkv_b)
        .reshape(batch_size, num_tokens, 3, groups, channels // groups)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = q * (float(num_tokens) ** -0.5)
    attention = q.transpose(-1, -2) @ k
    attention = attention.softmax(dim=-1)
    out = (attention @ v.transpose(-1, -2)).transpose(-1, -2)
    out = out.transpose(1, 2).reshape(batch_size, num_tokens, channels)
    return torch.nn.functional.linear(out, proj_w, proj_b)


def channel_attention_restructured(
    x: torch.Tensor, qkv_w: torch.Tensor, qkv_b: torch.Tensor, proj_w: torch.Tensor,
    proj_b: torch.Tensor, groups: int, pad_rows: bool = True, zero_fill_k: bool = True,
) -> torch.Tensor:
    """Torch model of the fused device graph of ``TTNNDaViTPreNormChannelAttnFused``:

        qkv = linear(x)                                   # [B, Tp, 3C]  (Tp = tile-padded T)
        q, k, v = split_heads(transpose_key=False)        # [B, g, Tp, cpg]
        k = fill_implicit_tile_padding(k, 0)              # rows T..Tp of k := 0 (exact)
        scores = transpose(q) @ k                         # [B, g, cpg, cpg], contraction over Tp
        probs = softmax(scores * T^-0.5)                  # scale_mask_softmax_in_place(scale)
        out = v @ transpose(probs)                        # [B, g, Tp, cpg]  == (probs @ v^T)^T
        ctx = concat_heads(out)                           # [B, Tp, C]
        y = linear_proj(ctx)                              # rows >= T are garbage, dropped by unpad

    ``pad_rows`` models what the device sees: the padded token rows of the LN output are NOT
    zero (LayerNorm(0) = beta, then + qkv bias), so without ``zero_fill_k`` the T-contraction
    would sum garbage (this is the bug the row fill prevents; the test asserts it). Only the
    first T rows of the result are meaningful."""
    B, T, C = x.shape
    cpg = C // groups
    Tp = tile_pad(T) if pad_rows else T
    if Tp != T:
        # Device: LN(zero rows) = beta -> model with a nonzero constant row through qkv.
        beta_rows = torch.full((B, Tp - T, C), 0.37, dtype=x.dtype)
        x_dev = torch.cat([x, beta_rows], dim=1)
    else:
        x_dev = x
    qkv = torch.nn.functional.linear(x_dev, qkv_w, qkv_b)
    q, k, v = split_heads_torch(qkv, groups, transpose_key=False)
    if Tp != T and zero_fill_k:
        k = k.clone()
        k[:, :, T:, :] = 0
    scores = q.transpose(-1, -2) @ k
    probs = (scores * (float(T) ** -0.5)).softmax(dim=-1)
    out = v @ probs.transpose(-1, -2)
    ctx = concat_heads_torch(out)
    y = torch.nn.functional.linear(ctx, proj_w, proj_b)
    return y[:, :T]


def channel_attention_launches(T: int, fused: bool) -> int:
    """Launch count per channel-attention call as written in the code (documentation)."""
    if not fused:
        return 21
    return 13 if tile_pad(T) != T else 12  # incl. tilize + untilize, + fill_pad when padded


# -------------------------------------------------------- DaViT window partition
def window_partition_ref(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """lerobot ``window_partition``: ``[B, Hp, Wp, C]`` -> ``[B*nh*nw, ws, ws, C]``."""
    batch_size, height, width, channels = x.shape
    x = x.view(batch_size, height // window_size, window_size, width // window_size, window_size, channels)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, channels)


def window_reverse_ref(windows: torch.Tensor, batch_size: int, window_size: int, height: int, width: int) -> torch.Tensor:
    """lerobot ``window_reverse``: ``[B*nh*nw, ws, ws, C]`` -> ``[B, Hp, Wp, C]``."""
    x = windows.view(batch_size, height // window_size, width // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, height, width, -1)


def window_partition_4d(x4p: torch.Tensor, ws: int) -> torch.Tensor:
    """The device formulation (rank-4 ops only): ``[B, Hp, Wp, C]`` -> view ``[B*nh, ws, nw, ws*C]``
    -> permute(0, 2, 1, 3) -> view ``[B*nh*nw, ws*ws, C]``. Equals
    ``window_partition_ref(x4p, ws).view(-1, ws*ws, C)`` exactly (pure data movement)."""
    B, Hp, Wp, C = x4p.shape
    nh, nw = Hp // ws, Wp // ws
    v = x4p.reshape(B * nh, ws, nw, ws * C)
    p = v.permute(0, 2, 1, 3).contiguous()
    return p.reshape(B * nh * nw, ws * ws, C)


def window_reverse_4d(windows: torch.Tensor, B: int, ws: int, Hp: int, Wp: int) -> torch.Tensor:
    """Inverse of ``window_partition_4d``: ``[B*nh*nw, ws*ws, C]`` -> ``[B, Hp, Wp, C]``."""
    nh, nw = Hp // ws, Wp // ws
    C = windows.shape[-1]
    v = windows.reshape(B * nh, nw, ws, ws * C)
    p = v.permute(0, 2, 1, 3).contiguous()
    return p.reshape(B, Hp, Wp, C)


def window_pad(H: int, W: int, ws: int) -> Tuple[int, int, int, int]:
    """(pad_b, pad_r, Hp, Wp) exactly as ``WindowAttention.forward`` computes them."""
    pad_r = (ws - W % ws) % ws
    pad_b = (ws - H % ws) % ws
    return pad_b, pad_r, H + pad_b, W + pad_r


# ------------------------------------------------------------------ BART mask
def bart_mask_is_trivial(attention_mask: Optional[torch.Tensor]) -> bool:
    """True when the additive 4-D mask ``_prepare_4d_attention_mask`` would build is all zeros,
    i.e. the 2-D mask has no zero entry. ``XVLAModel.forward_vlm`` always passes
    ``cat(ones(50), ones(32))`` (``_merge_input_ids_with_image_features``), so the 12 ``add``
    launches and the 0.17 MB mask upload are dropped; ``x + 0.0 == x`` for finite x (exact)."""
    if attention_mask is None:
        return True
    return bool(torch.all(attention_mask != 0).item())


# ------------------------------------------------------------- DRAM / launch tally
def fused_launches_per_chunk(num_steps: int = 1) -> Dict[str, int]:
    """Per action chunk launch counts as written in the fused code (default knobs, bfp8 fc2):
    documentation for DEVICE_VALIDATION.md; NOT a measurement."""
    stack_block = 10  # LN, qkv, split, SDPA, concat, dit(proj), LN, fc1(gelu), fc2, add
    stack = 2 + 24 * stack_block  # + tilize, untilize
    bart_layer = 9  # qkv, split, SDPA, concat, out, LN(res), fc1(gelu), fc2, LN(res)
    bart = 2 + 1 + 12 * bart_layer  # tilize, untilize, LN_emb
    wattn = 8  # tilize, LN, qkv, split, SDPA, concat, proj, untilize
    ffn = 6  # tilize, LN, fc1(gelu), fc2, add, untilize
    blocks_per_stage = (1, 1, 9, 1)
    cattn = sum(n * channel_attention_launches(s["T"], True) for n, s in zip(blocks_per_stage, DAVIT_STAGES))
    davit = 12 * wattn + 24 * ffn + cattn
    return dict(stack_per_step=stack, stack=stack * num_steps, bart=bart, davit=davit,
                total=stack * num_steps + bart + davit)


# ======================================================================= device side
class DeviceKit:
    """Fused-op wrappers bound to one device + config. Only ``__init__`` and the weight
    helpers touch ``ttnn`` at build time; everything else is called inside the graphs."""

    def __init__(self, ttnn_mod, device, cfg: FusedConfig, log: Callable[[str], None] = print) -> None:
        self.ttnn = ttnn_mod
        self.device = device
        self.cfg = cfg
        self.log = log
        ttnn = ttnn_mod
        self.grid = device.compute_with_storage_grid_size()
        # rf-detr recipe (ran on this p150a): 8x4x4 / 2x2 for the wide matmuls (qkv, fc1),
        # 4x4x4 / 2x2 for the fused matmul+residual (default 8x8x8 clashed L1 CBs there).
        self.mm_config = ttnn.MinimalMatmulConfig(
            M_block_size=8, K_block_size=4, N_block_size=4, subblock_h=2, subblock_w=2,
            compute_with_storage_grid_size=self.grid,
        )
        self.dit_config = ttnn.MinimalMatmulConfig(
            M_block_size=4, K_block_size=4, N_block_size=4, subblock_h=2, subblock_w=2,
            compute_with_storage_grid_size=self.grid,
        )
        # None = the op's own default (HiFi2 + fp32 accumulation + packer L1 acc).
        self.mm_kernel_config = None
        if cfg.mm_fidelity or cfg.mm_fp32acc:
            fidelity = getattr(ttnn.MathFidelity, cfg.mm_fidelity or "HiFi2")
            fp32 = (cfg.mm_fp32acc == "1") if cfg.mm_fp32acc else True
            self.mm_kernel_config = ttnn.init_device_compute_kernel_config(
                device.arch(), math_fidelity=fidelity, fp32_dest_acc_en=fp32, packer_l1_acc=True
            )
        self._sdpa_pc: Dict[int, Any] = {}
        self._ones: Dict[int, Any] = {}
        self._zero_masks: Dict[Tuple[int, int, int], Any] = {}

    # ------------------------------------------------------------- weights
    def weight(self, w_in_out: torch.Tensor, bf8: bool = False):
        """2-D ``[K, N]`` weight (already transposed) -> TILE tensor on device."""
        ttnn = self.ttnn
        return ttnn.from_torch(
            w_in_out.detach().to(torch.bfloat16).contiguous(),
            dtype=ttnn.bfloat8_b if bf8 else ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
        )

    def bias_row(self, b: torch.Tensor):
        """1-D bias -> ``[1, N]`` bf16 TILE (minimal_matmul / dit want a row; ttnn.linear too)."""
        ttnn = self.ttnn
        return ttnn.from_torch(
            b.detach().to(torch.bfloat16).reshape(1, -1).contiguous(),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
        )

    def vec(self, v: torch.Tensor):
        """1-D LayerNorm weight/bias exactly as the legacy path uploads them."""
        ttnn = self.ttnn
        return ttnn.from_torch(
            v.detach().to(torch.bfloat16).contiguous(),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
        )

    def ones_row(self, n: int):
        """``[1, n]`` bf16 ones in DRAM: the dit scale vector. The fused kernel compiles one
        TensorAccessor type for both addcmul inputs, so this must live in the same buffer type
        as the residual (all residuals here are DRAM interleaved)."""
        if n not in self._ones:
            ttnn = self.ttnn
            self._ones[n] = ttnn.from_torch(
                torch.ones(1, n, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return self._ones[n]

    def zero_mask(self, batch: int, rows: int = TILE, cols: int = TILE):
        """``[batch, 1, rows, cols]`` bf16 zeros in DRAM: the additive mask that lets
        ``scale_mask_softmax_in_place`` fuse the scale. This tree's softmax validate refuses a
        scale without a mask (softmax_device_operation.cpp: "Scale value must not be set when
        mask is not present" -- hit on the p150a, 2026-09-13) and, with the default program
        config, wants ``mask.padded_shape[0] == input.padded_shape[0]`` and 1 in the intermediate
        dims. Adding 0.0 is exact; one constant per (B, rows, cols), 6 KB for the [3, 1, 32, 32]
        channel-attention scores."""
        key = (int(batch), int(rows), int(cols))
        if key not in self._zero_masks:
            ttnn = self.ttnn
            self._zero_masks[key] = ttnn.from_torch(
                torch.zeros(key[0], 1, key[1], key[2], dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return self._zero_masks[key]

    def scale_softmax_(self, scores, scale: float):
        """In-place ``softmax(scores * scale)`` over the last dim of a 4-D TILE ``[B, H, R, C]``
        tensor as ONE launch: ``scale_mask_softmax_in_place(scores, scale, zeros[B, 1, R, C],
        numeric_stable=True)`` (numeric_stable as ``ttnn.softmax``, which the legacy chain uses;
        this op's own default is False)."""
        ttnn = self.ttnn
        b, _, r, c = (int(d) for d in scores.padded_shape)
        return ttnn.scale_mask_softmax_in_place(scores, float(scale), self.zero_mask(b, r, c), numeric_stable=True)

    # ------------------------------------------------------------------ I/O
    def host_rm(self, t: torch.Tensor):
        """Host bf16 ROW_MAJOR tensor (memcpy, no host tilize): what traces copy into their
        persistent input, and what ``upload`` sends when ``rm_io`` is on."""
        ttnn = self.ttnn
        return ttnn.from_torch(t.to(torch.bfloat16).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    def host_tensor(self, t: torch.Tensor):
        """Host tensor in the layout the device graph ingests (RM when rm_io, else TILE)."""
        ttnn = self.ttnn
        if self.cfg.rm_io:
            return self.host_rm(t)
        return ttnn.from_torch(t.to(torch.bfloat16).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def upload(self, t: torch.Tensor):
        """Torch -> device tensor (RM when rm_io else TILE, i.e. the legacy host tilize)."""
        ttnn = self.ttnn
        return ttnn.to_device(self.host_tensor(t), self.device)

    def upload_rm(self, t: torch.Tensor):
        """Torch -> device ROW_MAJOR bf16 tensor regardless of ``rm_io`` (device-side data
        movement such as the window permutes needs ROW_MAJOR)."""
        return self.ttnn.to_device(self.host_rm(t), self.device)

    def free(self, *tensors) -> None:
        """Deallocate device tensors that are still allocated (aliases/views are skipped once
        their buffer is gone)."""
        for t in tensors:
            if t is not None and t.is_allocated():
                self.ttnn.deallocate(t)

    def ingest(self, x):
        """Device input -> TILE (``tilize_with_zero_padding`` on RM inputs; identity on TILE)."""
        ttnn = self.ttnn
        if x.layout == ttnn.ROW_MAJOR_LAYOUT:
            return ttnn.tilize_with_zero_padding(x, use_multicore=True)
        return x

    def egress(self, x_tile, logical_shape: Sequence[int]):
        """TILE -> RM ``logical_shape`` (``untilize_with_unpadding``; ``output_tensor_end`` is
        inclusive, output dim = end + 1) when rm_io; else the TILE tensor (legacy host untilize)."""
        ttnn = self.ttnn
        if not self.cfg.rm_io:
            return x_tile
        return ttnn.untilize_with_unpadding(x_tile, output_tensor_end=[int(d) - 1 for d in logical_shape])

    def readback(self, x_dev) -> torch.Tensor:
        return self.ttnn.to_torch(x_dev)

    # -------------------------------------------------------------- compute
    def layer_norm(self, x, w, b, eps: float, residual=None):
        """``LN(x [+ residual])``. eps: the module's eps when ``ln_eps_ref`` else ttnn's default
        (1e-12, what the legacy path uses -- the legacy port never passed epsilon). The residual
        is fused via ``residual_input_tensor`` when ``ln_residual`` (bf16-rounding class), else
        added with ``ttnn.add`` first (the legacy math, for the exact A/B)."""
        ttnn = self.ttnn
        kw: Dict[str, Any] = dict(weight=w, bias=b)
        if self.cfg.ln_eps_ref:
            kw["epsilon"] = float(eps)
        if residual is None:
            return ttnn.layer_norm(x, **kw)
        if self.cfg.ln_residual:
            kw["residual_input_tensor"] = residual
            return ttnn.layer_norm(x, **kw)
        summed = ttnn.add(residual, x)
        y = ttnn.layer_norm(summed, **kw)
        ttnn.deallocate(summed)
        return y

    def linear(self, h, w, b, gelu: bool = False):
        """``h @ w + b`` (+ exact erf GELU). ``minimal_mm``: minimal_matmul (+ separate
        ``ttnn.gelu``: fusing gelu into the kernel was slower in rf-detr). Otherwise
        ``ttnn.linear`` (its ``activation="gelu"`` is the exact erf GELU, unary_op_utils.cpp)."""
        ttnn = self.ttnn
        if self.cfg.minimal_mm:
            y = ttnn.experimental.minimal_matmul(
                h, w, bias_tensor=b, config=self.mm_config, compute_kernel_config=self.mm_kernel_config
            )
            if gelu:
                y2 = ttnn.gelu(y, fast_and_approximate_mode=False)
                ttnn.deallocate(y)
                return y2
            return y
        if gelu:
            return ttnn.linear(h, w, bias=b, activation="gelu")
        return ttnn.linear(h, w, bias=b)

    def can_fuse_residual(self, w) -> bool:
        """dit fusion needs residual dtype (bf16 activations) == weight dtype (program factory
        line 334: ``ternary_a_data_format == in1_data_format``)."""
        return bool(self.cfg.dit) and w.dtype == self.ttnn.bfloat16

    def linear_residual(self, h, w, b, residual):
        """``residual + (h @ w + b)`` as ONE ``dit_minimal_matmul_addcmul_fused`` call (scalar 1.0,
        ones scale vector) when the weight is bf16, else ``linear`` + ``add``. Returns a new
        tensor; like every kit method it never frees its inputs (the caller owns ``h`` and
        ``residual`` -- a graph must not free a trace's persistent input)."""
        ttnn = self.ttnn
        if self.can_fuse_residual(w):
            n = int(w.shape[-1])
            return ttnn.experimental.dit_minimal_matmul_addcmul_fused(
                h, w, 1.0, residual, self.ones_row(n),
                bias_tensor=b, config=self.dit_config, compute_kernel_config=self.mm_kernel_config,
            )
        y = self.linear(h, w, b)
        out = ttnn.add(residual, y)
        ttnn.deallocate(y)
        return out

    def sdpa_program_config(self, seq_len: int):
        ttnn = self.ttnn
        chunk = sdpa_chunk(seq_len)
        if chunk not in self._sdpa_pc:
            self._sdpa_pc[chunk] = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.grid, q_chunk_size=chunk, k_chunk_size=chunk,
                exp_approx_mode=False,
            )
        return self._sdpa_pc[chunk]

    def attention(self, qkv, num_heads: int, seq_len: int, scale: float, mask=None):
        """Fused-qkv ``[B, S, 3C]`` (TILE, S tile-padded) -> context ``[B, S, C]``.

        ``sdpa`` on and no mask: split(transpose_key=False) -> scaled_dot_product_attention
        (is_causal=False -- the default is True in this tree; scale fused; NO attn_mask: the
        program factory generates the -inf mask for the padded key columns itself, and an explicit
        zero-padded mask was numerically wrong on Blackhole in the rf-detr port) -> concat heads.
        Otherwise the legacy chain (split(k transposed), matmul, multiply, [add mask], softmax,
        matmul, concat) so that ``TT_FUSED_SDPA=0`` isolates the kernel in an A/B."""
        ttnn = self.ttnn
        if self.cfg.sdpa and mask is None:
            q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
                qkv, num_heads=num_heads, transpose_key=False
            )
            ctx = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, is_causal=False, scale=float(scale), program_config=self.sdpa_program_config(seq_len)
            )
            ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(v)
        else:
            q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(qkv, num_heads=num_heads)
            scores = ttnn.matmul(q, k)
            ttnn.deallocate(q); ttnn.deallocate(k)
            scores = ttnn.multiply(scores, float(scale))
            if mask is not None:
                scores = ttnn.add(scores, mask)
            probs = ttnn.softmax(scores, dim=-1)
            ttnn.deallocate(scores)
            ctx = ttnn.matmul(probs, v)
            ttnn.deallocate(probs); ttnn.deallocate(v)
        out = ttnn.transformer.concatenate_heads(ctx)
        ttnn.deallocate(ctx)
        return out


class TracedGraph:
    """One device graph ``graph(x_dev) -> y_dev`` captured as a metal trace with a persistent
    input buffer.

    First call: ``to_device(host)`` allocates the persistent input, an eager run populates the
    program cache (capture must not JIT), then ``begin_trace_capture`` / graph / ``end_trace_capture``
    records the graph and keeps its output tensor alive; the same call then replays the trace for
    the real result. Later calls: ``copy_host_to_device_tensor`` into the persistent input,
    ``execute_trace(blocking=False)``, ``to_torch`` of the persistent output (blocks). Eager
    fallback (same graph, no trace) when tracing is disabled, when the capture raised (the
    device is left usable: the capture is ended/released and the failure logged once), or when
    a call arrives with a shape other than the captured one (shapes are locked per process:
    ``XVLA_LANG_TOKENS``, 3 views, batch 1).

    Trace hazard (BRIEF §1A / DEVICE_VALIDATION.md §5): weights, constants and the lazily
    created ``ones_row`` vectors are allocated before either capture (DaViT eager pass), and
    kit methods never free their inputs, so no graph can free a trace's persistent buffer.
    The two traces are NOT independent, though: the block-stack trace's ``_in``/``_out`` (and
    its warm-run temporaries) are allocated AFTER the BART trace is captured, so BART's scratch
    addresses may alias them (the allocator only logs 'Allocating device buffers is unsafe due
    to the existence of an active trace'). This is harmless as used: ``_replay`` ends with a
    blocking ``to_torch(self._out)`` before the other trace's input is refreshed with
    ``copy_host_to_device_tensor`` and the stack trace is captured while BART's buffers are
    live (its scratch avoids them). Argued, not measured -- §5 gives the fallback (capture the
    stack trace first, or ``allocate_tensor_on_device`` inputs).
    """

    def __init__(self, ttnn_mod, device, graph: Callable[[Any], Any], name: str,
                 enabled: bool, log: Callable[[str], None] = print) -> None:
        self.ttnn = ttnn_mod
        self.device = device
        self.graph = graph
        self.name = name
        self.enabled = bool(enabled)
        self.log = log
        self.trace_id = None
        self.failed = False
        self.captured_shape: Optional[Tuple[int, ...]] = None
        self._in = None
        self._out = None
        self.executions = 0
        self.eager_runs = 0

    # ---------------------------------------------------------------- status
    @property
    def captured(self) -> bool:
        return self.trace_id is not None

    def status(self) -> Dict[str, Any]:
        return dict(name=self.name, trace_enabled=self.enabled, trace_captured=self.captured,
                    trace_failed=self.failed, captured_shape=self.captured_shape,
                    trace_executions=self.executions, eager_runs=self.eager_runs)

    # ------------------------------------------------------------------ run
    def run(self, host) -> torch.Tensor:
        """``host``: host ttnn tensor (RM or TILE) -> torch result of ``graph``."""
        if not self.enabled or self.failed:
            return self._eager(host)
        shape = tuple(int(d) for d in host.shape)
        if self.trace_id is None:
            try:
                self._capture(host)
            except Exception as e:  # noqa: BLE001
                self._abort_capture(e)
                return self._eager(host)
        elif shape != self.captured_shape:
            self.log(f"[xvla.fused] {self.name}: input shape {shape} != captured {self.captured_shape}; eager")
            return self._eager(host)
        return self._replay(host)

    def _eager(self, host) -> torch.Tensor:
        ttnn = self.ttnn
        x = ttnn.to_device(host, self.device)
        y = self.graph(x)
        out = ttnn.to_torch(y)
        ttnn.deallocate(y)
        if x.is_allocated():
            ttnn.deallocate(x)
        self.eager_runs += 1
        return out

    def _capture(self, host) -> None:
        ttnn = self.ttnn
        dev = self.device
        self._in = ttnn.to_device(host, dev)
        # Eager warm run: compiles/caches every program of the graph (capture must be cache hits).
        y = self.graph(self._in)
        ttnn.synchronize_device(dev)
        ttnn.deallocate(y)
        tid = ttnn.begin_trace_capture(dev)
        try:
            self._out = self.graph(self._in)
            ttnn.end_trace_capture(dev, tid)
        except Exception:
            try:
                ttnn.end_trace_capture(dev, tid)
            except Exception:  # noqa: BLE001
                pass
            try:
                ttnn.release_trace(dev, tid)
            except Exception:  # noqa: BLE001
                pass
            raise
        ttnn.synchronize_device(dev)
        self.trace_id = tid
        self.captured_shape = tuple(int(d) for d in host.shape)
        self.log(f"[xvla.fused] {self.name}: trace captured for input {self.captured_shape}")

    def _abort_capture(self, err: BaseException) -> None:
        self.failed = True
        self.trace_id = None
        self._out = None
        self.log(f"[xvla.fused] {self.name}: trace capture FAILED ({type(err).__name__}: {err}); running eagerly")

    def _replay(self, host) -> torch.Tensor:
        ttnn = self.ttnn
        ttnn.copy_host_to_device_tensor(host, self._in)
        ttnn.execute_trace(self.device, self.trace_id, blocking=False)
        self.executions += 1
        return ttnn.to_torch(self._out)

    def release(self) -> None:
        if self.trace_id is not None:
            try:
                self.ttnn.release_trace(self.device, self.trace_id)
            except Exception:  # noqa: BLE001
                pass
            self.trace_id = None
