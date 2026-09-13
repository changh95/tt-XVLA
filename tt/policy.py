# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TT-NN-backed X-VLA policy loader.

The benchmark harness routes `--backend ttnn` to `load_policy_ttnn()`. This
file is the only place where the optimization branch diverges from the
upstream lerobot reference; the harness itself stays frozen.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

# Make ttnn imports route to the xvla tt-metal tree. Must happen before the
# first `import ttnn` anywhere in the process.
_HERE = Path(__file__).resolve().parent
_TTNN_ENV_FILE = _HERE / "ttnn_env.py"
_spec = importlib.util.spec_from_file_location("xvla_ttnn_env", str(_TTNN_ENV_FILE))
_mod = importlib.util.module_from_spec(_spec)
sys.modules["xvla_ttnn_env"] = _mod
_spec.loader.exec_module(_mod)
_mod.install()

import torch  # noqa: E402

# Bootstrap the lerobot import patches.
_BOOTSTRAP_FILE = _HERE.parent / "benchmark" / "lerobot_bootstrap.py"
_bspec = importlib.util.spec_from_file_location("xvla_lerobot_bootstrap", str(_BOOTSTRAP_FILE))
_bmod = importlib.util.module_from_spec(_bspec)
sys.modules["xvla_lerobot_bootstrap"] = _bmod
_bspec.loader.exec_module(_bmod)
_bmod.install()

_TTNN_DEVICE = None

# TT_FUSED knobs (tt/fused.py), read ONCE per process by `fused_config()`; the DeviceKit and
# the fused modules built by `load_policy_ttnn` (for `fused_status()`).
_FUSED_CFG = None
_FUSED_KIT = None
_FUSED_MODULES: dict = {}


def _load_module_file(name: str, filename: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(_HERE / filename))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def fused_config():
    """The ``TT_FUSED*`` knobs, read from the environment exactly once (first call = model
    build). ``TT_FUSED`` unset or 1 -> the fused / traced path (the default since the p150a
    validation of 2026-09-13); ``TT_FUSED=0`` -> ``enabled=False`` -> the legacy path, bit-for-bit."""
    global _FUSED_CFG
    if _FUSED_CFG is None:
        fused_mod = _load_module_file("xvla_ttnn_fused", "fused.py")
        _FUSED_CFG = fused_mod.FusedConfig.from_env()
    return _FUSED_CFG


def _open_device():
    """Open (and cache) the Blackhole device.

    The chip id comes from ``TT_DEVICE_ID`` (default 0 -- the only chip a
    tt-model container sees). The original port hard-coded id 3 for the
    author's multi-card box; export ``TT_DEVICE_ID=3`` to reproduce that.
    A caller may also pre-seed ``_TTNN_DEVICE`` with an already-open device.

    Fused path (default; ``TT_FUSED`` unset or 1): the device is opened with
    ``trace_region_size`` (v0.71's ``DEFAULT_TRACE_REGION_SIZE`` is 0, so metal traces need it)
    and the program cache is enabled (trace capture must hit the cache). ``TT_FUSED=0``
    (legacy): default kwargs, as before.
    """
    global _TTNN_DEVICE
    if _TTNN_DEVICE is None:
        import ttnn

        cfg = fused_config()
        device_id = int(os.environ.get("TT_DEVICE_ID", "0"))
        if cfg.enabled:
            _TTNN_DEVICE = ttnn.open_device(device_id=device_id, trace_region_size=cfg.trace_region_mb << 20)
        else:
            _TTNN_DEVICE = ttnn.open_device(device_id=device_id)
    return _TTNN_DEVICE


def fused_status() -> dict:
    """Knobs + per-module trace state (for the server log, /info and the warm-up gate)."""
    cfg = fused_config()
    st = {"config": cfg.describe(), "modules": {}}
    for name, mod in _FUSED_MODULES.items():
        st["modules"][name] = mod.status() if hasattr(mod, "status") else {}
    if cfg.enabled:
        traced = [m for m in st["modules"].values() if m.get("trace_enabled")]
        st["all_traces_captured"] = bool(traced) and all(m.get("trace_captured") for m in traced)
    return st


def load_policy_ttnn(weights: Path):
    """Load X-VLA with the optimized backend.

    Iter1-iter16: see results.tsv. Best so far: iter10 = 118.4 fps, PCC 99.98%
        (24 SoftPromptedTransformer blocks on chip, MLP weights bfp8_b).
    Iter17 (current): also offload Florence-2's 12-layer BART encoder to
        the Blackhole device. This is ~75% of the remaining torch CPU
        time. Each layer is post-LN BART (fused QKV, attention with
        additive mask, output proj, FFN). Embedding lookups stay on
        torch (small integer-indexed tables).
    TT_FUSED (default 1 since the p150a validation of 2026-09-13; TT_FUSED=0 =
        the legacy classes above, bit-for-bit): the `*Fused` classes -- SDPA,
        minimal_matmul, matmul+residual / LN+residual fusions, ROW_MAJOR I/O with
        device tilize/untilize, DaViT window permutes on device, and the block
        stack + BART encoder captured as metal traces (tt/fused.py,
        DEVICE_VALIDATION.md "Results"). Measured: 89 ms vs 177 ms per 1-step
        action chunk, PCC vs fp32 0.999982 (legacy 0.999981).
    """
    global _FUSED_KIT
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

    torch.set_grad_enabled(False)
    config = PreTrainedConfig.from_pretrained(str(weights))
    # config.json says device: cuda; lerobot already falls back to CPU with a
    # warning. Say it explicitly so from_pretrained's `.to(config.device)` is
    # deterministic: the torch side of this port is CPU, the rest is ttnn.
    config.device = "cpu"
    config.dtype = "bfloat16"
    config.num_denoising_steps = 1
    policy = XVLAPolicy.from_pretrained(str(weights), config=config)
    policy.eval()

    device = _open_device()
    cfg = fused_config()
    stack_mod = _load_module_file("xvla_ttnn_block_stack", "ttnn_block_stack.py")
    bart_mod = _load_module_file("xvla_ttnn_bart_encoder", "ttnn_bart_encoder.py")
    davit_mod = _load_module_file("xvla_ttnn_davit_ffn", "ttnn_davit_ffn.py")
    cattn_mod = _load_module_file("xvla_ttnn_davit_channel_attn", "ttnn_davit_channel_attn.py")
    wattn_mod = _load_module_file("xvla_ttnn_davit_window_attn", "ttnn_davit_window_attn.py")
    transformer = policy.model.transformer
    lm = policy.model.vlm.language_model
    vision_tower = policy.model.vlm.vision_tower

    if cfg.enabled:
        # Fused path (default; TT_FUSED=0 = legacy): fused ops + metal traces, see tt/fused.py.
        # Legacy numerics are not touched; every fused class is a separate implementation.
        import ttnn

        fused_mod = _load_module_file("xvla_ttnn_fused", "fused.py")
        device.enable_program_cache()  # idempotent; capture needs cache hits
        kit = fused_mod.DeviceKit(ttnn, device, cfg, log=lambda m: print(m, flush=True))
        _FUSED_KIT = kit
        out_rows = fused_mod.stack_out_rows(int(policy.config.chunk_size), cfg.stack_out_rows)
        transformer.blocks = stack_mod.TTNNTransformerBlockStackFused(
            transformer.blocks, device, num_heads=transformer.blocks[0].attn.num_heads, kit=kit,
            out_rows=out_rows, log=kit.log,
        )
        lm.model.encoder = bart_mod.TTNNBartEncoderFused(lm.model.encoder, device, kit, log=kit.log).to(torch.bfloat16)
        davit_mod.swap_davit_ffns(vision_tower, device, kit=kit)
        cattn_mod.swap_davit_channel_attns(vision_tower, device, kit=kit)
        wattn_mod.swap_davit_window_attns(vision_tower, device, kit=kit)
        _FUSED_MODULES.clear()
        _FUSED_MODULES["block_stack"] = transformer.blocks
        _FUSED_MODULES["bart_encoder"] = lm.model.encoder
        print(f"[xvla.fused] TT_FUSED config (default on; TT_FUSED=0 = legacy): {cfg.describe()}", flush=True)
        return policy

    # ---- legacy path (TT_FUSED=0): unchanged, the published 2026-09-12 behaviour -----------
    transformer.blocks = stack_mod.TTNNTransformerBlockStack(
        transformer.blocks, device, num_heads=transformer.blocks[0].attn.num_heads
    )

    # Replace Florence-2 BART encoder with on-device version.
    lm.model.encoder = bart_mod.TTNNBartEncoder(lm.model.encoder, device).to(torch.bfloat16)

    # Iter18: replace the FFN inside every DaViT SpatialBlock and
    # ChannelBlock with an on-device PreNorm+MLP+residual unit. This
    # touches the largest weight matrices in the vision tower (each
    # FFN's fc1/fc2 is dim x 4*dim).
    davit_mod.swap_davit_ffns(vision_tower, device)

    # Iter19v2: ChannelAttention on chip with 4D-only ttnn ops (split,
    # reshape, permute, transpose, matmul) — no intermediate torch
    # round-trips this time.
    cattn_mod.swap_davit_channel_attns(vision_tower, device)

    # Iter20: WindowAttention on chip (qkv linear + multi-head attention +
    # proj linear). Pad/window_partition/window_reverse stay on torch
    # because they are 6D reshape+permute ops; ttnn doesn't support 6D
    # ergonomically, but the matmul-heavy interior moves on-chip.
    wattn_mod.swap_davit_window_attns(vision_tower, device)

    return policy
