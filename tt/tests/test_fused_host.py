# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests for the TT_FUSED=1 path (BRIEF §4 item 3): torch only, NO device, NO ttnn
tensors. Every exact reformulation the fused device code relies on is checked against the
reference math, the constant/geometry tables are validated, and the knob plumbing is tested
(TT_FUSED unset -> legacy classes).

Run with the host python of the model's tt-metal tree (torch 2.7.1, pytest 9):

    cd models/xvla-base-p150
    TREE=/home/deepgadget/experiments/base/tt-metal
    TT_METAL_HOME=$TREE PYTHONPATH=code $TREE/python_env/bin/python -m pytest code/tt/tests/test_fused_host.py -q

or as a plain script (asserts): ``PYTHONPATH=code python code/tt/tests/test_fused_host.py``.

Exactness classes used below: ``torch.equal`` where the reformulation is pure data movement
or adds exact zeros; ``<= 1e-9`` in float64 where only the floating-point accumulation ORDER
differs (same math, e.g. ``(P @ v^T)^T == v @ P^T``).
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_CODE = Path(__file__).resolve().parents[2]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from tt import fused  # noqa: E402

torch.manual_seed(0)


# ------------------------------------------------------------------------ knobs
def test_exact_only_configuration():
    """The knob set the device pass used for the bit-identity check against legacy (DEVICE_VALIDATION
    §3.1, measured bit-identical on the p150a 2026-09-13): only the exact levers stay on (RM I/O, mask
    drop, traces, output-row unpad); every bf16-rounding or precision-affecting lever is off -- since
    the default flip that includes the three kernel-swap knobs that are now on by default."""
    cfg = fused.FusedConfig.from_env({"TT_FUSED": "1", "TT_FUSED_SDPA": "0", "TT_FUSED_DIT": "0",
                                      "TT_FUSED_CATTN": "0", "TT_FUSED_LN_RESIDUAL": "0",
                                      "TT_FUSED_MINIMAL_MM": "0", "TT_FUSED_FC2_BF16": "0",
                                      "TT_FUSED_WINDOWS_ON_DEVICE": "0"})
    assert cfg.enabled and cfg.rm_io and cfg.trace
    assert not (cfg.sdpa or cfg.dit or cfg.cattn or cfg.ln_residual or cfg.minimal_mm or cfg.fc2_bf16
                or cfg.windows_on_device or cfg.ln_eps_ref)


def test_knob_unset_is_fused_and_zero_is_legacy():
    """Default flipped after the p150a validation (2026-09-13): unset -> the validated fused
    set; TT_FUSED=0 -> legacy (every sub-knob irrelevant then). The empty string counts as
    unset (tt-model passes `--env K=` for blank values)."""
    cfg = fused.FusedConfig.from_env({})
    assert cfg.enabled and cfg.trace and cfg.rm_io and cfg.sdpa and cfg.dit and cfg.cattn and cfg.ln_residual
    assert cfg.minimal_mm and cfg.fc2_bf16 and cfg.windows_on_device and not cfg.ln_eps_ref
    assert fused.FusedConfig.from_env({"TT_FUSED": ""}).enabled is True
    for off in ("0", "false", "no", "off"):
        c = fused.FusedConfig.from_env({"TT_FUSED": off, "TT_FUSED_SDPA": "1"})
        assert c.enabled is False and c == fused.FusedConfig(enabled=False)


def test_knob_on_defaults_and_parsing():
    cfg = fused.FusedConfig.from_env({"TT_FUSED": "1"})
    assert cfg == fused.FusedConfig.from_env({}), "TT_FUSED=1 is the default"
    assert cfg.enabled and cfg.trace and cfg.rm_io and cfg.sdpa and cfg.dit and cfg.cattn and cfg.ln_residual
    assert cfg.fc2_bf16 and cfg.minimal_mm and cfg.windows_on_device and not cfg.ln_eps_ref
    assert cfg.trace_region_mb == 64 and cfg.mm_fidelity == "" and cfg.mm_fp32acc == "" and cfg.stack_out_rows == -1
    # the 3.2 "default fused" configuration of the validation (kernel swaps off) is still reachable
    c32 = fused.FusedConfig.from_env({"TT_FUSED_MINIMAL_MM": "0", "TT_FUSED_FC2_BF16": "0", "TT_FUSED_WINDOWS_ON_DEVICE": "0"})
    assert c32.enabled and not (c32.minimal_mm or c32.fc2_bf16 or c32.windows_on_device)
    cfg = fused.FusedConfig.from_env({
        "TT_FUSED": "1", "TT_FUSED_TRACE": "0", "TT_FUSED_TRACE_REGION_MB": "96", "TT_FUSED_FC2_BF16": "1",
        "TT_FUSED_MINIMAL_MM": "1", "TT_FUSED_MM_FIDELITY": "HiFi4", "TT_FUSED_MM_FP32ACC": "0",
        "TT_FUSED_WINDOWS_ON_DEVICE": "1", "TT_FUSED_STACK_OUT_ROWS": "0", "TT_FUSED_LN_EPS_REF": "1",
        "TT_FUSED_LN_RESIDUAL": "0",
    })
    assert not cfg.ln_residual
    assert not cfg.trace and cfg.trace_region_mb == 96 and cfg.fc2_bf16 and cfg.minimal_mm
    assert cfg.mm_fidelity == "HiFi4" and cfg.mm_fp32acc == "0" and cfg.windows_on_device
    assert cfg.stack_out_rows == 0 and cfg.ln_eps_ref
    for bad in ({"TT_FUSED": "1", "TT_FUSED_MM_FIDELITY": "HiFi9"}, {"TT_FUSED": "1", "TT_FUSED_MM_FP32ACC": "yes"}):
        try:
            fused.FusedConfig.from_env(bad)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"{bad} should be rejected")
    assert set(cfg.describe()) == set(fused.FusedConfig.__dataclass_fields__)


def test_policy_reads_knob_once_and_zero_is_legacy():
    """``tt.policy.fused_config()`` with TT_FUSED=0 -> enabled False (the legacy classes are
    selected in ``load_policy_ttnn``); unset -> the fused default. Read once per process. Needs
    the tree's ttnn/transformers for ``import tt.policy``."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        return  # torch-only venv: covered by test_swap_dispatch below
    os.environ["TT_FUSED"] = "0"
    import tt.policy as policy

    policy._FUSED_CFG = None
    try:
        cfg = policy.fused_config()
        assert cfg.enabled is False
        os.environ["TT_FUSED"] = "1"
        assert policy.fused_config() is cfg, "knob must be read once, not per call"
        st = policy.fused_status()
        assert st["config"]["enabled"] is False and st["modules"] == {}
        policy._FUSED_CFG = None
        os.environ.pop("TT_FUSED", None)
        assert policy.fused_config().enabled is True, "unset -> fused default"
    finally:
        os.environ.pop("TT_FUSED", None)
        policy._FUSED_CFG = None


def test_swap_dispatch_legacy_without_kit(monkeypatch=None):
    """The three DaViT swap functions pick the legacy class when ``kit`` is None (TT_FUSED unset)
    and the fused class when a kit is given; the legacy call signature is unchanged."""
    from torch import nn

    from tt import ttnn_davit_channel_attn as cattn, ttnn_davit_ffn as ffn, ttnn_davit_window_attn as wattn

    class Stub(nn.Module):
        made = []

        def __init__(self, *args):
            super().__init__()
            Stub.made.append((type(self).__name__, len(args)))

    class Tower(nn.Module):
        def __init__(self):
            super().__init__()
            dual = nn.Module()
            dual.spatial_block = nn.Module()
            dual.spatial_block.ffn = nn.Identity()
            dual.spatial_block.window_attn = nn.Identity()
            dual.channel_block = nn.Module()
            dual.channel_block.ffn = nn.Identity()
            dual.channel_block.channel_attn = nn.Identity()
            self.blocks = nn.ModuleList([nn.ModuleList([dual])])

    saved = {}
    pairs = [
        (ffn, "TTNNDaViTPreNormFFN", "TTNNDaViTPreNormFFNFused", ffn.swap_davit_ffns, 2),
        (cattn, "TTNNDaViTPreNormChannelAttn", "TTNNDaViTPreNormChannelAttnFused", cattn.swap_davit_channel_attns, 1),
        (wattn, "TTNNDaViTPreNormWindowAttn", "TTNNDaViTPreNormWindowAttnFused", wattn.swap_davit_window_attns, 1),
    ]
    try:
        for mod, legacy, fusedname, _, _ in pairs:
            saved[(mod, legacy)] = getattr(mod, legacy)
            saved[(mod, fusedname)] = getattr(mod, fusedname)
            setattr(mod, legacy, type(legacy, (Stub,), {}))
            setattr(mod, fusedname, type(fusedname, (Stub,), {}))
        for mod, legacy, fusedname, swap, n in pairs:
            Stub.made.clear()
            assert swap(Tower(), device="dev") == n  # legacy signature: (tower, device)
            assert Stub.made == [(legacy, 2)] * n, Stub.made
            Stub.made.clear()
            assert swap(Tower(), "dev", kit=object()) == n
            assert Stub.made == [(fusedname, 3)] * n, Stub.made
    finally:
        for (mod, name), cls in saved.items():
            setattr(mod, name, cls)


# --------------------------------------------------------------- tile arithmetic
def test_tile_pad_and_sdpa_chunks():
    assert [fused.tile_pad(n) for n in (244, 82, 144, 3136, 784, 196, 49, 32, 1)] == [256, 96, 160, 3136, 800, 224, 64, 32, 32]
    assert fused.sdpa_chunk(244) == 256 and fused.sdpa_chunk(82) == 96 and fused.sdpa_chunk(144) == 160
    for s in (244, 82, 144):
        assert fused.sdpa_chunk(s) % 32 == 0 and fused.sdpa_chunk(s) >= s  # validate: chunk % 32 == 0


def test_stack_out_rows():
    assert fused.stack_out_rows(30, -1) == 32
    assert fused.stack_out_rows(30, 0) is None
    assert fused.stack_out_rows(30, 64) == 64
    try:
        fused.stack_out_rows(30, 16)
    except ValueError:
        pass
    else:
        raise AssertionError("rows below chunk_size must be rejected")


def test_davit_stage_geometry():
    for s in fused.DAVIT_STAGES:
        assert s["T"] == s["H"] * s["W"]
        assert s["C"] == s["heads"] * 32 == s["groups"] * 32  # head_dim / channels-per-group tile-aligned
        pad_b, pad_r, Hp, Wp = fused.window_pad(s["H"], s["W"], fused.DAVIT_WINDOW)
        assert Hp % 12 == 0 and Wp % 12 == 0 and (Hp // 12) * (Wp // 12) == s["windows_per_view"]
        # WindowAttention.forward's own formula
        assert pad_r == (12 - s["W"] % 12) % 12 and pad_b == (12 - s["H"] % 12) % 12
    assert [fused.tile_pad(s["T"]) for s in fused.DAVIT_STAGES] == [3136, 800, 224, 64]


# ---------------------------------------------------------- head split / merge
def test_split_heads_matches_reference_layout():
    """ttnn split_query_key_value_and_split_heads (torch model of its docstring) produces the same
    q/k/v as the reference ``reshape(B, S, 3, H, dh).permute(2, 0, 3, 1, 4)`` -- so replacing the
    manual chain by split(transpose_key=False) + SDPA + concat changes no data layout."""
    for b, s, h, dh in ((1, 244, 16, 64), (1, 82, 16, 64), (12, 144, 32, 32), (3, 196, 32, 32)):
        qkv = torch.randn(b, s, 3 * h * dh)
        q, k, v = fused.split_heads_torch(qkv, h, transpose_key=False)
        rq, rk, rv = fused.reference_qkv_split(qkv, h)
        assert torch.equal(q, rq) and torch.equal(k, rk) and torch.equal(v, rv)
        _, kt, _ = fused.split_heads_torch(qkv, h, transpose_key=True)
        assert torch.equal(kt, rk.transpose(-1, -2))
        ctx = torch.randn(b, h, s, dh)
        assert torch.equal(fused.concat_heads_torch(ctx), ctx.transpose(1, 2).reshape(b, s, h * dh))


def test_sdpa_math_is_the_manual_chain():
    """The math SDPA computes (torch's reference kernel) equals the legacy chain
    softmax(q k^T * scale) v -- in float64 to 1e-9 (accumulation order). The device kernel's
    bf16/fp32 numerics are the precision-affecting part gated on hardware."""
    for b, s, h, dh in ((1, 244, 16, 64), (1, 82, 16, 64), (3, 144, 8, 32)):
        q, k, v = (torch.randn(b, h, s, dh, dtype=torch.float64) for _ in range(3))
        scale = float(dh) ** -0.5
        manual = torch.softmax(q @ k.transpose(-1, -2) * scale, dim=-1) @ v
        ref = F.scaled_dot_product_attention(q, k, v, scale=scale)
        assert (manual - ref).abs().max().item() < 1e-9


# ---------------------------------------------------------- channel attention
def _cattn_params(C: int, dtype=torch.float64):
    qkv_w = torch.randn(3 * C, C, dtype=dtype) / math.sqrt(C)
    qkv_b = torch.randn(3 * C, dtype=dtype) * 0.1
    proj_w = torch.randn(C, C, dtype=dtype) / math.sqrt(C)
    proj_b = torch.randn(C, dtype=dtype) * 0.1
    return qkv_w, qkv_b, proj_w, proj_b


def test_channel_attention_restructure_exact_all_stages():
    """Fused graph model (head split, zero-filled padded k rows, q^T k, scaled softmax, v probs^T,
    concat, proj) == reference ChannelAttention for the four DaViT stage shapes (float64, 1e-9:
    same sums in a different order). Also: WITHOUT the padded-row zero fill the padded stages
    (T = 784/196/49) are WRONG -- the fill is load-bearing, stage 0 (T = 3136) needs none."""
    for s in fused.DAVIT_STAGES:
        T, C, g = s["T"], s["C"], s["groups"]
        B = 1
        x = torch.randn(B, T, C, dtype=torch.float64)
        params = _cattn_params(C)
        ref = fused.channel_attention_reference(x, *params, groups=g)
        got = fused.channel_attention_restructured(x, *params, groups=g, pad_rows=True, zero_fill_k=True)
        assert got.shape == ref.shape == (B, T, C)
        err = (got - ref).abs().max().item()
        assert err < 1e-9, (s["stage"], err)
        if fused.tile_pad(T) != T:
            wrong = fused.channel_attention_restructured(x, *params, groups=g, pad_rows=True, zero_fill_k=False)
            assert (wrong - ref).abs().max().item() > 1e-3, "padded rows must corrupt the result without the fill"
            assert fused.channel_attention_launches(T, True) == 13
        else:
            assert fused.channel_attention_launches(T, True) == 12
        # Unpadded model == padded+filled model (the extra contraction terms are exact zeros).
        unpadded = fused.channel_attention_restructured(x, *params, groups=g, pad_rows=False)
        assert (unpadded - got).abs().max().item() < 1e-9


def test_channel_attention_probs_transpose_identity():
    """``(P @ v^T)^T == v @ P^T`` (the fused path's second matmul) -- exact in float64 up to order."""
    P = torch.softmax(torch.randn(3, 32, 32, 32, dtype=torch.float64), dim=-1)
    v = torch.randn(3, 32, 224, 32, dtype=torch.float64)
    a = (P @ v.transpose(-1, -2)).transpose(-1, -2)
    b = v @ P.transpose(-1, -2)
    assert (a - b).abs().max().item() < 1e-12


# ---------------------------------------------------------- window partition
def test_window_partition_4d_equals_reference_all_stages():
    ws = fused.DAVIT_WINDOW
    for s in fused.DAVIT_STAGES:
        B, C = 3, 16
        pad_b, pad_r, Hp, Wp = fused.window_pad(s["H"], s["W"], ws)
        x4p = torch.randn(B, Hp, Wp, C)
        ref = fused.window_partition_ref(x4p, ws).view(-1, ws * ws, C)
        got = fused.window_partition_4d(x4p, ws)
        assert got.shape == ref.shape == (B * s["windows_per_view"], ws * ws, C)
        assert torch.equal(got, ref)
        back = fused.window_reverse_4d(got, B, ws, Hp, Wp)
        assert torch.equal(back, x4p)
        assert torch.equal(back, fused.window_reverse_ref(ref.view(-1, ws, ws, C), B, ws, Hp, Wp))


def test_window_attention_data_path_roundtrip():
    """pad -> partition_4d -> (identity attention) -> reverse_4d -> unpad == identity, i.e. the
    device data path around the attention is exact (the residual add sees the same rows)."""
    ws = fused.DAVIT_WINDOW
    for s in fused.DAVIT_STAGES:
        B, C, H, W = 3, 8, s["H"], s["W"]
        x = torch.randn(B, H * W, C)
        pad_b, pad_r, Hp, Wp = fused.window_pad(H, W, ws)
        x4p = F.pad(x.view(B, H, W, C), (0, 0, 0, pad_r, 0, pad_b))
        win = fused.window_partition_4d(x4p, ws)
        r4 = fused.window_reverse_4d(win, B, ws, Hp, Wp)[:, :H, :W, :]
        assert torch.equal(r4.reshape(B, H * W, C), x)


# ------------------------------------------------------------------ BART mask
def _prepare_4d_attention_mask_like_transformers(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """transformers.modeling_attn_mask_utils._prepare_4d_attention_mask (no transformers import)."""
    bsz, src_len = mask.shape
    expanded = mask[:, None, None, :].expand(bsz, 1, src_len, src_len).to(dtype)
    inverted = 1.0 - expanded
    return inverted.masked_fill(inverted.to(torch.bool), torch.finfo(dtype).min)


def test_bart_all_ones_mask_add_is_exact_noop():
    ones = torch.ones(1, 82)
    assert fused.bart_mask_is_trivial(ones) and fused.bart_mask_is_trivial(None)
    m4 = _prepare_4d_attention_mask_like_transformers(ones, torch.bfloat16)
    assert m4.shape == (1, 1, 82, 82) and torch.equal(m4, torch.zeros_like(m4))
    scores = torch.randn(1, 16, 82, 82).to(torch.bfloat16)
    assert torch.equal(scores + m4, scores)  # x + 0.0 == x for finite x (bf16 too)
    holed = ones.clone()
    holed[0, 40] = 0
    assert not fused.bart_mask_is_trivial(holed)
    m4h = _prepare_4d_attention_mask_like_transformers(holed, torch.float32)
    assert (m4h[..., 40] == torch.finfo(torch.float32).min).all()


# ------------------------------------------------------- fused-op algebra models
def test_dit_ones_vector_is_plain_residual_add():
    """dit_minimal_matmul_addcmul_fused: out = r + 1.0 * (h @ W + b) * ones == r + (h @ W + b);
    multiplying by 1.0 and by a ones row is exact in every float format."""
    h = torch.randn(1, 244, 1024).to(torch.bfloat16)
    W = torch.randn(1024, 1024).to(torch.bfloat16) * 0.02
    b = torch.randn(1, 1024).to(torch.bfloat16)
    r = torch.randn(1, 244, 1024).to(torch.bfloat16)
    y = (h.float() @ W.float() + b.float()).to(torch.bfloat16)
    assert torch.equal(r + 1.0 * y * torch.ones(1, 1024, dtype=torch.bfloat16), r + y)


def test_layer_norm_residual_fusion_is_the_same_math():
    """layer_norm(a, residual_input_tensor=x) == LN(x + a) (post-LN BART); addition commutes exactly."""
    a = torch.randn(1, 82, 1024)
    x = torch.randn(1, 82, 1024)
    w, b = torch.randn(1024), torch.randn(1024)
    assert torch.equal(F.layer_norm(a + x, (1024,), w, b, 1e-5), F.layer_norm(x + a, (1024,), w, b, 1e-5))


def test_stack_output_rows_cover_the_consumed_slice():
    """SoftPromptedTransformer.forward reads x[:, :chunk_size] (30 rows) only; the traced stack
    unpads to 32 rows: the slice is unchanged."""
    S, C, chunk = 244, 1024, 30
    x = torch.randn(1, S, C)
    rows = fused.stack_out_rows(chunk, -1)
    assert rows == 32 and torch.equal(x[:, :rows][:, :chunk], x[:, :chunk])


# ----------------------------------------------- channel attention buffer lifetimes
class _StubBuf:
    def __init__(self):
        self.alive = True


class _StubTensor:
    """Device-tensor stand-in: ``buf`` is the (possibly shared) device buffer."""

    def __init__(self, shape, buf=None):
        self.shape = tuple(shape)
        self.buf = buf if buf is not None else _StubBuf()

    @property
    def padded_shape(self):
        return tuple(-(-int(d) // 32) * 32 if i >= len(self.shape) - 2 else int(d) for i, d in enumerate(self.shape))

    def is_allocated(self):
        return self.buf.alive


class _StubTTNN:
    """Models the v0.71 semantics that matter for ``_context_head_ops`` (BRIEF: no ttnn tensors
    on the host): ``fill_implicit_tile_padding`` is IN PLACE -- for rank > 3 it returns a reshape
    VIEW (new Python object, same buffer; fill_pad.cpp / fill_pad_device_operation.cpp) --
    and ``deallocate`` defaults to ``force=True`` (frees a shared buffer too). Every consuming
    op asserts its operands are allocated, like TT_FATAL 'Operands need to be allocated in
    buffers on device'."""

    bfloat16 = "bf16"
    TILE_LAYOUT = "tile"
    DRAM_MEMORY_CONFIG = "dram"

    def __init__(self):
        self.ops = []
        self.made = []
        self.constants = []

        class _Transformer:
            pass

        self.transformer = _Transformer()
        self.transformer.split_query_key_value_and_split_heads = self._split
        self.transformer.concatenate_heads = self._concat

    def _new(self, shape):
        t = _StubTensor(shape)
        self.made.append(t)
        return t

    def _use(self, name, *ts):
        for t in ts:
            assert t.is_allocated(), f"{name}: operand reads a deallocated buffer"
        self.ops.append(name)

    def _split(self, qkv, num_heads, transpose_key):
        self._use("split", qkv)
        B, T, C3 = qkv.shape
        C = C3 // 3
        return tuple(self._new((B, num_heads, T, C // num_heads)) for _ in range(3))

    def _concat(self, o):
        self._use("concat", o)
        B, g, T, d = o.shape
        return self._new((B, T, g * d))

    def fill_implicit_tile_padding(self, t, value):
        self._use("fill", t)
        assert value == 0.0
        if t.shape[-2] % 32 == 0 and t.shape[-1] % 32 == 0:
            return t  # no padding -> the input itself
        return _StubTensor(t.shape, buf=t.buf)  # view of the same buffer

    def transpose(self, t, a, b):
        self._use("transpose", t)
        sh = list(t.shape); sh[a], sh[b] = sh[b], sh[a]
        return self._new(sh)

    def matmul(self, a, b):
        self._use("matmul", a, b)
        assert a.shape[-1] == b.shape[-2]
        return self._new(tuple(a.shape[:-1]) + (b.shape[-1],))

    def from_torch(self, t, dtype=None, layout=None, device=None, memory_config=None):
        """Constants (the zero mask) live outside ``made``: they are meant to stay allocated."""
        c = _StubTensor(tuple(t.shape))
        self.constants.append(c)
        return c

    def scale_mask_softmax_in_place(self, s, scale, mask, numeric_stable=False):
        """v0.71 validate (softmax_device_operation.cpp, hit on the p150a 2026-09-13): a scale
        needs a mask; default program config: mask batch == input batch, intermediate dims 1."""
        self._use("softmax", s, mask)
        assert scale is not None and mask is not None, "Scale value must not be set when mask is not present"
        assert mask.padded_shape[0] == s.padded_shape[0], "Input and mask batch sizes must match"
        assert all(d == 1 for d in mask.padded_shape[1:-2]), "Non-sharded mask intermediate dimensions must be 1"
        assert mask.padded_shape[-2:] == s.padded_shape[-2:]
        assert numeric_stable is True, "the legacy chain used ttnn.softmax (numeric_stable=True)"
        return s

    def deallocate(self, t, force=True):
        assert t.is_allocated(), "double free"
        t.buf.alive = False


def _head_ops_module(T: int, groups: int):
    from torch import nn

    from tt import ttnn_davit_channel_attn as cattn

    stub = _StubTTNN()
    kit = fused.DeviceKit.__new__(fused.DeviceKit)  # no device: skip the ctor's ttnn configs
    kit.ttnn = stub
    kit.cfg = fused.FusedConfig()
    kit.device = None
    kit._zero_masks = {}
    Cls = cattn.TTNNDaViTPreNormChannelAttnFused
    m = Cls.__new__(Cls)
    nn.Module.__init__(m)
    m.kit = kit
    m._ttnn = stub
    m.groups = groups
    return m, stub


def test_channel_attention_head_ops_never_read_a_freed_buffer():
    """Review finding (critical, 2026-09-13): the fused channel attention freed k right after
    ``fill_implicit_tile_padding`` although the op is in place and its rank-4 result is a view of
    k's buffer, so ``transpose(q) @ k`` read deallocated memory in stages 1-3 (T = 784/196/49).
    With the stub semantics above every op must see allocated operands, the fill must run
    exactly in the padded stages, and nothing but the returned context may stay allocated."""
    for T, g, C in [(3136, 8, 256), (784, 16, 512), (196, 32, 1024), (49, 64, 2048)]:
        m, stub = _head_ops_module(T, g)
        qkv = _StubTensor((3, T, 3 * C))
        ctx = m._context_head_ops(qkv, T)
        assert ctx.shape == (3, T, C) and ctx.is_allocated()
        assert stub.ops.count("fill") == (0 if T % 32 == 0 else 1)
        assert stub.ops.count("matmul") == 2
        if T % 32:
            assert stub.ops.index("fill") < stub.ops.index("matmul"), "fill must precede q^T k"
        leaked = [t for t in stub.made if t.is_allocated() and t is not ctx]
        assert not leaked, f"T={T}: {len(leaked)} temporaries left allocated"
        assert qkv.is_allocated(), "the caller frees qkv, not the head ops"
        assert len(stub.constants) == 1 and stub.constants[0].shape == (3, 1, C // g, C // g), "one zero mask per (B, 32, 32)"
        assert stub.constants[0].is_allocated(), "the zero mask is a persistent constant"


# ------------------------------------------------------------------ tallies
def test_fused_launch_tally_is_self_consistent():
    t = fused.fused_launches_per_chunk(1)
    assert t["stack_per_step"] == 2 + 24 * 10 == 242
    assert t["bart"] == 2 + 1 + 12 * 9 == 111
    cattn = 12 + 13 + 9 * 13 + 13
    assert t["davit"] == 12 * 8 + 24 * 6 + cattn == 395
    assert t["total"] == 242 + 111 + 395 == 748
    assert fused.fused_launches_per_chunk(10)["total"] == 10 * 242 + 111 + 395


if __name__ == "__main__":  # plain-script mode: run every test_* function with asserts
    here = sys.modules[__name__]
    names = [n for n in dir(here) if n.startswith("test_")]
    for n in names:
        getattr(here, n)()
        print("PASS", n)
    print(f"{len(names)} tests passed")
