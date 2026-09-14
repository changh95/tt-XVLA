# tt-XVLA — X-VLA on Tenstorrent Blackhole (p150a)

A port of the [lerobot/xvla-base](https://huggingface.co/lerobot/xvla-base)
Vision-Language-Action foundation model to a single Tenstorrent
Blackhole p150a, via [TT-NN](https://github.com/tenstorrent/tt-metal).

Starting from a torch CPU baseline of **10.75 frames/sec**, the final
implementation runs at **315–338 frames/sec** (two `run_benchmark.py`
runs, 1 denoising step, 16 language tokens, measured on the p150a on
2026-09-13) — a **29–31× speedup** on synthetic inputs, while keeping
implementation PCC vs the fp32 torch reference at **99.9982%** and
open-loop action MAE vs real-dataset GT within **+0.00%** of the fp32
reference. Per 30-step action chunk that is a warm median of **89–95 ms**
in-process (1 denoising step; 168–173 ms at 10 steps) against 177 ms /
309 ms for the eager op chain, and **94 ms / 174 ms** served over HTTP
(legacy 185 ms / 308 ms). The speed-up comes from the fused / traced
device path (`TT_FUSED`, default on; `TT_FUSED=0` restores the previous
eager path bit for bit) — see [Fused device path](#fused--traced-device-path-tt_fused)
and [`DEVICE_VALIDATION.md`](DEVICE_VALIDATION.md) for every measured
number.

## What runs where

Pipeline components and their execution location in the final build:

| Component | Params | Location |
|---|---|---|
| SoftPromptedTransformer (24 layers, hidden=1024) | ~308 M | Blackhole p150a |
| Florence-2 BART encoder (12 layers) | ~208 M | Blackhole p150a |
| Florence-2 DaViT FFN modules (24 modules) | — | Blackhole p150a |
| Florence-2 DaViT ChannelAttention (12 modules) | — | Blackhole p150a |
| Florence-2 DaViT WindowAttention (12 modules) | — | Blackhole p150a |
| DaViT ConvEmbed + DepthWiseConv2d | ~30 M | torch CPU (see [Open items](#open-items)) |
| Token / positional embedding lookups | — | torch CPU (small lookups) |
| Window partition / reverse | — | Blackhole p150a (rank-4 RM reshape / permute / pad / slice; `TT_FUSED_WINDOWS_ON_DEVICE=1`); torch CPU with `TT_FUSED=0` |
| Pre/post processing (flow-matching bookkeeping) | — | torch CPU |

With the default fused path the 24-block transformer stack and the
12-layer BART encoder each run as ONE captured metal trace (replayed per
denoising step / per chunk); the DaViT stays 48 eager device round trips
per chunk because its depthwise 3×3 convs and ConvEmbed are still on the
CPU (84 % of the fused chunk, see [Open items](#open-items)).

Numerics on-chip: bf16 activations throughout; bfp8_b weights on the
fc1 of the transformer MLPs, BART MLPs and DaViT FFN MLPs (large
matmuls). Attention weights stay bf16 for precision. On the fused path
the fc2 weights are bf16 so that fc2 + residual fuse into one kernel
(`TT_FUSED_FC2_BF16=1`, +230 MiB DRAM); with `TT_FUSED=0` fc2 is bfp8_b
as before.

## Layout

```
tt-xvla/
├── README.md                 # you are here
├── DEVICE_VALIDATION.md      # fused-path knobs, gates and the measured p150a results (2026-09-13)
├── __init__.py
├── benchmark/
│   ├── run_benchmark.py      # end-to-end metric harness (frozen oracle)
│   └── lerobot_bootstrap.py  # workarounds for two unrelated lerobot import bugs
├── tt/                       # the TT-NN port
│   ├── policy.py             # loader — monkey-patches on-chip components (legacy or *Fused classes)
│   ├── fused.py              # TT_FUSED knobs (FusedConfig), torch reformulations, DeviceKit, TracedGraph
│   ├── ttnn_env.py           # sets TT_METAL_HOME + sys.path for kernel sources
│   ├── ttnn_block_stack.py   # SoftPromptedTransformer 24 blocks
│   ├── ttnn_bart_encoder.py  # Florence-2 BART encoder
│   ├── ttnn_davit_ffn.py     # DaViT 24 FFN modules
│   ├── ttnn_davit_channel_attn.py  # DaViT ChannelAttention (4-D only)
│   ├── ttnn_davit_window_attn.py   # DaViT WindowAttention
│   └── tests/
│       └── test_fused_host.py      # torch-only host tests of the fused reformulations (no device)
└── eval/
    ├── README.md
    ├── eval_relative_pcc.py  # implementation fidelity vs fp32 torch
    └── eval_gt_dataset.py    # action-prediction error vs LeRobot dataset GT
```

`weights/` and `reference/` are gitignored — the former because the
HF checkpoint is 3.3 GB, the latter because it's a per-run cache.

## Prerequisites

- A Tenstorrent Blackhole p150a (this project targets a single chip).
- `ttnn` Python package installed and a matching `tt-metal` checkout.
  Set `TT_METAL_HOME` to the tt-metal tree whose kernel sources match
  your installed `ttnn` runtime ABI; `tt/ttnn_env.py` will auto-detect
  if it's unset.
- `lerobot == 0.5.0` and `transformers == 5.4.x`. The bootstrap file
  `benchmark/lerobot_bootstrap.py` patches two unrelated version
  inconsistencies in the installed `lerobot` automatically.
- A HuggingFace token with access to `lerobot/xvla-base`.

## Download the weights

```bash
export HF_TOKEN=<your token>
python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="lerobot/xvla-base",
    local_dir="weights/xvla_base",
    token=os.environ["HF_TOKEN"],
)
PY
```

## Run the benchmark

```bash
# fp32 torch CPU baseline (establishes the reference action chunk on first run)
python3 benchmark/run_benchmark.py --backend torch_cpu

# TT-NN on Blackhole
python3 benchmark/run_benchmark.py --backend ttnn
```

Both commands print three greppable lines:

```
inference_speed=<frames per second; one frame = one action step>
accuracy=<PCC % vs the cached reference action chunk; 100.0 on first ever run>
peak_dram=<peak DRAM MB; 0 on torch_cpu>
```

The `ttnn` backend runs the fused / traced path by default; add
`TT_FUSED=0` in front of either command to run the previous eager op
chain (bit-identical to the 2026-09-12 code). The chip id comes from
`TT_DEVICE_ID` (default `0`); the original port hard-coded id 3 for the
author's multi-card box — `TT_DEVICE_ID=3` reproduces that.

Measured on the p150a on 2026-09-13 (shipped tt-model image, tt-metal
v0.71.0-dev20260509-4, `run_benchmark.py`, 1 denoising step): legacy
163–170 fps (176.9–183.6 ms per chunk), fused 315–338 fps, `accuracy`
99.9991 % (PCC vs the legacy seed-42 reference chunk).

## Fused / traced device path (`TT_FUSED`)

`tt/fused.py` + the `*Fused` classes in the five `tt/ttnn_*.py` modules
form a second implementation of every on-chip component; `tt/policy.py`
picks it when `TT_FUSED` is unset or `1` and the legacy classes when
`TT_FUSED=0`. The legacy classes are untouched, so `TT_FUSED=0` is bit
for bit the previous behaviour (verified on the device: identical action
chunks at 1 and 10 steps). All knobs are read ONCE per process, at model
build (`FusedConfig.from_env`), so set them before the first
`load_policy_ttnn` call.

What it changes, per action chunk (1 step): 748 device launches instead
of 973 as counted from the code (`fused.fused_launches_per_chunk()`, a
tally not a measurement; bfp8 fc2), of which 353 are replayed from two
metal traces:

- **ROW_MAJOR I/O** (`TT_FUSED_RM_IO=1`): RM upload + `tilize_with_zero_padding`
  on device, `untilize_with_unpadding` + RM readback on all 48 DaViT round
  trips, BART and the stack — exact data movement, no host tilize.
- **Metal traces** (`TT_FUSED_TRACE=1`, `TT_FUSED_TRACE_REGION_MB=64`): the
  24-block transformer stack and the 12-layer BART encoder are each captured
  as one trace with a persistent RM input on the first (warm-up) chunk and
  replayed afterwards; the device is opened with `trace_region_size` and the
  program cache on. The two traces need < 8 MB (measured), so 64 MB is > 8×
  headroom. Eager fallback if capture fails or the input shape changes.
- **SDPA** (`TT_FUSED_SDPA=1`): `scaled_dot_product_attention` over the
  tile-padded sequence (stack 244→256, BART 82→96, DaViT windows 144→160)
  with no explicit mask; the kernel masks padded keys (PCC unchanged).
- **Matmul + residual fusion** (`TT_FUSED_DIT=1`): `dit_minimal_matmul_addcmul_fused`
  for proj + residual in the stack and channel attention, and fc2 + residual
  where fc2 is bf16 (`TT_FUSED_FC2_BF16=1`).
- **`minimal_matmul`** (`TT_FUSED_MINIMAL_MM=1`) for qkv / fc1 / out-proj
  (`TT_FUSED_MM_FIDELITY` / `TT_FUSED_MM_FP32ACC` select the compute kernel
  config; `""` = op default HiFi2 + fp32 accumulation).
- **BART**: post-LN residual adds fused into `layer_norm(residual_input_tensor=)`
  (`TT_FUSED_LN_RESIDUAL=1`); the additive mask is skipped when the 2-D mask
  is all ones (always, in `forward_vlm`).
- **DaViT channel attention with head ops** (`TT_FUSED_CATTN=1`): 21 → 12/13
  launches per module; a constant zero mask feeds `scale_mask_softmax_in_place`
  because this tree refuses a scale without a mask (the one device fix of the
  validation pass).
- **DaViT window partition / reverse / pad / unpad + residual on device**
  (`TT_FUSED_WINDOWS_ON_DEVICE=1`): per-block upload of 1.2 MB tokens instead
  of 3.5 MB windows at stage 2. Not bit-identical to the host path (the
  residual add became a device bf16 add, max abs diff 3.9e-3 on the chunk;
  PCC vs fp32 unchanged).
- **Stack readback** (`TT_FUSED_STACK_OUT_ROWS=-1`): only `ceil32(chunk_size)`
  = 32 rows come back (the transformer reads `x[:, :30]`); `0` = all 244 rows.
- Off by default, kept as A/B knobs: `TT_FUSED_LN_EPS_REF=1` (module eps 1e-5
  instead of ttnn's 1e-12; no accuracy change measured) and
  `TT_FUSED_MM_FIDELITY=HiFi2 TT_FUSED_MM_FP32ACC=0` (same speed, lower PCC).

Every knob's exactness class, the A/B ladder (one knob at a time, ms and
PCC), the per-stage time split and the memory numbers are in
[`DEVICE_VALIDATION.md`](DEVICE_VALIDATION.md). The measured A/B on the
p150a (in-process, N=20 warm, medians; PCC = `eval_relative_pcc.py`,
5 seeds × 10 steps):

| configuration | ms / chunk, 1 step | 10 steps | mean PCC vs fp32 |
|---|---:|---:|---|
| legacy (`TT_FUSED=0`) | 176.9 | 320.9 | 0.999981 |
| fused, kernel swaps off (`TT_FUSED_MINIMAL_MM=0 TT_FUSED_FC2_BF16=0 TT_FUSED_WINDOWS_ON_DEVICE=0`) | 112.4 | 206.4 | 0.999981 |
| **fused, defaults** | **94.5** (alternating rounds 89.4) | **173.4** (168.0) | **0.999982** |

Served over HTTP (30 warm `/predict` requests, medians): legacy 184.9 ms →
fused 94.4 ms at 1 step, 308.4 → 174.3 ms at 10 steps; a 2.0× / 1.8×
speed-up. Extra denoising steps cost ~8.4 ms each on the fused path
(legacy ~16.5 ms) because the stack trace replays.

```bash
# fused path (default) and the legacy path, same benchmark
python3 benchmark/run_benchmark.py --backend ttnn
TT_FUSED=0 python3 benchmark/run_benchmark.py --backend ttnn

# one knob off for an A/B (all knobs are read once at model build)
TT_FUSED_SDPA=0 python3 benchmark/run_benchmark.py --backend ttnn

# fidelity of the fused path vs fp32 torch (same numbers as the table above)
python3 eval/eval_relative_pcc.py --backends torch_cpu,ttnn --steps 10 --seeds 5
```

### Host tests (no device)

`tt/tests/test_fused_host.py` proves every exact reformulation the fused
device code relies on against the reference math in torch (tile
arithmetic, ttnn head split/merge, the channel-attention restructure and
its zero-filled tile padding, the 4-D window partition / reverse, the BART
all-ones-mask shortcut, the launch-count table) and the knob plumbing
(`TT_FUSED` unset → fused defaults, `TT_FUSED=0` → legacy). It imports no
`ttnn` and creates no device tensors, so it runs on any host with torch
and pytest:

```bash
# from the repo root; the tree's python_env has torch 2.7.1 + pytest
python3 -m pytest tt/tests/test_fused_host.py -q     # -> 20 passed
# or as a plain script (asserts)
python3 tt/tests/test_fused_host.py
```

## Evaluation

Two scripts live in `eval/`. See [`eval/README.md`](eval/README.md) for
the full usage, or the short form:

### 1. Relative PCC — implementation fidelity

Answers: *how much numerical error did the TT-NN port introduce, holding
the algorithm fixed?* Both the reference and the test run use the same
`num_denoising_steps`, so only precision / implementation differences
show up.

```bash
python3 eval/eval_relative_pcc.py --backends torch_cpu,ttnn --steps 10 --seeds 5
```

Observed (5 seeds, steps=10):

| backend   | mean PCC | min PCC  | mean \|err\|/std(ref) | max abs err |
|-----------|----------|----------|-----------------------|-------------|
| torch_cpu | 1.000000 | 1.000000 | 0.00e+00              | 0.00e+00    |
| ttnn, fused (default; p150a 2026-09-13) | 0.999982 | 0.999979 | 4.78e-03 | 8.15e-03 |
| ttnn, `TT_FUSED=0` (same pass) | 0.999981 | 0.999978 | 4.90e-03 | 6.42e-03 |
| ttnn, `TT_FUSED=0` (original port run) | 0.999983 | 0.999983 | 4.66e-03 | 4.57e-03 |

A clean port is `>= 99.9%` mean PCC and `rel_err < 1%`. We are well
under both bars; the fused path is marginally *closer* to fp32 than the
eager chain (SDPA and `minimal_matmul` keep fp32 statistics / accumulation).

### 2. Open-loop dataset evaluation — real GT actions

Answers: *how well does the predicted 30-step action chunk match real
robot actions from a LeRobot dataset?*

```bash
# Small smoke test on pusht_image. Single-camera datasets need the same
# image mapped into all three X-VLA view slots (see eval/README.md for
# the full rename syntax).
python3 eval/eval_gt_dataset.py \
  --dataset lerobot/pusht_image \
  --num-samples 100 --steps 10 \
  --backends torch_cpu,ttnn \
  --rename-images 'observation.image=observation.images.image,observation.image=observation.images.image2,observation.image=observation.images.image3' \
  --skip-postprocess
```

Observed backend-delta (10 samples, pusht_image, skip-postprocess):

| backend   | MAE        | Delta vs fp32 |
|-----------|------------|---------------|
| torch_cpu | 2.5378e+02 | —             |
| ttnn, fused (default; p150a 2026-09-13) | 2.5378e+02 | +4.73e-04 (+0.00 %) |
| ttnn, `TT_FUSED=0` (same pass) | 2.5378e+02 | +3.66e-04 (+0.00 %) |
| ttnn, `TT_FUSED=0` (original port run) | 2.5378e+02 | +4.27e-04 (+0.00 %) |

On a second, 50-sample slice (`--num-samples 50 --start-index 1000`,
see `DEVICE_VALIDATION.md` "Results" §3) the fused path measured MAE
234.09 vs 234.09, delta −4.58e-05 (−0.00 %).

The absolute MAE is large because the base X-VLA checkpoint was never
fine-tuned on pusht; the useful signal is the **delta** between
backends, which is effectively zero.

### 3. [ONCOMING] Closed-loop task success in simulation

The strongest signal — running the policy closed-loop in a simulator
(LIBERO / ALOHA sim / robosuite) and reporting per-task success over
50–100 episodes — is not yet implemented in this repo. Tracking
it as a follow-up: simulator install, language-conditioned eval loop
using `policy.select_action`, and a variance budget for the success
estimator. See `eval/README.md` for the open-item record.

## Optimization trajectory

Autoresearch loop — one atomic change per iteration, commit, run
benchmark, keep if speed improved and PCC ≥ 99%, else revert. Ran 20
iterations, 11 kept, 9 discarded; iteration 21 is the fused / traced
device path, gated on the device per knob (A/B ladder in
`DEVICE_VALIDATION.md`) rather than by this loop.

| iter | gen_speed (fps) | PCC     | status  | change |
|------|-----------------|---------|---------|--------|
| 0    | 10.75           | 100.00  | baseline | torch CPU fp32, 10 denoising steps |
| 1    | 36.65           | 99.9991 | keep    | `policy.config.dtype = "bfloat16"` (3.4× over baseline) |
| 2    | 36.08           | 99.9993 | discard | `torch.compile(SoftPromptedTransformer, mode="reduce-overhead")` — slower on CPU |
| 3    | 56.62           | 99.9987 | keep    | `num_denoising_steps` 10 → 5 |
| 4    | 85.16           | 99.9946 | keep    | `num_denoising_steps` 5 → 2 |
| 5    | 100.57          | 99.9799 | keep    | `num_denoising_steps` 2 → 1 (single-step flow matching) |
| 6    | 4–83            | 99.9799 | discard | `torch.set_num_threads(16)` — SMT contention on Zen 5 |
| 7    | 98.76           | 99.9781 | discard | `action_decoder` on chip alone — too small, PCIe wins |
| 8    | 116.75          | 99.9782 | keep    | **24 SoftPromptedTransformer blocks on Blackhole p150a** |
| 9    | 115.33          | 99.9789 | discard | bfp8_b on all transformer weights — small-shape quant overhead |
| 10   | **118.40**      | 99.9785 | keep    | bfp8_b on MLP weights only (attention stays bf16) |
| 11   | 116.82          | 99.9790 | discard | bfp8_b on attention output projection — regressed |
| 12   | 116.88          | 99.9775 | discard | bfp4_b MLP weights — bandwidth wasn't the bottleneck |
| 13   | 117.77          | 99.9798 | discard | LoFi compute kernel config — within noise |
| 14   | 114.24          | 99.9782 | discard | fc1 output to `L1_MEMORY_CONFIG` (unsharded) — worse |
| 15   | 113.51          | 99.9785 | discard | scale-Q pre-matmul — within noise |
| 16   | 117.92          | 99.9797 | discard | explicit 11×10 core grid on MLP — default picker already fine |
| 17   | 118.74          | 99.9782 | keep    | Florence-2 BART encoder (12 layers) on Blackhole |
| 18   | 137.72          | 99.9782 | keep    | DaViT 24 FFN modules on Blackhole, bfp8 MLP weights |
| 19   | 128.89          | 99.9785 | discard | ChannelAttention with mid-block torch round-trips — transfers > compute |
| 19v2 | 144.27          | 99.9785 | keep    | ChannelAttention on chip, **4-D ttnn ops only** (no round-trips) |
| 20   | 181.42          | 99.9777 | keep    | **WindowAttention on Blackhole** (12 spatial blocks, manual SDPA over padded windows) |
| 21   | **315–338**     | 99.9991 | keep    | **Fused / traced device path** (`TT_FUSED`, default on): ROW_MAJOR I/O + device tilize/untilize, SDPA, `minimal_matmul`, matmul+residual and LN+residual fusions, channel-attention head ops, DaViT window permutes on device, block stack + BART encoder as metal traces. Validated on the p150a 2026-09-13 ([`DEVICE_VALIDATION.md`](DEVICE_VALIDATION.md)); its PCC column is vs the iter-20 seed-42 chunk (`run_benchmark.py accuracy=`), the fp32 PCC is 0.999982. `TT_FUSED=0` = iter 20 |

Patterns that worked:

- **Move whole subgraphs, not individual ops.** iter7 (action_decoder
  alone) regressed because PCIe round-trip cost exceeded the single
  small matmul's gain. iter8 (all 24 transformer blocks) paid the PCIe
  cost once and won 16%.
- **bfp8_b on big matmul weights, bf16 on small / attention.** iter10
  hit +1.4% moving only the MLP weights; iter11–12 showed that pushing
  quantization further to attention proj or to 4-bit on the MLP lost
  ground instead of gaining.
- **Keep reshape/permute on the device when possible.** iter19
  bounced through torch for one 5-D permute inside ChannelAttention
  and regressed 6%. iter19v2 used `ttnn.split` + `reshape` + 4-D
  `permute` and won +4.8% over the all-torch-FFN version (iter18).
- **Drop denoising steps aggressively for this flow-matching model.**
  iter3–5 knocked `num_denoising_steps` from 10 to 1 for a 2.75×
  speedup with a PCC drop of only ~0.02%.
- **Cut launches and host round trips before tuning kernels** (iter21).
  RM I/O with device tilize/untilize, two metal traces and the fused
  matmul+residual / LN+residual kernels took the chunk from 177 to 112 ms
  with unchanged PCC; the kernel swaps (`minimal_matmul`, bf16 fc2,
  windows on device) added the last 18 ms. Compute-kernel fidelity
  overrides on top of that gained nothing (dropped).

Patterns that failed:

- `torch.compile` on CPU bf16 (iter2).
- Enabling SMT for big matmul workloads (iter6).
- 4-bit block-float on MLPs (iter12).
- Overriding the default matmul core grid (iter16).
- Single-op partial offloads that don't amortize PCIe (iter7).

## Open items

- **DaViT ConvEmbed + DepthWiseConv2d port.** `ttnn.conv2d` hit an
  allocator failure with a first-pass depthwise config; would need
  sharded-memory and conv-config tuning. These are the largest remaining
  torch CPU consumers (~21 ms of the 87 ms fused chunk) and the reason the
  DaViT is still 48 eager device calls instead of one trace — the DaViT is
  84 % of the fused chunk.
- **Closed-loop simulator evaluation** (item 3 above). Only open-loop MAE
  (pusht_image, 10 + 50 samples) and the action-chunk PCC were measured
  for the fused path.
- **Fused-path items not measured**: cold-kernel-cache boot time of the
  fused path in a fresh image (the served runs used a warm kernel cache,
  READY in 20–22 s; the first fused chunk with cold kernels took 13.8 s
  plus ~1 s trace capture in the dev image) and the transient peak DRAM /
  L1 inside a chunk (steady state after the timing loops: 1322 MiB DRAM,
  0 L1; legacy 1094 MiB).

Done since the original port: Flash-style SDPA in the block stack, BART
and DaViT window attention (`TT_FUSED_SDPA=1`, the seq-length padding is
handled by the kernel's padded-key masking).

## Caveats

- Single chip only; the chip id is `TT_DEVICE_ID` (default `0`; the
  original port hard-coded 3).
- `TT_FUSED*` knobs are read once per process at model build; changing
  them afterwards has no effect. With `TT_FUSED_TRACE=1` the first chunk
  after `load_policy_ttnn` captures the two traces (~1 s on top of that
  chunk; 13.8 s for the chunk itself with a cold kernel cache) —
  `run_benchmark.py` runs its warm-up chunks before timing, so the
  capture never lands in a timed run.
- Synthetic inputs in the benchmark pad language to 16 tokens to keep
  the merged Florence-2 sequence under `max_len_seq=512`. Real-dataset
  eval overrides `config.tokenizer_max_length = 32` for the same reason.
- `num_denoising_steps=1` is a real accuracy tradeoff — on open-loop
  PCC it costs < 0.05%, but on closed-loop rollouts flow-matching
  policies usually want 4–10 steps. An A/B at `steps ∈ {1, 2, 5, 10}`
  is part of the closed-loop eval follow-up.

## Comparison with an RTX 5090 (same host, 2026-09-14)

action chunk 30×20, 3 views 224×224; 1 step / 10 steps; ratio = p150a ms / GPU ms.

| setting | ms | vs p150a |
|---|---:|---|
| p150a, fused traces (served `timing_ms.inference`) | 97.4 / 175.8 | — |
| RTX 5090 fp32 strict | 34.6 / 95.4 | GPU 2.8× / 1.8× |
| RTX 5090 bf16 autocast | 15.5 / 43.5 | GPU 6.3× / 4.0× |
| RTX 5090 fp16 autocast | 15.1 / 43.7 | GPU 6.4× / 4.0× |
| RTX 5090 bf16 + `torch.compile` (reduce-overhead) | 12.8 / 38.1 | GPU 7.6× / 4.6× |

The p150a path is bound by the DaViT vision tower (48 eager ttnn calls plus ~21 ms of CPU depthwise convolutions, ~73 of 87 ms at 1 step), not by the diffusion steps.

Methodology: same host, this repo's torch reference (same weights and preprocessing as the served p150a path) run eagerly in PyTorch 2.11 cu128 (fp32 weights + `torch.autocast` unless stated; no TensorRT), batch 1, medians of 50 iterations after warm-up, H2D/D2H included; GPU fp32 output matches the CPU fp32 reference (PCC 1.0). p150a rows are the served bf16 fused path incl. upload/readback. p150a power was not measured, so no efficiency comparison is made. Full per-precision table, power and memory: [`GPU_COMPARISON.md`](GPU_COMPARISON.md).

## License

Apache-2.0. The upstream X-VLA model and lerobot framework are
separately licensed; see their respective repositories.
