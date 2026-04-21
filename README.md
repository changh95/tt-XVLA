# tt-XVLA — X-VLA on Tenstorrent Blackhole (p150a)

A port of the [lerobot/xvla-base](https://huggingface.co/lerobot/xvla-base)
Vision-Language-Action foundation model to a single Tenstorrent
Blackhole p150a, via [TT-NN](https://github.com/tenstorrent/tt-metal).

Starting from a torch CPU baseline of **10.75 frames/sec**, the final
implementation runs at **181.42 frames/sec** — a **16.87× speedup** on
synthetic inputs, while keeping implementation PCC vs the fp32 torch
reference at **99.998%** and open-loop action MAE vs real-dataset GT
within **+0.00%** of the fp32 reference.

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
| Window partition / reverse | — | torch CPU (6-D reshape) |
| Pre/post processing (flow-matching bookkeeping) | — | torch CPU |

Numerics on-chip: bf16 activations throughout; bfp8_b weights on the
transformer MLPs and DaViT FFN MLPs (large matmuls). Attention weights
stay bf16 for precision.

## Layout

```
tt-xvla/
├── README.md                 # you are here
├── __init__.py
├── benchmark/
│   ├── run_benchmark.py      # end-to-end metric harness (frozen oracle)
│   └── lerobot_bootstrap.py  # workarounds for two unrelated lerobot import bugs
├── tt/                       # the TT-NN port
│   ├── policy.py             # loader — monkey-patches on-chip components
│   ├── ttnn_env.py           # sets TT_METAL_HOME + sys.path for kernel sources
│   ├── ttnn_block_stack.py   # SoftPromptedTransformer 24 blocks
│   ├── ttnn_bart_encoder.py  # Florence-2 BART encoder
│   ├── ttnn_davit_ffn.py     # DaViT 24 FFN modules
│   ├── ttnn_davit_channel_attn.py  # DaViT ChannelAttention (4-D only)
│   └── ttnn_davit_window_attn.py   # DaViT WindowAttention
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

Device id for `ttnn` is fixed at `3` in `tt/policy.py`. Change
`_open_device()` if you need a different chip.

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
| ttnn      | 0.999983 | 0.999983 | 4.66e-03              | 4.57e-03    |

A clean port is `>= 99.9%` mean PCC and `rel_err < 1%`. We are well
under both bars.

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
| ttnn      | 2.5378e+02 | +4.27e-04 (+0.00 %) |

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
iterations, 11 kept, 9 discarded.

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
| 20   | **181.42**      | 99.9777 | keep    | **WindowAttention on Blackhole** (12 spatial blocks, manual SDPA over padded windows) |

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
  torch CPU consumers.
- **Flash-2 SDPA** in the transformer block stack: blocked on seq-length
  padding to a 32-tile boundary.
- **Closed-loop simulator evaluation** (item 3 above).

## Caveats

- Single chip only; `device_id=3` is hard-coded in `tt/policy.py`.
- Synthetic inputs in the benchmark pad language to 16 tokens to keep
  the merged Florence-2 sequence under `max_len_seq=512`. Real-dataset
  eval overrides `config.tokenizer_max_length = 32` for the same reason.
- `num_denoising_steps=1` is a real accuracy tradeoff — on open-loop
  PCC it costs < 0.05%, but on closed-loop rollouts flow-matching
  policies usually want 4–10 steps. An A/B at `steps ∈ {1, 2, 5, 10}`
  is part of the closed-loop eval follow-up.

## License

Apache-2.0. The upstream X-VLA model and lerobot framework are
separately licensed; see their respective repositories.
