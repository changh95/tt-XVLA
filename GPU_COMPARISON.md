# xvla-base-p150 — Blackhole p150a vs RTX 5090 (same host, same weights, same input)

Date 2026-09-14. Facts only; every GPU number below was measured in this pass, every p150a number is copied (with its source line) from the validation / publish reports and logs. The p150a was NOT touched.

## What was run

| | |
|---|---|
| Model | X-VLA-base (`lerobot/xvla-base`): Florence-2-large encoder (DaViT vision tower 256/512/1024/2048 wide, depths 1/1/9/1, 3 views batched -> 3 x 49 tokens; BART-large encoder, 12 layers, 1024 wide) + SoftPromptedTransformer flow-matching action head (24 blocks, 1024 wide, 16 heads, 32 soft prompts, 30 domains), 879.5 M parameters. The reference is upstream **lerobot 0.5.0 `XVLAPolicy`** loaded with `config.dtype='float32'` from the pinned snapshot — exactly what `code/eval/eval_relative_pcc.py load_policy('torch_cpu')` and the validation session's `probe_xvla.py ref` build, i.e. the fp32 network the p150a's PCC gate (0.999982) was computed against. The port has no standalone reference module: `code/tt/policy.py` loads this same `XVLAPolicy` and swaps its blocks for ttnn modules |
| Weights | `lerobot/xvla-base` @ `cdb7964e4fe842935d671bfab5a5ebe00a96648c` (tt-model.yaml `weights.revision` = `serve.env.TT_WEIGHTS_REVISION`): model.safetensors (3.52 GB fp32), config.json from the HF cache `/home/deepgadget/.cache/huggingface/hub/models--lerobot--xvla-base/snapshots/cdb7964e4fe842935d671bfab5a5ebe00a96648c`, `HF_HUB_OFFLINE=1`. Tokenizer: the vendored `facebook/bart-large` files in `code/tt/assets/bart-large-tokenizer` (the server default) |
| Input | `media/pusht_synthetic.png` (256x256 RGB) sent as all 3 camera views, instruction `"push the T"`, state 8 zeros, `domain_id` 0, `seed` 42 — the `smoke_test.py` / card quickstart / Hub warm-run payload. Preprocessing is the server's own code (`code/tt/server/app.py`): `_prepare_view` x3 = base64 -> PNG decode -> /255 -> ImageNet normalise -> `resize_with_pad` 224x224 (bilinear, zero-pad) -> 3 x `[1,3,224,224]` fp32; `_tokenize` = BART tokenizer right-padded to 32 tokens (5 real: `[0, 41935, 5, 255, 2, 1]...`); state `[1,8]` fp32 (padded to 20 inside the policy); `domain_id` LongTensor. H2D payload 1.81 MB. Flow-matching noise `torch.manual_seed(42); torch.randn(1,30,20)` (see deviation below). `num_denoising_steps` **1** (the served default, `XVLA_NUM_DENOISING_STEPS=1`) and **10** (upstream default); batch 1 |
| Output | `[1,30,20]` ee6d action chunk (raw model space; the base checkpoint has no dataset stats) |
| GPU | NVIDIA GeForce RTX 5090 (sm_120), driver 580.126.18, power limit 600.00 W, 32607 MiB; idle 29.5 W |
| venv | `/home/deepgadget/experiments/tt-models/.venv-gpu/xvla` — Python 3.12.13, torch 2.7.1+cu128, CUDA 12.8, cuDNN 90701, torchvision 0.22.1+cu128, lerobot 0.5.0, transformers 5.4.0 (= the tt-model.yaml pins), numpy 2.2.6, safetensors 0.8.0, huggingface_hub 1.31.0, einops 0.8.2, pillow 12.3.0, tokenizers 0.22.2, triton 3.3.1; added in this pass: fastapi 0.141.1 (+ starlette) so `tt.server.app` — the served preprocessing — imports (pydantic 2.13.5 was already there; no ttnn) |
| Scripts | `logs/gpu-vs-p150/xvla-base/bench_xvla_gpu.py` (eager legs, uses `logs/gpu-vs-p150/bench_common.py`), `bench_xvla_compile.py` (torch.compile leg), `make_report.py` (this file from the JSONs); logs `full_run.log`, `recheck.log`, `compile_run.log`, smoke run `smoke.log`; raw JSON `result.json`, `recheck_result.json`, `compile_result.json` (merged into `reports/gpu-vs-p150/xvla-base.json`); CPU reference `cpu_fp32_reference.pt` |
| Commands | `.venv-gpu/xvla/bin/python bench_xvla_gpu.py --iters 50 --warmup 10 --served-iters 50 --stage-iters 20 --out result.json` ; `.venv-gpu/xvla/bin/python bench_xvla_gpu.py --iters 50 --warmup 10 --served-iters 50 --stage-iters 20 --precisions tf32,fp16_autocast --steps 1 --skip-cpu --no-bf16-resident --out recheck_result.json   # isolated re-run of tf32 / fp16_autocast at 1 step (recheck.log); spliced in as steps1, first pass kept as steps1_first_pass` ; `.venv-gpu/xvla/bin/python bench_xvla_compile.py --iters 50 --warmup 10 --stage-iters 20 --precision bf16_autocast --out compile_result.json` (all with `HF_HUB_OFFLINE=1`) |
| Loop | per precision and step count: 10 warm-ups + 50 timed iterations, `torch.cuda.synchronize()` before/after each; wall-clock (perf_counter) is the primary number, CUDA-event time recorded alongside; power sampled by `nvidia-smi -lms 200` during the timed loop (the loop is padded with untimed iterations to a 2 s window) |
| p150a source | `reports/gpu-vs-p150/p150_numbers.json` -> `reports/megakernel/PUBLISH_SUMMARY.md:21` (Hub `tt serve`, idle host, 30 warm requests: inference **97.44** ms 1 step, **175.75** ms 10 steps); the same run's `timing_ms.total` medians **99.71 / 177.89** from `logs/publish-megakernel/xvla-base/hub-serve-results.txt:8-10` (not carried into p150_numbers.json, which lists the total as unrecorded); in-process stage split `models/xvla-base-p150/DEVICE_VALIDATION.md:273-275` (c2 87.1 = DaViT 73.2 + BART 3.9 + stack 7.8 + other 2.6 ms at 1 step; stack 78.3 at 10 steps); accuracy `:282` (real-image PCC vs fp32 0.999820 / 0.999981), `:288` (gate), served chunks `logs/megakernel-validate/xvla-base/serve/default_response_s{1,10}_seed42.json` |

One deviation from the literal reference (`bench_xvla_gpu.py::generate_actions_fixed_noise`): `XVLAModel.generate_actions` draws the flow-matching noise `x1` with `torch.randn` on the model's device, and a CUDA `randn` under the same seed is a different sequence from the CPU draw the reference and the p150a use. The noise is therefore drawn once on the CPU (`torch.manual_seed(42); torch.randn(1,30,20)` — the exact tensor the CPU fp32 reference draws at that point) and handed to a verbatim copy of `generate_actions` that uses it instead of `randn`. The CPU reference was run both ways (original method + seed, and the fixed-noise copy) and the chunks are identical at 1 and 10 steps (`cpu_reference.orig_vs_fixed_noise_identical`). On the GPU the noise tensor is resident (it is part of the model, not an upload). Everything else — DaViT, BART encoder, the transformer, the action-space pre/post-processing, `predict_action_chunk` itself — is lerobot's code path unchanged.

Timing definitions (matching the p150a `timing_ms` keys):

- **incl_h2d** = `.to('cuda')` of the batch dict (3 views + tokens + state + domain_id, pageable host tensors, what `_build_batch` produces) + `XVLAPolicy.predict_action_chunk` (DaViT on the 3 views batched, BART encoder, N SoftPromptedTransformer steps) + `actions.float().cpu()`. Compare with p150a `timing_ms.inference` = the same `predict_action_chunk` on the ttnn-swapped policy incl. uploads (RM I/O + device tilize) and action readback (**97.44 / 175.75 ms**, PUBLISH_SUMMARY.md:21).
- **excl_h2d** = the same forward with inputs resident and the actions left on the device. The p150a has no separate number for this: its device graph is eager per-op for the DaViT (48 device calls incl. upload/readback each) and two Metal traces (BART, block stack), so the in-process 87.1 ms split (DEVICE_VALIDATION.md:273) is the nearest analogue.
- **served-like** = base64 -> `_build_batch` (3 x PNG decode + normalise + `resize_with_pad` + tokenise) + incl_h2d forward + `actions[0].tolist()`. Compare with p150a `timing_ms.total` (**99.71 / 177.89 ms** = preprocess ~2.3 + inference; hub-serve-results.txt:10).
- **stages** = `Florence2._encode_image` (DaViT, 3 views) / BART encoder call / SoftPromptedTransformer calls (N steps) / other, each bracketed by a sync, inputs resident (median of 20). Compare with the p150a in-process split DaViT 73.2 / BART 3.9 / stack 7.8 / other 2.6 (1 step) and stack 78.3 (10 steps).

## Correctness check (GPU vs CPU fp32 reference)

CPU fp32 reference (`XVLAPolicy` on the host, 16 threads, 7.68 s to load): one `predict_action_chunk` = 0.83 s at 1 step, 2.28 s at 10 steps. The chunks are **bit-identical** to the validation session's cached reference (`logs/megakernel-validate/xvla-base/ab/ref/real_s1_seed42.npy` / `real_s10_seed42.npy`, max abs diff 0.0 — same lerobot, same weights, same payload), so this is exactly the network the p150a was gated against. GPU fp32 strict (no TF32) on the same inputs, PCC over the `[1,30,20]` action chunk (the quantity the p150a's gate uses):

| metric | 1 step | 10 steps |
|---|---:|---:|
| PCC actions (600 values) | **1.0000000** | **1.0000000** |
| max abs diff | 3.874e-07 | 5.662e-07 |
| actions[0][:6] GPU | [-0.0448, -0.1145, 0.2539, 0.1483, -0.2, 0.1447] | [-0.0437, -0.116, 0.2558, 0.1398, -0.1959, 0.1435] |
| CPU fp32 reference actions[0][:6] | [-0.0448, -0.1145, 0.2539, 0.1483, -0.2, 0.1447] | [-0.0437, -0.116, 0.2558, 0.1398, -0.1959, 0.1435] |
| first GPU call (fp32 strict, incl. cuBLAS/cuDNN init) | 200.2 ms | 94.38 ms (second call in the process) |

PCC > 0.999 holds; the GPU runs the right model. Cross-check against the p150a itself: the chunks the p150a served for this exact payload in the validation session (`serve/default_response_s1_seed42.json` / `_s10_`, fused bf16 path) vs the GPU fp32 chunk: **PCC 0.999820** (1 step, max abs 1.304e-02), **0.999981** (10 steps, max abs 5.085e-03) — identical to the p150a-vs-fp32-reference figures DEVICE_VALIDATION.md:282 reports (0.999820 / 0.999981), as it must be since GPU fp32 == CPU fp32 to 6e-7.

Per-precision accuracy vs the CPU fp32 reference (same input):

| GPU precision | PCC 1 step | max abs 1 step | PCC 10 steps | max abs 10 steps | PCC vs p150a served chunk (1 / 10) |
|---|---:|---:|---:|---:|---:|
| fp32 strict (`allow_tf32=False`, `'highest'`) | **1.0000000** | 3.87e-07 | **1.0000000** | 5.66e-07 | 0.999820 / 0.999981 |
| tf32 (`allow_tf32=True`, `'high'`; PyTorch default is `'highest'`) | **1.0000000** | 2.01e-04 | **1.0000000** | 1.99e-04 | 0.999821 / 0.999982 |
| bf16 autocast (fp32 weights, +TF32 remainder) | **0.9999965** | 2.49e-03 | **0.9999960** | 2.32e-03 | 0.999816 / 0.999979 |
| fp16 autocast (fp32 weights, +TF32 remainder) | **0.9999999** | 2.45e-04 | **0.9999999** | 2.98e-04 | 0.999821 / 0.999981 |
| bf16 weights resident (eager, no autocast; = the port's `config.dtype='bfloat16'`) | **0.9999877** | 4.79e-03 | **0.9999917** | 3.08e-03 | 0.999802 / 0.999978 |
| bf16 autocast + `torch.compile` default | **0.9999966** | 2.49e-03 | **0.9999958** | 2.32e-03 | — |
| bf16 autocast + `torch.compile` reduce-overhead | **0.9999966** | 2.49e-03 | **0.9999958** | 2.32e-03 | — |
| p150a fused bf16 path vs the same fp32 reference (DEVICE_VALIDATION.md:282; gate :288) | 0.999820 | 1.30e-02 | 0.999981 | 5.09e-03 | (is the p150a) |

fp16 autocast is numerically fine on this input (no overflow in the BART / DaViT activations; PCC 0.9999999), so it is timed. Every GPU row is above the p150a's own 1-step PCC (0.99982); the bf16 rows land where bf16 should (0.99999x).

## GPU latency (batch 1, 3 x 224x224 + 32 tokens; median / min / p90 of 50 iterations, wall-clock ms)

Eager PyTorch, fp32 weights on the device (autocast rows re-cast the weights on every call — the autocast weight cache is off under `inference_mode`); the last row holds the weights in bf16, the dtype `tt/policy.py` sets for the port's torch side:

**1 denoising step** (p150a `timing_ms.inference` 97.44 ms):

| precision | incl_h2d median / min / p90 | excl_h2d median / min / p90 | CUDA-event excl | first call ms | power mean W (incl / excl loop) | GPU util % | peak mem alloc MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp32 strict (`allow_tf32=False`, `'highest'`) | **34.55 / 34.19 / 35.03** | **34.33 / 33.95 / 34.56** | 34.31 | 34.03 | 452.8 / 481.2 | 89.0 | 3548 |
| tf32 (`allow_tf32=True`, `'high'`; PyTorch default is `'highest'`) | **18.75 / 18.69 / 18.81** | **18.55 / 18.50 / 18.68** | 18.53 | 18.82 | 286.9 / 386.3 | 77.9 | 3548 |
| bf16 autocast (fp32 weights, +TF32 remainder) | **15.46 / 15.39 / 15.77** | **15.30 / 15.21 / 15.81** | 15.29 | 19.27 | 344.5 / 354.3 | 65.9 | 5118 |
| fp16 autocast (fp32 weights, +TF32 remainder) | **15.12 / 15.00 / 15.80** | **14.85 / 14.80 / 14.97** | 14.84 | 18.49 | 331.9 / 365.5 | 58.5 | 5080 |
| bf16 weights resident (eager, no autocast; = the port's `config.dtype='bfloat16'`) | **14.16 / 14.05 / 14.26** | **13.90 / 13.80 / 14.00** | 13.89 | 15.31 | 331.6 / 363.8 | 64.2 | 1783 |
| bf16 autocast + `torch.compile` default | **14.79 / 14.70 / 14.92** | **14.53 / 14.45 / 14.99** | 14.51 | (6.0 s: recompile / graph re-record inside the timing wrapper; compile time in the compile table) | 236.7 / 326.7 | 68.1 | 3456 |
| bf16 autocast + `torch.compile` reduce-overhead (CUDA graphs) | **12.81 / 12.75 / 12.88** | **12.56 / 12.50 / 12.78** | 12.54 | (7.2 s: recompile / graph re-record inside the timing wrapper; compile time in the compile table) | 239.5 / 350.9 | 61.0 | 3382 |

**10 denoising steps** (p150a `timing_ms.inference` 175.75 ms):

| precision | incl_h2d median / min / p90 | excl_h2d median / min / p90 | CUDA-event excl | first call ms | power mean W (incl / excl loop) | GPU util % | peak mem alloc MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp32 strict (`allow_tf32=False`, `'highest'`) | **95.37 / 94.52 / 100.03** | **95.00 / 94.14 / 99.40** | 94.97 | 94.25 | 518.3 / 519.3 | 92.8 | 3548 |
| tf32 (`allow_tf32=True`, `'high'`; PyTorch default is `'highest'`) | **57.90 / 57.16 / 60.62** | **56.94 / 56.22 / 59.27** | 56.92 | 61.08 | 377.8 / 384.0 | 86.2 | 3548 |
| bf16 autocast (fp32 weights, +TF32 remainder) | **43.52 / 43.22 / 45.31** | **42.96 / 42.56 / 43.44** | 42.95 | 45.84 | 335.2 / 357.9 | 71.6 | 5117 |
| fp16 autocast (fp32 weights, +TF32 remainder) | **43.65 / 43.47 / 44.12** | **43.31 / 43.05 / 44.49** | 43.29 | 45.91 | 349.5 / 366.4 | 68.7 | 5080 |
| bf16 weights resident (eager, no autocast; = the port's `config.dtype='bfloat16'`) | **39.00 / 38.20 / 39.37** | **39.16 / 38.99 / 39.42** | 39.14 | 39.26 | 353.2 / 351.4 | 72.4 | 1783 |
| bf16 autocast + `torch.compile` default | **45.19 / 44.99 / 46.79** | **44.77 / 44.59 / 46.40** | 44.75 | 45.24 (no recompile) | 331.7 / 321.2 | 78.4 | 3456 |
| bf16 autocast + `torch.compile` reduce-overhead (CUDA graphs) | **38.08 / 37.86 / 38.79** | **37.63 / 37.50 / 38.21** | 37.61 | (7.9 s: recompile / graph re-record inside the timing wrapper; compile time in the compile table) | 286.4 / 363.7 | 86.0 | 3382 |

Stage split (inputs resident, a sync after each stage, median of 20 ms; p150a in-process split from DEVICE_VALIDATION.md:273-275, c2 configuration, N=20, 1 step; the p150a DaViT figure includes ~21 ms of host work — ConvEmbed 12.3 + depthwise conv 9.0 — that stays on the CPU in the port):

| precision | DaViT (3 views) | BART encoder | transformer 1 step | transformer 10 steps | per step | other | total 1 step | total 10 steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fp32 strict (`allow_tf32=False`, `'highest'`) | 22.17 | 2.63 | 6.71 | 66.91 | 6.69 | 3.02 | 34.54 | 95.63 |
| tf32 (`allow_tf32=True`, `'high'`; PyTorch default is `'highest'`) | 9.70 | 1.88 | 4.14 | 41.40 | 4.14 | 2.89 | 18.61 | 56.98 |
| bf16 autocast (fp32 weights, +TF32 remainder) | 7.20 | 2.09 | 3.08 | 30.34 | 3.03 | 3.02 | 15.39 | 43.53 |
| fp16 autocast (fp32 weights, +TF32 remainder) | 6.99 | 2.07 | 3.02 | 30.40 | 3.04 | 2.80 | 14.88 | 43.61 |
| bf16 weights resident (eager, no autocast; = the port's `config.dtype='bfloat16'`) | 6.44 | 1.84 | 2.73 | 27.09 | 2.71 | 3.03 | 14.04 | 39.38 |
| bf16 autocast + `torch.compile` default | 6.73 | 1.63 | 3.31 | 32.83 | 3.28 | 2.95 | 14.64 | 45.40 |
| bf16 autocast + `torch.compile` reduce-overhead | 5.71 | 1.29 | 2.67 | 26.53 | 2.65 | 2.96 | 12.64 | 37.66 |
| p150a c2 fused (bf16; DaViT = 48 eager device calls + 21 ms host ConvEmbed/depthwise; BART + block stack = Metal traces) | 73.2 | 3.9 | 7.8 | 78.3 | 7.8 | 2.6 | 87.1 (profiling run; the c2 in-process median is 94.5, DEVICE_VALIDATION.md:266) | 173.41 (c2 in-process median, :266) |

Where the time goes: on the GPU the DaViT is 22.2 ms at fp32 strict and 7.2 ms under bf16 autocast for 3 views (the p150a spends 73.2 ms there, 21 of them on the host), the BART encoder 2.6 -> 2.1 ms (p150a trace 3.9), and one transformer step 6.7 -> 3.1 ms (p150a trace replay 7.8). The transformer step is launch-bound at 244 tokens x 24 blocks (about 3 ms for ~1 GFLOP), which is why the 10-step chunk costs 28 ms more than the 1-step chunk under bf16 autocast versus 78.3 ms on the p150a (served delta, 8.7 ms per extra step; 7.8 ms per replayed trace in-process). 'other' (~3-4 ms: embeddings, the merge of image tokens and text, action-space pre/post, Python) is the same on both sides.

`torch.compile` (inductor, `dynamic=False`) on the three compute regions the p150a port also moves on-device — `vision_tower.forward_features_unpool` (DaViT), `language_model.model.encoder.forward` (BART), `transformer.forward` (SoftPromptedTransformer, called once per step) — under bf16 autocast; the Python step loop, `_encode_image`'s data-dependent mask indexing and the action-space pre/post stay eager. `reduce-overhead` = CUDA graphs with `torch.compiler.cudagraph_mark_step_begin()` once per chunk. Accuracy re-checked after compile (table above):

| variant | compile s (first call) | 1 step incl_h2d median / min / p90 | 1 step excl_h2d | 10 steps incl_h2d median / min / p90 | 10 steps excl_h2d | power W (10-step incl loop) | peak mem MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| bf16 autocast + `torch.compile` default | 30.6 | **14.79 / 14.70 / 14.92** | 14.53 | **45.19 / 44.99 / 46.79** | 44.77 | 331.7 | 3456 |
| bf16 autocast + `torch.compile` reduce-overhead (CUDA graphs) | 24.3 | **12.81 / 12.75 / 12.88** | 12.56 | **38.08 / 37.86 / 38.79** | 37.63 | 286.4 | 3382 |

Other facts: model load (safetensors 3.52 GB fp32 -> host `XVLAPolicy.from_pretrained`) **7.68 s**; host -> cuda `policy.to('cuda')` **0.36 s** (3356 MiB of fp32 weights on the device; 1689 MiB bf16-resident); first fp32 call 200.2 ms; idle GPU power 29.5 W. Only the fp32 strict 10-step loop pushes the GPU hard (518.3 W, 92.8 % util); the bf16 rows draw 330-350 W at batch 1 — this model is small enough that the 5090 is launch/bandwidth-bound, not compute-bound.

Served-like loop (same host work as `server/app.py::predict`, 50 iterations, medians ms; p90 of total in brackets):

| GPU precision | steps | preprocess (base64 + PNG decode + normalise + resize_with_pad x3 + tokenise 32) | inference (incl_h2d) | postprocess (`tolist`) | **total** | power W |
|---|---:|---:|---:|---:|---:|---:|
| fp32 strict (`allow_tf32=False`, `'highest'`) | 1 | 2.20 | 34.56 | 0.01 | **36.77** (37.73) | 507.3 |
| tf32 (`allow_tf32=True`, `'high'`; PyTorch default is `'highest'`) | 1 | 2.21 | 18.72 | 0.01 | **20.97** (21.59) | 374.1 |
| bf16 autocast (fp32 weights, +TF32 remainder) | 1 | 2.23 | 15.53 | 0.01 | **17.80** (18.29) | 345.7 |
| fp16 autocast (fp32 weights, +TF32 remainder) | 1 | 2.13 | 15.11 | 0.01 | **17.27** (17.43) | 355.5 |
| bf16 weights resident (eager, no autocast; = the port's `config.dtype='bfloat16'`) | 1 | 2.41 | 14.78 | 0.02 | **17.24** (18.75) | 346.6 |
| p150a Hub run r1 (hub-serve-results.txt:8-10; preprocess from serve/default_response_s1_seed42.json) | 1 | 2.37 | 97.44 | (inside inference: readback) | **99.71** | not measured |
| fp32 strict (`allow_tf32=False`, `'highest'`) | 10 | 2.58 | 96.63 | 0.04 | **99.30** (105.51) | 506.8 |
| tf32 (`allow_tf32=True`, `'high'`; PyTorch default is `'highest'`) | 10 | 2.36 | 56.61 | 0.02 | **59.02** (63.63) | 385.4 |
| bf16 autocast (fp32 weights, +TF32 remainder) | 10 | 2.20 | 43.42 | 0.02 | **45.60** (47.87) | 350.8 |
| fp16 autocast (fp32 weights, +TF32 remainder) | 10 | 2.25 | 43.55 | 0.02 | **45.82** (49.78) | 355.4 |
| bf16 weights resident (eager, no autocast; = the port's `config.dtype='bfloat16'`) | 10 | 2.39 | 39.66 | 0.02 | **42.05** (45.09) | 358.3 |
| p150a Hub run r1 (hub-serve-results.txt:8-10; preprocess from serve/default_response_s10_seed42.json) | 10 | 2.37 | 175.75 | (inside inference: readback) | **177.89** | not measured |

The host stages are the same code on both sides (`_prepare_view` x3 + `_tokenize`); the GPU-side preprocess measured in this venv's PIL/tokenizers (2.1-2.6 ms) matches the p150a server's 2.2-2.4 ms.

## Comparison with the p150a (matching definitions)

Ratio = p150a ms / GPU ms (> 1 means the GPU is faster). p150a precision: bf16 activations, bf16 weights (fc2 bf16), `minimal_matmul` HiFi4 + fp32 acc, SDPA; DaViT window/channel attention and FFN as eager ttnn ops, BART encoder and the 24-block stack as Metal traces (DEVICE_VALIDATION.md §3-5); p150a numbers = Hub `tt serve` run r1 on an idle host (PUBLISH_SUMMARY.md:21).

| row | p150a (definition) | GPU precision | GPU ms 1 step | ratio | GPU ms 10 steps | ratio |
|---|---|---|---:|---:|---:|---:|
| device forward (p150a `timing_ms.inference` incl. uploads + readback vs GPU incl_h2d) | 97.44 / 175.75 (PUBLISH_SUMMARY.md:21; alt: c2 session 94.43 / 174.26, DEVICE_VALIDATION.md:298) | fp32 strict (`allow_tf32=False`, `'highest'`) | 34.548 | **2.82** | 95.366 | **1.84** |
|  |  | tf32 (`allow_tf32=True`, `'high'`; PyTorch default is `'highest'`) | 18.752 | **5.20** | 57.903 | **3.04** |
|  |  | bf16 autocast (fp32 weights, +TF32 remainder) | 15.463 | **6.30** | 43.516 | **4.04** |
|  |  | fp16 autocast (fp32 weights, +TF32 remainder) | 15.122 | **6.44** | 43.650 | **4.03** |
|  |  | bf16 weights resident (eager, no autocast; = the port's `config.dtype='bfloat16'`) | 14.161 | **6.88** | 38.999 | **4.51** |
| | | bf16 autocast + `torch.compile` default | 14.788 | **6.59** | 45.189 | **3.89** |
| | | bf16 autocast + `torch.compile` reduce-overhead | 12.808 | **7.61** | 38.078 | **4.62** |
| served e2e (p150a `timing_ms.total` = preprocess + inference vs GPU served-like total) | 99.71 / 177.89 (hub-serve-results.txt:10, same r1 runs) | fp32 strict (`allow_tf32=False`, `'highest'`) | 36.765 | **2.71** | 99.299 | **1.79** |
|  |  | tf32 (`allow_tf32=True`, `'high'`; PyTorch default is `'highest'`) | 20.973 | **4.75** | 59.016 | **3.01** |
|  |  | bf16 autocast (fp32 weights, +TF32 remainder) | 17.798 | **5.60** | 45.603 | **3.90** |
|  |  | fp16 autocast (fp32 weights, +TF32 remainder) | 17.271 | **5.77** | 45.822 | **3.88** |
|  |  | bf16 weights resident (eager, no autocast; = the port's `config.dtype='bfloat16'`) | 17.238 | **5.78** | 42.049 | **4.23** |
| stage: DaViT 3 views (p150a 48 eager device calls + 21 ms host conv, DEVICE_VALIDATION.md:273) | 73.2 | bf16 autocast / bf16 resident | 7.20 / 6.44 | **10.17** / **11.37** | | |
| stage: BART encoder (p150a Metal trace) | 3.9 | bf16 autocast / bf16 resident | 2.09 / 1.84 | **1.86** / **2.12** | | |
| stage: transformer per step (p150a trace replay 7.8 ms) | 7.8 | bf16 autocast / bf16 resident | 3.03 / 2.71 | **2.57** / **2.88** | | |

Summary of the measured facts: at the served 1-step setting the p150a's fused path (97.44 ms) is 2.82x slower than the RTX 5090 running the same network in strict fp32 (34.55 ms), 6.30x slower than bf16 autocast (15.46 ms) and 6.88x slower than the fastest eager GPU row (bf16 weights resident, 14.16 ms); at 10 steps the ratios are 1.84x / 4.04x / 4.51x. With `torch.compile` the best GPU row is 12.81 ms at 1 step. The p150a's largest share (DaViT, 73 ms of 87 in-process) is the part its port runs as 48 eager device calls plus 21 ms of host convolutions; on the GPU the same DaViT takes 6-7 ms in bf16.

Not measured / not claimed: p150a power (no measurement exists in any pass, so no power-efficiency statement); p150a excl-H2D time (its per-op eager DaViT has no upload-free variant); GPU with a whole-graph CUDA graph of the entire chunk (the data-dependent mask indexing in `_encode_image` prevents a single graph without editing the reference); cost. The 'ratio' columns compare host wall-clock medians with the same definitions on both sides; the p150a numbers are 30-request server medians, the GPU numbers 50-iteration in-process medians.

Note for `p150_numbers.json`: its xvla entry lists `served_e2e_ms.median` as unrecorded for the fused Hub run and derives 100.5 / 178.8; the Hub run did record it — `logs/publish-megakernel/xvla-base/hub-serve-results.txt:10` (`total_ms 1 step r1 99.71 r2 97.00 r3 97.92 ; 10 step r1 177.89 r2 178.83`) and the per-request `total_ms` arrays in `hub-warm30_*step_r*.json`. This report uses the measured r1 values; the shared JSON was not edited.

## Cleanup

`nvidia-smi --query-compute-apps=pid --format=csv,noheader` empty after every script; the p150a was not opened (no `tt-smi`, no docker, nothing under `models/xvla-base-p150` modified — read-only use).
