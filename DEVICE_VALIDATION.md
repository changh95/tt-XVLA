# DEVICE_VALIDATION — xvla-base-p150, branch `opt/xvla-base-p150-megakernel`

Hardware validation plan for the fused / traced device path (`TT_FUSED`) implemented host-only on
2026-09-13 (no device, no Docker, no ttnn tensors were created during the implementation; BRIEF §0)
and **validated on the p150a the same day -- see "Results (device, 2026-09-13)" at the end; the
fused path is now the default (`TT_FUSED=0` = legacy).** Sections 0-5 are the plan as written before
the hardware pass (knob defaults quoted there are the pre-flip ones). The evaluation that motivated
the levers is `reports/megakernel/xvla-base-p150.md` (tt-models repo).

Baseline = branch `tt-model-package` @ e8afbd8, the published bundle `changh95/xvla-base-p150`
(image `tt-model/xvla-base-p150:e7d0e7b7b215`, tt-metal v0.71.0-dev20260509-4 / 2a6ddd8e572), served
numbers from `reports/publish-p150/xvla-base-p150.json`:

| metric (legacy path, TT_FUSED unset) | value |
|---|---|
| smoke test, 3 views, seed 42, 1 step | `PASS shape=30x20 ... max|a|=0.574 rms=0.233 a[0][:10]=[-0.041, -0.11, 0.256, 0.146, -0.196, 0.145, 0.024, -0.016, -0.414, 0.406] inference_ms=193.61` |
| `/predict` inference (warm, 3 views, 1 step) | 193.6–211 ms (total 197–213 ms) |
| `/predict` inference (warm, 3 views, 10 steps) | 341.8 ms (total 344.5 ms) -> ~16.5 ms per extra step |
| action-chunk PCC vs fp32 torch (`eval_relative_pcc.py`, 5 seeds, 10 steps) | mean 0.999983, max abs err 4.57e-3 |
| open-loop MAE vs fp32 torch (`eval_gt_dataset.py`, lerobot/pusht_image, 10 samples, 10 steps) | 253.78 vs 253.78 (delta +4.27e-4, +0.00 %) |
| boot | weights 21.1 s + warm-up 31.7 s (cold kernel JIT) = READY 54.7 s |

## 0. What the knob changes (all read once at model build, `tt/fused.py:FusedConfig.from_env`)

| env | default | lever | exactness class |
|---|---|---|---|
| `TT_FUSED` | 0 | master switch; 0 = legacy classes, bit-for-bit the published behaviour | – |
| `TT_FUSED_RM_IO` | 1 | ROW_MAJOR upload + `tilize_with_zero_padding` on device, `untilize_with_unpadding` + ROW_MAJOR readback (all 48 DaViT round trips, BART, stack) | exact (data movement) |
| `TT_FUSED_TRACE` | 1 | block stack (24 blocks) and BART encoder (12 layers) each captured as ONE metal trace with a persistent RM input; `ttnn.open_device(trace_region_size=TT_FUSED_TRACE_REGION_MB<<20)` (default 64 MB; v0.71's default region is 0) + `enable_program_cache()` | exact (same programs replayed) |
| `TT_FUSED_STACK_OUT_ROWS` | -1 | stack readback unpadded to `ceil32(chunk_size)` = 32 rows (the transformer reads `x[:, :30]`); 0 = full 244 rows | exact |
| (always on with TT_FUSED) | – | BART: the additive mask is skipped when the 2-D mask is all ones (always, `forward_vlm`), 12 adds + 1 upload gone | exact (x + 0.0) |
| `TT_FUSED_SDPA` | 1 | `split_query_key_value_and_split_heads(transpose_key=False)` + `scaled_dot_product_attention(is_causal=False, scale, q/k chunk = padded S, NO attn_mask)` + `concatenate_heads` in the stack (S 244->256), BART (82->96), DaViT window attention (144->160) | precision-affecting (fp32 softmax statistics in-kernel; historically accuracy UP on rf-detr) |
| `TT_FUSED_DIT` | 1 | `dit_minimal_matmul_addcmul_fused(h, W, 1.0, residual, ones[1,N], bias)` for proj+residual (stack), channel-attention proj+residual, and fc2+residual where fc2 is bf16; blocks 4x4x4 / subblocks 2x2, ones vector in DRAM like the residual | bf16-rounding (residual added in the accumulator) |
| `TT_FUSED_LN_RESIDUAL` | 1 | BART post-LN: `layer_norm(a, residual_input_tensor=x)` for both residual adds | bf16-rounding |
| `TT_FUSED_CATTN` | 1 | DaViT channel attention with head ops: split(groups, k untransposed), `fill_implicit_tile_padding(k, 0)` for T = 784/196/49, `transpose(q) @ k`, `scale_mask_softmax_in_place(T^-0.5)`, `v @ transpose(probs)`, `concatenate_heads` (21 -> 12/13 launches) | exact data movement + bf16-rounding (fused scale, accumulation order) |
| `TT_FUSED_FC2_BF16` | 0 | upload fc2 weights as bf16 (stack, BART, DaViT FFN) so fc2+residual fuses into dit (dtype rule of the kernel); +4.2 MB DRAM per 1024-wide layer | precision-affecting (upwards) |
| `TT_FUSED_MINIMAL_MM` | 0 | `ttnn.experimental.minimal_matmul` for qkv / fc1 / out-proj (blocks 8x4x4 / 2x2, grid = device grid; fc1 then a separate exact `ttnn.gelu`) | precision-affecting (HiFi2 + fp32 acc default vs `ttnn.linear` default) |
| `TT_FUSED_MM_FIDELITY`, `TT_FUSED_MM_FP32ACC` | "" | compute kernel config of the minimal / dit kernels (`""` = op default HiFi2 + fp32 acc) | precision-affecting |
| `TT_FUSED_WINDOWS_ON_DEVICE` | 0 | DaViT window pad / partition / reverse / unpad / residual on device with rank-4 RM `reshape`/`permute`/`pad`/`slice`/`add` (per-block upload 1.2 MB tokens instead of 3.5 MB windows at stage 2) | exact data movement; device ops unverified |
| `TT_FUSED_LN_EPS_REF` | 0 | pass the modules' LayerNorm eps (1e-5) instead of ttnn's default 1e-12 that the legacy path silently uses | precision-affecting (towards the reference) |

Launch counts as written (per action chunk, 1 step, default knobs, bfp8 fc2), from
`fused.fused_launches_per_chunk()` (host test asserts the arithmetic): stack 242 (24 x 10 + tilize +
untilize; legacy 336), BART 111 (legacy 181), DaViT 395 (legacy 456 + hidden reshape conversions,
and now no host tilize/untilize on the 48 round trips) = **748** (legacy 973); of these 353 are traced.
The DaViT still enters the device 48 times per chunk (the 48 depthwise 3x3 convs and 4 ConvEmbed
convs stay on the CPU — the `generic_op` depthwise kernel of the evaluation §5a is NOT part of this pass).

## 1. Host checks (done here; re-run after any rebase)

```bash
cd $ROOT/models/xvla-base-p150            # ROOT=/home/deepgadget/experiments/tt-models
TREE=/home/deepgadget/experiments/base/tt-metal
TT_METAL_HOME=$TREE PYTHONPATH=code $TREE/python_env/bin/python -m pytest code/tt/tests/test_fused_host.py -q -p no:cacheprovider
# -> 20 passed (torch-only: exact reformulations, geometry tables, knob plumbing, buffer-lifetime
#    model of the channel-attention head ops). `-p no:cacheprovider`: the repo has no .gitignore,
#    so a .pytest_cache/ would dirty the tree.
TT_METAL_HOME=$TREE PYTHONPATH=code:$TREE:$TREE/ttnn:$TREE/tools $TREE/python_env/bin/python -c \
  "import tt.fused, tt.ttnn_block_stack, tt.ttnn_bart_encoder, tt.ttnn_davit_ffn, tt.ttnn_davit_window_attn, tt.ttnn_davit_channel_attn, tt.policy"
```

## 2. Running the fused path on the device

(Plan text, pre-flip.) The published manifest's `serve.env` did NOT set `TT_FUSED` (default off had
to stay until the gates passed; after the pass `serve.env` pins `TT_FUSED=1` + the kept knobs and the
code default is on). Three ways to flip it for the hardware pass:

1. **Host serve** (SERVING.md §3, needs a Python 3.12 ttnn build of the base tree + `requirements.lock`):
   `TT_FUSED=1 ... uvicorn tt.server.app:app --lifespan on --port 20000`.
2. **Container**: build as usual (`tt-model package --container tt-model.yaml --out $ROOT/build`), then run
   the image by hand with `docker run ... -e TT_FUSED=1 ...` (copy the device/hugepage mounts and env from
   `docker inspect tt-model-xvla-base-p150-default` of a normal `tt-model serve`), or add `TT_FUSED: "1"` to
   `serve.env` in a LOCAL, uncommitted copy of `tt-model.yaml` and `tt-model serve` the rebuilt manifest.
3. **Eval scripts on the host** (`code/eval/*.py`, `code/benchmark/run_benchmark.py`) read the same env at
   `load_policy_ttnn` time: `TT_FUSED=1 python code/eval/eval_relative_pcc.py ...`.

READY contract: with `TT_FUSED=1` and `TT_FUSED_TRACE=1` the warm-up chunk runs each traced module as
eager-warm -> capture -> replay; the lifespan then checks `tt.policy.fused_status()["all_traces_captured"]`
and **refuses READY** (RuntimeError before `Application startup complete`) if a capture failed and the
module fell back to eager. Expected log lines: `[xvla.fused] block_stack: trace captured for input
(1, 244, 1024)`, `[xvla.fused] bart_encoder: trace captured for input (1, 82, 1024)`, then
`TT_FUSED module block_stack: {... 'trace_captured': True, 'trace_executions': 1, 'eager_runs': 0 ...}`.
`GET /info` -> `"fused"` shows the config and the per-module trace status.

## 3. Gates (in this order; stop at the first failure and report)

### 3.1 Exact-only configuration must be bit-identical to legacy

```bash
TT_FUSED=1 TT_FUSED_SDPA=0 TT_FUSED_DIT=0 TT_FUSED_CATTN=0 TT_FUSED_LN_RESIDUAL=0   # + serve
python code/tt/server/smoke_test.py --url http://127.0.0.1:20000
```

On in this configuration: RM I/O + device tilize/untilize, the zero-mask drop, the two traces, the
32-row stack readback, `[1, N]` bias rows. Every op is pure data movement or the legacy program with
the same inputs, so `a[0][:10]` must equal the legacy smoke line above **exactly** (the JSON floats of a
seed-42 request must match the legacy server's byte for byte; compare the whole 30x20 chunk with
`/predict` seed 42 against a legacy run). Any difference here is a data-movement bug (untilize end
indices, tilize padding, trace input refresh), not numerics — fix before going on. Also check:
`num_denoising_steps=10` equal to legacy (the stack trace is replayed 10x with `copy_host_to_device_tensor`
in between: a stale-input bug shows up as steps 2..10 repeating step 1).

### 3.2 Default fused path (`TT_FUSED=1`, nothing else)

| check | command | gate | expected |
|---|---|---|---|
| smoke + determinism | `python code/tt/server/smoke_test.py --url ...` (same seed twice -> identical JSON; finite; single-view request ok) | PASS | PASS; `inference_ms` below 193.6 |
| action-chunk fidelity | `TT_FUSED=1 python code/eval/eval_relative_pcc.py --backends torch_cpu,ttnn --steps 10 --seeds 5` | mean PCC >= 0.9990 AND rel_err < 0.01 | ~0.99998 (legacy 0.999983); max abs err <= ~5e-3 |
| task metric | `TT_FUSED=1 python code/eval/eval_gt_dataset.py --dataset lerobot/pusht_image --num-samples 10 --steps 10 --backends torch_cpu,ttnn --rename-images 'observation.image=observation.images.image,observation.image=observation.images.image2,observation.image=observation.images.image3'` | `MAE(ttnn) - MAE(torch_cpu)` within +0.5 % of the reference MAE (253.78 -> delta <= +1.27) | legacy delta +4.27e-4 |
| benchmark oracle | `XVLA_BACKEND=ttnn TT_FUSED=1 python code/benchmark/run_benchmark.py --backend ttnn --weights <dir>` | `accuracy` (PCC % vs the cached seed-42 reference chunk) >= 99.9 | legacy 99.998 |
| speed | `/predict` timing_ms.inference at 1 and 10 steps, 20 requests each, warm | report; no gate | estimate (unverified, evaluation §4): ~150-160 ms at 1 step (-35..-45 ms: -26 ms host layout work measured on this box, -(973-748) x ~35 us launches, -(353) x ~15 us eager dispatch, +48 x 2 x 35 us for the device tilize/untilize pairs); ~270 ms at 10 steps |

PCC alone is not the gate: rf-detr's fidelity changes moved the task metric while PCC rose — the MAE
delta and the smoke `a[0][:10]` drift (should stay within ~1e-2 of the legacy values) are read together.

### 3.3 A/B ladder (one knob at a time on top of 3.2; record PCC, MAE delta, ms at 1 and 10 steps)

1. `TT_FUSED_MINIMAL_MM=1` — the largest and least predictable term (the ~660 GFLOP of matmuls: if
   `ttnn.linear`'s default programs run at rf-detr's 3.8 TFLOP/s class they are 33-130 ms of the budget;
   the minimal kernel was 2-5x faster there). Also try `TT_FUSED_MM_FIDELITY=HiFi2 TT_FUSED_MM_FP32ACC=0`
   (rf-detr's kept setting) vs the op default (HiFi2 + fp32 acc). Watch L1: the 8x4x4 blocks on
   `[9408, 256] x [256, 1024]` (stage-0 FFN fc1) and `[432, 2048] x [2048, 6144]` (stage-3 qkv) are
   the untested extremes; a CB-allocation failure means smaller blocks for that shape.
2. `TT_FUSED_SDPA=0` (A/B of the SDPA kernel: expect slower, PCC slightly lower or equal).
3. `TT_FUSED_FC2_BF16=1` (fc2+residual dit fusion; +~200 MB DRAM for the stack/BART fc2 in bf16, more
   weight traffic on M = 256-row matmuls — may be slower even with one launch fewer per block).
4. `TT_FUSED_CATTN=0` (A/B of the channel-attention restructure) and `TT_FUSED_DIT=0`.
5. `TT_FUSED_WINDOWS_ON_DEVICE=1` (device window permutes; measure per-stage, it adds ~8 launches per
   window attention and saves host permutes + 2.9x smaller stage-2 uploads).
6. `TT_FUSED_LN_EPS_REF=1` (LayerNorm eps 1e-5 as in the reference vs ttnn's 1e-12 that both paths
   use by default) — a precision item, not a speed one.
7. `TT_FUSED_TRACE=0` (eager fused graph) only to quantify the trace gain (expect +5 ms).

### 3.4 Where the time goes (trace-by-difference, as rf-detr's `profile_s5.py`)

Wrap `policy.predict_action_chunk` with `time.perf_counter()` around: (a) the three `forward_vlm` parts
(`_encode_image` = DaViT, BART encoder call, merges), (b) each `TTNNTransformerBlockStackFused._run_all`
call, (c) everything else. With `TT_FUSED_TRACE=1` the traced regions are one `execute_trace` +
readback each; compare against the same wrapper on the legacy path. Report the per-stage table next
to the estimate in the evaluation §1.2 (host DaViT CPU 30-36 ms, host layout 26 ms, launch floor,
matmul time), so the device pass can decide whether the depthwise `generic_op` kernel (evaluation
§5a, the only route to a single whole-model trace) is the next step.

## 4. Failure modes and what they mean

| symptom | likely cause | action |
|---|---|---|
| READY refused: `TT_FUSED=1: a metal trace was not captured during warm-up` | capture raised: trace region too small (`TT_FUSED_TRACE_REGION_MB`, try 128), an op JIT-compiled during capture (program cache off / shape drift between warm run and capture), or a device pre-seeded without a trace region | read the `[xvla.fused] <module>: trace capture FAILED (...)` line above it; serve with `TT_FUSED_TRACE=0` to validate the fused ops eagerly first |
| 3.1 not bit-identical | `untilize_with_unpadding` end indices (inclusive: out dim = end + 1), `tilize_with_zero_padding` on a `[75, 144, 256]`-class RM tensor, stale persistent input (`copy_host_to_device_tensor` skipped), `[1, N]` bias broadcast | isolate with `TT_FUSED_RM_IO=0`, `TT_FUSED_TRACE=0`, `TT_FUSED_STACK_OUT_ROWS=0` one at a time |
| PCC drop with default knobs but fine with `TT_FUSED_SDPA=0` | padded-key masking of this tree's SDPA at S = 244/82/144 (inferred from `sdpa_program_factory.cpp:230-253`, never run here) | keep SDPA off for that module (evaluation §6); do NOT add an explicit mask (rf-detr PCC 0.78 case) |
| stage 1-3 channel attention wrong or `Operands need to be allocated in buffers on device` (`TT_FUSED_CATTN=1`), stage 0 fine | the fill path (T = 784/196/49 -> 800/224/64) differs from stage 0 only in `fill_implicit_tile_padding(k, 0)`. Known and FIXED before this pass (review 2026-09-13): the op is in place -- `FillPadDeviceOperation::create_output_tensors` returns its input and, for rank > 3, `fill_pad.cpp` returns a `ttnn::reshape` VIEW of the same buffer -- so the first version freed `k` after the fill (`ttnn.deallocate` defaults to `force=True`) and the matmul read a deallocated buffer; `_context_head_ops` now frees k once, after the contraction. If the symptom still appears the remaining suspects are the fill kernel itself on 4-D `[3, 32, 196, 32]`-class TILE tensors (unverified) | check the padded rows of k after the fill (`to_torch` of the padded shape); fallback = multiply k by a `[1, 1, T, 32]` ones mask (its tile padding is zero) |
| dit fused op fails validate / L1 | `ternary_a_data_format == in1_data_format` (residual bf16 vs fc2 bfp8: only with `TT_FUSED_FC2_BF16=1` is fc2 fused), scale vector buffer type != residual, 4x4x4 blocks too large for the `[9408, 1024] x [1024, 256]` stage-0 fc2 | see `fused.DeviceKit.linear_residual`; try 2x4x4 / 2x2 for that shape |
| non-deterministic same-seed chunks | rf-detr saw this only with SDPA's on-device block-diagonal mask (`cu_window_seqlens`), which is not used here | if it appears, `TT_FUSED_SDPA=0` isolates it |
| `TT_FUSED_WINDOWS_ON_DEVICE=1` errors | RM `reshape` with a changing last dim `[B, Hp, Wp, C] -> [B*nh, ws, nw, ws*C]`, RM 4-D `permute(0, 2, 1, 3)`, RM `pad`/`slice`, RM `add` — none verified on hardware | leave the knob off; the host path is the default |

## 5. Not verified without hardware (honest list)

* Any device behaviour of the new ops on these shapes: `tilize_with_zero_padding` / `untilize_with_unpadding`
  on `[75, 144, 256]`, `[3, 3136, 256]`, `[1, 244, 1024]`-class tensors (validate read: RM bf16 in,
  interleaved, output width x 2 B aligned — ok on paper); SDPA padded-key masking with `is_causal=False`
  and no mask at S = 244/82/144 (program factory generates the -inf mask for padded K; not run);
  `split_query_key_value_and_split_heads(transpose_key=False)` with batch 75 and 64 heads;
  `concatenate_heads` on `[3, 64, 49, 32]`; `fill_implicit_tile_padding` on 4-D tensors (source-verified:
  in place, rank-4 path returns a reshape view -- the fused class therefore frees k once, after the
  contraction; kernel behaviour on these shapes not run);
  `scale_mask_softmax_in_place(scale)` on `[3, 64, 32, 32]`; `dit_minimal_matmul_addcmul_fused` with the
  4x4x4 / 2x2 config on M = 244 / 588 / 9408 rows (L1 CB fit); `minimal_matmul` bf16 act x bfp8 weight
  (validate accepts it; kernel path untested) and the 8x4x4 blocks on the stage-0/3 extremes;
  `copy_host_to_device_tensor` with ROW_MAJOR host tensors; `begin_trace_capture` of a 242-op graph in a
  64 MB trace region; `ttnn.slice` end convention (used with `TT_FUSED_RM_IO=0` on TILE and by
  `TT_FUSED_WINDOWS_ON_DEVICE` on RM; both knobs default off). The tree's `ttnn.slice` docstring says
  ends must be `< input_tensor_shape[i]`, but that text is stale: `slice_device_operation.cpp:120`
  validates `slice_end[i] <= padded_shape[i]`, `:210` sizes the output as
  `(end - start + step - 1) / step` (exclusive end), and the tree's own
  `tests/ttnn/unit_tests/operations/data_movement/test_slice.py` passes `ends=(1, 3, 320, 320)` /
  `ends = [b + 1, dims[1], dims[2], dims[3]]`, i.e. exclusive ends equal to the dim size, which is
  what both call sites do -- source-verified, not run; every RM data-movement op of
  `TT_FUSED_WINDOWS_ON_DEVICE`.
* Trace aliasing: the two traces are captured in the first warm-up chunk (BART first, then the stack);
  each trace's scratch region may alias the other's persistent input, which is harmless because every
  call refreshes its input with `copy_host_to_device_tensor` and reads its output before the other
  trace runs — but it is an argument, not a measurement. If the device pass sees corruption at
  10 steps, capture the stack trace BEFORE the BART trace (run a stack-only dummy call first) or give
  each trace its own `ttnn.allocate_tensor_on_device` input.
* The ~35 us launch floor, ~15 us eager dispatch overhead and every ms in §3.2 are estimates from the
  gbp-tt (v0.78) measurements and this box's host timings; v0.71 on p150a may differ.
* `ttnn.layer_norm`'s default eps (1e-12) is what the legacy path uses; `TT_FUSED_LN_EPS_REF=1` is the
  first time the port would use the modules' 1e-5 — direction of the PCC change unknown.
* The `[1, N]` bias rows (fused path) vs the legacy 1-D `[N]` biases in `ttnn.linear`: same math, but a
  different program variant may be picked; 3.1 catches any difference.
* `TT_FUSED_STACK_OUT_ROWS`: the transformer reads only `x[:, :chunk_size]`; if a future lerobot version
  reads more rows, set `TT_FUSED_STACK_OUT_ROWS=0`.

## Results (device, 2026-09-13)

Hardware pass on the p150a (validation agent, VALIDATE_BRIEF.md). Evidence:
`tt-models/logs/megakernel-validate/xvla-base/` (every log, probe and script named below; one row per
experiment in `reports/megakernel/VALIDATION.md`). Commits on this branch: `13354a8` (the one device
fix), `83b15a7` (default flip + card), then the docs commit carrying this section. Nothing merged, nothing
pushed, **the tt-model package was NOT rebuilt** -- the shipped image `e7d0e7b7b215` still holds the
pre-pass code, so `tt-model package` must run before the new `tt-model.yaml` (which pins `TT_FUSED=1`)
is pushed: with the old image code `TT_FUSED=1` would hit the softmax validate error of 3.2 below in the
warm-up chunk and the server would not reach READY.

### Environment and harness

* Shipped image `tt-model/xvla-base-p150:e7d0e7b7b215` (py3.12, tt-metal v0.71.0-dev20260509-4 / 2a6ddd8e572);
  dev image `xvla-base-dev:latest` = shipped + pytest (`docker commit`, `dev_image_build.log`). All device
  runs used the `tt-model serve --print` flags + `--rm`, this repo's `code/{tt,benchmark,eval}` bind-mounted
  over `/opt/tt-metal`, weights from the host HF cache (`run_dev.sh`, `serve.sh`, `s5_gate.sh`, `s4_serve_one.sh`).
  The A/B ladder ran in the dev image; the final gate (3.5) and the served A/B (4) in the shipped image.
* `probe_xvla.py`: one process = one knob configuration (knobs are read once at model build): torch fp32
  reference chunks cached once (`ab/ref/`), PCC / rel_err / max abs err on the synthetic batch (5 seeds,
  10 steps = the `eval_relative_pcc.py` metric), the real-image chunk (`media/pusht_synthetic.png` x 3,
  "push the T", seed 42) at 1 and 10 steps saved for bit-identity checks, same-seed determinism, timing of
  `predict_action_chunk` (N=20 warm, median / min / max), per-stage split (DaViT / BART / stack) and
  per-module DaViT split, DRAM / L1 memory view. `probe_served.py`: N warm `/predict` requests, the
  server's `timing_ms.inference` / `.total`, response identity, malformed requests.
* Host tests: `code/tt/tests/test_fused_host.py` 20 passed (tree python_env, torch only) before and after
  every commit (`s2_host_tests.log`, `s5_host_tests_after_flip.log`).

### 1. Legacy regression (`TT_FUSED` unset, HEAD 3f92ad3 = the published code path)

| metric | this pass | published (card / publish record) |
|---|---|---|
| `eval_relative_pcc` 5 seeds, 10 steps | mean 0.999981, min 0.999978, rel_err 4.90e-3, max abs 6.42e-3 | 0.999983, max abs 4.57e-3 |
| `eval_gt_dataset` pusht_image 10 samples, 10 steps | MAE 253.78 vs 253.78, delta +3.66e-4 (+0.00 %) | +4.27e-4 (+0.00 %) |
| `run_benchmark.py` | 163.4 / 169.5 fps (176.9-183.6 ms per 1-step chunk, 16 tokens); reference chunk cached | 181.42 fps (author's box) |
| in-process, real image, N=20 | 176.86 / 174.15 / 188.09 ms at 1 step; 320.88 / 314.81 / 340.37 at 10 steps | -- |
| served, 30 warm requests (shipped image) | inference 184.88 / 175.25 / 190.11 ms (1 step), 308.37 / 301.34 / 318.45 (10 steps); smoke `a[0][:10]` = the published line exactly, inference_ms 194.88 | 193.6-211 ms / 341.8 ms |

Within noise of the published numbers; the branch did not change the default path. (`s1_legacy.log`,
`ab/legacy.log`, `serve/legacy_probe.log`.)

### 2. Fused path: what the hardware said

* **3.1 exact-only configuration** (`TT_FUSED=1 TT_FUSED_SDPA=0 TT_FUSED_DIT=0 TT_FUSED_CATTN=0
  TT_FUSED_LN_RESIDUAL=0`, pre-flip defaults = kernel swaps off): first device run, no validate / L1 / trace
  error; both traces captured in the first chunk; real-image chunk **bit-identical to legacy at 1 AND 10
  steps** (max abs diff 0). 148.58 ms vs legacy 176.86 (-28 ms) at 1 step, 280.61 vs 320.88 at 10 steps.
  (`ab/s31_exact.log`)
* **3.2 default fused** (`TT_FUSED=1`): first run failed in the fused channel attention --
  `TT_FATAL: Scale value must not be set when mask is not present` (softmax_device_operation.cpp:294):
  this tree refuses `scale_mask_softmax_in_place(scores, scale)` without a mask. Fix `13354a8`:
  `DeviceKit.zero_mask(B, 32, 32)` = one constant `[B, 1, 32, 32]` bf16 zero tensor in DRAM (the default
  program config wants mask batch == input batch, intermediate dims 1), `DeviceKit.scale_softmax_` passes
  it with `numeric_stable=True` (as the legacy `ttnn.softmax`); adding 0.0 is exact, launch count unchanged.
  With the fix: 112.35 / 107.61 / 118.28 ms (1 step), 206.38 (10 steps); PCC 0.999981 / 0.999978 /
  rel_err 4.90e-3 = the legacy figures; deterministic. (`ab/s32_default.log`, `ab/s32_default_b.log`)
* **3.3 A/B ladder** (one knob on top of the default; 1-step / 10-step medians, N=20, PCC mean / min /
  rel_err on 5 seeds x 10 steps; every configuration ran with NO device error):

| knob | ms 1 step | ms 10 steps | PCC | verdict |
|---|---:|---:|---|---|
| default (`TT_FUSED=1`) | 112.35 | 206.38 | 0.999981 / 0.999978 / 4.90e-3 | -- |
| `TT_FUSED_MINIMAL_MM=1` | **95.88** | 177.69 | 0.999982 / 0.999979 / 4.78e-3 | keep (-16.5 ms, PCC up; L1 fits at `[9408,256]x[256,1024]` and `[432,2048]x[2048,6144]`) |
| + `TT_FUSED_MM_FIDELITY=HiFi2 TT_FUSED_MM_FP32ACC=0` | 95.38 | 175.00 | 0.999981 / 0.999976 / 4.98e-3 | drop (no speed gain in 3 alternating rounds with the final set: 88.39 vs 88.74; lower PCC) |
| `TT_FUSED_SDPA=0` | 139.23 | 266.19 | 0.999981 / 0.999979 / 4.89e-3 | SDPA keep (worth -27 ms; padded-key masking at S = 244 / 82 / 144 correct: PCC unchanged, no explicit mask) |
| `TT_FUSED_FC2_BF16=1` | 106.72 | **183.82** | 0.999981 / 0.999979 / 4.87e-3 | keep (-5.6 / -22.6 ms; +230 MiB DRAM) |
| `TT_FUSED_CATTN=0` | 122.03 | 214.91 | 0.999981 / 0.999979 / 4.87e-3 | head ops keep (-9.7 ms; `fill_implicit_tile_padding` on 4-D k correct at T = 784 / 196 / 49) |
| `TT_FUSED_DIT=0` | 122.76 | 220.30 | 0.999981 / 0.999979 / 4.88e-3 | dit keep (-10.4 ms; 4x4x4 / 2x2 fits at M = 244 / 588 / 9408) |
| `TT_FUSED_LN_RESIDUAL=0` | 120.56 | 215.77 | 0.999981 / 0.999978 / 4.96e-3 | LN-residual keep (-8.2 ms) |
| `TT_FUSED_WINDOWS_ON_DEVICE=1` | **103.76** | 197.36 | 0.999982 / 0.999979 / 4.83e-3 | keep (-8.6 ms; every RM reshape / permute / pad / slice / add accepted) |
| `TT_FUSED_LN_EPS_REF=1` | 115.87 | 209.66 | 0.999981 / 0.999978 / 4.91e-3 | drop (no accuracy change; knob stays off) |
| `TT_FUSED_TRACE=0` | 119.50 | 216.45 | identical to default | trace keep (-7.2 / -10.1 ms; eager == traced) |
| **c2 = default + MINIMAL_MM + FC2_BF16 + WINDOWS_ON_DEVICE** | **94.50** (rounds 85.7-97.1) | **173.41** | **0.999982 / 0.999979 / 4.78e-3** | the new default |

* **Alternating rounds** (3 x legacy / default / c2, fresh process each): 1-step medians 176.65 / 118.94 /
  **89.42** ms; 10-step 309.07 / 208.93 / **168.03**. Run-to-run noise ~+-3 ms at 1 step (host DaViT).
  (`ab/rr*_*.log`)
* **Where the time goes** (1 step, per chunk): legacy 186.0 = DaViT 162.5 (ConvEmbed 19.9 + depthwise 10.7
  on the host, window_attn 64.7, FFN 35.8, channel_attn 29.5) + BART 8.6 + stack 13.7 + other 2.8;
  **c2 87.1 = DaViT 73.2 (ConvEmbed 12.3 + depthwise 9.0 host = 21 ms; window_attn 14.7, FFN 19.7,
  channel_attn 15.5 = 48 eager device calls at ~1.0 ms each incl. upload / readback) + BART 3.9 + stack
  7.8 + other 2.6**; at 10 steps the stack is 78.3 ms (7.8 ms per replayed trace, legacy 13.5 per eager
  pass). The DaViT is 84 % of the fused chunk. (`ab/prof_legacy.log`, `ab/prof_c2.log`)
* **Memory**: DRAM allocated after the timing loops 1322 MiB of 32 091 (legacy 1094; fc2 bf16 +230), L1 0
  at rest; trace region 64 MB (headroom check: see 6).
* Real-image drift vs legacy: `a[0][:10]` c2 = `[-0.04, -0.11, 0.258, 0.146, -0.197, 0.146, 0.025, -0.019,
  -0.414, 0.406]` (legacy `-0.041, -0.11, 0.256, 0.146, -0.196, 0.145, 0.024, -0.016, -0.414, 0.406`; max
  abs diff of the 30x20 chunk 3.9e-3, PCC 0.99999); the 10-step chunk equals legacy's first row to 3
  decimals; real-image PCC vs fp32 0.999820 (legacy 0.999817) at 1 step, 0.999981 (0.999980) at 10.

### 3. Final gate, shipped image (c2 knobs via `--env`; `s5_c2.log`)

| check | result | gate |
|---|---|---|
| `eval_relative_pcc` 5 seeds, 10 steps | mean **0.999982**, min 0.999979, rel_err 4.78e-3, max abs 8.15e-3 | >= 0.9990 and < 0.01: pass |
| `eval_gt_dataset` pusht_image 10 samples (published setting) | MAE 253.78 vs 253.78, delta **+4.73e-4 (+0.00 %)** | <= +1.27: pass |
| `eval_gt_dataset` pusht_image 50 samples from index 1000 (robustness set) | MAE 234.09 vs 234.09, delta **-4.58e-5 (-0.00 %)** | pass |
| `run_benchmark.py` (16 lang tokens: the BART / stack traces capture at S = 66 / 228 too) | accuracy **99.9991** (PCC % vs the legacy seed-42 chunk), 314.8 / 338.4 fps | >= 99.9: pass |

### 4. Served A/B, shipped image, 30 warm requests per setting (`serve/*_probe.log`)

| setting | READY | smoke | inference ms 1 step (median / min / max) | 10 steps | identical | malformed | stop |
|---|---|---|---|---|---|---|---|
| legacy (`TT_FUSED` unset, pre-flip code) | 22 s (warm cache) | PASS 194.88 ms | 184.88 / 175.25 / 190.11 | 308.37 / 301.34 / 318.45 | 30/30 | 400 / 400 / 400 | SIGTERM -> "Device closed" -> exit 0, 1.69 s |
| c2 (knobs via env, pre-flip code) | 22 s; both traces captured in the warm-up | PASS 102.21 ms | **94.43 / 90.41 / 108.59** | **174.26 / 168.39 / 180.65** | 30/30 | 400 / 400 / 400 | exit 0, 1.73 s |

`/info.fused` shows the config and per-module trace status; `tt-smi -s` OK after every stop. Speed-up served:
2.0x at 1 step, 1.8x at 10 steps.

### 5. Default flip (`83b15a7`)

`FusedConfig` defaults `enabled` / `minimal_mm` / `fc2_bf16` / `windows_on_device` = True; `TT_FUSED=0`
selects the legacy classes (bit-for-bit the 2026-09-12 path); `tt-model.yaml` `serve.env` pins `TT_FUSED=1
TT_FUSED_MINIMAL_MM=1 TT_FUSED_FC2_BF16=1 TT_FUSED_WINDOWS_ON_DEVICE=1`; card speed / accuracy rows,
example response, README headline, eval tables and trajectory row 21 = the measured numbers above (range,
no cold-boot / best-case rows). Host tests 20 passed. Post-flip verification in the shipped image with the
flipped tree bind-mounted: see the rows below.

| post-flip check (shipped image, tree 83b15a7 bind-mounted) | result |
|---|---|
| final gate with NO `TT_FUSED` env (`s5_default.log`) | PCC 0.999982 / 0.999979 / 4.78e-3 / 8.15e-3; pusht 10: +4.73e-4; pusht 50: -4.58e-5; benchmark accuracy 99.9991, 303-333 fps -- identical to the c2 gate |
| served, NO env (`serve/default_probe.log`) | boot log `TT_FUSED config (default on; TT_FUSED=0 = legacy)`, both traces captured, READY 20 s; smoke PASS; 30 warm requests 97.56 / 91.80 / 112.42 ms (1 step), 176.84 / 169.30 / 181.80 (10 steps); responses byte-identical to the c2 served run; malformed -> 400; clean stop 1.57 s |
| served, `TT_FUSED=0` (`serve/knob0_probe.log`) | boot log `TT_FUSED=0: legacy device path`; 191.90 / 178.57 / 204.11 ms (1 step), 321.13 / 301.89 / 337.04 (10 steps); responses byte-identical to the legacy served run at 1 and 10 steps; clean stop 1.75 s |

### 6. Depth checks, kept / dropped, what remains

* **Windows-on-device is bf16-rounding, not exact** (plan §0 said "exact data movement"): on top of the
  exact-only configuration it is NOT bit-identical to legacy (max abs diff 3.9e-3 on the action chunk,
  PCC vs fp32 0.999984 vs 0.999984 for the exact-only recheck, which is still bit-identical) -- the
  window-attention residual add moved from torch to a device bf16 add. Kept: every gate passes and it
  is -8.6 ms. Docstrings corrected (`fused.py` knob table, `ttnn_davit_window_attn.py`).
  (`ab/s7_exact_windows.log`, `ab/s7_exact_recheck.log`)
* **Trace region**: both traces capture with `TT_FUSED_TRACE_REGION_MB=16` and `=8` (chunk bit-identical
  to c2), so the two traces need < 8 MB and the 64 MB default has > 8x headroom. (`ab/s7_region*.log`)
* **Served load**: 100 warm requests (1 step, default config, shipped image): inference 98.01 / 91.24 /
  111.73 ms, responses identical 100/100, clean stop. (`serve/load100_probe.log`)
* **16 language tokens**: `run_benchmark.py` (16 tokens -> BART S = 66, stack S = 228) ran fused with
  accuracy 99.9991, i.e. the traces capture and replay at the second validated `XVLA_LANG_TOKENS` too.
* **Per-request `num_denoising_steps`** (served, default config): 4 steps 173.33 / 127.31 / 190.59 ms,
  50 steps 507.76 / 503.44 / 510.21 ms (N=5, identical 5/5) -- the stack trace replays 50x from the
  same persistent input without a stale-input symptom; ~8.4 ms per extra step (legacy ~16.5).
  (`serve/steps4_50_probe.log`)

**Kept (default on):** RM I/O + device tilize/untilize, block-stack and BART metal traces, 32-row stack
readback, BART zero-mask drop, SDPA (q/k chunk = padded S, no mask), dit matmul+residual (4x4x4 / 2x2),
channel-attention head ops (with the zero-mask scale-softmax fix), BART LN+residual, `minimal_matmul`
(M8K4N4 / 2x2, op-default HiFi2 + fp32 acc) for qkv / fc1 / out-proj, fc2 bf16 + fused residual, DaViT
window permutes + residual on device.
**Dropped (knob kept, default off):** `TT_FUSED_LN_EPS_REF` (no accuracy change), `TT_FUSED_MM_FIDELITY=HiFi2
TT_FUSED_MM_FP32ACC=0` (same speed in alternating rounds, lower PCC). `TT_FUSED_TRACE=0` and `TT_FUSED_SDPA=0`
etc. remain as A/B knobs only.

**Not verified / open:**
* Closed-loop task success in a simulator (the port's own open item, README "ONCOMING") -- only open-loop
  MAE (pusht_image, 10 + 50 samples) and the action-chunk PCC were measured.
* Cold-kernel-cache boot time of the fused path in a fresh image cache (all served runs used the host kernel
  cache warmed by the probe runs: READY 20-22 s; the first fused chunk with cold kernels took 13.8 s in the
  dev image, plus ~1 s trace capture -- far inside the 30 min `tt-model serve` wait).
* Peak (transient) DRAM / L1 inside a chunk: only the steady-state allocation after the timing loops was read
  (1322 MiB DRAM, 0 L1); no allocator peak counter in this tree's Python API.
* The DaViT remains 48 eager device calls (84 % of the fused chunk, 21 ms of it host convs): the next lever is
  the evaluation's §5a depthwise `generic_op` kernel + `conv2d` ConvEmbed -> one DaViT trace. Not attempted.
* The shipped image `e7d0e7b7b215` carries the pre-pass code: `tt-model package` must be rebuilt before the
  new manifest is pushed (with the old image code, `TT_FUSED=1` fails in the warm-up on the softmax validate).
