# SVD-2DTM search throughput optimization — working report

_Branch `mdg-correlation-impl`, RTX 6000 Ada (142 SMs, 96 MB L2, 300 W power cap, sm_89),
torch 2.13+cu130 (test env) / 2.11+cu130 (dev env), cuFFTDx 1.5. One "correlation" =
one `(pixel, hypothesis, psi)` evaluation. Production shape unless noted:
`NumFreq=64`, 64 eigenvectors per frequency (`r = 4096`), `n_psi = 256`._

## 1. Where the time went (baseline)

Per hypothesis batch at `P=512, N=2048`:

| stage | kernel | time | note |
|---|---|---|---|
| contraction | cuBLAS `cgemm` (complex64) | 1.1–1.7 ms | 20 TFLOP/s; FP32 CUDA cores, no tensor cores |
| psi recovery + stats | cuFFTDx fused kernel `(EPT=8, FPB=8)` | 2.06 ms | 130 Gcorr/s |
| accumulate | CUDA-graphed elementwise | 0.02 ms | |
| **inner step** | | **~3.3 ms** | **~80 Gcorr/s** (pure torch: ~23) |

ncu on the fused kernel: **not** DRAM-bound (16 %); L1/shared-memory pipe at 84 %, 35 % of
stall cycles on shared-memory scoreboard, 23/32 threads active per warp, ~950
warp-instructions per 256-point FFT. The FFT arithmetic is ~5 GFLOP total (0.06 ms at
FP32 peak): the cost was cuFFTDx's inter-thread register↔smem exchanges and index math,
not the transform. The July "82 % FP64 pipe" ncu note predates the current kernel; the
compiled SASS contains zero FP64 instructions.

## 2. What was done

### 2.1 Reduce stage: register-resident kernel (`csrc/lean_irfft_stats.cuh`)
* `corr[R j + r] = Re IDFT64(D_k t^{k r})[j]`: the 256-point zero-padded real iRFFT is
  factored into 64-point complex IDFTs over residues `r`; two real residues are
  Hermitian-packed into one complex IDFT, so a thread owns residues `{h, h+R/2}` —
  2 threads per (pixel, hypothesis) at `n_psi=256`, 1 at `n_psi=128`.
* The 64 complex working values live in 128 registers; the radix-4 IDFT, residue twiddles
  and Hermitian packing are generated fully unrolled with literal immediates
  (`gen_fft64.py` → `fft64_gen.cuh`). No shared memory in the FFT path; 168 registers,
  0 spills, block-level shuffle reduction, 3 atomics per 64 hypotheses.
* Reads complex64 **or complex32** in the GEMM's native `(F, P, Q)` layout; any
  `NumFreq ≤ 64` (templated on `ceil8(F)`).
* Optional **in-kernel accumulation**: adds moments into the running `corr_sum/corr_sum2`
  and merges a packed `(value, global (hyp, psi) index)` max into `best_packed` —
  a stage's whole hypothesis loop needs no per-batch accumulate or decode kernels.
* Result: bit-faithful to the cuFFTDx kernel and the torch path (1e-7, 0 argmax flips);
  `P=512, N=2048`: **2.06 ms → 0.32 ms (845 Gcorr/s)**, DRAM-bandwidth bound for both
  `n_psi`. From L2 (`512×512` batches) it reaches ~1 Tcorr/s.
* cuFFTDx fallback (`NumFreq > 64`, `n_psi = 64`) retuned: `EPT=32/16` instead of 8
  → 1.4–2.4×, bit-identical.

### 2.2 Contraction: FP16 tensor cores (`tiling.py`, `precision="fp16"`)
* Real-valued "4M" GEMM `[Yr|Yi] @ B` with interleaved `(re, im)` output columns, so the
  fp16 `(k, P, 2N)` output **is** a `(k, P, N)` complex tensor in memory — feeds the
  reduce kernel with zero copies. `~4×` cuBLAS cgemm (82–113 TFLOP/s), rel. error 4.4e-4
  (TF32: 3e-4 at 1.9×; also available as `precision="tf32"`).
* **k-interval plan**: the `k` axis is partitioned into runs with a constant set of
  covering rectangles; their `m`-ranges are concatenated, so any multi-rectangle tiling
  is one GEMM per run written straight into its `k`-slab (`out=`) — no accumulate pass,
  no zero-fill. Multi-region search went from 5× slower than single-region to parity.
* Features sliced/converted once per pixel batch (`prepare_features`); compacted
  layouts slice instead of `index_select`.

### 2.3 Loop: CUDA-graphed hypothesis loop (`compressed._HypLoopGraph`, `use_cuda_graph`)
* One graph per pixel-batch size capturing feature conversion + every GEMM + every fused
  reduce-with-accumulate; consecutive batches alternate over two streams so a GEMM
  overlaps the previous reduce. Python/launch overhead per batch → 0.

### 2.4 Numbers (synthetic operands, `P=512`, 2048 hyps in batches of 512, `r=4096`)

| n_psi | torch reference | fp32 (exact) | fp16 eager | fp16 + graph |
|---|---|---|---|---|
| 256 | 23 Gcorr/s | 145 | 219 | **350** |
| 128 | 19 | 64 | 95 | **140** |

Isolated inner step (GEMM + reduce only) at L2-resident shapes with graphs:
~500 Gcorr/s (`n_psi=256`), ~340 (`n_psi=128`).

### 2.5 Real data (60S ribosome, K_MAX=160 / EIG_MAX=128 decomposition, rectangle `(0,0,64,64)`)

`scratch/svd_search_opt/validate_real.py`, xenon particle images `640×640` → `129×129`
valid-correlation pixels each, first 2048 of 6602 hypotheses, `pixel_batch=512`,
`hyp_batch=512`. `|mip| ≤ 32`, z-scores up to 11.9 — comfortably inside fp16 range.

| config | vs fp32 baseline (`mip` rel / z-score abs / `best_index` / `best_psi` agreement) |
|---|---|
| fp32, lean kernel | **bit-identical** (0 / 9.5e-7 / 100 % / 100 %) |
| fp32 + CUDA graph | bit-identical |
| fp16 | 1.5e-4 / 1.9e-3 (rel 1.6e-4) / 99.92 % / 99.93 %; at every disagreeing pixel the two candidates differ in `mip` by ≤ 1.4e-3 (near-ties) |
| fp16 + CUDA graph | same as fp16 |

(`n_psi=128` behaves identically: 1.5e-4 / 1.6e-3 / 99.92 % / 99.94 %.)

#### Steady-state throughput on real data

4 particle images → `n_px = 66 564`, 2048 hypotheses, `NumFreq = 64`, `r = 4096`.
"Steady-state" = marginal rate `(T_all − T_one_batch_per_image) / Δcorrelations`, which
cancels the per-call fixed cost. That fixed cost is **~215 ms per `compressed_search`
call**, almost all of it `build_layout_weights` (gathering `U·S` for all 6602 hypotheses
from the decomposition result) — irrelevant for a full micrograph, dominant for small
calls, and the reason the tiny first validation run showed ~35 Gcorr/s for everything.

| config (`pixel_batch × hyp_batch = 512×512`) | n_psi=256 | n_psi=128 |
|---|---|---|
| pure torch (cgemm + `irfft` + compiled reduce) | 21 Gcorr/s | (~19 synthetic) |
| fp32 cgemm + cuFFTDx fused kernel, retuned EPT (the pre-existing design) | 96 | |
| _(same with the original EPT=8 kernel, estimated from the 2.4× kernel ratio)_ | _~70_ | |
| fp32 cgemm + lean kernel (**exact**, default now) | 121 | 64 |
| fp16 tensor-core GEMM + lean kernel, eager loop | **313–329** | **175** |
| fp16 + CUDA graph, 1 stream | 312–319 (≈ eager) | 175 |
| fp16 + CUDA graph, 2 streams | 310–312 | 157 |

fp16 batch-shape sensitivity on real data (`n_psi=256`): `512×512` ≈ `1024×256` ≈
`1024×512` ≈ 300–320 Gcorr/s, `256×1024` / `512×1024` / `512×2048` 5–15 % lower. Two
streams were never better and cost 10 % at `n_psi=128`.

On this workload the eager fp16 loop is already GPU-bound (≈0.2 ms of GPU work per
hypothesis batch against ≈40 µs of Python), so the graph's only remaining gain is the
few-µs gaps between kernels — an ~10 % effect in isolation (0.71 vs 0.79 ms per pixel
batch) that is inside run-to-run noise on real data. `use_cuda_graph` is therefore
optional here; it matters when batches are made small (many tiny launches).

Kernel-level profile of one `512 × (4 × 512)` pixel batch (torch profiler, identical on
torch 2.11 and 2.13): GEMM `ampere_fp16_s1688gemm_fp16_128x128` 0.080 ms and
`lean_irfft_stats_transposed<64,4,half2>` 0.074 ms per hypothesis batch; everything
else (feature conversion, fills, the 16 MB input copy) < 0.05 ms per pixel batch.
Eager 0.79 ms vs graph 0.71 ms per pixel batch (339 vs 381 Gcorr/s). With two streams
both kernels slow down (0.42 / 0.45 ms) — concurrent execution costs more than the
overlap gains, so `use_cuda_graph=True` now means one stream. The lower graph numbers
in the first real-data pass came from a second graph capture for the trailing partial
pixel batch; partial batches now run eagerly.

## 3. Things that did not pan out / were ruled out
* fp16 *input* to the cuFFTDx kernel: no gain (it was compute/MIO-bound, not DRAM-bound).
* Two-stream GEMM/reduce overlap at large batches: 0.95–1.04× (both kernels fill the GPU).
* Full GEMM+FFT fusion: tiles small enough to hold `C` on-chip re-stream the operands
  3–6× (A/B are 16 KB per pixel / hypothesis); the L2-resident round trip is cheaper.
* DFT-as-GEMM on tensor cores for the psi transform: 65k flops/(p,n) vs ~5k for the FFT;
  not competitive once the FFT is register-resident.
* Bounding/skipping the transform via Parseval: bound is ~16σ vs a ~5σ running best,
  never prunes.
* `n_psi=128`: per correlation the GEMM costs 2×; the kernel is bandwidth-bound at the
  same bytes, so throughput is inherently ~½ of `n_psi=256`.

## 4. Issues encountered
* Another process (your `panther-em-dev` Jupyter kernel) held 28 GB on GPU 0 throughout;
  absolute timings may be perturbed by a few %. The card also power-throttles from
  2.5 GHz to 1.2–1.7 GHz under sustained load — compare ratios, not absolutes.
* The test conda env's `torch_fourier_slice` is too old for the current Leopard-EM
  checkout; real-data validation ran in the `panther-em-dev` env (separate
  `TORCH_EXTENSIONS_DIR`). That env's Leopard-EM indexes the particle DataFrame by label,
  so the scratch validation script re-indexes it.
* A runtime `NumFreq` loop bound silently pushed the 128-float working set to local
  memory (6× slower); fixed by templating on `ceil8(F)` with a predicate on the tail only.
* `merge_packed` must update `best_corr` in place — a captured graph's `clear()` addresses
  the original buffers.

## 5. Future directions
Ordered by expected payoff / effort:

1. **Batch-shape policy in `_run_stage`.** Throughput now depends mostly on keeping the
   per-batch spectrum L2-resident (`pixel_batch * hyp_batch * NumFreq * 4 B ≲ 64 MB`) while
   keeping the GEMM wide enough. Auto-pick `hyp_batch` from `NumFreq`/L2 size instead of
   leaving both knobs to the caller; the real-data sweep below is the starting table.
2. **Lean kernel, 4 threads per pair at `n_psi=256`** (two 32-point Hermitian-packed
   IDFTs per thread, 64 registers): occupancy 2×, likely 1.3–1.5× on the L2-resident
   reduce where the kernel is compute-bound (from L2 it already does ~1 Tcorr/s).
3. **fp16 spectrum from the multi-region path with TF32-style accuracy**: currently
   multi-region fp16 also writes complex32 slabs; if 4e-4 turns out too coarse for the
   multi-precision error bounds, the `tf32` precision is a 1.9× drop-in with 3e-4 error,
   or keep fp16 GEMMs but with `out_dtype=float32` slabs (2× reduce bytes).
4. **Overlap featurization / H2D staging with the search loop** on a second stream for
   CPU-resident feature stores — outside this scope but now the next-largest gap once
   the inner loop is fast.
5. **Deterministic accumulation**: the in-kernel path uses fp32 atomics for `corr_sum`
   / `corr_sum2` (order nondeterministic at the 1e-7 level). A per-block partial-sum
   buffer + tree reduce would make results bit-reproducible at small cost.
6. **Larger `NumFreq` (65–128) in the lean kernel**: needs a 128-point register FFT per
   pair (256 registers → 2 threads per residue pair); the cuFFTDx fallback covers it
   today at the retuned EPT.
7. **Fusion is not the next step.** Holding `C` on-chip forces small tiles and 3–6×
   operand re-streaming; the L2-resident write-once/read-once round trip is cheaper on
   this GPU. Revisit only on parts with much larger shared memory (Hopper's 228 KB +
   TMA) where a `64×64`-pair tile of complex32 `C` (1 MB) still does not fit — i.e.
   probably never for this problem shape.
8. **FP8 contraction**: 2× the fp16 rate on Ada, but 3-bit mantissa; would need
   per-row scaling and an accuracy study against the reconstruction error bounds.
