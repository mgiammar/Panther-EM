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

REAL_DATA_PLACEHOLDER

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
FUTURE_PLACEHOLDER
