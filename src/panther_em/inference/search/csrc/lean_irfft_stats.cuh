/* lean_irfft_stats.cuh -- register-resident fused iRFFT + Parseval moments + max/argmax.
 *
 * Replaces the cuFFTDx block FFT with a hand-scheduled transform that never touches
 * shared memory:
 *   corr[psi] = Re(C_0) + 2 sum_{k>=1} Re(C_k e^{2 pi i k psi / NPSI}),   NPSI = 64*R
 *   corr[R j + r] = Re IDFT64( D_k t^{k r} )[j],   t = e^{2 pi i / NPSI},  D = (1,2,2,..) * C
 * Two real residues r, r+R/2 are Hermitian-packed into ONE complex 64-pt IDFT, so a
 * thread handles residues {h, h+R/2}: T = R/2 threads per (pixel, hypothesis) pair.
 *   R=2 (n_psi=128): 1 thread/pair.   R=4 (n_psi=256): 2 threads/pair.
 * The 64 complex working values live in 128 registers; all twiddles are immediates.
 * Input layout is the GEMM's native (NumFreq, P, Q) with NumFreq <= 64 (runtime), as float2
 * or __half2 (a float16 (F,P,2Q) tensor, i.e. complex32 (F,P,Q)). FFT arithmetic is fp32.
 *
 * Measured on an RTX 6000 Ada at (P,Q,F)=(512,2048,64): 0.32 ms for both n_psi, i.e. the
 * kernel is DRAM-bandwidth-bound; 6.5x faster than the cuFFTDx block-FFT kernel at n_psi=256.
 */
#pragma once
#include <cfloat>
#include <cuda_fp16.h>
#include "fft64_gen.cuh"

namespace lean {

__device__ __forceinline__ float2 to_f2(float2 v) { return v; }
__device__ __forceinline__ float2 to_f2(__half2 v) { return __half22float2(v); }

__device__ __forceinline__ unsigned f2sortable(float f) {
  unsigned b = __float_as_uint(f);
  return b ^ ((b & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u);
}
__device__ __forceinline__ unsigned long long pack(float v, unsigned idx) {
  return (static_cast<unsigned long long>(f2sortable(v)) << 32) | idx;
}

// Block = 128 threads; PAIRS = 128 / T hypotheses per block; grid = P * ceil(Q / PAIRS).
template <unsigned R, typename InT>
__global__ void __launch_bounds__(128, 3)
lean_irfft_stats_transposed(const InT* __restrict__ c, float* __restrict__ s1,
                            float* __restrict__ s2, unsigned long long* __restrict__ argmax_packed,
                            unsigned F, unsigned P, unsigned Q, unsigned q_tiles,
                            unsigned hyp_offset) {
  static_assert(R == 2 || R == 4, "R in {2,4}");
  // F (NumFreq, 1..64) is a runtime argument: the unrolled loops below predicate on it.
  constexpr unsigned T = R / 2;
  constexpr unsigned NPSI = 64 * R;
  constexpr unsigned PAIRS = 128 / T;

  const unsigned p = blockIdx.x / q_tiles;
  const unsigned tile = blockIdx.x - p * q_tiles;
  const unsigned t = threadIdx.x;
  const unsigned lq = t / T;
  const unsigned h = t - lq * T;
  const unsigned q = tile * PAIRS + lq;
  const bool active = q < Q;

  const size_t PQ = static_cast<size_t>(P) * Q;
  const size_t base = static_cast<size_t>(p) * Q + (active ? q : 0u);

  float re[64], im[64];
  // ---- 1. load C_k (natural order). Inactive lanes load zeros.
#pragma unroll
  for (unsigned k = 0; k < 64; ++k) {
    if (k < F) {
      const float2 v = to_f2(c[k * PQ + base]);
      re[k] = active ? v.x : 0.f;
      im[k] = active ? v.y : 0.f;
    } else {
      re[k] = 0.f; im[k] = 0.f;
    }
  }
  // ---- 2. Parseval moments (h == 0 lanes only), unscaled by NPSI (done at finalize).
  float dc = 0.f, pw = 0.f;
  if (h == 0) {
    dc = re[0];
    pw = re[0] * re[0];
    // Constant trip count + predicate (NOT `k < F` as the bound): a runtime trip count
    // would stop the full unroll and push re[]/im[] out of registers into local memory.
#pragma unroll
    for (unsigned k = 1; k < 64; ++k) {
      if (k < F) { pw = fmaf(2.f * re[k], re[k], pw); pw = fmaf(2.f * im[k], im[k], pw); }
    }
  }
  // ---- 3. D scaling (2x for k>=1), residue twiddle t^{k h}, Hermitian pack.
#pragma unroll
  for (unsigned k = 1; k < 64; ++k) { re[k] *= 2.f; im[k] *= 2.f; }
  if constexpr (T > 1) {
    if (h == 1) twiddle_h1_256(re, im);   // half the lanes of each warp; ~250 FMAs
  }
  hermitian_pack_inplace(re, im);
  // ---- 4. digit-reversed copy (register renaming only) + in-place radix-4 IDFT64.
  constexpr unsigned perm[64] = {FFT64_PERM_LIST};
  float wr[64], wi[64];
#pragma unroll
  for (unsigned i = 0; i < 64; ++i) { wr[i] = re[perm[i]]; wi[i] = im[perm[i]]; }
  ifft64_inplace(wr, wi);
  // ---- 5. max / argmax over this thread's 128 psi samples.
  //   wr[j] -> psi = R*j + h ;  wi[j] -> psi = R*j + h + T
  float best = -FLT_MAX; unsigned bidx = 0;
#pragma unroll
  for (unsigned j = 0; j < 64; ++j) {
    if (wr[j] > best) { best = wr[j]; bidx = R * j + h; }
    if (wi[j] > best) { best = wi[j]; bidx = R * j + h + T; }
  }
  if (!active) best = -FLT_MAX;
  // Flat (hypothesis, psi) index, global across hypothesis batches via hyp_offset so the
  // packed (value, index) max can accumulate over a whole stage without a decode step.
  unsigned flat = (hyp_offset + q) * NPSI + bidx;
  // ---- 6. block reduction: warp shuffles, then one thread per block does 3 atomics.
#pragma unroll
  for (unsigned off = 16; off > 0; off >>= 1) {
    const float ov = __shfl_down_sync(0xffffffffu, best, off);
    const unsigned oi = __shfl_down_sync(0xffffffffu, flat, off);
    if (ov > best || (ov == best && oi < flat)) { best = ov; flat = oi; }
    dc += __shfl_down_sync(0xffffffffu, dc, off);
    pw += __shfl_down_sync(0xffffffffu, pw, off);
  }
  __shared__ float s_dc[4], s_pw[4]; __shared__ float s_best[4]; __shared__ unsigned s_idx[4];
  const unsigned warp = t >> 5, lane = t & 31;
  if (lane == 0) { s_dc[warp] = dc; s_pw[warp] = pw; s_best[warp] = best; s_idx[warp] = flat; }
  __syncthreads();
  if (t == 0) {
    float bdc = 0.f, bpw = 0.f, bb = -FLT_MAX; unsigned bi = 0;
#pragma unroll
    for (unsigned w = 0; w < 4; ++w) {
      bdc += s_dc[w]; bpw += s_pw[w];
      if (s_best[w] > bb || (s_best[w] == bb && s_idx[w] < bi)) { bb = s_best[w]; bi = s_idx[w]; }
    }
    atomicAdd(s1 + p, static_cast<float>(NPSI) * bdc);
    atomicAdd(s2 + p, static_cast<float>(NPSI) * bpw);
    atomicMax(argmax_packed + p, pack(bb, bi));
  }
}

template <unsigned R, typename InT>
inline void launch_lean_transposed(const InT* c, float* s1, float* s2, unsigned long long* pk,
                                   unsigned F, unsigned P, unsigned Q, unsigned hyp_offset,
                                   cudaStream_t stream) {
  constexpr unsigned PAIRS = 128 / (R / 2);
  const unsigned q_tiles = (Q + PAIRS - 1) / PAIRS;
  lean_irfft_stats_transposed<R, InT><<<P * q_tiles, 128, 0, stream>>>(c, s1, s2, pk, F, P, Q, q_tiles,
                                                                        hyp_offset);
}

}  // namespace lean
