/* lean_irfft_stats.cuh -- register-resident fused iRFFT + Parseval moments + max/argmax.
 *
 * Replaces the cuFFTDx block FFT with a hand-scheduled transform that never touches
 * shared memory. Generated straight-line code (lean_fft_gen.cuh, from gen_lean_fft.py):
 *
 *   corr[psi] = Re(C_0) + 2 sum_{k>=1} Re(C_k e^{2 pi i k psi / NPSI}),   NPSI = 64*R
 *   corr[R j + r] = Re IDFT64( X_r )[j],   X_r[k'] = sum_m D_{k'+64m} t^{(k'+64m) r}
 *
 * with t = e^{2 pi i / NPSI} and D = (1, 2, 2, ...) * C (Nyquist bin: 1, imag dropped).
 * The 64-point kernel e^{2 pi i k j / 64} is periodic in k, so any NumFreq up to
 * NPSI/2 + 1 folds exactly onto 64 slots. Two real residues r and r + R/2 are
 * Hermitian-packed into ONE complex 64-point IDFT, so a thread handles residues
 * {h, h + R/2}: T = R/2 threads per (pixel, hypothesis) pair.
 *   R=2 (n_psi=128): 1 thread/pair.   R=4 (n_psi=256): 2 threads/pair.
 * The 64 complex working values live in 128 registers; all twiddles are immediates.
 *
 * Two load paths, chosen at compile time from FMAX = ceil8(NumFreq):
 *   FMAX <= 64  load all bins, then scale / residue-twiddle / Hermitian-pack in place;
 *   FMAX  > 64  the same for bins k < 64, then a linear correction pass folds bins
 *               k' + 64 (and the Nyquist bin) onto the packed slots, two just-in-time
 *               loads per slot pair, so the working set stays one 64-point spectrum.
 *
 * Input layout is the GEMM's native (NumFreq, P, Q), as float2 (complex64) or __half2
 * (a complex32 / float16-pairs tensor). FFT arithmetic is fp32 either way. Measured on an
 * RTX 6000 Ada at (P,Q,F)=(512,2048,64): 0.32 ms for both n_psi (DRAM-bandwidth bound),
 * 6.5x faster than the cuFFTDx block-FFT kernel at n_psi=256.
 */
#pragma once
#include <cfloat>
#include <cuda_fp16.h>
#include "lean_fft_gen.cuh"

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

__device__ __forceinline__ float2 cmul(float2 a, float2 b) {
  return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// ---------------------------------------------------------------------------
// Raw bin load. `k` is a compile-time constant at every call site (all loops are
// unrolled), so only the last group of 8 bins carries the runtime `k < F` test:
// F > FMAX - 8 by construction of FMAX. Zero for k >= F and for inactive lanes.
// ---------------------------------------------------------------------------
template <unsigned FMAX, typename InT>
__device__ __forceinline__ float2 load_bin(const InT* __restrict__ c, size_t pq, size_t base,
                                           unsigned F, bool active, unsigned k) {
  const bool in_range = (k < FMAX) && (k + 8 < FMAX || k < F);
  if (!in_range) return make_float2(0.f, 0.f);
  // Inactive lanes have `base` clamped to a valid row: load unconditionally (no branch,
  // no predicate register) and select the zero afterwards.
  const float2 v = to_f2(c[static_cast<size_t>(k) * pq + base]);
  return active ? v : make_float2(0.f, 0.f);
}

// Parseval contribution of bin k (matches _reduce_stats / the cuFFTDx kernel): DC and
// Nyquist bins count once (real part), every other bin twice.
template <unsigned NYQ>
__device__ __forceinline__ void accumulate_moments(unsigned k, float2 v, float& dc, float& pw) {
  if (k == 0) { dc += v.x; pw = fmaf(v.x, v.x, pw); }
  else if (k == NYQ) { pw = fmaf(v.x, v.x, pw); }
  else { pw = fmaf(2.f * v.x, v.x, pw); pw = fmaf(2.f * v.y, v.y, pw); }
}

// D_k = a_k C_k: a_0 = 1, a_k = 2, Nyquist a = 1 with the imaginary part dropped.
template <unsigned NYQ>
__device__ __forceinline__ float2 scale_bin(unsigned k, float2 v) {
  if (k == NYQ) return make_float2(v.x, 0.f);
  if (k == 0) return v;
  return make_float2(2.f * v.x, 2.f * v.y);
}

// ---------------------------------------------------------------------------
// Fold pass (FMAX > 64): bins k = 64 + k' (and the Nyquist bin 32R) onto slot k'.
// With A = D0 + f D1 (+ f^2 D2) and B = D0 - f D1 (+ f^2 D2), f = e^{2 pi i h / R}, the
// packed spectrum is linear in the bins, so after the low bins are packed (Z0):
//   Z = Z0 + H(Y) - i H(u Y) + Re(f^2 D2) (1 + i) at slot 0,   Y[k'] = t^{k' h} f D_{64+k'}.
// Pairs (k', 64-k') go together: with H_c = H(Y)[k'] = a + ib and H_cu = H(uY)[k'] = c + id,
//   Z[k'] += (a + d) + i (b - c),   Z[64-k'] += (a - d) - i (b + c).
// Two just-in-time loads per pair; the working set stays one 64-point spectrum. h is a
// runtime value here (selects, not branches) so there is a single code path per kernel.
// ---------------------------------------------------------------------------
template <unsigned FMAX, unsigned R, typename InT>
__device__ __forceinline__ void fold_high_bins(const InT* __restrict__ c, size_t pq, size_t base,
                                               unsigned F, bool active, bool h1,
                                               float* __restrict__ re, float* __restrict__ im,
                                               float& dc, float& pw) {
  constexpr unsigned NYQ = 32 * R;

  // Y[k'] = t^{k' h} f D_{64 + k'} ;  f = 1 (h = 0) or i (h = 1, only for R = 4)
  auto y_of = [&](unsigned kp) -> float2 {
    const float2 v = load_bin<FMAX, InT>(c, pq, base, F, active, 64 + kp);
    accumulate_moments<NYQ>(64 + kp, v, dc, pw);
    float2 d = scale_bin<NYQ>(64 + kp, v);
    if (R == 4) {
      const float2 rot = cmul(make_float2(-d.y, d.x), lean_tw256(kp));  // i d t^{k'}
      d = h1 ? rot : d;
    }
    return d;
  };

#pragma unroll
  for (unsigned kp = 0; kp <= 32; ++kp) {
    const float2 yk = y_of(kp);
    if (kp == 0 || kp == 32) {
      // self-paired slot: H_c = Re Y, H_cu = Re(u Y);  Z += H_c - i H_cu
      const float2 uy = cmul(yk, lean_u128(kp));
      re[kp] += yk.x;
      im[kp] -= uy.x;
      continue;
    }
    const unsigned kq = 64 - kp;
    const float2 yq = y_of(kq);
    const float2 uyk = cmul(yk, lean_u128(kp));
    const float2 uyq = cmul(yq, lean_u128(kq));
    const float a = 0.5f * (yk.x + yq.x), b = 0.5f * (yk.y - yq.y);      // H_c
    const float cc = 0.5f * (uyk.x + uyq.x), d = 0.5f * (uyk.y - uyq.y); // H_cu
    re[kp] += a + d;  im[kp] += b - cc;
    re[kq] += a - d;  im[kq] -= b + cc;
  }
  // Nyquist bin of n_psi = 256 (k = 128, m = 2): A[0] and B[0] both gain f^2 D2 = +-Re D2.
  if constexpr (R == 4 && FMAX > 128) {
    const float2 v = load_bin<FMAX, InT>(c, pq, base, F, active, 128);
    accumulate_moments<NYQ>(128, v, dc, pw);
    const float d2 = h1 ? -v.x : v.x;
    re[0] += d2;
    im[0] += d2;
  }
}

// ---------------------------------------------------------------------------
// The kernel. Block = 128 threads; PAIRS = 128 / T hypotheses per block;
// grid = P * ceil(Q / PAIRS). FMAX = ceil8(NumFreq), NumFreq <= NPSI/2 + 1.
// Register cap (see min_blocks_per_sm): 3 blocks/SM (<= 168 registers) wherever ptxas
// fits the working set, 2 blocks/SM (<= 255) where it does not -- the fold path, whose
// correction pass adds in-flight loads, and a few n_psi = 128 instantiations where
// constant-folding the zero bins through the Hermitian pack inflates register pressure.
// All of these are bandwidth-bound; 8 warps/SM with 64+ outstanding loads each saturate DRAM.
// ---------------------------------------------------------------------------
constexpr unsigned min_blocks_per_sm(unsigned fmax, unsigned r) {
  if (fmax > 64) return 2;
  if (r == 2 && (fmax == 24 || fmax == 40 || fmax == 48 || fmax == 56)) return 2;
  return 3;
}

template <unsigned FMAX, unsigned R, typename InT>
__global__ void __launch_bounds__(128, min_blocks_per_sm(FMAX, R))
lean_irfft_stats_transposed(const InT* __restrict__ c, float* __restrict__ s1,
                            float* __restrict__ s2, unsigned long long* __restrict__ argmax_packed,
                            unsigned F, unsigned P, unsigned Q, unsigned q_tiles,
                            unsigned hyp_offset) {
  static_assert(R == 2 || R == 4, "R in {2,4}");
  constexpr unsigned T = R / 2;
  constexpr unsigned NPSI = 64 * R;
  constexpr unsigned NYQ = NPSI / 2;
  constexpr unsigned PAIRS = 128 / T;
  static_assert(FMAX % 8 == 0 && FMAX >= 8 && FMAX <= NYQ + 8, "FMAX in {8, ..., ceil8(NPSI/2 + 1)}");

  const unsigned p = blockIdx.x / q_tiles;
  const unsigned tile = blockIdx.x - p * q_tiles;
  const unsigned t = threadIdx.x;
  const unsigned lq = t / T;
  const unsigned h = t - lq * T;
  const unsigned q = tile * PAIRS + lq;
  const bool active = q < Q;

  const size_t pq = static_cast<size_t>(P) * Q;
  const size_t base = static_cast<size_t>(p) * Q + (active ? q : 0u);

  // ---- 1. low bins (k < 64), natural order, straight into the working array ----
  float re[64], im[64];
#pragma unroll
  for (unsigned k = 0; k < 64; ++k) {
    const float2 v = load_bin<FMAX, InT>(c, pq, base, F, active, k);
    re[k] = v.x; im[k] = v.y;
  }
  // ---- 2. Parseval moments of the low bins (h == 0 lanes; the others contribute 0) ----
  float dc = 0.f, pw = 0.f;
  if (h == 0) {
#pragma unroll
    for (unsigned k = 0; k < 64; ++k) {
      if (k < FMAX) accumulate_moments<NYQ>(k, make_float2(re[k], im[k]), dc, pw);
    }
  }
  // ---- 3. D scaling, residue twiddle t^{k h}, Hermitian pack -> Z0 ----
#pragma unroll
  for (unsigned k = 1; k < 64; ++k) { re[k] *= 2.f; im[k] *= 2.f; }  // (k = 64 = NYQ of n_psi=128 lives in the fold pass)
  if constexpr (T > 1) {
    if (h == 1) twiddle_residue_256(re, im);
  }
  hermitian_pack_inplace(re, im);
  // ---- 4. fold pass for bins >= 64 (FMAX > 64 only) ----
  if constexpr (FMAX > 64) {
    float dc_hi = 0.f, pw_hi = 0.f;
    fold_high_bins<FMAX, R, InT>(c, pq, base, F, active, h == 1, re, im, dc_hi, pw_hi);
    if (h == 0) { dc += dc_hi; pw += pw_hi; }
  }

  // ---- 5. digit-reversed copy (register renaming only) + in-place radix-4 IDFT64 ----
  constexpr unsigned perm[64] = {IFFT64_PERM_LIST};
  float wr[64], wi[64];
#pragma unroll
  for (unsigned i = 0; i < 64; ++i) { wr[i] = re[perm[i]]; wi[i] = im[perm[i]]; }
  ifft64_inplace(wr, wi);

  // ---- 6. max / argmax over this thread's 128 psi samples ----
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

  // ---- 7. block reduction: warp shuffles, then one thread per block does 3 atomics ----
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

// ---------------------------------------------------------------------------
// Launch: runtime NumFreq F -> the smallest FMAX (multiple of 8) instantiation.
// ---------------------------------------------------------------------------
template <unsigned FMAX, unsigned R, typename InT>
inline void launch_lean_transposed_fmax(const InT* c, float* s1, float* s2, unsigned long long* pk,
                                        unsigned F, unsigned P, unsigned Q, unsigned hyp_offset,
                                        cudaStream_t stream) {
  constexpr unsigned PAIRS = 128 / (R / 2);
  const unsigned q_tiles = (Q + PAIRS - 1) / PAIRS;
  lean_irfft_stats_transposed<FMAX, R, InT><<<P * q_tiles, 128, 0, stream>>>(
      c, s1, s2, pk, F, P, Q, q_tiles, hyp_offset);
}

template <unsigned FMAX, unsigned R, typename InT>
inline void launch_lean_by_fmax(const InT* c, float* s1, float* s2, unsigned long long* pk,
                                unsigned F, unsigned P, unsigned Q, unsigned hyp_offset,
                                cudaStream_t stream) {
  if constexpr (FMAX > 8) {
    if (F <= FMAX - 8)
      return launch_lean_by_fmax<FMAX - 8, R, InT>(c, s1, s2, pk, F, P, Q, hyp_offset, stream);
  }
  launch_lean_transposed_fmax<FMAX, R, InT>(c, s1, s2, pk, F, P, Q, hyp_offset, stream);
}

// F in [1, 32R + 1] (i.e. NumFreq <= n_psi/2 + 1).
template <unsigned R, typename InT>
inline void launch_lean_transposed(const InT* c, float* s1, float* s2, unsigned long long* pk,
                                   unsigned F, unsigned P, unsigned Q, unsigned hyp_offset,
                                   cudaStream_t stream) {
  constexpr unsigned FTOP = ((32 * R + 1) + 7) / 8 * 8;  // ceil8(NPSI/2 + 1): 72 (R=2), 136 (R=4)
  launch_lean_by_fmax<FTOP, R, InT>(c, s1, s2, pk, F, P, Q, hyp_offset, stream);
}

// ---------------------------------------------------------------------------
// On-device test of the generated transforms: one thread per unnormalized inverse DFT,
// natural-order in/out. Not a production path (the 128-point one spills registers).
// ---------------------------------------------------------------------------
template <unsigned N>
__global__ void lean_debug_ifft_kernel(const float2* __restrict__ in, float2* __restrict__ out,
                                       unsigned batch) {
  const unsigned b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) return;
  float re[N], im[N];
  if constexpr (N == 64) {
    constexpr unsigned perm[64] = {IFFT64_PERM_LIST};
#pragma unroll
    for (unsigned i = 0; i < 64; ++i) { const float2 v = in[b * 64 + perm[i]]; re[i] = v.x; im[i] = v.y; }
    ifft64_inplace(re, im);
  } else {
    constexpr unsigned perm[128] = {IFFT128_PERM_LIST};
#pragma unroll
    for (unsigned i = 0; i < 128; ++i) { const float2 v = in[b * 128 + perm[i]]; re[i] = v.x; im[i] = v.y; }
    ifft128_inplace(re, im);
  }
#pragma unroll
  for (unsigned i = 0; i < N; ++i) out[b * N + i] = make_float2(re[i], im[i]);
}

}  // namespace lean
