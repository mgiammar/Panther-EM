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
 *
 * Two load paths, chosen at compile time from FMAX = ceil8(NumFreq):
 *   FMAX <= 64  load all bins, then scale / residue-twiddle / Hermitian-pack in place;
 *   FMAX  > 64  the same for bins k < 64, then a linear correction pass folds bins
 *               k' + 64 m, m = 1 .. R/2 - 1, (and the Nyquist bin) onto the packed
 *               slots, loading a chunk of slot pairs at a time, so the working set
 *               stays one 64-point spectrum plus the chunk.
 *
 * Input layout is the GEMM's native (NumFreq, P, Q), as float2 (complex64) or __half2
 * (a complex32 / float16-pairs tensor). FFT arithmetic is fp32 either way.
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
// Fold pass (FMAX > 64): bins k = 64 m + k', m = 1 .. R/2 - 1, and the Nyquist bin 32R,
// onto slot k'. Thread residues are h (real output) and h + R/2 (imaginary output);
// with Y_m[k'] = t^{(64 m + k') h} D_{64 m + k'} their folded spectra are
//   X_a = sum_m Y_m,   X_b = u * sum_m (-1)^m Y_m,   u_k' = e^{2 pi i k' / 128},
// and the packed spectrum Z = H(X_a) + i H(X_b) is linear in the bins, so after the low
// bins (m = 0) are packed each m adds  Z += H(Y_m) + i s H(u Y_m),  s = (-1)^m.
// Pairs (k', 64-k') go together: with H(Y)[k'] = a + ib and H(uY)[k'] = c + id,
//   Z[k'] += (a - s d) + i (b + s c),   Z[64-k'] += (a + s d) + i (s c - b).
// Loads are issued FOLD_CHUNK pairs at a time ahead of their use to hide DRAM latency.
// h is a runtime (warp-uniform) value selecting the per-bin twiddle t^{k h}; keep this a
// single code path, as per-h instantiations make ptxas spill.
// ---------------------------------------------------------------------------
constexpr unsigned FOLD_CHUNK = 16;  // pairs (= 32 bins) in flight per thread in the fold pass

template <unsigned FMAX, unsigned R, typename InT>
__device__ __forceinline__ void fold_high_bins(const InT* __restrict__ c, size_t pq, size_t base,
                                               unsigned F, bool active, unsigned h,
                                               float* __restrict__ re, float* __restrict__ im,
                                               float& dc, float& pw) {
  constexpr unsigned NPSI = 64 * R;
  constexpr unsigned NYQ = 32 * R;

  // Y[k] = t^{k h} D_k from the raw bin v. Moments are accumulated by every lane; the
  // kernel keeps h == 0's.
  auto y_from = [&](unsigned k, float2 v) -> float2 {
    accumulate_moments<NYQ>(k, v, dc, pw);
    float2 d = scale_bin<NYQ>(k, v);
    if constexpr (R >= 4) {
      float2 w = lean_root<NPSI>(k % NPSI);  // h = 1
      if constexpr (R >= 8) {
        const float2 w2 = lean_root<NPSI>((2 * k) % NPSI), w3 = lean_root<NPSI>((3 * k) % NPSI);
        w = (h == 2) ? w2 : w;
        w = (h == 3) ? w3 : w;
      }
      const float2 dw = cmul(d, w);
      d = (h == 0) ? d : dw;
    }
    return d;
  };
  auto raw = [&](unsigned k) -> float2 { return load_bin<FMAX, InT>(c, pq, base, F, active, k); };

  // Bins at or beyond FMAX do not exist: every test below is a compile-time constant
  // once the loops unroll, so their pairs are not emitted at all (nvcc cannot fold
  // the arithmetic on a zero bin under IEEE semantics: x + 0 and 0 * c stay).
#pragma unroll
  for (unsigned m = 1; m < R / 2; ++m) {
    if (64 * m >= FMAX) break;
    const float s = (m & 1) ? -1.f : 1.f;
    // self-paired slots 0 and 32: H(Y) = Re Y, H(uY) = Re(u Y)
#pragma unroll
    for (unsigned kp = 0; kp <= 32; kp += 32) {
      if (64 * m + kp >= FMAX) continue;
      const float2 yk = y_from(64 * m + kp, raw(64 * m + kp));
      const float2 uy = cmul(yk, lean_u128(kp));
      re[kp] += yk.x;
      im[kp] += s * uy.x;
    }
    // pairs (k', 64 - k'), k' = 1..31, in chunks: issue the chunk's loads, then consume.
    // Bin 64m + k' < 64m + 64 - k', so a pair is either complete, k'-only, or absent.
#pragma unroll
    for (unsigned kp0 = 1; kp0 < 32; kp0 += FOLD_CHUNK) {
      float2 vk[FOLD_CHUNK], vq[FOLD_CHUNK];
#pragma unroll
      for (unsigned i = 0; i < FOLD_CHUNK; ++i) {
        const unsigned kp = kp0 + i;
        if (kp < 32) {
          if (64 * m + kp < FMAX) vk[i] = raw(64 * m + kp);
          if (64 * m + 64 - kp < FMAX) vq[i] = raw(64 * m + 64 - kp);
        }
      }
#pragma unroll
      for (unsigned i = 0; i < FOLD_CHUNK; ++i) {
        const unsigned kp = kp0 + i;
        if (kp >= 32) break;
        const unsigned kq = 64 - kp;
        const bool has_k = 64 * m + kp < FMAX, has_q = 64 * m + kq < FMAX;
        if (!has_k) continue;
        const float2 yk = y_from(64 * m + kp, vk[i]);
        const float2 uyk = cmul(yk, lean_u128(kp));
        float a = 0.5f * yk.x, b = 0.5f * yk.y;      // H(Y)[k']
        float cc = 0.5f * uyk.x, d = 0.5f * uyk.y;   // H(uY)[k']
        if (has_q) {
          const float2 yq = y_from(64 * m + kq, vq[i]);
          const float2 uyq = cmul(yq, lean_u128(kq));
          a = 0.5f * (yk.x + yq.x); b = 0.5f * (yk.y - yq.y);
          cc = 0.5f * (uyk.x + uyq.x); d = 0.5f * (uyk.y - uyq.y);
        }
        re[kp] += a - s * d;  im[kp] += b + s * cc;
        re[kq] += a + s * d;  im[kq] += s * cc - b;
      }
    }
  }
  // Nyquist bin k = 32R (m = R/2, k' = 0): slot 0 gains f^{R/2} Re D = (-1)^h Re D for
  // residue h and (-f)^{R/2} Re D for residue h + R/2 (the extra (-1)^{R/2} matters for R = 2).
  if constexpr (FMAX > NYQ) {
    const float2 v = load_bin<FMAX, InT>(c, pq, base, F, active, NYQ);
    accumulate_moments<NYQ>(NYQ, v, dc, pw);
    const float da = (h & 1u) ? -v.x : v.x;
    const float db = ((R / 2) & 1u) ? -da : da;
    re[0] += da;
    im[0] += db;
  }
}

// ---------------------------------------------------------------------------
// The kernel. Block = 128 threads; PAIRS = 128 / T hypotheses per block, thread t
// handles pair t % PAIRS with residue group h = t / PAIRS (so h is constant across each
// warp: T <= 4); grid = P * ceil(Q / PAIRS). FMAX = ceil8(NumFreq), NumFreq <= NPSI/2 + 1.
// Register cap: 3 blocks/SM (<= 168 registers) where the working set fits, otherwise
// 2 blocks/SM (<= 255) -- see min_blocks_per_sm.
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
  static_assert(R == 2 || R == 4 || R == 8, "R in {2,4,8}");
  constexpr unsigned T = R / 2;
  constexpr unsigned NPSI = 64 * R;
  constexpr unsigned NYQ = NPSI / 2;
  constexpr unsigned PAIRS = 128 / T;
  static_assert(FMAX % 8 == 0 && FMAX >= 8 && FMAX <= NYQ + 8, "FMAX in {8, ..., ceil8(NPSI/2 + 1)}");
  static_assert(PAIRS % 32 == 0, "each residue group must be whole warps");

  const unsigned p = blockIdx.x / q_tiles;
  const unsigned tile = blockIdx.x - p * q_tiles;
  const unsigned t = threadIdx.x;
  const unsigned h = t / PAIRS;  // warp-uniform
  const unsigned lq = t - h * PAIRS;
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
  // ---- 2. Parseval moments of the low bins (h == 0 warps; the others contribute 0) ----
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
  if constexpr (T > 1) twiddle_residue<NPSI>(re, im, h);
  hermitian_pack_inplace(re, im);
  // ---- 4. fold pass for bins >= 64 (FMAX > 64 only) ----
  if constexpr (FMAX > 64) {
    float dc_hi = 0.f, pw_hi = 0.f;
    fold_high_bins<FMAX, R, InT>(c, pq, base, F, active, h, re, im, dc_hi, pw_hi);
    if (h == 0) { dc += dc_hi; pw += pw_hi; }
  }

  // ---- 5. digit-reversed copy (register renaming only) + in-place radix-4 IDFT64 ----
  constexpr unsigned perm[64] = {IFFT64_PERM_LIST};
  float wr[64], wi[64];
#pragma unroll
  for (unsigned i = 0; i < 64; ++i) { wr[i] = re[perm[i]]; wi[i] = im[perm[i]]; }
  ifft64_inplace(wr, wi);

  // ---- 6. max / argmax over this thread's 128 psi samples ----
  //   wr[j] -> psi = R*j + h ;  wi[j] -> psi = R*j + h + T   (residues h and h + R/2)
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
  constexpr unsigned FTOP = ((32 * R + 1) + 7) / 8 * 8;  // ceil8(NPSI/2 + 1): 72 (R=2), 136 (R=4), 264 (R=8)
  launch_lean_by_fmax<FTOP, R, InT>(c, s1, s2, pk, F, P, Q, hyp_offset, stream);
}

// ---------------------------------------------------------------------------
// On-device test of the generated transforms: one thread per unnormalized inverse DFT,
// natural-order in/out. Not a production path (the 128/256-point ones spill registers).
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
  } else if constexpr (N == 128) {
    constexpr unsigned perm[128] = {IFFT128_PERM_LIST};
#pragma unroll
    for (unsigned i = 0; i < 128; ++i) { const float2 v = in[b * 128 + perm[i]]; re[i] = v.x; im[i] = v.y; }
    ifft128_inplace(re, im);
  } else {
    static_assert(N == 256, "N in {64, 128, 256}");
    constexpr unsigned perm[256] = {IFFT256_PERM_LIST};
#pragma unroll
    for (unsigned i = 0; i < 256; ++i) { const float2 v = in[b * 256 + perm[i]]; re[i] = v.x; im[i] = v.y; }
    ifft256_inplace(re, im);
  }
#pragma unroll
  for (unsigned i = 0; i < N; ++i) out[b * N + i] = make_float2(re[i], im[i]);
}

}  // namespace lean
