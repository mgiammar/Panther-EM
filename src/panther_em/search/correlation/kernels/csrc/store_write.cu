/* store_write.cu
 *
 * Writes a featurization chunk into a feature store's group tensor in one launch.
 *
 * A chunk is (n_c, P) complex64 features: cells [cell0, cell0 + n_c) of a rectangle's
 * row-major (k, m) cells at store columns [col0, col0 + P). The group tensor holds
 * blocks of the contraction's operand layout,
 *
 *   float16   (n_blocks, D, K, BP, 2M)  [Re | Im]: re at m, im at M + m
 *   complex64 (n_blocks, D, K, BP, M)   interleaved: re at 2m, im at 2m + 1 (as float)
 *
 * so column c lands in block c / BP at row c % BP. Each thread block takes 32 columns
 * and one k-row's run of (at most 64) eigenvectors: it reads the run's features along
 * the columns (coalesced) into shared memory and writes the 32 rows' runs back as one
 * linear span (a whole k-row of whole pixels is contiguous in the store), so every
 * sector of the store is written by one thread block. DRAM-bandwidth bound.
 */

#include <cstdint>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/complex.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

constexpr int kCols = 32;     // columns per thread block
constexpr int kRun = 64;      // eigenvectors per thread block
constexpr int kThreads = 256;

template <typename T> __device__ __forceinline__ T cast_to(float v);
template <> __device__ __forceinline__ __half cast_to<__half>(float v) {
  return __float2half_rn(v);
}
template <> __device__ __forceinline__ float cast_to<float>(float v) { return v; }

/**
 * grid = (ceil(P / 32), k-rows spanned by the cells * ceil(M / 64)); 256 threads.
 *
 * kSplit: float16 [Re | Im] rows (re at m, im at M + m); otherwise interleaved float
 * pairs (re at 2m, im at 2m + 1). A row is 2M scalars either way.
 */
template <typename T, bool kSplit>
__global__ void __launch_bounds__(kThreads)
    store_write_kernel(const float2* __restrict__ src, int64_t src_stride,
                       T* __restrict__ dst, int64_t n_c, int64_t n_p, int64_t cell0,
                       int64_t col0, int64_t plane, int64_t n_planes, int64_t num_k,
                       int64_t num_m, int64_t block_pixels) {
  __shared__ float re[kRun][kCols + 1];
  __shared__ float im[kRun][kCols + 1];

  const int64_t runs_per_row = (num_m + kRun - 1) / kRun;
  const int64_t k = cell0 / num_m + blockIdx.y / runs_per_row;
  const int64_t run = blockIdx.y % runs_per_row;
  // this block's eigenvectors: the run, clipped to the chunk's cells in row k
  const int64_t m0 = max(run * kRun, cell0 - k * num_m);
  const int64_t m1 = min(min((run + 1) * kRun, num_m), cell0 + n_c - k * num_m);
  if (m0 >= m1) return;
  const int n_m = static_cast<int>(m1 - m0);
  const int64_t c_first = k * num_m + m0 - cell0;  // chunk row of eigenvector m0
  const int64_t p_base = static_cast<int64_t>(blockIdx.x) * kCols;
  const int n_cols = static_cast<int>(min(static_cast<int64_t>(kCols), n_p - p_base));

  // Read: consecutive threads along the columns.
  for (int e = threadIdx.x; e < n_m * kCols; e += kThreads) {
    const int i = e / kCols, j = e % kCols;
    if (j < n_cols) {
      const float2 v = src[(c_first + i) * src_stride + p_base + j];
      re[i][j] = v.x;
      im[i][j] = v.y;
    }
  }
  __syncthreads();

  // Write: consecutive threads along each column's row (re run, then im run).
  const int64_t row_width = 2 * num_m;
  const int per_col = 2 * n_m;
  for (int e = threadIdx.x; e < n_cols * per_col; e += kThreads) {
    const int j = e / per_col, r = e % per_col;
    int i, part;
    if (kSplit) {
      part = r / n_m;
      i = r % n_m;
    } else {
      part = r & 1;
      i = r >> 1;
    }
    const int64_t col = col0 + p_base + j;
    const int64_t blk = col / block_pixels, row = col - blk * block_pixels;
    const int64_t at =
        (((blk * n_planes + plane) * num_k + k) * block_pixels + row) * row_width +
        (kSplit ? part * num_m + m0 + i : 2 * (m0 + i) + part);
    dst[at] = cast_to<T>(part ? im[i][j] : re[i][j]);
  }
}

}  // namespace

/**
 * store[blocks, plane, cells, columns] = values, for a (n_blocks, D, K, BP, W) float16
 * ([Re | Im], W = 2M) or complex64 (W = M) store and (n_c, P) complex64 values whose
 * columns are contiguous.
 */
void store_write(torch::Tensor store, torch::Tensor values, int64_t plane,
                 int64_t cell0, int64_t col0) {
  TORCH_CHECK(store.is_cuda() && values.is_cuda(), "store and values must be CUDA");
  TORCH_CHECK(store.device() == values.device(),
              "store and values must be on one device");
  TORCH_CHECK(store.dim() == 5 && store.is_contiguous(),
              "store must be a contiguous (n_blocks, D, K, BP, W) tensor");
  TORCH_CHECK(values.dim() == 2 && values.scalar_type() == torch::kComplexFloat,
              "values must be a (n_cells, P) complex64 tensor");
  TORCH_CHECK(values.size(1) <= 1 || values.stride(1) == 1,
              "values' columns must be contiguous");

  const bool split = store.scalar_type() == torch::kHalf;
  TORCH_CHECK(split || store.scalar_type() == torch::kComplexFloat,
              "store must be float16 or complex64");

  const int64_t n_blocks = store.size(0), n_planes = store.size(1),
                num_k = store.size(2), block_pixels = store.size(3);
  const int64_t num_m = split ? store.size(4) / 2 : store.size(4);
  const int64_t n_c = values.size(0), n_p = values.size(1);
  TORCH_CHECK(!split || store.size(4) % 2 == 0, "a float16 store's width is 2M");
  TORCH_CHECK(0 <= plane && plane < n_planes, "plane out of range");
  TORCH_CHECK(cell0 >= 0 && cell0 + n_c <= num_k * num_m, "cells out of range");
  TORCH_CHECK(col0 >= 0 && col0 + n_p <= n_blocks * block_pixels,
              "columns out of range");
  if (n_c == 0 || n_p == 0) return;

  const c10::cuda::CUDAGuard guard(store.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t k_rows = (cell0 + n_c - 1) / num_m - cell0 / num_m + 1;
  const int64_t y = k_rows * ((num_m + kRun - 1) / kRun);
  TORCH_CHECK(y <= 65535, "too many cells in one chunk");
  const dim3 grid(static_cast<unsigned>((n_p + kCols - 1) / kCols),
                  static_cast<unsigned>(y));
  const auto* src = reinterpret_cast<const float2*>(values.data_ptr());

  if (split) {
    store_write_kernel<__half, true><<<grid, kThreads, 0, stream>>>(
        src, values.stride(0), reinterpret_cast<__half*>(store.data_ptr()), n_c, n_p,
        cell0, col0, plane, n_planes, num_k, num_m, block_pixels);
  } else {
    store_write_kernel<float, false><<<grid, kThreads, 0, stream>>>(
        src, values.stride(0), reinterpret_cast<float*>(store.data_ptr()), n_c, n_p,
        cell0, col0, plane, n_planes, num_k, num_m, block_pixels);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
