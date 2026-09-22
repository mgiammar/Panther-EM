/* irfft_stats_1d.cu
 *
 * Torch binding for the fused (frequency-padded inverse real FFT) + (Parseval
 * moments) + (max/argmax) kernel in fused_irfft_and_stats_kernel.cuh.
 *
 * Returns raw (s1, s2, argmax_packed) rather than decoding vmax/amax here --
 * the packed-atomicMax decode (see fused_irfft_and_stats_kernel.cuh's
 * pack_val_idx / sortable-float trick) is done in Python (loader.py) via
 * ordinary vectorized bitwise tensor ops, since it's a P-sized (not P*Q*num_psi
 * -sized) elementwise op and not worth a dedicated decode kernel.
 */

#include <cstring>

#include <array>
#include <functional>
#include <tuple>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/complex.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "fused_irfft_and_stats_kernel.cuh"
#include "lean_irfft_stats.cuh"
#include "zipfft_common.hpp"

// ---------------------------------------------------------------------------
// Architecture dispatch for fused kernel
// ---------------------------------------------------------------------------
template <unsigned int NPsi, unsigned int NumFreq, unsigned int FPB,
          unsigned int EPT>
void fused_irfft_stats_launch(const c10::complex<float> *c, float *s1,
                              float *s2, unsigned long long *argmax_packed,
                              unsigned int p_total, unsigned int q_total) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto arch = zipfft::get_cuda_device_arch();
  const float2 *c2 = reinterpret_cast<const float2 *>(c);

  /* clang-format off */
    switch (arch) {
#ifdef ENABLE_CUDA_ARCH_750
        case 750: {
            using Config = fused_stats::FusedIrfftStatsConfig<750, NPsi, NumFreq, FPB, EPT>;
            fused_stats::launch_fused_irfft_stats<Config>(
                reinterpret_cast<const typename Config::complex_type*>(c2), s1, s2, argmax_packed,
                p_total, q_total, stream);
            break;
        }
#endif
#ifdef ENABLE_CUDA_ARCH_800
        case 800: {
            using Config = fused_stats::FusedIrfftStatsConfig<800, NPsi, NumFreq, FPB, EPT>;
            fused_stats::launch_fused_irfft_stats<Config>(
                reinterpret_cast<const typename Config::complex_type*>(c2), s1, s2, argmax_packed,
                p_total, q_total, stream);
            break;
        }
#endif
#ifdef ENABLE_CUDA_ARCH_860
        case 860: {
            using Config = fused_stats::FusedIrfftStatsConfig<860, NPsi, NumFreq, FPB, EPT>;
            fused_stats::launch_fused_irfft_stats<Config>(
                reinterpret_cast<const typename Config::complex_type*>(c2), s1, s2, argmax_packed,
                p_total, q_total, stream);
            break;
        }
#endif
#ifdef ENABLE_CUDA_ARCH_870
        case 870: {
            using Config = fused_stats::FusedIrfftStatsConfig<870, NPsi, NumFreq, FPB, EPT>;
            fused_stats::launch_fused_irfft_stats<Config>(
                reinterpret_cast<const typename Config::complex_type*>(c2), s1, s2, argmax_packed,
                p_total, q_total, stream);
            break;
        }
#endif
#ifdef ENABLE_CUDA_ARCH_890
        case 890: {
            using Config = fused_stats::FusedIrfftStatsConfig<890, NPsi, NumFreq, FPB, EPT>;
            fused_stats::launch_fused_irfft_stats<Config>(
                reinterpret_cast<const typename Config::complex_type*>(c2), s1, s2, argmax_packed,
                p_total, q_total, stream);
            break;
        }
#endif
#ifdef ENABLE_CUDA_ARCH_900
        case 900: {
            using Config = fused_stats::FusedIrfftStatsConfig<900, NPsi, NumFreq, FPB, EPT>;
            fused_stats::launch_fused_irfft_stats<Config>(
                reinterpret_cast<const typename Config::complex_type*>(c2), s1, s2, argmax_packed,
                p_total, q_total, stream);
            break;
        }
#endif
#ifdef ENABLE_CUDA_ARCH_1000
        case 1000: {
            using Config = fused_stats::FusedIrfftStatsConfig<1000, NPsi, NumFreq, FPB, EPT>;
            fused_stats::launch_fused_irfft_stats<Config>(
                reinterpret_cast<const typename Config::complex_type*>(c2), s1, s2, argmax_packed,
                p_total, q_total, stream);
            break;
        }
#endif
#ifdef ENABLE_CUDA_ARCH_1030
        case 1030: {
            using Config = fused_stats::FusedIrfftStatsConfig<1030, NPsi, NumFreq, FPB, EPT>;
            fused_stats::launch_fused_irfft_stats<Config>(
                reinterpret_cast<const typename Config::complex_type*>(c2), s1, s2, argmax_packed,
                p_total, q_total, stream);
            break;
        }
#endif
#ifdef ENABLE_CUDA_ARCH_1200
        case 1200: {
            using Config = fused_stats::FusedIrfftStatsConfig<1200, NPsi, NumFreq, FPB, EPT>;
            fused_stats::launch_fused_irfft_stats<Config>(
                reinterpret_cast<const typename Config::complex_type*>(c2), s1, s2, argmax_packed,
                p_total, q_total, stream);
            break;
        }
#endif
#ifdef ENABLE_CUDA_ARCH_1210
        case 1210: {
            using Config = fused_stats::FusedIrfftStatsConfig<1210, NPsi, NumFreq, FPB, EPT>;
            fused_stats::launch_fused_irfft_stats<Config>(
                reinterpret_cast<const typename Config::complex_type*>(c2), s1, s2, argmax_packed,
                p_total, q_total, stream);
            break;
        }
#endif
        default:
            throw std::runtime_error("fused_irfft_stats: unsupported CUDA architecture: " +
                                     std::to_string(arch));
    }
  /* clang-format on */
  // SM Operator (cufftdx::SM<unsigned int CC>()) supported architectures):
  //   Turing: 750 (sm_75).
  //   Ampere: 800, 860 and 870 (sm_80, sm_86, sm_87).
  //   Ada: 890 (sm_89).
  //   Hopper: 900 (sm_90).
  //   Blackwell: 1000, 1030, 1200, 1210 (sm_100, sm_103, sm_120, sm_121).
}

// ---------------------------------------------------------------------------
// Architecture dispatch for the transposed-input ((NumFreq, P, Q)) kernel.
// Mirrors fused_irfft_stats_launch above exactly, just calling into
// launch_fused_irfft_stats_transposed instead.
// ---------------------------------------------------------------------------
template <unsigned int NPsi, unsigned int NumFreq, unsigned int FPB,
          unsigned int EPT>
void fused_irfft_stats_launch_transposed(const c10::complex<float> *c,
                                         float *s1, float *s2,
                                         unsigned long long *argmax_packed,
                                         unsigned int p_total,
                                         unsigned int q_total) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto arch = zipfft::get_cuda_device_arch();
  const float2 *c2 = reinterpret_cast<const float2 *>(c);

  /* clang-format off */
    switch (arch) {
#ifdef ENABLE_CUDA_ARCH_750
        case 750: {
            using Config = fused_stats::FusedIrfftStatsConfig<750, NPsi, NumFreq, FPB, EPT>;
            fused_stats::launch_fused_irfft_stats_transposed<Config>(
                reinterpret_cast<const typename Config::complex_type*>(c2), s1, s2, argmax_packed,
                p_total, q_total, stream);
            break;
        }
#endif
#ifdef ENABLE_CUDA_ARCH_800
        case 800: {
            using Config = fused_stats::FusedIrfftStatsConfig<800, NPsi, NumFreq, FPB, EPT>;
            fused_stats::launch_fused_irfft_stats_transposed<Config>(
                reinterpret_cast<const typename Config::complex_type*>(c2), s1, s2, argmax_packed,
                p_total, q_total, stream);
            break;
        }
#endif
#ifdef ENABLE_CUDA_ARCH_860
        case 860: {
            using Config = fused_stats::FusedIrfftStatsConfig<860, NPsi, NumFreq, FPB, EPT>;
            fused_stats::launch_fused_irfft_stats_transposed<Config>(
                reinterpret_cast<const typename Config::complex_type*>(c2), s1, s2, argmax_packed,
                p_total, q_total, stream);
            break;
        }
#endif
#ifdef ENABLE_CUDA_ARCH_870
        case 870: {
            using Config = fused_stats::FusedIrfftStatsConfig<870, NPsi, NumFreq, FPB, EPT>;
            fused_stats::launch_fused_irfft_stats_transposed<Config>(
                reinterpret_cast<const typename Config::complex_type*>(c2), s1, s2, argmax_packed,
                p_total, q_total, stream);
            break;
        }
#endif
#ifdef ENABLE_CUDA_ARCH_890
        case 890: {
            using Config = fused_stats::FusedIrfftStatsConfig<890, NPsi, NumFreq, FPB, EPT>;
            fused_stats::launch_fused_irfft_stats_transposed<Config>(
                reinterpret_cast<const typename Config::complex_type*>(c2), s1, s2, argmax_packed,
                p_total, q_total, stream);
            break;
        }
#endif
#ifdef ENABLE_CUDA_ARCH_900
        case 900: {
            using Config = fused_stats::FusedIrfftStatsConfig<900, NPsi, NumFreq, FPB, EPT>;
            fused_stats::launch_fused_irfft_stats_transposed<Config>(
                reinterpret_cast<const typename Config::complex_type*>(c2), s1, s2, argmax_packed,
                p_total, q_total, stream);
            break;
        }
#endif
#ifdef ENABLE_CUDA_ARCH_1000
        case 1000: {
            using Config = fused_stats::FusedIrfftStatsConfig<1000, NPsi, NumFreq, FPB, EPT>;
            fused_stats::launch_fused_irfft_stats_transposed<Config>(
                reinterpret_cast<const typename Config::complex_type*>(c2), s1, s2, argmax_packed,
                p_total, q_total, stream);
            break;
        }
#endif
#ifdef ENABLE_CUDA_ARCH_1030
        case 1030: {
            using Config = fused_stats::FusedIrfftStatsConfig<1030, NPsi, NumFreq, FPB, EPT>;
            fused_stats::launch_fused_irfft_stats_transposed<Config>(
                reinterpret_cast<const typename Config::complex_type*>(c2), s1, s2, argmax_packed,
                p_total, q_total, stream);
            break;
        }
#endif
#ifdef ENABLE_CUDA_ARCH_1200
        case 1200: {
            using Config = fused_stats::FusedIrfftStatsConfig<1200, NPsi, NumFreq, FPB, EPT>;
            fused_stats::launch_fused_irfft_stats_transposed<Config>(
                reinterpret_cast<const typename Config::complex_type*>(c2), s1, s2, argmax_packed,
                p_total, q_total, stream);
            break;
        }
#endif
#ifdef ENABLE_CUDA_ARCH_1210
        case 1210: {
            using Config = fused_stats::FusedIrfftStatsConfig<1210, NPsi, NumFreq, FPB, EPT>;
            fused_stats::launch_fused_irfft_stats_transposed<Config>(
                reinterpret_cast<const typename Config::complex_type*>(c2), s1, s2, argmax_packed,
                p_total, q_total, stream);
            break;
        }
#endif
        default:
            throw std::runtime_error("fused_irfft_stats_transposed: unsupported CUDA architecture: " +
                                     std::to_string(arch));
    }
  /* clang-format on */
}

// ===========================================================================
//                         Config table + Python binding
// ===========================================================================

struct IrfftStatsConfigEntry {
  unsigned int num_psi;
  unsigned int num_freq;
  unsigned int fpb;
  unsigned int ept;
};

// Spectrum configuration table for the cuFFTDx block-FFT kernel: (num_psi, num_freq,
// ept, fpb) tuples. NOTE: 'num_freq' cannot exceed 'num_psi/2 + 1' (Nyquist for RFFT).
// NOTE: When adding more configurations, update element number for array.
//
// EPT (cuFFTDx ElementsPerThread) was retuned on an RTX 6000 Ada (Sep 2026): the
// kernel is bound by the FFT's inter-thread shared-memory exchanges, so fewer threads
// per FFT wins -- EPT=32 for n_psi=256 and EPT=16 for n_psi=128 (8 threads per FFT)
// are 1.4-2.4x faster than the previous uniform EPT=8, bit-for-bit identical results.
// This kernel is now the fallback for shapes the register-resident lean kernel
// (lean_irfft_stats.cuh: n_psi in {128, 256}, num_freq <= 64) does not cover.
static constexpr std::array<
    std::tuple<unsigned int, unsigned int, unsigned int, unsigned int>, 17>
    SUPPORTED_CONFIGS = {{
        {64, 16, 8, 8},
        {64, 32, 8, 8},
        {64, 33, 8, 8}, // Nyquist-bin filled, testing branch
        {128, 16, 16, 8},
        {128, 32, 16, 8},
        {128, 48, 16, 8},
        {128, 64, 16, 8},
        {128, 65, 16, 8}, // Nyquist-bin filled, testing branch
        {256, 16, 32, 8},
        {256, 32, 32, 8},
        {256, 48, 32, 8},
        {256, 64, 32, 8},
        {256, 80, 32, 8},
        {256, 96, 32, 8},
        {256, 112, 32, 8},
        {256, 128, 32, 8},
        {256, 129, 32, 8}, // Nyquist-bin filled, testing branch
    }};

template <unsigned int NPsi, unsigned int NumFreq, unsigned int FPB,
          unsigned int EPT>
void dispatch_stats(const c10::complex<float> *c, float *s1, float *s2,
                    unsigned long long *argmax_packed, unsigned int p_total,
                    unsigned int q_total) {
  fused_irfft_stats_launch<NPsi, NumFreq, FPB, EPT>(c, s1, s2, argmax_packed,
                                                    p_total, q_total);
}

template <std::size_t... Is>
constexpr auto make_dispatch_table(std::index_sequence<Is...>) {
  return std::array<
      std::pair<IrfftStatsConfigEntry,
                std::function<void(const c10::complex<float> *, float *,
                                   float *, unsigned long long *, unsigned int,
                                   unsigned int)>>,
      sizeof...(Is)>{
      {{IrfftStatsConfigEntry{std::get<0>(SUPPORTED_CONFIGS[Is]),
                              std::get<1>(SUPPORTED_CONFIGS[Is]),
                              std::get<3>(SUPPORTED_CONFIGS[Is]),
                              std::get<2>(SUPPORTED_CONFIGS[Is])},
        []() {
          constexpr auto c = SUPPORTED_CONFIGS[Is];
          return dispatch_stats<std::get<0>(c), std::get<1>(c), std::get<3>(c),
                                std::get<2>(c)>;
        }()}...}};
}

static const auto dispatch_table =
    make_dispatch_table(std::make_index_sequence<SUPPORTED_CONFIGS.size()>{});

static std::function<void(const c10::complex<float> *, float *, float *,
                          unsigned long long *, unsigned int, unsigned int)>
get_fn(unsigned int num_psi, unsigned int num_freq) {
  for (const auto &e : dispatch_table) {
    if (e.first.num_psi == num_psi && e.first.num_freq == num_freq)
      return e.second;
  }
  return nullptr;
}

// ---------------------------------------------------------------------------
// Dispatch table for the transposed-input ((NumFreq, P, Q)) kernel. Reuses
// SUPPORTED_CONFIGS unchanged -- same (num_psi, num_freq, fpb, ept) space,
// just a different kernel/launch path.
// ---------------------------------------------------------------------------
template <unsigned int NPsi, unsigned int NumFreq, unsigned int FPB,
          unsigned int EPT>
void dispatch_stats_transposed(const c10::complex<float> *c, float *s1,
                               float *s2, unsigned long long *argmax_packed,
                               unsigned int p_total, unsigned int q_total) {
  fused_irfft_stats_launch_transposed<NPsi, NumFreq, FPB, EPT>(
      c, s1, s2, argmax_packed, p_total, q_total);
}

template <std::size_t... Is>
constexpr auto make_dispatch_table_transposed(std::index_sequence<Is...>) {
  return std::array<
      std::pair<IrfftStatsConfigEntry,
                std::function<void(const c10::complex<float> *, float *,
                                   float *, unsigned long long *, unsigned int,
                                   unsigned int)>>,
      sizeof...(Is)>{
      {{IrfftStatsConfigEntry{std::get<0>(SUPPORTED_CONFIGS[Is]),
                              std::get<1>(SUPPORTED_CONFIGS[Is]),
                              std::get<3>(SUPPORTED_CONFIGS[Is]),
                              std::get<2>(SUPPORTED_CONFIGS[Is])},
        []() {
          constexpr auto c = SUPPORTED_CONFIGS[Is];
          return dispatch_stats_transposed<std::get<0>(c), std::get<1>(c),
                                           std::get<3>(c), std::get<2>(c)>;
        }()}...}};
}

static const auto dispatch_table_transposed = make_dispatch_table_transposed(
    std::make_index_sequence<SUPPORTED_CONFIGS.size()>{});

static std::function<void(const c10::complex<float> *, float *, float *,
                          unsigned long long *, unsigned int, unsigned int)>
get_fn_transposed(unsigned int num_psi, unsigned int num_freq) {
  for (const auto &e : dispatch_table_transposed) {
    if (e.first.num_psi == num_psi && e.first.num_freq == num_freq)
      return e.second;
  }
  return nullptr;
}

std::vector<std::tuple<int, int, int, int>> get_supported_configs() {
  std::vector<std::tuple<int, int, int, int>> v;
  v.reserve(SUPPORTED_CONFIGS.size());
  for (const auto &c : SUPPORTED_CONFIGS)
    v.emplace_back(std::get<0>(c), std::get<1>(c), std::get<2>(c),
                   std::get<3>(c));
  return v;
}

static int64_t sentinel_packed_i64() {
  unsigned long long s = fused_stats::sentinel_packed_host();
  int64_t out;
  std::memcpy(&out, &s, sizeof(out));
  return out;
}

/**
 * Fused psi-recovery + statistics-update for one hypothesis batch, matching
 * panther_em.inference.search.statistics._reduce_stats's I/O contract:
 *
 *     corr = torch.fft.irfft(C, n=num_psi, dim=-1, norm="forward")
 *     s1, s2, vmax, amax = _reduce_stats(corr, torch.view_as_real(C))
 *
 * except computed directly from C (corr is never materialized).
 *
 * @param c      complex64, CUDA, contiguous, shape (P, Q, NumFreq). The
 *               (un-padded) low-frequency spectrum for one hypothesis batch.
 * @param num_psi  full in-plane-angle length. (num_psi, NumFreq) must be in
 *               get_supported_configs().
 * @return (s1, s2, vmax, amax): s1/s2 float32 (P,), vmax float32 (P,), amax
 *         int64 (P,) -- flat index into the (Q, num_psi) grid for THIS batch
 *         (matching _reduce_stats's own un-decoded amax convention).
 */
std::vector<torch::Tensor> fused_irfft_stats(torch::Tensor c, int64_t num_psi) {
  TORCH_CHECK(c.is_cuda(), "c must be CUDA");
  TORCH_CHECK(c.dtype() == torch::kComplexFloat, "c must be complex64");
  TORCH_CHECK(c.dim() == 3, "c must be (P, Q, NumFreq)");

  const c10::cuda::CUDAGuard device_guard(c.device());

  c = c.contiguous();

  const auto p_total = static_cast<unsigned int>(c.size(0));
  const auto q_total = static_cast<unsigned int>(c.size(1));
  const auto num_freq = static_cast<unsigned int>(c.size(2));

  auto fn = get_fn(static_cast<unsigned int>(num_psi), num_freq);
  TORCH_CHECK(fn != nullptr, "Unsupported (num_psi, num_freq) = (", num_psi,
              ", ", num_freq, "). See get_supported_configs().");

  auto opts_f32 =
      torch::TensorOptions().dtype(torch::kFloat32).device(c.device());
  auto opts_i64 =
      torch::TensorOptions().dtype(torch::kInt64).device(c.device());

  torch::Tensor s1 = torch::zeros({p_total}, opts_f32);
  torch::Tensor s2 = torch::zeros({p_total}, opts_f32);
  torch::Tensor argmax_packed =
      torch::full({p_total}, sentinel_packed_i64(), opts_i64);

  const c10::complex<float> *c_ptr = c.data_ptr<c10::complex<float>>();
  float *s1_ptr = s1.data_ptr<float>();
  float *s2_ptr = s2.data_ptr<float>();
  unsigned long long *packed_ptr =
      reinterpret_cast<unsigned long long *>(argmax_packed.data_ptr<int64_t>());

  fn(c_ptr, s1_ptr, s2_ptr, packed_ptr, p_total, q_total);

  return {s1, s2, argmax_packed};
}

/**
 * Zero-copy variant of fused_irfft_stats for input already laid out as
 * (NumFreq, P, Q) contiguous -- the layout a cuBLAS strided-batched GEMM
 * produces natively (frequency as the batch axis), with no permute/copy
 * needed to feed this kernel. Same I/O contract as fused_irfft_stats
 * otherwise (see its docstring); (s1, s2, vmax, amax) values are identical
 * for the same logical spectrum, only the input tensor's physical layout
 * differs.
 *
 * @param c      complex64, CUDA, contiguous, shape (NumFreq, P, Q).
 *               Deliberately NOT materialized via .contiguous() here --
 *               that would defeat the purpose of this entry point. Callers
 *               that can't guarantee contiguity should use
 *               fused_irfft_stats() instead.
 * @param num_psi  full in-plane-angle length. (num_psi, NumFreq) must be in
 *               get_supported_configs().
 * @return (s1, s2, argmax_packed), same shapes/dtypes as fused_irfft_stats.
 */
std::vector<torch::Tensor> fused_irfft_stats_transposed(torch::Tensor c,
                                                         int64_t num_psi) {
  TORCH_CHECK(c.is_cuda(), "c must be CUDA");
  TORCH_CHECK(c.dtype() == torch::kComplexFloat, "c must be complex64");
  TORCH_CHECK(c.dim() == 3, "c must be (NumFreq, P, Q)");
  TORCH_CHECK(c.is_contiguous(),
              "c must already be (NumFreq,P,Q)-contiguous; this entry point "
              "exists specifically to avoid a copy -- call fused_irfft_stats() "
              "instead if c is not already contiguous");

  const c10::cuda::CUDAGuard device_guard(c.device());

  const auto num_freq = static_cast<unsigned int>(c.size(0));
  const auto p_total = static_cast<unsigned int>(c.size(1));
  const auto q_total = static_cast<unsigned int>(c.size(2));

  auto fn = get_fn_transposed(static_cast<unsigned int>(num_psi), num_freq);
  TORCH_CHECK(fn != nullptr, "Unsupported (num_psi, num_freq) = (", num_psi,
              ", ", num_freq, "). See get_supported_configs().");

  auto opts_f32 =
      torch::TensorOptions().dtype(torch::kFloat32).device(c.device());
  auto opts_i64 =
      torch::TensorOptions().dtype(torch::kInt64).device(c.device());

  torch::Tensor s1 = torch::zeros({p_total}, opts_f32);
  torch::Tensor s2 = torch::zeros({p_total}, opts_f32);
  torch::Tensor argmax_packed =
      torch::full({p_total}, sentinel_packed_i64(), opts_i64);

  const c10::complex<float> *c_ptr = c.data_ptr<c10::complex<float>>();
  float *s1_ptr = s1.data_ptr<float>();
  float *s2_ptr = s2.data_ptr<float>();
  unsigned long long *packed_ptr =
      reinterpret_cast<unsigned long long *>(argmax_packed.data_ptr<int64_t>());

  fn(c_ptr, s1_ptr, s2_ptr, packed_ptr, p_total, q_total);

  return {s1, s2, argmax_packed};
}

/**
 * Register-resident fused psi-recovery + statistics for the GEMM's native
 * (NumFreq, P, Q) layout -- see lean_irfft_stats.cuh. Same I/O contract as
 * fused_irfft_stats_transposed(), but:
 *   - accepts complex64 (NumFreq, P, Q) OR complex32/float16 input: a complex32
 *     (chalf) tensor of shape (NumFreq, P, Q), or an equivalent float16 tensor of
 *     shape (NumFreq, P, 2Q) holding interleaved (re, im) pairs -- the layout the
 *     fp16 real-valued "4M" contraction in FeatureTiling.run produces directly;
 *   - any NumFreq in [1, 64] (runtime), num_psi in {128, 256};
 *   - can accumulate straight into a caller-owned running state: pass `outs` =
 *     [corr_sum (P,) f32, corr_sum2 (P,) f32, best_packed (P,) i64] and the batch's
 *     global starting hypothesis index `hyp_offset`. s1/s2 are atomically added to
 *     the sums and the packed (value, (hyp_offset + q) * num_psi + psi) max is
 *     atomically merged into best_packed, so a whole stage of hypothesis batches
 *     reduces to the running state with no per-batch decode/accumulate kernels.
 *     With `outs` empty, fresh zero / sentinel-filled outputs are returned as usual.
 * Roughly 6x faster than the cuFFTDx kernel at n_psi=256 and DRAM-bandwidth bound.
 */
std::vector<torch::Tensor>
lean_irfft_stats_transposed(torch::Tensor c, int64_t num_psi, int64_t hyp_offset,
                            std::vector<torch::Tensor> outs) {
  TORCH_CHECK(c.is_cuda(), "c must be CUDA");
  TORCH_CHECK(c.dim() == 3, "c must be (NumFreq, P, Q) [or (NumFreq, P, 2Q) float16]");
  TORCH_CHECK(c.is_contiguous(),
              "c must be (NumFreq,P,Q)-contiguous; this entry point exists to avoid "
              "a copy");
  TORCH_CHECK(num_psi == 128 || num_psi == 256,
              "lean_irfft_stats_transposed supports num_psi in {128, 256}, got ",
              num_psi);
  const bool is_c64 = c.dtype() == torch::kComplexFloat;
  const bool is_c32 = c.dtype() == torch::kComplexHalf;
  const bool is_f16 = c.dtype() == torch::kFloat16;
  TORCH_CHECK(is_c64 || is_c32 || is_f16,
              "c must be complex64, complex32 or float16 (interleaved pairs)");
  if (is_f16)
    TORCH_CHECK(c.size(2) % 2 == 0, "float16 input must be (NumFreq, P, 2Q)");

  const c10::cuda::CUDAGuard device_guard(c.device());

  const auto num_freq = static_cast<unsigned int>(c.size(0));
  const auto p_total = static_cast<unsigned int>(c.size(1));
  const auto q_total =
      static_cast<unsigned int>(is_f16 ? c.size(2) / 2 : c.size(2));
  TORCH_CHECK(num_freq >= 1 && num_freq <= 64,
              "lean_irfft_stats_transposed supports NumFreq in [1, 64], got ",
              num_freq);
  TORCH_CHECK(hyp_offset >= 0 &&
                  (hyp_offset + static_cast<int64_t>(q_total)) * num_psi <
                      (int64_t(1) << 32),
              "(hyp_offset + Q) * num_psi must fit in 32 bits");

  torch::Tensor s1, s2, argmax_packed;
  if (outs.size() == 3) {
    s1 = outs[0]; s2 = outs[1]; argmax_packed = outs[2];
    TORCH_CHECK(s1.dtype() == torch::kFloat32 && s2.dtype() == torch::kFloat32 &&
                    argmax_packed.dtype() == torch::kInt64,
                "outs must be (float32, float32, int64)");
    TORCH_CHECK(s1.numel() == p_total && s2.numel() == p_total &&
                    argmax_packed.numel() == p_total,
                "outs must each have P elements");
    TORCH_CHECK(s1.is_contiguous() && s2.is_contiguous() &&
                    argmax_packed.is_contiguous(),
                "outs must be contiguous");
  } else {
    TORCH_CHECK(outs.empty(), "outs must be empty or [s1, s2, argmax_packed]");
    auto opts_f32 =
        torch::TensorOptions().dtype(torch::kFloat32).device(c.device());
    auto opts_i64 =
        torch::TensorOptions().dtype(torch::kInt64).device(c.device());
    s1 = torch::zeros({p_total}, opts_f32);
    s2 = torch::zeros({p_total}, opts_f32);
    argmax_packed = torch::full({p_total}, sentinel_packed_i64(), opts_i64);
  }
  float *s1_ptr = s1.data_ptr<float>();
  float *s2_ptr = s2.data_ptr<float>();
  unsigned long long *packed_ptr =
      reinterpret_cast<unsigned long long *>(argmax_packed.data_ptr<int64_t>());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto off = static_cast<unsigned int>(hyp_offset);

  const unsigned int r = static_cast<unsigned int>(num_psi) / 64;
  if (is_c64) {
    const float2 *ptr =
        reinterpret_cast<const float2 *>(c.data_ptr<c10::complex<float>>());
    if (r == 4)
      lean::launch_lean_transposed<4, float2>(ptr, s1_ptr, s2_ptr, packed_ptr,
                                              num_freq, p_total, q_total, off, stream);
    else
      lean::launch_lean_transposed<2, float2>(ptr, s1_ptr, s2_ptr, packed_ptr,
                                              num_freq, p_total, q_total, off, stream);
  } else {
    const __half2 *ptr = reinterpret_cast<const __half2 *>(c.data_ptr());
    if (r == 4)
      lean::launch_lean_transposed<4, __half2>(ptr, s1_ptr, s2_ptr, packed_ptr,
                                               num_freq, p_total, q_total, off, stream);
    else
      lean::launch_lean_transposed<2, __half2>(ptr, s1_ptr, s2_ptr, packed_ptr,
                                               num_freq, p_total, q_total, off, stream);
  }
  auto launch_err = cudaGetLastError();
  TORCH_CHECK(launch_err == cudaSuccess, "lean_irfft_stats_transposed launch failed: ",
              cudaGetErrorString(launch_err));
  return {s1, s2, argmax_packed};
}

int64_t lean_sentinel_packed() { return sentinel_packed_i64(); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() =
      "Fused frequency-padded inverse real FFT + Parseval moments + max/argmax "
      "(cuFFTDx)";
  m.def("fused_irfft_stats", &fused_irfft_stats,
        "(s1, s2, argmax_packed) = fused psi-recovery + statistics-update for "
        "one hypothesis batch",
        pybind11::arg("c"), pybind11::arg("num_psi"));
  m.def("fused_irfft_stats_transposed", &fused_irfft_stats_transposed,
        "Zero-copy variant of fused_irfft_stats for (NumFreq, P, Q)-contiguous "
        "input",
        pybind11::arg("c"), pybind11::arg("num_psi"));
  m.def("lean_irfft_stats_transposed", &lean_irfft_stats_transposed,
        "Register-resident variant for (NumFreq<=64, P, Q) complex64/complex32/"
        "float16-pairs input, num_psi in {128, 256}; optionally accumulates into "
        "caller-owned [corr_sum, corr_sum2, best_packed] with a global hyp_offset",
        pybind11::arg("c"), pybind11::arg("num_psi"), pybind11::arg("hyp_offset") = 0,
        pybind11::arg("outs") = std::vector<torch::Tensor>{});
  m.def("lean_sentinel_packed", &lean_sentinel_packed,
        "int64 bit pattern of the packed (-inf, 0) sentinel best_packed is pre-filled "
        "with");
  m.def("get_supported_configs", &get_supported_configs,
        "List of supported (num_psi, num_freq, fpb, ept) tuples for the cuFFTDx "
        "block-FFT kernel");
}
