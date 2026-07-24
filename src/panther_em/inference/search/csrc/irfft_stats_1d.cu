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

// ===========================================================================
//                         Config table + Python binding
// ===========================================================================

struct IrfftStatsConfigEntry {
  unsigned int num_psi;
  unsigned int num_freq;
  unsigned int fpb;
  unsigned int ept;
};

// Spectrum configuration table: (num_psi, num_freq, ept, fpb) tuples.
// NOTE: 'num_freq' cannot exceed 'num_psi/2 + 1' (Nyquist for RFFT)
// NOTE: When adding more configurations, update element number for array
// TODO: (future) Relax EPT and FPB to allow cuFFTDx to choose based on arch
static constexpr std::array<
    std::tuple<unsigned int, unsigned int, unsigned int, unsigned int>, 17>
    SUPPORTED_CONFIGS = {{
        {64, 16, 8, 8},
        {64, 32, 8, 8},
        {64, 33, 8, 8}, // Nyquist-bin filled, testing branch
        {128, 16, 8, 8},
        {128, 32, 8, 8},
        {128, 48, 8, 8},
        {128, 64, 8, 8},
        {128, 65, 8, 8}, // Nyquist-bin filled, testing branch
        {256, 16, 8, 8},
        {256, 32, 8, 8},
        {256, 48, 8, 8},
        {256, 64, 8, 8},
        {256, 80, 8, 8},
        {256, 96, 8, 8},
        {256, 112, 8, 8},
        {256, 128, 8, 8},
        {256, 129, 8, 8}, // Nyquist-bin filled, testing branch
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() =
      "Fused frequency-padded inverse real FFT + Parseval moments + max/argmax "
      "(cuFFTDx)";
  m.def("fused_irfft_stats", &fused_irfft_stats,
        "(s1, s2, argmax_packed) = fused psi-recovery + statistics-update for "
        "one hypothesis batch",
        pybind11::arg("c"), pybind11::arg("num_psi"));
  m.def("get_supported_configs", &get_supported_configs,
        "List of supported (num_psi, num_freq, fpb, ept) tuples");
}
