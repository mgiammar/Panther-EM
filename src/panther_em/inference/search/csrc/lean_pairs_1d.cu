/* lean_pairs_1d.cu
 *
 * Torch binding for the per-pair emission variant of the register-resident
 * iRFFT + statistics kernel (lean_irfft_stats.cuh, EMIT=true): the same per-pixel
 * accumulation into caller-owned [corr_sum, corr_sum2, best_packed] as
 * irfft_stats_1d.cu's lean_irfft_stats_transposed, plus, per (pixel, hypothesis)
 * pair, the maximum of the recovered correlogram over psi written to a caller-owned
 * (P, Q) float16/float32 slab view with an arbitrary row stride.
 *
 * Built as its own extension (panther_em_lean_pairs): it needs no cuFFTDx headers,
 * its instantiations never touch the dense kernel's codegen, and its compile time is
 * only paid when a schedule emits pairs. Nothing is allocated per call, so the entry
 * point is CUDA-graph safe.
 */

#include <cstring>

#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/complex.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "lean_irfft_stats.cuh"

namespace {

// Packed encoding of the (-inf, 0) sentinel best_packed must be pre-filled with
// (identical to fused_irfft_and_stats_kernel.cuh's sentinel_packed_host).
int64_t sentinel_packed_i64() {
  unsigned int bits = 0xFF800000u;  // -inf
  unsigned int sortable = bits ^ 0xFFFFFFFFu;  // sign bit set -> flip all bits
  unsigned long long s = static_cast<unsigned long long>(sortable) << 32;
  int64_t out;
  std::memcpy(&out, &s, sizeof(out));
  return out;
}

template <typename InT, typename OutT>
void dispatch(const InT *ptr, float *s1, float *s2, unsigned long long *pk,
              unsigned num_freq, unsigned p_total, unsigned q_total, unsigned off,
              unsigned r, cudaStream_t stream, OutT *pair_out, size_t pair_ld) {
  if (r == 4)
    lean::launch_lean_transposed<4, InT, true, OutT>(ptr, s1, s2, pk, num_freq, p_total,
                                                     q_total, off, stream, pair_out, pair_ld);
  else
    lean::launch_lean_transposed<2, InT, true, OutT>(ptr, s1, s2, pk, num_freq, p_total,
                                                     q_total, off, stream, pair_out, pair_ld);
}

}  // namespace

/**
 * lean_irfft_stats_pairs_transposed(c, num_psi, hyp_offset, outs, pair_out)
 *
 * c        : contiguous (NumFreq, P, Q) complex64 / complex32, or (NumFreq, P, 2Q) float16.
 * outs     : [corr_sum (P,) f32, corr_sum2 (P,) f32, best_packed (P,) i64] -- required,
 *            accumulated in place exactly like lean_irfft_stats_transposed.
 * pair_out : (P, Q) float16 or float32 view with stride(1) == 1; row stride may exceed Q
 *            (a column window of a wider (P, Q_total) slab). Written, never read.
 * Returns outs (for symmetry with the dense entry point).
 */
std::vector<torch::Tensor>
lean_irfft_stats_pairs_transposed(torch::Tensor c, int64_t num_psi, int64_t hyp_offset,
                                  std::vector<torch::Tensor> outs, torch::Tensor pair_out) {
  TORCH_CHECK(c.is_cuda(), "c must be CUDA");
  TORCH_CHECK(c.dim() == 3, "c must be (NumFreq, P, Q) [or (NumFreq, P, 2Q) float16]");
  TORCH_CHECK(c.is_contiguous(), "c must be (NumFreq,P,Q)-contiguous");
  TORCH_CHECK(num_psi == 128 || num_psi == 256,
              "lean_irfft_stats_pairs_transposed supports num_psi in {128, 256}, got ",
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
  const auto q_total = static_cast<unsigned int>(is_f16 ? c.size(2) / 2 : c.size(2));
  TORCH_CHECK(num_freq >= 1 && num_freq <= static_cast<unsigned int>(num_psi / 2 + 1),
              "NumFreq must lie in [1, num_psi/2 + 1], got ", num_freq, " at num_psi=",
              num_psi);
  TORCH_CHECK(hyp_offset >= 0 &&
                  (hyp_offset + static_cast<int64_t>(q_total)) * num_psi < (int64_t(1) << 32),
              "(hyp_offset + Q) * num_psi must fit in 32 bits");

  TORCH_CHECK(outs.size() == 3, "outs = [corr_sum, corr_sum2, best_packed] is required");
  torch::Tensor s1 = outs[0], s2 = outs[1], argmax_packed = outs[2];
  TORCH_CHECK(s1.dtype() == torch::kFloat32 && s2.dtype() == torch::kFloat32 &&
                  argmax_packed.dtype() == torch::kInt64,
              "outs must be (float32, float32, int64)");
  TORCH_CHECK(s1.numel() == p_total && s2.numel() == p_total && argmax_packed.numel() == p_total,
              "outs must each have P elements");
  TORCH_CHECK(s1.is_contiguous() && s2.is_contiguous() && argmax_packed.is_contiguous(),
              "outs must be contiguous");
  TORCH_CHECK(s1.device() == c.device() && s2.device() == c.device() &&
                  argmax_packed.device() == c.device(),
              "outs must be on c's device");

  TORCH_CHECK(pair_out.is_cuda() && pair_out.device() == c.device(),
              "pair_out must be on c's device");
  TORCH_CHECK(pair_out.dim() == 2 && pair_out.size(0) == p_total && pair_out.size(1) == q_total,
              "pair_out must be (P, Q) = (", p_total, ", ", q_total, "), got ", pair_out.sizes());
  TORCH_CHECK(pair_out.stride(1) == 1, "pair_out must be unit-stride along Q");
  TORCH_CHECK(pair_out.stride(0) >= static_cast<int64_t>(q_total) || p_total == 1,
              "pair_out row stride must be >= Q");
  const bool out_f16 = pair_out.dtype() == torch::kFloat16;
  TORCH_CHECK(out_f16 || pair_out.dtype() == torch::kFloat32,
              "pair_out must be float16 or float32");
  const auto pair_ld = static_cast<size_t>(pair_out.stride(0));

  float *s1_ptr = s1.data_ptr<float>();
  float *s2_ptr = s2.data_ptr<float>();
  unsigned long long *packed_ptr =
      reinterpret_cast<unsigned long long *>(argmax_packed.data_ptr<int64_t>());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto off = static_cast<unsigned int>(hyp_offset);
  const unsigned int r = static_cast<unsigned int>(num_psi) / 64;

  if (is_c64) {
    const float2 *ptr = reinterpret_cast<const float2 *>(c.data_ptr<c10::complex<float>>());
    if (out_f16)
      dispatch<float2, __half>(ptr, s1_ptr, s2_ptr, packed_ptr, num_freq, p_total, q_total, off,
                               r, stream, reinterpret_cast<__half *>(pair_out.data_ptr()),
                               pair_ld);
    else
      dispatch<float2, float>(ptr, s1_ptr, s2_ptr, packed_ptr, num_freq, p_total, q_total, off,
                              r, stream, pair_out.data_ptr<float>(), pair_ld);
  } else {
    const __half2 *ptr = reinterpret_cast<const __half2 *>(c.data_ptr());
    if (out_f16)
      dispatch<__half2, __half>(ptr, s1_ptr, s2_ptr, packed_ptr, num_freq, p_total, q_total, off,
                                r, stream, reinterpret_cast<__half *>(pair_out.data_ptr()),
                                pair_ld);
    else
      dispatch<__half2, float>(ptr, s1_ptr, s2_ptr, packed_ptr, num_freq, p_total, q_total, off,
                               r, stream, pair_out.data_ptr<float>(), pair_ld);
  }
  auto launch_err = cudaGetLastError();
  TORCH_CHECK(launch_err == cudaSuccess, "lean_irfft_stats_pairs_transposed launch failed: ",
              cudaGetErrorString(launch_err));
  return {s1, s2, argmax_packed};
}

int64_t lean_pairs_sentinel_packed() { return sentinel_packed_i64(); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Register-resident iRFFT + statistics kernel with per-pair max-over-psi emission";
  m.def("lean_irfft_stats_pairs_transposed", &lean_irfft_stats_pairs_transposed,
        "Accumulate [corr_sum, corr_sum2, best_packed] in place (global hyp_offset) and "
        "write each (pixel, hypothesis) pair's max over psi into pair_out (P, Q)",
        pybind11::arg("c"), pybind11::arg("num_psi"), pybind11::arg("hyp_offset"),
        pybind11::arg("outs"), pybind11::arg("pair_out"));
  m.def("lean_pairs_sentinel_packed", &lean_pairs_sentinel_packed,
        "int64 bit pattern of the packed (-inf, 0) sentinel best_packed is pre-filled with");
}
