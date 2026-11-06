/* lean_irfft_stats.cu
 *
 * Torch binding for the register-resident fused (frequency-padded inverse real FFT)
 * + (Parseval moments) + (max/argmax) kernel in lean_irfft_stats.cuh.
 *
 * Returns raw (s1, s2, argmax_packed); the packed (sortable value, index) maximum is
 * decoded in Python (packed.py) with ordinary vectorized bitwise tensor ops, a
 * P-sized elementwise op not worth a dedicated kernel.
 */

#include <cstring>

#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/complex.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "lean_irfft_stats.cuh"

// Packed (sortable(-inf), 0): the value best_packed is pre-filled with. The sortable
// map flips every bit of a negative float, so -inf (0xFF800000) maps to 0x007FFFFF.
static int64_t sentinel_packed_i64() {
  const unsigned int bits = 0xFF800000u;  // -inf
  const unsigned int sortable = bits ^ 0xFFFFFFFFu;
  const unsigned long long packed = static_cast<unsigned long long>(sortable) << 32;
  int64_t out;
  std::memcpy(&out, &packed, sizeof(out));
  return out;
}

/**
 * Register-resident fused psi-recovery + statistics for the GEMM's native
 * (NumFreq, P, Q) layout -- see lean_irfft_stats.cuh. Returns raw (s1, s2,
 * argmax_packed) (decoded in Python), and:
 *   - accepts complex64 (NumFreq, P, Q) OR complex32/float16 input: a complex32
 *     (chalf) tensor of shape (NumFreq, P, Q), or an equivalent float16 tensor of
 *     shape (NumFreq, P, 2Q) holding interleaved (re, im) pairs -- the layout the
 *     fp16 real-valued "4M" contraction in FeatureTiling.run produces directly;
 *   - any NumFreq in [1, num_psi/2 + 1] (runtime; bins above 64 fold onto the
 *     64-point transform exactly, the Nyquist bin is handled), num_psi in {128, 256, 512};
 *   - can accumulate straight into a caller-owned running state: pass `outs` =
 *     [corr_sum (P,) f32, corr_sum2 (P,) f32, best_packed (P,) i64] and the batch's
 *     global starting hypothesis index `hyp_offset`. s1/s2 are atomically added to
 *     the sums and the packed (value, (hyp_offset + q) * num_psi + psi) max is
 *     atomically merged into best_packed, so a whole stage of hypothesis batches
 *     reduces to the running state with no per-batch decode/accumulate kernels.
 *     With `outs` empty, fresh zero / sentinel-filled outputs are returned as usual.
 * DRAM-bandwidth bound.
 */
std::vector<torch::Tensor>
lean_irfft_stats_transposed(torch::Tensor c, int64_t num_psi, int64_t hyp_offset,
                            std::vector<torch::Tensor> outs) {
  TORCH_CHECK(c.is_cuda(), "c must be CUDA");
  TORCH_CHECK(c.dim() == 3, "c must be (NumFreq, P, Q) [or (NumFreq, P, 2Q) float16]");
  TORCH_CHECK(c.is_contiguous(),
              "c must be (NumFreq,P,Q)-contiguous; this entry point exists to avoid "
              "a copy");
  TORCH_CHECK(num_psi == 128 || num_psi == 256 || num_psi == 512,
              "lean_irfft_stats_transposed supports num_psi in {128, 256, 512}, got ",
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
  TORCH_CHECK(num_freq >= 1 && num_freq <= static_cast<unsigned int>(num_psi / 2 + 1),
              "lean_irfft_stats_transposed supports NumFreq in [1, num_psi/2 + 1], got ",
              num_freq, " at num_psi=", num_psi);
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
    if (r == 8)
      lean::launch_lean_transposed<8, float2>(ptr, s1_ptr, s2_ptr, packed_ptr,
                                              num_freq, p_total, q_total, off, stream);
    else if (r == 4)
      lean::launch_lean_transposed<4, float2>(ptr, s1_ptr, s2_ptr, packed_ptr,
                                              num_freq, p_total, q_total, off, stream);
    else
      lean::launch_lean_transposed<2, float2>(ptr, s1_ptr, s2_ptr, packed_ptr,
                                              num_freq, p_total, q_total, off, stream);
  } else {
    const __half2 *ptr = reinterpret_cast<const __half2 *>(c.data_ptr());
    if (r == 8)
      lean::launch_lean_transposed<8, __half2>(ptr, s1_ptr, s2_ptr, packed_ptr,
                                               num_freq, p_total, q_total, off, stream);
    else if (r == 4)
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

/**
 * On-device check of the generated unnormalized inverse DFTs (lean_fft_gen.cuh):
 * x complex64 (B, N), N in {64, 128, 256} -> IDFT_N(x) * N per row (matches
 * torch.fft.ifft(x, norm="forward")). Test/validation entry point only.
 */
torch::Tensor lean_debug_ifft(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kComplexFloat && x.dim() == 2,
              "x must be complex64 (B, N)");
  const c10::cuda::CUDAGuard device_guard(x.device());
  x = x.contiguous();
  const auto batch = static_cast<unsigned int>(x.size(0));
  const auto n = x.size(1);
  TORCH_CHECK(n == 64 || n == 128 || n == 256, "N must be 64, 128 or 256");
  torch::Tensor out = torch::empty_like(x);
  const float2 *in_ptr = reinterpret_cast<const float2 *>(x.data_ptr<c10::complex<float>>());
  float2 *out_ptr = reinterpret_cast<float2 *>(out.data_ptr<c10::complex<float>>());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const unsigned int blocks = (batch + 127) / 128;
  if (n == 64)
    lean::lean_debug_ifft_kernel<64><<<blocks, 128, 0, stream>>>(in_ptr, out_ptr, batch);
  else if (n == 128)
    lean::lean_debug_ifft_kernel<128><<<blocks, 128, 0, stream>>>(in_ptr, out_ptr, batch);
  else
    lean::lean_debug_ifft_kernel<256><<<blocks, 128, 0, stream>>>(in_ptr, out_ptr, batch);
  auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "lean_debug_ifft launch failed: ", cudaGetErrorString(err));
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() =
      "Register-resident fused frequency-padded inverse real FFT + Parseval moments "
      "+ max/argmax";
  m.def("lean_irfft_stats_transposed", &lean_irfft_stats_transposed,
        "Register-resident kernel for (NumFreq, P, Q) complex64/complex32/"
        "float16-pairs input, num_psi in {128, 256, 512}; optionally accumulates into "
        "caller-owned [corr_sum, corr_sum2, best_packed] with a global hyp_offset",
        pybind11::arg("c"), pybind11::arg("num_psi"), pybind11::arg("hyp_offset") = 0,
        pybind11::arg("outs") = std::vector<torch::Tensor>{});
  m.def("lean_debug_ifft", &lean_debug_ifft,
        "Test entry: unnormalized inverse DFT (N=64, 128 or 256) of complex64 (B, N) rows via "
        "the generated register-resident transforms",
        pybind11::arg("x"));
  m.def("lean_sentinel_packed", &lean_sentinel_packed,
        "int64 bit pattern of the packed (-inf, 0) sentinel best_packed is pre-filled "
        "with");
}
