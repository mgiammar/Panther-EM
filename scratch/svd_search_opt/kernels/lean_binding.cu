#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <cstring>
#include <vector>
#include "lean_irfft_stats.cuh"
using ull = unsigned long long;
using FnF = void (*)(const float2*, float*, float*, ull*, unsigned, unsigned, cudaStream_t);
using FnH = void (*)(const __half2*, float*, float*, ull*, unsigned, unsigned, cudaStream_t);
struct Entry { unsigned F, R; FnF f; FnH hf; };
#define E(F_, R_) Entry{F_, R_, &lean::launch_lean_transposed<F_, R_, float2>, &lean::launch_lean_transposed<F_, R_, __half2>}
static const Entry TABLE[] = {E(64,4), E(64,2), E(48,4), E(48,2), E(32,4), E(32,2), E(16,4), E(16,2)};
static const Entry* find(unsigned F, unsigned R) { for (auto& e : TABLE) if (e.F == F && e.R == R) return &e; return nullptr; }
static int64_t sentinel_i64() { unsigned b = 0xFF800000u ^ 0xFFFFFFFFu; ull s = (ull)b << 32; int64_t o; std::memcpy(&o, &s, 8); return o; }
static std::vector<torch::Tensor> outputs(unsigned P, const torch::Tensor& like, std::vector<torch::Tensor> outs) {
  if (outs.size() == 3) return outs;
  auto f = torch::TensorOptions().dtype(torch::kFloat32).device(like.device());
  auto i = torch::TensorOptions().dtype(torch::kInt64).device(like.device());
  return {torch::zeros({P}, f), torch::zeros({P}, f), torch::full({P}, sentinel_i64(), i)};
}
// c: complex64 (F,P,Q) or float16 (F,P,2Q), contiguous.
std::vector<torch::Tensor> lean_transposed(torch::Tensor c, int64_t num_psi, std::vector<torch::Tensor> outs) {
  TORCH_CHECK(c.is_cuda() && c.dim() == 3 && c.is_contiguous(), "need contiguous 3D CUDA tensor");
  TORCH_CHECK(num_psi == 128 || num_psi == 256, "num_psi in {128,256}");
  const c10::cuda::CUDAGuard guard(c.device());
  const bool half = c.dtype() == torch::kFloat16;
  TORCH_CHECK(half || c.dtype() == torch::kComplexFloat, "complex64 or float16");
  unsigned F = c.size(0), P = c.size(1), Q = half ? c.size(2) / 2 : c.size(2);
  auto e = find(F, num_psi / 64); TORCH_CHECK(e, "unsupported NumFreq ", F);
  auto o = outputs(P, c, std::move(outs));
  auto st = at::cuda::getCurrentCUDAStream();
  if (half) e->hf(reinterpret_cast<const __half2*>(c.data_ptr<at::Half>()), o[0].data_ptr<float>(), o[1].data_ptr<float>(), reinterpret_cast<ull*>(o[2].data_ptr<int64_t>()), P, Q, st);
  else e->f(reinterpret_cast<const float2*>(c.data_ptr<c10::complex<float>>()), o[0].data_ptr<float>(), o[1].data_ptr<float>(), reinterpret_cast<ull*>(o[2].data_ptr<int64_t>()), P, Q, st);
  return o;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;
  m.def("lean_transposed", &lean_transposed, "c"_a, "num_psi"_a, "outs"_a = std::vector<torch::Tensor>{});
  m.def("sentinel_packed", &sentinel_i64);
}
