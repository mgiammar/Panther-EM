// Experimental binding: compact single-arch dispatch, float2 and __half2 inputs,
// explicit (fpb, ept) selection, optional caller-provided output buffers.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <cstring>
#include <vector>
#include "fused_irfft_and_stats_kernel_exp.cuh"

#ifndef EXP_ARCH
#define EXP_ARCH 890
#endif

using ull = unsigned long long;
template <unsigned NPsi, unsigned NumFreq, unsigned FPB, unsigned EPT, typename InT>
void launch_t(const InT* c, float* s1, float* s2, ull* pk, unsigned P, unsigned Q) {
  using Config = fused_stats::FusedIrfftStatsConfig<EXP_ARCH, NPsi, NumFreq, FPB, EPT>;
  fused_stats::launch_fused_irfft_stats_transposed<Config, InT>(c, s1, s2, pk, P, Q, at::cuda::getCurrentCUDAStream());
}
template <unsigned NPsi, unsigned NumFreq, unsigned FPB, unsigned EPT, typename InT>
void launch_r(const InT* c, float* s1, float* s2, ull* pk, unsigned P, unsigned Q) {
  using Config = fused_stats::FusedIrfftStatsConfig<EXP_ARCH, NPsi, NumFreq, FPB, EPT>;
  fused_stats::launch_fused_irfft_stats<Config, InT>(c, s1, s2, pk, P, Q, at::cuda::getCurrentCUDAStream());
}
using FnF = void (*)(const float2*, float*, float*, ull*, unsigned, unsigned);
using FnH = void (*)(const __half2*, float*, float*, ull*, unsigned, unsigned);
struct Entry { unsigned psi, freq, fpb, ept; FnF tf, rf; FnH th, rh; };
#define CFG(PSI, FREQ, FPB, EPT) Entry{PSI, FREQ, FPB, EPT, &launch_t<PSI, FREQ, FPB, EPT, float2>, &launch_r<PSI, FREQ, FPB, EPT, float2>, &launch_t<PSI, FREQ, FPB, EPT, __half2>, &launch_r<PSI, FREQ, FPB, EPT, __half2>}
static const Entry TABLE[] = {
    CFG(256, 64, 4, 32), CFG(256, 64, 8, 32), CFG(256, 64, 16, 32), CFG(256, 64, 8, 16), CFG(256, 64, 8, 8),
    CFG(128, 64, 8, 16), CFG(128, 64, 4, 16), CFG(128, 64, 16, 16), CFG(128, 64, 8, 32), CFG(128, 64, 8, 8),
    CFG(256, 32, 4, 32), CFG(256, 32, 8, 32), CFG(256, 48, 4, 32), CFG(256, 48, 8, 32),
    CFG(128, 32, 8, 16), CFG(128, 48, 8, 16),
};
static const Entry* find(unsigned psi, unsigned freq, unsigned fpb, unsigned ept) {
  for (const auto& e : TABLE) if (e.psi == psi && e.freq == freq && e.fpb == fpb && e.ept == ept) return &e;
  return nullptr;
}
static int64_t sentinel_i64() { ull s = fused_stats::sentinel_packed_host(); int64_t o; std::memcpy(&o, &s, 8); return o; }

// Output buffers: (s1, s2, argmax_packed), each (P,). If `outs` is empty they are
// allocated + initialized here; otherwise the caller guarantees s1=s2=0 and
// argmax_packed = sentinel (e.g. re-initialized inside a CUDA graph).
static std::vector<torch::Tensor> outputs(unsigned P, const torch::Tensor& like, std::vector<torch::Tensor> outs) {
  if (outs.size() == 3) return outs;
  auto f = torch::TensorOptions().dtype(torch::kFloat32).device(like.device());
  auto i = torch::TensorOptions().dtype(torch::kInt64).device(like.device());
  return {torch::zeros({P}, f), torch::zeros({P}, f), torch::full({P}, sentinel_i64(), i)};
}

// c: complex64 (F, P, Q) contiguous
std::vector<torch::Tensor> transposed_f32(torch::Tensor c, int64_t psi, int64_t fpb, int64_t ept, std::vector<torch::Tensor> outs) {
  TORCH_CHECK(c.is_cuda() && c.dtype() == torch::kComplexFloat && c.dim() == 3 && c.is_contiguous(), "need contiguous complex64 (F,P,Q)");
  const c10::cuda::CUDAGuard guard(c.device());
  unsigned F = c.size(0), P = c.size(1), Q = c.size(2);
  auto e = find(psi, F, fpb, ept); TORCH_CHECK(e, "unsupported config (psi,F,fpb,ept)=(", psi, ",", F, ",", fpb, ",", ept, ")");
  auto o = outputs(P, c, std::move(outs));
  e->tf(reinterpret_cast<const float2*>(c.data_ptr<c10::complex<float>>()), o[0].data_ptr<float>(), o[1].data_ptr<float>(), reinterpret_cast<ull*>(o[2].data_ptr<int64_t>()), P, Q);
  return o;
}
// c: float16 (F, P, 2Q) contiguous, interleaved (re, im) pairs
std::vector<torch::Tensor> transposed_f16(torch::Tensor c, int64_t psi, int64_t fpb, int64_t ept, std::vector<torch::Tensor> outs) {
  TORCH_CHECK(c.is_cuda() && c.dtype() == torch::kFloat16 && c.dim() == 3 && c.is_contiguous() && c.size(2) % 2 == 0, "need contiguous float16 (F,P,2Q)");
  const c10::cuda::CUDAGuard guard(c.device());
  unsigned F = c.size(0), P = c.size(1), Q = c.size(2) / 2;
  auto e = find(psi, F, fpb, ept); TORCH_CHECK(e, "unsupported config (psi,F,fpb,ept)=(", psi, ",", F, ",", fpb, ",", ept, ")");
  auto o = outputs(P, c, std::move(outs));
  e->th(reinterpret_cast<const __half2*>(c.data_ptr<at::Half>()), o[0].data_ptr<float>(), o[1].data_ptr<float>(), reinterpret_cast<ull*>(o[2].data_ptr<int64_t>()), P, Q);
  return o;
}
// c: complex64 (P, Q, F) contiguous
std::vector<torch::Tensor> regular_f32(torch::Tensor c, int64_t psi, int64_t fpb, int64_t ept, std::vector<torch::Tensor> outs) {
  TORCH_CHECK(c.is_cuda() && c.dtype() == torch::kComplexFloat && c.dim() == 3, "need complex64 (P,Q,F)");
  const c10::cuda::CUDAGuard guard(c.device());
  c = c.contiguous();
  unsigned P = c.size(0), Q = c.size(1), F = c.size(2);
  auto e = find(psi, F, fpb, ept); TORCH_CHECK(e, "unsupported config");
  auto o = outputs(P, c, std::move(outs));
  e->rf(reinterpret_cast<const float2*>(c.data_ptr<c10::complex<float>>()), o[0].data_ptr<float>(), o[1].data_ptr<float>(), reinterpret_cast<ull*>(o[2].data_ptr<int64_t>()), P, Q);
  return o;
}
// c: float16 (P, Q, 2F) contiguous
std::vector<torch::Tensor> regular_f16(torch::Tensor c, int64_t psi, int64_t fpb, int64_t ept, std::vector<torch::Tensor> outs) {
  TORCH_CHECK(c.is_cuda() && c.dtype() == torch::kFloat16 && c.dim() == 3 && c.size(2) % 2 == 0, "need float16 (P,Q,2F)");
  const c10::cuda::CUDAGuard guard(c.device());
  c = c.contiguous();
  unsigned P = c.size(0), Q = c.size(1), F = c.size(2) / 2;
  auto e = find(psi, F, fpb, ept); TORCH_CHECK(e, "unsupported config");
  auto o = outputs(P, c, std::move(outs));
  e->rh(reinterpret_cast<const __half2*>(c.data_ptr<at::Half>()), o[0].data_ptr<float>(), o[1].data_ptr<float>(), reinterpret_cast<ull*>(o[2].data_ptr<int64_t>()), P, Q);
  return o;
}
std::vector<std::tuple<int, int, int, int>> configs() {
  std::vector<std::tuple<int, int, int, int>> v;
  for (const auto& e : TABLE) v.emplace_back(e.psi, e.freq, e.fpb, e.ept);
  return v;
}
int64_t sentinel() { return sentinel_i64(); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;
  m.def("transposed_f32", &transposed_f32, "c"_a, "num_psi"_a, "fpb"_a, "ept"_a, "outs"_a = std::vector<torch::Tensor>{});
  m.def("transposed_f16", &transposed_f16, "c"_a, "num_psi"_a, "fpb"_a, "ept"_a, "outs"_a = std::vector<torch::Tensor>{});
  m.def("regular_f32", &regular_f32, "c"_a, "num_psi"_a, "fpb"_a, "ept"_a, "outs"_a = std::vector<torch::Tensor>{});
  m.def("regular_f16", &regular_f16, "c"_a, "num_psi"_a, "fpb"_a, "ept"_a, "outs"_a = std::vector<torch::Tensor>{});
  m.def("get_supported_configs", &configs);
  m.def("sentinel_packed", &sentinel);
}
