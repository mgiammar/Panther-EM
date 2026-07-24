/* fused_irfft_and_stats_kernel.cuh
 *
 * Fused zero-padded inverse real FFT and statistics update
 * (mean/var + vmax/amax) for the SVD-2DTM search algorithm. Does moment
 * calculation upon loading frequency values using Parseval's theorem rather
 * than in real-space after inverse FFT. The full cross-correlogram is never
 * materialized in global memory and rather reduced in register space.
 *
 * Computation (in PyTorch-like notation) is as follows:
 *
 *      C[p, q, :]   # complex spectrum of (num_pixel, num_hypothesis, num_freq)
 *      s1[p]   = n_psi * sum_q Re(C[p, q, 0])
 *      s2[p]   = n_psi * sum_q ( Re(C[p,q,0])^2 + 2*sum_{k=1}^{NumFreq-1}
 *          |C[p,q,k]|^2 )
 *      corr[p, q, :] = irfft(C[p, q, :], n=num_psi, dim=-1,
 * norm="forward") vmax[p], amax[p] = corr[p].reshape(-1).max(dim=0)
 */

#include <cfloat>
#include <stdexcept>
#include <string>

#include <cufftdx.hpp>

#include "zipfft_block_io.hpp"
#include "zipfft_common.hpp"
#include "zipfft_padded_io.hpp"

namespace fused_stats {

// ---------------------------------------------------------------------------
// Sortable-float <-> packed (float, index) atomicMax/ArgMax helpers.
// ---------------------------------------------------------------------------
inline __device__ unsigned int float_to_sortable_u32(float f) {
  unsigned int bits = __float_as_uint(f);
  unsigned int mask = (bits & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
  return bits ^ mask;
}

inline __device__ float sortable_u32_to_float(unsigned int s) {
  unsigned int mask = (s & 0x80000000u) ? 0x80000000u : 0xFFFFFFFFu;
  return __uint_as_float(s ^ mask);
}

inline __device__ unsigned long long pack_val_idx(float val, unsigned int idx) {
  unsigned long long sortable =
      static_cast<unsigned long long>(float_to_sortable_u32(val));
  return (sortable << 32) | static_cast<unsigned long long>(idx);
}

inline __device__ float unpack_val(unsigned long long packed) {
  return sortable_u32_to_float(static_cast<unsigned int>(packed >> 32));
}

inline __device__ unsigned int unpack_idx(unsigned long long packed) {
  return static_cast<unsigned int>(packed & 0xFFFFFFFFull);
}

// Packed encoding of (-inf, 0) sentinel to pre-fill output
inline unsigned long long sentinel_packed_host() {
  unsigned int bits = 0xFF800000u; // -inf
  unsigned int mask = 0xFFFFFFFFu; // sign bit set -> flip all bits
  unsigned int sortable = bits ^ mask;
  return (static_cast<unsigned long long>(sortable) << 32);
}

// ---------------------------------------------------------------------------
// Load the NumFreq low complex frequency bins with zero-padding remaining FFT
// registers. At the same time, accumulate first and second moments from the
// spectrum via Parseval's theorem. Note the moments are not scaled by NumPsi.
// ---------------------------------------------------------------------------
template <class FFT, unsigned int NumFreq, bool HasNyquist>
inline __device__ void load_padded_with_parseval_moments(
    const typename FFT::value_type *__restrict__ input,
    typename FFT::value_type *thread_data, unsigned long long flat_batch_idx,
    bool active, double *dc_real_out, double *power_sum_out) {

  using complex_type = typename FFT::value_type;
  using scalar_type = typename complex_type::value_type;

  const unsigned int stride = FFT::stride;
  const unsigned long long batch_offset =
      static_cast<unsigned long long>(NumFreq) * flat_batch_idx;

  double dc_real = 0.0;
  double power_sum = 0.0;

#pragma unroll
  for (unsigned int i = 0; i < FFT::input_ept; ++i) {
    // Load from global memory. Set to zero if read index exceeds NumFreq.
    const unsigned int read_idx = i * stride + threadIdx.x;
    const bool valid_read = active && (read_idx < NumFreq);
    complex_type val = valid_read
                           ? input[batch_offset + read_idx]
                           : complex_type{scalar_type(0), scalar_type(0)};

    thread_data[i] = val;

    // Accumulate moments from valid frequency bins.
    if (valid_read) {
      const double re = static_cast<double>(val.real());
      const double im = static_cast<double>(val.imag());

      if (read_idx == 0) {
        dc_real += re; // DC bin only has real component
      } else if (HasNyquist && read_idx == NumFreq - 1) {
        power_sum += re * re; // Nyquist bin only has real component, no double
      } else {
        power_sum += 2.0 * (re * re + im * im); // Double since RFFT symmetric
      }
    }
  }

  power_sum += dc_real * dc_real; // Add DC contribution to power sum

  *dc_real_out = dc_real;
  *power_sum_out = power_sum;
}

// ---------------------------------------------------------------------------
// Scan this thread's slice of the post-FFT registers for local vmax/amax.
// ---------------------------------------------------------------------------
template <class FFT, unsigned int NumPsi>
inline __device__ void
local_argmax_from_registers(const typename FFT::value_type *thread_data,
                            bool active, float *best_val_out,
                            unsigned int *best_idx_out) {
  using complex_type = typename FFT::value_type;
  using scalar_type = typename complex_type::value_type;
  using output_t = typename FFT::output_type;

  constexpr unsigned int inner_loop_limit =
      sizeof(output_t) / sizeof(scalar_type);
  const unsigned int stride = FFT::stride;

  float best_val = -FLT_MAX;
  unsigned int best_idx = 0;

  if (active) {
    const scalar_type *data =
        reinterpret_cast<const scalar_type *>(thread_data);
#pragma unroll
    // For each output FFT element
    for (unsigned int i = 0; i < FFT::output_ept; ++i) {
#pragma unroll
      // For each real component of complex output (real1, real2)
      for (unsigned int j = 0; j < inner_loop_limit; ++j) {
        const unsigned int read_idx =
            i * stride * inner_loop_limit + j + threadIdx.x * inner_loop_limit;

        if (read_idx < NumPsi) {
          const float val = data[i * inner_loop_limit + j];
          if (val > best_val) {
            best_val = val;
            best_idx = read_idx;
          }
        }
      }
    }
  }

  *best_val_out = best_val;
  *best_idx_out = best_idx;
}

// ---------------------------------------------------------------------------
// Compile-time configuration of (Arch, NumPsi, NumFreq, FPB, EPT).
// ---------------------------------------------------------------------------
template <unsigned int Arch, unsigned int NumPsi, unsigned int NumFreq,
          unsigned int FPB, unsigned int EPT>
struct FusedIrfftStatsConfig {
  using real_fft_options =
      cufftdx::RealFFTOptions<cufftdx::complex_layout::natural,
                              cufftdx::real_mode::folded>;

  using FFT =
      decltype(cufftdx::Block() + cufftdx::Size<NumPsi>() +
               cufftdx::Type<cufftdx::fft_type::c2r>() + real_fft_options() +
               cufftdx::Direction<cufftdx::fft_direction::inverse>() +
               cufftdx::Precision<float>() + cufftdx::SM<Arch>() +
               cufftdx::ElementsPerThread<EPT>() +
               cufftdx::FFTsPerBlock<FPB>());

  using complex_type = typename FFT::value_type;
  using scalar_type = typename complex_type::value_type;

  static constexpr unsigned int num_psi = NumPsi;
  static constexpr unsigned int num_freq = NumFreq;
  static constexpr unsigned int elements_per_thread = EPT;
  static constexpr unsigned int ffts_per_block = FPB;
  static constexpr bool has_nyquist = (NumFreq == NumPsi / 2 + 1);

  static_assert(FFT::ffts_per_block == FPB, "FFTs per block mismatch");
  static_assert(FFT::elements_per_thread == EPT,
                "Elements per thread mismatch");
  static_assert(FFT::implicit_type_batching == 1,
                "Real-valued implicit batching mismatch");
  static_assert(NumFreq <= NumPsi / 2 + 1,
                "NumFreq must be <= NumPsi/2 + 1 for RFFT");

  static constexpr size_t fft_shared_memory_bytes = FFT::shared_memory_size;
};

// ---------------------------------------------------------------------------
// Fused kernel definition.
// - blockIdx.x enumerates (pixel, hypothesis-tile) pairs in row-major order of
//   (P, ceil(Q / FPB)).
// ---------------------------------------------------------------------------
template <class Config>
__launch_bounds__(Config::FFT::max_threads_per_block) __global__
    void fused_irfft_and_stats_kernel(
        const typename Config::complex_type *__restrict__ c, // (P, Q, NumFreq)
        float *__restrict__ s1,                              // (P), pre-zeroed
        float *__restrict__ s2,                              // (P), pre-zeroed
        unsigned long long
            *__restrict__ argmax_packed, // (P), sentinel pre-filled
        unsigned int q_total, unsigned int n_tiles_per_pixel) {

  using FFT = typename Config::FFT;
  using complex_type = typename FFT::value_type;

  // Extract execution parameters from configuration
  constexpr unsigned int num_freq = Config::num_freq;
  constexpr unsigned int n_psi = Config::num_psi;
  constexpr unsigned int fpb = Config::ffts_per_block;
  constexpr bool has_nyquist = Config::has_nyquist;

  // Compute constants
  const unsigned int p = blockIdx.x / n_tiles_per_pixel;
  const unsigned int tile = blockIdx.x % n_tiles_per_pixel;
  const unsigned int q0 = tile * fpb;
  const unsigned int local_fft_id = threadIdx.y;
  const unsigned int q = q0 + local_fft_id;
  const bool active = q < q_total;

  __shared__ double s_dc[fpb];
  __shared__ double s_power[fpb];
  __shared__ unsigned long long s_best_packed[fpb];

  // Initialize shared memory
  if (threadIdx.x == 0) {
    s_dc[local_fft_id] = 0.0;
    s_power[local_fft_id] = 0.0;
    s_best_packed[local_fft_id] = 0ull; // sortable(-inf) has nonzero high bits
  }
  __syncthreads();

  // Initialize registers, load frequency data, and accumulate moments
  double dc_real = 0.0;
  double power_sum = 0.0;
  const unsigned long long flat_batch_idx =
      static_cast<unsigned long long>(p) * q_total + q;
  complex_type thread_data[FFT::storage_size];
  load_padded_with_parseval_moments<FFT, num_freq, has_nyquist>(
      c, thread_data, flat_batch_idx, active, &dc_real, &power_sum);

  if (active) {
    atomicAdd(&s_dc[local_fft_id], dc_real);
    atomicAdd(&s_power[local_fft_id], power_sum);
  }

  // Execute the inverse FFT
  extern __shared__ __align__(16) unsigned char fft_smem_raw[];
  complex_type *fft_shared_mem = reinterpret_cast<complex_type *>(fft_smem_raw);
  __syncthreads();
  FFT().execute(thread_data, fft_shared_mem);

  // vmax/amax reduction across threads
  float local_best_val;
  unsigned int local_best_idx;
  local_argmax_from_registers<FFT, n_psi>(thread_data, active, &local_best_val,
                                          &local_best_idx);
  if (active) {
    atomicMax(&s_best_packed[local_fft_id],
              pack_val_idx(local_best_val, local_best_idx));
  }
  __syncthreads();

  // Thread zero accumulates results into global memory
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    double block_dc_sum = 0.0;
    double block_power_sum = 0.0;
    float best_val = -FLT_MAX;
    unsigned int best_flat_idx = 0;

#pragma unroll
    for (unsigned int l = 0; l < fpb; ++l) {
      if (q0 + l >= q_total)
        continue; // masks the trailing partial tile

      block_dc_sum += s_dc[l];
      block_power_sum += s_power[l];

      const unsigned long long packed = s_best_packed[l];
      const float val = unpack_val(packed);
      if (val > best_val) {
        best_val = val;
        best_flat_idx = (q0 + l) * n_psi + unpack_idx(packed);
      }
    }

    const double n_psi_d = static_cast<double>(n_psi);
    atomicAdd(s1 + p, static_cast<float>(n_psi_d * block_dc_sum));
    atomicAdd(s2 + p, static_cast<float>(n_psi_d * block_power_sum));
    atomicMax(argmax_packed + p, pack_val_idx(best_val, best_flat_idx));
  }
}

template <class Config>
inline void launch_fused_irfft_stats(const typename Config::complex_type* c, float* s1,
                                      float* s2, unsigned long long* argmax_packed,
                                      unsigned int p_total, unsigned int q_total,
                                      cudaStream_t stream = 0) {
  static bool attr_set = false;
  if (!attr_set) {
    auto err = cudaFuncSetAttribute(fused_irfft_and_stats_kernel<Config>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     static_cast<int>(Config::fft_shared_memory_bytes));
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("cudaFuncSetAttribute failed: ") +
                                cudaGetErrorString(err));
    }
    attr_set = true;
  }

  const unsigned int n_tiles_per_pixel =
      (q_total + Config::ffts_per_block - 1) / Config::ffts_per_block;
  const unsigned int num_blocks = p_total * n_tiles_per_pixel;

  fused_irfft_and_stats_kernel<Config>
      <<<num_blocks, Config::FFT::block_dim, Config::fft_shared_memory_bytes, stream>>>(
          c, s1, s2, argmax_packed, q_total, n_tiles_per_pixel);

  auto launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    throw std::runtime_error(std::string("fused_irfft_stats launch failed: ") +
                              cudaGetErrorString(launch_err) +
                              " num_blocks=" + std::to_string(num_blocks));
  }
}

} // namespace fused_stats