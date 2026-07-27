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
    bool active, float *dc_real_out, float *power_sum_out) {

  using complex_type = typename FFT::value_type;
  using scalar_type = typename complex_type::value_type;

  const unsigned int stride = FFT::stride;
  const unsigned long long batch_offset =
      static_cast<unsigned long long>(NumFreq) * flat_batch_idx;

  float dc_real = 0.0f;
  float power_sum = 0.0f;

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
      const float re = val.real();
      const float im = val.imag();

      if (read_idx == 0) {
        dc_real += re; // DC bin only has real component
      } else if (HasNyquist && read_idx == NumFreq - 1) {
        power_sum = fmaf(re, re, power_sum); // Nyquist bin real component only
      } else {
        // 2*(re^2+im^2), as two FMAs: fmaf(2*im,im,sum) then fmaf(2*re,re,.)
         // Doubled since RFFT symmetric
        power_sum = fmaf(2.0f * im, im, power_sum);
        power_sum = fmaf(2.0f * re, re, power_sum);
      }
    }
  }

  power_sum = fmaf(dc_real, dc_real, power_sum); // Add DC contribution to power sum

  *dc_real_out = dc_real;
  *power_sum_out = power_sum;
}

// ---------------------------------------------------------------------------
// Variant of load_padded_with_parseval_moments that reads from a per-block
// shared-memory staging tile (populated by a coalesced global->shared copy,
// see fused_irfft_and_stats_kernel_transposed) instead of directly from
// global memory. The register-to-frequency mapping (i*stride+threadIdx.x)
// and the DC-bin/Nyquist-bin Parseval accumulation are identical to the
// global-memory version above -- only the source address changes.
// `tile` is shaped (FPB, PaddedNumFreq), lane-major/freq-minor.
// ---------------------------------------------------------------------------
template <class FFT, unsigned int NumFreq, unsigned int PaddedNumFreq,
          bool HasNyquist>
inline __device__ void load_padded_with_parseval_moments_shared(
    const typename FFT::value_type *__restrict__ tile,
    typename FFT::value_type *thread_data, unsigned int local_fft_id,
    bool active, float *dc_real_out, float *power_sum_out) {

  using complex_type = typename FFT::value_type;
  using scalar_type = typename complex_type::value_type;

  const unsigned int stride = FFT::stride;
  const unsigned int row_offset = local_fft_id * PaddedNumFreq;

  float dc_real = 0.0f;
  float power_sum = 0.0f;

#pragma unroll
  for (unsigned int i = 0; i < FFT::input_ept; ++i) {
    const unsigned int read_idx = i * stride + threadIdx.x;
    const bool valid_read = active && (read_idx < NumFreq);
    complex_type val = valid_read
                           ? tile[row_offset + read_idx]
                           : complex_type{scalar_type(0), scalar_type(0)};

    thread_data[i] = val;

    if (valid_read) {
      const float re = val.real();
      const float im = val.imag();

      if (read_idx == 0) {
        dc_real += re;
      } else if (HasNyquist && read_idx == NumFreq - 1) {
        power_sum = fmaf(re, re, power_sum);
      } else {
        power_sum = fmaf(2.0f * im, im, power_sum);
        power_sum = fmaf(2.0f * re, re, power_sum);
      }
    }
  }

  power_sum = fmaf(dc_real, dc_real, power_sum);

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
// Conservative (lower-bound) nominal per-block opt-in shared memory budgets,
// keyed by the same SM/Arch values dispatched in irfft_stats_1d.cu. These
// back a compile-time static_assert in FusedIrfftStatsConfig as a
// best-effort guard for the transposed-input kernel's staging tile.
// ---------------------------------------------------------------------------
inline constexpr size_t arch_shared_memory_budget_bytes(unsigned int arch) {
  switch (arch) {
  case 750:
    return 64ull * 1024; // Turing
  case 800:
    return 163ull * 1024; // Ampere (A100)
  case 860:
  case 870:
    return 99ull * 1024; // Ampere (consumer / Orin)
  case 890:
    return 99ull * 1024; // Ada
  case 900:
    return 227ull * 1024; // Hopper
  case 1000:
  case 1030:
  case 1200:
  case 1210:
    return 99ull * 1024; // Blackwell (conservative floor)
  default:
    return 48ull * 1024; // conservative floor for unlisted/older archs
  }
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

  // -- Sizing for the transposed-input ((NumFreq, P, Q)) kernel variant,
  //    which stages each block's data through a (FPB, NumFreq+1) shared tile
  //    before the FFT's own working memory is used. The "+1" pads the row
  //    stride since common NumFreq values are powers of two and would lead to
  //    FPB-way bank conflicts on staging writes.
  static constexpr unsigned int padded_num_freq = NumFreq + 1;
  static constexpr size_t persistent_bytes_raw =
      static_cast<size_t>(FPB) * padded_num_freq * sizeof(complex_type);
  // Round up to 16 bytes so fft_shared_mem (below) stays 16-byte aligned.
  static constexpr size_t persistent_bytes =
      ((persistent_bytes_raw + 15) / 16) * 16;
  static constexpr size_t total_shared_bytes_transposed =
      persistent_bytes + fft_shared_memory_bytes;

  static_assert(total_shared_bytes_transposed <=
                    arch_shared_memory_budget_bytes(Arch),
                "Transposed-input staging tile + FFT scratch exceeds this "
                "architecture's nominal shared memory budget");
};

// ---------------------------------------------------------------------------
// Final per-block reduction: thread 0 sums the FPB lanes' Parseval moments
// and best (val, idx) pairs out of shared memory and atomically accumulates
// them into the (P,)-sized global outputs.
// ---------------------------------------------------------------------------
template <unsigned int FPB, unsigned int NumPsi>
inline __device__ void
finalize_block_stats(unsigned int p, unsigned int q0, unsigned int q_total,
                     const float *__restrict__ s_dc,
                     const float *__restrict__ s_power,
                     const unsigned long long *__restrict__ s_best_packed,
                     float *__restrict__ s1, float *__restrict__ s2,
                     unsigned long long *__restrict__ argmax_packed) {
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    float block_dc_sum = 0.0f;
    float block_power_sum = 0.0f;
    float best_val = -FLT_MAX;
    unsigned int best_flat_idx = 0;

#pragma unroll
    for (unsigned int l = 0; l < FPB; ++l) {
      if (q0 + l >= q_total)
        continue; // masks the trailing partial tile

      block_dc_sum += s_dc[l];
      block_power_sum += s_power[l];

      const unsigned long long packed = s_best_packed[l];
      const float val = unpack_val(packed);
      if (val > best_val) {
        best_val = val;
        best_flat_idx = (q0 + l) * NumPsi + unpack_idx(packed);
      }
    }

    const float n_psi_f = static_cast<float>(NumPsi);
    atomicAdd(s1 + p, n_psi_f * block_dc_sum);
    atomicAdd(s2 + p, n_psi_f * block_power_sum);
    atomicMax(argmax_packed + p, pack_val_idx(best_val, best_flat_idx));
  }
}

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

  __shared__ float s_dc[fpb];
  __shared__ float s_power[fpb];
  __shared__ unsigned long long s_best_packed[fpb];

  // Initialize shared memory
  if (threadIdx.x == 0) {
    s_dc[local_fft_id] = 0.0f;
    s_power[local_fft_id] = 0.0f;
    s_best_packed[local_fft_id] = 0ull; // sortable(-inf) has nonzero high bits
  }
  __syncthreads();

  // Initialize registers, load frequency data, and accumulate moments
  float dc_real = 0.0f;
  float power_sum = 0.0f;
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

  finalize_block_stats<fpb, n_psi>(p, q0, q_total, s_dc, s_power, s_best_packed,
                                   s1, s2, argmax_packed);
}

// ---------------------------------------------------------------------------
// Variant of fused_irfft_and_stats_kernel that consumes a (NumFreq, P, Q)
// contiguous input (frequency as the OUTERMOST/batch axis, instead of
// (P, Q, NumFreq). Each block does one coalesced global->shared copy of its
// (pixel, Q-tile) span.
// ---------------------------------------------------------------------------
template <class Config>
__launch_bounds__(Config::FFT::max_threads_per_block) __global__
    void fused_irfft_and_stats_kernel_transposed(
        const typename Config::complex_type *__restrict__ c, // (NumFreq, P, Q)
        float *__restrict__ s1,                              // (P), pre-zeroed
        float *__restrict__ s2,                              // (P), pre-zeroed
        unsigned long long
            *__restrict__ argmax_packed, // (P), sentinel pre-filled
        unsigned int p_total, unsigned int q_total,
        unsigned int n_tiles_per_pixel) {

  using FFT = typename Config::FFT;
  using complex_type = typename FFT::value_type;

  constexpr unsigned int num_freq = Config::num_freq;
  constexpr unsigned int n_psi = Config::num_psi;
  constexpr unsigned int fpb = Config::ffts_per_block;
  constexpr bool has_nyquist = Config::has_nyquist;
  constexpr unsigned int padded_num_freq = Config::padded_num_freq;
  constexpr unsigned int real_tile_elems = fpb * num_freq;

  const unsigned int p = blockIdx.x / n_tiles_per_pixel;
  const unsigned int tile = blockIdx.x % n_tiles_per_pixel;
  const unsigned int q0 = tile * fpb;
  const unsigned int local_fft_id = threadIdx.y;
  const unsigned int q = q0 + local_fft_id;
  const bool active = q < q_total;

  __shared__ float s_dc[fpb];
  __shared__ float s_power[fpb];
  __shared__ unsigned long long s_best_packed[fpb];

  if (threadIdx.x == 0) {
    s_dc[local_fft_id] = 0.0f;
    s_power[local_fft_id] = 0.0f;
    s_best_packed[local_fft_id] = 0ull;
  }

  // Partition dynamic shared memory: staging tile first, FFT scratch after.
  extern __shared__ __align__(16) unsigned char smem_raw[];
  complex_type *stage_tile = reinterpret_cast<complex_type *>(smem_raw);
  complex_type *fft_shared_mem =
      reinterpret_cast<complex_type *>(smem_raw + Config::persistent_bytes);

  __syncthreads(); // protects s_dc/s_power/s_best_packed init above

  // Cooperative, block-wide coalesced global->shared staging copy.
  {
    const unsigned int tid = threadIdx.y * FFT::stride + threadIdx.x;
    constexpr unsigned int block_threads = FFT::stride * fpb;
    const unsigned long long pq_stride =
        static_cast<unsigned long long>(p_total) * q_total;
    const unsigned long long row_base =
        static_cast<unsigned long long>(p) * q_total + q0;

    for (unsigned int elem = tid; elem < real_tile_elems;
         elem += block_threads) {
      const unsigned int k = elem / fpb;
      const unsigned int l = elem % fpb;
      const bool valid = (q0 + l) < q_total;
      complex_type val = valid ? c[k * pq_stride + row_base + l]
                               : zipfft::get_zero<complex_type>();
      stage_tile[l * padded_num_freq + k] = val;
    }
  }
  __syncthreads(); // staging tile fully written before any thread reads it

  float dc_real = 0.0f;
  float power_sum = 0.0f;
  complex_type thread_data[FFT::storage_size];
  load_padded_with_parseval_moments_shared<FFT, num_freq, padded_num_freq,
                                           has_nyquist>(
      stage_tile, thread_data, local_fft_id, active, &dc_real, &power_sum);

  if (active) {
    atomicAdd(&s_dc[local_fft_id], dc_real);
    atomicAdd(&s_power[local_fft_id], power_sum);
  }

  __syncthreads(); // existing barrier before FFT execute
  FFT().execute(thread_data, fft_shared_mem);

  float local_best_val;
  unsigned int local_best_idx;
  local_argmax_from_registers<FFT, n_psi>(thread_data, active, &local_best_val,
                                          &local_best_idx);
  if (active) {
    atomicMax(&s_best_packed[local_fft_id],
              pack_val_idx(local_best_val, local_best_idx));
  }
  __syncthreads();

  finalize_block_stats<fpb, n_psi>(p, q0, q_total, s_dc, s_power, s_best_packed,
                                   s1, s2, argmax_packed);
}

template <class Config>
inline void launch_fused_irfft_stats(const typename Config::complex_type *c,
                                     float *s1, float *s2,
                                     unsigned long long *argmax_packed,
                                     unsigned int p_total, unsigned int q_total,
                                     cudaStream_t stream = 0) {
  static bool attr_set = false;
  if (!attr_set) {
    auto err =
        cudaFuncSetAttribute(fused_irfft_and_stats_kernel<Config>,
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
      <<<num_blocks, Config::FFT::block_dim, Config::fft_shared_memory_bytes,
         stream>>>(c, s1, s2, argmax_packed, q_total, n_tiles_per_pixel);

  auto launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    throw std::runtime_error(std::string("fused_irfft_stats launch failed: ") +
                             cudaGetErrorString(launch_err) +
                             " num_blocks=" + std::to_string(num_blocks));
  }
}

template <class Config>
inline void launch_fused_irfft_stats_transposed(
    const typename Config::complex_type *c, float *s1, float *s2,
    unsigned long long *argmax_packed, unsigned int p_total,
    unsigned int q_total, cudaStream_t stream = 0) {
  static bool attr_set = false;
  if (!attr_set) {
    int device = 0;
    cudaGetDevice(&device);
    int max_optin = 0;
    auto attr_err = cudaDeviceGetAttribute(
        &max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (attr_err == cudaSuccess && static_cast<size_t>(max_optin) <
                                       Config::total_shared_bytes_transposed) {
      throw std::runtime_error(
          "fused_irfft_stats_transposed: device's max shared memory per "
          "block (" +
          std::to_string(max_optin) + " bytes) is smaller than the " +
          std::to_string(Config::total_shared_bytes_transposed) +
          " bytes required for this configuration");
    }

    auto err = cudaFuncSetAttribute(
        fused_irfft_and_stats_kernel_transposed<Config>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(Config::total_shared_bytes_transposed));
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("cudaFuncSetAttribute failed: ") +
                               cudaGetErrorString(err));
    }
    attr_set = true;
  }

  const unsigned int n_tiles_per_pixel =
      (q_total + Config::ffts_per_block - 1) / Config::ffts_per_block;
  const unsigned int num_blocks = p_total * n_tiles_per_pixel;

  fused_irfft_and_stats_kernel_transposed<Config>
      <<<num_blocks, Config::FFT::block_dim,
         Config::total_shared_bytes_transposed, stream>>>(
          c, s1, s2, argmax_packed, p_total, q_total, n_tiles_per_pixel);

  auto launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    throw std::runtime_error(
        std::string("fused_irfft_stats_transposed launch failed: ") +
        cudaGetErrorString(launch_err) +
        " num_blocks=" + std::to_string(num_blocks));
  }
}

} // namespace fused_stats