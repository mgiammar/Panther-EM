# Installing the fused iRFFT + stats CUDA kernel

Panther-EM's SVD-2DTM search always works out of the box: the reduction of each
hypothesis batch (inverse real FFT over the in-plane angle, moments, running maximum)
has a pure-PyTorch reference. On a CUDA device the search uses the faster
register-resident **fused kernel** instead (sources in
`src/panther_em/search/correlation/kernels/csrc/`, loader in
`.../correlation/kernels/fused_kernel_loader.py`), for ``n_psi`` in
``{128, 256, 512}``; other ``n_psi`` reduce through the reference.

The kernel is **JIT-compiled lazily**, the first time a search on a CUDA device is
planned, and cached in ``TORCH_EXTENSIONS_DIR`` afterwards. If it fails to compile,
Panther-EM warns once and uses the reference -- there is no install-time build step.
The kernel is plain CUDA C++: no header libraries (cuFFTDx, zipFFT) are needed.

## Prerequisites

- An NVIDIA GPU; the kernel is built for the device's own compute capability.
- The CUDA Toolkit with `nvcc` on `PATH`, matching the CUDA version PyTorch was
  built against (`nvcc --version`, `python -c "import torch; print(torch.version.cuda)"`).
- A CUDA-enabled PyTorch install (`torch.cuda.is_available()` returns `True`).

## Install

The `fused-kernels` extra installs `ninja`, which speeds up the JIT build:

```bash
uv sync --extra fused-kernels
# or, with plain pip:
pip install -e ".[fused-kernels]"
```

## Verify

The first call triggers the build (about a minute); it is cached afterwards.

```bash
python -c "
from panther_em.search.correlation.kernels import fused_kernel_loader
print('fused kernel OK' if fused_kernel_loader._try_compile() else 'fused kernel unavailable')
"
```

If it printed `fused kernel unavailable`, rerun with
`PANTHER_EM_FUSED_KERNEL_VERBOSE=1` to see the full `nvcc`/build log.
`scripts/check_lean_ptxas.py` builds the kernel and checks its register use and
spills per instantiation.

## Troubleshooting

- **nvcc / compiler errors**: usually a CUDA Toolkit / PyTorch / GPU driver
  version mismatch.
- **A stale build**: remove ``$TORCH_EXTENSIONS_DIR/panther_em_lean_irfft_stats``.
- **Force the pure-torch reference** (e.g. to A/B test or work around a broken
  toolchain):

  ```bash
  export PANTHER_EM_DISABLE_FUSED_KERNEL=1
  ```
