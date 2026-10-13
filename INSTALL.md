# Installing the fused iRFFT + stats CUDA kernel

Panther-EM's SVD-2DTM search always works out of the box using a pure-PyTorch
"psi recovery + statistics update" path. This document is only for users who want
the faster, lower-peak-memory **fused CUDA kernel** (see
`src/panther_em/inference/search/fused_kernel_loader.py` and
`FusedPixelStats`) that replaces it when available.

The kernel is **JIT-compiled lazily**, the first time it's actually called, and
cached for the rest of the process. If it's unavailable or fails to compile for
any reason, Panther-EM transparently falls back to the pure-torch path -- there is
no install-time build step and nothing else breaks if you skip this document.

## Prerequisites

- An NVIDIA GPU with compute capability **8.0, 8.6, 8.7, 8.9, or 9.0** (Ampere,
  Ada, or Hopper). Other architectures compile but the dispatch table has no
  matching case, so the kernel raises at first call and Panther-EM falls back.
- The CUDA Toolkit with `nvcc` on `PATH` (matching the CUDA version PyTorch was
  built against):

  ```bash
  nvcc --version
  ```

- A CUDA-enabled PyTorch install (`torch.cuda.is_available()` returns `True`).

## Install

Install Panther-EM with the `fused-kernels` extra, which pulls in
[`nvidia-mathdx`](https://pypi.org/project/nvidia-mathdx/) (cuFFTDx headers) and
`ninja` (speeds up the JIT build):

```bash
uv sync --extra fused-kernels
# or, with plain pip:
pip install -e ".[fused-kernels]"
```

zipFFT's own headers are already vendored in-tree
(`src/panther_em/inference/search/csrc/include/`) -- no separate zipFFT checkout
or environment variable is needed for those.

## Verify

The first call below triggers the actual JIT compile (a few seconds to a couple
of minutes); it's cached afterward for the life of the process.

```bash
python -c "
from panther_em.inference.search import fused_kernel_loader
configs = fused_kernel_loader.get_supported_configs()
print('fused kernel compiled OK' if configs else 'fused kernel unavailable (check warnings above)')
"
```

If it printed `fused kernel unavailable`, rerun with
`PANTHER_EM_FUSED_KERNEL_VERBOSE=1` to see the full `nvcc`/build log:

```bash
PANTHER_EM_FUSED_KERNEL_VERBOSE=1 python -c "
from panther_em.inference.search import fused_kernel_loader
fused_kernel_loader.get_supported_configs()
"
```

## Troubleshooting

- **"Could not locate cuFFTDx headers"**: `nvidia-mathdx` isn't installed, or its
  headers are somewhere unusual. Set `CUFFTDX_INCLUDE_DIR` (and optionally
  `CUTLASS_INCLUDE_DIR`) to point directly at a directory containing
  `cufftdx.hpp`.
- **nvcc / compiler errors**: usually a CUDA Toolkit / PyTorch / GPU driver
  version mismatch. Check `nvcc --version` matches the CUDA version `torch` was
  built against (`python -c "import torch; print(torch.version.cuda)"`).
- **Force the pure-torch fallback** (e.g. to A/B test or work around a broken
  toolchain) without uninstalling anything:

  ```bash
  export PANTHER_EM_DISABLE_FUSED_KERNEL=1
  ```

## Uninstall / disable

There's nothing to "uninstall" beyond the optional pip packages -- removing
`nvidia-mathdx` (or setting `PANTHER_EM_DISABLE_FUSED_KERNEL=1`) reverts to the
pure-torch path automatically.
