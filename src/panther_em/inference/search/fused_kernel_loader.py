"""JIT-compile glue for the fused iRFFT + Parseval-moments + max/argmax CUDA kernel.

cuFFTDx headers are located automatically, in this order:
  1. ``$CUFFTDX_INCLUDE_DIR`` (or ``$MATHDX_INCLUDE_DIR``) env override.
  2. the ``nvidia-mathdx`` pip wheel (``pip install panther-em[fused-kernels]``).
  3. common conda locations (``$CONDA_PREFIX/include``).

Notes
-----
Set ``PANTHER_EM_DISABLE_FUSED_KERNEL=1`` to force the pure-torch fallback path even
when a CUDA device an toolchain are available and fused kernel requested. Set the
environment variable ``PANTHER_EM_FUSED_KERNEL_VERBOSE=1`` to see the
``torch.utils.cpp_extension.load`` build log.
"""

from __future__ import annotations

import functools
import glob
import os
import warnings
from typing import Any

import torch

from panther_em.inference.search.statistics import decode_argmax_packed

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CSRC_DIR = os.path.join(_THIS_DIR, "csrc")
_INCLUDE_DIR = os.path.join(_CSRC_DIR, "include")

_warned_compile_failure = False
_warned_runtime_failure = False


# --------------------------------------------------------------------------- #
# Locating headers
# --------------------------------------------------------------------------- #
def _first_existing(*paths: str) -> str | None:
    for p in paths:
        if p and os.path.isdir(p):
            return p
    return None


def _find_cufftdx_includes() -> list[str]:
    """Return include dirs resolving ``<cufftdx.hpp>``.

    Returns
    -------
    list[str]
        List of include directories to pass to ``extra_include_paths`` in
        ``torch.utils.cpp_extension.load``. The first entry is guaranteed to contain
        ``cufftdx.hpp``. The second entry (if present) is the CUTLASS include dir, which
        is only needed for the fused kernel build.

    Raises
    ------
    RuntimeError
        If the cuFFTDx headers cannot be located.
    """
    includes: list[str] = []

    env = os.environ.get("CUFFTDX_INCLUDE_DIR") or os.environ.get("MATHDX_INCLUDE_DIR")
    if env and os.path.isfile(os.path.join(env, "cufftdx.hpp")):
        includes.append(env)
        cutlass = os.environ.get("CUTLASS_INCLUDE_DIR")
        if cutlass:
            includes.append(cutlass)
        return includes

    try:
        import nvidia  # namespace package installed by nvidia-mathdx

        for base in getattr(nvidia, "__path__", []):
            mathdx_inc = os.path.join(base, "mathdx", "include")
            if os.path.isfile(os.path.join(mathdx_inc, "cufftdx.hpp")):
                includes.append(mathdx_inc)
                cutlass_inc = os.path.join(
                    base, "mathdx", "external", "cutlass", "include"
                )
                if os.path.isdir(cutlass_inc):
                    includes.append(cutlass_inc)
                return includes
    except Exception:
        pass

    conda = os.environ.get("CONDA_PREFIX", "")
    candidates = []
    if conda:
        candidates += [os.path.join(conda, "include")]
        candidates += glob.glob(
            os.path.join(conda, "**", "cufftdx", "include"), recursive=True
        )
    candidates += ["/usr/local/cufftdx/include", "/opt/nvidia/mathdx/include"]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "cufftdx.hpp")):
            includes.append(c)
            cutlass = _first_existing(
                os.path.join(os.path.dirname(c), "external", "cutlass", "include"),
                os.path.join(conda, "include"),
            )
            if cutlass and cutlass != c:
                includes.append(cutlass)
            return includes

    raise RuntimeError(
        "Could not locate cuFFTDx headers (cufftdx.hpp). Set CUFFTDX_INCLUDE_DIR, "
        "or `pip install panther-em[fused-kernels]`."
    )


# --------------------------------------------------------------------------- #
# Build flags
# --------------------------------------------------------------------------- #
def _arch_flags() -> tuple[list[str], list[str]]:
    major, minor = torch.cuda.get_device_capability()
    cc = f"{major}{minor}"
    defines = [f"-DENABLE_CUDA_ARCH_{cc}0"]
    gencode = ["-gencode", f"arch=compute_{cc},code=sm_{cc}"]
    return defines, gencode


@functools.lru_cache(maxsize=1)
def _try_compile() -> Any:
    """Attempt to JIT-compile the fused extension with a LUR cache for repeated calls.

    Returns
    -------
    "compiled module" | None
        When successful, the compiled module is returned. If compilation fails for any
        reason, ``None`` is returned instead of raising an exception.
    """
    global _warned_compile_failure

    if not torch.cuda.is_available():
        return None
    if os.environ.get("PANTHER_EM_DISABLE_FUSED_KERNEL"):
        return None

    try:
        from torch.utils.cpp_extension import load

        cufftdx_incs = _find_cufftdx_includes()
        defines, gencode = _arch_flags()

        nvcc_flags = [
            "-O3",
            "-std=c++17",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "--expt-relaxed-constexpr",
            *defines,
            *gencode,
        ]

        return load(
            name="panther_em_fused_irfft_stats",
            sources=[os.path.join(_CSRC_DIR, "irfft_stats_1d.cu")],
            extra_include_paths=[_CSRC_DIR, _INCLUDE_DIR, *cufftdx_incs],
            extra_cuda_cflags=nvcc_flags,
            extra_cflags=["-O3", "-std=c++17"],
            verbose=bool(int(os.environ.get("PANTHER_EM_FUSED_KERNEL_VERBOSE", "0"))),
        )
    except Exception as exc:
        if not _warned_compile_failure:
            warnings.warn(
                "Fused iRFFT+stats CUDA kernel unavailable, falling back to the "
                f"pure-torch path for this process: {exc}",
                UserWarning,
                stacklevel=2,
            )
            _warned_compile_failure = True
        return None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def get_supported_configs() -> list[tuple[int, int, int, int]] | None:
    """List of supported ``(n_psi, num_freq, fpb, ept)`` tuples.

    Returns
    -------
    list[tuple[int, int, int, int]] | None
        List of supported ``(n_psi, num_freq, fpb, ept)`` tuples, or ``None`` if the
        kernel is unavailable in this process.
    """
    module = _try_compile()
    return list(module.get_supported_configs()) if module is not None else None


def fused_irfft_stats(
    c: torch.Tensor, n_psi: int, decode: bool = True
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | None
):
    """Fused psi-recovery + statistics-update for one hypothesis batch.

    Matches :func:`panther_em.inference.search.statistics._reduce_stats`'s I/O
    contract (except ``corr`` is never materialized)::

        corr = torch.fft.irfft(c, n=n_psi, dim=-1, norm="forward")
        s1, s2, vmax, amax = _reduce_stats(corr, torch.view_as_real(c))

    Parameters
    ----------
    c : torch.Tensor
        complex64, CUDA, shape (P, Q, NumFreq).
    n_psi : int
        Full in-plane-angle length.
    decode : bool, optional
        If ``True`` (default), decode the kernel's raw packed argmax into
        ``(vmax, amax)`` before returning. Pass ``False`` to skip that decode and get
        the raw packed value back instead -- useful when a caller wants to batch
        several calls' raw outputs together and decode the whole batch once (see
        :func:`panther_em.inference.search.statistics.decode_argmax_packed` and
        :meth:`~panther_em.inference.search.statistics.PixelStats.accumulate_batch`).

    Returns
    -------
    tuple | None
        ``(s1, s2, vmax, amax)`` with dtypes (``float32``, ``float32``, ``float32``,
        ``int64``) when ``decode=True``; ``(s1, s2, argmax_packed)`` with dtypes
        (``float32``, ``float32``, ``int64``) when ``decode=False``. All shapes
        ``(P,)``. If an exception was raised internally, then None is returned and a
        warning is issued, and finally the torch fallback path is used instead.
    """
    global _warned_runtime_failure

    module = _try_compile()
    if module is None:
        return None

    n_freq = c.shape[-1]
    configs = {(cfg[0], cfg[1]) for cfg in module.get_supported_configs()}
    if (int(n_psi), int(n_freq)) not in configs:
        return None

    try:
        s1, s2, argmax_packed = module.fused_irfft_stats(c.contiguous(), int(n_psi))
    except Exception as exc:
        if not _warned_runtime_failure:
            warnings.warn(
                "Fused iRFFT+stats CUDA kernel raised at runtime, falling back "
                f"to the pure-torch path for this call: {exc}",
                UserWarning,
                stacklevel=2,
            )
            _warned_runtime_failure = True
        return None

    if not decode:
        return s1, s2, argmax_packed
    vmax, amax = decode_argmax_packed(argmax_packed)
    return s1, s2, vmax, amax


def fused_irfft_stats_transposed(
    c: torch.Tensor, n_psi: int, decode: bool = True
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | None
):
    """Zero-copy variant of :func:`fused_irfft_stats` for ``(NumFreq, P, Q)`` input.

    Notes
    -----
    Spectrum ``c`` is expected to be  contiguous in ``(NumFreq, P, Q)`` layout (the
    layout a cuBLAS strided-batched GEMM produces natively). No ``.contiguous()`` call
    is made which skips a strided copy kernel. Internal strided CUDA kernel uses staged
    shared memory to transpose small tiles of memory upon load.

    Parameters
    ----------
    c : torch.Tensor
        complex64, CUDA, contiguous, shape (NumFreq, P, Q).
    n_psi : int
        Full in-plane-angle length.
    decode : bool, optional
        See :func:`fused_irfft_stats`'s ``decode`` parameter -- same contract here.

    Returns
    -------
    tuple | None
        ``(s1, s2, vmax, amax)`` when ``decode=True``, or ``(s1, s2, argmax_packed)``
        when ``decode=False`` -- same shapes/dtypes as :func:`fused_irfft_stats`.
        ``None`` on any internal failure (unsupported config, non-contiguous input,
        runtime error), in which case callers should fall back to
        :func:`fused_irfft_stats` or the pure-torch path.
    """
    global _warned_runtime_failure

    module = _try_compile()
    if module is None:
        return None

    n_freq = c.shape[0]
    configs = {(cfg[0], cfg[1]) for cfg in module.get_supported_configs()}
    if (int(n_psi), int(n_freq)) not in configs:
        return None

    try:
        s1, s2, argmax_packed = module.fused_irfft_stats_transposed(c, int(n_psi))
    except Exception as exc:
        if not _warned_runtime_failure:
            warnings.warn(
                "Fused iRFFT+stats (transposed) CUDA kernel raised at runtime, "
                f"falling back to the pure-torch path for this call: {exc}",
                UserWarning,
                stacklevel=2,
            )
            _warned_runtime_failure = True
        return None

    if not decode:
        return s1, s2, argmax_packed
    vmax, amax = decode_argmax_packed(argmax_packed)
    return s1, s2, vmax, amax
