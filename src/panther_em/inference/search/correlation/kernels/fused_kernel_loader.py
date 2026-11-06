"""JIT-compile glue for the register-resident fused iRFFT + moments + max kernel.

The kernel (``csrc/lean_irfft_stats.cuh``, generated transforms in
``csrc/lean_fft_gen.cuh``) recovers every in-plane angle of a hypothesis batch's
angular spectrum and reduces it to the per-pixel moment sums and the packed maximum,
in the contraction's native ``(F, P, N)`` layout, for ``n_psi`` in ``{128, 256, 512}``.
It is built with ``torch.utils.cpp_extension.load`` for the current device's
architecture the first time it is needed.

Notes
-----
Set ``PANTHER_EM_DISABLE_FUSED_KERNEL=1`` to skip the build and reduce through the
torch reference. Set ``PANTHER_EM_FUSED_KERNEL_VERBOSE=1`` to see the build log.
"""

from __future__ import annotations

import functools
import os
import warnings
from typing import Any, Literal, overload

import torch

from panther_em.inference.search.correlation.kernels.packed import (
    decode_argmax_packed,
)

EXTENSION_NAME = "panther_em_lean_irfft_stats"
_CSRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")

_warned_compile_failure = False
_warned_runtime_failure = False


@functools.lru_cache(maxsize=1)
def _try_compile() -> Any:
    """JIT-compile the lean kernel's extension once per process.

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

        major, minor = torch.cuda.get_device_capability()
        gencode = ["-gencode", f"arch=compute_{major}{minor},code=sm_{major}{minor}"]

        nvcc_flags = [
            "-O3",
            "-std=c++17",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "--expt-relaxed-constexpr",
            "-Xptxas",
            "-v",  # register/spill report (PANTHER_EM_FUSED_KERNEL_VERBOSE build log)
            *gencode,
        ]

        return load(
            name=EXTENSION_NAME,
            sources=[os.path.join(_CSRC_DIR, "lean_irfft_stats.cu")],
            extra_include_paths=[_CSRC_DIR],
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


LEAN_NUM_PSI = (128, 256, 512)


def lean_supported(n_psi: int, n_freq: int) -> bool:
    """Whether the register-resident kernel covers ``(n_psi, n_freq)``.

    Any ``n_freq`` in ``[1, n_psi // 2 + 1]`` at ``n_psi`` in ``{128, 256, 512}`` (bins
    above the base transform fold exactly onto it; the Nyquist bin is handled).
    Anything else reduces through the torch reference.
    """
    return int(n_psi) in LEAN_NUM_PSI and 1 <= int(n_freq) <= int(n_psi) // 2 + 1


@overload
def lean_irfft_stats_transposed(
    c: torch.Tensor,
    n_psi: int,
    decode: Literal[True] = True,
    hyp_offset: int = 0,
    outs: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None: ...
@overload
def lean_irfft_stats_transposed(
    c: torch.Tensor,
    n_psi: int,
    decode: Literal[False],
    hyp_offset: int = 0,
    outs: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None: ...
def lean_irfft_stats_transposed(
    c: torch.Tensor,
    n_psi: int,
    decode: bool = True,
    hyp_offset: int = 0,
    outs: list[torch.Tensor] | None = None,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | None
):
    """Register-resident fused psi-recovery + statistics (see lean_irfft_stats.cuh).

    Returns ``(s1, s2, vmax, amax)`` -- the moment sums over every hypothesis and
    in-plane angle of the batch, the maximum and its flat ``(q, psi)`` index -- or
    the raw ``(s1, s2, argmax_packed)``; accepts the fp16 contraction output directly.

    Parameters
    ----------
    c : torch.Tensor
        Contiguous ``(NumFreq, P, Q)`` spectrum, complex64 **or complex32**; a float16
        ``(NumFreq, P, 2Q)`` tensor of interleaved ``(re, im)`` pairs is also accepted.
        ``NumFreq`` may be anything in ``[1, n_psi // 2 + 1]``.
    n_psi : int
        Full in-plane-angle length, 128, 256 or 512.
    decode : bool, optional
        Decode the packed maximum into ``(vmax, amax)`` (default) or return it raw.
    hyp_offset : int, optional
        Global index of this batch's first hypothesis. Baked into the packed argmax
        as ``(hyp_offset + q) * n_psi + psi`` so results accumulated across batches
        (see ``outs``) decode straight to global hypothesis ids. Defaults to ``0``.
    outs : list[torch.Tensor], optional
        ``[corr_sum, corr_sum2, best_packed]`` running state (``float32``,
        ``float32``, ``int64``; each ``(P,)``) to accumulate into in place instead of
        returning fresh outputs. ``best_packed`` must start at
        :func:`lean_sentinel_packed`. With this the whole hypothesis loop needs no
        per-batch accumulate kernels at all.

    Returns
    -------
    tuple | None
        ``None`` when the extension is unavailable, the config is unsupported, or the
        kernel raised (a warning is issued once) -- callers fall back.
    """
    global _warned_runtime_failure

    module = _try_compile()
    if module is None:
        return None
    n_freq = int(c.shape[0])
    if not lean_supported(n_psi, n_freq):
        return None
    try:
        s1, s2, argmax_packed = module.lean_irfft_stats_transposed(
            c, int(n_psi), int(hyp_offset), list(outs) if outs is not None else []
        )
    except Exception as exc:
        if not _warned_runtime_failure:
            warnings.warn(
                f"Lean iRFFT+stats CUDA kernel raised at runtime for this call: {exc}",
                UserWarning,
                stacklevel=3,
            )
            _warned_runtime_failure = True
        return None
    if not decode:
        return s1, s2, argmax_packed
    vmax, amax = decode_argmax_packed(argmax_packed)
    return s1, s2, vmax, amax


def lean_debug_ifft(x: torch.Tensor) -> torch.Tensor | None:
    """Unnormalized inverse DFT of complex64 ``(B, N)`` rows, ``N`` 64, 128 or 256.

    Runs the generated register-resident transforms (``csrc/lean_fft_gen.cuh``) on the
    device; equals ``torch.fft.ifft(x, norm="forward")``. Test entry point only;
    ``None`` when the extension is unavailable.
    """
    module = _try_compile()
    return module.lean_debug_ifft(x) if module is not None else None


def lean_sentinel_packed() -> int | None:
    """int64 bit pattern of the packed ``(-inf, 0)`` sentinel (``None``: no kernel)."""
    module = _try_compile()
    return int(module.lean_sentinel_packed()) if module is not None else None
