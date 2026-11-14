"""zipFFT as a backend of the FFT engine (optional dependency).

zipFFT's ``padded_cconv2d.corr`` fuses the kernel's zero-padded forward FFT, the
product with the image's spectrum and the inverse FFT, writing only the valid
correlation rows and columns. Its kernels are compiled for fixed shapes: a kernel box,
an FFT shape ``(S, FX)`` and a batch (kernels per call). The FFT engine runs a strip on
zipFFT when its shape is compiled for the box and splits a chunk of cells into calls of
the compiled batches, largest first.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from types import ModuleType

FFTBackend = Literal["auto", "torch", "zipfft"]
"""``"torch"``: cuFFT at any size; ``"zipfft"``: zipFFT only (an error where a strip
is not compiled); ``"auto"``: zipFFT where compiled, cuFFT elsewhere."""

FFT_BACKENDS = ("auto", "torch", "zipfft")

DEFAULT_ZIPFFT_BATCH = 1
"""Largest batch (kernels per zipFFT call) a chunk of cells is split into."""


@functools.cache
def zipfft_module() -> ModuleType | None:
    """The ``zipfft`` package, or ``None`` if it is not installed (or fails to load)."""
    try:
        import zipfft  # type: ignore[import-not-found]
    except (ImportError, OSError):
        return None
    module: ModuleType = zipfft
    return module


@functools.cache
def _compiled() -> dict[tuple[int, int], dict[tuple[int, int], tuple[int, ...]]]:
    """``{box: {(S, FX): batches (descending)}}`` of the compiled correlations."""
    module = zipfft_module()
    if module is None:
        return {}
    found: dict[tuple[int, int], dict[tuple[int, int], set[int]]] = {}
    for config in module.padded_cconv2d.get_supported_conv_configs():
        k_h, k_w, fft_h, fft_w, batch, cross_correlate = config[:6]
        if cross_correlate:
            shapes = found.setdefault((int(k_h), int(k_w)), {})
            shapes.setdefault((int(fft_h), int(fft_w)), set()).add(int(batch))
    return {
        box: {
            shape: tuple(sorted(batches, reverse=True))
            for shape, batches in shapes.items()
            if 1 in batches  # every chunk splits into calls
        }
        for box, shapes in found.items()
    }


def zipfft_unavailable(device: torch.device | str) -> str | None:
    """Why zipFFT cannot run on ``device`` (``None``: it can)."""
    device = torch.device(device)
    if device.type != "cuda":
        return f"zipFFT runs on CUDA devices only, not {device}"
    module = zipfft_module()
    if module is None:
        return "zipFFT is not installed"
    major, minor = torch.cuda.get_device_capability(device)
    archs = list(module.get_cuda_architectures())
    for arch in archs:
        a_major, _, a_minor = str(arch).partition(".")
        if int(a_major) == major and int(a_minor or 0) <= minor:
            return None
    return f"zipFFT is compiled for {archs}, not compute capability {major}.{minor}"


def zipfft_shapes(
    box: tuple[int, int], device: torch.device | str
) -> dict[tuple[int, int], tuple[int, ...]]:
    """``{(S, FX): batches}`` zipFFT runs for kernel ``box`` on ``device`` (or ``{}``).

    Only shapes compiled at batch 1 (so any chunk of cells splits into calls), batches
    in descending order.
    """
    if zipfft_unavailable(device) is not None:
        return {}
    return dict(_compiled().get((int(box[0]), int(box[1])), {}))


def split_batch(n: int, batches: tuple[int, ...], largest: int) -> list[int]:
    """``n`` kernels as calls of ``batches`` (descending, with 1), largest first.

    No call is above ``largest``.
    """
    sizes = [b for b in batches if b <= max(1, largest)] or [1]
    out = []
    for size in sizes:
        count, n = divmod(n, size)
        out.extend([size] * count)
    return out


def correlate(
    kernels: torch.Tensor,
    workspace: torch.Tensor,
    image: torch.Tensor,
    output: torch.Tensor,
    fft_shape: tuple[int, int],
) -> None:
    """``output = valid(ifft2(image * conj(fft2(kernels, fft_shape))))`` by zipFFT.

    Parameters
    ----------
    kernels : torch.Tensor
        ``(b, kH, kW)`` complex64, contiguous.
    workspace : torch.Tensor
        ``(b, S, FX)`` complex64 scratch.
    image : torch.Tensor
        ``(FX, S)`` complex64: the image's spectrum, transposed and contiguous.
    output : torch.Tensor
        ``(b, S - kH + 1, FX - kW + 1)`` complex64, contiguous.
    fft_shape : tuple[int, int]
        ``(S, FX)``.
    """
    module = zipfft_module()
    assert module is not None
    module.padded_cconv2d.corr(kernels, workspace, image, output, *fft_shape)
