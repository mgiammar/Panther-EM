"""The reduction: iRFFT over the in-plane angle, moments and the running maximum.

Per pixel block the correlation loop hands every (GEMM sub-batch, hypothesis batch)
spectrum ``C`` (``F`` angular frequencies of ``P_g`` pixels x ``N_b`` hypotheses) to
a :class:`BlockStats`, which recovers the ``n_psi`` in-plane angles by an inverse real
FFT and keeps, per pixel, the first two moment sums over every ``(hypothesis, psi)``
and the maximum with its hypothesis and angle -- never the correlogram itself. After
the block's last batch :meth:`BlockStats.finalize_into` writes the maps (mip, mean,
variance, z-score, best hypothesis, best psi).

Two implementations of one interface:

* :class:`LeanBlockStats` -- CUDA, ``n_psi`` in ``{128, 256, 512}``: the register-
  resident kernel (:mod:`.kernels`) reads the spectrum in the contraction's native
  ``(F, P, N)`` layout and adds each batch's moments and its packed
  ``(value, index)`` maximum straight into the block's buffers; no per-batch
  Python-side work, so the block loop can be captured as one CUDA graph. The
  packed maximum is decoded once per block.
* :class:`TorchBlockStats` -- the reference, on any device and shape: an explicit
  ``torch.fft.irfft``, the moments from Parseval and an argmax per batch.

Both reverse the psi axis of the argmax (the correlogram is ``irfft(C)``, not
``irfft(conj C)``) and report the hypothesis as the batch's virtual offset plus its
index within the batch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from panther_em.inference.search.correlation.kernels import fused_kernel_loader
from panther_em.inference.search.correlation.kernels.packed import (
    decode_argmax_packed,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from panther_em.inference.search.correlation.contraction import Spectrum
    from panther_em.inference.search.hypotheses import HypBatch


# --------------------------------------------------------------------------- #
# The reference reduction of one batch
# --------------------------------------------------------------------------- #


def hermitian_spectrum(spectrum: torch.Tensor, n_psi: int) -> tuple[torch.Tensor, bool]:
    """``(spectrum, nyquist)``: complex64 with real-valued DC (and a Nyquist) bin."""
    spectrum = spectrum.to(torch.complex64)
    n_freq = spectrum.shape[-1]
    nyquist = n_psi % 2 == 0 and n_freq == n_psi // 2 + 1
    stop = n_freq - 1 if nyquist and n_freq > 1 else n_freq
    parts = [spectrum[..., :1].real.to(torch.complex64), spectrum[..., 1:stop]]

    if stop < n_freq:
        parts.append(spectrum[..., stop:].real.to(torch.complex64))

    return torch.cat(parts, dim=-1), nyquist


def max_spectrum(
    spectrum: torch.Tensor, n_psi: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(vmax, amax)`` of :func:`reduce_spectrum` without the moment sums."""
    corr = torch.fft.irfft(
        hermitian_spectrum(spectrum, n_psi)[0], n=n_psi, dim=-1, norm="forward"
    )
    vmax, amax = corr.reshape(corr.shape[0], -1).max(dim=1)

    return vmax, amax


def reduce_spectrum(
    spectrum: torch.Tensor, n_psi: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Per-pixel reduction of one batch's angular spectrum.

    Parameters
    ----------
    spectrum : torch.Tensor
        ``(P, N, F)`` complex rfft-form angular spectrum ``C`` of ``N`` hypotheses.
    n_psi : int
        In-plane angles recovered by ``irfft(C, n=n_psi, norm="forward")``.

    Returns
    -------
    tuple of torch.Tensor
        ``(s1, s2, vmax, amax)``, each ``(P,)``: the first and second moment sums
        over the batch's ``N * n_psi`` correlations, their maximum and its flat index
        into the ``(N, n_psi)`` grid (``hyp = amax // n_psi``, ``psi = amax % n_psi``).

    Notes
    -----
    The moments come from the spectrum by Parseval (``F <= n_psi // 2 + 1`` bins
    instead of ``n_psi`` samples): ``s1 = n_psi \sum Re C_0``,
    ``s2 = n_psi \sum (C_0^2 + 2 \sum_{k>0} |C_k|^2)``, with a Nyquist bin counted
    once and by its real part.

    The correlogram is real: the DC bin and a Nyquist bin enter by their real parts
    only (as in the fused kernel). Their imaginary parts are dropped here explicitly
    because cuFFT's complex-to-real transform leaves them undefined: with a nonzero
    imaginary Nyquist part its output depends on the batch shape (up to 4 % on the
    maximum).
    """
    spectrum, nyquist = hermitian_spectrum(spectrum, n_psi)
    corr = torch.fft.irfft(spectrum, n=n_psi, dim=-1, norm="forward")
    p = corr.shape[0]

    re, im = spectrum.real, spectrum.imag
    dc_real = re[..., 0]
    power = re * re + im * im  # |C|^2, (P, N, F)

    s1 = n_psi * dc_real.sum(dim=1)
    per_hyp = n_psi * (dc_real * dc_real + 2.0 * power[..., 1:].sum(dim=2))
    if nyquist:
        # the Nyquist bin is real and not doubled: replace its 2 |C|^2 by Re(C)^2
        per_hyp = per_hyp - n_psi * (2.0 * power[..., -1] - re[..., -1] * re[..., -1])
    s2 = per_hyp.sum(dim=1)

    vmax, amax = corr.reshape(p, -1).max(dim=1)
    return s1, s2, vmax, amax


# --------------------------------------------------------------------------- #
# Block statistics
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SpectrumSpec:
    """What a pass's spectra look like; decides which reduction applies.

    Attributes
    ----------
    dtype : torch.dtype
        complex64 or complex32 spectrum.
    n_freq : int
        ``F = k_stop`` angular frequencies.
    n_psi : int
        In-plane samples recovered by the inverse FFT.
    gemm_pixels : int
        Pixels per contraction (``P_g``).
    n_sub : int
        GEMM sub-batches per block (``block_pixels / gemm_pixels``).
    hyp_batch : int
        Hypotheses per contraction.
    n_virtual : int
        Hypotheses enumerated per pixel (all defocus planes x selected templates).
    device : torch.device
        Compute device.
    """

    dtype: torch.dtype
    n_freq: int
    n_psi: int
    gemm_pixels: int
    n_sub: int
    hyp_batch: int
    n_virtual: int
    device: torch.device


class BlockStats(ABC):
    """Per-pixel moments and maximum of one block over every hypothesis batch.

    Parameters
    ----------
    n_sub : int
        GEMM sub-batches per block.
    gemm_pixels : int
        Pixels per sub-batch.
    n_psi : int
        In-plane samples.
    device : torch.device
        Where the state lives.

    Attributes
    ----------
    corr_sum, corr_sum2 : torch.Tensor
        ``(n_sub * gemm_pixels,)`` float32 first and second moment sums.
    n_evaluated : int
        ``(hypothesis, psi)`` pairs every pixel of the block has seen.
    """

    #: Whether the reduction runs on the fused kernel (and the loop may be graphed).
    fused: bool = False

    def __init__(
        self, *, n_sub: int, gemm_pixels: int, n_psi: int, device: torch.device
    ) -> None:
        self.n_sub = int(n_sub)
        self.gemm_pixels = int(gemm_pixels)
        self.n_psi = int(n_psi)
        self.device = torch.device(device)
        n = self.n_sub * self.gemm_pixels
        self.corr_sum = torch.zeros(n, device=self.device, dtype=torch.float32)
        self.corr_sum2 = torch.zeros(n, device=self.device, dtype=torch.float32)
        self.n_evaluated = 0

    def rows(self, sub: int) -> slice:
        """The block rows of GEMM sub-batch ``sub``."""
        return slice(sub * self.gemm_pixels, (sub + 1) * self.gemm_pixels)

    @torch.no_grad()
    def clear(self) -> None:
        """Reset for a new block (in place: a captured loop holds the buffers)."""
        self.corr_sum.zero_()
        self.corr_sum2.zero_()
        self._clear_best()
        self.n_evaluated = 0

    @torch.no_grad()
    def consume(self, spectrum: Spectrum, batch: HypBatch, sub: int) -> None:
        """Reduce one hypothesis batch's spectrum into sub-batch ``sub``'s rows."""
        self._consume(spectrum, batch, self.rows(sub))
        if sub == 0:  # every sub-batch sees the same hypotheses
            self.n_evaluated += batch.size * self.n_psi

    def note_replayed(self, n_evaluated: int) -> None:
        """Set the evaluation count after a CUDA-graph replay (no Python ran)."""
        self.n_evaluated = int(n_evaluated)

    @torch.no_grad()
    def finalize_into(
        self, outs: Mapping[str, torch.Tensor], start: int, n_valid: int
    ) -> None:
        """Write the block's maps into rows ``[start, start + n_valid)`` of ``outs``.

        Keys: ``mip``, ``mean``, ``variance``, ``zscore``, ``best_index``, ``best_psi``.
        """
        best, best_hyp, best_psi = self._best()
        n = max(self.n_evaluated, 1)
        mean = self.corr_sum / n
        variance = (self.corr_sum2 / n - mean * mean).clamp_min_(0.0)
        zscore = (best - mean) / variance.sqrt().clamp_min(1e-12)
        maps = {
            "mip": best,
            "mean": mean,
            "variance": variance,
            "zscore": zscore,
            "best_index": best_hyp,
            "best_psi": best_psi,
        }
        for key, value in maps.items():
            target = outs[key]
            target[start : start + n_valid] = value[:n_valid].to(
                target.device, target.dtype
            )

    @abstractmethod
    def _clear_best(self) -> None:
        """Reset the running maximum."""

    @abstractmethod
    def _consume(self, spectrum: Spectrum, batch: HypBatch, rows: slice) -> None:
        """Accumulate one batch into ``rows`` of the block."""

    @abstractmethod
    def _best(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(value, virtual hypothesis id, psi index)`` of every row's maximum."""


class TorchBlockStats(BlockStats):
    """The reference reduction: ``irfft``, Parseval moments and argmax per batch."""

    def __init__(
        self, *, n_sub: int, gemm_pixels: int, n_psi: int, device: torch.device
    ) -> None:
        super().__init__(
            n_sub=n_sub, gemm_pixels=gemm_pixels, n_psi=n_psi, device=device
        )
        n = self.n_sub * self.gemm_pixels
        self.best_corr = torch.full((n,), float("-inf"), device=self.device)
        self.best_hypothesis = torch.full(
            (n,), -1, device=self.device, dtype=torch.long
        )
        self.best_psi = torch.full((n,), -1, device=self.device, dtype=torch.long)

    def _clear_best(self) -> None:
        self.best_corr.fill_(float("-inf"))
        self.best_hypothesis.fill_(-1)
        self.best_psi.fill_(-1)

    def _consume(self, spectrum: Spectrum, batch: HypBatch, rows: slice) -> None:
        n_psi = self.n_psi
        s1, s2, vmax, amax = reduce_spectrum(spectrum.pnf(), n_psi)
        self.corr_sum[rows] += s1
        self.corr_sum2[rows] += s2
        best = self.best_corr[rows]
        improved = vmax > best
        hyp = batch.virtual_offset + torch.div(amax, n_psi, rounding_mode="floor")
        psi = (n_psi - amax % n_psi) % n_psi  # irfft(C), not irfft(conj C)
        torch.where(improved, vmax, best, out=best)
        best_hyp = self.best_hypothesis[rows]
        torch.where(improved, hyp, best_hyp, out=best_hyp)
        best_psi = self.best_psi[rows]
        torch.where(improved, psi, best_psi, out=best_psi)

    def _best(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.best_corr, self.best_hypothesis, self.best_psi


class LeanBlockStats(BlockStats):
    """The fused kernel accumulates every batch in-kernel into the block's buffers.

    The kernel adds each batch's moment sums to :attr:`corr_sum` /
    :attr:`corr_sum2` and merges its packed ``(value, (offset + q) * n_psi + psi)``
    maximum into :attr:`best_packed` by ``atomicMax``; :meth:`finalize_into` decodes
    the packed maximum once per block.
    """

    fused = True

    def __init__(
        self, *, n_sub: int, gemm_pixels: int, n_psi: int, device: torch.device
    ) -> None:
        super().__init__(
            n_sub=n_sub, gemm_pixels=gemm_pixels, n_psi=n_psi, device=device
        )
        sentinel = fused_kernel_loader.lean_sentinel_packed()
        if sentinel is None:
            raise RuntimeError("the fused iRFFT + statistics kernel is unavailable")
        self.sentinel = sentinel
        n = self.n_sub * self.gemm_pixels
        self.best_packed = torch.full(
            (n,), sentinel, device=self.device, dtype=torch.int64
        )

    def _clear_best(self) -> None:
        self.best_packed.fill_(self.sentinel)

    def _consume(self, spectrum: Spectrum, batch: HypBatch, rows: slice) -> None:
        ran = fused_kernel_loader.lean_irfft_stats_transposed(
            spectrum.native,
            self.n_psi,
            decode=False,
            hyp_offset=batch.virtual_offset,
            outs=[self.corr_sum[rows], self.corr_sum2[rows], self.best_packed[rows]],
        )
        if ran is None:
            raise RuntimeError(
                f"the fused kernel rejected a ({spectrum.native.dtype}, "
                f"{tuple(spectrum.native.shape)}) spectrum at n_psi={self.n_psi}"
            )

    def _best(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        value, flat = decode_argmax_packed(self.best_packed)
        n_psi = self.n_psi
        hyp = torch.div(flat, n_psi, rounding_mode="floor")
        psi = (n_psi - flat % n_psi) % n_psi
        return value, hyp, psi


def fused_stats_supported(spec: SpectrumSpec) -> bool:
    """Whether a pass with ``spec`` reduces on the fused kernel.

    True when the spectrum is a CUDA complex64/complex32 tensor, ``(n_psi, n_freq)``
    is covered by the register-resident kernel, the packed index
    ``n_virtual * n_psi`` fits 32 bits and the extension compiled.
    """
    return bool(
        spec.device.type == "cuda"
        and spec.dtype in (torch.complex64, torch.complex32)
        and fused_kernel_loader.lean_supported(spec.n_psi, spec.n_freq)
        and spec.n_virtual * spec.n_psi < 2**32
        and fused_kernel_loader.lean_sentinel_packed() is not None
    )


def make_block_stats(spec: SpectrumSpec, *, fused: bool) -> BlockStats:
    """A fresh block accumulator for ``spec`` (the fused kernel when ``fused``)."""
    cls: type[BlockStats] = LeanBlockStats if fused else TorchBlockStats
    return cls(
        n_sub=spec.n_sub,
        gemm_pixels=spec.gemm_pixels,
        n_psi=spec.n_psi,
        device=spec.device,
    )
