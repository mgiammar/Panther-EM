"""Per-pixel search statistics tracked online over a batch of hypotheses.

:class:`PixelStats` accumulates the first two moments plus the running best correlation
(and the hypothesis / in-plane angle that produced it) for a batch of image pixels, so a
search never has to materialize the full ``(pixels, hypotheses, psi)`` correlogram.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, overload

import torch

from panther_em.inference.search import fused_kernel_loader
from panther_em.inference.search.packed import decode_argmax_packed

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from panther_em.inference.search.hypotheses import HypBatch


def _bind_capture_stream(
    device: torch.device, warmup: Callable[[], None]
) -> torch.cuda.Stream:
    """Bind a private stream to ``device``, run ``warmup`` on it, and return it.

    The returned stream is ready to pass as ``torch.cuda.graph(..., stream=...)``'s
    ``stream`` argument for the actual capture.

    Notes
    -----
    Explicitly binding both warm-up and capture to a stream on ``device`` matters:
    without ``stream=`` here, ``torch.cuda.graph()`` falls back to a lazily-created,
    process-wide *default* capture stream pinned to whatever device happened to be
    ambient-current the first time any CUDA graph was captured in this process -- if
    ``device`` differs from that (e.g. a multi-GPU process that never called
    ``torch.cuda.set_device(device.index)``), the capture silently records zero nodes
    (a "CUDA Graph is empty" warning) and replay becomes a no-op.
    """
    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(capture_stream):
        warmup()
    torch.cuda.current_stream(device).wait_stream(capture_stream)
    torch.cuda.synchronize(device)
    return capture_stream


@functools.cache
def _cuda_graph_capture_supported(device_index: int) -> bool:
    """One-time capability probe for CUDA graph capture on a given device index."""
    try:
        device = torch.device("cuda", device_index)
        x = torch.zeros(1, device=device)

        def _warmup() -> None:
            x.add_(1)

        capture_stream = _bind_capture_stream(device, _warmup)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=capture_stream):
            x.add_(1)
        g.replay()
        torch.cuda.synchronize(device)
        return bool(x.item() == 2.0)
    except Exception:
        return False


@torch.compile
def _reduce_stats(
    corr: torch.Tensor,
    spectrum_ri: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Fused per-pixel reductions over a ``(P, hyp, psi)`` correlogram batch.

    NOTE: Uses Parseval's theorem to compute the first two moment sums from the angular
          frequency spectrum. This is slightly faster since num_k < num_psi (frequency
          zero padding).

    Parameters
    ----------
    corr : torch.Tensor
        Tensor with correlation values with shape (num_pixels, hyp_batch, num_psi)
        where `num_pixels` is the number of pixels in the batch, `hyp_batch` is the
        number of hypotheses in the batch, and `num_psi` are all in-plane angles.
    spectrum_ri : torch.Tensor
        The rfft-form angular spectrum ``C`` this ``corr`` was produced from. Must be
        viewed in real-mode so shape (num_pixels, hyp_batch, n_freq, 2). The two moment
        sums are computed from it directly instead of by reducing the full ``corr``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Tuple of four tensors, each with shape (num_pixels,):
        * s1 : torch.Tensor
            The first moment sum over all hypotheses and in-plane angles.
        * s2 : torch.Tensor
            The second moment sum over all hypotheses and in-plane angles.
        * vmax : torch.Tensor
            The maximum correlation value over all hypotheses and in-plane angles.
        * amax : torch.Tensor
            The flattened index of the maximum correlation value over all hypotheses
            and in-plane angles. Can be decoded to hypothesis and psi indices with
            ``hyp_idx = amax // num_psi`` and ``psi_idx = amax % num_psi``.
    """
    p, _, num_psi = corr.shape
    n_freq = spectrum_ri.shape[2]

    re, im = spectrum_ri[..., 0], spectrum_ri[..., 1]
    dc_real = re[..., 0]
    power = re * re + im * im  # abs(C)^2, (P, hyp, n_freq)

    s1 = num_psi * dc_real.sum(dim=1)
    per_hyp = num_psi * (dc_real * dc_real + 2.0 * power[..., 1:].sum(dim=2))
    if num_psi % 2 == 0 and n_freq == num_psi // 2 + 1:
        # Last bin is Nyquist: realified and NOT doubled, so replace it with Re(C)^2
        per_hyp = per_hyp - num_psi * (2.0 * power[..., -1] - re[..., -1] * re[..., -1])
    s2 = per_hyp.sum(dim=1)

    vmax, amax = corr.reshape(p, -1).max(dim=1)
    return s1, s2, vmax, amax


class PixelStats:
    """Helper for tracing statistics over hypothesis space in a small pixel batch.

    Attributes
    ----------
    hypothesis_count : int
        The number of hypotheses processed.
    num_pixels : int
        The number of pixels in the batch.
    corr_sum : torch.Tensor
        The sum of correlation values.
    corr_sum2 : torch.Tensor
        The sum of squared correlation values.
    best_corr : torch.Tensor
        The best correlation value found.
    best_hypothesis : torch.Tensor
        Global index corresponding to the hypothesis which produced the best correlation
        value.
    best_psi_angle : torch.Tensor
        The in-plane rotation angle (also a hypothesis, but separate axis) which
        produced the best correlation value.

    Methods
    -------
    clear : None
        Clear all tracked statistics.
    update(spectrum: torch.Tensor, hyp_global_idx: torch.Tensor, num_psi: int) : None
        Update tracked statistics with a new hypothesis batch's raw spectrum.
    finalize : dict[str, torch.Tensor]
        Return a dictionary of the final statistics (maximum intensity projection /
        maximum inner product, z-score, mean, variance, best hypothesis index,
        best psi index).

    Notes
    -----
    With ``fused=True`` (CUDA) the tracker uses the fused CUDA kernels. Two fast
    paths, tried in order:

    1. **In-kernel accumulation** (:meth:`update_graphed`): for the register-resident
       kernel's domain (``n_psi`` in ``{128, 256, 512}``, ``NumFreq <= n_psi/2 + 1``,
       spectrum in the GEMM's native ``(NumFreq, P, N)`` layout, complex64 or
       complex32) the kernel adds the batch's moments straight into :attr:`corr_sum`
       / :attr:`corr_sum2` and merges a packed ``(value, (hyp, psi) index)`` maximum
       into :attr:`best_packed`; the packed best is decoded into ``best_corr`` /
       ``best_hypothesis`` / ``best_psi_angle`` once, in :meth:`merge_packed`.
    2. **Fused reduce** (:meth:`_reduce`): the lean kernel or, failing that, the
       cuFFTDx block-FFT kernel, feeding the ordinary accumulate.

    Everything falls back to the pure-torch reduction when a kernel does not apply.
    """

    hypothesis_count: int
    num_pixels: int
    corr_sum: torch.Tensor
    corr_sum2: torch.Tensor
    best_corr: torch.Tensor
    best_hypothesis: torch.Tensor
    best_psi_angle: torch.Tensor

    def __init__(
        self,
        num_pixels: int,
        device: torch.device,
        accumulate_dtype: torch.dtype = torch.float32,
        *,
        fused: bool = False,
    ):
        """Initialize a new pixel statistics tracker (``fused``: use the CUDA kernels)."""
        self.hypothesis_count = 0
        self.num_pixels = num_pixels
        self.fused = bool(fused)
        self._lean_sentinel: int | None = (
            fused_kernel_loader.lean_sentinel_packed() if self.fused else None
        )
        # in-kernel packed best (fused only); the num_psi it was accumulated with, or
        # None if untouched since the last clear()/merge_packed()
        self.best_packed: torch.Tensor | None = None
        if self.fused:
            self.best_packed = torch.full(
                (num_pixels,),
                self._lean_sentinel if self._lean_sentinel is not None else 0,
                device=device,
                dtype=torch.int64,
            )
        self._packed_num_psi: int | None = None
        # the streaming-accumulate CUDA graph, captured on first use
        self._graph: torch.cuda.CUDAGraph | None = None
        self._graph_num_psi: int | None = None
        self._graph_reverse_psi_axis: bool | None = None
        self.corr_sum = torch.zeros(num_pixels, device=device, dtype=accumulate_dtype)
        self.corr_sum2 = torch.zeros(num_pixels, device=device, dtype=accumulate_dtype)
        self.best_corr = torch.full(
            (num_pixels,), float("-inf"), device=device, dtype=accumulate_dtype
        )
        self.best_hypothesis = torch.full(
            (num_pixels,), -1, device=device, dtype=torch.long
        )
        self.best_psi_angle = torch.full(
            (num_pixels,), -1, device=device, dtype=torch.long
        )

    BUFFERS: tuple[str, ...] = (
        "corr_sum",
        "corr_sum2",
        "best_corr",
        "best_hypothesis",
        "best_psi_angle",
    )

    @classmethod
    def view_of(cls, parent: PixelStats, start: int, stop: int) -> PixelStats:
        """A tracker whose buffers are views ``[start:stop]`` of ``parent``'s.

        Lets a block keep one set of buffers for all its GEMM sub-batches: the
        sub-batches accumulate into their slices and the block finalizes once.
        Scalar bookkeeping (``hypothesis_count``) is per view.
        """
        obj = cls.__new__(cls)
        obj.hypothesis_count = 0
        obj.num_pixels = stop - start
        for name in cls.BUFFERS:
            setattr(obj, name, getattr(parent, name)[start:stop])
        obj.fused = parent.fused
        obj._lean_sentinel = parent._lean_sentinel
        obj.best_packed = (
            None if parent.best_packed is None else parent.best_packed[start:stop]
        )
        obj._packed_num_psi = None
        obj._graph = None
        obj._graph_num_psi = None
        obj._graph_reverse_psi_axis = None
        return obj

    def clear_bookkeeping(self) -> None:
        """Reset the scalar state only (a view whose buffers the block cleared)."""
        self.hypothesis_count = 0
        self._packed_num_psi = None

    @torch.no_grad()
    def clear(self) -> None:
        """Clear all tracked statistics."""
        self.clear_bookkeeping()
        self.corr_sum.zero_()
        self.corr_sum2.zero_()
        self.best_corr.fill_(float("-inf"))
        self.best_hypothesis.fill_(-1)
        self.best_psi_angle.fill_(-1)
        if self.best_packed is not None and self._lean_sentinel is not None:
            self.best_packed.fill_(self._lean_sentinel)

    def _reduce(
        self, spectrum: torch.Tensor, num_psi: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reduce one hypothesis batch's spectrum to ``(s1, s2, vmax, amax)``.

        With ``fused`` the CUDA kernels are tried first (:meth:`_try_fused_reduce`).

        Parameters
        ----------
        spectrum : torch.Tensor
            The rfft-form angular spectrum ``C``, shape (num_pixels, hyp_batch, n_freq).
        num_psi : int
            Number of in-plane angles to reconstruct via irfft.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            ``(s1, s2, vmax, amax)``, each shape (num_pixels,). See
            :func:`_reduce_stats`.
        """
        if self.fused:
            result = self._try_fused_reduce(spectrum, num_psi, decode=True)
            if result is not None:
                return result
        corr = torch.fft.irfft(spectrum, n=num_psi, dim=-1, norm="forward")
        return _reduce_stats(  # type: ignore[no-any-return]
            corr.to(self.best_corr.dtype), torch.view_as_real(spectrum)
        )

    @torch.no_grad()
    def update(
        self,
        spectrum: torch.Tensor,
        hyp_global_idx: torch.Tensor,
        num_psi: int,
        reverse_psi_axis: bool = True,
    ) -> None:
        """Update tracked statistics with a new hypothesis batch's raw spectrum.

        Parameters
        ----------
        spectrum : torch.Tensor
            The rfft-form angular spectrum ``C``, shape (num_pixels, hyp_batch, n_freq)
            where `num_pixels` must match held in `self.num_pixels`.
        hyp_global_idx : torch.Tensor
            Integer tensor referencing which indices these hypotheses correspond to,
            shape (hyp_batch,).
        num_psi : int
            Number of in-plane angles (the full irfft output length).
        reverse_psi_axis : bool, optional
            Whether to reverse the psi axis when updating the best psi angle. By default
            True because correlogram is produced by irfft(C) rather than irfft(C.conj())
        """
        total_hypotheses = spectrum.shape[1] * num_psi
        s1, s2, vmax, amax = self._reduce(spectrum, num_psi)
        self._accumulate(
            s1,
            s2,
            vmax,
            amax,
            hyp_global_idx,
            num_psi,
            total_hypotheses,
            reverse_psi_axis,
        )

        return None

    @torch.compile
    def _accumulate(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor,
        vmax: torch.Tensor,
        amax: torch.Tensor,
        hyp_global_idx: torch.Tensor,
        num_psi: int,
        total_hypotheses: int,
        reverse_psi_axis: bool,
    ) -> None:
        """Shared accumulate + branchless best-update logic for one reduced batch.

        Parameters
        ----------
        s1, s2 : torch.Tensor
            First and second moment sums, shape (num_pixels,).
        vmax : torch.Tensor
            Maximum correlation value per pixel, shape (num_pixels,).
        amax : torch.Tensor
            Flattened index of maximum (into the (hyp_batch, num_psi) grid), shape
            (num_pixels,).
        hyp_global_idx : torch.Tensor
            Global hypothesis indices for this batch, shape (hyp_batch,).
        num_psi : int
            Number of in-plane angles (for decoding amax).
        total_hypotheses : int
            Number of (hypothesis, psi) pairs contributed by this batch, added to
            ``self.hypothesis_count``.
        reverse_psi_axis : bool
            Whether to reverse the psi axis when updating the best psi angle.
        """
        vmax_cast = vmax.to(self.best_corr.dtype)
        self.corr_sum += s1.to(self.best_corr.dtype)
        self.corr_sum2 += s2.to(self.best_corr.dtype)
        self.hypothesis_count += total_hypotheses

        improved = vmax_cast > self.best_corr

        # Decode indices for all pixels to undo flattening of last dim
        n_local = torch.div(amax, num_psi, rounding_mode="floor")
        global_hyp = hyp_global_idx[n_local]
        if reverse_psi_axis:
            psi = (num_psi - amax % num_psi) % num_psi
        else:
            psi = amax % num_psi

        # In place: a BlockStats sub-batch is a view of the block's buffers.
        torch.where(improved, vmax_cast, self.best_corr, out=self.best_corr)
        torch.where(
            improved, global_hyp, self.best_hypothesis, out=self.best_hypothesis
        )
        torch.where(improved, psi, self.best_psi_angle, out=self.best_psi_angle)

        return None

    def _build_graphed_accumulate(self, num_psi: int, reverse_psi_axis: bool) -> None:
        """One-time capture of a tiny CUDA graph for the streaming accumulate step.

        Notes
        -----
        Only supports the contiguous-arange ``hyp_offset`` fast path.
        """
        device = self.corr_sum.device
        self._graph_s1 = torch.zeros(
            self.num_pixels, device=device, dtype=self.corr_sum.dtype
        )
        self._graph_s2 = torch.zeros_like(self._graph_s1)
        self._graph_vmax = torch.zeros(
            self.num_pixels, device=device, dtype=self.best_corr.dtype
        )
        self._graph_amax = torch.zeros(self.num_pixels, device=device, dtype=torch.long)
        self._graph_h0 = torch.zeros((), device=device, dtype=torch.long)

        def _step() -> None:
            vmax_cast = self._graph_vmax.to(self.best_corr.dtype)
            self.corr_sum.add_(self._graph_s1.to(self.corr_sum.dtype))
            self.corr_sum2.add_(self._graph_s2.to(self.corr_sum2.dtype))

            n_local = torch.div(self._graph_amax, num_psi, rounding_mode="floor")
            global_hyp = self._graph_h0 + n_local
            if reverse_psi_axis:
                psi = (num_psi - self._graph_amax % num_psi) % num_psi
            else:
                psi = self._graph_amax % num_psi

            improved = vmax_cast > self.best_corr
            torch.where(improved, vmax_cast, self.best_corr, out=self.best_corr)
            torch.where(
                improved, global_hyp, self.best_hypothesis, out=self.best_hypothesis
            )
            torch.where(improved, psi, self.best_psi_angle, out=self.best_psi_angle)

        def _warmup() -> None:
            for _ in range(3):
                _step()

        # See _bind_capture_stream's docstring for why `stream=` must be explicit here
        # -- without it, a multi-GPU process can silently capture an empty graph whose
        # replay is a no-op, leaving corr_sum/best_corr/etc. stuck at their initial
        # values.
        capture_stream = _bind_capture_stream(device, _warmup)

        # Undo the warm-up's accumulation before it becomes "real" tracked state.
        self.corr_sum.zero_()
        self.corr_sum2.zero_()
        self.best_corr.fill_(float("-inf"))
        self.best_hypothesis.fill_(-1)
        self.best_psi_angle.fill_(-1)

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph, stream=capture_stream):
            _step()
        self._graph_num_psi = num_psi
        self._graph_reverse_psi_axis = reverse_psi_axis

    @torch.no_grad()
    def accumulate_graphed(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor,
        vmax: torch.Tensor,
        amax: torch.Tensor,
        hyp_offset: int,
        num_psi: int,
        reverse_psi_axis: bool = True,
    ) -> None:
        """Streaming accumulate via a captured CUDA graph replay.

        Raises
        ------
        ValueError
            If called with a different ``(num_psi, reverse_psi_axis)`` than the one
            this instance's graph was captured with -- use a fresh instance instead.
        """
        if self._graph is None:
            self._build_graphed_accumulate(num_psi, reverse_psi_axis)
        elif (
            num_psi != self._graph_num_psi
            or reverse_psi_axis != self._graph_reverse_psi_axis
        ):
            raise ValueError(
                "accumulate_graphed's captured graph is specialized to "
                f"(num_psi={self._graph_num_psi}, "
                f"reverse_psi_axis={self._graph_reverse_psi_axis}); "
                f"got (num_psi={num_psi}, reverse_psi_axis={reverse_psi_axis}). "
                "Use a fresh PixelStats instance for a different (num_psi, "
                "reverse_psi_axis)."
            )

        self._graph_s1.copy_(s1, non_blocking=True)
        self._graph_s2.copy_(s2, non_blocking=True)
        self._graph_vmax.copy_(vmax, non_blocking=True)
        self._graph_amax.copy_(amax, non_blocking=True)
        self._graph_h0.fill_(hyp_offset)
        assert self._graph is not None
        self._graph.replay()

    @torch.no_grad()
    def update_graphed(
        self,
        spectrum: torch.Tensor,
        hyp_offset: int,
        num_psi: int,
        reverse_psi_axis: bool = True,
    ) -> None:
        """Like :meth:`update`, but the accumulate step replays a captured CUDA graph.

        Requires the caller to already know ``hyp_global_idx`` for this batch is a
        contiguous arange starting at ``hyp_offset`` -- callers without that guarantee
        should use :meth:`update` instead. Transparently falls back to :meth:`update`
        when CUDA graph capture isn't supported on this device (checked once per device
        via :func:`_cuda_graph_capture_supported`), so this is always safe to call.
        """
        if (
            self.fused
            and reverse_psi_axis
            and self._lean_sentinel is not None
            and self.best_packed is not None
            and self.corr_sum.dtype == torch.float32
            and self.inkernel_supported(spectrum, num_psi, hyp_offset)
        ):
            ran = (
                fused_kernel_loader.lean_irfft_stats_transposed(
                    spectrum.permute(2, 0, 1),
                    num_psi,
                    decode=False,
                    hyp_offset=hyp_offset,
                    outs=[self.corr_sum, self.corr_sum2, self.best_packed],
                )
                is not None
            )
            if ran:
                self.hypothesis_count += spectrum.shape[1] * num_psi
                self._packed_num_psi = num_psi
                return None
        if not (
            spectrum.is_cuda and _cuda_graph_capture_supported(spectrum.device.index)
        ):
            hyp_global_idx = torch.arange(
                hyp_offset, hyp_offset + spectrum.shape[1], device=spectrum.device
            )
            return self.update(spectrum, hyp_global_idx, num_psi, reverse_psi_axis)

        s1, s2, vmax, amax = self._reduce(spectrum, num_psi)
        self.accumulate_graphed(
            s1, s2, vmax, amax, hyp_offset, num_psi, reverse_psi_axis
        )
        self.hypothesis_count += spectrum.shape[1] * num_psi
        return None

    @staticmethod
    def inkernel_supported(
        spectrum: torch.Tensor, num_psi: int, hyp_offset: int
    ) -> bool:
        """Whether :meth:`update_graphed` can accumulate ``spectrum`` in-kernel."""
        if not (
            spectrum.is_cuda
            and spectrum.dtype in (torch.complex64, torch.complex32)
            and spectrum.dim() == 3
        ):
            return False
        transposed = spectrum.permute(2, 0, 1)
        return bool(
            transposed.is_contiguous()
            and fused_kernel_loader.lean_supported(num_psi, transposed.shape[0])
            and (hyp_offset + transposed.shape[2]) * num_psi < 2**32
        )

    @torch.no_grad()
    def merge_packed(self) -> None:
        """Fold the in-kernel packed best into ``best_corr`` / ``_hypothesis`` / ``_psi``.

        Idempotent; a no-op unless the in-kernel path ran since the last
        :meth:`clear`. Uses ``reverse_psi_axis=True`` decoding, the only mode routed
        through the in-kernel path.
        """
        if self._packed_num_psi is None or self.best_packed is None:
            return
        num_psi = self._packed_num_psi
        vmax, flat = decode_argmax_packed(self.best_packed)
        global_hyp = torch.div(flat, num_psi, rounding_mode="floor")
        psi = (num_psi - flat % num_psi) % num_psi
        vmax_cast = vmax.to(self.best_corr.dtype)
        improved = vmax_cast > self.best_corr
        # In place (`out=`), never rebinding the attributes: a captured CUDA graph
        # (executor.HypLoop) holds these exact buffers, and its captured clear()
        # must keep resetting the tensors finalize() reads.
        torch.where(improved, vmax_cast, self.best_corr, out=self.best_corr)
        torch.where(
            improved, global_hyp, self.best_hypothesis, out=self.best_hypothesis
        )
        torch.where(improved, psi, self.best_psi_angle, out=self.best_psi_angle)
        if self._lean_sentinel is not None:
            self.best_packed.fill_(self._lean_sentinel)
        self._packed_num_psi = None

    @overload
    def _try_fused_reduce(
        self, spectrum: torch.Tensor, num_psi: int, decode: Literal[True]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None: ...
    @overload
    def _try_fused_reduce(
        self, spectrum: torch.Tensor, num_psi: int, decode: Literal[False]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None: ...
    def _try_fused_reduce(
        self, spectrum: torch.Tensor, num_psi: int, decode: bool
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | None
    ):
        """Fast-path selection for :meth:`_reduce`.

        Returns ``None`` (rather than raising) whenever the fused kernel can't handle
        this call -- spectrum isn't CUDA complex64, the kernel failed to compile, or
        ``(num_psi, NumFreq)`` isn't a supported config -- signaling the caller to fall
        back to the parent's pure-torch reduction.
        """
        if not (
            spectrum.is_cuda and spectrum.dtype in (torch.complex64, torch.complex32)
        ):
            return None

        # Zero-copy fast path for internal transpose.
        # Triggers when `spectrum` is (P, N, NumFreq) view of a (NumFreq, P, N)
        # contiguous tensor (layout torch.bmm produces natively, in either precision).
        # `decode` is branched on explicitly (rather than forwarded as a plain bool)
        # so each call below keeps its `Literal[True]`/`Literal[False]` overload.
        transposed = spectrum.permute(2, 0, 1)
        if transposed.is_contiguous():
            # 1. Register-resident kernel: n_psi in {128, 256, 512}, NumFreq <= n_psi/2+1, both
            #    complex dtypes. ~6x faster than the cuFFTDx kernel at n_psi=256.
            result = (
                fused_kernel_loader.lean_irfft_stats_transposed(
                    transposed, num_psi, decode=True
                )
                if decode
                else fused_kernel_loader.lean_irfft_stats_transposed(
                    transposed, num_psi, decode=False
                )
            )
            if result is not None:
                return result
            # 2. cuFFTDx block-FFT kernel (complex64 only): a complex32 spectrum
            #    (fp16 contraction, n_psi = 64) is converted once, in place of
            #    layout, rather than converted and then copied again below.
            transposed64 = (
                transposed
                if spectrum.dtype == torch.complex64
                else transposed.to(torch.complex64)
            )
            result = (
                fused_kernel_loader.fused_irfft_stats_transposed(
                    transposed64, num_psi, decode=True
                )
                if decode
                else fused_kernel_loader.fused_irfft_stats_transposed(
                    transposed64, num_psi, decode=False
                )
            )
            if result is not None:
                return result

        # 3. Non-native layout: cuFFTDx kernel with an explicit .contiguous() copy.
        if spectrum.dtype != torch.complex64:
            spectrum = spectrum.to(torch.complex64)
        if decode:
            return fused_kernel_loader.fused_irfft_stats(spectrum, num_psi, decode=True)
        return fused_kernel_loader.fused_irfft_stats(spectrum, num_psi, decode=False)

    @torch.no_grad()
    def finalize(self) -> dict[str, torch.Tensor]:
        """Finalize tracked statistics into summary dictionary.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing the final tracked statistics. Tensors exists on device
            used to initialize the PixelStats object.
        """
        self.merge_packed()
        mean = self.corr_sum / self.hypothesis_count
        variance = (self.corr_sum2 / self.hypothesis_count) - (mean**2)
        variance.clamp_min_(0.0)
        std = variance.sqrt()
        zscore = (self.best_corr - mean) / std.clamp_min(1e-12)

        # Calling .clone() on tensors ensures no overwriting on re-use after stats.clear
        return {
            "mip": self.best_corr.clone(),
            "zscore": zscore.clone(),
            "mean": mean.clone(),
            "variance": variance.clone(),
            "best_index": self.best_hypothesis.clone(),
            "best_psi": self.best_psi_angle.clone(),
        }


# --------------------------------------------------------------------------- #
# Block-level accumulation (the redesigned executor's view of the statistics)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SpectrumSpec:
    """What a stage's spectra look like; decides which statistics backend applies.

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


class BlockStats:
    """Per-pixel statistics of one block: one :class:`PixelStats` per GEMM sub-batch.

    Parameters
    ----------
    n_sub : int
        Sub-batches per block.
    gemm_pixels : int
        Pixels per sub-batch.
    device : torch.device
        Where the state lives.
    n_psi : int
        In-plane samples.
    fused : bool
        ``True`` when every batch takes the fused in-kernel accumulate path (the
        packed best is folded once per block in :meth:`finalize_into`).
    """

    def __init__(
        self,
        *,
        n_sub: int,
        gemm_pixels: int,
        device: torch.device,
        n_psi: int,
        fused: bool,
    ) -> None:
        # One tracker for the whole block; the sub-batches are views of its buffers,
        # so finalization runs once per block instead of once per sub-batch (that
        # was ~20 small launches x n_sub per block, the dominant loop overhead).
        self.block = PixelStats(int(n_sub) * int(gemm_pixels), device=device, fused=fused)
        self.stats = [
            PixelStats.view_of(self.block, sub * gemm_pixels, (sub + 1) * gemm_pixels)
            for sub in range(n_sub)
        ]
        self.n_sub = int(n_sub)
        self.gemm_pixels = int(gemm_pixels)
        self.n_psi = int(n_psi)
        self.fused = bool(fused)
        self.device = device

    @torch.no_grad()
    def clear(self) -> None:
        """Reset the block's buffers (one pass) and every sub-batch's bookkeeping."""
        self.block.clear()
        for st in self.stats:
            st.clear_bookkeeping()

    @torch.no_grad()
    def consume(self, spectrum_pnf: torch.Tensor, batch: HypBatch, sub: int) -> None:
        """Accumulate one batch's spectrum ``(P_g, N_b, F)`` into sub-batch ``sub``."""
        self.stats[sub].update_graphed(
            spectrum_pnf, batch.virtual_offset, num_psi=self.n_psi, reverse_psi_axis=True
        )

    def note_replayed(self, n_evaluated: int) -> None:
        """Set the per-pixel evaluation count after a CUDA-graph replay.

        A captured hypothesis loop does not execute the Python-side bookkeeping, so the
        caller states how many ``(hypothesis, psi)`` pairs each pixel saw.
        """
        for st in self.stats:
            st.hypothesis_count = int(n_evaluated)
            if st.fused:
                st._packed_num_psi = self.n_psi

    @torch.no_grad()
    def finalize_into(
        self, outs: Mapping[str, torch.Tensor], start: int, n_valid: int
    ) -> None:
        """Write the finalized maps into rows ``[start, start + n_valid)`` of ``outs``.

        Keys: ``mip``, ``mean``, ``variance``, ``zscore``, ``best_index``, ``best_psi``.
        """
        for st in self.stats:  # fold any pending in-kernel best into the shared buffers
            st.merge_packed()
        self.block.hypothesis_count = self.stats[0].hypothesis_count
        result = self.block.finalize()
        for key, value in result.items():
            target = outs[key]
            target[start : start + n_valid] = value[:n_valid].to(
                target.device, target.dtype
            )


def fused_stats_supported(spec: SpectrumSpec) -> bool:
    """Whether every batch of a stage with ``spec`` takes the in-kernel path.

    True when the spectrum is a CUDA complex64/complex32 tensor in the kernel-native
    layout, ``(n_psi, n_freq)`` is covered by the register-resident kernel and the
    fused extension compiled; the hypothesis loop is then CUDA-graph capturable.
    """
    return bool(
        spec.device.type == "cuda"
        and spec.dtype in (torch.complex64, torch.complex32)
        and fused_kernel_loader.lean_supported(spec.n_psi, spec.n_freq)
        and fused_kernel_loader.lean_sentinel_packed() is not None
    )


def make_block_stats(spec: SpectrumSpec, *, fused: bool) -> BlockStats:
    """A fresh block accumulator for ``spec`` (fused kernels when ``fused``)."""
    return BlockStats(
        n_sub=spec.n_sub,
        gemm_pixels=spec.gemm_pixels,
        device=spec.device,
        n_psi=spec.n_psi,
        fused=fused,
    )
