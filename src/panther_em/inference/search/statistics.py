"""Per-pixel search statistics tracked online over a batch of hypotheses.

:class:`PixelStats` accumulates the first two moments plus the running best correlation
(and the hypothesis / in-plane angle that produced it) for a batch of image pixels, so a
search never has to materialize the full ``(pixels, hypotheses, psi)`` correlogram.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable


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


# --------------------------------------------------------------------------- #
# Sortable-float <-> packed (float, index) encoding, mirroring the CUDA
# kernel's atomicMax-friendly (float_to_sortable_u32 / pack_val_idx) trick in
# fused_irfft_and_stats_kernel.cuh.
# --------------------------------------------------------------------------- #
def encode_argmax_packed(val: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Pack ``(val, idx)`` into a single sortable ``int64``.

    Parameters
    ----------
    val : torch.Tensor
        float32 tensor of values to pack.
    idx : torch.Tensor
        Integer tensor of indices to pack (must fit in 32 bits), same shape as ``val``.

    Returns
    -------
    torch.Tensor
        int64 tensor, same shape as ``val``. Comparing two packed values via ordinary
        *unsigned* 64-bit comparison recovers the ``(val, idx)`` lexicographic order --
        see the module-level note on :func:`decode_argmax_packed` about why a plain
        ``torch.max`` on this tensor (a signed int64 view of what is conceptually
        unsigned data) is unsafe.
    """
    bits = val.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    sign_set = (bits & 0x80000000) != 0
    mask = torch.where(
        sign_set, torch.full_like(bits, 0xFFFFFFFF), torch.full_like(bits, 0x80000000)
    )
    sortable = (bits ^ mask) & 0xFFFFFFFF
    return (sortable << 32) | (idx.to(torch.int64) & 0xFFFFFFFF)


def decode_argmax_packed(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`encode_argmax_packed`; also decodes the fused kernel's output.

    Parameters
    ----------
    packed : torch.Tensor
        int64 tensor produced by :func:`encode_argmax_packed`, or the fused kernel's raw
        ``argmax_packed`` output.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(val, idx)``, dtypes ``(float32, int64)``, same shape as ``packed``.

    Notes
    -----
    Never reduce (``.max()``/``>``/etc.) a **packed** tensor directly: the encoding's
    order-preserving trick assumes *unsigned* 64-bit comparison, which is what CUDA's
    ``atomicMax`` on ``unsigned long long`` performs, but PyTorch has no native uint64
    dtype -- comparing the raw ``int64`` view as *signed* silently inverts the ordering
    whenever the packed value's top bit is set. Always decode first (elementwise, safe
    regardless of tensor shape), then reduce the decoded float ``val`` in ordinary
    floating-point comparison space.
    """
    idx = (packed & 0xFFFFFFFF).to(torch.int64)
    sortable = ((packed >> 32) & 0xFFFFFFFF).to(torch.int64)

    sign_set = (sortable & 0x80000000) != 0
    mask = torch.where(
        sign_set,
        torch.full_like(sortable, 0x80000000),
        torch.full_like(sortable, 0xFFFFFFFF),
    )
    bits = (sortable ^ mask) & 0xFFFFFFFF
    val = bits.to(torch.int32).view(torch.float32)
    return val, idx


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
    :meth:`update` delegates the "reduce one hypothesis batch's spectrum to
    ``(s1, s2, vmax, amax)``" step to the overridable :meth:`_reduce` method, so a
    subclass can substitute a faster reduction (e.g. a fused CUDA kernel) without
    duplicating the accumulate / branchless-best-update / psi-axis-decode logic in
    :meth:`_accumulate`. See
    :class:`panther_em.inference.search.fused_statistics.FusedPixelStats`.
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
    ):
        """Initialize a new pixel statistics tracker."""
        self.hypothesis_count = 0
        self.num_pixels = num_pixels
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

    @torch.no_grad()
    def clear(self) -> None:
        """Clear all tracked statistics."""
        self.hypothesis_count = 0
        self.corr_sum.zero_()
        self.corr_sum2.zero_()
        self.best_corr.fill_(float("-inf"))
        self.best_hypothesis.fill_(-1)
        self.best_psi_angle.fill_(-1)

    def _reduce(
        self, spectrum: torch.Tensor, num_psi: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reduce one hypothesis batch's spectrum to ``(s1, s2, vmax, amax)``.

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
        corr = torch.fft.irfft(spectrum, n=num_psi, dim=-1, norm="forward")

        return _reduce_stats(  # type: ignore[no-any-return]
            corr.to(self.best_corr.dtype), torch.view_as_real(spectrum)
        )

    def _reduce_raw(
        self, spectrum: torch.Tensor, num_psi: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reduce one hypothesis batch to ``(s1, s2, packed)`` without decoding argmax.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``(s1, s2, packed)``, each shape ``(num_pixels,)``, dtypes
            ``(accumulate_dtype, accumulate_dtype, int64)``.
        """
        s1, s2, vmax, amax = self._reduce(spectrum, num_psi)
        packed = encode_argmax_packed(vmax.to(torch.float32), amax)
        return s1.to(self.corr_sum.dtype), s2.to(self.corr_sum.dtype), packed

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

        self.best_corr = torch.where(improved, vmax_cast, self.best_corr)
        self.best_hypothesis = torch.where(improved, global_hyp, self.best_hypothesis)
        self.best_psi_angle = torch.where(improved, psi, self.best_psi_angle)

    def _build_graphed_accumulate(self, num_psi: int, reverse_psi_axis: bool) -> None:
        """One-time capture of a tiny CUDA graph for the streaming accumulate step.

        Notes
        -----
        Only supports the contiguous-arange ``hyp_offset`` fast path (see
        :meth:`accumulate_batch`'s ``hyp_offset`` parameter).
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
        if not hasattr(self, "_graph"):
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
        contiguous arange starting at ``hyp_offset`` (see :meth:`accumulate_batch`'s
        ``hyp_offset`` parameter) -- callers without that guarantee should use
        :meth:`update` instead. Transparently falls back to :meth:`update` when CUDA
        graph capture isn't supported on this device (checked once per device via
        :func:`_cuda_graph_capture_supported`), so this is always safe to call.
        """
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

    @torch.no_grad()
    def begin_hypothesis_batches(self, n_batches: int) -> None:
        """Preallocate staging buffers for the batch-then-reduce accumulation path."""
        device = self.corr_sum.device
        self._batch_capacity = n_batches
        self._batch_cursor = 0
        self._total_hypotheses_batched = 0
        self._s1_stack = torch.empty(
            (n_batches, self.num_pixels), device=device, dtype=self.corr_sum.dtype
        )
        self._s2_stack = torch.empty_like(self._s1_stack)
        self._packed_stack = torch.empty(
            (n_batches, self.num_pixels), device=device, dtype=torch.int64
        )
        self._hyp_offset_stack: list[int | None] = [None] * n_batches
        self._hyp_idx_stack: torch.Tensor | None = None

    @torch.no_grad()
    def accumulate_batch(
        self,
        spectrum: torch.Tensor,
        hyp_global_idx: torch.Tensor,
        num_psi: int,
        hyp_offset: int | None = None,
    ) -> None:
        """Stage one hypothesis batch's reduced result for deferred reduction.

        Must be called after :meth:`begin_hypothesis_batches` and before
        :meth:`reduce_hypothesis_batches`, once per hypothesis batch (in place of
        :meth:`update`).

        Parameters
        ----------
        spectrum : torch.Tensor
            Same as :meth:`update`'s ``spectrum``.
        hyp_global_idx : torch.Tensor
            Same as :meth:`update`'s ``hyp_global_idx``.
        num_psi : int
            Same as :meth:`update`'s ``num_psi``.
        hyp_offset : int, optional
            Pass this batch's starting global hypothesis index when
            ``hyp_global_idx`` is known to equal
            ``arange(hyp_offset, hyp_offset + hyp_global_idx.numel())`` -- true
            whenever the caller sliced this batch off a plain, unshuffled ``arange``
            (the common case in
            :func:`~panther_em.inference.search.compressed._run_stage`).
            When every staged batch provides this, :meth:`reduce_hypothesis_batches`
            recovers the winning global hypothesis id with a plain addition instead of
            a gather. If any staged batch omits it, all batches fall back to a gather
            against a stored copy of each batch's ``hyp_global_idx``.
        """
        i = self._batch_cursor
        s1, s2, packed = self._reduce_raw(spectrum, num_psi)
        self._s1_stack[i] = s1
        self._s2_stack[i] = s2
        self._packed_stack[i] = packed
        self._hyp_offset_stack[i] = hyp_offset

        if hyp_offset is None:
            m = hyp_global_idx.numel()
            if self._hyp_idx_stack is None:
                self._hyp_idx_stack = torch.zeros(
                    (self._batch_capacity, m),
                    device=hyp_global_idx.device,
                    dtype=torch.long,
                )
            self._hyp_idx_stack[i, :m] = hyp_global_idx

        self._total_hypotheses_batched += hyp_global_idx.numel() * num_psi
        self._batch_cursor += 1

    @torch.no_grad()
    def reduce_hypothesis_batches(
        self, num_psi: int, reverse_psi_axis: bool = True
    ) -> None:
        """Decode and reduce all staged hypothesis batches, updating running stats once.

        Equivalent to calling :meth:`update` once per staged batch, but the argmax
        decode, best-of compare, and running-sum update each run once over the whole
        ``(n_batches, num_pixels)`` stack instead of once per batch. No-op if no
        batches were staged since the last :meth:`begin_hypothesis_batches` call.

        Parameters
        ----------
        num_psi : int
            Same ``num_psi`` used for every staged :meth:`accumulate_batch` call.
        reverse_psi_axis : bool, optional
            Same as :meth:`update`'s ``reverse_psi_axis``.
        """
        n = self._batch_cursor
        if n == 0:
            return

        s1_total = self._s1_stack[:n].sum(dim=0)
        s2_total = self._s2_stack[:n].sum(dim=0)

        # Decode the WHOLE stack in one shot -- and only ever compare/reduce the
        # DECODED float vmax, never the raw packed int64 (see decode_argmax_packed's
        # docstring: the packed encoding assumes unsigned comparison, which a plain
        # torch.max on this signed-int64-typed tensor would get backwards).
        vmax_stack, amax_local_stack = decode_argmax_packed(self._packed_stack[:n])

        best_vmax, best_batch = vmax_stack.max(dim=0)
        best_amax_local = amax_local_stack.gather(0, best_batch.unsqueeze(0)).squeeze(0)
        n_local = torch.div(best_amax_local, num_psi, rounding_mode="floor")

        offsets = self._hyp_offset_stack[:n]
        if all(offset is not None for offset in offsets):
            offset_stack = torch.tensor(
                offsets, device=best_batch.device, dtype=torch.long
            )
            global_hyp = offset_stack[best_batch] + n_local
        else:
            assert self._hyp_idx_stack is not None
            global_hyp = self._hyp_idx_stack[best_batch, n_local]

        if reverse_psi_axis:
            psi = (num_psi - best_amax_local % num_psi) % num_psi
        else:
            psi = best_amax_local % num_psi

        vmax_cast = best_vmax.to(self.best_corr.dtype)
        self.corr_sum += s1_total.to(self.best_corr.dtype)
        self.corr_sum2 += s2_total.to(self.best_corr.dtype)
        self.hypothesis_count += self._total_hypotheses_batched

        improved = vmax_cast > self.best_corr
        self.best_corr = torch.where(improved, vmax_cast, self.best_corr)
        self.best_hypothesis = torch.where(improved, global_hyp, self.best_hypothesis)
        self.best_psi_angle = torch.where(improved, psi, self.best_psi_angle)

        self._batch_cursor = 0

    @torch.no_grad()
    def finalize(self) -> dict[str, torch.Tensor]:
        """Finalize tracked statistics into summary dictionary.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing the final tracked statistics. Tensors exists on device
            used to initialize the PixelStats object.
        """
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


# ---------------------------------------------------------------------------
# Error-aware partitioning (multi-precision follow-up) -- NOT YET IMPLEMENTED
# ---------------------------------------------------------------------------
#
# Between incremental stages the reconstructed correlations are partitioned,
# using the reconstruction-accuracy bounds implied by the retained singular
# values, into three sets:
#
#   * accepted   -- score far above expectation; record and stop refining.
#   * rejected   -- score far below expectation; record and stop refining.
#   * follow-up  -- ambiguous; carry forward to the next (higher-rank) stage.
#
# The follow-up set becomes the pixel mask (and joint hypothesis mask) handed to the
# next stage. This is the home for that logic; it is intentionally unimplemented for
# now.
