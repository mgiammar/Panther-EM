"""A ``Template`` bundle of kernels, weights, and features for a search.

Everything here is built once per decomposition and reused across stages and images:

* :class:`KernelBank` -- lazily builds, caches and serves the Cartesian feature kernels
  ``V[k, m]`` (spatial, spectral or planar form) under a byte budget.
* :class:`WeightSource` -- holds ``U * S`` device-resident once and serves rectangular
  ``(N, num_k, num_m)`` blocks.
* :class:`PreparedWeightCache` -- caches the engine-specific B operands per
  ``(k-run, hypothesis batch, precision)``.
* :class:`Template` -- the bundle a
  :class:`~panther_em.inference.search.searcher.Searcher` is built from.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np
import torch

from panther_em.decomposition.result import (
    DecompositionResult,
    SparseDecompositionResult,
)
from panther_em.inference.projection_reconstruction import ProjectionReconstructor
from panther_em.inference.search.hypotheses import HypothesisSpace
from panther_em.inference.search.types import KernelConfig, Rectangle

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Sequence
    from pathlib import Path

    from panther_em.inference.search.contraction import ContractionEngine
    from panther_em.inference.search.hypotheses import HypBatch
    from panther_em.inference.search.tiling import FeatureLayout, KRun

Cell = tuple[int, int]


def _as_cells(cells: torch.Tensor | Iterable[Sequence[int]]) -> list[Cell]:
    """Normalise ``(n, 2)`` tensors / iterables of pairs to Python ``(k, m)`` tuples."""
    if isinstance(cells, torch.Tensor):
        if cells.dim() != 2 or cells.shape[1] != 2:
            raise ValueError(f"cells must have shape (n, 2), got {tuple(cells.shape)}")
        return [(int(k), int(m)) for k, m in cells.tolist()]
    return [(int(c[0]), int(c[1])) for c in cells]


class _ByteLRU:
    """Insertion-ordered cache with a byte budget and pinning."""

    def __init__(self, budget: int | None) -> None:
        self.budget = budget  # None = unbounded
        self._items: OrderedDict[Hashable, torch.Tensor] = OrderedDict()
        self._pins: dict[Hashable, set[Hashable]] = {}
        self.nbytes = 0

    def __contains__(self, key: Hashable) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return len(self._items)

    def get(self, key: Hashable) -> torch.Tensor | None:
        t = self._items.get(key)
        if t is None:
            return None
        self._items.move_to_end(key)
        return t

    def _pinned(self, key: Hashable) -> bool:
        return any(key in keys for keys in self._pins.values())

    def put(
        self, key: Hashable, value: torch.Tensor
    ) -> list[tuple[Hashable, torch.Tensor]]:
        """Insert ``value``; returns the (key, tensor) pairs evicted for the budget."""
        size = value.numel() * value.element_size()
        if key in self._items:
            old = self._items[key]
            self.nbytes -= old.numel() * old.element_size()
        self._items[key] = value
        self._items.move_to_end(key)
        self.nbytes += size
        return self._evict()

    def _evict(self) -> list[tuple[Hashable, torch.Tensor]]:
        evicted: list[tuple[Hashable, torch.Tensor]] = []
        if self.budget is None:
            return evicted
        for key in list(self._items.keys()):
            if self.nbytes <= self.budget:
                break
            if self._pinned(key) or key is next(reversed(self._items)):
                continue  # never evict pinned entries or the entry just inserted
            t = self._items.pop(key)
            self.nbytes -= t.numel() * t.element_size()
            evicted.append((key, t))
        return evicted

    def pop(self, key: Hashable) -> torch.Tensor | None:
        t = self._items.pop(key, None)
        if t is not None:
            self.nbytes -= t.numel() * t.element_size()
        return t

    def pin(self, tag: Hashable, keys: Iterable[Hashable]) -> None:
        self._pins.setdefault(tag, set()).update(keys)

    def unpin(self, tag: Hashable) -> None:
        self._pins.pop(tag, None)

    def clear(self) -> None:
        self._items.clear()
        self._pins.clear()
        self.nbytes = 0


class KernelBank:
    """Lazily built, byte-budgeted cache of Cartesian feature kernels.

    Parameters
    ----------
    reconstructor : ProjectionReconstructor
        Supplies the SVD tensors and the polar-to-Cartesian transform.
    config : KernelConfig, optional
        Warp settings for every kernel built by this bank.
    device : torch.device or str, optional
        Where served kernels live. Defaults to ``reconstructor.device``.
    device_budget : int, optional
        Bytes of spatial kernels kept on ``device`` (LRU). ``None`` = unbounded.
    host_budget : int, optional
        Bytes of evicted spatial kernels kept on the CPU (pinned when CUDA is the
        device) before they must be rebuilt. Defaults to ``0`` (no host tier).
    spectral_budget : int, optional
        Bytes of ``conj(fft2(pad(kernel)))`` kept per FFT shape. ``None`` =
        unbounded; ``0`` disables spectral caching.
    build_chunk : int, optional
        Maximum kernels built per polar-to-Cartesian call. Defaults to 64.

    Notes
    -----
    Kernels are built grouped by angular frequency ``k``: the analytic angular phase
    is shared across the group and the radial vectors ``Vh`` are gathered with one
    host-to-device transfer per chunk, instead of one per cell.
    """

    def __init__(
        self,
        reconstructor: ProjectionReconstructor,
        config: KernelConfig | None = None,
        *,
        device: torch.device | str | None = None,
        device_budget: int | None = None,
        host_budget: int = 0,
        spectral_budget: int | None = None,
        build_chunk: int = 64,
    ) -> None:
        if reconstructor.result.is_complex_projection:
            raise NotImplementedError("Complex-valued projections not yet supported")
        self.reconstructor = reconstructor
        self.config = config or KernelConfig()
        self.device = (
            torch.device(device) if device is not None else reconstructor.device
        )
        self.template_shape: tuple[int, int] = reconstructor.image_shape
        self.build_chunk = int(build_chunk)
        self._device_cache = _ByteLRU(device_budget)
        self._host_cache = _ByteLRU(host_budget if host_budget > 0 else 0)
        self._host_enabled = host_budget > 0
        self._spectral_cache = _ByteLRU(spectral_budget)
        self._spectral_enabled = spectral_budget is None or spectral_budget > 0
        self._planar_cache: dict[tuple[Cell, torch.dtype], torch.Tensor] = {}

    # -- construction --------------------------------------------------------

    def _build_group(self, k: int, ms: Sequence[int]) -> torch.Tensor:
        """Kernels ``(k, m)`` for every ``m`` in ``ms`` -> ``(len(ms), kH, kW)``."""
        recon = self.reconstructor
        indices = np.column_stack([np.full(len(ms), k), np.asarray(ms)])
        _, _, vh = recon.result.get_svd_tensors(indices, recon.device)  # (M, R)
        angular = recon._get_angular_phase_component(k, 0.0)  # (A,)
        polar = angular[None, :, None] * vh[:, None, :]  # (M, A, R) complex64
        cfg = self.config
        cart = recon._coordinate_transform.to_cartesian(
            polar,
            preserve_energy=cfg.preserve_energy,
            wrap_angular_axis=cfg.wrap_angular_axis,
            order=cfg.order,
            mode=cfg.mode,
            cval=cfg.cval,
        )
        out = torch.as_tensor(cart, device=self.device).to(torch.complex64)
        return out

    def _build(self, missing: Sequence[Cell]) -> dict[Cell, torch.Tensor]:
        built: dict[Cell, torch.Tensor] = {}
        by_k: dict[int, list[int]] = {}
        for k, m in missing:
            by_k.setdefault(k, []).append(m)
        for k in sorted(by_k):
            ms = sorted(set(by_k[k]))
            for start in range(0, len(ms), self.build_chunk):
                chunk = ms[start : start + self.build_chunk]
                kernels = self._build_group(k, chunk)
                for m, kern in zip(chunk, kernels, strict=True):
                    built[(k, m)] = kern.contiguous()
        return built

    def _lookup(self, cell: Cell) -> torch.Tensor | None:
        t = self._device_cache.get(cell)
        if t is not None:
            return t
        if self._host_enabled:
            h = self._host_cache.pop(cell)
            if h is not None:
                t = h.to(self.device, non_blocking=True)
                self._store(cell, t)
                return t
        return None

    def _store(self, cell: Cell, kernel: torch.Tensor) -> None:
        """Insert on the device tier, demoting whatever that evicts to the host tier."""
        evicted = self._device_cache.put(cell, kernel)
        if self._host_enabled:
            for key, t in evicted:
                host = t.to("cpu", non_blocking=False)
                if self.device.type == "cuda":
                    host = host.pin_memory()
                self._host_cache.put(key, host)

    # -- public API ----------------------------------------------------------

    def spatial(self, cells: torch.Tensor | Iterable[Sequence[int]]) -> torch.Tensor:
        """``(n, kH, kW)`` complex64 kernels for ``(k, m)`` cells, in request order."""
        keys = _as_cells(cells)
        found: dict[Cell, torch.Tensor] = {}
        missing: list[Cell] = []
        for cell in dict.fromkeys(keys):
            t = self._lookup(cell)
            if t is None:
                missing.append(cell)
            else:
                found[cell] = t
        if missing:
            for cell, kern in self._build(missing).items():
                self._store(cell, kern)
                found[cell] = kern
        if not keys:
            return torch.empty(
                (0, *self.template_shape), dtype=torch.complex64, device=self.device
            )
        return torch.stack([found[c] for c in keys], dim=0)

    def spectral(
        self,
        cells: torch.Tensor | Iterable[Sequence[int]],
        fft_shape: tuple[int, int],
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``conj(fft2(kernel, s=fft_shape))`` per cell, ``(n, Fh, Fw)`` complex64.

        Cached per ``(cell, fft_shape)`` under ``spectral_budget``; written into
        ``out`` when given (and then not cached).
        """
        keys = _as_cells(cells)
        fft_shape = (int(fft_shape[0]), int(fft_shape[1]))
        if out is not None and tuple(out.shape) != (len(keys), *fft_shape):
            raise ValueError(f"out must have shape {(len(keys), *fft_shape)}")
        result = (
            out
            if out is not None
            else torch.empty(
                (len(keys), *fft_shape), dtype=torch.complex64, device=self.device
            )
        )
        todo: list[int] = []
        for i, cell in enumerate(keys):
            cached = (
                self._spectral_cache.get((cell, fft_shape))
                if self._spectral_enabled
                else None
            )
            if cached is None:
                todo.append(i)
            else:
                result[i] = cached
        if todo:
            spatial = self.spatial([keys[i] for i in todo])
            spec = torch.fft.fft2(spatial, s=fft_shape).conj()
            for j, i in enumerate(todo):
                result[i] = spec[j]
                if self._spectral_enabled and out is None:
                    self._spectral_cache.put((keys[i], fft_shape), spec[j].contiguous())
        return result

    def planar(
        self,
        cells: torch.Tensor | Iterable[Sequence[int]],
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """``(2, n, kH * kW)`` real rows ``[Re K ; -Im K]`` for the 1x1 inner product.

        ``sum_ij I[i, j] * conj(K[i, j]) = I . Re K  -  i (I . Im K)``, so the features
        of a centred box are ``planar @ I.flatten()`` with real part row 0 and imaginary
        part ``-``row 1.
        """
        spatial = self.spatial(cells).reshape(
            -1, self.template_shape[0] * self.template_shape[1]
        )
        return torch.stack((spatial.real, -spatial.imag), dim=0).to(dtype)

    def prefetch(
        self,
        cells: torch.Tensor | Iterable[Sequence[int]],
        stream: torch.cuda.Stream | None = None,
    ) -> torch.cuda.Event | None:
        """Build/load ``cells`` (on ``stream`` when given); returns its event."""
        if stream is None or self.device.type != "cuda":
            self.spatial(cells)
            return None
        with torch.cuda.stream(stream):
            self.spatial(cells)
            event = torch.cuda.Event()  # type: ignore[no-untyped-call]
            event.record(stream)  # type: ignore[no-untyped-call]
        return event

    def bytes(self) -> tuple[int, int]:
        """``(device_bytes, host_bytes)`` of cached spatial kernels."""
        return self._device_cache.nbytes, self._host_cache.nbytes

    def clear(self) -> None:
        """Drop every cached kernel."""
        self._device_cache.clear()
        self._host_cache.clear()
        self._spectral_cache.clear()
        self._planar_cache.clear()


class WeightSource:
    """``U * S`` weights, device-resident once, served as rectangular blocks.

    Parameters
    ----------
    result : DecompositionResult
        Dense or sparse decomposition (sparse pairs absent from the stored set are
        zeros, as in :meth:`~DecompositionResult.get_component`).
    device : torch.device or str
        Where the weights live.
    cells_union : Rectangle or sequence of Rectangle, optional
        Bounding box of every rectangle any stage will contract; only that box of
        ``(k, m)`` space is resident. Defaults to the whole stored grid.
    """

    def __init__(
        self,
        result: DecompositionResult,
        device: torch.device | str,
        cells_union: Rectangle | Sequence[Rectangle] | None = None,
    ) -> None:
        if result.is_complex_projection:
            raise NotImplementedError("Complex-valued projections not yet supported")
        self.result = result
        self.device = torch.device(device)
        self.n_filters = int(result.num_fourier_filters)
        self.n_orientations = int(result.num_orientations)
        k_blocks, e_max = int(result.k_max), int(result.eig_max)
        if cells_union is None:
            k0, m0, k1, m1 = 0, 0, k_blocks, e_max
        else:
            rects: list[Rectangle] = (
                [cells_union]  # type: ignore[list-item]
                if isinstance(cells_union[0], int)
                else list(cells_union)  # type: ignore[arg-type]
            )
            k0 = min(r[0] for r in rects)
            m0 = min(r[1] for r in rects)
            k1 = max(r[0] + r[2] for r in rects)
            m1 = max(r[1] + r[3] for r in rects)
            if k1 > k_blocks or m1 > e_max:
                raise ValueError(
                    f"cells_union {(k0, m0, k1, m1)} exceeds the stored "
                    f"(k_max={k_blocks}, "
                    f"eig_max={e_max}) grid"
                )
        self.k_start, self.m_start, self.k_stop, self.m_stop = k0, m0, k1, m1

        ks, ms = np.meshgrid(np.arange(k0, k1), np.arange(m0, m1), indexing="ij")
        u, s, _ = result.get_component(ks.reshape(-1), ms.reshape(-1), return_vh=False)
        assert u is not None and s is not None
        us = (u * s[None, None, :]).reshape(
            self.n_filters * self.n_orientations, k1 - k0, m1 - m0
        )
        self.us: torch.Tensor = torch.as_tensor(us, dtype=torch.complex64).to(
            self.device
        )

    @property
    def n_templates(self) -> int:
        """``FF * O``."""
        return self.n_filters * self.n_orientations

    @property
    def nbytes(self) -> int:
        """Resident bytes."""
        return self.us.numel() * self.us.element_size()

    def block(
        self, rect: Rectangle, template_rows: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``(N_sel, k_extent, m_extent)`` complex64 block of ``U * S`` for ``rect``."""
        k0, m0, k_ext, m_ext = (int(v) for v in rect)
        if (
            k0 < self.k_start
            or m0 < self.m_start
            or k0 + k_ext > self.k_stop
            or m0 + m_ext > self.m_stop
        ):
            raise ValueError(f"rectangle {rect} lies outside the resident box")
        view = self.us[
            :,
            k0 - self.k_start : k0 - self.k_start + k_ext,
            m0 - self.m_start : m0 - self.m_start + m_ext,
        ]
        if template_rows is None:
            return view
        return view.index_select(0, template_rows.to(self.device))

    def layout_weights(
        self, layout: FeatureLayout, template_rows: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``(N_sel, r)`` weights in ``layout``'s compacted slot order."""
        blocks = [
            self.block(rect, template_rows).reshape(-1, rect[2] * rect[3])
            for rect in layout.rectangles
        ]
        return torch.cat(blocks, dim=1)


class PreparedWeightCache:
    """Engine-specific B operands per ``(k-run, hypothesis batch, precision)``.

    Entries are shared across defocus planes (weights do not depend on the image),
    across stages that reuse a run, and across images. Entries referenced by a
    captured CUDA graph are pinned via :meth:`pin` and never evicted.
    """

    def __init__(self, source: WeightSource, byte_budget: int | None = None) -> None:
        self.source = source
        self._cache = _ByteLRU(byte_budget)

    @staticmethod
    def key(
        layout: FeatureLayout,
        run: KRun,
        batch: HypBatch,
        engine: ContractionEngine,
        rows_key: Hashable,
    ) -> tuple:
        """Cache key for one prepared operand."""
        parts = tuple(
            (layout.rectangles[p.rect], p.m_start, p.m_stop) for p in run.parts
        )
        return (
            parts,
            run.k_start,
            run.k_stop,
            engine.precision,
            rows_key,
            batch.j0,
            batch.j1,
        )

    def get(
        self,
        layout: FeatureLayout,
        run: KRun,
        batch: HypBatch,
        engine: ContractionEngine,
        template_rows: torch.Tensor | None = None,
        rows_key: Hashable = None,
    ) -> torch.Tensor:
        """Concatenated B operand for ``run`` and hypotheses ``batch.j0:batch.j1``."""
        key = self.key(layout, run, batch, engine, rows_key)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        rows = (
            torch.arange(batch.j0, batch.j1, device=self.source.device)
            if template_rows is None
            else template_rows.to(self.source.device)[batch.j0 : batch.j1]
        )
        parts = []
        for part in run.parts:
            sub = (
                run.k_start,
                part.m_start,
                run.k_stop - run.k_start,
                part.m_stop - part.m_start,
            )
            parts.append(engine.prepare_weights(self.source.block(sub, rows)))
        prepared = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
        self._cache.put(key, prepared)
        return prepared

    def pin(self, tag: Hashable, keys: Iterable[Hashable]) -> None:
        """Protect ``keys`` from eviction while ``tag`` (e.g. a graph) is alive."""
        self._cache.pin(tag, keys)

    def unpin(self, tag: Hashable) -> None:
        """Release a pin."""
        self._cache.unpin(tag)

    @property
    def nbytes(self) -> int:
        """Cached bytes."""
        return self._cache.nbytes

    def clear(self) -> None:
        """Drop every entry (including pinned ones; only call with no live graphs)."""
        self._cache.clear()


class Template:
    """A decomposition plus everything derived from it that a search reuses.

    Parameters
    ----------
    result : DecompositionResult
        The (dense or sparse) decomposition, with its coordinate transform.
    device : torch.device or str
        Compute device for kernels and weights.
    kernel : KernelConfig, optional
        Polar-to-Cartesian warp settings.
    kernel_budget, kernel_host_budget, spectral_budget : int, optional
        Byte budgets forwarded to :class:`KernelBank`.
    weights_cells : Rectangle or sequence of Rectangle, optional
        Bounding box of the feature space any stage will use; restricts the
        resident ``U * S`` (see :class:`WeightSource`).
    """

    def __init__(
        self,
        result: DecompositionResult,
        *,
        device: torch.device | str,
        kernel: KernelConfig | None = None,
        kernel_budget: int | None = None,
        kernel_host_budget: int = 0,
        spectral_budget: int | None = None,
        weights_cells: Rectangle | Sequence[Rectangle] | None = None,
    ) -> None:
        self.result = result
        self.device = torch.device(device)
        self.reconstructor = ProjectionReconstructor(result, device=self.device)
        self.template_shape: tuple[int, int] = self.reconstructor.image_shape
        self.kernel_config = kernel or KernelConfig()
        self.kernels = KernelBank(
            self.reconstructor,
            self.kernel_config,
            device=self.device,
            device_budget=kernel_budget,
            host_budget=kernel_host_budget,
            spectral_budget=spectral_budget,
        )
        self._weights_cells = weights_cells
        self._weights: WeightSource | None = None
        self.space = HypothesisSpace.from_result(result)

    @property
    def weights(self) -> WeightSource:
        """Device-resident ``U * S`` (built on first access)."""
        if self._weights is None:
            self._weights = WeightSource(self.result, self.device, self._weights_cells)
        return self._weights

    @property
    def k_max(self) -> int:
        """Stored angular-frequency blocks."""
        return int(self.result.k_max)

    @property
    def eig_max(self) -> int:
        """Stored eigenvectors per frequency."""
        return int(self.result.eig_max)

    @classmethod
    def from_result(
        cls, result: DecompositionResult, device: torch.device | str, **kwargs: object
    ) -> Template:
        """Alias of the constructor for symmetry with :meth:`from_hdf5`."""
        return cls(result, device=device, **kwargs)  # type: ignore[arg-type]

    @classmethod
    def from_hdf5(
        cls, path: str | Path, device: torch.device | str, **kwargs: object
    ) -> Template:
        """Load a dense or sparse decomposition and wrap it."""
        try:
            result: DecompositionResult = DecompositionResult.load(path)
        except ValueError:
            result = SparseDecompositionResult.load(path)
        return cls(result, device=device, **kwargs)  # type: ignore[arg-type]

    def release(self) -> None:
        """Drop cached kernels and the resident weights."""
        self.kernels.clear()
        self._weights = None
