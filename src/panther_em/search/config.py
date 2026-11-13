"""Execution parameters for a search plus tuning knobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import torch

    from panther_em._precision import Precision
    from panther_em.search.correlation.contraction import BatchShape
    from panther_em.search.featurization.direct_featurization import (
        ContractionName,
    )

MIB = 1024**2


@dataclass(frozen=True)
class Tuning:
    """Overrides of the planner's and the engines' own choices.

    Attributes
    ----------
    batch : BatchShape, optional
        Explicit ``(pixels, hypotheses)`` per contraction; ``None`` lets the planner
        pick from the L2-residency rule.
    block_pixels : int, optional
        Explicit store block size (a multiple of the GEMM pixel batch).
    l2_target : int, optional
        Bytes of spectrum per contraction to keep L2-resident (the hypothesis batch
        follows from it). Will default to a quarter of the device's L2.
    block_bytes_max : int
        Byte budget of one feature-store block (256 MiB).
    use_cuda_graph, use_fused_kernel : bool
        Allow the graphed hypothesis loop / the fused statistics kernel.
    plane_major : bool
        Search a multi-plane stage one defocus plane at a time. When True, each pass
        holds only one plane features and the per-pixel maxima, means and variances are
        merged across passes. When False, the passes hold all planes' features and the
        statistics are computed in one pass.
    cell_chunk : int
        Kernel spectra per chunk of the FFT engine.
    image_batch : int
        Positions per chunk of the direct engine.
    prepared_budget : int, optional
        Byte budget of the prepared-weight cache (``None`` = unbounded).
    ledger_device : device, optional
        Where the dense per-pixel state lives; defaults to the compute device.
    local_bytes : int
        Working-memory budget of one chunk of a local stage's tiles (512 MiB).
    """

    batch: BatchShape | None = None
    block_pixels: int | None = None
    l2_target: int | None = None
    block_bytes_max: int = 256 * MIB
    use_cuda_graph: bool = True
    use_fused_kernel: bool = True
    plane_major: bool = True
    cell_chunk: int = 32
    image_batch: int = 1024
    prepared_budget: int | None = None
    ledger_device: torch.device | str | None = None
    local_bytes: int = 512 * MIB


@dataclass(frozen=True)
class ExecutionConfig:
    """How a search runs (not what it searches: that is the stages').

    Attributes
    ----------
    featurize_mode : {"auto", "fft", "direct"}
        Featurization engine: overlap-save FFT correlation of the selection's bounding
        box, or box gather + contraction per position. ``"auto"`` applies
        :func:`~panther_em.search.featurization.choose_mode` to each
        stage's selection.
    direct_contraction : {"adjoint", "planar"}
        How the direct engine turns boxes into features: ``"adjoint"`` scatters each
        box onto the polar grid through the transposed warp (no Cartesian kernels;
        falls back to ``"planar"`` where the transform is not mirrored), ``"planar"``
        the ``[Re K; -Im K]`` kernel GEMM.
    direct_precision : {"fp16", "tf32", "fp32"}
        Operand precision of the direct engine, independent of the stage's (store)
        precision: fp16 is 5e-4 relative on the adjoint's features, fp32 exact to
        1e-6.
    device_store_fraction : float
        A stage whose store would exceed this fraction of the free device memory runs
        in *bands*: row-contiguous chunks of its pixels whose store fits, each
        featurized and reduced in turn; the maps are concatenated before the decision.
    show_progress : bool
        Print per-stage progress bars.
    trace : bool
        Synchronize the device at every session mark (``SearchSession.marks``) so
        the marks' wall-clock deltas are per-phase device times.
    tuning : Tuning
        Overrides of the planner's and the engines' choices.
    """

    featurize_mode: Literal["auto", "fft", "direct"] = "auto"
    direct_contraction: ContractionName = "adjoint"
    direct_precision: Precision = "fp16"
    device_store_fraction: float = 0.5
    show_progress: bool = False
    trace: bool = False
    tuning: Tuning = field(default_factory=Tuning)

    def __post_init__(self) -> None:
        """Validate the choices."""
        if self.featurize_mode not in ("auto", "fft", "direct"):
            raise ValueError(f"unknown featurize_mode {self.featurize_mode!r}")
        if not 0.0 < self.device_store_fraction <= 1.0:
            raise ValueError("device_store_fraction must lie in (0, 1]")
        if not isinstance(self.tuning, Tuning):
            raise ValueError("tuning must be a Tuning")
