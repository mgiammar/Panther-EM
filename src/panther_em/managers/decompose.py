"""Program ``decompose``: the template volume's polar projection decomposition."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import Field

from panther_em.coordinates import (
    CoordinateTransform,
    OffsetPolarTransform,
    SpiralPolarTransform,
    WarpConfig,
)
from panther_em.decomposition import DecompositionResult, PolarProjectionDecomposer
from panther_em.filters import WhiteningFilter
from panther_em.managers._common import Model, ProgramConfig, read_volume, save_output
from panther_em.orientations import healpix_orientations

TRANSFORMS: dict[str, type[OffsetPolarTransform] | type[SpiralPolarTransform]] = {
    "offset": OffsetPolarTransform,
    "spiral": SpiralPolarTransform,
}


class PolarGrid(Model):
    """The polar grid the projections are decomposed on.

    Parameters
    ----------
    kind : {"spiral", "offset"}
        :class:`~panther_em.coordinates.SpiralPolarTransform` (default) or
        :class:`~panther_em.coordinates.OffsetPolarTransform`.
    num_angle, num_radius : int
        Grid size; ``num_radius`` defaults to the transform's own choice.
    options : dict
        Further keywords of the transform's ``from_image`` (e.g. a spiral's ``c``).
    """

    kind: Literal["offset", "spiral"] = "spiral"
    num_angle: int = Field(ge=2)
    num_radius: int | None = Field(default=None, ge=1)
    options: dict[str, Any] = Field(default_factory=dict)

    def build(self, box: tuple[int, int]) -> CoordinateTransform:
        """The transform on the ``box`` template grid."""
        return TRANSFORMS[self.kind].from_image(
            image_shape=box,
            num_angle=self.num_angle,
            num_radius=self.num_radius,
            **self.options,
        )


class DecomposeConfig(ProgramConfig):
    """Configuration of :func:`run`.

    Parameters
    ----------
    volume : Path
        The cubic template volume (MRC); its ``(d, d)`` is the template box.
    whitening : Path, optional
        The dataset's whitening filter; the projections are filtered by its
        ``W_avg * band`` (needed by the search). Unfiltered when omitted.
    output : Path
        The decomposition's HDF5 file.
    polar_grid : PolarGrid
        The polar grid.
    healpix_nside : int
        Orientations: the HEALPix pixel centers at this ``nside`` (``12 nside^2``),
        which a search's level and local stages need.
    warp : dict
        :class:`~panther_em.coordinates.WarpConfig` fields; default its defaults.
    k_max, eig_max : int, optional
        Stored angular blocks and components (default all).
    normalize_projections : bool
        Normalize each projection to zero mean, unit variance. Default True.
    svd_method : {"dense", "low_rank"}
        Default ``"dense"``; ``"low_rank"`` is faster when ``eig_max`` is small.
    projection_batch_size, block_batch_size : int
        Projections and angular blocks per device batch.
    device : str
        Compute device. Default ``"cpu"``.
    """

    program: ClassVar[str] = "decompose"

    volume: Path
    whitening: Path | None = None
    output: Path
    polar_grid: PolarGrid
    healpix_nside: int = Field(ge=1)
    warp: dict[str, Any] = Field(default_factory=dict)
    k_max: int | None = Field(default=None, ge=1)
    eig_max: int | None = Field(default=None, ge=1)
    normalize_projections: bool = True
    svd_method: Literal["dense", "low_rank"] = "dense"
    projection_batch_size: int = Field(default=128, ge=1)
    block_batch_size: int = Field(default=8, ge=1)
    device: str = "cpu"


def run(config: DecomposeConfig) -> DecompositionResult:
    """Decompose the volume and write the result to ``config.output``."""
    volume = read_volume(config.volume)
    box = (int(volume.shape[-2]), int(volume.shape[-1]))
    filters = None
    if config.whitening is not None:
        filters = WhiteningFilter.load(config.whitening).template_filter(box)[None]
    phi, theta = healpix_orientations(config.healpix_nside)
    result = PolarProjectionDecomposer(
        volume,
        phi,
        theta,
        coordinate_transform=config.polar_grid.build(box),
        fourier_filters=filters,
        warp=WarpConfig(**config.warp),
        device=config.device,
    ).do_decomposition(
        k_max=config.k_max,
        eig_max=config.eig_max,
        projection_batch_size=config.projection_batch_size,
        block_batch_size=config.block_batch_size,
        normalize_projections=config.normalize_projections,
        svd_method=config.svd_method,
    )
    save_output(result.save, config.output, config)
    return result
