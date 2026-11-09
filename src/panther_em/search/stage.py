"""One search stage: what it searches and the decision that ends it.

Feature rectangles, precision, ``n_psi``, orientation sampling, CTF planes, an
optional local neighborhood (seeded by the previous stage) or self-seeding
(:class:`~.seeding.Seeding`), and the decision.

A :class:`Stage` is a frozen value: build it with :meth:`Stage.create` (which
coerces loose inputs) and read its derived state (``k_stop``, ``num_features``,
``layout()``, ``hypothesis_ids()``). ``stage_to_json`` / ``stage_from_json`` give
the HDF5 attribute form used by :class:`~.results.SearchResult`; a saved decision
records its threshold test's calibration and table summary but is restored without the
test (the error table is a separate file).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, cast

import torch

from panther_em.decomposition.rectangles import as_rectangle
from panther_em.orientations import healpix_level_ids
from panther_em.search.correlation.contraction import Precision
from panther_em.search.correlation.layout import (
    FeatureLayout,
    rectangles_disjoint,
)
from panther_em.search.decision import Decision
from panther_em.search.hypotheses import template_rows_for
from panther_em.search.seeding import Seeding

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy as np

    from panther_em.decomposition.rectangles import Rectangle
    from panther_em.search.hypotheses import HypothesisSpace


@dataclass(frozen=True, eq=False)
class Stage:
    """One stage of the search: which features, over which hypotheses, decided how.

    Build with :meth:`create`, which accepts loose inputs; the constructor only
    validates already-normalised fields.

    Parameters
    ----------
    rectangles : tuple of Rectangle
        Pairwise cell-disjoint ``(k_start, m_start, k_extent, m_extent)`` blocks of
        feature space contracted in this stage.
    precision : Precision
        Contraction precision.
    n_psi : int
        In-plane samples recovered by the inverse FFT; ``k_stop <= n_psi // 2 + 1``.
    orientations : int or torch.Tensor, optional
        ``None`` searches every orientation; an int ``L`` keeps one orientation per
        HEALPix pixel ``L`` levels coarser than the template's sampling (the
        template's Euler angles must be HEALPix pixel centers, see
        :func:`~panther_em.orientations.healpix_orientations`); a 1-D int64 tensor
        lists explicit orientation ids. Every Fourier filter is searched either way.
    planes : tuple of int, optional
        CTF (defocus) planes searched in this stage, sorted; ``None`` = all.
    decision : Decision, optional
        Applied to the stage's pixels after the search; ``None`` leaves them
        AMBIGUOUS (sweeps and benchmarks).
    local_rings : int, optional
        Search each pixel in only the local neighborhood of orientations from the
        previous search stage.
    seeding : Seeding, optional
        Seed the stage itself: a seed pass over coarse cell templates names each
        pixel's cells, the exact local search covers their neighborhoods (see
        :mod:`.seeding`). Needs a HEALPix template; excludes ``local_rings`` and
        explicit orientation ids.
    name : str
        Free-form label.
    """

    rectangles: tuple[Rectangle, ...]
    precision: Precision = Precision.FP32
    n_psi: int = 256
    orientations: int | torch.Tensor | None = None
    planes: tuple[int, ...] | None = None
    decision: Decision | None = None
    local_rings: int | None = None
    seeding: Seeding | None = None
    name: str = ""

    @classmethod
    def create(
        cls,
        rectangles: Rectangle | Sequence[Rectangle],
        *,
        precision: Precision | str = Precision.FP32,
        n_psi: int = 256,
        orientations: int | Sequence[int] | torch.Tensor | None = None,
        planes: Sequence[int] | None = None,
        decision: Decision | None = None,
        local_rings: int | None = None,
        seeding: Seeding | Mapping[str, object] | None = None,
        name: str = "",
    ) -> Stage:
        """Build a stage from loose inputs.

        One rectangle or many, precision by name, planes in any order, orientation
        ids as any int sequence or tensor, the seeding as a :class:`Seeding` or its
        fields in a mapping.
        """
        if seeding is not None and not isinstance(seeding, Seeding):
            seeding = Seeding(**{str(k): v for k, v in seeding.items()})  # type: ignore[arg-type]
        rects: tuple[Rectangle, ...]
        if len(rectangles) > 0 and isinstance(rectangles[0], int):
            rects = (as_rectangle(cast("Sequence[int]", rectangles)),)
        else:
            rects = tuple(as_rectangle(cast("Sequence[int]", r)) for r in rectangles)
        if orientations is not None and not isinstance(orientations, int):
            orientations = (
                torch.as_tensor(orientations, dtype=torch.long).reshape(-1).cpu()
            )
        return cls(
            rectangles=rects,
            precision=Precision.coerce(precision),
            n_psi=int(n_psi),
            orientations=orientations,
            planes=None if planes is None else tuple(sorted({int(p) for p in planes})),
            decision=decision,
            local_rings=None if local_rings is None else int(local_rings),
            seeding=seeding,
            name=str(name),
        )

    def __post_init__(self) -> None:
        """Validate (never coerce); :meth:`create` normalises loose inputs."""
        rects = self.rectangles
        if not isinstance(rects, tuple) or not rects:
            raise ValueError(
                "a stage needs a non-empty tuple of rectangles; see Stage.create()"
            )
        for r in rects:
            if r != as_rectangle(r):
                raise ValueError(f"bad rectangle {r!r}; see Stage.create()")
        if not rectangles_disjoint(rects):
            raise ValueError(
                "rectangles within a stage must be pairwise cell-disjoint or the "
                "contraction would double count"
            )
        if not isinstance(self.precision, Precision):
            raise ValueError("precision must be a Precision; see Stage.create()")
        if self.k_stop > self.n_psi // 2 + 1:
            raise ValueError(
                f"k_stop ({self.k_stop}) exceeds n_psi // 2 + 1 "
                f"({self.n_psi // 2 + 1}); increase n_psi to at least "
                f"{2 * (self.k_stop - 1)} or cap k_extent at {self.n_psi // 2 + 1} so "
                "the angular-frequency content is not aliased"
            )
        if self.planes is not None and (
            not isinstance(self.planes, tuple)
            or not self.planes
            or self.planes != tuple(sorted(set(self.planes)))
            or self.planes[0] < 0
        ):
            raise ValueError(
                "planes must be a sorted tuple of distinct plane ids >= 0; see "
                "Stage.create()"
            )
        if isinstance(self.orientations, int):
            if self.orientations < 0:
                raise ValueError("orientations level must be >= 0")
        elif self.orientations is not None and (
            not isinstance(self.orientations, torch.Tensor)
            or self.orientations.dtype is not torch.long
            or self.orientations.dim() != 1
        ):
            raise ValueError(
                "orientations must be None, a HEALPix level or a 1-D int64 tensor of "
                "orientation ids; see Stage.create()"
            )
        if self.decision is not None and not isinstance(self.decision, Decision):
            raise ValueError("decision must be a Decision or None")
        if self.local_rings is not None:
            if not isinstance(self.local_rings, int) or self.local_rings < 0:
                raise ValueError("local_rings must be an int >= 0; see Stage.create()")
            if isinstance(self.orientations, torch.Tensor):
                raise ValueError(
                    "a local stage samples orientations on a HEALPix grid: leave "
                    "orientations None or give a level"
                )
        if self.seeding is not None:
            if not isinstance(self.seeding, Seeding):
                raise ValueError("seeding must be a Seeding; see Stage.create()")
            if self.local_rings is not None:
                raise ValueError(
                    "a stage is either self-seeded (seeding) or local to the previous "
                    "stage (local_rings), not both"
                )
            if isinstance(self.orientations, torch.Tensor):
                raise ValueError(
                    "a seeded stage samples orientations on a HEALPix grid: leave "
                    "orientations None or give a level"
                )
            if self.seeding.level <= self.level:
                raise ValueError(
                    f"seeding level {self.seeding.level} must exceed the stage's "
                    f"orientation level {self.level}"
                )
            if self.k_stop > self.seed_n_psi // 2 + 1:
                raise ValueError(
                    f"k_stop ({self.k_stop}) exceeds the seed pass's n_psi // 2 + 1 "
                    f"({self.seed_n_psi // 2 + 1}); raise seeding.n_psi to at least "
                    f"{2 * (self.k_stop - 1)}"
                )

    @property
    def is_local(self) -> bool:
        """Whether the stage searches per-pixel neighborhoods of the previous stage."""
        return self.local_rings is not None

    @property
    def is_seeded(self) -> bool:
        """Whether the stage seeds itself (see :mod:`.seeding`)."""
        return self.seeding is not None

    @property
    def seed_n_psi(self) -> int:
        """In-plane samples of the seed pass (the stage's unless the seeding says)."""
        if self.seeding is None or self.seeding.n_psi is None:
            return self.n_psi
        return self.seeding.n_psi

    def seed_stage(self) -> Stage:
        """The seed pass as a stage: this stage's features at the seed ``n_psi``.

        Its hypotheses are the seed templates' (:class:`~.seeding.SeedResources`),
        not this stage's; it carries no decision.
        """
        if self.seeding is None:
            raise ValueError("only a seeded stage has a seed pass")
        return replace(
            self,
            n_psi=self.seed_n_psi,
            orientations=None,
            decision=None,
            seeding=None,
            name=f"{self.name}:seed" if self.name else "seed",
        )

    @property
    def level(self) -> int:
        """HEALPix sampling level (0 = the template's grid, also for explicit ids)."""
        return self.orientations if isinstance(self.orientations, int) else 0

    @property
    def k_stop(self) -> int:
        """Largest angular frequency any rectangle reaches."""
        return max(r[0] + r[2] for r in self.rectangles)

    @property
    def num_features(self) -> int:
        """Total ``(k, m)`` cells (the stage's rank)."""
        return sum(r[2] * r[3] for r in self.rectangles)

    def layout(self) -> FeatureLayout:
        """The stage's feature layout."""
        return FeatureLayout(self.rectangles)

    def hypothesis_ids(
        self, space: HypothesisSpace, ipix: np.ndarray | None = None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """``(template_ids, defocus_ids)`` of this stage (``None`` = all).

        ``ipix`` are the template's NESTED HEALPix pixel ids
        (:func:`~panther_em.orientations.healpix_ipix`), needed when ``orientations``
        is a level.
        """
        planes = None if self.planes is None else torch.tensor(self.planes)
        if self.orientations is None or (
            isinstance(self.orientations, int) and self.orientations == 0
        ):
            return None, planes
        if isinstance(self.orientations, int):
            if ipix is None:
                raise ValueError(
                    f"stage {self.name!r}: orientations={self.orientations} needs a "
                    "template whose Euler angles are HEALPix pixel centers (see "
                    "healpix_orientations); pass explicit orientation ids otherwise"
                )
            ids = healpix_level_ids(ipix, self.orientations)
        else:
            ids = self.orientations
        return template_rows_for(space, ids), planes


def stage_to_json(stage: Stage) -> dict:
    """JSON-serialisable description of a stage (for result files)."""
    orientations: object
    if stage.orientations is None or isinstance(stage.orientations, int):
        orientations = stage.orientations
    else:
        orientations = stage.orientations.tolist()
    decision = None
    if stage.decision is not None:
        test = None
        if stage.decision.test is not None:
            test = {
                **asdict(stage.decision.test.calibration),
                "table": stage.decision.test.table.summary(),
            }
        decision = {
            "threshold": stage.decision.threshold,
            "accept": stage.decision.accept,
            "absorb_radius": stage.decision.absorb_radius,
            "test": test,
        }
    return {
        "rectangles": [list(r) for r in stage.rectangles],
        "precision": stage.precision.value,
        "n_psi": stage.n_psi,
        "orientations": orientations,
        "planes": None if stage.planes is None else list(stage.planes),
        "decision": decision,
        "local_rings": stage.local_rings,
        "seeding": None if stage.seeding is None else asdict(stage.seeding),
        "name": stage.name,
    }


def stage_from_json(data: dict) -> Stage:
    """Inverse of :func:`stage_to_json`.

    The decision's threshold test is not restored; its calibration and table stay
    readable in the saved description.
    """
    decision = None
    if data.get("decision") is not None:
        d = data["decision"]
        decision = Decision(
            float(d["threshold"]),
            None,
            bool(d.get("accept", True)),
            int(d.get("absorb_radius", 0)),
        )
    orientations = data.get("orientations")
    return Stage.create(
        [tuple(r) for r in data["rectangles"]],
        precision=data["precision"],
        n_psi=int(data["n_psi"]),
        orientations=orientations,
        planes=data.get("planes"),
        decision=decision,
        local_rings=data.get("local_rings"),
        seeding=data.get("seeding"),
        name=data.get("name", ""),
    )
