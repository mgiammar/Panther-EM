"""Numeric precision of the feature/weight contraction.

A plain ``Literal`` of names so every layer (approximation, search, the program
configs) shares one type without importing the search.
"""

from __future__ import annotations

from typing import Literal, TypeAlias, get_args

Precision: TypeAlias = Literal["fp32", "tf32", "fp16"]
PRECISIONS: tuple[Precision, ...] = get_args(Precision)


def check_precision(value: str) -> Precision:
    """``value`` lowercased, checked to be one of :data:`PRECISIONS`."""
    name = str(value).lower()
    if name not in PRECISIONS:
        raise ValueError(
            f"unknown precision {value!r}; expected one of {list(PRECISIONS)}"
        )
    return name
