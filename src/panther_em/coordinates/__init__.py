"""Coordinate transform system for Panther-EM.

Provides the abstract base class, registry, and built-in polar coordinate
transform implementations.
"""

# Import built-in transforms to trigger @register_transform
# from .nonuniform_polar import NonUniformPolarTransform
# # TODO: add nonuniform_polar.py before exporting
from .offset_polar import OffsetPolarTransform
from .spiral_polar import SpiralPolarTransform
from .standard_polar import StandardPolarTransform
from .transform_base import (
    CoordinateTransform,
    GridTransform,
    get_transform,
    get_transform_class,
    reconstruct_transform,
    register_transform,
)

__all__ = [
    "CoordinateTransform",
    "GridTransform",
    # "NonUniformPolarTransform",  # TODO: enable when implemented
    "OffsetPolarTransform",
    "SpiralPolarTransform",
    "StandardPolarTransform",
    "get_transform",
    "get_transform_class",
    "reconstruct_transform",
    "register_transform",
]
