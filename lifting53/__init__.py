"""lifting53 — reversible integer 5/3 lifting wavelet transform for 2D images.

Public API
----------
dwt2d(image, levels=1) -> TransformResult
idwt2d(result) -> tuple of tuples (the reconstructed integer image)
dwt1d(signal) / idwt1d(coefficients)
max_levels(height, width)
Subband, LevelLayout, TransformResult
IntegerLiftingError, IntegerOverflowError

Constants: INT63_MAX = 2**63 - 1, MAX_SIDE = 128.
"""
from ._core import (
    INT63_MAX,
    MAX_SIDE,
    CoefficientReader,
    IntegerLiftingError,
    IntegerOverflowError,
    LevelLayout,
    Subband,
    TransformResult,
    dwt1d,
    dwt2d,
    idwt2d_roi,
    idwt1d,
    idwt2d,
    make_reader,
    max_levels,
)

__all__ = [
    "INT63_MAX",
    "MAX_SIDE",
    "CoefficientReader",
    "IntegerLiftingError",
    "IntegerOverflowError",
    "LevelLayout",
    "Subband",
    "TransformResult",
    "dwt1d",
    "dwt2d",
    "idwt2d_roi",
    "idwt1d",
    "idwt2d",
    "make_reader",
    "max_levels",
]

__version__ = "1.0.0"
