"""lifting53 — reversible integer 5/3 lifting wavelet transform for 2D images.

Public API
----------
dwt2d(image, levels=1) -> TransformResult
idwt2d(result, *, resolution=0) -> tuple of tuples
    # resolution=0: full original-resolution reconstruction
    # resolution=k (1..levels): only the level-k low-resolution LL image,
    # shape (ceil(H/2**k), ceil(W/2**k)); unrelated detail coefficients are
    # never read.  Corrupted level metadata is rejected before processing.
dwt1d(signal) / idwt1d(coefficients)
max_levels(height, width)
TransformResult.shape_at_level(level) -> (height, width) of the level-k LL
Subband, LevelLayout, TransformResult
IntegerLiftingError, IntegerOverflowError

Constants: INT63_MAX = 2**63 - 1, MAX_SIDE = 128.
"""
from ._core import (
    INT63_MAX,
    MAX_SIDE,
    IntegerLiftingError,
    IntegerOverflowError,
    LevelLayout,
    Subband,
    TransformResult,
    dwt1d,
    dwt2d,
    idwt1d,
    idwt2d,
    max_levels,
)

__all__ = [
    "INT63_MAX",
    "MAX_SIDE",
    "IntegerLiftingError",
    "IntegerOverflowError",
    "LevelLayout",
    "Subband",
    "TransformResult",
    "dwt1d",
    "dwt2d",
    "idwt1d",
    "idwt2d",
    "max_levels",
]

__version__ = "1.0.0"
