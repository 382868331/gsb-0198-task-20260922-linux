"""lifting53 — reversible integer 5/3 lifting wavelet transform for 2D images.

Public API
----------
dwt2d(image, levels=1) -> TransformResult
idwt2d(result) -> tuple of tuples (the reconstructed integer image)
idwt2d_roi(result, rows, cols, read_coeff) -> ROI-local reconstruction
roi_access_plan(result, rows, cols) -> ROIPlan (exact forced coefficients)
dwt1d(signal) / idwt1d(coefficients)
max_levels(height, width)
Subband, LevelLayout, TransformResult, ROIPlan, ROIBandAccess, CoeffAccess
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
from ._roi import (
    CoeffAccess,
    ROIBandAccess,
    ROIError,
    ROIPlan,
    idwt2d_roi,
    roi_access_plan,
)

__all__ = [
    "INT63_MAX",
    "MAX_SIDE",
    "IntegerLiftingError",
    "IntegerOverflowError",
    "LevelLayout",
    "CoeffAccess",
    "ROIBandAccess",
    "ROIError",
    "ROIPlan",
    "Subband",
    "TransformResult",
    "dwt1d",
    "dwt2d",
    "idwt1d",
    "idwt2d",
    "idwt2d_roi",
    "max_levels",
    "roi_access_plan",
]

__version__ = "1.1.0"
