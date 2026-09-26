"""lifting53 — reversible integer 5/3 lifting wavelet transform for 2D images.

Public API
----------
dwt2d(image, levels=1) -> TransformResult
idwt2d(result) -> tuple of tuples (the reconstructed integer image)
dwt1d(signal) / idwt1d(coefficients)
max_levels(height, width)
Subband, LevelLayout, TransformResult

Adaptive quadtree decomposition (all four subbands may be decomposed,
cost-based optimal pruning, tree-validated exact inverse):
dwt2d_adaptive(image, depth, cost=None) -> AdaptiveResult
idwt2d_adaptive(result) -> tuple of tuples
AdaptiveNode, AdaptiveResult, InvalidTreeError, integer_coding_length

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
from ._adaptive import (
    BAND_NAMES,
    DEFAULT_COST_DESCRIPTION,
    AdaptiveNode,
    AdaptiveResult,
    InvalidTreeError,
    dwt2d_adaptive,
    idwt2d_adaptive,
    integer_coding_length,
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
    "BAND_NAMES",
    "DEFAULT_COST_DESCRIPTION",
    "AdaptiveNode",
    "AdaptiveResult",
    "InvalidTreeError",
    "dwt2d_adaptive",
    "idwt2d_adaptive",
    "integer_coding_length",
]

__version__ = "1.1.0"
