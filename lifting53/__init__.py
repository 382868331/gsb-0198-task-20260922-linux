"""lifting53 — reversible integer 5/3 lifting wavelet transform for 2D images.

Public API
----------
dwt2d(image, levels=1) -> TransformResult
idwt2d(result) -> tuple of tuples (the reconstructed integer image)
dwt1d(signal) / idwt1d(coefficients)
max_levels(height, width)

Adaptive quad-tree decomposition (all four subbands may be decomposed):
build_full_tree(image, depth) -> PacketNode
prune_tree(root, cost=None) -> PacketNode   (cost-optimal, ties stay shallow)
adaptive_decompose(image, depth, cost=None) -> PacketNode
inverse_packet(root) -> reconstructed image
default_cost(coefficients) -> explicit integer encoding-length estimate
leaf_cost_total(root, cost=None) / validate_tree(root)
PacketNode, InvalidTreeError, CHILD_BANDS
Subband, LevelLayout, TransformResult
IntegerLiftingError, IntegerOverflowError

Constants: INT63_MAX = 2**63 - 1, MAX_SIDE = 128.
"""
from ._packet import (
    CHILD_BANDS,
    InvalidTreeError,
    PacketNode,
    adaptive_decompose,
    build_full_tree,
    default_cost,
    inverse_packet,
    leaf_cost_total,
    prune_tree,
    validate_tree,
)
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
    "CHILD_BANDS",
    "INT63_MAX",
    "InvalidTreeError",
    "PacketNode",
    "adaptive_decompose",
    "build_full_tree",
    "default_cost",
    "inverse_packet",
    "leaf_cost_total",
    "prune_tree",
    "validate_tree",
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

__version__ = "1.1.0"
