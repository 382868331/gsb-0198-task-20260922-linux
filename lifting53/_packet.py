"""Adaptive quad-tree (wavelet-packet style) integer 5/3 decomposition.

Unlike :func:`lifting53.dwt2d`, which only recurses into the LL block, this
module lets every one of the four subbands (LL, LH, HL, HH) be decomposed
further with the same reversible integer 5/3 lifting step.  A full tree is
built to a requested depth, then pruned bottom-up with a caller-supplied
per-block cost: for each internal node we either keep the parent's
coefficient block or expand it into its four children, whichever has the
smaller total cost; ties are resolved in favour of the shallower
representation (the parent is kept).

The pruned tree, its leaf coefficient blocks and their shapes fully describe
the transform; :func:`inverse_packet` replays exactly the recorded tree and
restores the original integer image bit-for-bit.  Odd side lengths and
single-row/single-column blocks are handled by the same clamped-boundary
lifting rules as the base module -- nothing is padded, so nothing has to be
cropped afterwards.  Structurally invalid trees (overlapping leaves, missing
children, wrong shapes, out-of-range integers) are rejected before any
arithmetic happens.

The default cost is an explicit integer encoding-length estimate (one sign
bit plus the magnitude bit-length per coefficient).  It is an estimate of
how many bits a trivial explicit integer encoding would need -- it is NOT
the size of any real compressed file.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from ._core import (
    MAX_SIDE,
    IntegerLiftingError,
    _bounded,
    _fwd_block,
    _inv_block,
    _positive_int_param,
    _validate_matrix,
)

#: Band names of the four children of an internal node, in canonical order.
CHILD_BANDS = ("LL", "LH", "HL", "HH")

#: Hard recursion guard for validating user-supplied trees.  A 128x128 image
#: never needs more than 7 productive splits along any path; 64 is generous.
_MAX_TREE_DEPTH = 64


class InvalidTreeError(IntegerLiftingError, ValueError):
    """Raised when a decomposition tree is structurally invalid.

    Examples: a missing child band, children whose shapes/origins do not
    exactly tile the parent block (overlapping or gapping leaves), a
    coefficient block whose shape disagrees with its node, or non-integer /
    out-of-range coefficient values.
    """


@dataclass(frozen=True)
class PacketNode:
    """One node of an adaptive decomposition tree.

    ``band`` is ``"ROOT"`` for the root and one of ``"LL"``, ``"LH"``,
    ``"HL"``, ``"HH"`` otherwise.  ``depth`` is 0 at the root.  ``origin``
    is the ``(row, col)`` offset of this block inside the root's canvas and
    ``shape`` its ``(height, width)``.  ``coefficients`` is the block's
    coefficient matrix as stored at this node (before its own
    decomposition); it always has exactly ``shape``.  ``children`` is empty
    for a leaf, otherwise exactly four :class:`PacketNode` in
    :data:`CHILD_BANDS` order.
    """

    band: str
    depth: int
    origin: tuple[int, int]
    shape: tuple[int, int]
    coefficients: tuple[tuple[int, ...], ...]
    children: tuple["PacketNode", ...] = ()

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def leaves(self) -> tuple["PacketNode", ...]:
        """All leaf nodes of the subtree, in canonical (pre-order) order."""
        if self.is_leaf:
            return (self,)
        out: list[PacketNode] = []
        for child in self.children:
            out.extend(child.leaves())
        return tuple(out)


# ---------------------------------------------------------------------------
# Cost functions
# ---------------------------------------------------------------------------

def default_cost(coefficients: Sequence[Sequence[int]]) -> int:
    """Explicit integer encoding-length estimate for one coefficient block.

    Returns ``sum(1 + abs(v).bit_length())`` over all coefficients: one sign
    bit plus the magnitude bit-length per value (zero costs a single bit).
    This is only an encoding-length estimate, not the size of any real
    compressed file.
    """
    total = 0
    for row in coefficients:
        for v in row:
            total += 1 + abs(v).bit_length()
    return total


def _check_cost_value(value: object, where: str):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"cost function must return an int or float, got "
            f"{type(value).__name__} (at {where})"
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"cost must be finite, got {value!r} (at {where})")
    if value < 0:
        raise ValueError(f"cost must be >= 0, got {value!r} (at {where})")
    return value


# ---------------------------------------------------------------------------
# Shape arithmetic (shared by construction and validation)
# ---------------------------------------------------------------------------

def _child_geometry(h: int, w: int) -> dict:
    """Derived {band: (origin, shape)} of the four children of an hxw block."""
    lr = (h + 1) // 2
    lc = (w + 1) // 2
    return {
        "LL": ((0, 0), (lr, lc)),
        "LH": ((0, lc), (lr, w - lc)),
        "HL": ((lr, 0), (h - lr, lc)),
        "HH": ((lr, lc), (h - lr, w - lc)),
    }


def _decomposable(h: int, w: int) -> bool:
    return h >= 1 and w >= 1 and (h > 1 or w > 1)


# ---------------------------------------------------------------------------
# Forward tree construction
# ---------------------------------------------------------------------------

def _split_block(block: list[list[int]]) -> dict:
    """One 2D 5/3 lifting step on a copy of ``block``; returns the 4 subbands."""
    h = len(block)
    w = len(block[0]) if h else 0
    a = [row[:] for row in block]
    _fwd_block(a, 0, h, 0, w)
    geo = _child_geometry(h, w)
    out: dict = {}
    for band, ((r0, c0), (bh, bw)) in geo.items():
        out[band] = [row[c0:c0 + bw] for row in a[r0:r0 + bh]]
    return out


def _build_node(
    band: str,
    depth: int,
    origin: tuple[int, int],
    shape: tuple[int, int],
    block: list[list[int]],
    max_depth: int,
) -> PacketNode:
    # The shape comes from the parent's tiling geometry; it cannot be derived
    # from the block alone because an empty subband (0 rows or 0 cols, e.g.
    # the high bands of a 1xN block) carries no rows to measure.
    h, w = shape
    if h == 0 or w == 0:
        # An empty subband carries no samples regardless of how the slicing
        # produced it (e.g. a 1x0 block may arrive as one empty row).
        coeffs = ()
    else:
        coeffs = tuple(tuple(row) for row in block)
    node = PacketNode(band, depth, origin, shape, coeffs)
    if depth >= max_depth or not _decomposable(h, w):
        return node
    sub = _split_block(block)
    geo = _child_geometry(h, w)
    r, c = origin
    children = tuple(
        _build_node(
            name,
            depth + 1,
            (r + geo[name][0][0], c + geo[name][0][1]),
            geo[name][1],
            sub[name],
            max_depth,
        )
        for name in CHILD_BANDS
    )
    return replace(node, children=children)


def build_full_tree(image: Sequence[Sequence[int]], depth: int) -> PacketNode:
    """Decompose ``image`` into a full quad-tree down to ``depth`` levels.

    Every subband (not just LL) is recursively decomposed with the same
    reversible integer 5/3 lifting step.  Blocks that reach 1x1 (or become
    empty in one dimension for 1xN / Nx1 inputs) stop early; such blocks are
    leaves and are never padded.  The input matrix is never mutated.
    """
    depth = _positive_int_param(depth, "depth")
    a = _validate_matrix(image, "image")
    h, w = len(a), len(a[0])
    if h > MAX_SIDE or w > MAX_SIDE:
        raise ValueError(
            f"image shape {h}x{w} exceeds the supported maximum "
            f"{MAX_SIDE}x{MAX_SIDE}"
        )
    return _build_node("ROOT", 0, (0, 0), (h, w), a, depth)


# ---------------------------------------------------------------------------
# Tree validation
# ---------------------------------------------------------------------------

def _validate_coeff_block(node: PacketNode, where: str) -> None:
    h, w = node.shape
    coeffs = node.coefficients
    if h == 0 or w == 0:
        if coeffs != ():
            raise InvalidTreeError(
                f"{where}: empty block (shape {h}x{w}) must carry no "
                f"coefficients, got {len(coeffs)} row(s)"
            )
        return
    rows = _validate_matrix(coeffs, f"{where}.coefficients")
    if len(rows) != h or len(rows[0]) != w:
        raise InvalidTreeError(
            f"{where}: coefficient block is {len(rows)}x{len(rows[0])} but "
            f"the node declares shape {h}x{w}"
        )


def _validate_node(node: object, expect_band: str, expect_depth: int,
                   expect_origin: tuple[int, int], expect_shape: tuple[int, int],
                   where: str) -> None:
    if not isinstance(node, PacketNode):
        raise InvalidTreeError(
            f"{where}: expected a PacketNode, got {type(node).__name__}"
        )
    if node.band != expect_band:
        raise InvalidTreeError(
            f"{where}: expected band {expect_band!r}, got {node.band!r}"
        )
    if node.depth != expect_depth:
        raise InvalidTreeError(
            f"{where}: expected depth {expect_depth}, got {node.depth}"
        )
    if expect_depth > _MAX_TREE_DEPTH:
        raise InvalidTreeError(f"{where}: tree depth exceeds {_MAX_TREE_DEPTH}")
    if node.origin != expect_origin:
        raise InvalidTreeError(
            f"{where}: block origin {node.origin} does not match the tiling "
            f"origin {expect_origin} (leaves would overlap or leave gaps)"
        )
    if node.shape != expect_shape:
        raise InvalidTreeError(
            f"{where}: block shape {node.shape} does not match the tiling "
            f"shape {expect_shape} (leaves would overlap or leave gaps)"
        )
    _validate_coeff_block(node, where)

    if node.is_leaf:
        return
    if len(node.children) != len(CHILD_BANDS):
        raise InvalidTreeError(
            f"{where}: an internal node must have exactly {len(CHILD_BANDS)} "
            f"children ({CHILD_BANDS}), got {len(node.children)}"
        )
    h, w = node.shape
    if not _decomposable(h, w):
        raise InvalidTreeError(
            f"{where}: a {h}x{w} block cannot be decomposed further and "
            f"must be a leaf"
        )
    geo = _child_geometry(h, w)
    r, c = node.origin
    for child, name in zip(node.children, CHILD_BANDS):
        (dr, dc), (bh, bw) = geo[name]
        _validate_node(child, name, expect_depth + 1,
                       (r + dr, c + dc), (bh, bw), f"{where}.{name}")


def validate_tree(root: PacketNode) -> PacketNode:
    """Validate the structure of a decomposition tree; return it unchanged.

    Raises :class:`InvalidTreeError` (or ``TypeError`` for non-integer
    coefficients) on any structural problem: missing children, children
    whose shapes or origins do not exactly tile the parent block (overlaps
    or gaps), coefficient/shape mismatches, or malformed shapes.
    """
    if not isinstance(root, PacketNode):
        raise InvalidTreeError(
            f"tree root must be a PacketNode, got {type(root).__name__}"
        )
    if root.band != "ROOT" or root.depth != 0 or root.origin != (0, 0):
        raise InvalidTreeError(
            "tree root must have band='ROOT', depth=0 and origin=(0, 0), got "
            f"band={root.band!r}, depth={root.depth}, origin={root.origin}"
        )
    h, w = root.shape
    for side, name in ((h, "height"), (w, "width")):
        if not isinstance(side, int) or isinstance(side, bool) or side < 1:
            raise InvalidTreeError(f"root: invalid {name} {side!r}")
        if side > MAX_SIDE:
            raise InvalidTreeError(
                f"root: {name} {side} exceeds the supported maximum {MAX_SIDE}"
            )
    _validate_node(root, "ROOT", 0, (0, 0), (h, w), "root")
    return root


# ---------------------------------------------------------------------------
# Cost-optimal pruning
# ---------------------------------------------------------------------------

def _prune(node: PacketNode, cost: Callable, where: str):
    keep = _check_cost_value(cost(node.coefficients), where)
    if node.is_leaf:
        return node, keep
    children = []
    expand = 0
    for child in node.children:
        pruned_child, child_cost = _prune(child, cost, f"{where}.{child.band}")
        children.append(pruned_child)
        expand += child_cost
    if keep <= expand:
        # Tie or strictly cheaper: keep the parent (shallower representation).
        return replace(node, children=()), keep
    return replace(node, children=tuple(children)), expand


def prune_tree(root: PacketNode, cost=None) -> PacketNode:
    """Prune a full tree so the total leaf cost is minimal.

    For every internal node the caller-supplied per-block ``cost`` decides
    whether the parent's coefficient block is kept or replaced by its four
    (recursively optimal) children.  Ties keep the parent, i.e. prefer the
    shallowest optimal representation.  ``cost`` defaults to
    :func:`default_cost`.  The input tree is not mutated; a new tree is
    returned.
    """
    if cost is None:
        cost = default_cost
    if not callable(cost):
        raise TypeError(f"cost must be callable, got {type(cost).__name__}")
    validate_tree(root)
    pruned, _ = _prune(root, cost, "root")
    return pruned


def leaf_cost_total(root: PacketNode, cost=None):
    """Sum of ``cost`` over the tree's leaf coefficient blocks."""
    if cost is None:
        cost = default_cost
    total = 0
    for leaf in validate_tree(root).leaves():
        total += _check_cost_value(cost(leaf.coefficients), f"leaf {leaf.band}")
    return total


def adaptive_decompose(image: Sequence[Sequence[int]], depth: int, cost=None) -> PacketNode:
    """Convenience: :func:`build_full_tree` followed by :func:`prune_tree`."""
    return prune_tree(build_full_tree(image, depth), cost)


# ---------------------------------------------------------------------------
# Inverse transform along the recorded tree
# ---------------------------------------------------------------------------

def _invert_node(node: PacketNode) -> list[list[int]]:
    if node.is_leaf:
        return [list(row) for row in node.coefficients]
    h, w = node.shape
    a = [[0] * w for _ in range(h)]
    geo = _child_geometry(h, w)
    for child in node.children:
        block = _invert_node(child)
        (r0, c0), (bh, bw) = geo[child.band]
        if bh == 0 or bw == 0:
            continue  # empty subband: nothing to place
        for i in range(bh):
            a[r0 + i][c0:c0 + bw] = block[i]
    _inv_block(a, 0, h, 0, w)
    return a


def inverse_packet(root: PacketNode) -> tuple[tuple[int, ...], ...]:
    """Exactly invert a (possibly pruned) decomposition tree.

    The tree is validated first; invalid trees (overlapping leaves, missing
    children, shape mismatches, out-of-range integers) are rejected before
    any arithmetic.  For a tree produced by :func:`build_full_tree` /
    :func:`prune_tree` / :func:`adaptive_decompose` the result is the
    original integer image, bit-for-bit.
    """
    validate_tree(root)
    a = _invert_node(root)
    for r, row in enumerate(a):
        for c, v in enumerate(row):
            _bounded(v, f"reconstructed[{r}][{c}]")
    return tuple(tuple(row) for row in a)


__all__ = [
    "CHILD_BANDS",
    "InvalidTreeError",
    "PacketNode",
    "adaptive_decompose",
    "build_full_tree",
    "default_cost",
    "inverse_packet",
    "leaf_cost_total",
    "prune_tree",
    "validate_tree",
]

