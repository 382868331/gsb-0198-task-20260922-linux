"""Adaptive quadtree 5/3 lifting decomposition with cost-based pruning.

Unlike lifting53._core (which recurses into the LL block only), the
adaptive transform allows *every* subband (LL, LH, HL, HH) to be decomposed
further with the same integer 5/3 lifting step, forming a quadtree up to a
caller-given depth.  The fully built tree is then pruned bottom-up with a
caller-supplied per-block cost: for each internal node the algorithm keeps
either the parent block or the expansion into its four (recursively pruned)
children, whichever has the smaller total cost; ties are resolved in favour
of the shallower representation (the parent block is kept).

The default cost is an explicit integer coding-length estimate
(integer_coding_length).  It is an estimate of how many bits the block's
coefficients would take under a trivial signed-magnitude coding; it is NOT
the size of any real compressed file.

The inverse transform (idwt2d_adaptive) validates the tree first and
rejects invalid trees (overlapping/mis-sized children, missing subblocks,
mis-shaped coefficient blocks, ...) with InvalidTreeError before
reconstructing anything.  Reconstruction then follows the tree exactly and
restores the original integer image bit-for-bit, including odd side lengths
and single-row/single-column shapes (no zero-padding is ever invented, so
nothing has to be cropped afterwards).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ._core import (
    MAX_SIDE,
    IntegerLiftingError,
    _fwd_block,
    _inv_block,
    _positive_int_param,
    _scalar,
    _validate_matrix,
    max_levels,
)

#: Child band order everywhere in this module (row-major quadrant order).
BAND_NAMES: tuple[str, str, str, str] = ("LL", "LH", "HL", "HH")


class InvalidTreeError(IntegerLiftingError, ValueError):
    """Raised when a decomposition tree is structurally invalid."""


# ---------------------------------------------------------------------------
# Default cost: explicit integer coding-length estimate
# ---------------------------------------------------------------------------

def integer_coding_length(block: Sequence[Sequence[int]]) -> int:
    """Explicit integer coding-length estimate of one coefficient block.

    Each coefficient v is estimated at 1 + bit_length(|v|) bits (one
    sign/zero bit plus the magnitude bits).  This is a transparent,
    deterministic estimate used only to compare representations; it is not
    and does not claim to be the size of a real compressed file.
    """
    total = 0
    for row in block:
        for v in row:
            total += 1 + abs(v).bit_length()
    return total


DEFAULT_COST_DESCRIPTION = (
    "integer coding-length estimate: sum over coefficients of "
    "1 + bit_length(|v|) bits (an estimate, not a real compressed file size)"
)


# ---------------------------------------------------------------------------
# Tree data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdaptiveNode:
    """One node of the adaptive decomposition quadtree.

    band is "ROOT" at the root and one of "LL", "LH", "HL", "HH"
    otherwise.  depth is 0 at the root.  shape is (height, width) of this
    node's own coefficient block; a zero extent (possible for 1xN / Nx1
    ancestors) means the block holds no samples and the node is always a
    leaf.

    coefficients is this node's own block (the values that are kept when
    the node is a leaf).  children is None for leaves, otherwise exactly
    four nodes in BAND_NAMES order whose shapes are the exact quadrant
    split of this node's shape.
    """

    band: str
    depth: int
    shape: tuple[int, int]
    coefficients: tuple[tuple[int, ...], ...]
    children: tuple[AdaptiveNode, ...] | None = None

    @property
    def is_leaf(self) -> bool:
        return self.children is None

    def leaves(self) -> tuple[AdaptiveNode, ...]:
        """All leaf nodes of this subtree, in row-major tree order."""
        if self.children is None:
            return (self,)
        out: list[AdaptiveNode] = []
        for child in self.children:
            out.extend(child.leaves())
        return tuple(out)


@dataclass(frozen=True)
class AdaptiveResult:
    """Result of dwt2d_adaptive.

    root is the pruned decomposition tree; its leaves carry the kept
    coefficient blocks and their shapes.  depth is the requested maximum
    build depth.  total_cost is the cost of the pruned representation
    under the cost function described by cost_description.
    """

    root: AdaptiveNode
    shape: tuple[int, int]
    depth: int
    total_cost: int
    cost_description: str

    def leaves(self) -> tuple[AdaptiveNode, ...]:
        return self.root.leaves()


# ---------------------------------------------------------------------------
# Forward: build the full quadtree, then prune
# ---------------------------------------------------------------------------

def _quadrant_shapes(h: int, w: int) -> tuple[tuple[int, int], ...]:
    """Shapes of the (LL, LH, HL, HH) children of an h x w block."""
    lh = (h + 1) // 2
    lw = (w + 1) // 2
    return ((lh, lw), (lh, w - lw), (h - lh, lw), (h - lh, w - lw))


def _build(
    block: tuple[tuple[int, ...], ...],
    h: int,
    w: int,
    band: str,
    depth: int,
    max_depth: int,
) -> AdaptiveNode:
    """Build the full (unpruned) quadtree for one h x w block."""
    if depth < max_depth and h >= 1 and w >= 1 and (h > 1 or w > 1):
        work = [list(row) for row in block]
        _fwd_block(work, 0, h, 0, w)
        shapes = _quadrant_shapes(h, w)
        lh, lw = shapes[0]
        children = []
        for name, (ch, cw) in zip(BAND_NAMES, shapes):
            r0, c0 = {
                "LL": (0, 0), "LH": (0, lw), "HL": (lh, 0), "HH": (lh, lw),
            }[name]
            sub = tuple(
                tuple(work[r][c0:c0 + cw]) for r in range(r0, r0 + ch)
            )
            children.append(
                _build(sub, ch, cw, name, depth + 1, max_depth)
            )
        return AdaptiveNode(band, depth, (h, w), block, tuple(children))
    return AdaptiveNode(band, depth, (h, w), block, None)


def _check_cost_value(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"cost function must return a non-negative int, got "
            f"{type(value).__name__} {value!r} (block at {path})"
        )
    if value < 0:
        raise ValueError(
            f"cost function returned a negative cost {value} (block at {path})"
        )
    return value


def _prune(
    node: AdaptiveNode,
    cost: Callable[[tuple[tuple[int, ...], ...]], int],
    path: str,
) -> tuple[AdaptiveNode, int]:
    """Bottom-up optimal pruning; ties keep the shallower representation."""
    if node.children is None:
        return node, _check_cost_value(cost(node.coefficients), path)
    new_children = []
    expanded_total = 0
    for name, child in zip(BAND_NAMES, node.children):
        pruned_child, child_cost = _prune(child, cost, f"{path}/{name}")
        new_children.append(pruned_child)
        expanded_total += child_cost
    keep_cost = _check_cost_value(cost(node.coefficients), path)
    if keep_cost <= expanded_total:
        # Tie or strictly cheaper: prefer the shallower representation.
        return (
            AdaptiveNode(node.band, node.depth, node.shape, node.coefficients, None),
            keep_cost,
        )
    return (
        AdaptiveNode(
            node.band, node.depth, node.shape, node.coefficients,
            tuple(new_children),
        ),
        expanded_total,
    )


def dwt2d_adaptive(
    image: Sequence[Sequence[int]],
    depth: int,
    cost: Callable[[tuple[tuple[int, ...], ...]], int] | None = None,
) -> AdaptiveResult:
    """Adaptive quadtree 5/3 lifting decomposition with optimal pruning.

    Builds the full quadtree (all four subbands may be decomposed further)
    down to depth levels, then prunes it bottom-up so that the total
    cost of the kept leaf blocks is minimal; ties prefer the shallower
    representation.  cost maps one coefficient block (tuple of rows) to
    a non-negative int; the default is integer_coding_length.

    The input matrix is never mutated; on any error no partial result is
    returned.
    """
    depth = _positive_int_param(depth, "depth")
    a = _validate_matrix(image, "image")
    h, w = len(a), len(a[0])
    if h > MAX_SIDE or w > MAX_SIDE:
        raise ValueError(
            f"image shape {h}x{w} exceeds the supported maximum "
            f"{MAX_SIDE}x{MAX_SIDE}"
        )
    limit = max(1, max_levels(h, w))
    if depth > limit:
        raise ValueError(
            f"depth={depth} exceeds the productive maximum {limit} for a "
            f"{h}x{w} image; deeper nodes would be identity on 1x1 blocks"
        )
    if cost is None:
        cost_fn = integer_coding_length
        description = DEFAULT_COST_DESCRIPTION
    else:
        if not callable(cost):
            raise TypeError(
                f"cost must be callable, got {type(cost).__name__}"
            )
        cost_fn = cost
        description = getattr(cost, "__name__", None) or repr(cost)

    block = tuple(tuple(row) for row in a)
    full = _build(block, h, w, "ROOT", 0, depth)
    pruned, total = _prune(full, cost_fn, "ROOT")
    return AdaptiveResult(pruned, (h, w), depth, total, description)


# ---------------------------------------------------------------------------
# Tree validation + exact inverse
# ---------------------------------------------------------------------------

def _validate_block_coeffs(
    coeffs: object, shape: tuple[int, int], path: str
) -> None:
    h, w = shape
    if h == 0:
        if not isinstance(coeffs, (tuple, list)) or len(coeffs) != 0:
            raise InvalidTreeError(
                f"{path}: block with 0 rows must have empty coefficients"
            )
        return
    if not isinstance(coeffs, (tuple, list)) or len(coeffs) != h:
        raise InvalidTreeError(
            f"{path}: expected {h} coefficient rows, got "
            f"{len(coeffs) if isinstance(coeffs, (tuple, list)) else type(coeffs).__name__}"
        )
    for r, row in enumerate(coeffs):
        if not isinstance(row, (tuple, list)) or len(row) != w:
            raise InvalidTreeError(
                f"{path} row {r}: expected {w} coefficients, got "
                f"{len(row) if isinstance(row, (tuple, list)) else type(row).__name__}"
            )
        for c, v in enumerate(row):
            try:
                _scalar(v, f"{path}[{r}][{c}]")
            except (TypeError, ValueError) as exc:
                raise InvalidTreeError(str(exc)) from exc


def _validate_node(
    node: object,
    shape: tuple[int, int],
    band: str,
    depth: int,
    path: str,
) -> None:
    if not isinstance(node, AdaptiveNode):
        raise TypeError(
            f"{path}: expected an AdaptiveNode, got {type(node).__name__}"
        )
    if node.band != band:
        raise InvalidTreeError(
            f"{path}: expected band {band!r}, got {node.band!r}"
        )
    if node.depth != depth:
        raise InvalidTreeError(
            f"{path}: expected depth {depth}, got {node.depth}"
        )
    if tuple(node.shape) != tuple(shape):
        raise InvalidTreeError(
            f"{path}: node shape {tuple(node.shape)} does not match the "
            f"expected quadrant shape {tuple(shape)} (children would overlap "
            f"or leave gaps)"
        )
    _validate_block_coeffs(node.coefficients, tuple(shape), path)

    if node.children is None:
        return
    h, w = shape
    if h < 1 or w < 1 or (h == 1 and w == 1):
        raise InvalidTreeError(
            f"{path}: a {h}x{w} block cannot have children; further "
            f"decomposition would be empty or identity"
        )
    if not isinstance(node.children, (tuple, list)) or len(node.children) != 4:
        got = (
            len(node.children)
            if isinstance(node.children, (tuple, list))
            else type(node.children).__name__
        )
        raise InvalidTreeError(
            f"{path}: internal node must have exactly 4 children "
            f"(LL, LH, HL, HH), got {got}; missing subblocks are not allowed"
        )
    shapes = _quadrant_shapes(h, w)
    for name, child, child_shape in zip(BAND_NAMES, node.children, shapes):
        _validate_node(child, child_shape, name, depth + 1, f"{path}/{name}")


def _reconstruct(node: AdaptiveNode) -> list[list[int]]:
    if node.children is None:
        return [list(row) for row in node.coefficients]
    h, w = node.shape
    blocks = [_reconstruct(child) for child in node.children]
    lh, lw = _quadrant_shapes(h, w)[0]
    a = [[0] * w for _ in range(h)]
    for name, block in zip(BAND_NAMES, blocks):
        r0, c0 = {
            "LL": (0, 0), "LH": (0, lw), "HL": (lh, 0), "HH": (lh, lw),
        }[name]
        for i, row in enumerate(block):
            a[r0 + i][c0:c0 + len(row)] = row
    _inv_block(a, 0, h, 0, w)
    return a


def idwt2d_adaptive(result: AdaptiveResult) -> tuple[tuple[int, ...], ...]:
    """Exact inverse of dwt2d_adaptive.

    The decomposition tree is validated first; invalid trees (overlapping or
    mis-sized children, missing subblocks, mis-shaped coefficient blocks,
    illegal internal nodes) are rejected with InvalidTreeError
    before anything is reconstructed.  Reconstruction then follows the tree
    exactly and restores the original integer image bit-for-bit.
    """
    if not isinstance(result, AdaptiveResult):
        raise TypeError(
            "idwt2d_adaptive expects an AdaptiveResult from dwt2d_adaptive "
            f"(got {type(result).__name__})"
        )
    shape = result.shape
    if (
        not isinstance(shape, (tuple, list)) or len(shape) != 2
        or not all(isinstance(s, int) and not isinstance(s, bool) for s in shape)
    ):
        raise InvalidTreeError(f"malformed shape in AdaptiveResult: {shape!r}")
    h, w = int(shape[0]), int(shape[1])
    if h < 1 or w < 1 or h > MAX_SIDE or w > MAX_SIDE:
        raise InvalidTreeError(
            f"shape {h}x{w} outside the supported 1..{MAX_SIDE} range"
        )
    if not isinstance(result.depth, int) or isinstance(result.depth, bool)             or result.depth < 1:
        raise InvalidTreeError(f"malformed depth in AdaptiveResult: {result.depth!r}")
    _validate_node(result.root, (h, w), "ROOT", 0, "ROOT")
    return tuple(tuple(row) for row in _reconstruct(result.root))

