"""Reversible integer 5/3 lifting transform (1D helpers + 2D image transform).

Contract implemented here (see README.md for the full specification):

* Prediction: odd samples  d = x_odd - floor((left_even + right_even)/2)
* Update:     even samples s = x_even + floor((left_detail + right_detail + 2)/4)
* Boundaries: a missing same-kind neighbour is replaced by the nearest valid
  same-kind sample (half-sample symmetric replication / clamping).
* Length 1 signals are the identity transform.
* 2D applies rows first, then columns; multi-level decompositions recurse into
  the LL block only.
* Pure integer arithmetic (no float anywhere in the algorithm); every stored
  input/output value and every checked intermediate must satisfy
  |v| <= 2^63 - 1, otherwise IntegerOverflowError is raised (never truncated).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

#: Hard magnitude limit for inputs, outputs and checked intermediates.
INT63_MAX = (1 << 63) - 1

#: Maximum accepted image side length (interface boundary from the task).
MAX_SIDE = 128


class IntegerLiftingError(Exception):
    """Base class for errors raised by this package."""


class IntegerOverflowError(IntegerLiftingError, OverflowError):
    """Raised when an integer value exceeds +/- (2^63 - 1).

    This is raised both for out-of-range inputs and for checked intermediate
    results.  Values are never wrapped or truncated.
    """


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _bounded(v: int, where: str) -> int:
    if v > INT63_MAX or v < -INT63_MAX:
        raise IntegerOverflowError(
            f"integer {v} (abs={abs(v)}) exceeds the limit +/- (2^63-1)"
            f" = +/-{INT63_MAX} at {where}"
        )
    return v


def _scalar(v: object, where: str) -> int:
    """Validate one integer pixel/coefficient.

    bool is rejected even though it is an int subclass; floats are rejected
    (with an explicit non-finite message for NaN/Infinity).
    """
    if isinstance(v, bool):
        raise TypeError(f"bool is not accepted as an integer value (at {where})")
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError(
                f"non-finite value (NaN/Infinity) is not allowed (at {where})"
            )
        raise TypeError(f"only integers are accepted, got float {v!r} (at {where})")
    if not isinstance(v, int):
        raise TypeError(
            f"only integers are accepted, got {type(v).__name__} {v!r} (at {where})"
        )
    return _bounded(v, where)


def _is_real_sequence(obj: object) -> bool:
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray))


def _validate_matrix(matrix: object, what: str) -> list[list[int]]:
    """Validate a rectangular integer matrix; return a private deep copy."""
    if not _is_real_sequence(matrix):
        raise TypeError(
            f"{what} must be a non-empty sequence of rows "
            f"(got {type(matrix).__name__})"
        )
    if len(matrix) == 0:
        raise ValueError(f"{what} must contain at least 1 row")

    rows: list[list[int]] = []
    width: int | None = None
    for r, row in enumerate(matrix):
        if not _is_real_sequence(row):
            raise TypeError(
                f"{what}[{r}] must be a sequence of integer samples "
                f"(got {type(row).__name__})"
            )
        if width is None:
            width = len(row)
            if width == 0:
                raise ValueError(f"{what}[{r}] is empty; width must be at least 1")
        elif len(row) != width:
            raise ValueError(
                f"{what}[{r}] has length {len(row)} but row 0 has length {width}; "
                f"{what} must be rectangular"
            )
        rows.append([_scalar(v, f"{what}[{r}][{c}]") for c, v in enumerate(row)])
    return rows


def _positive_int_param(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be int, bool is not accepted")
    if not isinstance(value, int):
        raise TypeError(f"{name} must be int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


# ---------------------------------------------------------------------------
# Core 1D lifting (operates on plain lists, context string for locatable errors)
# ---------------------------------------------------------------------------

def _fwd1d(x: list[int], ctx: str) -> list[int]:
    n = len(x)
    if n == 1:
        # Length 1 is the identity transform.
        return [x[0]]

    low_count = (n + 1) // 2
    high_count = n // 2
    ev = [x[2 * k] for k in range(low_count)]

    # Prediction: detail[j] = x[2j+1] - floor((even[j] + even[j+1_clamped]) / 2)
    detail = [0] * high_count
    for j in range(high_count):
        left = ev[j]
        right = ev[j + 1] if j + 1 < low_count else ev[j]  # right-edge mirror
        total = _bounded(left + right, f"{ctx}: even-neighbour sum, detail {j}")
        detail[j] = _bounded(
            x[2 * j + 1] - total // 2, f"{ctx}: prediction result, detail {j}"
        )

    # Update: approx[k] = even[k] + floor((detail[k-1_clamped] + detail[k_clamped] + 2) / 4)
    approx = [0] * low_count
    for k in range(low_count):
        dl = detail[k - 1] if k >= 1 else detail[0]          # left-edge mirror
        dr = detail[k] if k < high_count else detail[high_count - 1]
        total = _bounded(dl + dr, f"{ctx}: detail sum, approx {k}")
        total = _bounded(total + 2, f"{ctx}: detail sum + 2, approx {k}")
        approx[k] = _bounded(
            ev[k] + total // 4, f"{ctx}: update result, approx {k}"
        )

    # Deinterleaved layout: low-pass (even) samples first, high-pass after.
    return approx + detail


def _inv1d(y: list[int], ctx: str) -> list[int]:
    n = len(y)
    if n == 1:
        return [y[0]]

    low_count = (n + 1) // 2
    high_count = n // 2
    approx = y[:low_count]
    detail = y[low_count:]

    # Undo update (same mirror convention).
    ev = [0] * low_count
    for k in range(low_count):
        dl = detail[k - 1] if k >= 1 else detail[0]
        dr = detail[k] if k < high_count else detail[high_count - 1]
        total = _bounded(dl + dr, f"{ctx}: detail sum, even {k}")
        total = _bounded(total + 2, f"{ctx}: detail sum + 2, even {k}")
        ev[k] = _bounded(
            approx[k] - total // 4, f"{ctx}: inverse-update result, even {k}"
        )

    # Undo prediction and re-interleave.
    out = [0] * n
    for k in range(low_count):
        out[2 * k] = ev[k]
    for j in range(high_count):
        left = ev[j]
        right = ev[j + 1] if j + 1 < low_count else ev[j]
        total = _bounded(left + right, f"{ctx}: even-neighbour sum, odd {j}")
        out[2 * j + 1] = _bounded(
            detail[j] + total // 2, f"{ctx}: inverse-prediction result, odd {j}"
        )
    return out


# ---------------------------------------------------------------------------
# Public 1D API
# ---------------------------------------------------------------------------

def dwt1d(signal: Sequence[int]) -> tuple[int, ...]:
    """Forward 1D reversible 5/3 lifting transform.

    Returns a new tuple with ceil(n/2) approximation (even-indexed) samples
    followed by floor(n/2) detail samples.  The input is never mutated.
    """
    if not _is_real_sequence(signal):
        raise TypeError(
            f"signal must be a sequence of integers (got {type(signal).__name__})"
        )
    if len(signal) == 0:
        raise ValueError("signal must contain at least 1 sample")
    x = [_scalar(v, f"signal[{i}]") for i, v in enumerate(signal)]
    return tuple(_fwd1d(x, "signal"))


def idwt1d(coefficients: Sequence[int]) -> tuple[int, ...]:
    """Exact inverse of :func:`dwt1d`.  Input is never mutated."""
    if not _is_real_sequence(coefficients):
        raise TypeError(
            "coefficients must be a sequence of integers "
            f"(got {type(coefficients).__name__})"
        )
    if len(coefficients) == 0:
        raise ValueError("coefficients must contain at least 1 sample")
    y = [_scalar(v, f"coefficients[{i}]") for i, v in enumerate(coefficients)]
    return tuple(_inv1d(y, "signal"))


# ---------------------------------------------------------------------------
# Metadata data structures (fixed public contract)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Subband:
    """A rectangular subband region inside the coefficient matrix.

    Ranges are half-open Python-style intervals ``[row_start, row_end)`` and
    ``[col_start, col_end)``; a zero-extent range (possible for 1xN / Nx1
    images) means that subband contains no samples.
    """

    name: str  # "LL", "LH", "HL" or "HH"
    level: int  # 1-based decomposition level
    rows: tuple[int, int]
    cols: tuple[int, int]


@dataclass(frozen=True)
class LevelLayout:
    """Layout of one decomposition level within the coefficient matrix."""

    level: int
    rows: tuple[int, int]  # working region transformed at this level
    cols: tuple[int, int]
    low_rows: int  # ceil(working_height / 2)
    low_cols: int  # ceil(working_width / 2)
    bands: tuple[Subband, ...]  # exactly (LL, LH, HL, HH)

    def band(self, name: str) -> Subband:
        for b in self.bands:
            if b.name == name:
                return b
        raise KeyError(name)


@dataclass(frozen=True)
class TransformResult:
    """Result of :func:`dwt2d`.

    ``coefficients`` is the full coefficient matrix with an in-place recursive
    pyramid layout (level k+1 lives inside level k's LL block).  ``subbands``
    lists LH/HL/HH of every level plus the final deepest LL; per-level LL
    rectangles are available through ``layouts``.
    """

    coefficients: tuple[tuple[int, ...], ...]
    shape: tuple[int, int]
    levels: int
    layouts: tuple[LevelLayout, ...]
    subbands: tuple[Subband, ...]

    def band(self, name: str, level: int) -> Subband:
        for b in self.subbands:
            if b.name == name and b.level == level:
                return b
        # Interior LL blocks exist in layouts even though they are not in the
        # aggregate subband list.
        if name == "LL" and 1 <= level <= self.levels:
            return self.layouts[level - 1].band("LL")
        raise KeyError((name, level))

    def extract(self, name: str, level: int) -> tuple[tuple[int, ...], ...]:
        """Return a defensive copy of one subband's coefficient values."""
        b = self.band(name, level)
        r0, r1 = b.rows
        c0, c1 = b.cols
        if r1 <= r0 or c1 <= c0:
            # Zero extent in either dimension: the subband holds no samples.
            return ()
        return tuple(
            tuple(self.coefficients[r][c0:c1]) for r in range(r0, r1)
        )


# ---------------------------------------------------------------------------
# 2D block operations
# ---------------------------------------------------------------------------

def _fwd_block(a: list[list[int]], r0: int, r1: int, c0: int, c1: int) -> None:
    # Rows first.
    for r in range(r0, r1):
        a[r][c0:c1] = _fwd1d(a[r][c0:c1], f"row {r}")
    # Then columns.
    for c in range(c0, c1):
        col = [a[r][c] for r in range(r0, r1)]
        out = _fwd1d(col, f"column {c}")
        for i, r in enumerate(range(r0, r1)):
            a[r][c] = out[i]


def _inv_block(a: list[list[int]], r0: int, r1: int, c0: int, c1: int) -> None:
    # Strict reverse order: columns first, then rows.
    for c in range(c0, c1):
        col = [a[r][c] for r in range(r0, r1)]
        out = _inv1d(col, f"column {c}")
        for i, r in enumerate(range(r0, r1)):
            a[r][c] = out[i]
    for r in range(r0, r1):
        a[r][c0:c1] = _inv1d(a[r][c0:c1], f"row {r}")


def max_levels(height: int, width: int) -> int:
    """Number of productive decomposition levels for a given shape.

    A level is productive while at least one side is longer than 1; each side
    shrinks as ceil(s/2).  1xN images keep splitting horizontally.
    """
    h = _positive_int_param(height, "height")
    w = _positive_int_param(width, "width")
    count = 0
    while h > 1 or w > 1:
        h = (h + 1) // 2
        w = (w + 1) // 2
        count += 1
    return count


def _check_levels(h: int, w: int, levels: int) -> None:
    """Validate the level count against the shape (shared by dwt2d/codec)."""
    limit = max_levels(h, w)
    if levels > limit:
        # A 1x1 image (limit == 0) accepts exactly one level, which is the
        # documented length-1 identity; anything beyond it is rejected.
        if not (limit == 0 and levels == 1):
            raise ValueError(
                f"levels={levels} exceeds max_levels({h}, {w})={limit}; "
                f"further levels would be identity on a 1x1 LL block"
            )


def _build_layouts(h: int, w: int, levels: int) -> tuple[LevelLayout, ...]:
    """Deterministically derive every level layout from shape + level count.

    The bitstream codec stores only shape and levels; the decoder rebuilds
    layouts with this same function.
    """
    layouts: list[LevelLayout] = []
    r0 = c0 = 0
    r1, c1 = h, w
    for level in range(1, levels + 1):
        lr = (r1 - r0 + 1) // 2
        lc = (c1 - c0 + 1) // 2
        bands = (
            Subband("LL", level, (r0, r0 + lr), (c0, c0 + lc)),
            Subband("LH", level, (r0, r0 + lr), (c0 + lc, c1)),
            Subband("HL", level, (r0 + lr, r1), (c0, c0 + lc)),
            Subband("HH", level, (r0 + lr, r1), (c0 + lc, c1)),
        )
        layouts.append(
            LevelLayout(level, (r0, r1), (c0, c1), lr, lc, bands)
        )
        # Multi-level: recurse into the LL region only.
        r1 = r0 + lr
        c1 = c0 + lc
    return tuple(layouts)


def _aggregate_subbands(
    layouts: tuple[LevelLayout, ...], levels: int
) -> tuple[Subband, ...]:
    return tuple(
        b
        for layout in layouts
        for b in layout.bands
        if not (b.name == "LL" and layout.level < levels)
    )


def dwt2d(image: Sequence[Sequence[int]], levels: int = 1) -> TransformResult:
    """Forward 2D reversible 5/3 lifting decomposition.

    Returns a :class:`TransformResult`.  The input matrix is never mutated;
    on any error (validation or overflow) no partial result is returned.
    """
    levels = _positive_int_param(levels, "levels")
    a = _validate_matrix(image, "image")
    h, w = len(a), len(a[0])

    if h > MAX_SIDE or w > MAX_SIDE:
        raise ValueError(
            f"image shape {h}x{w} exceeds the supported maximum "
            f"{MAX_SIDE}x{MAX_SIDE}"
        )

    _check_levels(h, w, levels)

    layouts = _build_layouts(h, w, levels)
    for layout in layouts:
        _fwd_block(a, layout.rows[0], layout.rows[1],
                   layout.cols[0], layout.cols[1])

    aggregate = _aggregate_subbands(layouts, levels)
    coeffs = tuple(tuple(row) for row in a)
    return TransformResult(coeffs, (h, w), levels, layouts, aggregate)


def _validate_result(result: object) -> TransformResult:
    if not isinstance(result, TransformResult):
        raise TypeError(
            "idwt2d expects a TransformResult from dwt2d "
            f"(got {type(result).__name__})"
        )
    h, w = result.shape
    levels = result.levels
    if (not isinstance(h, int) or isinstance(h, bool)
            or not isinstance(w, int) or isinstance(w, bool)
            or not isinstance(levels, int) or isinstance(levels, bool)
            or h < 1 or w < 1 or h > MAX_SIDE or w > MAX_SIDE
            or levels < 1):
        raise ValueError(f"malformed shape/levels in TransformResult: "
                         f"shape={result.shape}, levels={levels}")
    if len(result.layouts) != levels:
        raise ValueError("malformed TransformResult: layouts/levels mismatch")

    # Every layout must be exactly the canonical layout deterministically
    # derived from shape + level count; this rejects tampered metadata rather
    # than trusting it.  The level-count rule is re-checked here as well.
    _check_levels(h, w, levels)
    canonical = _build_layouts(h, w, levels)
    if result.layouts != canonical:
        raise ValueError(
            "malformed TransformResult: layouts do not match the canonical "
            "layout derived from shape and levels"
        )

    expected_aggregate = _aggregate_subbands(canonical, levels)
    if result.subbands != expected_aggregate:
        raise ValueError(
            "malformed TransformResult: subband list does not match layouts"
        )
    return result


def idwt2d(result: TransformResult) -> tuple[tuple[int, ...], ...]:
    """Exact inverse of :func:`dwt2d`, reconstructing the original image.

    Levels are undone in strict reverse order.  Pure integer arithmetic;
    raises :class:`IntegerOverflowError` if a checked intermediate exceeds
    the int63 limit.
    """
    result = _validate_result(result)
    a = _validate_matrix(result.coefficients, "coefficients")
    if (len(a), len(a[0])) != result.shape:
        raise ValueError(
            f"coefficient matrix shape {len(a)}x{len(a[0])} does not match "
            f"recorded shape {result.shape[0]}x{result.shape[1]}"
        )

    for layout in reversed(result.layouts):
        r0, r1 = layout.rows
        c0, c1 = layout.cols
        _inv_block(a, r0, r1, c0, c1)

    return tuple(tuple(row) for row in a)
