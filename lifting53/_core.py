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

    limit = max_levels(h, w)
    if levels > limit:
        # A 1x1 image (limit == 0) accepts exactly one level, which is the
        # documented length-1 identity; anything beyond it is rejected.
        if not (limit == 0 and levels == 1):
            raise ValueError(
                f"levels={levels} exceeds max_levels({h}, {w})={limit}; "
                f"further levels would be identity on a 1x1 LL block"
            )

    layouts: list[LevelLayout] = []
    r0 = c0 = 0
    r1, c1 = h, w
    for level in range(1, levels + 1):
        _fwd_block(a, r0, r1, c0, c1)
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

    aggregate = tuple(
        b
        for layout in layouts
        for b in layout.bands
        if not (b.name == "LL" and layout.level < levels)
    )
    coeffs = tuple(tuple(row) for row in a)
    return TransformResult(coeffs, (h, w), levels, tuple(layouts), aggregate)


def _validate_result(result: object) -> TransformResult:
    if not isinstance(result, TransformResult):
        raise TypeError(
            "idwt2d expects a TransformResult from dwt2d "
            f"(got {type(result).__name__})"
        )
    h, w = result.shape
    if (not isinstance(h, int) or isinstance(h, bool)
            or not isinstance(w, int) or isinstance(w, bool)
            or h < 1 or w < 1 or h > MAX_SIDE or w > MAX_SIDE):
        raise ValueError(f"malformed shape in TransformResult: {result.shape}")
    if len(result.layouts) != result.levels:
        raise ValueError("malformed TransformResult: layouts/levels mismatch")
    for layout in result.layouts:
        r0, r1 = layout.rows
        c0, c1 = layout.cols
        if not (0 <= r0 < r1 <= h and 0 <= c0 < c1 <= w):
            raise ValueError(
                f"malformed layout at level {layout.level}: "
                f"rows {layout.rows}, cols {layout.cols} outside shape"
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


# ---------------------------------------------------------------------------
# Partial inverse driven by a requested output ROI (no full reconstruction)
# ---------------------------------------------------------------------------

def _dep1d_indices(out_indices: set[int], n: int) -> set[int]:
    """Coefficient indices needed to reconstruct out_indices of length n.

    Derived directly from the inverse lifting dependency:

    * even[2k] needs approx[k] (and details k-1/k to undo update);
    * odd[2j+1] needs detail[j] plus even[j] and clamped even[j+1].

    Returns indices into the deinterleaved coefficient vector (approx first,
    detail after).  Clamped (replicated) neighbours are real valid
    coefficients, so they appear as actual reads; missing out-of-range
    samples are never synthesised as zeros.
    """
    if n == 1:
        return {0} if 0 in out_indices else set()
    low_count = (n + 1) // 2
    high_count = n // 2
    approx_needed: set[int] = set()
    for i in out_indices:
        if i % 2 == 0:
            approx_needed.add(i // 2)
        else:
            j = i // 2
            approx_needed.add(j)
            approx_needed.add(min(j + 1, low_count - 1))
    detail_needed: set[int] = set()
    for k in approx_needed:
        # Undo-update reads detail[k-1 clamped] and detail[k clamped].
        if k >= 1:
            detail_needed.add(k - 1)
        if k < high_count:
            detail_needed.add(k)
        else:
            detail_needed.add(high_count - 1)
    return approx_needed | {low_count + j for j in detail_needed}


def _inv1d_partial(
    coeffs: dict[int, int],
    n: int,
    out_indices: set[int],
    ctx: str,
) -> dict[int, int]:
    """Inverse 1D lifting producing only requested output indices.

    coeffs must cover exactly the deinterleaved positions returned by
    _dep1d_indices.  Same arithmetic and overflow checks as the full _inv1d.
    """
    if n == 1:
        return {0: _bounded(coeffs[0], f"{ctx}: identity sample")}

    low_count = (n + 1) // 2
    high_count = n // 2

    approx_needed: set[int] = set()
    for i in out_indices:
        if i % 2 == 0:
            approx_needed.add(i // 2)
        else:
            j = i // 2
            approx_needed.add(j)
            approx_needed.add(min(j + 1, low_count - 1))

    ev: dict[int, int] = {}
    for k in approx_needed:
        dl = coeffs[low_count + (k - 1 if k >= 1 else 0)]
        dr = coeffs[low_count + (k if k < high_count else high_count - 1)]
        total = _bounded(dl + dr, f"{ctx}: detail sum, even {k}")
        total = _bounded(total + 2, f"{ctx}: detail sum + 2, even {k}")
        ev[k] = _bounded(
            coeffs[k] - total // 4, f"{ctx}: inverse-update result, even {k}"
        )

    out: dict[int, int] = {}
    for i in out_indices:
        if i % 2 == 0:
            out[i] = ev[i // 2]
        else:
            j = i // 2
            left = ev[j]
            right = ev[min(j + 1, low_count - 1)]
            total = _bounded(left + right, f"{ctx}: even-neighbour sum, odd {j}")
            out[i] = _bounded(
                coeffs[low_count + j] + total // 2,
                f"{ctx}: inverse-prediction result, odd {j}",
            )
    return out


def _pair_of_ints(value: object, name: str) -> tuple[int, int]:
    if not _is_real_sequence(value) or len(value) != 2:
        raise TypeError(
            f"{name} must be a sequence of exactly 2 ints "
            f"(got {type(value).__name__})"
        )
    out = []
    for i, v in enumerate(value):
        if isinstance(v, bool) or not isinstance(v, int):
            raise TypeError(
                f"{name}[{i}] must be int, got {type(v).__name__} {v!r}"
            )
        out.append(v)
    return out[0], out[1]


def _validate_roi(roi: object, h: int, w: int) -> tuple[int, int, int, int]:
    if not _is_real_sequence(roi) or len(roi) != 4:
        raise TypeError(
            "roi must be a sequence of exactly 4 ints "
            f"(r0, r1, c0, c1); got {type(roi).__name__}"
        )
    names = ("r0", "r1", "c0", "c1")
    vals: list[int] = []
    for name, v in zip(names, roi):
        if isinstance(v, bool) or not isinstance(v, int):
            raise TypeError(
                f"roi {name} must be int, got {type(v).__name__} {v!r}"
            )
        vals.append(v)
    r0, r1, c0, c1 = vals
    if not (0 <= r0 < r1 <= h):
        raise ValueError(
            f"roi rows [{r0}, {r1}) are invalid for image height {h}; "
            f"need 0 <= r0 < r1 <= {h}"
        )
    if not (0 <= c0 < c1 <= w):
        raise ValueError(
            f"roi cols [{c0}, {c1}) are invalid for image width {w}; "
            f"need 0 <= c0 < c1 <= {w}"
        )
    return r0, r1, c0, c1


def idwt2d_roi(
    shape: Sequence[int],
    levels: int,
    roi: Sequence[int],
    read,
) -> tuple[tuple[int, ...], ...]:
    """Reconstruct only the pixels of the half-open rectangle roi.

    shape is (height, width), levels is the number of decomposition levels,
    roi is (r0, r1, c0, c1) with 0 <= r0 < r1 <= height and
    0 <= c0 < c1 <= width.  Coefficients are fetched on demand through
    read(name, level, row, col), where name is "LL" (deepest level only),
    "LH", "HL" or "HH" and (row, col) are coefficient-matrix coordinates
    (the same coordinates reported by Subband).  Each needed coordinate is
    read at most once; the caller owns the access log and may cache.

    The required coefficient set is derived level by level from the 5/3
    inverse lifting dependencies (including same-kind clamped neighbours),
    then the pyramid is rebuilt bottom-up inside that halo.  Coefficients
    outside the dependency are never requested and never assumed zero; the
    image is never fully reconstructed merely to crop it afterwards.  Every
    argument, including roi, is validated before read is called.
    """
    if not callable(read):
        raise TypeError(f"read must be callable (got {type(read).__name__})")
    h, w = _pair_of_ints(shape, "shape")
    if h < 1 or w < 1 or h > MAX_SIDE or w > MAX_SIDE:
        raise ValueError(
            f"shape {h}x{w} must satisfy 1 <= side <= {MAX_SIDE}"
        )
    levels = _positive_int_param(levels, "levels")
    limit = max_levels(h, w)
    if levels > limit and not (limit == 0 and levels == 1):
        raise ValueError(
            f"levels={levels} exceeds max_levels({h}, {w})={limit}"
        )
    r0, r1, c0, c1 = _validate_roi(roi, h, w)

    # Working-region dimensions per level (all regions start at (0, 0)).
    dims: list[tuple[int, int]] = []
    hh, ww = h, w
    for _ in range(levels):
        dims.append((hh, ww))
        hh = (hh + 1) // 2
        ww = (ww + 1) // 2

    # Top-down dependency propagation.
    # need_rows/cols[L]: positions the level L inverse must produce.
    # coeff_rows/cols[L]: coefficient positions used at level L.
    need_rows: list[set[int] | None] = [None] * (levels + 1)
    need_cols: list[set[int] | None] = [None] * (levels + 1)
    coeff_rows: list[set[int] | None] = [None] * (levels + 1)
    coeff_cols: list[set[int] | None] = [None] * (levels + 1)
    need_rows[1] = set(range(r0, r1))
    need_cols[1] = set(range(c0, c1))
    for level in range(1, levels + 1):
        hh, ww = dims[level - 1]
        rows = _dep1d_indices(need_rows[level], hh)
        cols = _dep1d_indices(need_cols[level], ww)
        coeff_rows[level] = rows
        coeff_cols[level] = cols
        if level < levels:
            lr = (hh + 1) // 2
            lc = (ww + 1) // 2
            need_rows[level + 1] = {r for r in rows if r < lr}
            need_cols[level + 1] = {c for c in cols if c < lc}

    # Bottom-up partial reconstruction.  values holds the sparse values of
    # the level currently being inverted; its LL entries were produced by the
    # preceding deeper iteration.
    values: dict[tuple[int, int], int] = {}
    output: list[list[int]] = [[0] * (c1 - c0) for _ in range(r1 - r0)]

    for level in range(levels, 0, -1):
        hh, ww = dims[level - 1]
        lr = (hh + 1) // 2
        lc = (ww + 1) // 2
        rows = coeff_rows[level]
        cols = coeff_cols[level]
        out_rows = need_rows[level]
        out_cols = need_cols[level]

        # Fetch stored coefficients needed at this level.  The LL quadrant is
        # stored only at the deepest level; otherwise it is the deeper
        # level's reconstructed output already present in values.
        for rr in rows:
            for cc in cols:
                in_ll = rr < lr and cc < lc
                if level < levels and in_ll:
                    continue
                if in_ll:
                    band = "LL"
                elif rr < lr:
                    band = "LH"
                elif cc < lc:
                    band = "HL"
                else:
                    band = "HH"
                raw = read(band, level, rr, cc)
                values[(rr, cc)] = _scalar(
                    raw,
                    f"read({band!r}, level={level}, row={rr}, col={cc})",
                )

        # Strict inverse order: columns first.
        post_columns: dict[tuple[int, int], int] = {}
        for cc in cols:
            col_coeffs = {rr: values[(rr, cc)] for rr in rows}
            produced = _inv1d_partial(
                col_coeffs, hh, out_rows, f"column {cc} (level {level})"
            )
            for rr, v in produced.items():
                post_columns[(rr, cc)] = v

        # Then rows.
        for rr in out_rows:
            row_coeffs = {cc: post_columns[(rr, cc)] for cc in cols}
            produced = _inv1d_partial(
                row_coeffs, ww, out_cols, f"row {rr} (level {level})"
            )
            for cc, v in produced.items():
                if level == 1:
                    output[rr - r0][cc - c0] = v
                else:
                    # Becomes a stored LL coefficient for the level above.
                    values[(rr, cc)] = v

    return tuple(tuple(row) for row in output)


class CoefficientReader:
    """Read-only idwt2d_roi callback over a TransformResult.

    Calling the instance as reader(name, level, row, col) returns the
    recorded coefficient and appends the coefficient-matrix coordinate to
    accessed (a tuple (name, level, row, col)).  Reads outside the declared
    subband rectangle are rejected with ValueError, so the reader doubles as
    a locality guard.
    """

    def __init__(self, result: TransformResult):
        result = _validate_result(result)
        checked = _validate_matrix(result.coefficients, "coefficients")
        if (len(checked), len(checked[0])) != result.shape:
            raise ValueError(
                f"coefficient matrix shape {len(checked)}x{len(checked[0])} "
                f"does not match recorded shape {result.shape[0]}x{result.shape[1]}"
            )
        self._rows = checked
        self._result = result
        self.accessed: list[tuple[str, int, int, int]] = []

    def __call__(self, name: object, level: object, row: object, col: object) -> int:
        if (not isinstance(name, str)
                or name not in ("LL", "LH", "HL", "HH")):
            raise ValueError(f"unknown subband name {name!r}")
        if isinstance(level, bool) or not isinstance(level, int):
            raise TypeError(f"level must be int, got {type(level).__name__}")
        if isinstance(row, bool) or not isinstance(row, int):
            raise TypeError(f"row must be int, got {type(row).__name__}")
        if isinstance(col, bool) or not isinstance(col, int):
            raise TypeError(f"col must be int, got {type(col).__name__}")
        try:
            band = self._result.band(name, level)
        except KeyError:
            raise ValueError(
                f"subband {name!r} level {level} does not exist in this result"
            ) from None
        if name == "LL" and level < self._result.levels:
            raise ValueError(
                f"LL of level {level} is an interior working region, not a "
                f"stored subband; only LL of the deepest level "
                f"{self._result.levels} can be read"
            )
        r0, r1 = band.rows
        c0, c1 = band.cols
        if not (r0 <= row < r1 and c0 <= col < c1):
            raise ValueError(
                f"read outside {name}(level {level}) rectangle rows[{r0},{r1}) "
                f"cols[{c0},{c1}): ({row}, {col})"
            )
        self.accessed.append((name, level, row, col))
        return self._rows[row][col]


def make_reader(result: TransformResult) -> CoefficientReader:
    """Build a CoefficientReader for a TransformResult."""
    return CoefficientReader(result)
