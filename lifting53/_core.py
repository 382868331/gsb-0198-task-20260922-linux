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


def _validate_region(
    matrix: Sequence[Sequence[int]],
    h: int,
    w: int,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
) -> list[list[int]]:
    """Validate values only inside ``[r0,r1) x [c0,c1)``.

    A full ``h x w`` private workspace is returned; cells outside the needed
    rectangle are filled with 0 *without ever reading* the corresponding
    caller-supplied entries (they may belong to unrelated detail bands).
    Row lengths have already been structurally verified by
    :func:`_validate_result`.
    """
    a = [[0] * w for _ in range(h)]
    for r in range(r0, r1):
        row = matrix[r]
        a[r][c0:c1] = [
            _scalar(row[c], f"coefficients[{r}][{c}]")
            for c in range(c0, c1)
        ]
    return a


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

    def shape_at_level(self, level: int) -> tuple[int, int]:
        """Shape of the approximation available at the given level.

        ``level`` runs from 1 (coarsest stored LL) through ``self.levels``
        (the level-1 LL, whose shape equals the original image shape).
        Each side shrinks as ``ceil(s / 2)`` per level, so the returned
        height is the size of the level-``k`` LL block:
        ``(ceil(h / 2**k), ceil(w / 2**k))``.
        """
        if isinstance(level, bool) or not isinstance(level, int):
            raise TypeError("level must be int, bool is not accepted")
        if level < 1 or level > self.levels:
            raise ValueError(
                f"level must be an int in [1, {self.levels}], got {level!r}"
            )
        h, w = self.shape
        for _ in range(level):
            h = (h + 1) // 2
            w = (w + 1) // 2
        return (h, w)

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


def _check_int_pair(pair: object, name: str, level: int) -> tuple[int, int]:
    if (not isinstance(pair, tuple) or len(pair) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in pair)):
        raise ValueError(
            f"malformed layout at level {level}: {name} must be a pair of ints, "
            f"got {pair!r}"
        )
    return pair  # type: ignore[return-value]


def _validate_result(result: object) -> TransformResult:
    if not isinstance(result, TransformResult):
        raise TypeError(
            "idwt2d expects a TransformResult from dwt2d "
            f"(got {type(result).__name__})"
        )
    if (not isinstance(result.shape, tuple) or len(result.shape) != 2):
        raise ValueError(
            f"malformed shape in TransformResult: {result.shape!r}"
        )
    h, w = result.shape
    if (not isinstance(h, int) or isinstance(h, bool)
            or not isinstance(w, int) or isinstance(w, bool)
            or h < 1 or w < 1 or h > MAX_SIDE or w > MAX_SIDE):
        raise ValueError(f"malformed shape in TransformResult: {result.shape}")
    if isinstance(result.levels, bool) or not isinstance(result.levels, int):
        raise ValueError("malformed TransformResult: levels must be an int")
    levels = result.levels
    if levels < 1:
        raise ValueError("malformed TransformResult: levels must be >= 1")

    # --- Structural validation of the coefficient matrix (types/shape only;
    # scalar values are checked later, only inside the region that will be
    # read) -------------------------------------------------------------
    coeffs = result.coefficients
    if not _is_real_sequence(coeffs):
        raise ValueError(
            f"malformed coefficients: must be a tuple of rows "
            f"(got {type(coeffs).__name__})"
        )
    if len(coeffs) != h:
        raise ValueError(
            f"coefficient matrix has {len(coeffs)} rows but recorded shape is "
            f"{h}x{w}"
        )
    for r, row in enumerate(coeffs):
        if not _is_real_sequence(row):
            raise ValueError(
                f"malformed coefficients[{r}]: must be a sequence "
                f"(got {type(row).__name__})"
            )
        if len(row) != w:
            raise ValueError(
                f"coefficients[{r}] has length {len(row)} but recorded width "
                f"is {w}"
            )

    # --- Structural validation of every layout, *before* any arithmetic.
    if not isinstance(result.layouts, tuple) or len(result.layouts) != levels:
        raise ValueError("malformed TransformResult: layouts/levels mismatch")

    prev_r1 = prev_c1 = None
    for idx, layout in enumerate(result.layouts):
        expected_level = idx + 1
        if not isinstance(layout, LevelLayout):
            raise ValueError(
                f"malformed layout at level {expected_level}: not a LevelLayout"
            )
        if layout.level != expected_level:
            raise ValueError(
                f"malformed layout at position {idx}: level field is "
                f"{layout.level}, expected {expected_level}"
            )
        r0, r1 = _check_int_pair(layout.rows, "rows", expected_level)
        c0, c1 = _check_int_pair(layout.cols, "cols", expected_level)
        if not (0 <= r0 < r1 <= h and 0 <= c0 < c1 <= w):
            raise ValueError(
                f"malformed layout at level {expected_level}: "
                f"rows {layout.rows}, cols {layout.cols} outside shape {h}x{w}"
            )
        wh, ww = r1 - r0, c1 - c0
        lr = (wh + 1) // 2
        lc = (ww + 1) // 2
        if (layout.low_rows != lr or layout.low_cols != lc):
            raise ValueError(
                f"malformed layout at level {expected_level}: low block size "
                f"{layout.low_rows}x{layout.low_cols} does not follow the "
                f"ceil-half rule for working region {wh}x{ww}"
            )
        # All pyramid levels are anchored at (0, 0); each deeper working
        # region must be exactly the previous LL rectangle.
        if idx == 0:
            if (r0, c0, r1, c1) != (0, 0, h, w):
                raise ValueError(
                    f"malformed layout at level 1: working region "
                    f"{layout.rows}x{layout.cols} must cover the full "
                    f"{h}x{w} coefficient matrix"
                )
        else:
            if (r0, c0, r1, c1) != (0, 0, prev_r1, prev_c1):
                raise ValueError(
                    f"malformed layout at level {expected_level}: working "
                    f"region {layout.rows}x{layout.cols} is not the LL "
                    f"block {prev_r1}x{prev_c1} of the previous level"
                )
        # Bands must partition the working region into LL/LH/HL/HH with the
        # correct ceil-half split (zero-extent detail bands are allowed for
        # 1xN / Nx1 images).
        expected_bands = {
            ("LL", (r0, r0 + lr), (c0, c0 + lc)),
            ("LH", (r0, r0 + lr), (c0 + lc, c1)),
            ("HL", (r0 + lr, r1), (c0, c0 + lc)),
            ("HH", (r0 + lr, r1), (c0 + lc, c1)),
        }
        got_bands = set()
        if (not isinstance(layout.bands, tuple)
                or len(layout.bands) != 4):
            raise ValueError(
                f"malformed layout at level {expected_level}: "
                f"bands must be a 4-tuple"
            )
        for band in layout.bands:
            if not isinstance(band, Subband):
                raise ValueError(
                    f"malformed layout at level {expected_level}: "
                    f"band is not a Subband"
                )
            br = _check_int_pair(band.rows, "band rows", expected_level)
            bc = _check_int_pair(band.cols, "band cols", expected_level)
            got_bands.add((band.name, br, bc))
        if got_bands != expected_bands:
            raise ValueError(
                f"malformed layout at level {expected_level}: bands do not "
                f"partition the working region with the ceil-half split"
            )
        # Every forward block must be productive (shrink at least one side);
        # a 1x1 working region is the identity and is legal only for the
        # single level-1 transform of a 1x1 image.
        if wh == 1 and ww == 1 and not (idx == 0 and h == 1 and w == 1):
            raise ValueError(
                f"malformed layout at level {expected_level}: a 1x1 working "
                f"region is identity and cannot be a decomposition level"
            )
        prev_r1, prev_c1 = r0 + lr, c0 + lc

    # --- Aggregate subband list must agree with the layouts.
    if not isinstance(result.subbands, tuple):
        raise ValueError("malformed TransformResult: subbands must be a tuple")
    expected_subbands = set()
    for layout in result.layouts:
        for band in layout.bands:
            if band.name != "LL" or band.level == levels:
                expected_subbands.add((band.name, band.level,
                                       band.rows, band.cols))
    got_subbands = {
        (b.name, b.level, b.rows, b.cols)
        if isinstance(b, Subband) else None
        for b in result.subbands
    }
    if None in got_subbands or got_subbands != expected_subbands:
        raise ValueError(
            "malformed TransformResult: subbands list is inconsistent with "
            "the layouts"
        )
    return result


def _resolution_param(resolution: object, levels: int) -> int:
    if isinstance(resolution, bool) or not isinstance(resolution, int):
        raise TypeError(
            f"resolution must be int, got {type(resolution).__name__}"
        )
    if resolution < 0 or resolution > levels:
        raise ValueError(
            f"resolution must be in [0, {levels}] (0 = full original "
            f"resolution), got {resolution}"
        )
    return resolution


def idwt2d(
    result: TransformResult,
    *,
    resolution: int = 0,
) -> tuple[tuple[int, ...], ...]:
    """Inverse of :func:`dwt2d`, optionally stopping at a coarse level.

    ``resolution`` selects the output resolution without ever touching
    unrelated detail data:

    * ``resolution=0`` (default): undo all levels and return the original
      image at full resolution, bit-exact.
    * ``resolution=k`` with ``1 <= k <= result.levels``: return only the
      level-``k`` low-resolution approximation (the level-``k`` LL block
      after levels deeper than ``k`` have been undone).  Its shape is
      ``(ceil(H/2**k), ceil(W/2**k))`` -- obtainable up-front via
      ``result.shape_at_level(k)`` -- and the odd-side boundary handling is
      exactly the 5/3 half-sample symmetric rule used by the full inverse.
      Detail bands LH/HL/HH of levels 1..k and every coefficient outside the
      level-``k`` working region are neither validated nor read.

    All level/layout metadata is structurally validated *before* any
    coefficient is read or any arithmetic is performed, so corrupted level
    information is rejected up front.  Pure integer arithmetic; raises
    :class:`IntegerOverflowError` if a checked intermediate exceeds the
    int63 limit.
    """
    result = _validate_result(result)
    h, w = result.shape
    levels = result.levels
    resolution = _resolution_param(resolution, levels)

    if resolution == 0:
        r0 = c0 = 0
        r1, c1 = h, w
    else:
        # Only this level's LL block is needed: every deeper level lives
        # nested inside it, while this level's own LH/HL/HH detail bands
        # are irrelevant and must not be read.
        target_layout = result.layouts[resolution - 1]
        target_ll = target_layout.band("LL")
        r0, r1 = target_ll.rows
        c0, c1 = target_ll.cols

    # Validate (and copy) only the coefficient values inside the region that
    # the requested reconstruction will actually read.
    a = _validate_region(result.coefficients, h, w, r0, r1, c0, c1)

    # Undo levels in strict reverse order, but only the levels strictly
    # deeper than the requested resolution.  resolution=0 undoes every level.
    for layout in reversed(result.layouts[resolution:]):
        lr0, lr1 = layout.rows
        lc0, lc1 = layout.cols
        _inv_block(a, lr0, lr1, lc0, lc1)

    if resolution > 0:
        # After the deeper levels are undone, the level-``resolution`` LL
        # block holds the requested low-resolution image.
        r0, r1 = target_ll.rows
        c0, c1 = target_ll.cols
    return tuple(
        tuple(a[r][c0:c1]) for r in range(r0, r1)
    )
