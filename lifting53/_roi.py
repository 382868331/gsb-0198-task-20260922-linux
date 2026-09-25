"""ROI-local inverse 5/3 reconstruction.

Given a half-open target pixel rectangle of the fully reconstructed image,
this module derives exactly which wavelet coefficients are forced by the
lifting dependency graph, then performs the inverse level by level reading
each forced coefficient once through a user supplied read-only callback.

Coefficients *outside* the dependency cone are never read and never assumed
to be zero.  Boundary extension uses the same clamped same-kind neighbour
rule as the full transform: when a lifting step references a missing
same-kind neighbour, the nearest valid coefficient of that kind is read, so
the required coordinate set (the "halo") may extend beyond the ROI and
across subband borders.  This is not a full reconstruction followed by a
crop: only forced coordinates are requested from the callback.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ._core import (
    IntegerLiftingError,
    TransformResult,
    _bounded,
    _scalar,
)

#: Signature of a read-only coefficient source:
#: read_coeff(name, level, local_row, local_col) -> int.  row/column are
#: coordinates *inside that subband* (its top-left is (0, 0)).
CoeffCallback = Callable[[str, int, int, int], int]


@dataclass(frozen=True)
class CoeffAccess:
    """One coefficient coordinate forced by an ROI request."""

    name: str
    level: int
    row: int  # subband-local row
    col: int  # subband-local column


@dataclass(frozen=True)
class ROIBandAccess:
    """The exact forced coordinates of one subband for an ROI request.

    ``rows``/``cols`` are the bounding half-open rectangle in subband-local
    coordinates; ``coordinates`` lists every forced point, which need not
    fill that box.  Unneeded bands are absent from the plan altogether.
    """

    name: str
    level: int
    rows: tuple[int, int]
    cols: tuple[int, int]
    coordinates: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ROIPlan:
    """Static dependency analysis of one ROI inverse request.

    Produced by :func:`roi_access_plan`; identical ROI geometry always
    yields an identical plan regardless of coefficient values.
    """

    shape: tuple[int, int]
    levels: int
    rows: tuple[int, int]  # requested half-open interval
    cols: tuple[int, int]
    bands: tuple[ROIBandAccess, ...]

    def coordinates(self) -> frozenset[CoeffAccess]:
        """Set of every (subband, level, local row, local col) forced."""
        out: set[CoeffAccess] = set()
        for band in self.bands:
            for r, c in band.coordinates:
                out.add(CoeffAccess(band.name, band.level, r, c))
        return frozenset(out)


class ROIError(IntegerLiftingError, ValueError):
    """Raised for a malformed ROI rectangle before any coefficient is read."""


# ---------------------------------------------------------------------------
# 1D lifting dependency closure (deinterleaved layout)
# ---------------------------------------------------------------------------

def _coefficients_needed_for_output(n: int, p: int) -> frozenset[int]:
    """Coefficient indices forced by one reconstructed sample of length n.

    Indices use the deinterleaved layout: 0..low_count-1 are approximation
    samples, low_count..n-1 detail samples.  Same-kind boundary clamps are
    included, so the closure already matches the full inverse's extension.
    """
    if n == 1:
        return frozenset({0})
    low = (n + 1) // 2
    high = n // 2
    needed: set[int] = set()

    def detail_index(i: int) -> int:
        if i < 0:
            i = 0
        elif i >= high:
            i = high - 1
        return low + i

    def add_even(k: int) -> None:
        # Undo-update: ev[k] = approx[k] - floor((d[k-1]+d[k]+2)/4)
        needed.add(k)
        needed.add(detail_index(k - 1))
        needed.add(detail_index(k))

    if p % 2 == 0:
        add_even(p // 2)
    else:
        j = p // 2
        # Undo-prediction: odd[j] = detail[j] + floor((ev[j]+ev[j+1_cl])/2)
        needed.add(low + j)
        add_even(j)
        add_even(min(j + 1, low - 1))
    assert all(0 <= i < n for i in needed)
    return frozenset(needed)


def _closure_outputs_to_coeffs(n: int, outputs: set[int]) -> frozenset[int]:
    if n == 1:
        return frozenset({0}) if outputs else frozenset()
    out: set[int] = set()
    for p in outputs:
        out |= _coefficients_needed_for_output(n, p)
    return frozenset(out)


# ---------------------------------------------------------------------------
# Argument validation (completed before any coefficient is read)
# ---------------------------------------------------------------------------

def _check_interval(interval: object, name: str, limit: int) -> tuple[int, int]:
    if not isinstance(interval, tuple) or len(interval) != 2:
        raise ROIError(
            f"{name} must be a (start, end) half-open tuple, got {interval!r}"
        )
    start, end = interval
    if isinstance(start, bool) or not isinstance(start, int):
        raise ROIError(f"{name}[0] must be an int, got {type(start).__name__}")
    if isinstance(end, bool) or not isinstance(end, int):
        raise ROIError(f"{name}[1] must be an int, got {type(end).__name__}")
    if not (0 <= start < end <= limit):
        raise ROIError(
            f"ROI {name}={interval!r} is invalid for extent {limit}; require "
            f"0 <= start < end <= {limit} (half-open, non-empty)"
        )
    return start, end


def _validate_roi_arguments(result: object, rows: object, cols: object
                            ) -> tuple[TransformResult, tuple[int, int],
                                       tuple[int, int]]:
    if not isinstance(result, TransformResult):
        raise TypeError(
            "ROI inverse expects a TransformResult from dwt2d "
            f"(got {type(result).__name__})"
        )
    h, w = result.shape
    r_interval = _check_interval(rows, "rows", h)
    c_interval = _check_interval(cols, "cols", w)
    from ._core import _validate_result

    # Shape/layout validation only.  The ROI path never reads
    # result.coefficients: all coefficient values come from the callback.
    _validate_result(result)
    return result, r_interval, c_interval


# ---------------------------------------------------------------------------
# Per-level dependency planning (top-down through the pyramid)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _LevelNeed:
    layout_index: int
    out_rows: frozenset[int]  # region-relative rows reconstructed here
    out_cols: frozenset[int]  # region-relative columns reconstructed here
    coeff_rows: frozenset[int]  # forced deinterleaved row coordinates
    coeff_cols: frozenset[int]  # forced deinterleaved column coordinates


def _plan_levels(result: TransformResult,
                 rows: tuple[int, int], cols: tuple[int, int]
                 ) -> tuple[_LevelNeed, ...]:
    # The image ROI is this level's output demand; the closure's low-band
    # part becomes the *LL input* demand pushed to the next deeper level.
    deeper_rows = set(range(rows[0], rows[1]))
    deeper_cols = set(range(cols[0], cols[1]))
    plans: list[_LevelNeed] = []
    for index, layout in enumerate(result.layouts):
        rh = layout.rows[1] - layout.rows[0]
        cw = layout.cols[1] - layout.cols[0]
        rel_rows = {r - layout.rows[0] for r in deeper_rows}
        rel_cols = {c - layout.cols[0] for c in deeper_cols}
        coeff_rows = _closure_outputs_to_coeffs(rh, rel_rows)
        coeff_cols = _closure_outputs_to_coeffs(cw, rel_cols)
        plans.append(_LevelNeed(
            index, frozenset(rel_rows), frozenset(rel_cols),
            coeff_rows, coeff_cols,
        ))
        # LL block == next deeper working region.  Pyramid regions share the
        # (0, 0) origin, so forced low-band indices carry over directly.
        deeper_rows = {layout.rows[0] + i for i in coeff_rows
                       if i < layout.low_rows}
        deeper_cols = {layout.cols[0] + j for j in coeff_cols
                       if j < layout.low_cols}
    return tuple(plans)


# ---------------------------------------------------------------------------
# Sparse 1D inverse lifting
# ---------------------------------------------------------------------------

def _inv1d_sparse(get_coeff: Callable[[int], int], n: int,
                  wanted: frozenset[int], ctx: str) -> dict[int, int]:
    """Inverse 1D 5/3 producing only the requested output positions.

    ``get_coeff(i)`` reads deinterleaved coefficient ``i``; only coordinates
    in the precomputed dependency closure of ``wanted`` are requested.
    Every intermediate sum is checked exactly like the full inverse.
    """
    if n == 1:
        v = _scalar(get_coeff(0), f"{ctx}[0]")
        return {0: _bounded(v, f"{ctx}: length-1 copy")}

    low = (n + 1) // 2
    high = n // 2

    even_positions: set[int] = set()
    for p in wanted:
        if p % 2 == 0:
            even_positions.add(p // 2)
        else:
            j = p // 2
            even_positions.add(j)
            even_positions.add(min(j + 1, low - 1))

    def detail(i: int) -> int:
        if i < 0:
            i = 0
        elif i >= high:
            i = high - 1
        return _scalar(get_coeff(low + i), f"{ctx}: detail[{i}]")

    ev: dict[int, int] = {}
    for k in sorted(even_positions):
        approx = _scalar(get_coeff(k), f"{ctx}: approx[{k}]")
        dl = detail(k - 1)
        dr = detail(k)
        total = _bounded(dl + dr, f"{ctx}: detail sum, even {k}")
        total = _bounded(total + 2, f"{ctx}: detail sum + 2, even {k}")
        ev[k] = _bounded(approx - total // 4,
                         f"{ctx}: inverse-update result, even {k}")

    out: dict[int, int] = {}
    for p in sorted(wanted):
        if p % 2 == 0:
            out[p] = ev[p // 2]
        else:
            j = p // 2
            d = _scalar(get_coeff(low + j), f"{ctx}: detail[{j}]")
            left = ev[j]
            right = ev[min(j + 1, low - 1)]
            total = _bounded(left + right,
                             f"{ctx}: even-neighbour sum, odd {j}")
            out[p] = _bounded(d + total // 2,
                              f"{ctx}: inverse-prediction result, odd {j}")
    return out


# ---------------------------------------------------------------------------
# Public dependency analysis
# ---------------------------------------------------------------------------

def _collect_band_coords(result: TransformResult,
                         plans: tuple[_LevelNeed, ...]
                         ) -> dict[tuple[str, int], set[tuple[int, int]]]:
    deepest = result.levels
    collected: dict[tuple[str, int], set[tuple[int, int]]] = {}
    for plan in plans:
        layout = result.layouts[plan.layout_index]
        level = layout.level
        for band in layout.bands:
            if band.name == "LL" and level < deepest:
                continue  # interior LL is produced by the deeper inverse
            r0, r1 = band.rows
            c0, c1 = band.cols
            for rr in plan.coeff_rows:
                gr = layout.rows[0] + rr
                if not (r0 <= gr < r1):
                    continue
                for cc in plan.coeff_cols:
                    gc = layout.cols[0] + cc
                    if c0 <= gc < c1:
                        collected.setdefault((band.name, level), set()).add(
                            (gr - r0, gc - c0)
                        )
    return collected


def roi_access_plan(result: TransformResult, rows: tuple[int, int],
                    cols: tuple[int, int]) -> ROIPlan:
    """Return the exact coefficient coordinates an ROI request will read.

    Pure geometry: raises :class:`ROIError`/TypeError for malformed input
    and never invokes a coefficient callback.
    """
    result, r_iv, c_iv = _validate_roi_arguments(result, rows, cols)
    plans = _plan_levels(result, r_iv, c_iv)
    collected = _collect_band_coords(result, plans)

    band_entries: list[ROIBandAccess] = []
    for layout in result.layouts:
        for band in layout.bands:
            coords = collected.get((band.name, layout.level))
            if not coords:
                continue
            rs = sorted({r for r, _ in coords})
            cs = sorted({c for _, c in coords})
            band_entries.append(ROIBandAccess(
                band.name, layout.level,
                (rs[0], rs[-1] + 1), (cs[0], cs[-1] + 1),
                tuple(sorted(coords)),
            ))
    return ROIPlan(
        shape=result.shape, levels=result.levels,
        rows=r_iv, cols=c_iv, bands=tuple(band_entries),
    )



# ---------------------------------------------------------------------------
# Local reconstruction
# ---------------------------------------------------------------------------

def _subband_name(layout, rr: int, cc: int) -> str:
    low_r = rr < layout.low_rows
    low_c = cc < layout.low_cols
    if low_r and low_c:
        return "LL"
    if low_r:
        return "LH"
    if low_c:
        return "HL"
    return "HH"


def _column_output_positions(n: int, wanted_outputs: frozenset[int]
                             ) -> frozenset[int]:
    """Row positions the column inverse must produce for the row pass.

    The row inverse consumes, for every requested output row, its full
    deinterleaved input row: an even output needs that even intermediate,
    while an odd output needs its own reconstructed odd row plus the two
    bracketing even intermediates (2j and 2*min(j+1, low-1)).  Note odd
    rows are genuine column-inverse outputs (detail plus the bracketing
    even prediction) -- they are *not* raw high-band coefficients.
    """
    if n == 1:
        return frozenset({0})
    low = (n + 1) // 2
    positions: set[int] = set()
    for p in wanted_outputs:
        positions.add(p)
        if p % 2 == 0:
            continue
        j = p // 2
        positions.add(2 * j)
        positions.add(2 * min(j + 1, low - 1))
    return frozenset(positions)


def idwt2d_roi(result: TransformResult, rows: tuple[int, int],
               cols: tuple[int, int], read_coeff: CoeffCallback
               ) -> tuple[tuple[int, ...], ...]:
    """Reconstruct only the pixels inside the half-open ROI rectangle.

    ``rows=(r0, r1)``, ``cols=(c0, c1)`` select image pixels
    ``r0 <= r < r1``, ``c0 <= c < c1``; the returned matrix has shape
    ``(r1-r0, c1-c0)``.  ``read_coeff(name, level, local_row, local_col)``
    is called at most once per forced coefficient (cached within this call)
    and *only* for coordinates in the static dependency cone reported by
    :func:`roi_access_plan`.  Values outside the cone are neither read nor
    zero-filled: the clamped same-kind boundary rule supplies every needed
    neighbour instead.

    Invalid ROI geometry is rejected before the callback is ever invoked.
    Coefficient values go through the same int63 checks as the full inverse,
    so overflow behaviour is identical.
    """
    if not callable(read_coeff):
        raise TypeError(
            "read_coeff must be callable as read_coeff(name, level, row, "
            f"col); got {type(read_coeff).__name__}"
        )
    result, r_iv, c_iv = _validate_roi_arguments(result, rows, cols)
    plans = _plan_levels(result, r_iv, c_iv)

    cache: dict[tuple[str, int, int, int], int] = {}

    def fetch(layout_index: int, name: str, rr: int, cc: int) -> int:
        layout = result.layouts[layout_index]
        band = layout.band(name)
        gr = layout.rows[0] + rr
        gc = layout.cols[0] + cc
        local_r = gr - band.rows[0]
        local_c = gc - band.cols[0]
        key = (name, layout.level, local_r, local_c)
        if key not in cache:
            cache[key] = _scalar(
                read_coeff(name, layout.level, local_r, local_c),
                f"coefficient {name} L{layout.level}[{local_r}][{local_c}]",
            )
        return cache[key]

    # LL values reconstructed at level k+1, consumed as level k's LL input.
    produced_ll: dict[tuple[int, int], int] | None = None

    for plan in reversed(plans):
        layout = result.layouts[plan.layout_index]
        rh = layout.rows[1] - layout.rows[0]
        cw = layout.cols[1] - layout.cols[0]
        is_deepest = plan.layout_index == result.levels - 1

        def source(rr: int, cc: int) -> int:
            name = _subband_name(layout, rr, cc)
            if name == "LL" and not is_deepest:
                assert produced_ll is not None
                return produced_ll[(rr, cc)]
            return fetch(plan.layout_index, name, rr, cc)

        # --- inverse columns first (forward was rows then columns) --------
        # The row pass consumes column-inverse outputs at every requested
        # output row plus the bracketing even rows of odd outputs; the
        # closure guarantees every input they depend on is inside
        # plan.coeff_rows/plan.coeff_cols.
        inter_rows = _column_output_positions(rh, plan.out_rows)
        after_columns: dict[tuple[int, int], int] = {}
        for cc in sorted(plan.coeff_cols):
            vals = _inv1d_sparse(
                lambda rr, _cc=cc: source(rr, _cc),
                rh, inter_rows,
                f"ROI L{layout.level} column {layout.cols[0] + cc}",
            )
            for rr, value in vals.items():
                after_columns[(rr, cc)] = value

        # --- inverse rows -------------------------------------------------
        level_output: dict[tuple[int, int], int] = {}
        for rr in sorted(plan.out_rows):
            def row_get(cc: int, _rr=rr) -> int:
                # Every deinterleaved input of the row pass is a column
                # inverse output (at both even and odd row positions).
                return after_columns[(_rr, cc)]
            vals = _inv1d_sparse(
                row_get, cw, plan.out_cols,
                f"ROI L{layout.level} row {layout.rows[0] + rr}",
            )
            for cc, value in vals.items():
                level_output[(rr, cc)] = value

        if plan.layout_index == 0:
            out: list[tuple[int, ...]] = []
            for gr in range(r_iv[0], r_iv[1]):
                rr = gr - layout.rows[0]
                out.append(tuple(
                    level_output[(rr, gc - layout.cols[0])]
                    for gc in range(c_iv[0], c_iv[1])
                ))
            return tuple(out)

        # The deeper level reconstructs exactly the parent LL working block:
        # its own HL/HH rows are high-frequency *rows of that LL block*, so
        # the whole level_output (not only the deep LL sub-rectangle) is what
        # the shallower level consumes as its LL input.
        produced_ll = dict(level_output)
    raise AssertionError("unreachable: no level plans")  # pragma: no cover
