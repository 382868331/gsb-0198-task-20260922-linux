#!/usr/bin/env python3
"""Fixed-input demonstration for lifting53.

Run:  python demo.py

Part 1 builds a deterministic signed integer image with odd dimensions,
performs a 2-level reversible 5/3 decomposition, prints every subband with
its layout metadata, and verifies bit-exact recovery.  Part 2 genuinely
triggers the documented int63 overflow boundary (the exception is raised by
real arithmetic, not pre-recorded).  Everything is offline, integer-only and
takes well under a second.
"""
from __future__ import annotations

from lifting53 import (
    INT63_MAX,
    IntegerOverflowError,
    dwt1d,
    dwt2d,
    idwt2d,
    max_levels,
)


def make_image(h: int, w: int) -> list[list[int]]:
    """Deterministic signed pattern: a sloped gradient plus a sign flip."""
    return [
        [((i * 11 - j * 7) % 17) - 8 - (2 if (i + j) % 5 == 0 else 0)
         for j in range(w)]
        for i in range(h)
    ]


def print_matrix(title: str, rows) -> None:
    print(title)
    for row in rows:
        print("  " + "  ".join(f"{v:4d}" for v in row))
    print()


def print_subband(result, name: str, level: int) -> None:
    b = result.band(name, level)
    (r0, r1), (c0, c1) = b.rows, b.cols
    if r1 == r0 or c1 == c0:
        print(f"  {name} (L{level}) rows={b.rows} cols={b.cols}  "
              f"-> empty (zero extent)")
        return
    vals = result.extract(name, level)
    flat = [v for row in vals for v in row]
    preview = "  ".join(f"{v:4d}" for v in flat[:8])
    more = " ..." if len(flat) > 8 else ""
    print(f"  {name} (L{level}) rows={b.rows} cols={b.cols} "
          f"shape={r1 - r0}x{c1 - c0}: {preview}{more}")


def main() -> None:
    print("=" * 68)
    print("Part 1: normal result - 2-level reversible 5/3 transform")
    print("=" * 68)

    h, w, levels = 7, 5, 2  # both dimensions odd
    image = make_image(h, w)
    print(f"input: deterministic {h}x{w} signed integer image, "
          f"levels={levels}, max_levels={max_levels(h, w)}")
    print_matrix("original image:", image)

    result = dwt2d(image, levels=levels)
    print_matrix(f"coefficient matrix ({result.shape[0]}x{result.shape[1]}), "
                 "LL recursed in place:", result.coefficients)

    print("subband layout metadata:")
    for lv in range(1, levels + 1):
        layout = result.layouts[lv - 1]
        print(f" level {lv}: working region rows={layout.rows} "
              f"cols={layout.cols}, low block {layout.low_rows}x"
              f"{layout.low_cols}")
        for name in ("LL", "LH", "HL", "HH"):
            print_subband(result, name, lv)
    print()

    recovered = idwt2d(result)
    print_matrix("recovered image:", recovered)

    original = tuple(tuple(r) for r in image)
    bit_exact = recovered == original
    max_abs_diff = max(
        abs(a - b) for ra, rb in zip(original, recovered)
        for a, b in zip(ra, rb)
    )
    print(f"bit-exact recovery: {bit_exact} (element count {h * w}, "
          f"max abs difference {max_abs_diff})")
    assert bit_exact, "round trip must be exact"

    # Supplementary 1xN degenerate case: vertical detail bands are empty.
    one_row = [[3, -7, 2, 0, 5]]
    r1 = dwt2d(one_row, levels=2)
    assert idwt2d(r1) == tuple(map(tuple, one_row))
    print(f"1xN check (1x5, 2 levels): recovered exactly; "
          f"L1 HL/HH empty by construction -> "
          f"{r1.extract('HL', 1) == () and r1.extract('HH', 1) == ()}")
    print()

    print("=" * 68)
    print("Part 2: boundary actually triggered - int63 overflow")
    print("=" * 68)
    signal = [-INT63_MAX, INT63_MAX, -INT63_MAX]
    print(f"call dwt1d({signal!r})")
    print("all three inputs are within +/- (2^63-1), but the prediction")
    print("step adds the two even neighbours: -2*(2^63-1), whose magnitude")
    print("exceeds the limit, so it must raise instead of truncating.")
    try:
        dwt1d(signal)
    except IntegerOverflowError as exc:
        print(f"raised {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("overflow boundary was not triggered")
    print()

    # A second documented rejection: NaN is refused with a locatable error.
    print("secondary boundary - non-finite input rejected:")
    try:
        dwt2d([[1, 2], [3, float("nan")]])
    except ValueError as exc:
        print(f"raised {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("NaN boundary was not triggered")

    print("\ndemo finished: one normal multilevel result with bit-exact")
    print("recovery, and two genuinely triggered documented rejections.")


# ---------------------------------------------------------------------------
# Part 3: ROI-local inverse reconstruction
# ---------------------------------------------------------------------------

def demo_roi() -> None:
    print("=" * 68)
    print("Part 3: ROI inverse - reconstruct a half-open rectangle only")
    print("=" * 68)
    from lifting53 import idwt2d_roi, roi_access_plan, ROIError

    h, w, levels = 7, 5, 2
    image = make_image(h, w)
    result = dwt2d(image, levels=levels)
    full = idwt2d(result)

    rows, cols = (4, 7), (2, 5)  # bottom-right 3x3 half-open rectangle
    plan = roi_access_plan(result, rows, cols)
    accessed = []

    def strict_reader(name, level, r, c):
        key = (name, level, r, c)
        allowed = {(a.name, a.level, a.row, a.col)
                   for a in plan.coordinates()}
        if key not in allowed:
            raise AssertionError(f"unplanned access: {key}")
        band = result.band(name, level)
        accessed.append(key)
        return result.coefficients[band.rows[0] + r][band.cols[0] + c]

    print(f"requested ROI rows={rows} cols={cols} of {h}x{w}, "
          f"levels={levels}")
    print("static dependency plan (bounding boxes, subband-local):")
    for band in plan.bands:
        print(f"  {band.name} L{band.level}: rows={band.rows} cols={band.cols} "
              f"points={len(band.coordinates)}")
    n_forced = len(plan.coordinates())
    roi_pixels = (rows[1] - rows[0]) * (cols[1] - cols[0])
    print(f"forced coefficients: {n_forced} of {h * w} "
          f"(whole pyramid has {h * w}); ROI has {roi_pixels} pixels")

    roi = idwt2d_roi(result, rows, cols, strict_reader)
    expected = tuple(row[cols[0]:cols[1]]
                     for row in full[rows[0]:rows[1]])
    print("ROI output:")
    for row in roi:
        print("  " + "  ".join(f"{v:4d}" for v in row))
    print(f"matches independent full-inverse crop: {roi == expected}; "
          f"callback reads: {len(accessed)} (each forced coordinate once)")
    assert roi == expected
    assert len(accessed) == n_forced == len(set(accessed))

    # Single corner pixel: the halo crosses subbands and levels but is tiny.
    corner_plan = roi_access_plan(result, (0, 1), (0, 1))
    corner_cb = []

    def corner_reader(name, level, r, c):
        corner_cb.append((name, level, r, c))
        band = result.band(name, level)
        return result.coefficients[band.rows[0] + r][band.cols[0] + c]

    pixel = idwt2d_roi(result, (0, 1), (0, 1), corner_reader)
    print(f"corner pixel (0,0)={pixel[0][0]!r} forced "
          f"{len(corner_plan.coordinates())} coefficients across "
          f"{sorted({(a.name, a.level) for a in corner_plan.coordinates()})}")
    print("(the halo is larger than the 1 pixel ROI because clamped lifting "
          "neighbours and the deeper LL block are genuine dependencies)")

    # Invalid ROI: rejected before any coefficient is read.
    def must_not_run(*args):
        raise AssertionError("callback must not be called")

    for bad_rows, bad_cols in [((h, h + 1), (0, 1)), ((0, 1), (0, 0))]:
        try:
            idwt2d_roi(result, bad_rows, bad_cols, must_not_run)
        except ROIError as exc:
            print(f"invalid ROI rows={bad_rows} cols={bad_cols} -> "
                  f"{type(exc).__name__}: {exc}")
        else:
            raise AssertionError("invalid ROI was not rejected")
    print()


if __name__ == "__main__":
    main()
    demo_roi()
