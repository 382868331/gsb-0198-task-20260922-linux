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
    idwt2d_roi,
    idwt2d,
    make_reader,
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

    print()
    print("=" * 68)
    print("Part 3: ROI inverse with a read-only coefficient callback")
    print("=" * 68)
    roi_h, roi_w = 11, 9
    roi_img = make_image(roi_h, roi_w)
    roi_levels = max_levels(roi_h, roi_w)
    roi_result = dwt2d(roi_img, levels=roi_levels)
    full = idwt2d(roi_result)
    rect = (3, 7, 2, 8)
    r0, r1, c0, c1 = rect
    reader = make_reader(roi_result)
    got = idwt2d_roi((roi_h, roi_w), roi_levels, rect, reader)
    expected = tuple(row[c0:c1] for row in full[r0:r1])
    assert got == expected
    total = roi_h * roi_w
    by_band = {}
    for name, level, _r, _c in reader.accessed:
        by_band[(name, level)] = by_band.get((name, level), 0) + 1
    print(f"image {roi_h}x{roi_w}, {roi_levels} levels; ROI = rows [{r0},{r1}), "
          f"cols [{c0},{c1})")
    print("reconstructed ROI pixels:")
    for row in got:
        print("  " + "  ".join(f"{v:4d}" for v in row))
    print(f"coefficients actually read: {len(reader.accessed)} / {total} "
          f"(full image has {total}); per stored subband: {dict(sorted(by_band.items()))}")
    print("each access was within that subband (CoefficientReader rejects "
          "anything else), and bit-exactly equals the independent full "
          "reconstruction cropped to the ROI.")

    # Single corner pixel: the required halo is much larger than the pixel.
    corner_reader = make_reader(roi_result)
    corner = idwt2d_roi((roi_h, roi_w), roi_levels, (0, 1, 0, 1), corner_reader)
    assert corner == ((roi_img[0][0],),)
    print(f"corner pixel (0,0) alone needed {len(corner_reader.accessed)} "
          f"coefficients across the whole pyramid: the required halo is "
          f"necessarily larger than the requested pixel.")

    # Invalid ROI must be rejected before any coefficient is read.
    def no_reads(*_args):
        raise AssertionError("callback must not be called for an invalid ROI")

    try:
        idwt2d_roi((roi_h, roi_w), roi_levels, (0, 1, roi_w, roi_w + 1), no_reads)
    except ValueError as exc:
        print(f"invalid ROI rejected before any read: {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("invalid ROI boundary was not triggered")


if __name__ == "__main__":
    main()
