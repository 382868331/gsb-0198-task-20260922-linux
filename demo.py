#!/usr/bin/env python3
"""Fixed-input demonstration for lifting53.

Run:  python demo.py

Part 1 builds a deterministic signed integer image with odd dimensions,
performs a 2-level reversible 5/3 decomposition, prints every subband with
its layout metadata, and verifies bit-exact recovery.  Part 2 genuinely
triggers the documented int63 overflow boundary (the exception is raised by
real arithmetic, not pre-recorded).  Part 3 encodes the coefficients into
the fixed bitstream (sign bits + bit planes + versioned header + CRC),
recovers the original integer image bit-exactly from the complete stream,
then truncates the stream at successive complete bit-plane boundaries and
decodes each prefix into a progressively better approximate image; it also
shows a genuinely damaged header being rejected and a successful CRC catch.
Everything is offline, integer-only and takes well under a second.
"""
from __future__ import annotations

from lifting53 import (
    INT63_MAX,
    BitstreamError,
    IntegerOverflowError,
    decode_bitstream,
    decode_bitstream_partial,
    dwt1d,
    dwt2d,
    encode_bitstream,
    idwt2d,
    max_levels,
    truncate_bitstream,
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
    print()

    print("=" * 68)
    print("Part 3: sign-bit + bit-plane codec, truncation, header checks")
    print("=" * 68)

    # Use the same 7x5 / 2-level result from Part 1 so the printed
    # coefficients can be compared directly with what gets coded.
    result = dwt2d(image, levels=levels)
    stream = encode_bitstream(result)
    img_h, img_w = result.shape
    max_width = stream[11]
    side_bits = int.from_bytes(stream[12:16], "big")
    plane_bytes = int.from_bytes(stream[16:20], "big")
    print(f"encoded {img_h}x{img_w} coefficients, levels={levels}:")
    print(f"  stream length      = {len(stream)} bytes")
    print(f"  header magic/ver   = {stream[0:4]!r} / v{stream[4]}")
    print(f"  max magnitude width= {max_width} bit(s)")
    print(f"  side block         = {side_bits} bits "
          f"(zero flag + sign + width per coefficient)")
    print(f"  plane segments     = {plane_bytes} bytes "
          f"(each bit-plane byte-aligned)")
    print(f"  trailing CRC-32    = {stream[-4:].hex()}")

    decoded = decode_bitstream(stream)
    exact = idwt2d(decoded) == tuple(tuple(r) for r in image)
    print(f"strict decode + idwt reproduces the original exactly: {exact}")
    assert exact

    print()
    print("successive truncation at complete bit-plane boundaries:")
    original = tuple(tuple(r) for r in image)
    coeff_prev = None
    for keep in range(1, max_width + 1):
        cut = truncate_bitstream(stream, keep)
        partial = decode_bitstream_partial(cut)
        approx = idwt2d(partial.result)
        coeff_err = max(
            abs(a - b)
            for ra, rb in zip(result.coefficients, partial.result.coefficients)
            for a, b in zip(ra, rb)
        )
        img_err = max(
            abs(a - b) for ra, rb in zip(original, approx)
            for a, b in zip(ra, rb)
        )
        print(f"  keep top {keep}/{max_width} planes, {len(cut):3d} bytes: "
              f"complete={str(partial.complete):5s} "
              f"max coeff error {coeff_err:4d}, max pixel error {img_err:4d}")
        if coeff_prev is not None:
            assert coeff_err <= coeff_prev
        coeff_prev = coeff_err
    assert coeff_prev == 0

    print()
    # Strict decode refuses the truncated prefix.
    cut1 = truncate_bitstream(stream, 1)
    print(f"strict decode of the {len(cut1)}-byte truncated prefix is "
          "rejected:")
    try:
        decode_bitstream(cut1)
    except BitstreamError as exc:
        print(f"  raised {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("strict decode accepted a truncated stream")

    # Damaged header: bad magic is refused before any decoding happens.
    bad_head = bytearray(stream)
    bad_head[0] = ord("?")
    print("damaged header (bad magic) rejected:")
    try:
        decode_bitstream(bytes(bad_head))
    except BitstreamError as exc:
        print(f"  raised {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("bad magic was not rejected")

    # Truncated header (fewer than 23 bytes) is refused.
    print("incomplete header (10 bytes) rejected:")
    try:
        decode_bitstream(stream[:10])
    except BitstreamError as exc:
        print(f"  raised {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("short header was not rejected")

    # A single flipped payload bit is caught by the CRC (header intact).
    flipped = bytearray(stream)
    flipped[24] ^= 0x01
    print("single flipped payload bit caught by CRC-32:")
    try:
        decode_bitstream(bytes(flipped))
    except BitstreamError as exc:
        print(f"  raised {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("corrupted payload passed the CRC")

    print("\ndemo finished: exact transform round trip, genuinely triggered")
    print("overflow/NaN rejections, exact bitstream recovery, monotone")
    print("bit-plane approximate decoding, and four real codec rejections.")


if __name__ == "__main__":
    main()
