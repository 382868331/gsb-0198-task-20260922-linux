#!/usr/bin/env python3
"""Fixed-input demonstration for lifting53.

Run:  python demo.py

Part 1 builds a deterministic signed integer image with odd dimensions,
performs a 2-level reversible 5/3 decomposition, prints every subband with
its layout metadata, and verifies bit-exact recovery.  Part 2 genuinely
triggers the documented int63 overflow boundary (the exception is raised by
real arithmetic, not pre-recorded).

Part 3 exercises the embedded bitstream: explicit sign bits plus bit-plane
coding with a fixed CRC-protected header.  It shows the on-disk header, an
exact decode, progressive truncation to complete bit-planes (each prefix
decodes to a genuine approximation whose error shrinks as planes arrive), and
two genuinely triggered rejection paths (a corrupt header and an incomplete
stream).  Everything is offline, integer-only and takes well under a second.
"""
from __future__ import annotations

from lifting53 import (
    HEADER_SIZE,
    INT63_MAX,
    CorruptBitstreamError,
    IntegerOverflowError,
    decode_bitstream,
    dwt1d,
    dwt2d,
    encode_bitstream,
    idwt2d,
    max_levels,
    truncate_to_bitplanes,
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

    # ------------------------------------------------------------------
    print("=" * 68)
    print("Part 3: embedded bitstream - sign bits + bit-plane coding")
    print("=" * 68)
    bh, bw, blevels = 9, 7, 2
    image3 = make_image(bh, bw)
    bit_result = dwt2d(image3, levels=blevels)
    stream = encode_bitstream(bit_result)
    print(f"encoded {bh}x{bw}, {blevels} levels -> {len(stream)} bytes "
          f"(header {HEADER_SIZE} bytes)")
    print(f"magic={stream[:4]!r} version={stream[4]} flags={stream[5]} "
          f"dims={int.from_bytes(stream[6:8], 'big')}x"
          f"{int.from_bytes(stream[8:10], 'big')} levels={stream[10]} "
          f"block_shift={stream[11]}")
    plen = int.from_bytes(stream[12:16], "big")
    print(f"payload_len={plen}  first payload bytes="
          f"{stream[HEADER_SIZE:HEADER_SIZE + 8].hex()}...")

    decoded = decode_bitstream(stream)
    orig3 = tuple(tuple(r) for r in image3)
    print(f"strict decode: exact={decoded.exact} "
          f"planes={decoded.planes_decoded}/{decoded.planes_encoded} "
          f"image_restored={decoded.image == orig3}")
    assert decoded.exact and decoded.image == orig3

    print()
    print("progressive truncation to complete bit-planes:")
    print("  planes  stream_bytes  wrong_pixels  total_abs_error")
    total_planes = decoded.planes_encoded
    prev_err = None
    for p in range(total_planes + 1):
        prefix = truncate_to_bitplanes(stream, p)
        d = decode_bitstream(prefix, allow_truncated=True)
        wrong = sum(1 for ra, rb in zip(orig3, d.image)
                    for a, b in zip(ra, rb) if a != b)
        err = sum(abs(a - b) for ra, rb in zip(orig3, d.image)
                  for a, b in zip(ra, rb))
        tag = " (exact)" if d.exact else ""
        print(f"  {p:>7d}  {len(prefix):>12d}  {wrong:>12d}  "
              f"{err:>15d}{tag}")
        assert d.planes_decoded == p
        if prev_err is not None:
            assert err <= prev_err
        prev_err = err
    assert err == 0

    print()
    print("actual error 1 - strict decode of a truncated stream:")
    partial = truncate_to_bitplanes(stream, total_planes // 2)
    print(f"header declares {plen + 4} payload/CRC bytes but stream is "
          f"only {len(partial)} bytes")
    try:
        decode_bitstream(partial)
    except CorruptBitstreamError as exc:
        print(f"raised {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("truncation rejection was not triggered")

    print()
    print("actual error 2 - corrupt fixed header (CRC catches a byte flip):")
    bad = bytearray(stream)
    bad[7] ^= 0xFF  # flip one width byte
    print(f"flipped header byte 7: width field now reads "
          f"{int.from_bytes(bad[6:8], 'big')}")
    try:
        decode_bitstream(bytes(bad))
    except CorruptBitstreamError as exc:
        print(f"raised {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("corrupt-header rejection was not triggered")

    print()
    print("actual error 3 - payload bit flip defeats CRC on strict decode:")
    bad = bytearray(stream)
    bad[HEADER_SIZE + 1] ^= 0x01
    try:
        decode_bitstream(bytes(bad))
    except CorruptBitstreamError as exc:
        print(f"raised {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("payload corruption rejection was not triggered")

    print("\ndemo finished: one normal multilevel result with bit-exact")
    print("recovery, one overflow boundary, one NaN rejection, and three")
    print("genuinely triggered bitstream error paths plus progressive")
    print("complete-bit-plane approximations.")


if __name__ == "__main__":
    main()
