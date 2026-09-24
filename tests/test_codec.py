"""Independent tests for the lifting53 embedded bitstream codec.

Expected bytes/values come from sources that never call the codec under test:

1. Hand-computed byte constants (bit-level paper work, including negative
   coefficients and their explicit sign bits).
2. An independent spec-direct packer written in this file using a string-of-bits
   representation (a completely different implementation style than the
   integer-accumulator packer in lifting53/_codec.py).
3. The independent transform reference copied here from the math contract.

Exact round trips are only a supplementary check.
"""
from __future__ import annotations

import math
import unittest
import zlib

from lifting53 import (
    HEADER_MAGIC,
    HEADER_SIZE,
    CorruptBitstreamError,
    LevelLayout,
    TransformResult,
    UnsupportedFormatError,
    decode_bitstream,
    dwt2d,
    encode_bitstream,
    truncate_to_bitplanes,
)

BLOCK_SIDE = 8
COUNT_BITS = 6


# ---------------------------------------------------------------------------
# Independent math reference (spec-direct, no import of the implementation)
# ---------------------------------------------------------------------------

def _clamp(i, n):
    return 0 if i < 0 else n - 1 if i >= n else i


def ref_inv1d(y):
    n = len(y)
    if n == 1:
        return [y[0]]
    lc = (n + 1) // 2
    approx, detail = y[:lc], y[lc:]
    even = []
    for k, s in enumerate(approx):
        dl = detail[_clamp(k - 1, len(detail))]
        dr = detail[_clamp(k, len(detail))]
        even.append(s - math.floor((dl + dr + 2) / 4))
    out = [0] * n
    for k, e in enumerate(even):
        out[2 * k] = e
    for j, d in enumerate(detail):
        el = even[_clamp(j, len(even))]
        er = even[_clamp(j + 1, len(even))]
        out[2 * j + 1] = d + math.floor((el + er) / 2)
    return out


def ref_inv2d(coeffs, levels):
    """Spec-direct 2D inverse: levels reversed, columns then rows."""
    a = [list(row) for row in coeffs]
    h, w = len(a), len(a[0])
    regions = []
    r0 = c0 = 0
    r1, c1 = h, w
    for _ in range(levels):
        regions.append((r0, r1, c0, c1))
        r1 = r0 + (r1 - r0 + 1) // 2
        c1 = c0 + (c1 - c0 + 1) // 2
    for r0, r1, c0, c1 in reversed(regions):
        for c in range(c0, c1):
            col = [a[r][c] for r in range(r0, r1)]
            out = ref_inv1d(col)
            for i, r in enumerate(range(r0, r1)):
                a[r][c] = out[i]
        for r in range(r0, r1):
            seg = a[r][c0:c1]
            a[r][c0:c1] = ref_inv1d(seg)
    return a


def ref_blocks(h, w):
    blocks = []
    for br in range(0, h, BLOCK_SIDE):
        for bc in range(0, w, BLOCK_SIDE):
            blocks.append([
                (r, c)
                for r in range(br, min(br + BLOCK_SIDE, h))
                for c in range(bc, min(bc + BLOCK_SIDE, w))
            ])
    return blocks


# ---------------------------------------------------------------------------
# Independent spec-direct packer (string-of-bits, deliberately different)
# ---------------------------------------------------------------------------

def _bits(value, n):
    return "".join("1" if (value >> (n - 1 - i)) & 1 else "0"
                   for i in range(n))


def ref_pack_payload(coeffs):
    h, w = len(coeffs), len(coeffs[0])
    blocks = ref_blocks(h, w)
    counts = []
    for cells in blocks:
        counts.append(max(abs(coeffs[r][c]).bit_length() for r, c in cells))
    max_planes = max(counts, default=0)

    bitstring = "".join(_bits(k, COUNT_BITS) for k in counts)
    bitstring += "0" * ((-len(bitstring)) % 8)

    significant = [[False] * w for _ in range(h)]
    for p in range(max_planes):
        for bi, cells in enumerate(blocks):
            k = counts[bi]
            if k <= p:
                continue
            bi_idx = k - 1 - p
            for r, c in cells:
                v = coeffs[r][c]
                mb = (abs(v) >> bi_idx) & 1
                bitstring += str(mb)
                if mb and not significant[r][c]:
                    bitstring += "1" if v < 0 else "0"
                    significant[r][c] = True
        bitstring += "0" * ((-len(bitstring)) % 8)

    assert len(bitstring) % 8 == 0
    return bytes(int(bitstring[i:i + 8], 2)
                 for i in range(0, len(bitstring), 8)), counts


def ref_pack_stream(coeffs, h, w, levels, payload):
    body = b"".join((
        HEADER_MAGIC,
        bytes((1, 0)),
        h.to_bytes(2, "big"),
        w.to_bytes(2, "big"),
        bytes((levels, 3)),
        len(payload).to_bytes(4, "big"),
        (zlib.crc32(payload) & 0xFFFFFFFF).to_bytes(4, "big"),
        b"\x00" * 4,
    ))
    return body + (zlib.crc32(body) & 0xFFFFFFFF).to_bytes(4, "big") \
        + payload + (zlib.crc32(payload) & 0xFFFFFFFF).to_bytes(4, "big")


def ref_quantized_coeffs(coeffs, p):
    """Independent truncated-plane reconstruction of the coefficient matrix."""
    h, w = len(coeffs), len(coeffs[0])
    out = [[0] * w for _ in range(h)]
    for cells in ref_blocks(h, w):
        k = max(abs(coeffs[r][c]).bit_length() for r, c in cells)
        shift = max(0, k - p)
        for r, c in cells:
            v = coeffs[r][c]
            m = (abs(v) >> shift) << shift
            out[r][c] = -m if v < 0 else m
    return out


def make_image(h, w, seed_offset=0):
    return [
        [((i * 7 - j * 5 + seed_offset * 3) % 23) - 11 for j in range(w)]
        for i in range(h)
    ]


def payload_of(stream):
    n = int.from_bytes(stream[12:16], "big")
    return stream[HEADER_SIZE:HEADER_SIZE + n]


# ---------------------------------------------------------------------------
# Hand-computed byte constants
# ---------------------------------------------------------------------------

class HandComputedBytes(unittest.TestCase):
    def test_header_layout_1x1(self):
        stream = encode_bitstream(dwt2d([[42]]))
        self.assertEqual(len(stream), HEADER_SIZE + 7 + 4)
        self.assertEqual(stream[0:4], b"L53B")
        self.assertEqual(stream[4], 1)             # version
        self.assertEqual(stream[5], 0)             # flags
        self.assertEqual(int.from_bytes(stream[6:8], "big"), 1)   # height
        self.assertEqual(int.from_bytes(stream[8:10], "big"), 1)  # width
        self.assertEqual(stream[10], 1)            # levels
        self.assertEqual(stream[11], 3)            # block shift
        self.assertEqual(int.from_bytes(stream[12:16], "big"), 7)
        self.assertEqual(stream[20:24], b"\x00\x00\x00\x00")
        self.assertEqual(
            int.from_bytes(stream[24:28], "big"),
            zlib.crc32(stream[0:24]) & 0xFFFFFFFF,
        )
        # trailing payload CRC
        self.assertEqual(
            int.from_bytes(stream[-4:], "big"),
            zlib.crc32(stream[28:-4]) & 0xFFFFFFFF,
        )

    def test_payload_42_positive(self):
        # coeff 42 = 0b101010 -> 6 planes, sign bit 0 at the first one.
        # count: 000110|00 = 18 ; planes: 10|..=80, 00, 80, 00, 80, 00
        stream = encode_bitstream(dwt2d([[42]]))
        self.assertEqual(payload_of(stream),
                         bytes.fromhex("18800080008000"))

    def test_payload_negative_five_explicit_sign(self):
        # coeff -5: magnitude 101 -> 3 planes; first magnitude bit 1 is
        # followed by the explicit sign bit 1.
        stream = encode_bitstream(dwt2d([[-5]]))
        self.assertEqual(payload_of(stream), bytes.fromhex("0cc00080"))
        dec = decode_bitstream(stream)
        self.assertEqual(dec.image, ((-5,),))
        self.assertTrue(dec.exact)

    def test_payload_1x2_two_cells(self):
        # dwt1d([3,7]) = (5,4): magnitudes 101 and 100 -> k=3.
        # count 000011|00 = 0C
        # plane0 MSB: 1,sign0, 1,sign0 = 1010|0000 = A0
        # plane1: 0,0 = 00 ; plane2: 1 (already sig), 0 = 10|.. = 80
        stream = encode_bitstream(dwt2d([[3, 7]]))
        self.assertEqual(payload_of(stream), bytes.fromhex("0ca00080"))

    def test_payload_2x2_all_negative(self):
        # coeffs ((-3,-2),(3,4)): raster cells -3,-2,3,4, k=3.
        # plane0: 0,0,0,1,sign0 -> 00010|000 = 10
        # plane1: -3 ->1,sign1 ; -2 ->1,sign1 ; 3 ->1,sign0 ; 4 ->0
        #        = 11 11 10 0 |pad0 = F8
        # plane2: 1,0,1,0 = 1010|0000 = A0
        stream = encode_bitstream(dwt2d([[-3, -7], [-2, -2]]))
        self.assertEqual(payload_of(stream), bytes.fromhex("0c10f8a0"))
        dec = decode_bitstream(stream)
        self.assertEqual(dec.image, ((-3, -7), (-2, -2)))


# ---------------------------------------------------------------------------
# Cross-check against the independent packer
# ---------------------------------------------------------------------------

CODEC_SHAPES = [
    (1, 1, 1), (1, 2, 1), (1, 3, 1), (1, 8, 3), (1, 17, 3),
    (2, 1, 1), (3, 1, 1), (8, 1, 3),
    (2, 2, 1), (3, 3, 1), (5, 7, 2), (7, 5, 2),
    (8, 8, 3), (9, 9, 2), (13, 16, 2), (17, 13, 3),
    (31, 33, 3), (64, 64, 3), (128, 128, 2),
]


class MatchesIndependentPacker(unittest.TestCase):
    def test_payload_bytes_match_reference(self):
        for h, w, lv in CODEC_SHAPES:
            with self.subTest(shape=(h, w), levels=lv):
                img = make_image(h, w, seed_offset=h + w + lv)
                res = dwt2d(img, levels=lv)
                coeffs = [list(row) for row in res.coefficients]
                ref_payload, _ = ref_pack_payload(coeffs)
                self.assertEqual(payload_of(encode_bitstream(res)), ref_payload)

    def test_full_stream_matches_reference_builder(self):
        for h, w, lv in [(1, 1, 1), (9, 9, 2), (17, 13, 3)]:
            with self.subTest(shape=(h, w), levels=lv):
                img = make_image(h, w)
                res = dwt2d(img, levels=lv)
                ref_payload, _ = ref_pack_payload(
                    [list(r) for r in res.coefficients])
                expected = ref_pack_stream(
                    [list(r) for r in res.coefficients], h, w, lv, ref_payload)
                self.assertEqual(encode_bitstream(res), expected)

    def test_all_negative_payload_matches_reference(self):
        img = [[-(i * 3 + j * 2 + 1) for j in range(11)] for i in range(7)]
        res = dwt2d(img, levels=2)
        ref_payload, _ = ref_pack_payload([list(r) for r in res.coefficients])
        self.assertEqual(payload_of(encode_bitstream(res)), ref_payload)
        dec = decode_bitstream(encode_bitstream(res))
        self.assertEqual(dec.image, tuple(map(tuple, img)))

    def test_exact_round_trip_all_shapes(self):
        for h, w, lv in CODEC_SHAPES:
            with self.subTest(shape=(h, w), levels=lv):
                img = make_image(h, w, seed_offset=lv)
                stream = encode_bitstream(dwt2d(img, levels=lv))
                dec = decode_bitstream(stream)
                self.assertTrue(dec.exact)
                self.assertEqual(dec.planes_decoded, dec.planes_encoded)
                self.assertEqual(dec.image, tuple(map(tuple, img)))
                self.assertFalse(dec.truncated)

    def test_int63_boundary_coefficient(self):
        stream = encode_bitstream(dwt2d([[2**63 - 1]]))
        dec = decode_bitstream(stream)
        self.assertEqual(dec.image, ((2**63 - 1,),))


# ---------------------------------------------------------------------------
# Embedded truncation: complete bit-plane prefixes decode to approximations
# ---------------------------------------------------------------------------

class EmbeddedTruncation(unittest.TestCase):
    def test_plane_prefixes_match_independent_quantization(self):
        img = make_image(11, 9, seed_offset=4)
        res = dwt2d(img, levels=2)
        stream = encode_bitstream(res)
        coeffs = [list(row) for row in res.coefficients]
        total = decode_bitstream(stream).planes_encoded
        self.assertGreaterEqual(total, 2)
        for p in range(total + 1):
            with self.subTest(planes=p):
                prefix = truncate_to_bitplanes(stream, p)
                dec = decode_bitstream(prefix, allow_truncated=True)
                self.assertEqual(dec.planes_decoded, p)
                expected_coeffs = ref_quantized_coeffs(coeffs, p)
                self.assertEqual(
                    [list(r) for r in dec.result.coefficients],
                    expected_coeffs,
                )
                # Approximation independently reconstructed from the spec.
                expected_image = ref_inv2d(expected_coeffs, res.levels)
                self.assertEqual(
                    [list(r) for r in dec.image], expected_image)
                if p < total:
                    self.assertTrue(dec.truncated)
                    self.assertFalse(dec.exact)
                    with self.assertRaises(CorruptBitstreamError):
                        decode_bitstream(prefix)  # strict mode refuses
                else:
                    self.assertFalse(dec.truncated)
                    self.assertTrue(dec.exact)
                    self.assertEqual(dec.image, tuple(map(tuple, img)))
                    # Full-plane "truncation" is the intact CRC-terminated
                    # stream and therefore also decodes in strict mode.
                    self.assertEqual(prefix, stream)
                    self.assertTrue(decode_bitstream(prefix).exact)

    def test_approximation_converges_monotonically(self):
        img = make_image(13, 15)
        orig = tuple(map(tuple, img))
        stream = encode_bitstream(dwt2d(img, levels=2))
        total = decode_bitstream(stream).planes_encoded

        def l1(decoded):
            return sum(abs(a - b)
                       for ra, rb in zip(orig, decoded)
                       for a, b in zip(ra, rb))

        errors = []
        for p in range(total + 1):
            dec = decode_bitstream(truncate_to_bitplanes(stream, p),
                                   allow_truncated=True)
            errors.append(l1(dec.image))
        self.assertEqual(errors[-1], 0)
        for earlier, later in zip(errors, errors[1:]):
            self.assertLessEqual(later, earlier)
        self.assertGreater(errors[0], 0)  # zero-plane image is a real approx.

    def test_zero_plane_prefix_of_all_zero_image_is_intact(self):
        res = dwt2d([[0, 0], [0, 0]])
        stream = encode_bitstream(res)
        dec = decode_bitstream(stream)
        self.assertEqual(dec.planes_encoded, 0)
        self.assertEqual(dec.planes_decoded, 0)
        self.assertTrue(dec.exact)
        prefix = truncate_to_bitplanes(stream, 0)
        self.assertEqual(prefix, stream)

    def test_zero_plane_prefix_gives_zero_approximation(self):
        img = make_image(5, 5)
        stream = encode_bitstream(dwt2d(img))
        dec = decode_bitstream(truncate_to_bitplanes(stream, 0),
                               allow_truncated=True)
        self.assertEqual(dec.planes_decoded, 0)
        self.assertTrue(all(v == 0 for row in dec.result.coefficients
                            for v in row))

    def test_mid_plane_cut_discards_incomplete_plane(self):
        # A cut one byte *inside* a plane section is not a plane boundary;
        # in lenient mode that plane is discarded as if absent.
        img = make_image(13, 7, seed_offset=1)
        stream = encode_bitstream(dwt2d(img, levels=2))
        prefix_2 = truncate_to_bitplanes(stream, 2)
        prefix_3 = truncate_to_bitplanes(stream, 3)
        self.assertGreater(len(prefix_3), len(prefix_2))
        cut = prefix_3[:-1]
        dec = decode_bitstream(cut, allow_truncated=True)
        self.assertEqual(dec.planes_decoded, 2)
        boundary = decode_bitstream(prefix_2, allow_truncated=True)
        self.assertEqual(dec.image, boundary.image)

    def test_1x17_many_blocks_round_trip_and_prefixes(self):
        img = make_image(1, 17)
        stream = encode_bitstream(dwt2d(img, levels=3))
        dec = decode_bitstream(stream)
        self.assertTrue(dec.exact)
        self.assertEqual(dec.image, tuple(map(tuple, img)))
        partial = decode_bitstream(truncate_to_bitplanes(stream, 1),
                                   allow_truncated=True)
        self.assertEqual(partial.planes_decoded, 1)

    def test_prefix_lengths_are_byte_aligned_growth(self):
        stream = encode_bitstream(dwt2d(make_image(9, 9), levels=2))
        total = decode_bitstream(stream).planes_encoded
        lengths = [len(truncate_to_bitplanes(stream, p))
                   for p in range(total + 1)]
        self.assertEqual(lengths[0], HEADER_SIZE +
                         math.ceil(len(ref_blocks(9, 9)) * COUNT_BITS / 8))
        for a, b in zip(lengths, lengths[1:]):
            self.assertGreater(b, a)


# ---------------------------------------------------------------------------
# Corrupt / incomplete headers and payloads are rejected
# ---------------------------------------------------------------------------

def _recompute_header_crc(stream):
    out = bytearray(stream)
    out[24:28] = (zlib.crc32(bytes(out[:24])) & 0xFFFFFFFF).to_bytes(4, "big")
    return bytes(out)


class HeaderRejection(unittest.TestCase):
    def setUp(self):
        self.stream = encode_bitstream(dwt2d(make_image(6, 4)))

    def test_empty_and_short_inputs(self):
        for n in (0, 1, 27):
            with self.subTest(n=n):
                with self.assertRaisesRegex(CorruptBitstreamError,
                                            "incomplete header"):
                    decode_bitstream(b"\x00" * n)

    def test_bad_magic(self):
        bad = bytearray(self.stream)
        bad[0] = ord("X")
        with self.assertRaisesRegex(CorruptBitstreamError, "bad magic"):
            decode_bitstream(bytes(bad))

    def test_header_crc_each_byte(self):
        for i in range(HEADER_SIZE):
            bad = bytearray(self.stream)
            bad[i] ^= 0xFF
            with self.subTest(byte=i):
                with self.assertRaises(CorruptBitstreamError):
                    decode_bitstream(bytes(bad))

    def test_payload_crc_bit_flip_each_region(self):
        n = int.from_bytes(self.stream[12:16], "big")
        for i in range(HEADER_SIZE, HEADER_SIZE + n):
            bad = bytearray(self.stream)
            bad[i] ^= 0x01
            with self.subTest(byte=i):
                with self.assertRaisesRegex(CorruptBitstreamError,
                                            "payload CRC"):
                    decode_bitstream(bytes(bad))

    def test_truncated_strict_mode(self):
        for cut in (1, 4, 5, 10):
            with self.subTest(cut=cut):
                with self.assertRaisesRegex(CorruptBitstreamError,
                                            "incomplete bitstream"):
                    decode_bitstream(self.stream[:-cut])

    def test_truncated_inside_count_section_lenient(self):
        # 9x9 has four 8x8 blocks, so the count section spans 3 bytes;
        # a one-payload-byte prefix cannot carry all four counts.
        stream = encode_bitstream(dwt2d(make_image(9, 9), levels=2))
        with self.assertRaisesRegex(CorruptBitstreamError,
                                    "block count section"):
            decode_bitstream(stream[:HEADER_SIZE + 1],
                             allow_truncated=True)

    def test_payload_complete_but_crc_missing_rejected_even_lenient(self):
        # Having all payload bytes but only part of the 4-byte CRC is never a
        # bit-plane prefix; unverified bytes must not be accepted.
        n = int.from_bytes(self.stream[12:16], "big")
        for cut in (1, 2, 3):
            with self.subTest(crc_bytes_missing=cut):
                with self.assertRaisesRegex(CorruptBitstreamError, "CRC"):
                    decode_bitstream(
                        self.stream[:HEADER_SIZE + n + (4 - cut)],
                        allow_truncated=True,
                    )

    def test_trailing_bytes(self):
        with self.assertRaisesRegex(CorruptBitstreamError, "trailing"):
            decode_bitstream(self.stream + b"\x00")
        with self.assertRaisesRegex(CorruptBitstreamError, "trailing"):
            decode_bitstream(self.stream + b"abc")

    def test_undefined_flags(self):
        bad = bytearray(self.stream)
        bad[5] = 0x01
        bad = _recompute_header_crc(bytes(bad))
        with self.assertRaisesRegex(CorruptBitstreamError, "flag"):
            decode_bitstream(bad)

    def test_nonzero_reserved_bytes(self):
        bad = bytearray(self.stream)
        bad[22] = 0x80
        bad = _recompute_header_crc(bytes(bad))
        with self.assertRaisesRegex(CorruptBitstreamError, "reserved"):
            decode_bitstream(bad)

    def test_unsupported_version(self):
        bad = bytearray(self.stream)
        bad[4] = 2
        bad = _recompute_header_crc(bytes(bad))
        with self.assertRaises(UnsupportedFormatError):
            decode_bitstream(bad)

    def test_unsupported_block_shift(self):
        bad = bytearray(self.stream)
        bad[11] = 4
        bad = _recompute_header_crc(bytes(bad))
        with self.assertRaisesRegex(CorruptBitstreamError, "block_shift"):
            decode_bitstream(bad)

    def test_bad_dimensions(self):
        for h_byte, w_byte in ((0, 4), (4, 0), (129, 1), (1, 129)):
            bad = bytearray(self.stream)
            bad[6], bad[7] = h_byte >> 8, h_byte & 0xFF
            bad[8], bad[9] = w_byte >> 8, w_byte & 0xFF
            bad = _recompute_header_crc(bytes(bad))
            with self.subTest(dims=(h_byte, w_byte)):
                with self.assertRaisesRegex(CorruptBitstreamError,
                                            "dimensions"):
                    decode_bitstream(bad)

    def test_levels_inconsistent_with_shape(self):
        # 2x2 supports only one level; claim three.
        stream = encode_bitstream(dwt2d([[1, 2], [3, 4]]))
        bad = bytearray(stream)
        bad[10] = 3
        bad = _recompute_header_crc(bytes(bad))
        with self.assertRaisesRegex(CorruptBitstreamError, "levels"):
            decode_bitstream(bad)

    def test_declared_length_smaller_than_sections(self):
        # Lie about payload_len (shorter) with consistent CRCs; the coded
        # sections then do not fit, which must be detected.
        n = int.from_bytes(self.stream[12:16], "big")
        payload = self.stream[HEADER_SIZE:HEADER_SIZE + n]
        bad = bytearray(_recompute_header_crc(
            self.stream[:12] + (n - 1).to_bytes(4, "big")
            + self.stream[16:HEADER_SIZE]))
        bad += payload
        bad += (zlib.crc32(payload) & 0xFFFFFFFF).to_bytes(4, "big")
        with self.assertRaises(CorruptBitstreamError):
            decode_bitstream(bytes(bad))

    def test_nonzero_padding_bits_rejected(self):
        # A payload whose last padding bit is 1, CRC and header recomputed.
        n = int.from_bytes(self.stream[12:16], "big")
        payload = bytearray(self.stream[HEADER_SIZE:HEADER_SIZE + n])
        payload[-1] |= 0x01
        body = (
            HEADER_MAGIC + bytes((1, 0))
            + int.from_bytes(self.stream[6:8], "big").to_bytes(2, "big")
            + int.from_bytes(self.stream[8:10], "big").to_bytes(2, "big")
            + bytes((self.stream[10], 3))
            + n.to_bytes(4, "big")
            + (zlib.crc32(payload) & 0xFFFFFFFF).to_bytes(4, "big")
            + b"\x00" * 4
        )
        stream = (body
                  + (zlib.crc32(body) & 0xFFFFFFFF).to_bytes(4, "big")
                  + bytes(payload)
                  + (zlib.crc32(payload) & 0xFFFFFFFF).to_bytes(4, "big"))
        with self.assertRaisesRegex(CorruptBitstreamError, "padding"):
            decode_bitstream(stream)

    def test_wrong_type(self):
        with self.assertRaises(TypeError):
            decode_bitstream("L53B")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            decode_bitstream(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# truncate_to_bitplanes validation
# ---------------------------------------------------------------------------

class TruncateValidation(unittest.TestCase):
    def setUp(self):
        self.stream = encode_bitstream(dwt2d(make_image(7, 6), levels=2))
        self.total = decode_bitstream(self.stream).planes_encoded

    def test_bad_planes_argument(self):
        with self.assertRaises(TypeError):
            truncate_to_bitplanes(self.stream, True)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            truncate_to_bitplanes(self.stream, 1.0)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "planes"):
            truncate_to_bitplanes(self.stream, -1)
        with self.assertRaisesRegex(ValueError, "planes"):
            truncate_to_bitplanes(self.stream, self.total + 1)

    def test_bad_stream_type(self):
        with self.assertRaises(TypeError):
            truncate_to_bitplanes("ab", 1)  # type: ignore[arg-type]

    def test_requires_intact_stream(self):
        with self.assertRaises(CorruptBitstreamError):
            truncate_to_bitplanes(self.stream[:-1], 1)

    def test_rejects_corrupt_stream(self):
        bad = bytearray(self.stream)
        n = int.from_bytes(bad[12:16], "big")
        bad[HEADER_SIZE + n - 1] ^= 0xFF
        with self.assertRaises(CorruptBitstreamError):
            truncate_to_bitplanes(bytes(bad), 1)


# ---------------------------------------------------------------------------
# Encoder input validation / atomicity
# ---------------------------------------------------------------------------

class EncoderValidation(unittest.TestCase):
    def test_encode_rejects_foreign_object(self):
        with self.assertRaises(TypeError):
            encode_bitstream([[1, 2]])  # type: ignore[arg-type]

    def test_encode_rejects_tampered_layout(self):
        res = dwt2d(make_image(4, 4), levels=2)

        # levels/layouts length mismatch is rejected.
        with self.assertRaises(ValueError):
            encode_bitstream(TransformResult(
                coefficients=res.coefficients,
                shape=res.shape,
                levels=2,
                layouts=res.layouts[:1],
                subbands=res.subbands,
            ))

        # A structurally plausible but non-canonical layout region is rejected.
        bad_layout = LevelLayout(
            level=1,
            rows=(0, 3),  # canonical working region for 4x4 is (0, 4)
            cols=(0, 4),
            low_rows=2,
            low_cols=2,
            bands=res.layouts[0].bands,
        )
        with self.assertRaises(ValueError):
            encode_bitstream(TransformResult(
                coefficients=res.coefficients,
                shape=res.shape,
                levels=1,
                layouts=(bad_layout,),
                subbands=res.subbands,
            ))

    def test_encode_rejects_shape_mismatch(self):
        res = dwt2d(make_image(4, 4))
        tampered = TransformResult(
            coefficients=res.coefficients[:3],
            shape=(4, 4),
            levels=res.levels,
            layouts=res.layouts,
            subbands=res.subbands,
        )
        with self.assertRaises(ValueError):
            encode_bitstream(tampered)

    def test_encode_does_not_mutate_result(self):
        res = dwt2d(make_image(5, 3), levels=2)
        snapshot = tuple(tuple(r) for r in res.coefficients)
        encode_bitstream(res)
        self.assertEqual(res.coefficients, snapshot)


if __name__ == "__main__":
    unittest.main(verbosity=2)
