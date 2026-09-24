"""Tests for the sign-bit / bit-plane codec in lifting53.

Expected values come from:

1. A fully hand-computed byte-level example (2x2 image whose coefficients
   are already anchored by the hand tests in test_lifting.py): every header
   byte and every payload bit is written out below.
2. An independent spec-direct decoder written in this file (its own bit
   reader, its own scan-order/layout derivation); it never imports the
   package's codec internals.
3. Independent analytic expectations for bit-plane truncation (dropping
   the low planes is exactly a sign-preserving magnitude masking).

Round trips are only a supplementary check.
"""
from __future__ import annotations

import math
import unittest
import zlib

from lifting53 import (
    MAX_SIDE,
    BitstreamError,
    TransformResult,
    decode_bitstream,
    decode_bitstream_partial,
    dwt2d,
    encode_bitstream,
    idwt2d,
    max_levels,
    truncate_bitstream,
)

HEADER_SIZE = 23
MAGIC = b"L53B"


# ---------------------------------------------------------------------------
# Independent, spec-direct machinery (does NOT use the package codec)
# ---------------------------------------------------------------------------

class IndependentBits:
    def __init__(self, data, start=0):
        self.data = data
        self.pos = start * 8

    def bit(self):
        byte = self.data[self.pos >> 3]
        b = (byte >> (7 - (self.pos & 7))) & 1
        self.pos += 1
        return b

    def bits(self, n):
        v = 0
        for _ in range(n):
            v = (v << 1) | self.bit()
        return v


def independent_layouts(h, w, levels):
    """Spec-direct band rectangles: (name, level, r0, r1, c0, c1)."""
    bands = []
    r0 = c0 = 0
    r1, c1 = h, w
    for level in range(1, levels + 1):
        lr = (r1 - r0 + 1) // 2
        lc = (c1 - c0 + 1) // 2
        bands += [
            ("LL", level, r0, r0 + lr, c0, c0 + lc),
            ("LH", level, r0, r0 + lr, c0 + lc, c1),
            ("HL", level, r0 + lr, r1, c0, c0 + lc),
            ("HH", level, r0 + lr, r1, c0 + lc, c1),
        ]
        r1, c1 = r0 + lr, c0 + lc
    return bands


def independent_scan_positions(h, w, levels):
    bands = independent_layouts(h, w, levels)
    by = {(n, lv): (r0, r1, c0, c1) for n, lv, r0, r1, c0, c1 in bands}
    order = [("LL", levels)]
    for lv in range(levels, 0, -1):
        order += [("HL", lv), ("LH", lv), ("HH", lv)]
    pos = []
    for name, lv in order:
        r0, r1, c0, c1 = by[(name, lv)]
        for r in range(r0, r1):
            for c in range(c0, c1):
                pos.append((r, c))
    assert len(pos) == h * w
    return pos


def independent_decode(stream, keep_planes=None):
    """Decode strictly from the format spec; returns (h, w, levels, matrix).

    Every segment is independently byte aligned.  keep_planes selects a
    truncated prefix (no CRC expected); the cut bytes must contain exactly
    those plane segments.
    """
    assert stream[0:4] == MAGIC
    version = stream[4]
    assert version == 1 and stream[5] == 0
    h = int.from_bytes(stream[6:8], "big")
    w = int.from_bytes(stream[8:10], "big")
    levels = stream[10]
    max_width = stream[11]
    side_bits = int.from_bytes(stream[12:16], "big")
    side_len = (side_bits + 7) // 8

    n = h * w
    rd = IndependentBits(stream[HEADER_SIZE:HEADER_SIZE + side_len])
    signs = [1] * n
    widths = [0] * n
    for i in range(n):
        if rd.bit() == 0:
            continue
        signs[i] = -1 if rd.bit() else 1
        u = rd.bits(6)
        if u < 31:
            e = u
        else:
            e = 31 + rd.bits(6)
        widths[i] = e + 1
    # Side padding must be zero.
    for _ in range(side_len * 8 - side_bits):
        assert rd.bit() == 0

    # Plane segments, MSB first, each padded to a byte boundary.
    retained = max_width if keep_planes is None else keep_planes
    assert 1 <= retained <= max_width
    pos = independent_scan_positions(h, w, levels)
    matrix = [[0] * w for _ in range(h)]
    mags = [0] * n
    seg_offset = HEADER_SIZE + side_len
    seen = 0
    for p in range(max_width - 1, -1, -1):
        active = sum(1 for wd in widths if wd > p)
        seg_len = (active + 7) // 8
        present = seen < retained
        if present:
            prd = IndependentBits(stream[seg_offset:seg_offset + seg_len])
            bits = [prd.bit() for _ in range(active)]
            for _ in range(seg_len * 8 - active):
                assert prd.bit() == 0
            seen += 1
        else:
            bits = [0] * active
        it = iter(bits)
        for i in range(n):
            if widths[i] > p:
                mags[i] |= next(it) << p
        seg_offset += seg_len
    for i, (r, c) in enumerate(pos):
        if widths[i]:
            matrix[r][c] = signs[i] * mags[i]
    return h, w, levels, matrix


def make_image(h, w, seed=0, spread=64):
    """Deterministic signed pattern (independent of the package)."""
    return [
        [((i * 13 - j * 7 + seed * 5) % (2 * spread + 1)) - spread
         for j in range(w)]
        for i in range(h)
    ]


def reshell_with_coefficients(res, matrix):
    """A new TransformResult with the same canonical metadata, new coeffs."""
    return TransformResult(
        coefficients=tuple(tuple(row) for row in matrix),
        shape=res.shape, levels=res.levels,
        layouts=res.layouts, subbands=res.subbands,
    )


# ---------------------------------------------------------------------------
# Hand-computed byte-level example
# ---------------------------------------------------------------------------

class HandComputedBitstream(unittest.TestCase):
    def test_2x2_byte_level_layout(self):
        # Anchored in test_lifting.HandComputed2D.test_2x2:
        #   [[3,7],[2,2]] -> coefficients ((4,2),(-3,-4))
        img = [[3, 7], [2, 2]]
        res = dwt2d(img)
        stream = encode_bitstream(res)

        # --- header, byte for byte ---
        self.assertEqual(stream[0:4], b"L53B")
        self.assertEqual(stream[4], 1)            # version
        self.assertEqual(stream[5], 0)            # flags
        self.assertEqual(int.from_bytes(stream[6:8], "big"), 2)   # height
        self.assertEqual(int.from_bytes(stream[8:10], "big"), 2)  # width
        self.assertEqual(stream[10], 1)           # levels
        self.assertEqual(stream[11], 3)           # max_width (|4|=100)
        self.assertEqual(int.from_bytes(stream[12:16], "big"), 32)  # side bits
        # Three plane segments, each padded to one byte.
        self.assertEqual(int.from_bytes(stream[16:20], "big"), 3)   # plane bytes
        self.assertEqual(stream[20:23], b"\x00\x00\x00")  # reserved

        # --- side segment (byte aligned): scan LL=+4, HL=-3, LH=+2, HH=-4 ---
        # +4: nz 1, sign 0, w=3 -> e=2 -> 10 000010
        # -3: nz 1, sign 1, w=2 -> e=1 -> 11 000001
        # +2: nz 1, sign 0, w=2        -> 10 000001
        # -4: nz 1, sign 1, w=3        -> 11 000010
        self.assertEqual(stream[23:27], bytes.fromhex("82 c1 81 c2"))

        # --- one byte-aligned segment per plane, MSB first ---
        # p2 (w>2): 4->1, -4->1            => "11" + 6 pad => c0
        # p1 (w>1): 4->0,-3->1,2->1,-4->0  => "0110" + pad => 60
        # p0 (w>0): 4->0,-3->1,2->0,-4->0  => "0100" + pad => 40
        self.assertEqual(stream[27:30], bytes.fromhex("c0 60 40"))

        # --- CRC-32 over header + side + planes ---
        body = stream[:HEADER_SIZE + 4 + 3]
        self.assertEqual(
            int.from_bytes(stream[HEADER_SIZE + 7:
                                  HEADER_SIZE + 11], "big"),
            zlib.crc32(body) & 0xFFFFFFFF,
        )
        self.assertEqual(len(stream), HEADER_SIZE + 4 + 3 + 4)

    def test_zero_coefficient_token_is_a_single_zero_bit(self):
        # 1x3 image with a hand-picked zero detail; use the known
        # dwt1d [1,2,3,4] -> (1,3,0,1): image [[1,2,3,4]] level 1.
        res = dwt2d([[1, 2, 3, 4]])
        stream = encode_bitstream(res)
        # scan: LL positions (0,0)=1,(0,1)=3 ; HL empty ;
        #       LH (0,2)=0,(0,3)=1 ; HH empty
        # tokens: +1 -> 10 000000 ; +3 (|3|=11, w=2) -> 10 000001 ;
        #         0  -> 0        ; +1 -> 10 000000
        expect_bits = "10000000" "10000001" "0" "10000000"
        side_bits = int.from_bytes(stream[12:16], "big")
        self.assertEqual(side_bits, len(expect_bits))
        rd = IndependentBits(stream[HEADER_SIZE:])
        got = "".join(str(rd.bit()) for _ in range(side_bits))
        self.assertEqual(got, expect_bits)

    def test_escape_coded_width_hand_values(self):
        # 1x1 identity keeps the pixel itself: pick +2^40 -> width 41.
        v = 1 << 40
        stream = encode_bitstream(dwt2d([[v]], levels=1))
        # nz=1 sign=0; e=40 -> marker 011111 (u=31), then 40-31=9 = 001001
        expect = "10" + "011111" + f"{9:06b}"  # 14 bits total
        rd = IndependentBits(stream[HEADER_SIZE:HEADER_SIZE + 2])
        self.assertEqual("".join(str(rd.bit()) for _ in range(14)), expect)
        # With a single coefficient each plane is a one-byte segment; the
        # MSB plane starts right after the two-byte side segment.
        self.assertEqual((stream[HEADER_SIZE + 2] >> 7) & 1, 1)
        self.assertEqual(decode_bitstream(stream).coefficients, ((v,),))


# ---------------------------------------------------------------------------
# Independent-decoder cross checks
# ---------------------------------------------------------------------------

SHAPES = [
    (1, 1, 1), (1, 2, 1), (1, 3, 1), (2, 1, 1), (3, 1, 1),
    (2, 2, 1), (3, 3, 1), (4, 4, 1), (5, 7, 2), (7, 5, 2),
    (8, 8, 3), (1, 8, 3), (8, 1, 3), (13, 17, 2), (31, 33, 3),
]


class IndependentDecoderCrossCheck(unittest.TestCase):
    def test_full_stream_matches_independent_decoder(self):
        for h, w, lv in SHAPES:
            with self.subTest(shape=(h, w), levels=lv):
                img = make_image(h, w, seed=h + 37 * w + lv)
                res = dwt2d(img, levels=lv)
                stream = encode_bitstream(res)
                ih, iw, ilv, matrix = independent_decode(stream)
                self.assertEqual((ih, iw, ilv), (h, w, lv))
                self.assertEqual(tuple(map(tuple, matrix)), res.coefficients)
                # Package decoder agrees, and inverse is bit-exact.
                self.assertEqual(decode_bitstream(stream).coefficients,
                                 res.coefficients)
                self.assertEqual(idwt2d(decode_bitstream(stream)),
                                 tuple(map(tuple, img)))

    def test_all_zero_image(self):
        res = dwt2d([[0] * 5 for _ in range(4)])
        stream = encode_bitstream(res)
        self.assertEqual(stream[11], 0)  # max_width
        self.assertEqual(int.from_bytes(stream[16:20], "big"), 0)
        self.assertEqual(decode_bitstream(stream).coefficients, res.coefficients)
        pd = decode_bitstream_partial(stream)
        self.assertTrue(pd.complete)
        self.assertEqual(pd.planes_retained, 0)
        self.assertEqual(pd.max_width, 0)
        self.assertEqual(idwt2d(pd.result), tuple(tuple(r) for r in
                         [[0] * 5 for _ in range(4)]))

    def test_all_negative_signs_survive(self):
        img = [[-(i * 5 + j + 1) for j in range(4)] for i in range(3)]
        res = dwt2d(img, levels=2)
        stream = encode_bitstream(res)
        _, _, _, matrix = independent_decode(stream)
        self.assertEqual(tuple(map(tuple, matrix)), res.coefficients)
        self.assertEqual(idwt2d(decode_bitstream(stream)),
                         tuple(map(tuple, img)))

    def test_extreme_magnitude_widths_31_32_63(self):
        # Synthesise canonical coefficient matrices exercising the width
        # prefix boundary (e=30 direct / e=31..62 escaped) without relying
        # on the transform to generate such magnitudes.
        base = dwt2d(make_image(4, 4, seed=1, spread=1))
        picks = {
            (0, 0): (1 << 30),       # width 31, e=30 direct
            (0, 1): (1 << 31),       # width 32, e=31 escaped
            (1, 1): (1 << 40) - 1,   # width 40, e=39 escaped
            (3, 3): (1 << 62),       # width 63, e=62 escaped (max width)
            (2, 2): -((1 << 31) | 1),  # negative escaped
        }
        matrix = [[0] * 4 for _ in range(4)]
        for (r, c), v in picks.items():
            matrix[r][c] = v
        res = reshell_with_coefficients(base, matrix)
        stream = encode_bitstream(res)
        # The package and the independent decoder must recover exactly.
        self.assertEqual(decode_bitstream(stream).coefficients,
                         tuple(map(tuple, matrix)))
        _, _, _, got = independent_decode(stream)
        self.assertEqual(tuple(map(tuple, got)), tuple(map(tuple, matrix)))

    def test_encoding_is_deterministic_and_accepts_bytes_like(self):
        res = dwt2d(make_image(6, 5, seed=7), levels=2)
        self.assertEqual(encode_bitstream(res), encode_bitstream(res))
        stream = encode_bitstream(res)
        for view in (stream, bytearray(stream), memoryview(stream)):
            self.assertEqual(decode_bitstream(view).coefficients,
                             res.coefficients)


# ---------------------------------------------------------------------------
# Bit-plane truncation
# ---------------------------------------------------------------------------

def independent_truncated_matrix(res, keep_planes):
    """Mask low magnitude planes to zero, preserving sign and scan order."""
    h, w = res.shape
    max_width = max(
        (abs(res.coefficients[r][c]).bit_length()
         for r in range(h) for c in range(w)),
        default=0,
    )
    drop = max_width - keep_planes
    matrix = [list(row) for row in res.coefficients]
    if drop:
        for r in range(h):
            for c in range(w):
                v = matrix[r][c]
                if v:
                    m = (abs(v) >> drop) << drop
                    matrix[r][c] = (-m if v < 0 else m)
    return max_width, tuple(tuple(row) for row in matrix)


class TruncatedBitplanes(unittest.TestCase):
    def test_every_plane_prefix_matches_independent_masking(self):
        for h, w, lv in SHAPES:
            img = make_image(h, w, seed=h + w + lv, spread=100)
            res = dwt2d(img, levels=lv)
            stream = encode_bitstream(res)
            max_width = stream[11]
            for k in range(1, max_width + 1):
                with self.subTest(shape=(h, w), levels=lv, keep=k):
                    cut = truncate_bitstream(stream, k)
                    pd = decode_bitstream_partial(cut)
                    self.assertEqual(pd.planes_retained, k)
                    self.assertEqual(pd.max_width, max_width)
                    self.assertEqual(pd.complete, k == max_width)
                    mw, expect = independent_truncated_matrix(res, k)
                    self.assertEqual(pd.result.coefficients, expect)
                    # Independent decoder on the same byte prefix agrees.
                    _, _, _, ind = independent_decode(cut, keep_planes=k)
                    self.assertEqual(tuple(map(tuple, ind)), expect)

    def test_full_plane_count_reconstructs_original_exactly(self):
        img = make_image(9, 7, seed=3, spread=200)
        res = dwt2d(img, levels=3)
        stream = encode_bitstream(res)
        full = truncate_bitstream(stream, stream[11])
        pd = decode_bitstream_partial(full)
        self.assertTrue(pd.complete)
        self.assertEqual(idwt2d(pd.result), tuple(map(tuple, img)))

    def test_strict_decode_rejects_truncated_stream(self):
        res = dwt2d(make_image(5, 5, seed=2), levels=2)
        stream = encode_bitstream(res)
        cut = truncate_bitstream(stream, 1)
        with self.assertRaises(BitstreamError):
            decode_bitstream(cut)

    def test_raw_byte_cut_inside_a_plane_is_rejected(self):
        # With separately byte-aligned plane segments, any payload length
        # that does not end exactly after a whole plane segment is refused.
        res = dwt2d(make_image(6, 6, seed=4, spread=50), levels=1)
        stream = encode_bitstream(res)
        side_len = (int.from_bytes(stream[12:16], "big") + 7) // 8
        # The first (top) plane segment has at least one byte; a prefix
        # ending after the side segment but before its first full byte is
        # impossible (segments are >= 1 byte), so instead add a single
        # extra byte past the first plane boundary only if that lands
        # inside the second segment.  General check: walk lengths and
        # confirm only exact segment boundaries decode.
        from lifting53._bitstream import _scan_bands, _plane_byte_sizes
        vals = [res.coefficients[r][c] for r, c in _scan_bands(res)]
        widths = [0 if v == 0 else abs(v).bit_length() for v in vals]
        sizes = _plane_byte_sizes(widths, stream[11])
        boundaries = []
        off = 0
        for p in range(stream[11] - 1, -1, -1):
            off += sizes[p]
            boundaries.append(HEADER_SIZE + side_len + off)
        for length in range(HEADER_SIZE + side_len + 1, len(stream)):
            if length in boundaries or length == len(stream):
                continue
            with self.subTest(length=length):
                with self.assertRaises(BitstreamError):
                    decode_bitstream_partial(stream[:length])

    def test_truncate_helper_validation(self):
        res = dwt2d(make_image(3, 3, seed=5), levels=1)
        stream = encode_bitstream(res)
        mw = stream[11]
        with self.assertRaises(ValueError):
            truncate_bitstream(stream, 0)
        with self.assertRaises(ValueError):
            truncate_bitstream(stream, mw + 1)
        with self.assertRaises(TypeError):
            truncate_bitstream(stream, True)
        with self.assertRaises(TypeError):
            truncate_bitstream(stream, 1.0)
        zero = encode_bitstream(dwt2d([[0, 0], [0, 0]]))
        with self.assertRaises(ValueError):
            truncate_bitstream(zero, 1)

    def test_partial_rejects_header_only_and_side_block_cut(self):
        res = dwt2d(make_image(4, 4, seed=6), levels=2)
        stream = encode_bitstream(res)
        with self.assertRaises(BitstreamError):
            decode_bitstream_partial(stream[:HEADER_SIZE])
        side_len = (int.from_bytes(stream[12:16], "big") + 7) // 8
        # Cut one byte before the side segment ends.
        with self.assertRaises(BitstreamError):
            decode_bitstream_partial(
                stream[:HEADER_SIZE + max(1, side_len - 1)])


# ---------------------------------------------------------------------------
# Header / stream corruption rejection
# ---------------------------------------------------------------------------

def with_recomputed_crc(head_and_payload: bytearray) -> bytes:
    body = bytes(head_and_payload)
    return body + (zlib.crc32(body) & 0xFFFFFFFF).to_bytes(4, "big")


class CorruptStreams(unittest.TestCase):
    def setUp(self):
        self.res = dwt2d(make_image(5, 6, seed=8, spread=30), levels=2)
        self.stream = encode_bitstream(self.res)

    def test_incomplete_header(self):
        for n in (0, 1, 4, HEADER_SIZE - 1):
            with self.subTest(n=n):
                with self.assertRaisesRegex(BitstreamError, "incomplete header"):
                    decode_bitstream(self.stream[:n])
                with self.assertRaises(BitstreamError):
                    decode_bitstream_partial(self.stream[:n])

    def test_bad_magic(self):
        bad = bytearray(self.stream)
        bad[0] = ord("X")
        with self.assertRaisesRegex(BitstreamError, "magic"):
            decode_bitstream(bytes(bad))

    def test_bad_version(self):
        bad = bytearray(self.stream)
        bad[4] = 2
        with self.assertRaisesRegex(BitstreamError, "version"):
            decode_bitstream(bytes(bad))

    def test_nonzero_flags_and_reserved(self):
        for offset in (5, 20, 21, 22):
            bad = bytearray(self.stream)
            bad[offset] = 1
            with self.subTest(offset=offset):
                with self.assertRaises(BitstreamError):
                    decode_bitstream(bytes(bad))

    def test_shape_out_of_range(self):
        for offset in (6, 8):
            bad = bytearray(self.stream)
            bad[offset:offset + 2] = (0).to_bytes(2, "big")
            with self.subTest(offset=offset):
                with self.assertRaisesRegex(BitstreamError, "shape"):
                    decode_bitstream(bytes(bad))
        bad = bytearray(self.stream)
        bad[6:8] = (MAX_SIDE + 1).to_bytes(2, "big")
        with self.assertRaisesRegex(BitstreamError, "shape"):
            decode_bitstream(bytes(bad))

    def test_levels_bad_or_inconsistent_with_shape(self):
        bad = bytearray(self.stream)
        bad[10] = 0
        with self.assertRaisesRegex(BitstreamError, "levels"):
            decode_bitstream(bytes(bad))
        bad = bytearray(self.stream)
        bad[10] = 200
        with self.assertRaisesRegex(BitstreamError, "levels"):
            decode_bitstream(bytes(bad))
        # 5x6 allows 3 levels; claim 7 (incompatible with the shape).
        bad = bytearray(self.stream)
        bad[10] = 7
        with self.assertRaisesRegex(BitstreamError, "incompatible header"):
            decode_bitstream(bytes(bad))

    def test_max_width_inconsistent_with_tokens(self):
        # Valid CRC, but the claimed max_width disagrees with coded widths.
        body = bytearray(self.stream[:-4])
        body[11] = 20
        fixed = with_recomputed_crc(body)
        with self.assertRaisesRegex(BitstreamError, "max_width mismatch"):
            decode_bitstream(fixed)

    def test_side_bits_inconsistent(self):
        body = bytearray(self.stream[:-4])
        body[12:16] = (int.from_bytes(body[12:16], "big") + 1).to_bytes(4, "big")
        fixed = with_recomputed_crc(body)
        with self.assertRaises(BitstreamError):
            decode_bitstream(fixed)

    def test_plane_bits_inconsistent(self):
        body = bytearray(self.stream[:-4])
        val = int.from_bytes(body[16:20], "big")
        body[16:20] = (val + 8).to_bytes(4, "big")
        fixed = with_recomputed_crc(body)
        with self.assertRaises(BitstreamError):
            decode_bitstream(fixed)

    def test_crc_catches_payload_corruption(self):
        bad = bytearray(self.stream)
        bad[HEADER_SIZE + 2] ^= 0x01
        with self.assertRaisesRegex(BitstreamError, "CRC"):
            decode_bitstream(bytes(bad))
        with self.assertRaisesRegex(BitstreamError, "CRC"):
            decode_bitstream_partial(bytes(bad))

    def test_truncated_payload_strict(self):
        with self.assertRaisesRegex(BitstreamError, "incomplete stream"):
            decode_bitstream(self.stream[:-5])

    def test_trailing_garbage_strict_and_partial(self):
        with self.assertRaisesRegex(BitstreamError, "trailing"):
            decode_bitstream(self.stream + b"\x00")
        with self.assertRaises(BitstreamError):
            decode_bitstream_partial(self.stream + b"\x00")

    def test_wrong_input_type(self):
        with self.assertRaises(TypeError):
            decode_bitstream("not bytes")
        with self.assertRaises(TypeError):
            decode_bitstream(1234)
        with self.assertRaises(TypeError):
            decode_bitstream_partial(["nope"])
        with self.assertRaises(TypeError):
            encode_bitstream([[1, 2], [3, 4]])

    def test_encoder_rejects_tampered_metadata(self):
        # Same coefficient matrix, but a forged shape: must be refused rather
        # than emitting an unusable stream.
        good = encode_bitstream(self.res)
        forged = TransformResult(
            coefficients=self.res.coefficients,
            shape=(self.res.shape[0] + 1, self.res.shape[1]),
            levels=self.res.levels,
            layouts=self.res.layouts,
            subbands=self.res.subbands,
        )
        with self.assertRaises(ValueError):
            encode_bitstream(forged)
        self.assertEqual(encode_bitstream(self.res), good)  # nothing mutated


# ---------------------------------------------------------------------------
# Boundary sizes / supplementary round trips
# ---------------------------------------------------------------------------

class CodecBoundaries(unittest.TestCase):
    def test_max_side_round_trip(self):
        img = [[(i * 31 + j * 17) % 19 - 9 for j in range(MAX_SIDE)]
               for i in range(MAX_SIDE)]
        res = dwt2d(img, levels=2)
        stream = encode_bitstream(res)
        self.assertEqual(decode_bitstream(stream).coefficients, res.coefficients)
        self.assertEqual(idwt2d(decode_bitstream(stream)),
                         tuple(map(tuple, img)))

    def test_truncated_approx_is_decodable_image(self):
        # Every prefix must yield an idwt2d-able result (a defined image);
        # coefficient-wise absolute error must shrink plane by plane (an
        # exact sign-preserving magnitude-masking relation), and the final
        # prefix reconstructs the original exactly.
        img = make_image(11, 9, seed=11, spread=120)
        res = dwt2d(img, levels=2)
        stream = encode_bitstream(res)

        def coeff_max_err(matrix):
            return max(
                abs(a - b)
                for ra, rb in zip(res.coefficients, matrix)
                for a, b in zip(ra, rb)
            )

        prev_err = None
        for k in range(1, stream[11] + 1):
            pd = decode_bitstream_partial(truncate_bitstream(stream, k))
            idwt2d(pd.result)  # the approximate image must always decode
            err = coeff_max_err(pd.result.coefficients)
            if prev_err is not None:
                self.assertLessEqual(err, prev_err)
            prev_err = err
        self.assertEqual(prev_err, 0)
        self.assertEqual(
            idwt2d(decode_bitstream_partial(
                truncate_bitstream(stream, stream[11])).result),
            tuple(map(tuple, img)),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
