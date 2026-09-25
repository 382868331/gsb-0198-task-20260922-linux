"""Tests for the ROI entry point idwt2d_roi.

Expected pixels come from an independent spec-direct full inverse written in
this file (it never imports lifting53); agreement with idwt2d is only a
supplementary check.  A second callback independently derives a conservative
halo per level and rejects any read outside it, verifying locality.
"""
from __future__ import annotations

import math
import unittest

from lifting53 import (
    INT63_MAX,
    IntegerOverflowError,
    dwt2d,
    idwt2d,
    idwt2d_roi,
    make_reader,
    max_levels,
)


def _clamp(i, n):
    return 0 if i < 0 else (n - 1 if i >= n else i)


def ref_inv1d(y):
    n = len(y)
    if n == 1:
        return [y[0]]
    lc = (n + 1) // 2
    hc = n // 2
    approx, detail = y[:lc], y[lc:]
    even = []
    for k in range(lc):
        dl = detail[_clamp(k - 1, hc)]
        dr = detail[_clamp(k, hc)]
        even.append(approx[k] - math.floor((dl + dr + 2) / 4))
    out = [0] * n
    for k in range(lc):
        out[2 * k] = even[k]
    for j in range(hc):
        out[2 * j + 1] = detail[j] + math.floor(
            (even[j] + even[_clamp(j + 1, lc)]) / 2
        )
    return out


def ref_idwt2d(coeffs, h, w, levels):
    a = [list(row) for row in coeffs]
    hh, ww = h, w
    dims = []
    for _ in range(levels):
        dims.append((hh, ww))
        hh, ww = (hh + 1) // 2, (ww + 1) // 2
    for H, W in reversed(dims):
        for c in range(W):
            col = [a[r][c] for r in range(H)]
            out = ref_inv1d(col)
            for i in range(H):
                a[i][c] = out[i]
        for r in range(H):
            a[r][:W] = ref_inv1d(a[r][:W])
    return a


def make_image(h, w):
    return [
        [((i * 13 - j * 7 + (i ^ j) * 3) % 19) - 9 for j in range(w)]
        for i in range(h)
    ]


def halo_indices(a, b, n):
    """Conservative deinterleaved coeff indices for outputs [a, b)."""
    if n == 1:
        return {0}
    lc = (n + 1) // 2
    hc = n // 2
    k_lo = a // 2
    k_hi = min(lc, b // 2 + 1)
    allowed = set(range(k_lo, k_hi))
    d_lo = max(0, k_lo - 1)
    d_hi = min(hc, k_hi + 1)
    allowed |= {lc + j for j in range(d_lo, d_hi)}
    return allowed


def band_of(r, c, H, W):
    lr, lc = (H + 1) // 2, (W + 1) // 2
    if r < lr and c < lc:
        return "LL"
    if r < lr:
        return "LH"
    if c < lc:
        return "HL"
    return "HH"


def allowed_map(shape, levels, roi):
    """Map (level, row, col) -> band name for a conservative read superset."""
    h, w = shape
    r0, r1, c0, c1 = roi
    allowed = {}
    ra, rb = r0, r1
    ca, cb = c0, c1
    H, W = h, w
    for level in range(1, levels + 1):
        rows = halo_indices(ra, rb, H)
        cols = halo_indices(ca, cb, W)
        lr, lc = (H + 1) // 2, (W + 1) // 2
        for r in rows:
            for c in cols:
                band = band_of(r, c, H, W)
                if level < levels and band == "LL":
                    continue
                allowed[(level, r, c)] = band
        ra2 = min(rows) if rows else 0
        rb2 = (max(rows) + 1) if rows else 0
        ca2 = min(cols) if cols else 0
        cb2 = (max(cols) + 1) if cols else 0
        ra, rb = ra2, min(rb2, lr)
        ca, cb = ca2, min(cb2, lc)
        H, W = lr, lc
    return allowed


class GuardedReader:
    """Reject any read that is not in the independently derived halo."""

    def __init__(self, result):
        self._a = result.coefficients
        self.accessed = []

    def arm(self, shape, levels, roi):
        self.allowed = allowed_map(shape, levels, roi)
        return self

    def __call__(self, name, level, row, col):
        key = (level, row, col)
        if key not in self.allowed:
            raise AssertionError(
                f"unexpected local read {name}({level},{row},{col})"
            )
        if self.allowed[key] != name:
            raise AssertionError(
                f"wrong band label {name} for ({level},{row},{col}); "
                f"expected {self.allowed[key]}"
            )
        if key in self.accessed:
            raise AssertionError(f"duplicate read of {key}")
        self.accessed.append(key)
        return self._a[row][col]


class TestROI(unittest.TestCase):
    def assert_roi_matches_reference(self, h, w, levels, roi):
        image = make_image(h, w)
        result = dwt2d(image, levels)
        r0, r1, c0, c1 = roi
        expected_full = ref_idwt2d(result.coefficients, h, w, levels)
        expected = tuple(tuple(row[c0:c1]) for row in expected_full[r0:r1])
        guarded = GuardedReader(result).arm((h, w), levels, roi)
        got = idwt2d_roi((h, w), levels, roi, guarded)
        self.assertEqual(got, expected)
        self.assertEqual(
            len(guarded.accessed), len(set(guarded.accessed))
        )
        if roi != (0, h, 0, w) and max(h, w) >= 8:
            self.assertLess(len(guarded.accessed), h * w)
        full = idwt2d(result)
        self.assertEqual(
            got, tuple(row[c0:c1] for row in full[r0:r1])
        )

    def test_exhaustive_small_shapes_and_rois(self):
        for h in range(1, 6):
            for w in range(1, 6):
                limit = max(1, max_levels(h, w))
                for levels in range(1, limit + 1):
                    rois = [
                        (0, 1, 0, 1),
                        (h - 1, h, w - 1, w),
                        (0, 1, w - 1, w),
                        (h - 1, h, 0, 1),
                        (0, h, 0, w),
                        (0, h, 0, 1),
                        (0, 1, 0, w),
                        (h - 1, h, 0, w),
                        (0, h, w - 1, w),
                        (h // 2, h // 2 + 1, w // 2, w // 2 + 1),
                    ]
                    for roi in rois:
                        self.assert_roi_matches_reference(
                            h, w, levels, roi
                        )

    def test_all_rois_on_mixed_odd_shapes(self):
        for h, w in ((3, 4), (4, 3), (2, 2), (5, 2), (2, 5)):
            image = make_image(h, w)
            levels = max(1, max_levels(h, w))
            result = dwt2d(image, levels)
            ref = ref_idwt2d(result.coefficients, h, w, levels)
            for r0 in range(h):
                for r1 in range(r0 + 1, h + 1):
                    for c0 in range(w):
                        for c1 in range(c0 + 1, w + 1):
                            guarded = GuardedReader(result).arm(
                                (h, w), levels, (r0, r1, c0, c1)
                            )
                            got = idwt2d_roi(
                                (h, w), levels, (r0, r1, c0, c1), guarded
                            )
                            expected = tuple(
                                tuple(row[c0:c1]) for row in ref[r0:r1]
                            )
                            self.assertEqual(got, expected)

    def test_width_or_height_one(self):
        for h, w in ((1, 1), (1, 7), (7, 1), (1, 8), (8, 1), (1, 128)):
            image = make_image(h, w)
            for levels in range(1, max(1, max_levels(h, w)) + 1):
                result = dwt2d(image, levels)
                ref = ref_idwt2d(result.coefficients, h, w, levels)
                rois = [
                    (0, h, 0, w),
                    (0, 1, 0, 1),
                    (h - 1, h, w - 1, w),
                    (0, h, max(0, w // 2 - 1), min(w, w // 2 + 2)),
                ]
                for roi in rois:
                    r0, r1, c0, c1 = roi
                    got = idwt2d_roi(
                        (h, w), levels, roi, make_reader(result)
                    )
                    self.assertEqual(
                        got, tuple(tuple(row[c0:c1]) for row in ref[r0:r1])
                    )

    def test_large_odd_multi_level_corner_halo(self):
        h, w = 127, 128
        image = make_image(h, w)
        levels = max_levels(h, w)
        result = dwt2d(image, levels)
        roi = (0, 1, 0, 1)
        guarded = GuardedReader(result).arm((h, w), levels, roi)
        got = idwt2d_roi((h, w), levels, roi, guarded)
        ref = ref_idwt2d(result.coefficients, h, w, levels)
        self.assertEqual(got, ((ref[0][0],),))
        self.assertLess(len(guarded.accessed), 60)

    def test_reader_rejects_out_of_subband_reads(self):
        result = dwt2d(make_image(6, 6), 2)
        reader = make_reader(result)
        with self.assertRaises(ValueError):
            reader("LH", 1, 0, 0)
        with self.assertRaises(ValueError):
            reader("LL", 1, 0, 0)
        with self.assertRaises(ValueError):
            reader("HH", 2, 9, 9)
        self.assertEqual(reader.accessed, [])

    def test_invalid_roi_before_any_read(self):
        def reject(*_args):
            raise AssertionError("no read must happen for an invalid request")

        bad_rois = [
            (-1, 1, 0, 1),
            (0, 5, 0, 1),
            (2, 2, 0, 1),
            (0, 1, 1, 6),
            (0, 1, 3, 2),
        ]
        for roi in bad_rois:
            with self.assertRaises((TypeError, ValueError)):
                idwt2d_roi((4, 5), 1, roi, reject)
        with self.assertRaises(TypeError):
            idwt2d_roi((4, 5), 1, (0, True, 0, 1), reject)
        with self.assertRaises(TypeError):
            idwt2d_roi((4, 5), 1, (0, 1, 0, 1), None)
        with self.assertRaises(ValueError):
            idwt2d_roi((4, 5), 5, (0, 1, 0, 1), reject)
        with self.assertRaises((TypeError, ValueError)):
            idwt2d_roi((0, 5), 1, (0, 1, 0, 1), reject)

    def test_callback_return_validation(self):
        def bad_non_int(name, level, row, col):
            return 1.5

        with self.assertRaises(TypeError):
            idwt2d_roi((2, 2), 1, (0, 1, 0, 1), bad_non_int)

        def bad_overflow(name, level, row, col):
            return INT63_MAX

        with self.assertRaises(IntegerOverflowError):
            idwt2d_roi((2, 2), 1, (0, 1, 0, 1), bad_overflow)
