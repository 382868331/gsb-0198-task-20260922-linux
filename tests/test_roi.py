"""Tests for the ROI-local inverse entry point (idwt2d_roi).

Expected ROI pixels come from an *independent* reference: a complete
inverse reconstruction (written directly from the spec in this file with an
interleaved/clamped formulation) followed by a crop.  Locality is checked
with a strict callback that rejects every access outside the static
dependency plan and records the exact coordinates visited.
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
    roi_access_plan,
)

M = INT63_MAX


# Independent spec-direct reference (never imports lifting53 internals).

def _clamp(i, n):
    return 0 if i < 0 else (n - 1 if i >= n else i)


def ref_inv1d(y):
    n = len(y)
    if n == 1:
        return [y[0]]
    lc = (n + 1) // 2
    hc = n // 2
    approx, detail = y[:lc], y[lc:]
    even = [0] * lc
    for k in range(lc):
        dl = detail[_clamp(k - 1, hc)]
        dr = detail[_clamp(k, hc)]
        even[k] = approx[k] - math.floor((dl + dr + 2) / 4)
    out = [0] * n
    for k in range(lc):
        out[2 * k] = even[k]
    for j in range(hc):
        out[2 * j + 1] = detail[j] + math.floor(
            (even[j] + even[_clamp(j + 1, lc)]) / 2
        )
    return out


def ref_inv2d(coefficients, h, w, levels):
    """Independent full inverse; levels undone deepest-first, columns first."""
    a = [list(row) for row in coefficients]
    rh, rw, r0, c0 = h, w, 0, 0
    regions = []
    for _ in range(levels):
        regions.append((r0, rh, c0, rw))
        rh, rw = (rh + 1) // 2, (rw + 1) // 2
    for br0, brh, bc0, brw in reversed(regions):
        r1, c1 = br0 + brh, bc0 + brw
        for c in range(bc0, c1):
            col = [a[r][c] for r in range(br0, r1)]
            o = ref_inv1d(col)
            for i, r in enumerate(range(br0, r1)):
                a[r][c] = o[i]
        for r in range(br0, r1):
            a[r][bc0:c1] = ref_inv1d(a[r][bc0:c1])
    return a


def make_image(h, w, seed=0):
    return [
        [((i * 13 - j * 7 + seed * 5) % 29) - 14 for j in range(w)]
        for i in range(h)
    ]


class StrictCallback:
    """Read-only source; raises on unplanned/out-of-band access, logs reads."""

    def __init__(self, result, allow=None):
        self.result = result
        self.allow = allow
        self.calls = []

    def __call__(self, name, level, r, c):
        key = (name, level, r, c)
        if self.allow is not None and key not in self.allow:
            raise AssertionError(
                "unplanned coefficient access outside ROI dependency cone: "
                f"{key}"
            )
        band = self.result.band(name, level)
        band_h = band.rows[1] - band.rows[0]
        band_w = band.cols[1] - band.cols[0]
        if not (0 <= r < band_h and 0 <= c < band_w):
            raise AssertionError(f"callback given out-of-band coordinate {key}")
        self.calls.append(key)
        return self.result.coefficients[band.rows[0] + r][band.cols[0] + c]


SHAPES = [
    (1, 1, 1), (1, 2, 1), (1, 3, 1), (1, 5, 2), (1, 8, 3), (1, 17, 3),
    (2, 1, 1), (3, 1, 1), (5, 1, 2), (8, 1, 3), (13, 1, 3),
    (2, 2, 1), (3, 3, 1), (3, 4, 1), (4, 3, 1), (5, 5, 2),
    (7, 5, 2), (8, 8, 3), (9, 9, 3), (16, 13, 3), (33, 17, 3),
    (2, 3, 2), (3, 2, 2), (4, 7, 2), (6, 6, 2),
]


def _keys(plan):
    return {(a.name, a.level, a.row, a.col) for a in plan.coordinates()}


class ROIMatchesIndependentCrop(unittest.TestCase):
    def _check(self, result, rows, cols):
        h, w = result.shape
        plan = roi_access_plan(result, rows, cols)
        cb = StrictCallback(result, allow=_keys(plan))
        got = idwt2d_roi(result, rows, cols, cb)
        ref = ref_inv2d(result.coefficients, h, w, result.levels)
        r0, r1 = rows
        c0, c1 = cols
        expected = tuple(tuple(row[c0:c1]) for row in ref[r0:r1])
        self.assertEqual(got, expected)
        self.assertEqual((len(got), len(got[0])), (r1 - r0, c1 - c0))
        # Each forced coordinate is read at most once (per-request caching
        # absorbs repeated clamp references) and nothing off-plan is read.
        self.assertEqual(len(cb.calls), len(set(cb.calls)))
        self.assertEqual(set(cb.calls), _keys(plan))
        return plan, cb

    def test_single_pixel_every_position(self):
        for h, w, lv in SHAPES:
            image = make_image(h, w, seed=h + 3 * w + lv)
            result = dwt2d(image, levels=lv)
            with self.subTest(shape=(h, w), levels=lv):
                for r in range(h):
                    for c in range(w):
                        self._check(result, (r, r + 1), (c, c + 1))

    def test_all_corners_and_edges(self):
        for h, w, lv in SHAPES:
            result = dwt2d(make_image(h, w, seed=lv), levels=lv)
            rects = [
                (0, 1, 0, 1), (0, 1, w - 1, w),
                (h - 1, h, 0, 1), (h - 1, h, w - 1, w),
                (0, h, 0, 1), (0, h, w - 1, w),
                (0, 1, 0, w), (h - 1, h, 0, w),
                (0, h, 0, w),
            ]
            mid_r, mid_c = h // 2, w // 2
            rects += [
                (0, mid_r + 1, 0, mid_c + 1),
                (mid_r, h, mid_c, w),
                (max(0, mid_r - 1), min(h, mid_r + 2),
                 max(0, mid_c - 1), min(w, mid_c + 2)),
            ]
            for r0, r1, c0, c1 in rects:
                with self.subTest(shape=(h, w), levels=lv,
                                  rect=(r0, r1, c0, c1)):
                    self._check(result, (r0, r1), (c0, c1))

    def test_cross_subband_boundaries(self):
        # 2x2 ROIs stepping through every even/odd and subband boundary.
        for h, w, lv in [(7, 5, 2), (8, 8, 3), (9, 7, 3), (16, 13, 3)]:
            result = dwt2d(make_image(h, w, seed=7), levels=lv)
            for r0 in range(0, h - 1):
                for c0 in range(0, w - 1):
                    with self.subTest(shape=(h, w), anchor=(r0, c0)):
                        self._check(result, (r0, r0 + 2), (c0, c0 + 2))

    def test_full_roi_uses_every_coefficient_once(self):
        for h, w, lv in [(5, 5, 1), (7, 5, 2), (8, 8, 3), (1, 8, 3),
                         (8, 1, 3), (33, 17, 3)]:
            result = dwt2d(make_image(h, w), levels=lv)
            plan, cb = self._check(result, (0, h), (0, w))
            self.assertEqual(len(cb.calls), h * w)
            seen = {(a.name, a.level) for a in plan.bands}
            for layout in result.layouts:
                for band in layout.bands:
                    has_samples = (band.rows[1] > band.rows[0]
                                   and band.cols[1] > band.cols[0])
                    if has_samples and not (band.name == "LL"
                                            and layout.level < lv):
                        self.assertIn((band.name, band.level), seen)

    def test_matches_library_full_inverse_too(self):
        # Supplementary cross-check against the package's own full inverse.
        for h, w, lv in SHAPES:
            result = dwt2d(make_image(h, w, seed=3), levels=lv)
            full = idwt2d(result)
            for rows, cols in [((0, h), (0, w)),
                               ((0, min(2, h)), (w - 1, w)),
                               ((h - 1, h), (0, w))]:
                with self.subTest(shape=(h, w), roi=(rows, cols)):
                    cb = StrictCallback(result)
                    self.assertEqual(
                        idwt2d_roi(result, rows, cols, cb),
                        tuple(row[cols[0]:cols[1]]
                              for row in full[rows[0]:rows[1]]),
                    )


class HaloAndLocality(unittest.TestCase):
    def test_halo_can_exceed_roi_but_is_still_local(self):
        # One top-left pixel in a 7x5 two-level pyramid forces several
        # coefficients via update/predict clamps and the deeper LL block:
        # strictly more than the ROI pixel, but far below the full 35.
        result = dwt2d(make_image(7, 5), levels=2)
        plan = roi_access_plan(result, (0, 1), (0, 1))
        coords = list(plan.coordinates())
        self.assertGreater(len(coords), 1)
        self.assertLess(len(coords), 7 * 5)
        self.assertEqual({a.level for a in coords}, {1, 2})
        self.assertIn("LL", {a.name for a in coords})
        self.assertGreaterEqual(len({a.name for a in coords}), 2)

    def test_strict_callback_rejects_unrelated_access(self):
        # 16x16, 3 levels: the deep block is 2x2 so its 4 coefficients are
        # genuinely forced by clamps (expected halo), but level-1 HH is a
        # 4x4 region and its far corner has nothing to do with pixel (0,0).
        result = dwt2d(make_image(16, 16), levels=3)
        plan = roi_access_plan(result, (0, 1), (0, 1))
        cb = StrictCallback(result, allow=_keys(plan))
        idwt2d_roi(result, (0, 1), (0, 1), cb)
        hh1 = result.band("HH", 1)
        far = ("HH", 1, hh1.rows[1] - hh1.rows[0] - 1,
               hh1.cols[1] - hh1.cols[0] - 1)
        self.assertNotIn(far, cb.calls)
        with self.assertRaisesRegex(AssertionError, "unplanned"):
            cb(*far)

    def test_opposite_corner_coefficient_not_read(self):
        result = dwt2d(make_image(31, 33), levels=1)
        keys = _keys(roi_access_plan(result, (0, 1), (0, 1)))
        hh = result.band("HH", 1)
        self.assertNotIn(
            ("HH", 1, hh.rows[1] - hh.rows[0] - 1,
             hh.cols[1] - hh.cols[0] - 1),
            keys,
        )

    def test_plan_is_pure_geometry(self):
        r1 = dwt2d(make_image(4, 4), levels=2)
        r2 = dwt2d(make_image(4, 4, seed=9), levels=2)
        self.assertEqual(roi_access_plan(r1, (1, 3), (0, 2)),
                         roi_access_plan(r2, (1, 3), (0, 2)))

    def test_cache_absorbs_repeated_clamp_reference(self):
        result = dwt2d([[10, -5, 3, 2]], levels=1)
        cb = StrictCallback(result)
        idwt2d_roi(result, (0, 1), (0, 1), cb)
        # Pixel 0's clamped update references detail[0] twice; the callback
        # sees it only once thanks to per-request caching.
        self.assertEqual(cb.calls.count(("LH", 1, 0, 0)), 1)

    def test_caching_is_local_to_one_request(self):
        result = dwt2d(make_image(5, 5), levels=1)
        cb = StrictCallback(result)
        idwt2d_roi(result, (0, 1), (0, 1), cb)
        first = list(cb.calls)
        idwt2d_roi(result, (0, 1), (0, 1), cb)
        self.assertEqual(cb.calls, first + first)

    def test_callback_values_are_read_only_used(self):
        # Coefficient data must come from the callback, not result's stored
        # matrix: perturbing an approximation coefficient changes the ROI
        # pixel directly (a detail-only +1 may be swallowed by the floor).
        result = dwt2d(make_image(4, 4), levels=1)

        def shifted(name, level, r, c):
            band = result.band(name, level)
            v = result.coefficients[band.rows[0] + r][band.cols[0] + c]
            if (name, level, r, c) == ("LL", 1, 1, 1):
                return v + 1
            return v

        got = idwt2d_roi(result, (2, 3), (2, 3), shifted)
        ref = idwt2d(result)
        self.assertEqual(got, ((ref[2][2] + 1,),))


class ROIErrors(unittest.TestCase):
    def _assert_no_reads(self, result, rows, cols):
        def reader(*args):
            raise AssertionError("callback must not run for invalid ROI")

        with self.assertRaises((ValueError, TypeError)):
            idwt2d_roi(result, rows, cols, reader)

    def test_out_of_range_and_empty_rects_rejected_before_reads(self):
        result = dwt2d(make_image(3, 4))
        bad_rows = [(-1, 1), (0, 4), (2, 2), (3, 4), (0, 0), (1, 0)]
        bad_cols = [(-1, 1), (0, 5), (2, 2), (4, 5), (0, 0), (1, 0)]
        for rows in bad_rows:
            with self.subTest(rows=rows):
                self._assert_no_reads(result, rows, (0, 1))
        for cols in bad_cols:
            with self.subTest(cols=cols):
                self._assert_no_reads(result, (0, 1), cols)

    def test_bad_interval_types_rejected_before_reads(self):
        result = dwt2d(make_image(2, 2))
        for rows in [None, [0, 1], (0, 1.0), ("0", 1), (True, 1), (0,)]:
            with self.subTest(rows=rows):
                self._assert_no_reads(result, rows, (0, 1))

    def test_non_callable_rejected(self):
        result = dwt2d(make_image(2, 2))
        with self.assertRaises(TypeError):
            idwt2d_roi(result, (0, 1), (0, 1), 123)
        with self.assertRaises(TypeError):
            idwt2d_roi(result, (0, 1), (0, 1), None)

    def test_foreign_result_rejected(self):
        def reader(*a):
            raise AssertionError

        with self.assertRaises(TypeError):
            idwt2d_roi([[1, 2]], (0, 1), (0, 1), reader)


class ROIOverflowParity(unittest.TestCase):
    def test_callback_value_outside_int63_rejected(self):
        result = dwt2d([[1, 2], [3, 4]])

        def reader(name, level, r, c):
            if (name, level, r, c) == ("LL", 1, 0, 0):
                return M + 1
            band = result.band(name, level)
            return result.coefficients[band.rows[0] + r][band.cols[0] + c]

        with self.assertRaises(IntegerOverflowError):
            idwt2d_roi(result, (0, 1), (0, 1), reader)

    def test_overflow_agrees_with_full_inverse(self):
        # Stored coefficients individually within int63 whose inverse-update
        # even sample overflows: ev0 = M - floor((-2M+2)/4) exceeds the
        # limit.  Every ROI output depends on ev0, so every ROI raises just
        # like the full inverse.
        from lifting53 import TransformResult

        base = dwt2d([[0, 0, 0]])
        result = TransformResult(
            ((M, M, -M),), base.shape, base.levels,
            base.layouts, base.subbands,
        )
        with self.assertRaises(IntegerOverflowError):
            idwt2d(result)

        def reader(name, level, r, c):
            band = result.band(name, level)
            return result.coefficients[band.rows[0] + r][band.cols[0] + c]

        for rows, cols in [((0, 1), (0, 1)), ((0, 1), (1, 2)),
                           ((0, 1), (2, 3)), ((0, 1), (0, 3))]:
            with self.subTest(roi=(rows, cols)):
                with self.assertRaises(IntegerOverflowError):
                    idwt2d_roi(result, rows, cols, reader)


if __name__ == "__main__":
    unittest.main(verbosity=2)
