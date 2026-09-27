"""Independent tests for lifting53.

Expected values come from two sources that never call the package under test:

1. Hand-computed constants (worked examples on paper, including negatives).
2. A short spec-direct reference written independently in this test file
   (index/clamping formulation instead of the implementation's slicing one).

Exact forward-then-inverse round trips are only a supplementary check.
"""
from __future__ import annotations

import math
import unittest

from lifting53 import (
    INT63_MAX,
    MAX_SIDE,
    IntegerOverflowError,
    TransformResult,
    dwt1d,
    dwt2d,
    idwt1d,
    idwt2d,
    max_levels,
)

M = INT63_MAX


# ---------------------------------------------------------------------------
# Independent reference (written from the task text; does NOT import core)
# ---------------------------------------------------------------------------

def _clamp_index(i: int, n: int) -> int:
    """Nearest valid same-kind sample: clamp to [0, n-1]."""
    if i < 0:
        return 0
    if i >= n:
        return n - 1
    return i


def ref_fwd1d(x):
    """Spec-direct 1D forward, interleaved-index/clamping formulation."""
    n = len(x)
    if n == 1:
        # length 1 identity
        return [x[0]]
    even = [x[2 * k] for k in range((n + 1) // 2)]
    odd = [x[2 * k + 1] for k in range(n // 2)]
    detail = []
    for j, o in enumerate(odd):
        el = even[_clamp_index(j, len(even))]
        er = even[_clamp_index(j + 1, len(even))]
        detail.append(o - math.floor((el + er) / 2))
    approx = []
    for k, e in enumerate(even):
        dl = detail[_clamp_index(k - 1, len(detail))]
        dr = detail[_clamp_index(k, len(detail))]
        approx.append(e + math.floor((dl + dr + 2) / 4))
    return approx + detail


def ref_inv1d(y):
    """Spec-direct 1D inverse (independent reconstruction)."""
    n = len(y)
    if n == 1:
        return [y[0]]
    lc = (n + 1) // 2
    approx, detail = y[:lc], y[lc:]
    even = []
    for k, s in enumerate(approx):
        dl = detail[_clamp_index(k - 1, len(detail))]
        dr = detail[_clamp_index(k, len(detail))]
        even.append(s - math.floor((dl + dr + 2) / 4))
    out = [0] * n
    for k, e in enumerate(even):
        out[2 * k] = e
    for j, d in enumerate(detail):
        el = even[_clamp_index(j, len(even))]
        er = even[_clamp_index(j + 1, len(even))]
        out[2 * j + 1] = d + math.floor((el + er) / 2)
    return out


def ref_fwd2d(image, levels):
    """Spec-direct 2D forward: rows first, then columns, recurse into LL."""
    a = [list(row) for row in image]
    h, w = len(a), len(a[0])
    r0 = c0 = 0
    r1, c1 = h, w
    for _ in range(levels):
        for r in range(r0, r1):
            seg = a[r][c0:c1]
            a[r][c0:c1] = ref_fwd1d(seg)
        for c in range(c0, c1):
            col = [a[r][c] for r in range(r0, r1)]
            out = ref_fwd1d(col)
            for i, r in enumerate(range(r0, r1)):
                a[r][c] = out[i]
        r1 = r0 + (r1 - r0 + 1) // 2
        c1 = c0 + (c1 - c0 + 1) // 2
    return a


def ref_partial_idwt2d(image, levels, undo):
    """Spec-direct partial inverse: forward with the independent reference,
    then undo only the deepest 'undo' levels and crop the coarse image."""
    a = ref_fwd2d(image, levels)
    h, w = len(a), len(a[0])
    regions = []  # regions[j] = working region of level j+1
    r1, c1 = h, w
    for _ in range(levels):
        regions.append((r1, c1))
        r1 = (r1 + 1) // 2
        c1 = (c1 + 1) // 2
    for (r1, c1) in reversed(regions[levels - undo:]):
        for c in range(c1):
            col = [a[r][c] for r in range(r1)]
            out = ref_inv1d(col)
            for i, r in enumerate(range(r1)):
                a[r][c] = out[i]
        for r in range(r1):
            a[r][:c1] = ref_inv1d(a[r][:c1])
    if undo == 0:
        out_h = (regions[-1][0] + 1) // 2
        out_w = (regions[-1][1] + 1) // 2
    else:
        out_h, out_w = regions[levels - undo]
    return [row[:out_w] for row in a[:out_h]]


# ---------------------------------------------------------------------------
# Deterministic sample generation
# ---------------------------------------------------------------------------

def make_image(h, w, seed_offset=0):
    """Small signed integer pattern, fully deterministic."""
    return [
        [((i * 7 - j * 5 + seed_offset * 3) % 23) - 11 for j in range(w)]
        for i in range(h)
    ]


# ---------------------------------------------------------------------------
# Hand-computed constants (worked on paper)
# ---------------------------------------------------------------------------

class HandComputed1D(unittest.TestCase):
    def test_len2(self):
        # [3,7]: d = 7-floor(6/2)=4 ; s = 3+floor(10/4)=5
        self.assertEqual(dwt1d([3, 7]), (5, 4))
        self.assertEqual(idwt1d((5, 4)), (3, 7))

    def test_len3_odd_size(self):
        # [3,7,2]: d = 7-floor(5/2)=5 ; s0 = 3+floor(12/4)=6 ; s1 = 2+3 = 5
        self.assertEqual(dwt1d([3, 7, 2]), (6, 5, 5))
        self.assertEqual(idwt1d((6, 5, 5)), (3, 7, 2))

    def test_len4(self):
        # [1,2,3,4]: d=(0,1); s = 1+floor(2/4)=1, 3+floor(3/4)=3
        self.assertEqual(dwt1d([1, 2, 3, 4]), (1, 3, 0, 1))
        self.assertEqual(idwt1d((1, 3, 0, 1)), (1, 2, 3, 4))

    def test_len5_zeros_details(self):
        self.assertEqual(dwt1d([0, 1, 2, 3, 4]), (0, 2, 4, 0, 0))
        self.assertEqual(idwt1d((0, 2, 4, 0, 0)), (0, 1, 2, 3, 4))

    def test_len1_identity(self):
        self.assertEqual(dwt1d([42]), (42,))
        self.assertEqual(idwt1d((-9,)), (-9,))

    def test_all_negative_len3(self):
        # [-3,-7,-2]: d=-7-floor(-5/2)=-7+3=-4
        # s0=-3+floor(-6/4)=-5 ; s1=-2+floor(-6/4)=-4
        self.assertEqual(dwt1d([-3, -7, -2]), (-5, -4, -4))
        self.assertEqual(idwt1d((-5, -4, -4)), (-3, -7, -2))

    def test_negative_floor_division_semantics(self):
        # floor((-1)/2) must be -1 (truncation toward zero would give 0)
        # [-1,0]: d = 0 - floor(-1/2) = 1 ; s = -1 + floor(4/4) = 0
        self.assertEqual(dwt1d([-1, 0]), (0, 1))
        self.assertEqual(idwt1d((0, 1)), (-1, 0))


class HandComputed2D(unittest.TestCase):
    def test_2x2(self):
        img = [[3, 7], [2, 2]]
        res = dwt2d(img)
        # rows: [3,7]->(5,4), [2,2]->(2,0)
        # col0 [5,2]: d=-3, s=5+floor(-4/4)=4
        # col1 [4,0]: d=-4, s=4+floor(-6/4)=2
        self.assertEqual(res.coefficients, ((4, 2), (-3, -4)))
        self.assertEqual(res.extract("LL", 1), ((4,),))
        self.assertEqual(res.extract("LH", 1), ((2,),))
        self.assertEqual(res.extract("HL", 1), ((-3,),))
        self.assertEqual(res.extract("HH", 1), ((-4,),))
        self.assertEqual(idwt2d(res), tuple(map(tuple, img)))

    def test_2x2_all_negative(self):
        img = [[-3, -7], [-2, -2]]
        res = dwt2d(img)
        # rows: [-3,-7]->(-5,-4), [-2,-2]->(-2,0)
        # col0 [-5,-2]: d=3, s=-3 ; col1 [-4,0]: d=4, s=-2
        self.assertEqual(res.coefficients, ((-3, -2), (3, 4)))
        self.assertEqual(idwt2d(res), tuple(map(tuple, img)))

    def test_3x3_known_values(self):
        # Independently worked with ref formulas in this test's design:
        img = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        expected = ref_fwd2d(img, 1)
        res = dwt2d(img)
        self.assertEqual([list(r) for r in res.coefficients], expected)
        # Explicit hand-anchored values:
        # rows [1,2,3]->(1,3,0), [4,5,6]->(4,6,0), [7,8,9]->(7,9,0)
        # col0 [1,4,7] -> (1,7,0), so the LL column 0 is (1,7)
        self.assertEqual(res.coefficients[0][0], 1)
        self.assertEqual(res.coefficients[1][0], 7)
        self.assertEqual(res.extract("HH", 1), ((0,),))
        self.assertEqual(idwt2d(res), tuple(map(tuple, img)))


# ---------------------------------------------------------------------------
# Independent-reference cross check over many shapes and value ranges
# ---------------------------------------------------------------------------

SHAPES = [
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 7),
    (2, 1), (3, 1), (4, 1), (5, 1),
    (2, 2), (2, 3), (3, 2), (3, 3), (3, 4), (4, 3),
    (4, 4), (5, 5), (5, 7), (7, 5), (8, 8), (9, 9),
    (16, 13), (17, 1), (1, 17), (31, 33),
]


class MatchesIndependentReference(unittest.TestCase):
    def test_forward_matches_reference_all_shapes(self):
        for h, w in SHAPES:
            with self.subTest(shape=(h, w)):
                img = make_image(h, w)
                got = dwt2d(img, levels=1).coefficients
                exp = tuple(map(tuple, ref_fwd2d(img, 1)))
                self.assertEqual(got, exp)

    def test_forward_all_negative_matches_reference(self):
        for h, w in [(2, 2), (3, 3), (1, 5), (5, 1), (7, 4)]:
            with self.subTest(shape=(h, w)):
                img = [[-(i * 3 + j * 2 + 1) for j in range(w)] for i in range(h)]
                got = dwt2d(img, levels=1).coefficients
                exp = tuple(map(tuple, ref_fwd2d(img, 1)))
                self.assertEqual(got, exp)

    def test_multilevel_matches_reference_and_layout(self):
        for h, w, lv in [(5, 5, 2), (8, 8, 3), (1, 8, 3), (8, 1, 3),
                         (7, 5, 2), (16, 16, 4), (13, 1, 3), (33, 17, 3)]:
            with self.subTest(shape=(h, w), levels=lv):
                img = make_image(h, w, seed_offset=lv)
                res = dwt2d(img, levels=lv)
                exp = tuple(map(tuple, ref_fwd2d(img, lv)))
                self.assertEqual(res.coefficients, exp)
                # The deepest LL extent must follow ceil-halving.
                last = res.layouts[-1]
                eh, ew = h, w
                for _ in range(lv):
                    eh, ew = (eh + 1) // 2, (ew + 1) // 2
                ll = last.band("LL")
                self.assertEqual((ll.rows[1] - ll.rows[0],
                                  ll.cols[1] - ll.cols[0]), (eh, ew))
                # Subband partition of the last working region.
                for name in ("LL", "LH", "HL", "HH"):
                    b = last.band(name)
                    self.assertGreaterEqual(b.rows[0], 0)
                    self.assertGreaterEqual(b.cols[0], 0)
                self.assertEqual(idwt2d(res), tuple(map(tuple, img)))

    def test_1d_reference_cross_check(self):
        for n in range(1, 40):
            sig = [((i * 13 + n) % 19) - 9 for i in range(n)]
            with self.subTest(n=n):
                y = dwt1d(sig)
                self.assertEqual(list(y), ref_fwd1d(sig))
                self.assertEqual(list(idwt1d(y)), ref_inv1d(y))
                self.assertEqual(list(idwt1d(y)), sig)


# ---------------------------------------------------------------------------
# Layout / metadata contract
# ---------------------------------------------------------------------------

class LayoutMetadata(unittest.TestCase):
    def test_3x3_extents(self):
        res = dwt2d(make_image(3, 3))
        ll, lh, hl, hh = (res.layouts[0].band(n) for n in ("LL", "LH", "HL", "HH"))
        self.assertEqual((ll.rows, ll.cols), ((0, 2), (0, 2)))
        self.assertEqual((lh.rows, lh.cols), ((0, 2), (2, 3)))
        self.assertEqual((hl.rows, hl.cols), ((2, 3), (0, 2)))
        self.assertEqual((hh.rows, hh.cols), ((2, 3), (2, 3)))
        names = sorted((b.name, b.level) for b in res.subbands)
        self.assertEqual(names, [("HH", 1), ("HL", 1), ("LH", 1), ("LL", 1)])

    def test_1xN_empty_vertical_subbands(self):
        res = dwt2d([[1, 2, 3, 4]])
        ll, lh, hl, hh = (res.layouts[0].band(n) for n in ("LL", "LH", "HL", "HH"))
        self.assertEqual((ll.rows, ll.cols), ((0, 1), (0, 2)))
        self.assertEqual((lh.rows, lh.cols), ((0, 1), (2, 4)))
        self.assertEqual(hl.rows[1], hl.rows[0])  # zero-extent, no samples
        self.assertEqual(hh.rows[1], hh.rows[0])
        self.assertEqual(res.extract("HL", 1), ())
        self.assertEqual(res.extract("HH", 1), ())
        self.assertEqual(res.extract("LL", 1), ((1, 3),))
        self.assertEqual(res.extract("LH", 1), ((0, 1),))

    def test_Nx1_empty_horizontal_subbands(self):
        res = dwt2d([[1], [2], [3], [4]])
        lh = res.layouts[0].band("LH")
        self.assertEqual(lh.cols[1], lh.cols[0])
        self.assertEqual(res.extract("LH", 1), ())
        self.assertEqual(idwt2d(res), ((1,), (2,), (3,), (4,)))

    def test_1x1_identity_level(self):
        res = dwt2d([[42]], levels=1)
        self.assertEqual(res.coefficients, ((42,),))
        self.assertEqual(idwt2d(res), ((42,),))

    def test_multilevel_ll_nesting(self):
        res = dwt2d(make_image(8, 8), levels=3)
        # Level 2 LL sits inside level 1 LL; level 3 LL inside level 2 LL.
        ll1 = res.layouts[0].band("LL")
        ll2 = res.layouts[1].band("LL")
        ll3 = res.layouts[2].band("LL")
        self.assertEqual((ll1.rows, ll1.cols), ((0, 4), (0, 4)))
        self.assertEqual((ll2.rows, ll2.cols), ((0, 2), (0, 2)))
        self.assertEqual((ll3.rows, ll3.cols), ((0, 1), (0, 1)))
        # Aggregate subbands expose only LH/HL/HH of each level + final LL.
        exposed = sorted((b.name, b.level) for b in res.subbands)
        self.assertEqual(exposed, [
            ("HH", 1), ("HH", 2), ("HH", 3),
            ("HL", 1), ("HL", 2), ("HL", 3),
            ("LH", 1), ("LH", 2), ("LH", 3),
            ("LL", 3),
        ])
        # Interior LL blocks are still reachable via band().
        self.assertEqual(res.band("LL", 1).rows, (0, 4))


# ---------------------------------------------------------------------------
# Bounds and max_levels
# ---------------------------------------------------------------------------

class Bounds(unittest.TestCase):
    def test_max_levels_table(self):
        self.assertEqual(max_levels(1, 1), 0)
        self.assertEqual(max_levels(5, 5), 3)   # 5->3->2->1
        self.assertEqual(max_levels(1, 4), 2)   # 1x4->1x2->1x1
        self.assertEqual(max_levels(8, 1), 3)   # 8->4->2->1
        self.assertEqual(max_levels(128, 128), 7)

    def test_max_side_accepted(self):
        img = [[0] * MAX_SIDE for _ in range(MAX_SIDE)]
        res = dwt2d(img)
        self.assertEqual(res.shape, (MAX_SIDE, MAX_SIDE))
        self.assertEqual(idwt2d(res), tuple(map(tuple, img)))

    def test_too_large_rejected(self):
        with self.assertRaisesRegex(ValueError, "maximum"):
            dwt2d([[0] * (MAX_SIDE + 1)])
        with self.assertRaisesRegex(ValueError, "maximum"):
            dwt2d([[0]] * (MAX_SIDE + 1))

    def test_too_many_levels_rejected(self):
        with self.assertRaisesRegex(ValueError, "levels"):
            dwt2d([[1, 2]], levels=2)
        with self.assertRaisesRegex(ValueError, "levels"):
            dwt2d([[1]], levels=2)
        with self.assertRaisesRegex(ValueError, "levels"):
            dwt2d(make_image(5, 5), levels=4)

    def test_levels_zero_or_negative(self):
        with self.assertRaises(ValueError):
            dwt2d([[1, 2]], levels=0)
        with self.assertRaises(ValueError):
            dwt2d([[1, 2]], levels=-1)


# ---------------------------------------------------------------------------
# Input validation / error semantics
# ---------------------------------------------------------------------------

class InputValidation(unittest.TestCase):
    def test_nan_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            dwt2d([[1, float("nan")]])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            dwt1d([1, float("nan")])

    def test_infinity_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            dwt2d([[float("inf")]])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            dwt2d([[float("-inf")]])

    def test_float_rejected(self):
        with self.assertRaisesRegex(TypeError, "integers"):
            dwt2d([[1.5]])
        with self.assertRaisesRegex(TypeError, "integers"):
            dwt1d([1, 2.0])

    def test_bool_rejected_as_value(self):
        with self.assertRaisesRegex(TypeError, "bool"):
            dwt2d([[True, False]])
        with self.assertRaisesRegex(TypeError, "bool"):
            dwt1d([True])

    def test_bool_rejected_as_param(self):
        with self.assertRaisesRegex(TypeError, "levels"):
            dwt2d([[1, 2]], levels=True)
        with self.assertRaisesRegex(TypeError, "height"):
            max_levels(True, 3)

    def test_non_rectangular(self):
        with self.assertRaisesRegex(ValueError, "rectangular"):
            dwt2d([[1, 2, 3], [4, 5]])

    def test_empty(self):
        with self.assertRaises(ValueError):
            dwt2d([])
        with self.assertRaises(ValueError):
            dwt2d([[]])
        with self.assertRaises(ValueError):
            dwt1d([])

    def test_bad_types(self):
        with self.assertRaises(TypeError):
            dwt2d(123)
        with self.assertRaises(TypeError):
            dwt2d([1, 2, 3])  # rows must be sequences
        with self.assertRaises(TypeError):
            dwt2d([["a", "b"]])
        with self.assertRaises(TypeError):
            dwt2d(None)

    def test_error_is_locatable(self):
        with self.assertRaisesRegex(TypeError, r"image\[1\]\[2\]"):
            dwt2d([[1, 2, 3], [4, 5, True]])
        with self.assertRaisesRegex(ValueError, r"signal\[3\]"):
            dwt1d([1, 2, 3, float("nan")])


# ---------------------------------------------------------------------------
# Overflow semantics
# ---------------------------------------------------------------------------

class Overflow(unittest.TestCase):
    def test_boundary_value_accepted(self):
        for v in (M, -M):
            with self.subTest(v=v):
                self.assertEqual(dwt1d([v]), (v,))

    def test_input_above_limit_rejected(self):
        with self.assertRaises(IntegerOverflowError):
            dwt1d([M + 1])
        with self.assertRaises(IntegerOverflowError):
            dwt2d([[-(M + 1)]])

    def test_intermediate_overflow_rejected_not_truncated(self):
        # [-M, M, -M]: all three inputs are in range, but prediction gives
        # d = M - floor(-2M/2) = 2M, beyond the int63 limit.  The checked
        # even-neighbour sum (-2M) trips first; either way it must raise
        # rather than wrap or truncate.
        with self.assertRaises(IntegerOverflowError):
            dwt1d([-M, M, -M])
        with self.assertRaises(IntegerOverflowError):
            dwt2d([[-M, M, -M]])

    def test_overflow_does_not_mutate_input(self):
        img = [[-M, M], [1, 2]]
        snapshot = [row[:] for row in img]
        with self.assertRaises(IntegerOverflowError):
            dwt2d(img)
        self.assertEqual(img, snapshot)

    def test_overflow_is_overflow_error_subclass(self):
        self.assertTrue(issubclass(IntegerOverflowError, OverflowError))


# ---------------------------------------------------------------------------
# Atomicity / immutability / encapsulation
# ---------------------------------------------------------------------------

class MutationAndAtomicity(unittest.TestCase):
    def test_forward_does_not_mutate_input(self):
        img = make_image(5, 7)
        snapshot = [row[:] for row in img]
        dwt2d(img, levels=2)
        self.assertEqual(img, snapshot)

    def test_1d_does_not_mutate_input(self):
        sig = [3, 7, 2]
        dwt1d(sig)
        self.assertEqual(sig, [3, 7, 2])

    def test_result_coefficients_are_immutable_views(self):
        res = dwt2d([[1, 2], [3, 4]])
        with self.assertRaises(TypeError):
            res.coefficients[0] = (9, 9)  # type: ignore[misc]
        with self.assertRaises(TypeError):
            res.coefficients[0][0] = 9  # type: ignore[index]

    def test_idwt_rejects_foreign_object(self):
        with self.assertRaises(TypeError):
            idwt2d([[1, 2], [3, 4]])  # type: ignore[arg-type]

    def test_idwt_shape_mismatch_detected(self):
        res = dwt2d([[1, 2], [3, 4]])
        tampered = TransformResult(
            coefficients=((1, 2, 9), (3, 4, 9)),
            shape=(2, 2),
            levels=res.levels,
            layouts=res.layouts,
            subbands=res.subbands,
        )
        with self.assertRaises(ValueError):
            idwt2d(tampered)


# ---------------------------------------------------------------------------
# Partial (coarse-resolution) reconstruction
# ---------------------------------------------------------------------------

class PartialReconstruction(unittest.TestCase):
    def test_matches_independent_reference(self):
        for h, w, lv in [(7, 5, 2), (8, 8, 3), (5, 5, 3), (1, 8, 3),
                         (8, 1, 3), (9, 9, 2), (16, 13, 3), (3, 3, 1)]:
            img = make_image(h, w, seed_offset=lv)
            res = dwt2d(img, levels=lv)
            for undo in range(0, lv + 1):
                with self.subTest(shape=(h, w), levels=lv, undo=undo):
                    got = [list(r) for r in idwt2d(res, levels=undo)]
                    self.assertEqual(got, ref_partial_idwt2d(img, lv, undo))

    def test_full_levels_default_is_bit_exact(self):
        img = make_image(7, 5)
        res = dwt2d(img, levels=2)
        self.assertEqual(idwt2d(res), idwt2d(res, levels=2))
        self.assertEqual(idwt2d(res), tuple(map(tuple, img)))

    def test_output_shape_rule(self):
        # 7x5, 2 levels: undo 2 -> 7x5, undo 1 -> 4x3, undo 0 -> 2x2.
        res = dwt2d(make_image(7, 5), levels=2)
        for undo, (eh, ew) in [(2, (7, 5)), (1, (4, 3)), (0, (2, 2))]:
            with self.subTest(undo=undo):
                out = idwt2d(res, levels=undo)
                self.assertEqual((len(out), len(out[0])), (eh, ew))

    def test_levels_zero_returns_deepest_ll(self):
        res = dwt2d(make_image(8, 8), levels=3)
        self.assertEqual(idwt2d(res, levels=0), res.extract("LL", 3))

    def test_does_not_read_irrelevant_detail(self):
        # Corrupt (with values that are not even valid integers) the detail
        # subbands of the levels that are NOT undone; a partial inverse must
        # never read them and must still succeed.
        img = make_image(8, 8)
        res = dwt2d(img, levels=2)
        poisoned = [list(row) for row in res.coefficients]
        for name in ("LH", "HL", "HH"):
            b = res.band(name, 1)
            for r in range(*b.rows):
                for c in range(*b.cols):
                    poisoned[r][c] = float("nan")
        tampered = TransformResult(
            coefficients=tuple(tuple(row) for row in poisoned),
            shape=res.shape,
            levels=res.levels,
            layouts=res.layouts,
            subbands=res.subbands,
        )
        got = idwt2d(tampered, levels=1)
        self.assertEqual(got, idwt2d(res, levels=1))
        # The full inverse, which genuinely needs those bands, must reject.
        with self.assertRaises(ValueError):
            idwt2d(tampered)

    def test_levels_param_validation(self):
        res = dwt2d(make_image(4, 4), levels=2)
        with self.assertRaisesRegex(TypeError, "levels"):
            idwt2d(res, levels=True)
        with self.assertRaisesRegex(TypeError, "levels"):
            idwt2d(res, levels=1.5)
        with self.assertRaisesRegex(ValueError, "levels"):
            idwt2d(res, levels=-1)
        with self.assertRaisesRegex(ValueError, "levels"):
            idwt2d(res, levels=3)


# ---------------------------------------------------------------------------
# Corrupted level metadata is rejected before processing
# ---------------------------------------------------------------------------

class CorruptedMetadata(unittest.TestCase):
    def _tampered(self, **kw):
        res = dwt2d(make_image(8, 8), levels=3)
        base = dict(
            coefficients=res.coefficients,
            shape=res.shape,
            levels=res.levels,
            layouts=res.layouts,
            subbands=res.subbands,
        )
        base.update(kw)
        return TransformResult(**base)

    def test_wrong_layout_region_rejected(self):
        res = dwt2d(make_image(8, 8), levels=2)
        bad = list(res.layouts)
        lay = bad[1]
        bad[1] = type(lay)(lay.level, (0, 3), lay.cols, lay.low_rows,
                           lay.low_cols, lay.bands)
        with self.assertRaisesRegex(ValueError, "layouts"):
            idwt2d(self._tampered(layouts=tuple(bad)))

    def test_truncated_layouts_rejected(self):
        res = dwt2d(make_image(8, 8), levels=2)
        with self.assertRaisesRegex(ValueError, "layouts"):
            idwt2d(self._tampered(layouts=res.layouts[:1]))

    def test_bad_levels_value_rejected(self):
        with self.assertRaisesRegex(ValueError, "levels"):
            idwt2d(self._tampered(levels=0))
        with self.assertRaisesRegex(ValueError, "levels"):
            idwt2d(self._tampered(levels=True))

    def test_bad_subbands_rejected(self):
        with self.assertRaisesRegex(ValueError, "subbands"):
            idwt2d(self._tampered(subbands=()))

    def test_shape_mismatch_rejected(self):
        res = dwt2d([[1, 2], [3, 4]])
        tampered = TransformResult(
            coefficients=((1, 2, 9), (3, 4, 9)),
            shape=(2, 2),
            levels=res.levels,
            layouts=res.layouts,
            subbands=res.subbands,
        )
        with self.assertRaises(ValueError):
            idwt2d(tampered)


# ---------------------------------------------------------------------------
# Supplementary: exhaustive small-matrix exact round trips
# ---------------------------------------------------------------------------

class ExactRoundTrip(unittest.TestCase):
    def test_all_shapes_and_levels(self):
        for h, w in SHAPES:
            depth = max_levels(h, w)
            for lv in range(1, max(depth, 1) + 1):
                with self.subTest(shape=(h, w), levels=lv):
                    img = make_image(h, w, seed_offset=h + w + lv)
                    res = dwt2d(img, levels=lv)
                    self.assertEqual(idwt2d(res), tuple(map(tuple, img)))

    def test_extreme_values_round_trip(self):
        # Boundary magnitudes placed so that every checked intermediate stays
        # within int63 (edge details get doubled by mirroring, so a naive
        # corner placement of +/-M correctly overflows and is tested above).
        big = M // 8
        images = [
            [[M]],  # length-1 identity accepts the boundary value itself
            [[M, 0, 0]],  # 1x3: M as the first even sample
            [[M], [0], [0]],
            [
                [big, -big, 0],
                [-big, big, big],
                [0, -big, -big],
            ],
        ]
        for img in images:
            with self.subTest(img=img):
                res = dwt2d(img)
                self.assertEqual(idwt2d(res), tuple(map(tuple, img)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
