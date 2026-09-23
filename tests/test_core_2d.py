"""二维变换：手算 2x2/3x3 锚点、奇数尺寸、1xN/Nx1、全负、多层、布局元数据。"""

import unittest

from iwt53 import (
    IntegerOverflowError,
    InvalidParameterError,
    Subband,
    TransformLayout,
    forward_2d,
    inverse_2d,
    max_levels,
)
from reference import ref_forward_2d, ref_inverse_2d


def m(rows):
    return [list(r) for r in rows]


class Test2DHandAnchors(unittest.TestCase):
    def test_2x2_hand(self):
        # 行：[1,2]->L[2] H[1]，[3,4]->L[4] H[1]
        # 列：col0 [2,4]->L[3] H[2]；col1 [1,1]->L[1] H[0]
        res = forward_2d(m([[1, 2], [3, 4]]), levels=1)
        self.assertEqual(res.coefficients, [[3, 1], [2, 0]])
        names = {(b.name, b.row, b.col, b.rows, b.cols) for b in res.layout.bands}
        self.assertIn(("LH", 0, 1, 1, 1), names)
        self.assertIn(("HL", 1, 0, 1, 1), names)
        self.assertIn(("HH", 1, 1, 1, 1), names)
        self.assertEqual(res.layout.ll.rows, 1)
        self.assertEqual(res.layout.ll.cols, 1)
        self.assertEqual(inverse_2d(res), [[1, 2], [3, 4]])

    def test_3x3_hand(self):
        # 行变换（一维手算表）：
        # [1,2,3]->L[1,3] H[0]；[4,5,6]->L[4,6] H[0]；[7,8,9]->L[7,9] H[0]
        # 列方向（同样手算）：
        # col0 [1,4,7] -> L[1,7] H[0]
        # col1 [3,6,9] -> L[3,9] H[0]
        # col2 [0,0,0] -> L[0,0] H[0]
        res = forward_2d(m([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), levels=1)
        self.assertEqual(res.coefficients,
                         [[1, 3, 0], [7, 9, 0], [0, 0, 0]])
        # LL 为 2x2 左上，其余三个子带：LH 2x1、HL 1x2、HH 1x1
        self.assertEqual(res.band_matrix("LL"), [[1, 3], [7, 9]])
        self.assertEqual(res.band_matrix("LH", 1), [[0], [0]])
        self.assertEqual(res.band_matrix("HL", 1), [[0, 0]])
        self.assertEqual(res.band_matrix("HH", 1), [[0]])
        self.assertEqual(inverse_2d(res), [[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def test_band_matrix_returns_copy(self):
        res = forward_2d(m([[1, 2], [3, 4]]), 1)
        bm = res.band_matrix("HH", 1)
        bm[0][0] = 999
        self.assertEqual(res.coefficients[1][1], 0)


class TestShapes(unittest.TestCase):
    def setUp(self):
        self.sizes = [
            (1, 1), (1, 2), (1, 3), (1, 5), (1, 8),
            (2, 1), (3, 1), (5, 1), (8, 1),
            (2, 2), (3, 3), (3, 5), (5, 3), (7, 7), (8, 8),
            (1, 128), (128, 1),
        ]

    def _img(self, r, c):
        return [[((i * 31 + j * 17 + 5) % 23) - 11 for j in range(c)]
                for i in range(r)]

    def test_roundtrip_all_shapes_level1(self):
        for r, c in self.sizes:
            img = self._img(r, c)
            with self.subTest(size=(r, c)):
                res = forward_2d(img, 1)
                self.assertEqual(inverse_2d(res), img)
                # 输入未被修改
                self.assertEqual(img, self._img(r, c))

    def test_matches_independent_reference(self):
        for r, c in self.sizes:
            img = self._img(r, c)
            for lv in (1, 2):
                if lv > max(1, max_levels(r, c)):
                    continue
                with self.subTest(size=(r, c), levels=lv):
                    got = forward_2d(img, lv).coefficients
                    ref, _steps = ref_forward_2d(img, lv)
                    self.assertEqual(got, ref)
                    self.assertEqual(ref_inverse_2d(ref, _steps), img)
                    self.assertEqual(inverse_2d(forward_2d(img, lv)), img)

    def test_one_by_n_row_then_column_identity_axis(self):
        # 1xN：行方向做真正的变换，列方向长度 1 恒等。
        res = forward_2d(m([[1, 2, 3, 4]]), 1)
        self.assertEqual(res.coefficients, [[1, 3, 0, 1]])  # 与一维 n=4 一致
        self.assertEqual(inverse_2d(res), [[1, 2, 3, 4]])
        # Nx1：行方向长度 1 恒等，列方向变换。
        res2 = forward_2d(m([[1], [2], [3], [4]]), 1)
        self.assertEqual(res2.coefficients, [[1], [3], [0], [1]])
        self.assertEqual(inverse_2d(res2), [[1], [2], [3], [4]])

    def test_one_by_one_identity(self):
        res = forward_2d(m([[7]]), 1)
        self.assertEqual(res.coefficients, [[7]])
        self.assertEqual(res.layout.bands, ())
        self.assertEqual(res.layout.steps, ())
        self.assertEqual(inverse_2d(res), [[7]])


class TestNegativeImages(unittest.TestCase):
    def test_all_negative_3x5_multilevel(self):
        img = [[-(1 + (i * 5 + j) % 7) for j in range(5)] for i in range(3)]
        for lv in (1, 2):
            with self.subTest(levels=lv):
                res = forward_2d(img, lv)
                ref, steps = ref_forward_2d(img, lv)
                self.assertEqual(res.coefficients, ref)
                self.assertEqual(inverse_2d(res), img)

    def test_mixed_signs_against_reference(self):
        img = [[(-1) ** (i + j) * (1 + (i * 3 + j) % 9) for j in range(6)]
               for i in range(4)]
        res = forward_2d(img, 2)
        ref, _ = ref_forward_2d(img, 2)
        self.assertEqual(res.coefficients, ref)
        self.assertEqual(inverse_2d(res), img)


class TestMultiLevelLayout(unittest.TestCase):
    def test_8x8_three_levels_only_ll_recurses(self):
        img = [[(i * 8 + j) % 13 - 6 for j in range(8)] for i in range(8)]
        res = forward_2d(img, 3)
        # 每层 LL 尺寸：8 -> 4 -> 2 -> 1
        self.assertEqual(res.layout.steps, ((8, 8), (4, 4), (2, 2)))
        self.assertEqual((res.layout.ll.rows, res.layout.ll.cols), (1, 1))
        # 9 个细节子带（每层 3 个）
        self.assertEqual(len(res.layout.bands), 9)
        # 第 2 层 HL 位于 (2,0)，2x2
        hl2 = res.layout.band("HL", 2)
        self.assertEqual((hl2.row, hl2.col, hl2.rows, hl2.cols), (2, 0, 2, 2))
        # 第 3 层 HH 位于 (1,1)，1x1
        hh3 = res.layout.band("HH", 3)
        self.assertEqual((hh3.row, hh3.col, hh3.rows, hh3.cols), (1, 1, 1, 1))
        ref, steps = ref_forward_2d(img, 3)
        self.assertEqual(res.coefficients, ref)
        self.assertEqual(inverse_2d(res), img)

    def test_5x7_two_levels_odd_sizes(self):
        img = [[(i * 11 + j * 5) % 17 - 8 for j in range(7)] for i in range(5)]
        res = forward_2d(img, 2)
        # 5x7 -> LL 3x4 -> LL 2x2
        self.assertEqual(res.layout.steps, ((5, 7), (3, 4)))
        self.assertEqual((res.layout.ll.rows, res.layout.ll.cols), (2, 2))
        ref, _ = ref_forward_2d(img, 2)
        self.assertEqual(res.coefficients, ref)
        self.assertEqual(inverse_2d(res), img)

    def test_inverse_accepts_coeffs_and_layout(self):
        img = [[i * 4 + j for j in range(4)] for i in range(4)]
        res = forward_2d(img, 2)
        rebuilt = inverse_2d([row[:] for row in res.coefficients], res.layout)
        self.assertEqual(rebuilt, img)

    def test_layout_mismatch_raises(self):
        img = [[i * 4 + j for j in range(4)] for i in range(4)]
        res = forward_2d(img, 1)
        other = forward_2d([[0] * 8 for _ in range(8)], 1)
        with self.assertRaises(InvalidParameterError):
            inverse_2d(res.coefficients, other.layout)

    def test_max_levels_values(self):
        self.assertEqual(max_levels(1, 1), 0)
        self.assertEqual(max_levels(1, 2), 1)
        self.assertEqual(max_levels(3, 5), 3)   # 3x5 ->2x3->1x2->1x1
        self.assertEqual(max_levels(8, 8), 3)
        self.assertEqual(max_levels(128, 128), 7)

    def test_levels_out_of_range(self):
        img = [[0] * 3 for _ in range(3)]
        with self.assertRaises(InvalidParameterError):
            forward_2d(img, 0)
        with self.assertRaises(InvalidParameterError):
            forward_2d(img, 4)  # 3x3 最多 2 层


class TestOverflowBound(unittest.TestCase):
    def test_input_at_bound_accepted(self):
        big = 2 ** 63 - 1
        res = forward_2d([[big]], 1)  # 1x1 恒等
        self.assertEqual(res.coefficients, [[big]])

    def test_intermediate_overflow_raises_not_truncates(self):
        big = 2 ** 63 - 1
        # 一维: H_0 = big - big = 0；L_0 = big + floor(2/4)=big，未越界。
        # 改用 [big, -big]：H_0 = -big - big = -2*INT64_MAX，越界。
        with self.assertRaises(IntegerOverflowError):
            forward_2d([[big, -big]], 1)

    def test_input_exceeding_bound_rejected(self):
        with self.assertRaises(IntegerOverflowError):
            forward_2d([[2 ** 63]], 1)
        with self.assertRaises(IntegerOverflowError):
            forward_2d([[-2 ** 63]], 1)

    def test_overflow_leaves_no_partial_change_to_input(self):
        big = 2 ** 63 - 1
        img = [[big, -big, 0], [0, 0, 0]]
        snapshot = [row[:] for row in img]
        with self.assertRaises(IntegerOverflowError):
            forward_2d(img, 2)
        self.assertEqual(img, snapshot)


if __name__ == "__main__":
    unittest.main()
