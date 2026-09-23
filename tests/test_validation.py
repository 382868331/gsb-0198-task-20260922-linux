"""错误语义：NaN/Infinity、bool、非整数、错行、尺寸上限、可定位信息、不静默纠正。"""

import math
import unittest

from iwt53 import (
    INT64_MAX,
    MAX_SIDE,
    InvalidImageError,
    InvalidParameterError,
    SizeLimitError,
    WaveletError,
    forward_1d,
    forward_2d,
    inverse_1d,
)


class TestScalarRejection(unittest.TestCase):
    def test_bool_rejected_everywhere(self):
        # bool 虽是 int 子类，但必须显式拒绝
        with self.assertRaises(InvalidImageError) as cm:
            forward_1d([True, 2, 3])
        self.assertIn("bool", str(cm.exception))
        with self.assertRaises(InvalidImageError):
            forward_2d([[1, 2], [3, True]])
        with self.assertRaises(InvalidParameterError):
            forward_2d([[1, 2], [3, 4]], levels=True)
        with self.assertRaises(InvalidImageError):
            inverse_1d([1], [True])

    def test_nan_rejected(self):
        with self.assertRaises(InvalidImageError) as cm:
            forward_1d([1, float("nan"), 3])
        self.assertIn("NaN", str(cm.exception))
        with self.assertRaises(InvalidImageError):
            forward_2d([[float("nan")]])

    def test_infinity_rejected_both_signs(self):
        for v in (float("inf"), float("-inf")):
            with self.subTest(v=v):
                with self.assertRaises(InvalidImageError) as cm:
                    forward_1d([1, v, 3])
                self.assertIn("Infinity", str(cm.exception))
                with self.assertRaises(InvalidImageError):
                    forward_2d([[v]])

    def test_plain_float_rejected_even_if_integral(self):
        # 不静默接受 2.0：整数接口不做隐式转换
        with self.assertRaises(InvalidImageError):
            forward_1d([1, 2.0, 3])
        with self.assertRaises(InvalidImageError):
            forward_2d([[1, 2], [3, 4.0]])

    def test_other_types_rejected(self):
        for bad in ([1, "2", 3], [1, None], [1 + 0j]):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidImageError):
                    forward_1d(bad)
        with self.assertRaises(InvalidImageError):
            forward_2d([[(1, 2)]])


class TestStructureValidation(unittest.TestCase):
    def test_not_a_list(self):
        with self.assertRaises(InvalidImageError):
            forward_1d((1, 2, 3))
        with self.assertRaises(InvalidImageError):
            forward_2d(((1, 2), (3, 4)))

    def test_empty(self):
        with self.assertRaises(InvalidImageError):
            forward_1d([])
        with self.assertRaises(InvalidImageError):
            forward_2d([])
        with self.assertRaises(InvalidImageError):
            forward_2d([[]])

    def test_ragged_rows_localizable(self):
        with self.assertRaises(InvalidImageError) as cm:
            forward_2d([[1, 2, 3], [4, 5]])
        self.assertIn("image[1]", str(cm.exception))

    def test_row_not_list_localizable(self):
        with self.assertRaises(InvalidImageError) as cm:
            forward_2d([[1, 2], "ab"])
        self.assertIn("image[1]", str(cm.exception))

    def test_element_index_localizable(self):
        with self.assertRaises(InvalidImageError) as cm:
            forward_2d([[1, 2], [3, float("nan")]])
        self.assertIn("image[1][1]", str(cm.exception))


class TestSizeLimit(unittest.TestCase):
    def test_128_accepted(self):
        img = [[0] * MAX_SIDE for _ in range(MAX_SIDE)]
        res = forward_2d(img, 1)
        self.assertEqual(len(res.coefficients), MAX_SIDE)

    def test_129_rejected(self):
        img = [[0] * 5 for _ in range(MAX_SIDE + 1)]
        with self.assertRaises(SizeLimitError) as cm:
            forward_2d(img, 1)
        self.assertIn(str(MAX_SIDE), str(cm.exception))
        img = [[0] * (MAX_SIDE + 1) for _ in range(5)]
        with self.assertRaises(SizeLimitError):
            forward_2d(img, 1)

    def test_error_hierarchy(self):
        # 所有库错误都是 WaveletError（ValueError）
        self.assertTrue(issubclass(SizeLimitError, WaveletError))
        self.assertTrue(issubclass(InvalidParameterError, WaveletError))
        self.assertTrue(issubclass(InvalidImageError, WaveletError))

    def test_int64_bound_constant(self):
        self.assertEqual(INT64_MAX, 9223372036854775807)

    def test_invalid_inverse_lengths(self):
        with self.assertRaises(InvalidParameterError):
            inverse_1d([1, 2, 3], [4])  # low 与 high 长度差 2
        with self.assertRaises(InvalidParameterError):
            inverse_1d([], [])


if __name__ == "__main__":
    unittest.main()
