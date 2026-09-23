"""一维 5/3 提升：手算锚点、负数 floor、长度 1 恒等、独立参考交叉验证。"""

import math
import unittest

from iwt53 import forward_1d, inverse_1d
from reference import ref_forward_1d, ref_inverse_1d


# 每个用例: (输入, 手算 low, 手算 high)
# 手算严格按 H_k = O_k - floor((E_{k-1}+E_{k+1})/2)
#          L_k = E_k + floor((H_{k-1}+H_k+2)/4)，越界邻居端点夹取。
HAND_TABLE = [
    ([42], [42], []),                                  # 长度 1 恒等
    ([10, 1], [6], [-9]),                              # n=2 端点全夹取
    ([0, 7, 4], [3, 7], [5]),                          # n=3
    ([1, 2, 3, 4], [1, 3], [0, 1]),                    # n=4
    ([3, 1, 4, 1, 5, 9], [2, 3, 5], [-2, -3, 4]),      # n=6
    ([4, 2, 6, 8, 1, 3, 5], [3, 7, 2, 5], [-3, 5, 0]), # n=7
    # 全负/含负：验证 Python floor 语义（如 (-8)+(-9)=-17, floor(/2)=-9）
    ([-5, -3, -8, -1, -9], [-3, -5, -5], [4, 8]),
       ([-1, 0, -1], [0, 0], [1]),
    ([0, 0, 0, 0], [0, 0], [0, 0]),
    ([-7, -7, -7], [-7, -7], [0]),
]


class TestForward1DHand(unittest.TestCase):
    def test_hand_computed_table(self):
        for x, want_low, want_high in HAND_TABLE:
            with self.subTest(x=x):
                low, high = forward_1d(list(x))
                self.assertEqual(low, want_low)
                self.assertEqual(high, want_high)

    def test_negative_floor_prediction(self):
        # floor((-8 + -9)/2) = floor(-8.5) = -9（不是向零截断的 -8）
        low, high = forward_1d([-5, -3, -8, -1, -9])
        self.assertEqual(high[0], 4)   # -3 - (-7) = 4
        self.assertEqual(high[1], 8)   # -1 - (-9) = 8
        self.assertEqual(low, [-3, -5, -5])

    def test_length_one_is_identity(self):
        for v in (0, 1, -1, 123456, -98765):
            low, high = forward_1d([v])
            self.assertEqual((low, high), ([v], []))
            self.assertEqual(inverse_1d(low, high), [v])


class TestAgainstIndependentReference(unittest.TestCase):
    """独立参考 ref_* 不 import 被测核心；这里逐长度交叉比对。"""

    def setUp(self):
        self.cases = [
            list(range(-12, 13)),
            [((i * 37 + 11) % 17) - 8 for i in range(40)],
            [(-1) ** i * (i * i % 13 + 1) for i in range(40)],
            [5] * 40,
            [-6] * 40,
        ]

    def test_forward_matches_reference_lengths_1_to_40(self):
        for n in range(1, 41):
            for base in self.cases:
                x = base[:n]
                with self.subTest(n=n, x=x):
                    got_low, got_high = forward_1d(x)
                    ref_low, ref_high = ref_forward_1d(x)
                    self.assertEqual(got_low, ref_low)
                    self.assertEqual(got_high, ref_high)

    def test_inverse_matches_reference_lengths_1_to_40(self):
        for n in range(1, 41):
            for base in self.cases:
                x = base[:n]
                low, high = ref_forward_1d(x)  # 期望系数来自独立参考
                with self.subTest(n=n):
                    self.assertEqual(inverse_1d(low, high), x)
                    self.assertEqual(ref_inverse_1d(low, high), x)

    def test_core_inverse_is_exact_inverse_of_core_forward(self):
        # 往返一致仅作补充验证。
        for n in range(1, 33):
            x = [((i * 53 + 7) % 29) - 14 for i in range(n)]
            low, high = forward_1d(x)
            self.assertEqual(inverse_1d(low, high), x)

    def test_all_outputs_are_python_int(self):
        x = [((i * 7) % 11) - 5 for i in range(20)]
        low, high = forward_1d(x)
        self.assertTrue(all(type(v) is int for v in low + high))


if __name__ == "__main__":
    unittest.main()
