"""固定输入演示：真实计算 2 层可逆整数 5/3 提升，逐位恢复并展示子带；
随后实际触发两个拒绝/溢出边界。纯标准库，确定性数据，无网络、无预录结果。

运行：python demo.py
"""

import time

from iwt53 import (
    INT64_MAX,
    InvalidImageError,
    IntegerOverflowError,
    forward_1d,
    forward_2d,
    inverse_2d,
)


def show_matrix(title, mat):
    print(title)
    for row in mat:
        print("  " + "  ".join(f"{v:4d}" for v in row))


def main():
    t0 = time.perf_counter()

    # ---- 1. 一维手算负数参考（真实计算，用于对照 README 中的手算） ----------
    x = [-5, -3, -8, -1, -9]
    low, high = forward_1d(x)
    print("[1D] 输入(含负数):", x)
    print("     低频 L:", low, " 高频 H:", high)
    print("     关键: H1 = -1 - floor((-8 + -9)/2) = -1 - (-9) = 8 "
          "(floor, 非向零截断)")
    print()

    # ---- 2. 二维 5x7、两层：确定性固定数据 --------------------------------
    img = [[(i * 11 + j * 5) % 17 - 8 for j in range(7)] for i in range(5)]
    show_matrix("[2D] 原始整数图 5x7:", img)

    res = forward_2d(img, levels=2)
    lay = res.layout
    show_matrix("[2D] 两层提升后的系数矩阵（Mallat 子带布局）:", res.coefficients)

    print(f"[2D] 尺寸 {lay.rows}x{lay.cols}，层数 {lay.levels}，"
          f"LL 递归尺寸链: {lay.steps} -> ({lay.ll.rows}x{lay.ll.cols})")
    print("[2D] 子带布局元数据（name level row col rows×cols）:")
    print(f"     LL   L{lay.levels}  {lay.ll.row:2d},{lay.ll.col:2d}  "
          f"{lay.ll.rows}x{lay.ll.cols}  <- 仅此处被继续递归")
    for b in lay.bands:
        print(f"     {b.name:3s}  L{b.level}  {b.row:2d},{b.col:2d}  "
              f"{b.rows}x{b.cols}")

    show_matrix("[2D] 取出 HH@L2 子带:", res.band_matrix("HH", 2))

    rebuilt = inverse_2d(res)
    show_matrix("[2D] 严格反序逆变换恢复:", rebuilt)
    exact = rebuilt == img
    print(f"[2D] 逐位（bit-exact）恢复: {exact}")
    assert exact, "roundtrip must be exact"

    # ---- 3. 实际触发的拒绝边界：NaN 必须被拒绝，不静默纠正 ------------------
    print()
    print("[边界1] 传入含 NaN 的图像:")
    try:
        forward_2d([[1, 2], [3, float("nan")]])
    except InvalidImageError as e:
        print(f"       如预期拒绝 -> {type(e).__name__}: {e}")

    # ---- 4. 实际触发的溢出边界：中间整数 |v| > 2^63-1，报错不截断 -----------
    print("[边界2] 输入 [INT64_MAX, -INT64_MAX]，预测步产生越界中间值:")
    try:
        forward_1d([INT64_MAX, -INT64_MAX])
    except IntegerOverflowError as e:
        print(f"       如预期报错 -> {type(e).__name__}")
        print(f"       {e}")
    print(f"\n演示计算耗时 {time.perf_counter() - t0:.3f} 秒。")


if __name__ == "__main__":
    main()
