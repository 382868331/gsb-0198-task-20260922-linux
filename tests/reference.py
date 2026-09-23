"""独立参考实现（仅测试使用）。

与 ``iwt53.core`` 完全独立编写：一维直接按数学定义逐下标计算，
二维用字典网格 + 显式邻居夹取，不 import 被测包，也不共享任何函数。
作用是给测试提供"第二来源"的期望值；往返一致在测试中仅作补充。
"""

# ---------------------------------------------------------------------------
# 一维参考：直接套定义
# ---------------------------------------------------------------------------

def ref_forward_1d(x):
    n = len(x)
    p = (n + 1) // 2  # 偶样本数
    q = n // 2        # 奇样本数
    if n == 1:
        return [x[0]], []

    detail = []
    for k in range(q):
        left_even = x[2 * k]
        right_even = x[2 * min(p - 1, k + 1)]
        detail.append(x[2 * k + 1] - (left_even + right_even) // 2)

    low = []
    for k in range(p):
        dl = detail[max(0, k - 1)]
        dh = detail[min(q - 1, k)]
        low.append(x[2 * k] + (dl + dh + 2) // 4)
    return low, detail


def ref_inverse_1d(low, high):
    p, q = len(low), len(high)
    if p == 1 and q == 0:
        return [low[0]]

    evens = [
        low[k] - (high[max(0, k - 1)] + high[min(q - 1, k)] + 2) // 4
        for k in range(p)
    ]
    rebuilt = []
    for k in range(p):
        rebuilt.append(evens[k])
        if k < q:
            predictor = (evens[k] + evens[min(p - 1, k + 1)]) // 2
            rebuilt.append(high[k] + predictor)
    return rebuilt


# ---------------------------------------------------------------------------
# 二维参考：字典网格，逐层只动左上角；输出普通嵌套列表（Mallat 布局）
# ---------------------------------------------------------------------------

def ref_forward_2d(image, levels):
    r = len(image)
    c = len(image[0])
    grid = {(i, j): int(image[i][j]) for i in range(r) for j in range(c)}
    steps = []

    for _ in range(levels):
        if r == 1 and c == 1:
            break
        steps.append((r, c))
        lr, lc = (r + 1) // 2, (c + 1) // 2
        hr, hc = r // 2, c // 2

        # 行方向
        for i in range(r):
            vals = [grid[(i, j)] for j in range(c)]
            low, high = ref_forward_1d(vals)
            for t, v in enumerate(low):
                grid[(i, t)] = v
            for t, v in enumerate(high):
                grid[(i, lc + t)] = v

        # 列方向
        for j in range(c):
            vals = [grid[(i, j)] for i in range(r)]
            low, high = ref_forward_1d(vals)
            for t, v in enumerate(low):
                grid[(t, j)] = v
            for t, v in enumerate(high):
                grid[(lr + t, j)] = v

        r, c = lr, lc

    rows = len(image)
    cols = len(image[0])
    return [[grid[(i, j)] for j in range(cols)] for i in range(rows)], steps


def ref_inverse_2d(coeffs, steps):
    rows = len(coeffs)
    cols = len(coeffs[0])
    grid = {(i, j): int(coeffs[i][j]) for i in range(rows) for j in range(cols)}

    for r, c in reversed(steps):
        lr, lc = (r + 1) // 2, (c + 1) // 2
        hr, hc = r // 2, c // 2

        # 先反列
        for j in range(c):
            low = [grid[(t, j)] for t in range(lr)]
            high = [grid[(lr + t, j)] for t in range(hr)]
            vals = ref_inverse_1d(low, high)
            for t, v in enumerate(vals):
                grid[(t, j)] = v

        # 再反行
        for i in range(r):
            low = [grid[(i, t)] for t in range(lc)]
            high = [grid[(i, lc + t)] for t in range(hc)]
            vals = ref_inverse_1d(low, high)
            for t, v in enumerate(vals):
                grid[(i, t)] = v

    return [[grid[(i, j)] for j in range(cols)] for i in range(rows)]
