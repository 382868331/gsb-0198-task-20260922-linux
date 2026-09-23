"""可逆整数 5/3 提升小波变换（本题契约实现，纯标准库）。

本模块只实现题目约定的数学子集：
  * 一维 Le Gall 5/3 整数提升（预测—更新，两步均为 floor 整数运算）；
  * 二维"先行后列"可分离变换，多层时只递归左上角 LL（Mallat 布局）；
  * 边界：越界的同类样本用"最近有效同类样本"对称复制（端点 clamp）；
  * 长度 1 恒等；
  * 中间整数绝对值严格限制在 2^63-1 以内，越界报错，绝不截断/饱和/回绕。

本模块不宣称与 JPEG 2000 等任何行业标准完整兼容。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "INT64_MAX",
    "MAX_SIDE",
    "WaveletError",
    "InvalidImageError",
    "SizeLimitError",
    "InvalidParameterError",
    "IntegerOverflowError",
    "Subband",
    "TransformLayout",
    "TransformResult",
    "max_levels",
    "forward_1d",
    "inverse_1d",
    "forward_2d",
    "inverse_2d",
]

INT64_MAX = (1 << 63) - 1
"""允许的中间整数绝对值上界（含端点）。"""

MAX_SIDE = 128
"""输入图像边长上限（含端点）：1 <= rows, cols <= 128。"""


class WaveletError(ValueError):
    """本库所有输入/契约类错误的基类（也是 ValueError）。"""


class InvalidImageError(WaveletError):
    """图像或序列的结构/元素类型非法（含 NaN、Infinity、bool、float、错行等）。"""


class SizeLimitError(WaveletError):
    """图像边长超过 MAX_SIDE。"""


class InvalidParameterError(WaveletError):
    """levels 等标量参数非法（类型错、bool、越界、逆变换长度不配对等）。"""


class IntegerOverflowError(WaveletError, ArithmeticError):
    """某个中间整数的绝对值超过 INT64_MAX。不截断，直接抛出。"""


# --------------------------------------------------------------------------- #
# 输入校验
# --------------------------------------------------------------------------- #

def _reject_non_int(value: object, where: str) -> int:
    """要求真正的整数：拒绝 bool、NaN/Infinity、任何 float 与其他类型。"""
    if isinstance(value, bool):
        raise InvalidImageError(f"{where}: bool is not accepted as an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            raise InvalidImageError(f"{where}: NaN is not allowed")
        if math.isinf(value):
            kind = "+Infinity" if value > 0 else "-Infinity"
            raise InvalidImageError(f"{where}: {kind} is not allowed")
        raise InvalidImageError(
            f"{where}: float is not allowed, integers only (got {value!r})"
        )
    raise InvalidImageError(
        f"{where}: expected integer, got {type(value).__name__}"
    )


def _bounded(value: int, where: str) -> int:
    if value > INT64_MAX or value < -INT64_MAX:
        raise IntegerOverflowError(
            f"{where}: integer {value} has absolute value {abs(value)} "
            f"> INT64_MAX ({INT64_MAX}); no truncation is performed"
        )
    return value


def _add(a: int, b: int, where: str) -> int:
    return _bounded(a + b, where)


def _sub(a: int, b: int, where: str) -> int:
    return _bounded(a - b, where)


def _check_vector(seq: object, what: str, *, allow_empty: bool = False) -> list[int]:
    if not isinstance(seq, list):
        raise InvalidImageError(
            f"{what}: expected list, got {type(seq).__name__}"
        )
    if not seq and not allow_empty:
        raise InvalidImageError(f"{what}: must be a non-empty list")
    out: list[int] = []
    for i, v in enumerate(seq):
        iv = _reject_non_int(v, f"{what}[{i}]")
        _bounded(iv, f"{what}[{i}]")
        out.append(iv)
    return out


def _check_image(image: object) -> tuple[int, int]:
    """完整校验图像；返回 (rows, cols)。失败时不触碰调用方对象。"""
    if not isinstance(image, list):
        raise InvalidImageError(
            f"image: expected list of rows, got {type(image).__name__}"
        )
    if not image:
        raise InvalidImageError("image: must contain at least one row")
    rows = len(image)
    cols = -1
    for i, row in enumerate(image):
        if not isinstance(row, list):
            raise InvalidImageError(
                f"image[{i}]: expected list row, got {type(row).__name__}"
            )
        if i == 0:
            cols = len(row)
            if cols == 0:
                raise InvalidImageError("image[0]: row must be non-empty")
        elif len(row) != cols:
            raise InvalidImageError(
                f"image[{i}]: ragged row has {len(row)} elements, "
                f"expected {cols} (from image[0])"
            )
    if rows > MAX_SIDE:
        raise SizeLimitError(
            f"image: {rows} rows exceed MAX_SIDE={MAX_SIDE}"
        )
    if cols > MAX_SIDE:
        raise SizeLimitError(
            f"image: {cols} columns exceed MAX_SIDE={MAX_SIDE}"
        )
    for i, row in enumerate(image):
        for j, v in enumerate(row):
            iv = _reject_non_int(v, f"image[{i}][{j}]")
            _bounded(iv, f"image[{i}][{j}]")
    return rows, cols


# --------------------------------------------------------------------------- #
# 一维核心
# --------------------------------------------------------------------------- #

def _even_index(k: int, count: int) -> int:
    """同类（低频/偶）序列第 k 个邻居的夹取索引：越界取最近有效同类。"""
    if k < 0:
        k = 0
    elif k >= count:
        k = count - 1
    return k


def forward_1d(seq: list[int]) -> tuple[list[int], list[int]]:
    """一维前向可逆整数 5/3 提升。

    定义（n = len(seq)，E_k 为偶样本、O_k 为奇样本）::

        H_k = O_k - floor((E_k + E_{k+1}) / 2)
        L_k = E_k + floor((H_{k-1} + H_k + 2) / 4)

    越界的 E/H 下标按最近有效同类样本复制（端点 clamp）。
    n == 1 时恒等，返回 ``([seq[0]], [])``。

    返回 ``(low, high)``：``len(low)=ceil(n/2)``，``len(high)=floor(n/2)``。
    """
    x = _check_vector(seq, "seq")
    n = len(x)
    if n == 1:
        return [x[0]], []

    nl = (n + 1) // 2
    nh = n // 2
    high: list[int] = [0] * nh

    # 预测：奇样本 O_k 减去其左右偶样本 E_k、E_{k+1} 均值的 floor。
    for k in range(nh):
        il = 2 * k
        ir = 2 * _even_index(k + 1, nl)
        pred = _add(x[il], x[ir], f"forward_1d prediction sum at odd k={k}") // 2
        high[k] = _sub(x[2 * k + 1], pred, f"forward_1d detail at k={k}")

    # 更新：偶样本加上相邻细节 (..+2)/4 的 floor。
    low: list[int] = [0] * nl
    for k in range(nl):
        hl = high[0 if k - 1 < 0 else k - 1]
        hr = high[nh - 1 if k >= nh else k]
        total = _add(hl, hr, f"forward_1d update sum at even k={k}")
        total = _add(total, 2, f"forward_1d update rounding at k={k}")
        low[k] = _add(x[2 * k], total // 4, f"forward_1d low at k={k}")

    return low, high


def inverse_1d(low: list[int], high: list[int]) -> list[int]:
    """一维逆变换，严格按前向反序：先反更新再反预测。纯整数，无浮点。"""
    s = _check_vector(low, "low", allow_empty=True)
    d = _check_vector(high, "high", allow_empty=True)
    nl = len(s)
    nh = len(d)
    if nl == 0 or not (nl == nh or nl == nh + 1):
        raise InvalidParameterError(
            f"inverse_1d: invalid coefficient lengths low={nl}, high={nh}; "
            f"expected len(low)==len(high) or len(low)==len(high)+1"
        )
    if nl == 1 and nh == 0:
        return [s[0]]

    # 反更新，恢复偶样本。
    evens: list[int] = [0] * nl
    for k in range(nl):
        hl = d[0 if k - 1 < 0 else k - 1]
        hr = d[nh - 1 if k >= nh else k]
        total = _add(hl, hr, f"inverse_1d undo-update sum at k={k}")
        total = _add(total, 2, f"inverse_1d undo-update rounding at k={k}")
        evens[k] = _sub(s[k], total // 4, f"inverse_1d even at k={k}")

    # 反预测，恢复奇样本并交织回原排列。
    out: list[int] = [0] * (nl + nh)
    for k in range(nl):
        out[2 * k] = evens[k]
    for k in range(nh):
        il = k
        ir = _even_index(k + 1, nl)
        pred = _add(evens[il], evens[ir], f"inverse_1d prediction sum at k={k}") // 2
        out[2 * k + 1] = _add(d[k], pred, f"inverse_1d odd at k={k}")
    return out


# --------------------------------------------------------------------------- #
# 布局元数据
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Subband:
    """全系数矩阵中的一个子带矩形。

    命名按"行频率 + 列频率"：LH = 行低频、列高频（竖直方向条纹），
    HL = 行高频、列低频，HH = 双高频。坐标为全矩阵绝对下标。
    """

    name: str
    level: int
    row: int
    col: int
    rows: int
    cols: int


@dataclass(frozen=True)
class TransformLayout:
    """一次前向变换的完整布局描述。"""

    rows: int
    cols: int
    levels: int
    bands: tuple[Subband, ...]
    ll: Subband
    """最深处低频 LL（levels 层后）。"""
    steps: tuple[tuple[int, int], ...]
    """每层分解前 LL 矩形的 (rows, cols)，逆变换严格按它反序。"""

    def band(self, name: str, level: int | None = None) -> Subband:
        """按名字（及层号）取细节带；找不到抛 KeyError。"""
        for b in self.bands:
            if b.name == name and (level is None or b.level == level):
                return b
        if name == "LL":
            return self.ll
        raise KeyError(f"band {name!r} (level={level}) not present in layout")


@dataclass
class TransformResult:
    """前向变换输出：与原图同尺寸的系数矩阵 + 布局元数据。"""

    coefficients: list[list[int]]
    layout: TransformLayout

    def band_matrix(self, name: str, level: int | None = None) -> list[list[int]]:
        """拷贝取出某个子带的矩形系数。"""
        b = self.layout.band(name, level)
        return [
            list(row[b.col : b.col + b.cols])
            for row in self.coefficients[b.row : b.row + b.rows]
        ]


def max_levels(rows: int, cols: int) -> int:
    """到达 1x1 LL 所需的最大非平凡层数（1x1 输入返回 0）。"""
    if isinstance(rows, bool) or isinstance(cols, bool) or not (
        isinstance(rows, int) and isinstance(cols, int)
    ):
        raise InvalidParameterError("max_levels: rows and cols must be int")
    if rows < 1 or cols < 1:
        raise InvalidParameterError("max_levels: rows and cols must be >= 1")
    levels = 0
    while (rows, cols) != (1, 1):
        rows = (rows + 1) // 2
        cols = (cols + 1) // 2
        levels += 1
    return levels


# --------------------------------------------------------------------------- #
# 二维核心
# --------------------------------------------------------------------------- #

def _check_levels(levels: object, rows: int, cols: int) -> int:
    if isinstance(levels, bool) or not isinstance(levels, int):
        raise InvalidParameterError(
            f"levels: expected int, got {type(levels).__name__}"
        )
    cap = max(1, max_levels(rows, cols))
    if levels < 1 or levels > cap:
        raise InvalidParameterError(
            f"levels={levels} out of range for image {rows}x{cols}; "
            f"allowed 1..{cap}"
        )
    return levels


def forward_2d(image: list[list[int]], levels: int = 1) -> TransformResult:
    """二维前向变换，先行后列；多层只递归左上 LL。

    返回的系数矩阵与输入同尺寸（Mallat 布局），所有子带位置见
    ``TransformResult.layout``。输入矩阵不会被修改（内部复制后计算），
    任何校验/溢出错误都不会在调用方数据上留下部分变更。
    """
    rows, cols = _check_image(image)
    levels = _check_levels(levels, rows, cols)

    a = [list(row) for row in image]
    bands: list[Subband] = []
    steps: list[tuple[int, int]] = []
    r, c = rows, cols

    for lev in range(1, levels + 1):
        if r == 1 and c == 1:
            break  # 仅 1x1 输入 + levels=1：整图恒等
        steps.append((r, c))

        # 行方向（每一行独立一维变换，行内解交织为 [L...|H...]）。
        for i in range(r):
            low, high = forward_1d(a[i][:c])
            a[i][:c] = low + high

        lr = (r + 1) // 2
        hr = r // 2
        lc = (c + 1) // 2
        hc = c // 2

        # 列方向（每一列独立一维变换，列内解交织为上 L / 下 H）。
        for j in range(c):
            col = [a[i][j] for i in range(r)]
            low, high = forward_1d(col)
            for i in range(lr):
                a[i][j] = low[i]
            for i in range(hr):
                a[lr + i][j] = high[i]

        if hc > 0:
            bands.append(Subband("LH", lev, 0, lc, lr, hc))
        if hr > 0:
            bands.append(Subband("HL", lev, lr, 0, hr, lc))
        if hr > 0 and hc > 0:
            bands.append(Subband("HH", lev, lr, lc, hr, hc))

        r, c = lr, lc

    ll = Subband("LL", levels, 0, 0, r, c)
    layout = TransformLayout(
        rows=rows,
        cols=cols,
        levels=levels,
        bands=tuple(bands),
        ll=ll,
        steps=tuple(steps),
    )
    return TransformResult(coefficients=a, layout=layout)


def inverse_2d(
    result: TransformResult | list[list[int]],
    layout: TransformLayout | None = None,
) -> list[list[int]]:
    """二维逆变换。

    接受 ``forward_2d`` 返回的 :class:`TransformResult`，或
    ``inverse_2d(coeffs, layout)`` 形式。严格反序：从最深层到最浅层，
    每层先反列、后反行。返回新矩阵，不修改输入。
    """
    if layout is None:
        if not hasattr(result, "coefficients") or not hasattr(result, "layout"):
            raise InvalidParameterError(
                "inverse_2d: expected TransformResult or (coefficients, layout)"
            )
        coeffs = result.coefficients
        layout = result.layout
    else:
        coeffs = result

    rows, cols = _check_image(coeffs)
    if rows != layout.rows or cols != layout.cols:
        raise InvalidParameterError(
            f"inverse_2d: coefficient matrix {rows}x{cols} does not match "
            f"layout {layout.rows}x{layout.cols}"
        )

    a = [list(row) for row in coeffs]

    for lev in range(layout.levels, 0, -1):
        if lev - 1 >= len(layout.steps):
            continue  # 1x1 恒等层：无操作
        r, c = layout.steps[lev - 1]
        lr = (r + 1) // 2
        hr = r // 2
        lc = (c + 1) // 2
        hc = c // 2

        # 先反列。
        for j in range(c):
            low = [a[i][j] for i in range(lr)]
            high = [a[lr + i][j] for i in range(hr)]
            col = inverse_1d(low, high)
            for i in range(r):
                a[i][j] = col[i]

        # 再反行。
        for i in range(r):
            low = a[i][:lc]
            high = a[i][lc : c]
            row = inverse_1d(low, high)
            a[i][:c] = row

    return a
