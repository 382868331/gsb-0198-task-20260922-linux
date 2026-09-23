# 离散图像的可逆整数 5/3 提升变换（iwt53）

纯 Python 标准库实现的二维可逆整数 5/3（Le Gall）提升小波题库，无任何第三方依赖、
无浮点运算、完全离线。输入与输出都是整数，前向/逆变换**逐位精确**（不存在误差与
容差概念）。

- **Python 版本**：Python 3.14.7（3.10+ 亦可运行，仅用到标准库；本仓库按 3.14.7 验收）
- **依赖**：无。`pip install` 不需要任何包。
- **平台**：Linux 原生；不访问网络、账号、密钥、硬件、数据库、中间件、云 API 或 Docker。

## 运行命令

```bash
python -m unittest discover -s tests -v   # 必要测试（41 个），秒级完成
python demo.py                            # 固定输入演示，约 0.05 秒
```

## 数学定义（本题契约子集）

一维提升，记偶样本 E_k = x[2k]、奇样本 O_k = x[2k+1]：

```
H_k = O_k - floor((E_k + E_{k+1}) / 2)          # 预测（奇样本减左右偶样本均值）
L_k = E_k + floor((H_{k-1} + H_k + 2) / 4)      # 更新（偶样本加左右细节）
```

- 所有除法是 Python 整数 `//`（floor 语义），负数正确，例如
  `floor((-8 + -9)/2) = floor(-8.5) = -9`（**不是**向零截断的 -8）。
- **边界**：越界的同类样本用"最近有效同类样本"对称复制（端点夹取）。
  即 O_{nh-1} 在偶数长度下右偶样本缺失时复制 E_{nh-1}；E_0 的左细节、
  末位偶样本的右细节同理复制最近细节。
- **长度 1 恒等**：`forward_1d([v]) == ([v], [])`。

二维：**先行后列**，行、列各自做一维变换并在各自方向解交织
（低半段在前、高半段在后）。多层时只递归左上角 LL（标准 Mallat 金字塔布局），
每一层产生 LH / HL / HH 三个细节子带。奇数尺寸下 LL 尺寸为
`ceil(rows/2) × ceil(cols/2)`。逆变换严格反序：从最深层到最浅层，每层**先反列、
后反行**，每步先反更新再反预测。

本实现只按本题契约工作，**不宣称与 JPEG 2000 等任何行业标准完整兼容**
（量化、ROI、码流打包等均不在范围内）。

## 公开接口

```python
from iwt53 import (
    forward_1d, inverse_1d,      # list[int] -> (low, high) / 反解
    forward_2d, inverse_2d,      # list[list[int]] <-> TransformResult
    max_levels,                  # 给定 (rows, cols) 可分解的最大层数
    INT64_MAX, MAX_SIDE,         # 2**63-1, 128
    Subband, TransformLayout, TransformResult,
    WaveletError, InvalidImageError, SizeLimitError,
    InvalidParameterError, IntegerOverflowError,
)
```

- `result = forward_2d(image, levels=1)`：
  - `result.coefficients`：与原图同尺寸的整数系数矩阵（Mallat 布局）；
  - `result.layout`：`rows/cols/levels`、`steps`（每层分解前 LL 的尺寸）、
    `ll`（最深 LL 矩形）、`bands`（每个 LH/HL/HH 的名字、层号、绝对行列起点与尺寸）；
  - `result.band_matrix("HH", 2)`：按名字/层号拷贝取出某个子带。
- `inverse_2d(result)`，或 `inverse_2d(coeffs, layout)`；返回新矩阵，不修改输入。

## 输入上限与溢出语义

- 图像：非空、矩形（禁止错行）嵌套 `list`，`1 <= rows, cols <= 128`。
- 元素与所有中间整数必须是 Python 真 `int` 且 `|v| <= 2^63 - 1`。
  每一步预测/更新的中间和、差都**显式做界检查**；越界立即抛
  `IntegerOverflowError`，**不截断、不饱和、不回绕**。

## 错误语义（可定位，不静默纠正）

| 情况 | 异常 |
|---|---|
| 非 list、空序列、错行、元素非整数 | `InvalidImageError`（消息带 `image[i][j]` 定位） |
| `NaN` / `+Infinity` / `-Infinity`（即使包在 float 中） | `InvalidImageError`，消息指明 NaN/Infinity |
| 普通 float（含 `2.0` 这种整数值）、str、complex 等 | `InvalidImageError`，不做隐式转换 |
| `bool`（`True/False`，虽是 int 子类）出现在数值位置 | 一律拒绝 |
| 边长 > 128 | `SizeLimitError` |
| `levels` 类型错 / bool / 超出 `1..max_levels(...)`、逆变换长度不配对 | `InvalidParameterError` |
| 任一中间整数绝对值 > 2^63-1 | `IntegerOverflowError` |

所有异常继承 `WaveletError`（同时是 `ValueError`；溢出还是 `ArithmeticError`）。
前向变换在内部副本上计算，任何错误都不会修改调用方传入的矩阵。

## 设计取舍

- **端点夹取**实现"最近有效同类样本对称复制"：本题邻居只越界一个位置，
  端点夹取与端点镜像反射在此完全等价，且不需要任何长度特判（长度 1 除外）。
- 二维每列在一维调用边界做列表拆装（按列 O(n) 拷贝）。输入上限 128²、
  最多 7 层，规模很小，优先可读性与可审计性；未做原地缓存优化。
- 界检查围绕最小运算单元（预测和、更新和、加减结果）逐点进行，
  宁可早报错也不产生越界中间值。
- 子带命名按"行频率 + 列频率"：LH = 行低/列高，HL = 行高/列低，HH = 双高；
  各文献惯例不一，以本库文档与布局元数据为准。

## 已知限制

- 仅支持题面契约内的 5/3 提升与 Mallat 多层布局；不含其他小波、9/7 浮点变换、
  DWT 之外的任何编码环节。
- 不做行业标准互操作；子带命名/边界规则如与某标准默认值不同，以本文档为准。
- 无浮点输出，因此没有误差口径：正规模型下前向→逆变换结果与输入**逐位相等**；
  测试不使用容差。
- 128²、7 层是硬接口边界，不是性能验证范围；demo 刻意使用小规模固定数据。

## 目录

```
iwt53/core.py      核心算法（无第三方、无浮点）
iwt53/__init__.py  公开接口导出
tests/             unittest 测试
  reference.py     独立参考实现（不 import 被测核心，结构独立编写）
  test_core_1d.py  手算锚点（含负数 floor）、长度 1、独立参考交叉验证
  test_core_2d.py  手算 2x2/3x3、奇数尺寸、1xN/Nx1、全负、多层、布局元数据
  test_validation.py  NaN/Infinity/bool/错行/上限/错误类型
demo.py            固定输入演示：正常两层结果 + 真实触发的拒绝与溢出边界
```

独立参考 `tests/reference.py` 仅用于给出"第二来源"期望值，不调用被测核心；
往返一致性测试只作为补充。
