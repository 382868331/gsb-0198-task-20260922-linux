# lifting53 — 离散图像的可逆整数 5/3 提升变换

纯 Python 标准库实现的二维可逆整数 5/3 提升小波变换（Le Gall 5/3，整数提升
形式），含严格逆变换、子带布局元数据、多层递归与显式溢出边界。核心算法
全部离线、无浮点、无第三方依赖。

## 运行环境

- **Python 版本**：Python 3.14.7（仅使用 3.14 标准库；`from __future__ import
  annotations`，无第三方包）。
- **依赖**：无。`requirements.txt` 仅为占位说明。安装 Python 后直接运行，
  不需要网络、账号、密钥、硬件、数据库、中间件、云 API 或 Docker。

## 准确命令

```bash
# 必要测试（独立参考 + 手算常量 + 往返/ROI 补充）
./.venv/bin/python -m unittest discover -s tests -v

# 固定输入演示（真实计算：多层变换 + ROI 局部还原 + 拒绝边界）
./.venv/bin/python demo.py
```

`.venv` 是指向题目隔离环境的符号链接（Python 3.14.7，仅标准库）。
demo 为小规模固定输入（7×5 两层等），真实计算、无 `sleep`、无预录
结果，正常机器上耗时远小于 1 秒（含 ROI 演示，题面预算 90 秒内）。

## 接口

```python
from lifting53 import dwt2d, idwt2d, dwt1d, idwt1d, max_levels

result = dwt2d(image, levels=1)
# image: Sequence[Sequence[int]]，矩形整数矩阵，不被修改
# result: TransformResult
reconstructed = idwt2d(result)          # 逐位恢复原矩阵
```

ROI 局部逆变换（只还原指定半开矩形内的像素；详见下文
「ROI 局部逆变换」一节）：

```python
def read_coeff(name: str, level: int, row: int, col: int) -> int:
    # 只读回调：子带名、1-based 层数、子带内局部坐标 -> 整数

patch = idwt2d_roi(result, rows=(r0, r1), cols=(c0, c1),
                     read_coeff=read_coeff)
plan  = roi_access_plan(result, rows=(r0, r1), cols=(c0, c1))
```

公开数据结构（冻结 dataclass，不可变）：

- `TransformResult`
  - `coefficients: tuple[tuple[int, ...], ...]` — 完整系数矩阵，多层时各层
    LH/HL/HH 留在原工作区，深层变换原地嵌套在 LL 内（Mallat 式金字塔布局）。
  - `shape: (height, width)`
  - `levels: int`
  - `layouts: tuple[LevelLayout, ...]` — 每层工作区与四个子带的精确位置。
  - `subbands: tuple[Subband, ...]` — 每层 LH/HL/HH 加最深层 LL（中间层 LL
    是下一层工作区，可经 `layouts` 或 `band("LL", k)` 取得）。
  - `band(name, level)` / `extract(name, level)` — 取子带元数据 / 防御性拷贝。
- `LevelLayout`：`level`、`rows`、`cols`（半开区间）、`low_rows`、`low_cols`、
  `bands=(LL,LH,HL,HH)`。
- `Subband`：`name`、`level`、`rows=(r0,r1)`、`cols=(c0,c1)`，半开区间；
  零跨度区间（1×N / N×1 的空带子）表示该子带无样本。
- `ROIPlan`：`roi_access_plan` 的静态分析结果（`shape`、`levels`、
  `rows`/`cols` 请求区间、`bands: tuple[ROIBandAccess, ...]`）；
  `coordinates()` 汇总全部 `CoeffAccess(name, level, row, col)`。
- `ROIBandAccess`：单个子带被 ROI 请求强制访问的坐标
  （`rows`/`cols` 子带内包围盒 + `coordinates` 精确点集）。

一维：`dwt1d(Sequence[int]) -> tuple[int, ...]`（前 ceil(n/2) 个为偶样本
近似，后 floor(n/2) 个为奇样本细节），`idwt1d` 为严格逆。
`max_levels(h, w)` 返回对该形状仍非恒等的最大层数。

异常：

- `IntegerLiftingError`：本包异常基类。
- `IntegerOverflowError(IntegerLiftingError, OverflowError)`：输入或受检中间
  整数绝对值超过 `2^63 - 1` 时抛出，**绝不截断或回绕**。
- `ROIError(IntegerLiftingError, ValueError)`：ROI 矩形非法（越界、空矩形、
  端点类型错误）；在任何系数回调被调用之前抛出。
- 非法输入抛 `TypeError` / `ValueError`，错误信息带可定位坐标，例如
  `image[1][2]`、`signal[3]`、`row 2`、`column 4`。

## 数学定义（本题契约子集）

对一维序列 `x`（长度 n），下标按奇偶样本划分：

1. 预测（奇样本/细节）：

   ```
   d[j] = x[2j+1] - floor((e[j] + e[j+1]) / 2),  e[k] = x[2k]
   ```

2. 更新（偶样本/近似）：

   ```
   s[k] = e[k] + floor((d[k-1] + d[k] + 2) / 4)
   ```

   其中 `/2`、`/4` 为**向下取整的数学 floor**（负数时与向零截断不同，
   实现用 Python `//`），输出为整数，全程无浮点。

3. 边界：缺失的同类邻居用**最近有效同类样本对称复制**（半样本对称/夹取，
   即对同类下标夹取到 `[0, count-1]`）：右端最后一个奇样本的右偶邻居取最后
   一个偶样本；两端更新所需的越界细节取最邻近的细节。

4. **长度 1 恒等**：长度为 1 的行/列变换直接复制；1×1 图像接受恰好 1 层。

二维：**先行后列**，对每个 1×2 子块（行、列）分别做一维变换，输出
LL / LH（行高频）/ HL（列高频）/ HH 四个子带。多层时**只递归 LL 区域**，
其余子带保持不动。逆变换**严格反序**：逐层从最深层开始，先列后行，
每个一维步为前向的精确整数逆运算。

可逆性来自提升结构：逆步用同样的 floor 公式抵消（前向加的项逆向减、
前向减的项逆向加），因此恢复误差恒为零。

## 输入上限

- 图像边长：`1 <= height, width <= 128`（`MAX_SIDE = 128`），必须为矩形。
- 层数：`levels >= 1` 且不超过 `max_levels(h, w)`；1×1 图像仅允许 1 层
  （恒等）。
- 数值：所有输入、输出及受检中间整数必须满足 `|v| <= 2^63 - 1`
  （`INT63_MAX`）。

## 误差口径

- **无浮点结果**：算法仅用整数与 floor 整除，正变换/逆变换均为精确整数
  运算，不存在舍入误差。
- 逐元素往返误差恒为 `0`（测试对全部列出的形状与层数断言逐位相等）；
  README 与测试均不以宽容差掩盖任何差异。
- 边界采用本题指定的“最近有效同类样本对称复制”，不是零延拓也不是周期
  延拓；不同边界约定会得到不同系数，这是契约的一部分而非误差。

## 设计取舍

- 输出使用**去交错 + 原地金字塔**布局：一维输出近似在前、细节在后；二维
  多层只改写 LL 区，系数矩阵尺寸始终等于原图，子带位置全部由元数据给出。
- 输入先整体校验并深拷贝到私有工作矩阵，再开始任何变换；因此校验失败或
  中途溢出时**不会产生部分结果，也绝不修改调用方数据**（无“未说明的部分
  变更”）。溢出在每个关键中间表达式（邻值和、+2、预测/更新结果）后立即
  检查，先抛错，不继续计算。
- 对输入的校验：拒绝 `bool`（尽管是 `int` 子类）、拒绝 `float`
  （`NaN`/`±Infinity` 给明确的 non-finite 报错，其他 float 报 TypeError）、
  拒绝非矩形/空矩阵/字符串伪装序列；`levels` 等整数参数同样拒绝 `bool`。
- 逆变换只接受本库 `dwt2d` 产出的 `TransformResult`，并重新校验形状、
  布局区间与每个系数，防止被篡改的元数据导致越界。
- 测试中的独立参考实现采用“交错下标 + 下标夹取”的另一套写法（见
  `tests/test_lifting.py` 顶部），**不调用被测核心**；手算小例与负数参考
  以常量形式写死。往返一致只作为补充测试。

## ROI 局部逆变换

`idwt2d_roi(result, rows, cols, read_coeff)` 只重建图像中半开矩形
`rows=(r0, r1)`、`cols=(c0, c1)`（`r0 <= r < r1`、`c0 <= c < c1`）内的
像素，返回形状为 `(r1-r0) x (c1-c0)` 的整数矩阵。它不是“完整重建后
裁剪”：实现从目标像素出发，按 5/3 提升的真实依赖关系（撤销更新需
近似系数与左右细节、撤销预测需细节与两个偶样本、多层时 LL 依赖
逐层向更深层传播）自顶向下推导每层所需系数集合，再自底向上做稀疏
逆变换。矩形之外的系数**既不读取也不置零**；边界扩展沿用与全量
逆变换完全相同的“最近有效同类样本夹取”规则，因此所需的 halo 可能
超出 ROI 本身（例如单像素 ROI 在 2×2 深层块上会依赖全部 4 个系数，
这是数学上的必要依赖而非实现浪费）。

参数与回调：

- `result`：`dwt2d` 返回的 `TransformResult`（形状/布局校验与
  `idwt2d` 相同；其 `coefficients` 不会被 ROI 路径读取）。
- `rows` / `cols`：`(start, end)` 半开区间，必须满足
  `0 <= start < end <= 对应边长`，且两端均为 `int`（拒绝 `bool`）。
- `read_coeff(name, level, row, col)`：只读回调，按子带名
  （`LL`/`LH`/`HL`/`HH`）、1-based 层数、**子带内局部坐标**返回整数
  系数。同一请求内每个所需坐标至多调用一次（请求内缓存吸收夹取
  带来的重复引用）；回调返回值按与全量变换相同的规则校验
  （拒绝 `bool`/`float`/`NaN`/`Infinity`，绝对值超 `2^63-1` 抛
  `IntegerOverflowError`）。

`roi_access_plan(result, rows, cols)` 返回纯几何的 `ROIPlan`：
`bands` 逐子带列出所需坐标的包围盒与精确点集（`coordinates()` 汇总
为 `CoeffAccess` 集合）。同一 ROI 几何与系数值无关，可用于预取、
审计或验证回调局部性。

错误状态（全部在**任何系数读取之前**抛出）：

- `ROIError(IntegerLiftingError, ValueError)`：ROI 区间越界、空矩形
  （`start >= end`）、非 `int` 端点、非二元组等非法矩形。
- `TypeError`：`result` 不是 `TransformResult`，或 `read_coeff`
  不可调用。
- `IntegerOverflowError`：回调返回的系数或受检中间量超出 int63
  范围，与全量逆变换的溢出语义完全一致（不截断、不回绕）。

局部性保证：回调只会被请求 `roi_access_plan` 所列坐标；测试用“拒绝
计划外访问”的严格回调验证了这一点（含单像素角落、跨子带边界、
宽高为 1 的形状）。

## 已知限制 / 范围声明

- 实现本题契约的数学子集：单层与多层 Mallat 式二维分解、上述对称夹取
  边界、int63 数值边界，以及按提升依赖推导的 ROI 局部逆变换（子带只读
  回调 + 静态访问计划）。**不宣称**与 JPEG 2000 等行业标准的完整兼容
  （不包含量化、渐进码流、码块、分量变换、规范规定的边界模式切换、
  图像浏览器等）；ROI 入口不是渐进式码流接口。
- 不包含二维非矩形、非整数、复数或浮点输入；超出 128×128 直接拒绝。
- 大绝对值数据可能因受检中间和超界而抛 `IntegerOverflowError`（这是显式
  边界，非静默截断）；典型图像量级远低于此。
- 纯 Python 实现面向正确性与可复用性，未做向量化/原生加速；128×128 规模
  下为亚秒级。

## 仓库内容

```
lifting53/__init__.py   # 公开 API
lifting53/_core.py      # 核心：校验、一维/二维分块、元数据、全量逆变换
lifting53/_roi.py       # ROI：依赖闭包推导、稀疏逆变换、静态访问计划
tests/test_lifting.py    # 独立参考、手算常量、边界/溢出/元数据/往返
tests/test_roi.py       # ROI：独立全量逆变换裁剪参考、严格局部性回调、边界
demo.py                 # 固定输入演示（含 ROI 局部还原）
requirements.txt        # 无第三方依赖（占位说明）
```
