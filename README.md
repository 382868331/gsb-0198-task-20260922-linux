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
# 必要测试（独立参考 + 手算常量 + 往返补充）
./.venv/bin/python -m unittest discover -s tests -v

# 固定输入演示（真实计算：多层全量/ROI 结果与三个实际触发的拒绝边界）
./.venv/bin/python demo.py
```

隔离环境用 Linux 原生 Python 3.14.7 建立，仅需标准库：

```bash
/opt/task-python-3.14.7/bin/python3.14 -m venv .venv
```

demo 为小规模固定输入（7×5 两层及一个 11×9 四层 ROI 例等），真实计算、
无 `sleep`、无预录结果，正常机器上耗时远小于 1 秒（题面预算约 90 秒内）。

## 接口

```python
from lifting53 import (
    dwt2d, idwt2d, dwt1d, idwt1d, max_levels,
    idwt2d_roi, make_reader,
)

result = dwt2d(image, levels=1)
# image: Sequence[Sequence[int]]，矩形整数矩阵，不被修改
# result: TransformResult
reconstructed = idwt2d(result)          # 逐位恢复原矩阵

# 只还原一个半开矩形 ROI（先行后列的严格逆序，按 5/3 依赖逐层推导系数）
reader = make_reader(result)            # 只读回调，自动记录 accessed 坐标
pixels = idwt2d_roi((h, w), levels, (r0, r1, c0, c1), reader)
# pixels: r0..r1-1 行、c0..c1-1 列的像素；reader.accessed 为实际读取的
# (子带名, 层, 行, 列) 列表（坐标为完整系数矩阵坐标）
```

ROI 逆变换：

- `idwt2d_roi(shape, levels, roi, read)`
  - `shape=(height, width)`、`levels` 与正变换相同（奇数宽高、多层 LL
    尺寸均支持，1×1 接受恰好 1 层恒等）。
  - `roi=(r0, r1, c0, c1)` 为**半开**矩形：
    `0 <= r0 < r1 <= height`、`0 <= c0 < c1 <= width`。
  - `read(name, level, row, col) -> int` 为只读回调：`name` 为
    `"LL"`（仅最深层）、`"LH"`、`"HL"`、`"HH"`；`(row, col)` 是系数
    矩阵中的绝对坐标，与 `Subband.rows/cols` 一致。每个所需坐标在一次
    请求内至多读取一次；同一请求内允许缓存，矩形依赖之外的系数**不会**
    被读取，也不会被当作零。
  - 实现按 5/3 逆提升的真实依赖自顶向下推导每层所需的近似/细节下标
    （含夹取复制的同类邻居），再自底向上只重建该 halo；不是“完整重建后
    裁剪”。长度 1 的行/列走恒等，溢出规则（int63、不截断）与全量逆变换
    完全一致。
  - **必要 halo 可能比 ROI 大**：单个像素需要其提升邻域的近似与细节，
    多层时这些需求逐层向上传播；跨子带边界的 ROI 还会同时读取相邻多个
    子带。这是依赖关系决定的，不是冗余读取。
- `make_reader(result)` 返回 `CoefficientReader`：对 `TransformResult`
  的只读回调，`accessed` 列表记录每次实际访问；读取坐标越出该子带声明
  矩形、或在非最深层请求 `LL`（内部工作区而非存储子带）时抛
  `ValueError`，可直接用作局部性守卫。自定义回调只要满足同样的
  `(name, level, row, col)` 约定即可。

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

一维：`dwt1d(Sequence[int]) -> tuple[int, ...]`（前 ceil(n/2) 个为偶样本
近似，后 floor(n/2) 个为奇样本细节），`idwt1d` 为严格逆。
`max_levels(h, w)` 返回对该形状仍非恒等的最大层数。

异常：

- `IntegerLiftingError`：本包异常基类。
- `IntegerOverflowError(IntegerLiftingError, OverflowError)`：输入或受检中间
  整数绝对值超过 `2^63 - 1` 时抛出，**绝不截断或回绕**。
- ROI 入口的错误状态（全部在任何回调读取**之前**完成结构性校验，非法
  请求不产生任何读取）：
  - `TypeError`：`shape`/`levels`/`roi` 不是整数（`bool` 拒绝）、
    `read` 不可调用、回调返回非整数（含 `float`/`NaN`/`Infinity`，
    规则同输入校验）。
  - `ValueError`：ROI 越界或为空（`r0 >= r1`、`c0 >= c1`、端点超出
    图像）、边长不在 `1..128`、`levels` 超过 `max_levels`、`roi` 不是
    4 元素序列；回调返回的系数在重建中间表达式触发 int63 溢出时抛
    `IntegerOverflowError`，与全量逆变换同一检查点。
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

## 已知限制 / 范围声明

- 仅实现本题契约的数学子集：单层与多层 Mallat 式二维分解、上述对称夹取
  边界、int63 数值边界。**不宣称**与 JPEG 2000 等行业标准的完整兼容
  （不包含码流、分量变换、规范规定的边界模式切换等；像素域半开矩形
  ROI 局部逆变换已由 `idwt2d_roi` 提供，但不是 JPEG 2000 式渐进码流 ROI）。
- 不包含二维非矩形、非整数、复数或浮点输入；超出 128×128 直接拒绝。
- ROI 功能是“给定系数后的像素域局部逆变换”，不包含渐进码流、图像浏览
  器或任意形状 ROI；ROI 始终为轴对齐半开矩形。
- 大绝对值数据可能因受检中间和超界而抛 `IntegerOverflowError`（这是显式
  边界，非静默截断）；典型图像量级远低于此。
- 纯 Python 实现面向正确性与可复用性，未做向量化/原生加速；128×128 规模
  下为亚秒级。

## 仓库内容

```
lifting53/__init__.py   # 公开 API
lifting53/_core.py      # 校验、一维提升、二维分块、元数据、全量/ROI 逆变换
tests/test_lifting.py   # 独立参考、手算常量、边界/溢出/元数据/往返测试
tests/test_roi.py       # ROI：独立参考、角点/跨子带/1×N/全部边界、局部性
demo.py                 # 固定输入演示（含 ROI 局部还原与访问统计）
requirements.txt        # 无第三方依赖（占位说明）
```
