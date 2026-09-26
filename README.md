# lifting53 — 离散图像的可逆整数 5/3 提升变换

纯 Python 标准库实现的二维可逆整数 5/3 提升小波变换（Le Gall 5/3，整数提升
形式），含严格逆变换、子带布局元数据、多层递归与显式溢出边界。核心算法
全部离线、无浮点、无第三方依赖。

## 运行环境

- **Python 版本**：Python 3.14.7（仅使用 3.14 标准库；`from __future__ import
  annotations`，无第三方包）。
- **依赖**：无。`requirements.txt` 仅为占位说明。安装 Python 后直接运行，
  不需要网络、账号、密钥、硬件、数据库、中间件、云 API 或 Docker。

## 隔离 venv 准备

不需要任何第三方包，venv 仅用于隔离解释器环境：

```bash
python3.14 -m venv .venv          # 需要 Linux 原生 Python 3.14.7
. .venv/bin/activate              # 或直接使用 .venv/bin/python
python -V                         # 应显示 Python 3.14.7
```

## 准确命令

```bash
# 必要测试（基线回归 + 自适应增量：独立参考 + 枚举核对 + 往返补充）
python -m unittest discover -s tests -v

# 基线固定输入演示（真实计算，含一个正常多层结果与两个实际触发的拒绝边界）
python demo.py

# 自适应四叉树分解演示（真实离线计算，预算 90 秒，实测远低于 1 秒）
python demo_iteration.py
```

demo 为小规模固定输入（7×5 两层等），真实计算、无 `sleep`、无预录结果，
正常机器上耗时远小于 1 秒（题面预算约 8 秒内）。

## 接口

```python
from lifting53 import dwt2d, idwt2d, dwt1d, idwt1d, max_levels

result = dwt2d(image, levels=1)
# image: Sequence[Sequence[int]]，矩形整数矩阵，不被修改
# result: TransformResult
reconstructed = idwt2d(result)          # 逐位恢复原矩阵
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

一维：`dwt1d(Sequence[int]) -> tuple[int, ...]`（前 ceil(n/2) 个为偶样本
近似，后 floor(n/2) 个为奇样本细节），`idwt1d` 为严格逆。
`max_levels(h, w)` 返回对该形状仍非恒等的最大层数。

异常：

- `IntegerLiftingError`：本包异常基类。
- `IntegerOverflowError(IntegerLiftingError, OverflowError)`：输入或受检中间
  整数绝对值超过 `2^63 - 1` 时抛出，**绝不截断或回绕**。
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

## 自适应四叉树分解（增量功能）

基线 `dwt2d` 只递归分解 LL 低频块。增量接口 `dwt2d_adaptive` 允许
**四个子带（LL/LH/HL/HH）各自继续做同一 5/3 提升**，构建到给定深度
`depth` 的完整四叉树后，用调用方提供的**每块代价**自底向上剪枝：每个
内部节点在“保留父块”与“展开为四个（已递归剪枝的）子块”之间取总代价
更小者，**并列时优先浅层表示**（保留父块）。由于各子树相互独立，该
自底向上过程对整棵树是最优的；测试用小树穷举全部剪枝选择核对了最优性。

```python
from lifting53 import dwt2d_adaptive, idwt2d_adaptive, integer_coding_length

result = dwt2d_adaptive(image, depth=3)            # 默认代价
result = dwt2d_adaptive(image, depth=3, cost=my_cost)  # 调用方代价
# my_cost(block) -> 非负 int；block 为 tuple[tuple[int, ...], ...]
image2 = idwt2d_adaptive(result)                   # 按树精确还原
```

公开数据结构：

- `AdaptiveResult`：`root`（剪枝后的完整分解树）、`shape`、`depth`、
  `total_cost`（剪枝表示的总代价）、`cost_description`；`leaves()`
  返回全部叶节点。
- `AdaptiveNode`：`band`（`"ROOT"` 或 LL/LH/HL/HH）、`depth`（根为 0）、
  `shape`、`coefficients`（本节点自己的块，叶节点即被保留的系数）、
  `children`（叶为 `None`，否则恰好 4 个、按 LL/LH/HL/HH 顺序、形状为
  父块的精确象限划分）。各叶系数与形状均可直接从树读出。
- `integer_coding_length(block)`：**默认代价，显式整数编码长度估计**——
  每个系数估计 `1 + bit_length(|v|)` 比特。它只是用于比较表示优劣的
  确定估计，**不是也不宣称是任何真实压缩文件的大小**。

契约要点：

- 奇数边长与 1×N / N×1 / 1×1 直接按提升定义处理（长度 1 恒等、空子带为
  零样本叶子），**不靠补零**，因此逆变换也无需裁切。
- `idwt2d_adaptive` 先校验整棵树再重建：缺子块（子节点数不为 4）、子块
  形状不等于父块象限划分（重叠或留缝）、叶系数形状不符、1×1 或空块携带
  子节点、深度/带名不符等非法树一律抛 `InvalidTreeError` 拒绝。
- `depth >= 1` 且不超过 `max(1, max_levels(h, w))`；代价函数必须返回
  非负 `int`（拒绝 `bool`/`float`/负数）。

## 已知限制 / 范围声明

- 仅实现本题契约的数学子集：单层与多层 Mallat 式二维分解、上述对称夹取
  边界、int63 数值边界。**不宣称**与 JPEG 2000 等行业标准的完整兼容
  （不包含量化、ROI、码流、分量变换、规范规定的边界模式切换等）。
- 不包含二维非矩形、非整数、复数或浮点输入；超出 128×128 直接拒绝。
- 大绝对值数据可能因受检中间和超界而抛 `IntegerOverflowError`（这是显式
  边界，非静默截断）；典型图像量级远低于此。
- 纯 Python 实现面向正确性与可复用性，未做向量化/原生加速；128×128 规模
  下为亚秒级。
- 自适应分解的默认代价是显式整数编码长度**估计**，用于在合法表示之间做
  选择；它不对应任何真实压缩器输出，本库也不产出压缩文件格式。
- 剪枝最优性针对“每个节点二选一（保留/展开）”的代价模型；代价函数必须
  是纯函数式的每块非负整数代价。

## 仓库内容

```
lifting53/__init__.py   # 公开 API
lifting53/_core.py      # 基线核心：校验、一维提升、二维分块、元数据、逆变换
lifting53/_adaptive.py  # 增量：四叉树自适应分解、代价剪枝、树校验逆变换
tests/test_lifting.py   # 基线回归：独立参考、手算常量、边界/溢出/元数据/往返
tests/test_adaptive.py  # 增量：独立参考逐层核对、小树枚举最优性、非法树拒绝
demo.py                 # 基线固定输入演示
demo_iteration.py       # 自适应分解演示（真实离线计算，< 90 秒）
requirements.txt        # 无第三方依赖（占位说明）
```
