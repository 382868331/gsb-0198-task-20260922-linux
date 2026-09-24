# lifting53 — 离散图像的可逆整数 5/3 提升变换

纯 Python 标准库实现的二维可逆整数 5/3 提升小波变换（Le Gall 5/3，整数提升
形式），含严格逆变换、子带布局元数据、多层递归与显式溢出边界；并提供
**确定性符号位 + 位平面码流编解码**：固定版本化二进制头、每个非零系数
显式符号位、MSB 位平面分段、CRC-32 校验，完整码流逐整数还原原图，截断
到完整位平面可解出近似图，损坏/不完整头一律拒绝。核心算法全部离线、
无浮点、无第三方依赖。

## 运行环境

- **Python 版本**：Python 3.14.7（仅使用 3.14 标准库；`from __future__ import
  annotations`，无第三方包）。
- **依赖**：无。`requirements.txt` 仅为占位说明。安装 Python 后直接运行，
  不需要网络、账号、密钥、硬件、数据库、中间件、云 API 或 Docker。

## 准确命令

```bash
# 全部测试（独立参考 + 手算常量/手算字节 + 往返补充）
python -m unittest discover -s tests -v

# 固定输入演示（真实计算：正常多层结果、溢出/NaN 拒绝、
# 完整码流逐位还原、逐位平面截断近似、坏头/CRC 拒绝）
python demo.py
```

demo 为小规模固定输入（7×5 两层等），真实计算、无 `sleep`、无预录结果，
正常机器上耗时约 0.1 秒（题面预算 90 秒内）。

## 接口

```python
from lifting53 import dwt2d, idwt2d, dwt1d, idwt1d, max_levels
from lifting53 import (encode_bitstream, decode_bitstream,
                      decode_bitstream_partial, truncate_bitstream)

result = dwt2d(image, levels=1)
# image: Sequence[Sequence[int]]，矩形整数矩阵，不被修改
# result: TransformResult
reconstructed = idwt2d(result)          # 逐位恢复原矩阵
stream = encode_bitstream(result)       # 符号位+位平面码流（bytes）
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

## 码流编解码（本轮新增）

```python
from lifting53 import (encode_bitstream, decode_bitstream,
                      decode_bitstream_partial, truncate_bitstream,
                      BitstreamError)

stream: bytes = encode_bitstream(result)        # result 来自 dwt2d
result2 = decode_bitstream(stream)              # 严格完整解码 + CRC 校验
image2  = idwt2d(result2)                       # 逐整数还原原图

cut   = truncate_bitstream(stream, keep_planes=1)   # 只留最高 1 个位平面
part  = decode_bitstream_partial(cut)               # 截断码流也能解
approx = idwt2d(part.result)                        # 近似图（整数）
# part.complete / part.planes_retained / part.max_width
```

公开入口（均为无状态函数；输入不被修改，任何失败都不产生半更新）：

- `encode_bitstream(TransformResult) -> bytes`：编码前完整校验类型、规范
  布局与每个系数（含 int63 边界），校验失败直接抛错、不产出任何字节。
- `decode_bitstream(bytes-like) -> TransformResult`：**严格**解码。要求头、
  边信息、所有位平面、CRC 全部完整正确；截断、损坏、尾部多字节一律抛
  `BitstreamError`。
- `decode_bitstream_partial(bytes-like) -> PartialDecode`：接受完整码流，
  或**恰好截断在某个完整位平面段边界**的前缀；缺的低位平面按 0 解，得到
  合法 `TransformResult`，可直接喂 `idwt2d`。头损坏/不完整、边信息缺失、
  切在某个位平面段中间，一律抛 `BitstreamError`。
- `truncate_bitstream(stream, keep_planes) -> bytes`：确定性地生成保留最高
  `keep_planes` 个位平面的字节前缀（去 CRC）。`keep_planes` 必须为
  `1..max_width` 的 `int`（拒绝 `bool`/`float`/越界）；全零图无位平面，
  截断抛 `ValueError`。
- `PartialDecode`（冻结 dataclass）：
  - `result: TransformResult` — 由保留位平面重建的系数（始终通过完整校验）。
  - `complete: bool` — 所有位平面是否都在（平面覆盖性质；是否过 CRC 请用
    严格 `decode_bitstream`）。
  - `planes_retained: int` — 实际保留的最高位平面数。
  - `max_width: int` — 头中记录的最大系数量化宽度（全零图为 0）。
- `BitstreamError(IntegerLiftingError)`：码流损坏/不完整/不合规时抛出。

### 固定二进制格式

多字节整数一律无符号大端。头固定 **23 字节**：

| 偏移 | 长度 | 字段 |
|---|---|---|
| 0 | 4 | 魔数 `b"L53B"` |
| 4 | 1 | 版本（当前 1，其余拒绝） |
| 5 | 1 | flags（保留，必须 0） |
| 6 | 2 | 高 `height`（1..128） |
| 8 | 2 | 宽 `width`（1..128） |
| 10 | 1 | 分解层数（≥1，并按形状合法性校验） |
| 11 | 1 | `max_width`：最大系数量化宽度（0..63；0 表示全零图） |
| 12 | 4 | `side_bits`：符号/宽度块的位数 |
| 16 | 4 | `plane_bytes`：所有位平面段总字节数 |
| 20 | 3 | 保留（必须 0） |

净荷为若干**各自按字节对齐**的段（高位在前 MSB-first，段内不足 8 位用
0 填充）：

1. 边信息段：按固定扫描序（最深 LL 优先，再由深层到浅层 HL、LH、HH，
   带内光栅序）逐系数。零系数写 1 个 `0`；非零系数写 `1` + 符号位
   （1=负，0=正）+ 量化宽度 `w`（1..63）：令 `e=w-1`，`e` 在 0..30 用
   6 位直写，31..62 用保留标记 6 位 `011111`（即 31）后再接 6 位
   `e-31`。
2. 位平面段：从平面 `max_width-1`（最高）到平面 0，每段只含该平面上
   “宽度大于该平面号”的系数的幅度位，扫描序同上；每段独立补齐到整字节，
   因此**位平面边界就是无歧义的字节边界**。各平面的有效位数完全由解出的
   宽度推导，无需任何索引。

整流末尾附 4 字节 CRC-32（大端），覆盖头 + 全部净荷。截断前缀不含 CRC。

### 截断近似的解析关系

丢掉低 `d` 个位平面，等价于在系数域对每个非零幅度做**保号的向下掩膜**
`m' = (|c| >> d) << d`、符号不变、零仍为零。因此随保留平面数增加，逐
系数绝对误差单调不增；保留全部平面时误差为 0、`idwt2d` 逐整数还原。
像素域误差经逆提升混合，不保证逐像素单调，但每个完整前缀都能解出一幅
确定的整数近似图。

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

- 实现题面限定的数学子集与本轮码流：单层与多层 Mallat 式二维分解、
  对称夹取边界、int63 数值边界，以及本包自定义的符号位 + 位平面格式
  （固定头、逐段字节对齐、CRC-32）。**不宣称**与 JPEG 2000 等行业标准
  完整兼容（不含量化、ROI、分量变换、算术编码、规范规定的边界模式
  切换等）；码流格式是本库版本 1 的自有格式。
- 本轮**只做**“确定符号位 + 位平面、完整还原、完整位平面截断近似、固定
  头与损坏/不完整头拒绝”。不包含随机局部解码、跨块索引、码率联合优化。
- 不包含二维非矩形、非整数、复数或浮点输入；超出 128×128 直接拒绝。
- 大绝对值数据可能因受检中间和超界而抛 `IntegerOverflowError`（这是显式
  边界，非静默截断）；典型图像量级远低于此。
- 截断近似只保证在**完整位平面段边界**可解；切在段中间、边信息不完整、
  头损坏均报错而非猜测。`decode_bitstream_partial` 的 `complete` 表示
  位平面覆盖完整，不代表已过 CRC；需要防损坏请用严格 `decode_bitstream`。
- 纯 Python 实现面向正确性与可复用性，未做向量化/原生加速；128×128
  七层 + 编解码在普通机器上约 0.1 秒量级。

## 仓库内容

```
lifting53/__init__.py       # 公开 API
lifting53/_core.py          # 校验、一维提升、二维分块、规范布局、逆变换
lifting53/_bitstream.py     # 符号位+位平面编码、固定头、CRC、截断解码
tests/test_lifting.py       # 变换：独立参考、手算常量、边界/溢出/元数据
tests/test_bitstream.py     # 码流：手算字节、独立解码器、截断/损坏用例
demo.py                     # 固定输入演示（变换 + 码流 + 真实拒绝边界）
requirements.txt            # 无第三方依赖（占位说明）
```
