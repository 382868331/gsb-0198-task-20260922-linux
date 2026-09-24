# lifting53 — 离散图像的可逆整数 5/3 提升变换与嵌入式码流

纯 Python 标准库实现的二维可逆整数 5/3 提升小波变换（Le Gall 5/3，整数提升
形式），含严格逆变换、子带布局元数据、多层递归与显式溢出边界；并在变换
系数之上提供**确定性符号位 + 位平面嵌入式码流**：完整码流逐整数还原原图，
截断到完整位平面可解出渐进近似图，固定头格式带 CRC，拒绝损坏或不完整头。
核心算法全部离线、无浮点、无第三方依赖。

## 运行环境

- **Python 版本**：Python 3.14.7（仅使用 3.14 标准库；`from __future__ import
  annotations`，无第三方包）。
- **依赖**：无。`requirements.txt` 仅为占位说明。安装 Python 后直接运行，
  不需要网络、账号、密钥、硬件、数据库、中间件、云 API 或 Docker。

## 准确命令

```bash
# 全部测试（独立参考 + 手算常量 + 往返补充 + 码流编码/截断/损坏头）
python -m unittest discover -s tests -v

# 固定输入演示（正常多层结果、渐进位平面近似，以及溢出/NaN/损坏头、
# 不完整码流等实际触发的拒绝边界）
python demo.py
```

demo 为小规模固定输入（7×5 两层变换 + 9×7 两层码流等），真实计算、无
`sleep`、无预录结果，正常机器上耗时约 0.04 秒（题面预算 90 秒内）。

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

### 嵌入式码流（符号位 + 位平面）

```python
from lifting53 import encode_bitstream, decode_bitstream, truncate_to_bitplanes

stream = encode_bitstream(result)          # bytes，完整自描述码流
out = decode_bitstream(stream)             # 严格模式：必须完整且 CRC 正确
assert out.exact and out.image == original_image
partial = decode_bitstream(prefix, allow_truncated=True)  # 截断模式
```

- `encode_bitstream(TransformResult) -> bytes`：先整体校验（形状、层数、
  规范化布局、int63 系数），再编码；不修改入参，失败不返回半成品。
- `decode_bitstream(data, *, allow_truncated=False) -> DecodeResult`，接受
  `bytes` / `bytearray` / `memoryview`：
  - 默认严格：坏魔数、坏头 CRC、维度/层数/块尺寸非法、载荷 CRC 不匹配、
    长度与实际位段不符、尾部多余字节、非零保留位/填充位一律抛
    `CorruptBitstreamError`；版本不支持抛 `UnsupportedFormatError`。
  - `allow_truncated=True`：仅允许**短于**头声明长度的码流，且必须结束在
    某个完整位平面边界（在一个位平面中间切断时，该不完整平面被整体丢弃，
    回滚到前一完整平面）；头损坏、载荷翻转、尾部多余字节仍拒绝。
- `truncate_to_bitplanes(stream, planes) -> bytes`：对**完整**码流（含 CRC
  校验）求恰好包含前 `planes` 个完整位平面的字节前缀；`planes` 为非负
  `int`（拒绝 `bool`/浮点/越界）。请求全部平面时原样返回完整码流。
- `DecodeResult`（冻结 dataclass）字段：
  - `image`：逆变换后的还原图；`result`：解出的系数 `TransformResult`；
  - `planes_encoded`：头中各块声明的全局最大位平面数；
    `planes_decoded`：本次实际解出的完整位平面数；
  - `exact`：是否逐整数还原；`truncated` 属性 =
    `planes_decoded < planes_encoded`；
  - `stream_len`、`payload_len`：实际收到的字节数与头声明的载荷长度。

异常：`BitstreamError(Exception)` 为基类；
`CorruptBitstreamError(BitstreamError, ValueError)` 用于损坏/截断/坏头；
`UnsupportedFormatError(BitstreamError)` 用于未知版本。

#### 码流格式（版本 1，全部多字节整数为无符号大端）

固定 28 字节头：

| 偏移 | 长度 | 字段 |
|---:|---:|---|
| 0 | 4 | 魔数 `b"L53B"` |
| 4 | 1 | 版本，当前为 1 |
| 5 | 1 | 标志位，必须为 0 |
| 6 | 2 | 高 `height`（1..128） |
| 8 | 2 | 宽 `width`（1..128） |
| 10 | 1 | 层数（同 `dwt2d` 规则，1..127） |
| 11 | 1 | 块位移，固定为 3（8×8 系数块） |
| 12 | 4 | 载荷字节长度 `payload_len` |
| 16 | 4 | 载荷 `zlib.crc32` |
| 20 | 4 | 保留，必须全 0 |
| 24 | 4 | 头字节 0..23 的 `zlib.crc32` |

头后是 `payload_len` 字节载荷和 4 字节载荷 CRC（大端）。载荷各段独立按
字节对齐（零位填充）：

1. **块计数段**：按光栅顺序，每个 8×8 块（边缘按图像裁剪）6 位无符号数
   给出该块系数量值所需的位平面数（0..63）；
2. 每个全局位平面 `p = 0..max-1` 一段；仅含该平面的块，对块内单元按
   光栅顺序写 1 位量值位（MSB 在前）；该单元第一次出现量值 1 时，紧跟
   1 位**显式符号位**（0 正、1 负），之后的平面不再重复符号。

因此“前 p 个完整位平面”是码流的精确字节前缀；解码端把缺失低位补零，
再走普通整数逆变换，得到近似图，随平面增加误差单调不增、最终为 0。
常量：`HEADER_MAGIC = b"L53B"`、`BITSTREAM_VERSION = 1`、
`HEADER_SIZE = 28`、`BLOCK_SIDE = 8`。

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
- 码流测试的期望值来自三处，均不调用被测编码器：逐位手算的字节常量
  （含 `-5` 的显式负号位）、测试文件内用“位字符串拼包”实现的独立打包器
  （与核心的整数累加打包是不同写法）、以及按题面独立实现的逆变换参考；
  截断结果另与“逐块量值低位清零”的解析关系逐系数核对。

## 已知限制 / 范围声明

- 仅实现本题契约的数学子集：单层与多层 Mallat 式二维分解、上述对称夹取
  边界、int63 数值边界，以及自描述的符号位 + 位平面嵌入式码流。**不宣称**
  与 JPEG 2000 等行业标准的完整兼容（不包含量化、ROI、码率控制、分量
  变换、EBCOT 上下文算术编码、规范规定的边界模式切换等）。
- 本码流的渐进粒度是**全局完整位平面**：不提供随机局部解码、跨块索引、
  码率联合优化（本次迭代明确不做这三项）。近似只来自低位补零，没有额外
  量化步长参数。
- 8×8 块在边缘裁剪；位平面计数为每块 6 位的固定头开销，系数块边界与
  小波子带边界相互独立（块仅用于嵌入顺序，不影响变换数学）。
- 不包含二维非矩形、非整数、复数或浮点输入；超出 128×128 直接拒绝。
- 大绝对值数据可能因受检中间和超界而抛 `IntegerOverflowError`（这是显式
  边界，非静默截断）；典型图像量级远低于此。
- 纯 Python 实现面向正确性与可复用性，未做向量化/原生加速；128×128 规模
  下编码/解码为亚秒级。

## 仓库内容

```
lifting53/__init__.py   # 公开 API（变换 + 嵌入式码流）
lifting53/_core.py      # 核心：校验、一维提升、二维分块、元数据、逆变换
lifting53/_codec.py     # 固定头/CRC、显式符号位、8x8 块位平面编码与截断解码
tests/test_lifting.py   # 独立参考、手算常量、边界/溢出/元数据/往返测试
tests/test_codec.py     # 手算字节常量、独立拼包器、位平面截断、坏头拒绝
demo.py                 # 固定输入演示（变换 + 码流）
requirements.txt        # 无第三方依赖（占位说明）
```
