"""iwt53 —— 离散图像的可逆整数 5/3 提升小波变换（本题题库）。

公开接口见 :mod:`iwt53.core`。纯 Python 标准库，无第三方依赖，无浮点运算。
"""

from .core import (
    INT64_MAX,
    MAX_SIDE,
    IntegerOverflowError,
    InvalidImageError,
    InvalidParameterError,
    SizeLimitError,
    Subband,
    TransformLayout,
    TransformResult,
    WaveletError,
    forward_1d,
    forward_2d,
    inverse_1d,
    inverse_2d,
    max_levels,
)

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

__version__ = "1.0.0"
