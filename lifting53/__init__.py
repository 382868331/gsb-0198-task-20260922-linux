"""lifting53 — reversible integer 5/3 lifting wavelet transform for 2D images.

Public API
----------
Transform:
  dwt2d(image, levels=1) -> TransformResult
  idwt2d(result) -> tuple of tuples (the reconstructed integer image)
  dwt1d(signal) / idwt1d(coefficients)
  max_levels(height, width)

Bitstream codec (sign bit + bit-plane coding, fixed header, CRC):
  encode_bitstream(result) -> bytes
  decode_bitstream(data) -> TransformResult           (strict, complete)
  decode_bitstream_partial(data) -> PartialDecode     (truncated planes OK)
  truncate_bitstream(data, keep_planes) -> bytes
  PartialDecode, BitstreamError

Data/errors:
  Subband, LevelLayout, TransformResult
  IntegerLiftingError, IntegerOverflowError

Constants: INT63_MAX = 2**63 - 1, MAX_SIDE = 128.
"""
from ._bitstream import (
    BitstreamError,
    PartialDecode,
    decode_bitstream,
    decode_bitstream_partial,
    encode_bitstream,
    truncate_bitstream,
)
from ._core import (
    INT63_MAX,
    MAX_SIDE,
    IntegerLiftingError,
    IntegerOverflowError,
    LevelLayout,
    Subband,
    TransformResult,
    dwt1d,
    dwt2d,
    idwt1d,
    idwt2d,
    max_levels,
)

__all__ = [
    "INT63_MAX",
    "MAX_SIDE",
    "BitstreamError",
    "IntegerLiftingError",
    "IntegerOverflowError",
    "LevelLayout",
    "PartialDecode",
    "Subband",
    "TransformResult",
    "decode_bitstream",
    "decode_bitstream_partial",
    "dwt1d",
    "dwt2d",
    "encode_bitstream",
    "idwt1d",
    "idwt2d",
    "max_levels",
    "truncate_bitstream",
]

__version__ = "1.1.0"
