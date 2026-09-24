"""lifting53 — reversible integer 5/3 lifting wavelet transform for 2D images.

Public API
----------
dwt2d(image, levels=1) -> TransformResult
idwt2d(result) -> tuple of tuples (the reconstructed integer image)
dwt1d(signal) / idwt1d(coefficients)
max_levels(height, width)
Subband, LevelLayout, TransformResult
IntegerLiftingError, IntegerOverflowError

Embedded bitstream (deterministic sign bits + bit-plane coding)
---------------------------------------------------------------
encode_bitstream(result) -> bytes
decode_bitstream(data, allow_truncated=False) -> DecodeResult
truncate_to_bitplanes(stream, planes) -> bytes
DecodeResult, BitstreamError, CorruptBitstreamError, UnsupportedFormatError
HEADER_MAGIC, BITSTREAM_VERSION, HEADER_SIZE, BLOCK_SIDE

Constants: INT63_MAX = 2**63 - 1, MAX_SIDE = 128.
"""
from ._codec import (
    BITSTREAM_VERSION,
    BLOCK_SIDE,
    HEADER_MAGIC,
    HEADER_SIZE,
    BitstreamError,
    CorruptBitstreamError,
    DecodeResult,
    UnsupportedFormatError,
    decode_bitstream,
    encode_bitstream,
    truncate_to_bitplanes,
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
    "IntegerLiftingError",
    "IntegerOverflowError",
    "LevelLayout",
    "Subband",
    "TransformResult",
    "dwt1d",
    "dwt2d",
    "idwt1d",
    "idwt2d",
    "max_levels",
    "BITSTREAM_VERSION",
    "BLOCK_SIDE",
    "HEADER_MAGIC",
    "HEADER_SIZE",
    "BitstreamError",
    "CorruptBitstreamError",
    "UnsupportedFormatError",
    "DecodeResult",
    "encode_bitstream",
    "decode_bitstream",
    "truncate_to_bitplanes",
]

__version__ = "1.1.0"
