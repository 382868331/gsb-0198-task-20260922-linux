"""Deterministic embedded bitstream for the reversible integer 5/3 transform.

This module serialises a :class:`~lifting53._core.TransformResult` into a
fixed, self-describing byte stream and reconstructs it exactly.  It adds two
things on top of the raw integer coefficients:

* **Explicit sign bits.**  Every non-zero coefficient carries one sign bit
  (0 = positive, 1 = negative), emitted immediately after the magnitude bit at
  which the coefficient first becomes significant.  There is no implicit
  sign convention anywhere in the stream.
* **Bit-plane embedded coding.**  Within each fixed 8x8 coefficient block the
  magnitude of every coefficient is written MSB-first, one global bit-plane at
  a time.  A decoder that only receives the prefix through the last *complete*
  bit-plane recovers the high-order magnitude bits (unknown low bits read as
  zero), runs the ordinary integer inverse transform, and therefore produces a
  genuine lossy approximation that converges to the exact image.

On-disk layout (all multi-byte integers are unsigned big-endian)::

    offset  size  field
    0       4     magic        b"L53B"
    4       1     version      1
    5       1     flags        0 (all bits reserved, must be zero)
    6       2     height       1..128
    8       2     width        1..128
    10      1     levels       1..127 (same levels rule as dwt2d)
    11      1     block_shift  3  -> 8x8 coefficient blocks
    12      4     payload_len  byte length of the coded payload
    16      4     payload_crc  zlib.crc32 over the payload bytes
    20      4     reserved     zero
    24      4     header_crc   zlib.crc32 over header bytes 0..23
    (header is 28 bytes, followed by payload_len payload bytes and 4 CRC bytes)

Payload sections, each padded with zero bits to a byte boundary:

1. block count section: for every 8x8 block (raster order), 6 bits giving the
   number of magnitude bit-planes in that block (0..63);
2. one section per global bit-plane p = 0..max_planes-1; each block that has a
   plane p contributes, for every cell in raster order, one magnitude bit, and
   when the cell is first found significant its one sign bit immediately
   follows.

The per-section byte alignment makes "the prefix through p complete
bit-planes" an exact byte prefix of the stream.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass

from ._core import (
    INT63_MAX,
    MAX_SIDE,
    TransformResult,
    _build_layouts,
    _validate_matrix,
    _validate_result,
    idwt2d,
)

#: Magic bytes identifying an L53 bitstream.
HEADER_MAGIC = b"L53B"

#: Supported bitstream format version.
BITSTREAM_VERSION = 1

#: Fixed header size in bytes.
HEADER_SIZE = 28

#: Coefficient blocks are 2**BLOCK_SHIFT squares clipped to the image edge.
BLOCK_SHIFT = 3
BLOCK_SIDE = 1 << BLOCK_SHIFT

#: Width in bits of the per-block plane count field.
COUNT_BITS = 6
MAX_PLANES = 63  # coefficient magnitudes are bounded by 2**63 - 1

_RESERVED_BYTES = 4


class BitstreamError(Exception):
    """Base class for bitstream-related errors."""


class CorruptBitstreamError(BitstreamError, ValueError):
    """Raised when a bitstream is corrupt, truncated or has a bad header."""


class UnsupportedFormatError(BitstreamError):
    """Raised when a well-formed header requests an unsupported format."""


# ---------------------------------------------------------------------------
# Bit-level IO
# ---------------------------------------------------------------------------

class _EndOfBits(Exception):
    """Internal: a read ran past the end of the available payload bytes."""


class _BitWriter:
    """MSB-first bit packer; sections are explicitly byte-aligned."""

    __slots__ = ("buf", "acc", "nbits")

    def __init__(self) -> None:
        self.buf = bytearray()
        self.acc = 0
        self.nbits = 0

    def write(self, value: int, n: int) -> None:
        if n:
            self.acc = (self.acc << n) | (value & ((1 << n) - 1))
            self.nbits += n
            while self.nbits >= 8:
                self.nbits -= 8
                self.buf.append((self.acc >> self.nbits) & 0xFF)

    def align(self) -> None:
        if self.nbits:
            self.buf.append((self.acc << (8 - self.nbits)) & 0xFF)
            self.acc = 0
            self.nbits = 0

    def byte_len(self) -> int:
        return len(self.buf)

    def get_bytes(self) -> bytes:
        self.align()
        return bytes(self.buf)


class _BitReader:
    """MSB-first bit reader over a fixed bytes-like payload."""

    __slots__ = ("data", "total_bits", "pos")

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.total_bits = len(data) * 8
        self.pos = 0

    def read(self, n: int) -> int:
        if n == 0:
            return 0
        if self.pos + n > self.total_bits:
            self.pos = self.total_bits
            raise _EndOfBits
        value = 0
        for _ in range(n):
            byte = self.data[self.pos >> 3]
            bit = (byte >> (7 - (self.pos & 7))) & 1
            value = (value << 1) | bit
            self.pos += 1
        return value

    def byte_pos(self) -> int:
        return (self.pos + 7) // 8


# ---------------------------------------------------------------------------
# Geometry / validation helpers
# ---------------------------------------------------------------------------

def _blocks(h: int, w: int) -> list[list[tuple[int, int]]]:
    """Raster-ordered 8x8 blocks, clipped at the coefficient matrix edge."""
    blocks: list[list[tuple[int, int]]] = []
    for br in range(0, h, BLOCK_SIDE):
        for bc in range(0, w, BLOCK_SIDE):
            cells = [
                (r, c)
                for r in range(br, min(br + BLOCK_SIDE, h))
                for c in range(bc, min(bc + BLOCK_SIDE, w))
            ]
            blocks.append(cells)
    return blocks


def _check_levels(h: int, w: int, levels: int) -> None:
    """Mirror the levels validation rule used by ``dwt2d``."""
    limit = 0
    hh, ww = h, w
    while hh > 1 or ww > 1:
        hh = (hh + 1) // 2
        ww = (ww + 1) // 2
        limit += 1
    if levels < 1:
        raise CorruptBitstreamError(f"levels must be >= 1, got {levels}")
    if levels > limit and not (limit == 0 and levels == 1):
        raise CorruptBitstreamError(
            f"levels={levels} exceeds max_levels({h}, {w})={limit}"
        )


def _prepare_coefficients(
    result: TransformResult,
) -> tuple[list[list[int]], int, int]:
    """Validate a TransformResult for serialising; private working copy."""
    result = _validate_result(result)
    a = _validate_matrix(result.coefficients, "coefficients")
    h, w = len(a), len(a[0])
    if (h, w) != result.shape:
        raise ValueError(
            f"coefficient matrix shape {h}x{w} does not match recorded "
            f"shape {result.shape[0]}x{result.shape[1]}"
        )
    if len(result.layouts) != result.levels:
        raise ValueError("malformed TransformResult: layouts/levels mismatch")
    canonical = _build_layouts(h, w, result.levels)
    if tuple(result.layouts) != canonical:
        raise ValueError(
            "TransformResult layouts do not match the canonical layout for "
            f"shape {h}x{w} levels={result.levels}"
        )
    return a, h, w


def _result_from_coeffs(a: list[list[int]], h: int, w: int,
                        levels: int) -> TransformResult:
    layouts = _build_layouts(h, w, levels)
    aggregate = tuple(
        b
        for layout in layouts
        for b in layout.bands
        if not (b.name == "LL" and layout.level < levels)
    )
    coeffs = tuple(tuple(row) for row in a)
    return TransformResult(coeffs, (h, w), levels, layouts, aggregate)


# ---------------------------------------------------------------------------
# Header (de)serialisation
# ---------------------------------------------------------------------------

def _build_header(height: int, width: int, levels: int,
                  payload: bytes) -> bytes:
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    body = b"".join((
        HEADER_MAGIC,
        bytes((BITSTREAM_VERSION, 0)),
        height.to_bytes(2, "big"),
        width.to_bytes(2, "big"),
        bytes((levels, BLOCK_SHIFT)),
        len(payload).to_bytes(4, "big"),
        crc.to_bytes(4, "big"),
        b"\x00" * _RESERVED_BYTES,
    ))
    assert len(body) == HEADER_SIZE - 4
    return body + (zlib.crc32(body) & 0xFFFFFFFF).to_bytes(4, "big")


def _parse_header(data: bytes) -> tuple[int, int, int, int, int]:
    """Validate the fixed header; return (h, w, levels, payload_len, crc)."""
    if len(data) < HEADER_SIZE:
        raise CorruptBitstreamError(
            f"incomplete header: got {len(data)} bytes, "
            f"need at least {HEADER_SIZE}"
        )
    if data[0:4] != HEADER_MAGIC:
        raise CorruptBitstreamError(
            f"bad magic {data[0:4]!r}, expected {HEADER_MAGIC!r}"
        )
    stored_crc = int.from_bytes(data[24:28], "big")
    if (zlib.crc32(data[0:24]) & 0xFFFFFFFF) != stored_crc:
        raise CorruptBitstreamError("header CRC mismatch; header is corrupt")

    version = data[4]
    if version != BITSTREAM_VERSION:
        raise UnsupportedFormatError(
            f"unsupported bitstream version {version}, "
            f"supported version is {BITSTREAM_VERSION}"
        )
    flags = data[5]
    if flags != 0:
        raise CorruptBitstreamError(f"undefined flag bits set: 0x{flags:02x}")
    if data[20:24] != b"\x00" * _RESERVED_BYTES:
        raise CorruptBitstreamError("non-zero reserved header bytes")

    h = int.from_bytes(data[6:8], "big")
    w = int.from_bytes(data[8:10], "big")
    levels = data[10]
    block_shift = data[11]
    if not (1 <= h <= MAX_SIDE and 1 <= w <= MAX_SIDE):
        raise CorruptBitstreamError(
            f"header dimensions {h}x{w} outside 1..{MAX_SIDE}"
        )
    if levels < 1 or levels > 127:
        raise CorruptBitstreamError(f"header levels field invalid: {levels}")
    _check_levels(h, w, levels)
    if block_shift != BLOCK_SHIFT:
        raise CorruptBitstreamError(
            f"unsupported block_shift {block_shift}, expected {BLOCK_SHIFT}"
        )
    payload_len = int.from_bytes(data[12:16], "big")
    payload_crc = int.from_bytes(data[16:20], "big")
    return h, w, levels, payload_len, payload_crc


# ---------------------------------------------------------------------------
# Payload coding
# ---------------------------------------------------------------------------

def _encode_payload(a: list[list[int]], h: int, w: int) -> bytes:
    blocks = _blocks(h, w)
    magnitudes = [[abs(v) for v in row] for row in a]
    negative = [[v < 0 for v in row] for row in a]

    plane_counts: list[int] = []
    for cells in blocks:
        k = 0
        for r, c in cells:
            mb = magnitudes[r][c].bit_length()
            if mb > k:
                k = mb
        if k > MAX_PLANES:  # pragma: no cover - inputs are int63 bounded
            raise CorruptBitstreamError(f"coefficient needs {k} planes")
        plane_counts.append(k)
    max_planes = max(plane_counts, default=0)

    wr = _BitWriter()
    for k in plane_counts:
        wr.write(k, COUNT_BITS)
    wr.align()

    significant = [[False] * w for _ in range(h)]
    for p in range(max_planes):
        for bi, cells in enumerate(blocks):
            k = plane_counts[bi]
            if k <= p:
                continue
            bit_index = k - 1 - p
            for r, c in cells:
                mb = (magnitudes[r][c] >> bit_index) & 1
                wr.write(mb, 1)
                if mb and not significant[r][c]:
                    wr.write(1 if negative[r][c] else 0, 1)
                    significant[r][c] = True
        wr.align()
    return wr.get_bytes()


@dataclass(frozen=True)
class DecodeResult:
    """Outcome of :func:`decode_bitstream`.

    ``image`` is the inverse-transformed reconstruction; ``result`` is the
    decoded coefficient matrix with its canonical layout metadata;
    ``planes_decoded < planes_encoded`` means the stream was truncated at a
    complete-bit-plane boundary and the image is an approximation.
    """

    image: tuple[tuple[int, ...], ...]
    result: TransformResult
    stream_len: int
    payload_len: int
    planes_encoded: int
    planes_decoded: int
    exact: bool

    @property
    def truncated(self) -> bool:
        return self.planes_decoded < self.planes_encoded


def _decode_payload(payload: bytes, h: int, w: int, levels: int, *,
                    allow_truncated: bool, stream_len: int,
                    payload_len_declared: int, intact: bool) -> DecodeResult:
    blocks = _blocks(h, w)
    reader = _BitReader(payload)

    try:
        counts: list[int] = []
        for _ in blocks:
            k = reader.read(COUNT_BITS)
            if k > MAX_PLANES:
                raise CorruptBitstreamError(
                    f"block plane count {k} exceeds the limit {MAX_PLANES}"
                )
            counts.append(k)
        # Consume the count section's zero padding.
        pad = (-reader.pos) & 7
        reader.read(pad)
    except _EndOfBits:
        raise CorruptBitstreamError(
            "incomplete payload: block count section is missing bits"
        ) from None
    max_planes = max(counts, default=0)

    magnitudes = [[0] * w for _ in range(h)]
    negative = [[False] * w for _ in range(h)]
    significant = [[False] * w for _ in range(h)]

    planes_decoded = 0
    for p in range(max_planes):
        # A plane either commits fully or is discarded entirely, so a stream
        # cut inside a plane behaves exactly as if that plane were absent.
        snap_mag = [row[:] for row in magnitudes]
        snap_neg = [row[:] for row in negative]
        snap_sig = [row[:] for row in significant]
        section_start = reader.pos
        try:
            for bi, cells in enumerate(blocks):
                k = counts[bi]
                if k <= p:
                    continue
                bit_index = k - 1 - p
                for r, c in cells:
                    mb = reader.read(1)
                    if mb:
                        magnitudes[r][c] |= 1 << bit_index
                        if not significant[r][c]:
                            negative[r][c] = bool(reader.read(1))
                            significant[r][c] = True
            data_end = reader.pos
            pad = (-data_end) & 7
            if pad:
                padding = reader.read(pad)
                if intact and padding != 0:
                    raise CorruptBitstreamError(
                        f"non-zero padding bits after bit-plane {p}"
                    )
        except _EndOfBits:
            magnitudes, negative, significant = snap_mag, snap_neg, snap_sig
            reader.pos = section_start
            if not allow_truncated:
                raise CorruptBitstreamError(
                    f"incomplete payload: bit-plane {p} is truncated"
                ) from None
            break
        planes_decoded += 1

    if planes_decoded == 0 and max_planes > 0 and intact:
        raise CorruptBitstreamError(
            "payload declares bit-planes but contains no complete plane"
        )

    # An intact stream must consume exactly the declared payload bytes.
    if intact and reader.byte_pos() != payload_len_declared:
        raise CorruptBitstreamError(
            "payload length does not match the coded bit sections"
        )

    values = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            mag = magnitudes[r][c]
            if mag > INT63_MAX:
                raise CorruptBitstreamError(
                    f"decoded coefficient at ({r},{c}) exceeds +/- (2^63-1)"
                )
            values[r][c] = -mag if negative[r][c] else mag

    result = _result_from_coeffs(values, h, w, levels)
    image = idwt2d(result)
    return DecodeResult(
        image=image,
        result=result,
        stream_len=stream_len,
        payload_len=payload_len_declared,
        planes_encoded=max_planes,
        planes_decoded=planes_decoded,
        exact=intact and planes_decoded == max_planes,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode_bitstream(result: TransformResult) -> bytes:
    """Encode a 5/3 :class:`TransformResult` into the fixed bitstream format.

    The input is fully validated first; the function either returns a complete
    ``bytes`` value or raises, and never mutates its argument.
    """
    a, h, w = _prepare_coefficients(result)
    _check_levels(h, w, result.levels)
    payload = _encode_payload(a, h, w)
    crc = (zlib.crc32(payload) & 0xFFFFFFFF).to_bytes(4, "big")
    return _build_header(h, w, result.levels, payload) + payload + crc


def decode_bitstream(data: bytes | bytearray | memoryview, *,
                     allow_truncated: bool = False) -> DecodeResult:
    """Decode an L53 bitstream.

    With ``allow_truncated=False`` (default) the stream must be byte-exact:
    fixed header, header CRC, payload CRC and length all have to match,
    otherwise :class:`CorruptBitstreamError` is raised.

    With ``allow_truncated=True`` a stream that is *shorter* than its header
    declares is accepted as long as it ends on a complete-bit-plane boundary;
    the resulting :class:`DecodeResult` reports ``exact=False`` and the number
    of planes decoded.  Corruption (bad magic/CRC, flipped payload bytes,
    trailing data) is still rejected in that mode.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(
            "bitstream must be bytes-like (bytes/bytearray/memoryview), "
            f"got {type(data).__name__}"
        )
    data = bytes(data)

    h, w, levels, payload_len, payload_crc = _parse_header(data)
    available = len(data) - HEADER_SIZE

    if available > payload_len + 4:
        raise CorruptBitstreamError(
            f"trailing bytes after bitstream (declared {payload_len} payload "
            f"bytes + 4 CRC bytes, got {available})"
        )

    intact = available == payload_len + 4
    if intact:
        payload = data[HEADER_SIZE:HEADER_SIZE + payload_len]
        if (zlib.crc32(payload) & 0xFFFFFFFF) != payload_crc:
            raise CorruptBitstreamError(
                "payload CRC mismatch; payload is corrupt"
            )
    elif available >= payload_len:
        # The whole payload is present but the 4-byte CRC is missing or
        # damaged: there is no plane-prefix interpretation, and accepting it
        # would mean trusting unverified bytes.
        raise CorruptBitstreamError(
            f"incomplete bitstream: payload is present but its 4-byte CRC "
            f"is missing/truncated (got {available}, need {payload_len + 4})"
        )
    else:
        if not allow_truncated:
            raise CorruptBitstreamError(
                f"incomplete bitstream: header declares {payload_len + 4} "
                f"payload/CRC bytes but {available} are present"
            )
        payload = data[HEADER_SIZE:]

    return _decode_payload(
        payload, h, w, levels,
        allow_truncated=allow_truncated,
        stream_len=len(data),
        payload_len_declared=payload_len,
        intact=intact,
    )


def truncate_to_bitplanes(stream: bytes | bytearray | memoryview,
                          planes: int) -> bytes:
    """Return the exact byte prefix containing the first ``planes`` planes.

    Sections are byte-aligned, so the returned prefix decodes with exactly
    ``planes`` complete bit-planes under ``allow_truncated=True``.  ``planes``
    may be 0 (the prefix then ends after the block count section).  The intact
    input stream is fully validated (header and payload CRC included).
    """
    if not isinstance(stream, (bytes, bytearray, memoryview)):
        raise TypeError(
            "stream must be bytes-like, got "
            f"{type(stream).__name__}"
        )
    if isinstance(planes, bool) or not isinstance(planes, int):
        raise TypeError("planes must be int, got "
                        f"{type(planes).__name__}")
    stream = bytes(stream)
    h, w, levels, payload_len, payload_crc = _parse_header(stream)
    if len(stream) != HEADER_SIZE + payload_len + 4:
        raise CorruptBitstreamError(
            "truncate_to_bitplanes requires an intact, complete stream"
        )
    payload = stream[HEADER_SIZE:HEADER_SIZE + payload_len]
    if (zlib.crc32(payload) & 0xFFFFFFFF) != payload_crc:
        raise CorruptBitstreamError("payload CRC mismatch; stream is corrupt")

    blocks = _blocks(h, w)
    reader = _BitReader(payload)
    counts = [reader.read(COUNT_BITS) for _ in blocks]
    pad = (-reader.pos) & 7
    reader.read(pad)
    max_planes = max(counts, default=0)
    if planes < 0 or planes > max_planes:
        raise ValueError(
            f"planes must be in 0..{max_planes}, got {planes}"
        )
    significant: set[tuple[int, int]] = set()
    for p in range(planes):
        for bi, cells in enumerate(blocks):
            k = counts[bi]
            if k <= p:
                continue
            for cell in cells:
                mb = reader.read(1)
                if mb and cell not in significant:
                    reader.read(1)  # sign bit, only at first significance
                    significant.add(cell)
        pad = (-reader.pos) & 7
        reader.read(pad)

    # Asking for every plane is not a truncation: return the intact stream
    # (with its trailing CRC) so strict decoding succeeds as well.
    if planes == max_planes:
        return stream
    return stream[:HEADER_SIZE + reader.byte_pos()]
