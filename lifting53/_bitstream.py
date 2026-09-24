"""Deterministic bitstream codec for 5/3 lifting coefficients.

Adds, on top of the reversible transform in :mod:`lifting53._core`:

* an explicit sign bit for every non-zero coefficient;
* a fixed sign-magnitude bit-plane coding of the coefficient payload;
* a fixed, versioned binary header with reserved fields;
* exact integer recovery from the complete stream (forward then inverse is
  still bit-exact);
* decoding of a stream truncated at a *complete bit-plane boundary*, which
  yields a well-defined approximate image (missing low-order magnitude bits
  are read as zero).

Nothing here uses floats or third-party packages.

Binary format (all multi-byte integers are unsigned big-endian)
---------------------------------------------------------------

Header (exactly :data:`HEADER_SIZE` = 23 bytes):

=========  =====  =====================================================
offset     size   field
=========  =====  =====================================================
0          4      magic ``b"L53B"``
4          1      version (currently 1; any other value is rejected)
5          1      flags (reserved; must be 0)
6          2      height (1..128)
8          2      width  (1..128)
10         1      decomposition levels (>= 1; validated vs. shape)
11         1      max_width: magnitude bit width of the largest
                  coefficient (0..63; 0 means the image is all zero)
12         4      side_bits: length of the sign/width block in bits
16         4      plane_bytes: total size of all bit-plane segments in
                  bytes (each plane is separately byte-aligned)
20         3      reserved (must be zero)
=========  =====  =====================================================

Payload, a concatenation of byte-aligned segments (MSB-first packed bits,
each segment padded with zero bits to a byte boundary):

1. Side-information segment: ``ceil(side_bits / 8)`` bytes.  Coefficients
   are visited in a fixed scan order (see below).  Each coefficient
   contributes:

   * one bit ``0`` if the coefficient is zero (nothing follows);
   * otherwise bit ``1``, one sign bit (``1`` = negative, ``0`` =
     positive), then the magnitude bit width ``w`` (1..63) coded as
     ``e = w - 1`` (0..62): ``e`` in 0..30 is six direct bits; 31..62 is
     the reserved six-bit marker ``011111`` (the value 31) followed by
     six bits coding ``e - 31``.

2. One segment per bit-plane, from plane ``max_width - 1`` (most
   significant) down to plane 0.  Plane ``p`` carries magnitude bit ``p``
   of every coefficient whose width is greater than ``p``, in scan order,
   padded to a whole number of bytes.  The active count of every plane is
   derived from the coded widths, so a decoder knows each segment's exact
   byte length without an index.

Finally the complete stream carries a 4-byte CRC-32 (big-endian) computed
over all preceding bytes (header + payload).  A prefix ending on a
complete bit-plane segment boundary (CRC omitted) is a valid truncated
stream for :func:`decode_bitstream_partial`.

Scan order: the deepest LL band first, then for levels L..1 the bands
HL, LH, HH (raster order within each band).  Empty (zero-extent) bands
contribute nothing; every one of the ``height * width`` coefficient
positions occurs exactly once.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass

from ._core import (
    INT63_MAX,
    MAX_SIDE,
    IntegerLiftingError,
    TransformResult,
    _aggregate_subbands,
    _build_layouts,
    _check_levels,
    _validate_matrix,
    _validate_result,
)

#: Fixed header magic and version.
MAGIC = b"L53B"
VERSION = 1

#: Header size in bytes (see module docstring for the field table).
HEADER_SIZE = 23

#: Maximum coded magnitude bit width (|coefficient| <= 2^63 - 1).
MAX_WIDTH = INT63_MAX.bit_length()  # 63


class BitstreamError(IntegerLiftingError):
    """Raised on a corrupt, incomplete or non-conforming bitstream.

    Incomplete/corrupt headers, bad magic/version/reserved fields, failing
    CRCs, payloads that do not end on a complete bit-plane segment
    boundary and out-of-range header values all raise this error.
    Encoding and decoding are functional operations, so no caller-visible
    state is ever partially updated.
    """


# ---------------------------------------------------------------------------
# MSB-first bit packing
# ---------------------------------------------------------------------------

class _BitWriter:
    def __init__(self) -> None:
        self._bytes = bytearray()
        self._cur = 0
        self._nbits = 0
        self.total_bits = 0

    def bit(self, b: int) -> None:
        self._cur = (self._cur << 1) | (b & 1)
        self._nbits += 1
        self.total_bits += 1
        if self._nbits == 8:
            self._bytes.append(self._cur)
            self._cur = 0
            self._nbits = 0

    def bits(self, value: int, n: int) -> None:
        for k in range(n - 1, -1, -1):
            self.bit((value >> k) & 1)

    def data(self) -> bytes:
        if self._nbits:
            # Pad the final byte with zero bits.
            self._cur <<= 8 - self._nbits
            self._bytes.append(self._cur)
            self._cur = 0
            self._nbits = 0
        return bytes(self._bytes)


class _BitReader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._byte_pos = 0
        self._bit_pos = 0  # 0 = most significant bit of current byte

    def remaining(self) -> int:
        return len(self._data) * 8 - (self._byte_pos * 8 + self._bit_pos)

    def bit(self) -> int:
        if self._byte_pos >= len(self._data):
            raise BitstreamError("unexpected end of bitstream (missing bits)")
        b = (self._data[self._byte_pos] >> (7 - self._bit_pos)) & 1
        self._bit_pos += 1
        if self._bit_pos == 8:
            self._bit_pos = 0
            self._byte_pos += 1
        return b

    def bits(self, n: int) -> int:
        v = 0
        for _ in range(n):
            v = (v << 1) | self.bit()
        return v


# ---------------------------------------------------------------------------
# Width prefix coding (6-bit direct, or marker + 6 bits)
# ---------------------------------------------------------------------------

def _encode_width(width: int) -> tuple[int, int]:
    """Return (bits, bit_count) coding a width in 1..63."""
    e = width - 1
    if e < 31:
        return e, 6
    return (31 << 6) | (e - 31), 12


def _decode_width(reader: _BitReader) -> int:
    u = reader.bits(6)
    if u < 31:
        e = u
    else:
        e = 31 + reader.bits(6)
    width = e + 1
    if not 1 <= width <= MAX_WIDTH:
        raise BitstreamError(f"coded magnitude width {width} out of range 1..63")
    return width


# ---------------------------------------------------------------------------
# Scan order and plane geometry
# ---------------------------------------------------------------------------

def _scan_bands(result: TransformResult):
    """Yield (row, col) positions in the fixed codec scan order."""
    levels = result.levels
    order = [result.band("LL", levels)]
    for lv in range(levels, 0, -1):
        for name in ("HL", "LH", "HH"):
            order.append(result.band(name, lv))
    count = 0
    for band in order:
        r0, r1 = band.rows
        c0, c1 = band.cols
        for r in range(r0, r1):
            for c in range(c0, c1):
                yield r, c
                count += 1
    h, w = result.shape
    if count != h * w:
        # Invariant: the canonical bands partition the coefficient matrix.
        raise BitstreamError("internal error: scan does not cover the matrix")


def _plane_active_counts(widths: list[int], max_width: int) -> list[int]:
    """Active coefficient count in each plane, index 0 = LSB plane."""
    return [sum(1 for width in widths if width > p)
            for p in range(max_width)]


def _plane_byte_sizes(widths: list[int], max_width: int) -> list[int]:
    """Byte length of each plane segment, index 0 = LSB plane."""
    return [(count + 7) // 8
            for count in _plane_active_counts(widths, max_width)]


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_bitstream(result: TransformResult) -> bytes:
    """Encode a :class:`TransformResult` into the complete coded stream.

    The result is fully validated (type, canonical layouts and coefficient
    bounds) before any bytes are produced; no caller-visible state is
    modified.
    """
    result = _validate_result(result)
    # Re-validates every coefficient (type, bool/float, int63 bounds).
    _validate_matrix(result.coefficients, "coefficients")

    values = [result.coefficients[r][c] for r, c in _scan_bands(result)]

    # Side-information segment: zero/non-zero flag, sign, coded width.
    side = _BitWriter()
    for v in values:
        if v == 0:
            side.bit(0)
            continue
        side.bit(1)
        side.bit(1 if v < 0 else 0)
        width = abs(v).bit_length()
        if width > MAX_WIDTH:  # defensive; validation already enforces it
            raise BitstreamError(
                f"coefficient magnitude width {width} exceeds {MAX_WIDTH}"
            )
        coded, nbits = _encode_width(width)
        side.bits(coded, nbits)
    side_bits = side.total_bits
    side_segment = side.data()  # already zero-padded to a byte boundary

    widths = [(0 if v == 0 else abs(v).bit_length()) for v in values]
    max_width = max(widths, default=0)

    # One byte-aligned segment per magnitude plane, MSB plane first.
    plane_segments: list[bytes] = []
    for p in range(max_width - 1, -1, -1):
        plane = _BitWriter()
        for i, v in enumerate(values):
            if widths[i] > p:
                plane.bit((abs(v) >> p) & 1)
        plane_segments.append(plane.data())
    plane_bytes_total = sum(len(seg) for seg in plane_segments)

    h, w = result.shape
    header = bytearray(HEADER_SIZE)
    header[0:4] = MAGIC
    header[4] = VERSION
    header[5] = 0  # flags
    header[6:8] = h.to_bytes(2, "big")
    header[8:10] = w.to_bytes(2, "big")
    header[10] = result.levels
    header[11] = max_width
    header[12:16] = side_bits.to_bytes(4, "big")
    header[16:20] = plane_bytes_total.to_bytes(4, "big")
    # bytes 20..22 stay zero (reserved)

    out = bytes(header) + side_segment + b"".join(plane_segments)
    crc = zlib.crc32(out) & 0xFFFFFFFF
    return out + crc.to_bytes(4, "big")


# ---------------------------------------------------------------------------
# Header parsing / validation
# ---------------------------------------------------------------------------

def _parse_header(data: bytes) -> dict:
    if len(data) < HEADER_SIZE:
        raise BitstreamError(
            f"incomplete header: got {len(data)} bytes, "
            f"need at least {HEADER_SIZE}"
        )
    if data[0:4] != MAGIC:
        raise BitstreamError(
            f"bad magic {bytes(data[0:4])!r}; expected {MAGIC!r}"
        )
    version = data[4]
    if version != VERSION:
        raise BitstreamError(
            f"unsupported bitstream version {version}; supported: {VERSION}"
        )
    flags = data[5]
    if flags != 0:
        raise BitstreamError(f"non-zero reserved flags byte: {flags:#04x}")
    h = int.from_bytes(data[6:8], "big")
    w = int.from_bytes(data[8:10], "big")
    levels = data[10]
    max_width = data[11]
    side_bits = int.from_bytes(data[12:16], "big")
    plane_bytes = int.from_bytes(data[16:20], "big")
    reserved = data[20:23]
    if reserved != b"\x00\x00\x00":
        raise BitstreamError("non-zero reserved header bytes")
    if not 1 <= h <= MAX_SIDE or not 1 <= w <= MAX_SIDE:
        raise BitstreamError(
            f"header shape {h}x{w} outside 1..{MAX_SIDE} per side"
        )
    if levels < 1:
        raise BitstreamError(f"header levels {levels} must be >= 1")
    if levels > 7:
        raise BitstreamError(f"header levels {levels} implausibly large")
    # Shape/level compatibility (reuses the transform's own rule, including
    # the 1x1 identity-level exception).
    try:
        _check_levels(h, w, levels)
    except ValueError as exc:
        raise BitstreamError(f"incompatible header: {exc}") from None
    if not 0 <= max_width <= MAX_WIDTH:
        raise BitstreamError(
            f"header max_width {max_width} outside 0..{MAX_WIDTH}"
        )

    n = h * w
    # One bit per coefficient minimum; 14 bits maximum per non-zero token.
    if not n <= side_bits <= n * 14:
        raise BitstreamError(
            f"header side_bits {side_bits} inconsistent with {n} coefficients"
        )
    # Every plane segment holds at least one active coefficient bit per
    # byte except the last; total must fit n coefficients * max_width bits.
    if not 0 <= plane_bytes <= (n * MAX_WIDTH + 7) // 8 + MAX_WIDTH:
        raise BitstreamError(
            f"header plane_bytes {plane_bytes} implausible for {n} "
            f"coefficients of width <= {MAX_WIDTH}"
        )
    if max_width == 0 and plane_bytes != 0:
        raise BitstreamError(
            "header max_width=0 but plane_bytes is non-zero"
        )
    return {
        "h": h, "w": w, "levels": levels, "max_width": max_width,
        "side_bits": side_bits, "plane_bytes": plane_bytes,
    }


def _as_bytes(data: object) -> bytes:
    if isinstance(data, str):
        raise TypeError("bitstream must be bytes-like, not str")
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(
            "bitstream must be a bytes-like object "
            f"(got {type(data).__name__})"
        )
    return bytes(data)


# ---------------------------------------------------------------------------
# Side block parsing and coefficient reconstruction
# ---------------------------------------------------------------------------

def _parse_side(segment: bytes, count: int, side_bits: int,
                max_width: int) -> tuple[list[int], list[int]]:
    """Parse exactly ``side_bits`` bits into parallel sign/width arrays.

    Padding bits of the byte-aligned segment must be zero.
    """
    if len(segment) != (side_bits + 7) // 8:
        raise BitstreamError(
            f"side segment is {len(segment)} bytes, expected "
            f"{(side_bits + 7) // 8}"
        )
    reader = _BitReader(segment)
    signs = [0] * count
    widths = [0] * count
    for i in range(count):
        if reader.remaining() <= 0:
            raise BitstreamError(
                "side information is incomplete (token cut short)"
            )
        if reader.bit() == 0:
            continue
        if reader.remaining() < 7:
            raise BitstreamError(
                "side information is incomplete (sign/width cut short)"
            )
        signs[i] = -1 if reader.bit() else 1
        widths[i] = _decode_width(reader)
    if reader.remaining() != len(segment) * 8 - side_bits:
        raise BitstreamError("side block length mismatch (extra token bits)")
    spare = len(segment) * 8 - side_bits
    for _ in range(spare):
        if reader.bit() != 0:
            raise BitstreamError("non-zero side-segment padding bits")
    actual_max = max(widths, default=0)
    if actual_max != max_width:
        raise BitstreamError(
            f"max_width mismatch: header says {max_width}, tokens give "
            f"{actual_max}"
        )
    return signs, widths


def _build_result(h: int, w: int, levels: int, signs: list[int],
                  widths: list[int], plane_data: list[bytes] | None,
                  planes_available: list[bool]) -> TransformResult:
    """Reconstruct the coefficient matrix and a canonical TransformResult.

    ``plane_data[p]`` is the byte segment for plane ``p`` (index 0 = LSB);
    planes marked unavailable read as zero.
    """
    layouts = _build_layouts(h, w, levels)
    subbands = _aggregate_subbands(layouts, levels)
    result_shell = TransformResult(
        coefficients=((0,),), shape=(h, w), levels=levels,
        layouts=layouts, subbands=subbands,
    )
    positions = list(_scan_bands(result_shell))

    n = len(signs)
    mags = [0] * n
    if plane_data is not None:
        for p, segment in enumerate(plane_data):
            if not planes_available[p] or segment is None:
                continue
            active = sum(1 for width in widths if width > p)
            if len(segment) != (active + 7) // 8:
                raise BitstreamError(
                    f"plane {p} segment has {len(segment)} bytes, expected "
                    f"{(active + 7) // 8}"
                )
            reader = _BitReader(segment)
            for i in range(n):
                if widths[i] > p:
                    mags[i] |= reader.bit() << p
            for _ in range(len(segment) * 8 - active):
                if reader.bit() != 0:
                    raise BitstreamError(
                        f"non-zero padding bits in plane {p} segment"
                    )

    matrix = [[0] * w for _ in range(h)]
    for i, (r, c) in enumerate(positions):
        if widths[i] == 0:
            continue
        m = mags[i]
        if m.bit_length() > widths[i] or m > INT63_MAX:
            raise BitstreamError("reconstructed magnitude out of range")
        matrix[r][c] = signs[i] * m

    coeffs = tuple(tuple(row) for row in matrix)
    result = TransformResult(
        coefficients=coeffs, shape=(h, w), levels=levels,
        layouts=layouts, subbands=subbands,
    )
    # Full canonical validation + integer bounds on every decoded value.
    _validate_result(result)
    _validate_matrix(result.coefficients, "decoded coefficients")
    return result


# ---------------------------------------------------------------------------
# Strict decode
# ---------------------------------------------------------------------------

def decode_bitstream(data: object) -> TransformResult:
    """Decode a complete stream produced by :func:`encode_bitstream`.

    Requires the intact fixed header, the full side segment, every
    bit-plane segment and the trailing CRC-32.  Any corruption,
    truncation or trailing garbage raises :class:`BitstreamError`;
    nothing is returned on error.
    """
    raw = _as_bytes(data)
    head = _parse_header(raw)
    h, w = head["h"], head["w"]
    levels, max_width = head["levels"], head["max_width"]
    side_bits, plane_bytes = head["side_bits"], head["plane_bytes"]

    side_bytes_len = (side_bits + 7) // 8
    expected_len = HEADER_SIZE + side_bytes_len + plane_bytes + 4
    if len(raw) < expected_len:
        raise BitstreamError(
            f"incomplete stream: got {len(raw)} bytes, need {expected_len} "
            f"(header + {side_bytes_len} side + {plane_bytes} planes + 4 CRC)"
        )
    if len(raw) > expected_len:
        raise BitstreamError(
            f"trailing garbage after stream: {len(raw) - expected_len} "
            f"extra byte(s)"
        )

    body = raw[:HEADER_SIZE + side_bytes_len + plane_bytes]
    crc_stored = int.from_bytes(raw[expected_len - 4:expected_len], "big")
    crc_calc = zlib.crc32(body) & 0xFFFFFFFF
    if crc_stored != crc_calc:
        raise BitstreamError(
            f"CRC-32 mismatch: stored {crc_stored:#010x}, "
            f"computed {crc_calc:#010x}"
        )

    side_segment = raw[HEADER_SIZE:HEADER_SIZE + side_bytes_len]
    signs, widths = _parse_side(side_segment, h * w, side_bits, max_width)

    plane_sizes = _plane_byte_sizes(widths, max_width)
    if sum(plane_sizes) != plane_bytes:
        raise BitstreamError(
            f"plane_bytes mismatch: header says {plane_bytes}, coded widths "
            f"require {sum(plane_sizes)}"
        )

    plane_data: list[bytes | None] = [None] * max_width
    offset = HEADER_SIZE + side_bytes_len
    for p in range(max_width - 1, -1, -1):
        size = plane_sizes[p]
        plane_data[p] = raw[offset:offset + size]
        offset += size

    available = [True] * max_width
    return _build_result(h, w, levels, signs, widths, plane_data, available)


# ---------------------------------------------------------------------------
# Truncated (complete bit-plane boundary) decode
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PartialDecode:
    """Result of :func:`decode_bitstream_partial`.

    * ``result``: a fully valid :class:`TransformResult` built from the
      retained bit-planes; feed it to :func:`lifting53.idwt2d` for the
      approximate image.
    * ``complete``: True when every bit-plane is present (a CRC-bearing
      stream additionally passes its checksum, but completeness here is a
      plane-coverage property; use strict :func:`decode_bitstream` when the
      CRC must be enforced).
    * ``planes_retained``: number of most-significant bit-planes present
      (``max_width`` when complete; ``0`` only for an all-zero image).
    * ``max_width``: bit width of the largest coefficient (from the header).
    """

    result: TransformResult
    complete: bool
    planes_retained: int
    max_width: int


def decode_bitstream_partial(data: object) -> PartialDecode:
    """Decode a complete stream or one truncated at a plane boundary.

    The fixed header must be intact (any damaged/incomplete header is
    rejected exactly as in strict mode) and the whole byte-aligned side
    segment must be present.  After that, the payload must end exactly
    after some number ``k >= 1`` of complete most-significant bit-plane
    segments (each segment is byte aligned, so this boundary is
    unambiguous); ending inside a segment or before the first segment
    raises :class:`BitstreamError`.  Missing low-order planes decode as
    zero.

    A complete, CRC-bearing stream is accepted and fully CRC-verified.
    """
    raw = _as_bytes(data)
    head = _parse_header(raw)
    h, w = head["h"], head["w"]
    levels, max_width = head["levels"], head["max_width"]
    side_bits, plane_bytes = head["side_bits"], head["plane_bytes"]
    n = h * w

    side_bytes_len = (side_bits + 7) // 8
    full_len = HEADER_SIZE + side_bytes_len + plane_bytes + 4

    if len(raw) > full_len:
        raise BitstreamError(
            f"stream is longer than the header-described complete stream "
            f"({len(raw)} bytes > {full_len}); refusing to guess the cut"
        )

    crc_ok = len(raw) == full_len
    if crc_ok:
        body = raw[:full_len - 4]
        crc_stored = int.from_bytes(raw[full_len - 4:full_len], "big")
        if zlib.crc32(body) & 0xFFFFFFFF != crc_stored:
            raise BitstreamError("CRC-32 mismatch on complete stream")

    if len(raw) < HEADER_SIZE + side_bytes_len:
        raise BitstreamError(
            "side information is incomplete (stream cut inside the "
            "byte-aligned side segment)"
        )

    side_segment = raw[HEADER_SIZE:HEADER_SIZE + side_bytes_len]
    signs, widths = _parse_side(side_segment, n, side_bits, max_width)

    plane_sizes = _plane_byte_sizes(widths, max_width)
    if sum(plane_sizes) != plane_bytes:
        raise BitstreamError(
            f"plane_bytes mismatch: header says {plane_bytes}, coded widths "
            f"require {sum(plane_sizes)}"
        )

    if max_width == 0:
        if not crc_ok and len(raw) != HEADER_SIZE + side_bytes_len:
            raise BitstreamError("trailing bytes after an all-zero payload")
        result = _build_result(h, w, levels, signs, widths, None, [])
        # No planes exist, so "every plane" is vacuously present.
        return PartialDecode(result, True, 0, 0)

    # Walk plane segments (MSB first) and match the exact payload length.
    # For a CRC-verified stream the four CRC bytes are not payload.
    payload_end = full_len - 4 if crc_ok else len(raw)
    plane_data: list[bytes | None] = [None] * max_width
    available = [False] * max_width
    offset = HEADER_SIZE + side_bytes_len
    retained = 0
    for p in range(max_width - 1, -1, -1):
        size = plane_sizes[p]
        end = offset + size
        if payload_end < end:
            if retained == 0:
                raise BitstreamError(
                    "stream truncated before the end of the first complete "
                    "bit-plane segment"
                )
            raise BitstreamError(
                f"stream ends {end - payload_end} byte(s) inside bit-plane "
                f"segment {p}; truncation is only valid at a complete "
                f"bit-plane boundary"
            )
        plane_data[p] = raw[offset:end]
        available[p] = True
        retained += 1
        offset = end
        if payload_end == end:
            break

    if payload_end != offset:
        raise BitstreamError(
            f"payload length {payload_end} does not end on a bit-plane "
            f"segment boundary (next boundary at {offset} bytes)"
        )

    complete = retained == max_width
    result = _build_result(h, w, levels, signs, widths, plane_data, available)
    return PartialDecode(result, complete, retained, max_width)


# ---------------------------------------------------------------------------
# Deterministic truncation helper
# ---------------------------------------------------------------------------

def truncate_bitstream(data: object, keep_planes: int) -> bytes:
    """Return ``data`` cut after its first ``keep_planes`` bit-planes.

    ``keep_planes`` counts the most-significant planes retained and must
    satisfy ``1 <= keep_planes <= max_width`` (an all-zero image has no
    planes and cannot be truncated).  Because every plane segment is
    byte-aligned, the returned prefix is an exact byte-level cut: intact
    header, complete side segment and exactly the retained plane segments,
    with no CRC trailer.  It decodes with
    :func:`decode_bitstream_partial` and is rejected by strict
    :func:`decode_bitstream`.
    """
    raw = _as_bytes(data)
    head = _parse_header(raw)
    max_width = head["max_width"]
    side_bits = head["side_bits"]
    plane_bytes = head["plane_bytes"]
    if not isinstance(keep_planes, int) or isinstance(keep_planes, bool):
        raise TypeError("keep_planes must be int, bool is not accepted")
    if max_width == 0:
        raise ValueError("stream has no bit-planes (all-zero image)")
    if not 1 <= keep_planes <= max_width:
        raise ValueError(
            f"keep_planes must be in 1..{max_width}, got {keep_planes}"
        )

    side_bytes_len = (side_bits + 7) // 8
    payload_need = side_bytes_len + plane_bytes
    if len(raw) < HEADER_SIZE + payload_need:
        raise BitstreamError("source stream payload is incomplete")
    payload = raw[HEADER_SIZE:HEADER_SIZE + side_bytes_len]
    signs, widths = _parse_side(payload, head["h"] * head["w"],
                                side_bits, max_width)
    plane_sizes = _plane_byte_sizes(widths, max_width)
    if sum(plane_sizes) != plane_bytes:
        raise BitstreamError(
            f"plane_bytes mismatch: header says {plane_bytes}, coded widths "
            f"require {sum(plane_sizes)}"
        )

    kept = sum(plane_sizes[max_width - 1 - k] for k in range(keep_planes))
    return bytes(raw[:HEADER_SIZE + side_bytes_len + kept])
