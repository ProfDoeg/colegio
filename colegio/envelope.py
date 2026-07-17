"""envelope — the only header bytes the node parses (everything else is format).

Every quipu cabeza (strand 0) opens with 6 type-agnostic bytes:

    c1dd 0001   magic + version
    <type>      1 byte
    <tone>      1 byte

…and then a TYPE-SPECIFIC tail. The text family (text/essay/latex/scene) uses
`|title|key=value|`, but image carries binary dimensions, encrypted a subfamily
byte + ciphertext, celestial/cert/identity their own structures. A cross-check
of the whole on-chain corpus confirmed it: title and fields are NOT universal —
39 of 65 quipu put something other than a pipe-title right after byte 6.

So the node parses only what is genuinely universal — magic, version, type,
tone — and returns the raw header strand. The client (`canonical/*`) decodes
title/fields per type. This is the `header` of `quipuread`.
"""

MAGIC = b"\xc1\xdd"
VERSION_V1 = 0x0001
VERSION_V2 = 0x0002


def parse_envelope(header_bytes):
    """Parse the universal head of a quipu cabeza.

    Returns {magic, version, type, tone, raw} — the four type-agnostic fields
    plus the raw header-strand hex. Title and fields are deliberately absent:
    they are type-specific, and the client decodes `raw` per `type`.

    The magic is TWO bytes; the version is data at bytes 2..3 (u16-BE) and
    every version parses — deciding which versions to dispatch is a later
    layer's job (canonical/envelope.py, c1dd0002 §3). The old 4-byte
    MAGIC+VERSION compare rejected every non-v1 blob as bad magic.

    Raises ValueError if the magic is missing or the header is too short.
    """
    b = bytes(header_bytes)
    if len(b) < 6:
        raise ValueError(f"header too short: {len(b)} bytes (need >= 6)")
    if b[:2] != MAGIC:
        raise ValueError(
            f"not a quipu (magic c1dd missing; got {b[:2].hex()})")
    return {
        "magic": b[:2].hex(),
        "version": int.from_bytes(b[2:4], "big"),
        "type": b[4],
        "tone": b[5],
        "raw": b.hex(),
    }
