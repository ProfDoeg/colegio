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
VERSION = b"\x00\x01"


def parse_envelope(header_bytes):
    """Parse the universal head of a quipu cabeza.

    Returns {magic, version, type, tone, raw} — the four type-agnostic fields
    plus the raw header-strand hex. Title and fields are deliberately absent:
    they are type-specific, and the client decodes `raw` per `type`.

    Raises ValueError if the magic is missing or the header is too short.
    """
    b = bytes(header_bytes)
    if b[:4] != MAGIC + VERSION:
        raise ValueError("not a quipu (c1dd0001 magic missing from header)")
    if len(b) < 6:
        raise ValueError(f"header too short: {len(b)} bytes (need >= 6)")
    return {
        "magic": b[:2].hex(),
        "version": int.from_bytes(b[2:4], "big"),
        "type": b[4],
        "tone": b[5],
        "raw": b.hex(),
    }
