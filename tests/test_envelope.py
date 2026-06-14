"""Universal envelope parser — the four type-agnostic fields + raw header.

Title/fields are deliberately NOT parsed here (they are type-specific; the
client decodes `raw`). The corpus cross-check (test_corpus_groundtruth) proves
type/tone are correct against the on-chain catalog for every quipu.
"""

import pytest

from colegio.envelope import parse_envelope


def test_byte_exact_onchain_mi_perrito():
    # canonical text self-test asserts this header is byte-identical on chain
    h = b"\xc1\xdd\x00\x01\x00\x00|Mi Perrito|"
    env = parse_envelope(h)
    assert env == {"magic": "c1dd", "version": 1, "type": 0x00, "tone": 0x00,
                   "raw": h.hex()}
    assert "title" not in env and "fields" not in env   # raw-only contract


def test_universal_fields_across_types():
    # type/tone read from fixed byte positions, regardless of the tail's grammar
    for typ, tone in [(0x03, 0xff), (0x0e, 0x03), (0xce, 0x01), (0x5c, 0x02),
                      (0x01, 0x00), (0xcc, 0x02)]:
        h = b"\xc1\xdd\x00\x01" + bytes([typ, tone]) + b"\x00\x01binarytail\xff"
        env = parse_envelope(h)
        assert env["type"] == typ and env["tone"] == tone
        assert env["magic"] == "c1dd" and env["version"] == 1


def test_raw_round_trips():
    h = b"\xc1\xdd\x00\x01\x5c\x02" + \
        "|The Descent|engine=xelatex|author=El Ermitaño|".encode("utf-8")
    env = parse_envelope(h)
    assert bytes.fromhex(env["raw"]) == h          # raw is the exact header bytes


def test_minimum_length_ok():
    env = parse_envelope(b"\xc1\xdd\x00\x01\x00\xff")   # exactly 6 bytes
    assert env["type"] == 0x00 and env["tone"] == 0xff and env["raw"] == "c1dd000100ff"


def test_magic_missing_or_short_raises():
    with pytest.raises(ValueError):
        parse_envelope(b"\x00\x00\x00\x01\x00\x00|x|")
    with pytest.raises(ValueError):
        parse_envelope(b"\xc1\xdd\x00\x01\x00")        # 5 bytes, too short
