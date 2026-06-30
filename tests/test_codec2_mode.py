"""test_codec2_mode.py — colegio.audio Codec2 is MODE-AWARE (default 3200).

The inscription voice codec moved from legacy 700C to mode 3200 (cleaner speech),
storing the mode in codec_meta = mode:u16 + n_frames:u32 so any decoder recovers
the geometry. This pins that: a 3200 round-trip through colegio.audio recovers the
mode + frame geometry, the meta layout is the 6-byte form, and a legacy 2-byte
meta still decodes as 700C. When the Colegio_Invisible oracle (voice_codec.py) is
importable it also asserts byte-identity. Skips cleanly without libcodec2/pycodec2.
"""
import os
import struct
import sys

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pycodec2")     # needs libcodec2 + the binding

import numpy as np

from colegio import audio

_HERE = os.path.dirname(os.path.abspath(__file__))
_CINV = os.path.abspath(os.path.join(_HERE, "..", "..", "Colegio_Invisible"))


def _voice(seconds=0.6, sr=8000):
    """A short deterministic voiced-ish signal (formant-like sum + light noise)."""
    t = np.arange(int(seconds * sr)) / sr
    x = (0.6 * np.sin(2 * np.pi * 140 * t)
         + 0.3 * np.sin(2 * np.pi * 700 * t)
         + 0.1 * np.sin(2 * np.pi * 1900 * t))
    rng = np.random.default_rng(0)
    x = x + 0.02 * rng.standard_normal(len(t))
    return (x / np.max(np.abs(x))).astype(np.float32)


def test_mode3200_roundtrip_geometry():
    """Encode at 3200 → the meta is mode:u16+n_frames:u32, and decode recovers
    mode 3200 with 160-sample frames."""
    x = _voice()
    header, body = audio.encode_sound_codec2(x, mode=3200, title="hermit")
    rec = audio.read_sound_quipu(header, body)
    assert len(rec["codec_meta"]) == 6, "3200 meta must be 6 bytes (mode:u16 + n_frames:u32)"
    mode, n_frames = struct.unpack(">HI", rec["codec_meta"][:6])
    assert mode == 3200
    pcm, meta = audio.decode_sound(header, body)
    assert meta["mode"] == 3200
    assert meta["n_frames"] == n_frames
    assert len(pcm) == n_frames * 160        # mode 3200 = 160 samples/frame
    assert len(body) == n_frames * 8         # mode 3200 = 8 bytes/frame


def test_legacy_700c_meta_still_decodes():
    """A legacy 2-byte (n_frames:u16) codec_meta is read as 700C (320-sample
    frames) — back-compat for anything inscribed before the mode-aware switch."""
    x = _voice()
    _h, body700 = audio.encode_sound_codec2(x, mode=700)   # body is 700C frames
    nf = max(1, (len(x) + 320 - 1) // 320)                  # 700C = ceil(len/320)
    header, body = audio.build_sound_quipu(
        audio.CODEC_CODEC2, body700, sample_rate=8000, channels=1,
        codec_meta=struct.pack(">H", nf), title="legacy")
    pcm, meta = audio.decode_sound(header, body)
    assert meta["mode"] == 700
    assert len(pcm) == nf * 320              # 700C = 320 samples/frame


@pytest.fixture(scope="module")
def oracle():
    """The Colegio_Invisible voice_codec oracle, if present (root import needs
    canonical/ on the path for its `sound` dependency). Oracle only."""
    if not os.path.isfile(os.path.join(_CINV, "voice_codec.py")):
        pytest.skip("Colegio_Invisible not present")
    for p in (_CINV, os.path.join(_CINV, "canonical")):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import voice_codec
    except Exception as exc:                  # pragma: no cover
        pytest.skip(f"voice_codec oracle not importable: {exc}")
    return voice_codec


def test_differential_vs_oracle(oracle):
    """colegio.audio Codec2 (mode 3200) must be byte-identical to the oracle, and
    decode to the same PCM."""
    x = _voice()
    mine = audio.encode_sound_codec2(x, mode=3200, title="hermit")
    theirs = oracle.encode_sound_codec2(x, mode=3200, title="hermit")
    assert mine == theirs, "codec2 3200 container bytes differ from the oracle"
    pm, _ = audio.decode_sound(*mine)
    po, _ = oracle.decode_voice_c2(*theirs)
    assert np.array_equal(pm, po), "codec2 3200 decode diverges from the oracle"
