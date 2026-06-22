"""Differential gate: colegio.audio ≡ the Colegio_Invisible monolith oracle.

audio.py is the clean-package AUDIO media codec (mirrors imaging.py). It houses
the numpy vocoder DSP AND the keyless 0x07 sound container, SELF-CONTAINED — it
imports nothing from the monolith. This test proves byte-identity against the
monolith oracle (canonical/sound.py + voice_codec.py): the container header+body
and the vocoder-encoded bodies must be byte-for-byte identical.

The monolith is imported ONLY here (oracle), by adding Colegio_Invisible and its
canonical/ subdir to sys.path. audio.py itself stays import-clean of the
monolith. Skips cleanly when the sibling Colegio_Invisible repo isn't present.
"""
import os
import sys

import numpy as np
import pytest

from colegio import audio

_HERE = os.path.dirname(os.path.abspath(__file__))
_CINV = os.path.abspath(os.path.join(_HERE, "..", "..", "Colegio_Invisible"))
_CANON = os.path.join(_CINV, "canonical")


@pytest.fixture(scope="module")
def oracle():
    """Import the monolith oracle modules (canonical/sound.py + voice_codec.py).
    Oracle ONLY — audio.py never imports these."""
    if not (os.path.isdir(_CINV) and os.path.isfile(os.path.join(_CANON, "sound.py"))):
        pytest.skip("canonical Colegio_Invisible not present")
    # canonical/ first so `import sound`, `import tone` resolve; then the repo
    # root so `import voice_codec` resolves.
    for p in (_CANON, _CINV):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import sound as oracle_sound
        import voice_codec as oracle_voice
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"monolith oracle not importable: {exc}")
    return {"sound": oracle_sound, "voice": oracle_voice}


def _synthetic_audio(n=4096, seed=1234):
    """Deterministic synthetic mono utterance (vowel-like harmonics + a chirp).
    No RNG in the encoders, so the same array yields byte-identical bodies."""
    sr = audio.SR
    t = np.arange(n) / sr
    x = (0.6 * np.sin(2 * np.pi * 140.0 * t)
         + 0.3 * np.sin(2 * np.pi * 280.0 * t)
         + 0.15 * np.sin(2 * np.pi * 700.0 * t))
    # a slow chirp in the back half to exercise voiced + transition frames
    half = n // 2
    tc = t[half:] - t[half]
    x[half:] += 0.4 * np.sin(2 * np.pi * (300.0 * tc + 200.0 * tc * tc))
    return x.astype(np.float32)


# Container build tuples: (codec, sample_rate, channels, duration_ms,
#                          codec_meta, title, tone). Mix of opaque (wav) and
# vocoder (stft/lpc) codecs, including empty-meta and empty-title edges.
_CONTAINER_CASES = [
    (audio.CODEC_WAV, 44100, 2, 12000, b"", "Synthetic WAV", audio.TONE_REVERENCE),
    (audio.CODEC_WAV, 0, 0, 0, b"", "", audio.TONE_ORDINARY),
    (audio.CODEC_STFT, 8000, 1, 5008,
     audio.pack_stft_meta(313, -7.5, 3.25), "Ephemeris", audio.TONE_ORDINARY),
    (audio.CODEC_LPC, 8000, 1, 5000,
     audio.pack_frames_meta(200), "lpc title", audio.TONE_AI),
    (audio.CODEC_OPUS, 48000, 2, 30000, b"", "| opus clip |", audio.TONE_HOPE),
    (0x7F, 16000, 1, 1234, b"\x01\x02\x03", "unknown codec", audio.TONE_DEMONIC),
]


@pytest.mark.parametrize("codec,sr,ch,dur,meta,title,tone", _CONTAINER_CASES)
def test_container_byte_identity(oracle, codec, sr, ch, dur, meta, title, tone):
    """(1) Container header AND body are byte-for-byte identical to the oracle."""
    body = bytes(range(256)) * 4  # 1024 bytes of arbitrary opaque payload
    h_col, b_col = audio.build_sound_quipu(
        codec, body, sample_rate=sr, channels=ch, duration_ms=dur,
        codec_meta=meta, title=title, tone=tone)
    h_ora, b_ora = oracle["sound"].build_sound_quipu(
        codec, body, sample_rate=sr, channels=ch, duration_ms=dur,
        codec_meta=meta, title=title, tone=tone)
    assert h_col == h_ora, "header bytes differ"
    assert b_col == b_ora, "body bytes differ"


def test_stft_encode_byte_identity(oracle):
    """(2) STFT encoder header+body byte-identical to the oracle (deterministic)."""
    x = _synthetic_audio()
    h_col, b_col = audio.encode_sound_stft(x, title="stft diff", tone=audio.TONE_REVERENCE)
    h_ora, b_ora = oracle["voice"].encode_sound_stft(x, title="stft diff", tone=0xFF)
    assert h_col == h_ora
    assert b_col == b_ora


def test_lpc_encode_byte_identity(oracle):
    """(2) LPC encoder header+body byte-identical to the oracle (deterministic)."""
    x = _synthetic_audio()
    h_col, b_col = audio.encode_sound_lpc(x, title="lpc diff", tone=audio.TONE_AI)
    h_ora, b_ora = oracle["voice"].encode_sound_lpc(x, title="lpc diff", tone=0xA1)
    assert h_col == h_ora
    assert b_col == b_ora


@pytest.mark.parametrize("codec,sr,ch,dur,meta,title,tone", _CONTAINER_CASES)
def test_read_sound_quipu_agreement(oracle, codec, sr, ch, dur, meta, title, tone):
    """(3) read_sound_quipu returns equal dicts on the same blob."""
    body = bytes(range(200))
    h, b = audio.build_sound_quipu(
        codec, body, sample_rate=sr, channels=ch, duration_ms=dur,
        codec_meta=meta, title=title, tone=tone)
    assert audio.read_sound_quipu(h, b) == oracle["sound"].read_sound_quipu(h, b)


def test_roundtrip_stft_decodes_finite(oracle):
    """(4) STFT: encode -> read container -> decode_sound -> finite audio."""
    x = _synthetic_audio()
    h, b = audio.encode_sound_stft(x, title="rt")
    rec = audio.read_sound_quipu(h, b)
    assert rec["type"] == "sound" and rec["codec"] == audio.CODEC_STFT
    out, meta = audio.decode_sound(h, b)
    assert np.all(np.isfinite(out))
    assert len(out) == meta["n_frames"] * audio.HOP + (audio.FRAME - audio.HOP)


def test_roundtrip_lpc_decodes_finite(oracle):
    """(4) LPC: encode -> read container -> decode_sound -> finite audio."""
    x = _synthetic_audio()
    h, b = audio.encode_sound_lpc(x, title="rt")
    rec = audio.read_sound_quipu(h, b)
    assert rec["type"] == "sound" and rec["codec"] == audio.CODEC_LPC
    out, meta = audio.decode_sound(h, b)
    assert np.all(np.isfinite(out))
    assert len(out) == meta["n_frames"] * audio.LPC_FRAME


def test_audio_module_is_import_clean_of_monolith():
    """audio.py must not import the monolith — checked two ways: it must not
    pull the oracle modules into sys.modules on a fresh import, and its source
    must contain no import statement reaching into Colegio_Invisible."""
    import ast
    import importlib

    # 1) A fresh import of colegio.audio must not bring the oracle modules in.
    for name in ("sound", "voice_codec", "colegio.audio"):
        sys.modules.pop(name, None)
    mod = importlib.import_module("colegio.audio")
    assert "voice_codec" not in sys.modules, "audio.py pulled in voice_codec"

    # 2) Static check: no import statement names the monolith or its modules.
    #    (Prose in the docstring may mention 'Colegio_Invisible'; only imports
    #     are load-bearing, so we walk the AST rather than grepping text.)
    tree = ast.parse(open(mod.__file__).read())
    bad = {"sound", "voice_codec", "tone", "Colegio_Invisible"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in bad, f"imports {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            assert root not in bad, f"from {node.module} import ..."
