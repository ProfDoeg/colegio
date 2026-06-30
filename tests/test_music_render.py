"""test_music_render.py — the render DSP (colegio.music_render) must match the
Colegio_Invisible prototype mixer (working/music/music_codec.py) sample-for-sample.

colegio/music.py is the keyless reader (pcm as raw bytes); music_render.py is the
numpy mixer + the build-by-reference resolver. This proves render_music produces
the same audio as the proven prototype on a synth+sample module (no references,
so no external decoders needed) — and that the keyless reader feeds the DSP
correctly across 8- and 16-bit embedded PCM. Skips when numpy/scipy or the
sibling Colegio_Invisible repo aren't present.
"""
import os
import sys

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")

from colegio import music

_HERE = os.path.dirname(os.path.abspath(__file__))
_CINV = os.path.abspath(os.path.join(_HERE, "..", "..", "Colegio_Invisible"))
_PROTO = os.path.join(_CINV, "working", "music")


@pytest.fixture(scope="module")
def proto():
    """The prototype mixer (working/music/music_codec.py). Oracle ONLY."""
    if not os.path.isfile(os.path.join(_PROTO, "music_codec.py")):
        pytest.skip("Colegio_Invisible prototype not present")
    if _PROTO not in sys.path:
        sys.path.insert(0, _PROTO)
    try:
        import music_codec
    except Exception as exc:                       # pragma: no cover
        pytest.skip(f"prototype not importable: {exc}")
    return music_codec


def _module():
    """A no-reference module exercising synth (all waveforms incl. noise) and
    embedded sample voices at 8- and 16-bit, across two patterns + an order."""
    pcm8 = bytes((i * 7) & 0xFF for i in range(300))
    pcm16 = b"".join(((i * 211) & 0xFFFF).to_bytes(2, "big") for i in range(400))
    instruments = [
        music.synth_instrument("sq", music.WAVE_SQUARE, duty=80, volume=200,
                               attack_ms=2, decay_ms=40, sustain_level=170, release_ms=80),
        music.synth_instrument("nz", music.WAVE_NOISE, duty=120, volume=160,
                               attack_ms=1, decay_ms=20, sustain_level=120, release_ms=50),
        music.sample_instrument("s8", pcm8, srate=8000, base_note=60, bits=8,
                                loop_start=0, loop_end=0, volume=230),
        music.sample_instrument("s16", pcm16, srate=22050, base_note=57, bits=16,
                                loop_start=100, loop_end=380, volume=210),
    ]
    patterns = [
        music.pattern(48, [music.event(0, 0, 60, 0, 230), music.event(4, 1, 64, 1, 200),
                           music.event(8, 2, 1, 2, 255), music.event(0, 3, 62, 3, 180),
                           music.event(40, 0, 0, 0, 0)]),
        music.pattern(96, [music.event(0, 0, 67, 0, 220), music.event(48, 3, 60, 3, 200)]),
    ]
    return music.build_music_body(
        tempo_bpm=96, rows_per_beat=24, num_channels=4,
        instruments=instruments, patterns=patterns, order=[0, 1, 1, 0],
        master_volume=200)


def test_render_matches_prototype(proto):
    """render_music must match the prototype mixer to within float epsilon."""
    import numpy as np
    from colegio import music_render
    body = _module()
    mine = music_render.render_music(body, out_rate=22050)
    ref = proto.render_music(body, out_rate=22050)
    assert len(mine) == len(ref), f"length {len(mine)} != {len(ref)}"
    assert np.max(np.abs(mine - ref)) < 1e-4, "render diverges from prototype mixer"


def test_to_wav_bytes_roundtrip():
    """to_wav_bytes emits a valid 16-bit mono WAV at the given rate."""
    import io
    import wave
    import numpy as np
    from colegio import music_render
    pcm = music_render.render_music(_module(), out_rate=22050)
    wav = music_render.to_wav_bytes(pcm, 22050)
    w = wave.open(io.BytesIO(wav))
    assert w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getframerate() == 22050
    assert w.getnframes() == len(pcm)
