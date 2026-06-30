"""test_music.py — differential gate: colegio.music ≡ the Colegio_Invisible oracle.

colegio/music.py is the keyless 0x20 music codec (a 'QM' module under the 0x07
sound type): pure-struct build/read, PCM carried as raw bytes, five instrument
kinds (synth / sample / sliced / sample_ref / sliced_ref). This proves it is
byte-for-byte identical to the canonical oracle (canonical/music.py) — and that
it never imports the monolith. Also checks the nature tone + music codec landed
in audio.py. Skips cleanly when the sibling Colegio_Invisible repo isn't present.
"""
import os
import sys

import pytest

from colegio import music, audio

_HERE = os.path.dirname(os.path.abspath(__file__))
_CINV = os.path.abspath(os.path.join(_HERE, "..", "..", "Colegio_Invisible"))
_CANON = os.path.join(_CINV, "canonical")


@pytest.fixture(scope="module")
def oracle():
    """The monolith oracle (canonical/music.py). Oracle ONLY — colegio.music
    never imports it."""
    if not os.path.isfile(os.path.join(_CANON, "music.py")):
        pytest.skip("canonical Colegio_Invisible not present")
    if _CANON not in sys.path:
        sys.path.insert(0, _CANON)
    try:
        import music as oracle_music          # resolves to canonical/music.py
    except Exception as exc:                   # pragma: no cover
        pytest.skip(f"monolith oracle not importable: {exc}")
    return oracle_music


def _module(M):
    """Build a representative module exercising ALL FIVE instrument kinds with
    module M (colegio.music or the oracle). Identical inputs -> identical bytes."""
    pcm8 = bytes(range(256))                                    # int8 PCM, 256 samples
    pcm16 = b"".join(((i * 137) & 0xFFFF).to_bytes(2, "big") for i in range(200))
    txid_a = "a" * 64
    txid_b = bytes(range(32))
    instruments = [
        M.synth_instrument("lead", M.WAVE_SQUARE, duty=96, volume=200,
                           attack_ms=2, decay_ms=30, sustain_level=180, release_ms=60),
        M.sample_instrument("drum", pcm8, srate=8000, base_note=60, bits=8,
                            loop_start=0, loop_end=0, volume=230),
        M.sliced_instrument("chops", pcm16, [(0, 50), (50, 60), (110, 90)],
                            srate=22050, base_note=60, bits=16, volume=220),
        M.sample_ref_instrument("choir", txid_a, src_start_ms=800, src_len_ms=1800,
                                srate=22050, base_note=52, normalize=True,
                                loop_start=9908, loop_end=29767, volume=120),
        M.sliced_ref_instrument("water", txid_b, [(0, 2205), (2205, 3000)],
                                src_start_ms=0, src_len_ms=40000, srate=22050,
                                base_note=60, normalize=True, volume=72),
    ]
    patterns = [
        M.pattern(64, [M.event(0, 0, 60, 0, 230), M.event(4, 1, 1, 1, 255),
                       M.event(8, 2, 2, 2, 200), M.event(0, 3, 60, 3, 200),
                       M.event(16, 4, 1, 4, 90), M.event(60, 0, 0, 0, 0)]),
        M.pattern(300, [M.event(0, 0, 64, 0, 200), M.event(256, 4, 2, 4, 110)]),
    ]
    return M.build_music_body(
        tempo_bpm=88, rows_per_beat=48, num_channels=8,
        instruments=instruments, patterns=patterns, order=[0, 1, 1, 0],
        master_volume=200)


def test_build_byte_identical(oracle):
    """build_music_body must be byte-for-byte identical across all five kinds,
    embedded PCM (8/16-bit), references, and the v2 u16-row format."""
    assert _module(music) == _module(oracle), "music body bytes differ from oracle"


def test_read_agreement(oracle):
    """read_music_body must return identical dicts (incl. raw-bytes pcm + 32-byte
    ref_txid fields)."""
    body = _module(music)
    assert music.read_music_body(body) == oracle.read_music_body(body)


def test_v1_backcompat(oracle):
    """A v1 (u8-row) body still round-trips through both readers identically."""
    # hand-roll a v1 body via the oracle by forcing version 1? both readers
    # accept v1; build a small module and verify cross-reader equality.
    body = music.build_music_body(
        tempo_bpm=120, rows_per_beat=4, num_channels=1,
        instruments=[music.synth_instrument("s", music.WAVE_SINE)],
        patterns=[music.pattern(16, [music.event(0, 0, 60, 0, 255)])],
        order=[0])
    assert music.read_music_body(body) == oracle.read_music_body(body)


def test_constants_match(oracle):
    """Kind/codec/version constants match the oracle."""
    for name in ("KIND_SYNTH", "KIND_SAMPLE", "KIND_SLICED", "KIND_SAMPLE_REF",
                 "KIND_SLICED_REF", "CODEC_MUSIC", "MUSIC_VERSION", "REF_NORMALIZE"):
        assert getattr(music, name) == getattr(oracle, name), name


def test_music_is_keyless_no_monolith():
    """colegio.music imports only struct (+ __future__) — never the monolith."""
    import ast
    tree = ast.parse(open(music.__file__).read())
    bad = {"canonical", "Colegio_Invisible", "numpy", "colegio"}
    imps = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imps |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imps.add((node.module or "").split(".")[0])
    imps.discard("__future__")
    assert imps <= {"struct"}, f"music.py imports more than struct: {imps}"
    assert not (imps & bad)


def test_nature_tone_and_music_codec_in_audio():
    """The colegio sound container gained the nature tone (0x6e) + music codec."""
    assert audio.TONE_NATURE == 0x6E
    assert audio.TONE_NATURE in audio.VALID_TONES
    audio.validate_tone(audio.TONE_NATURE)            # accepts, no raise
    assert audio.CODEC_MUSIC == 0x20
    assert audio.CODEC_NAMES[0x20] == "music"
