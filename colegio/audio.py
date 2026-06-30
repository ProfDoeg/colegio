"""audio — the audio media codec for sound quipus (0x07).

Pure numpy: the speech vocoder DSP (STFT-magnitude + LPC-10) AND the keyless
0x07 sound container (pure struct). Mirrors `imaging.py` (the image media
codec) one layer down in the tower: colegio is BELOW the monolith, so this
module is SELF-CONTAINED — it imports nothing from Colegio_Invisible. The
small container build/read is inlined here exactly as canonical/sound.py does
it; the vocoder math is ported verbatim from voice_codec.py.

Wire format (type 0x07): a codec-agnostic envelope carrying a codec byte,
sample rate, channels, duration, a small codec-specific metadata blob
(codec_meta), a title, and an opaque body. Speech vocoders (STFT / LPC /
Codec2) and opaque standard formats (opus / mp3 / wav / flac) all share this
one type byte; the codec enum selects between them.

Public adapters
---------------
    encode_sound_stft(audio, title='', tone=0)   -> (header, body)  [codec 0x00]
    encode_sound_lpc(audio, title='', tone=0)    -> (header, body)  [codec 0x01]
    encode_sound_codec2(audio, title='', tone=0) -> (header, body)  [codec 0x02]
    decode_sound(header, body) -> (audio, meta)   # reads container, routes
                                                  # to the matching DSP
"""

import struct

import numpy as np


# ---------------------------------------------------------------------------
# Tone vocabulary (inlined — do NOT import the monolith's tone.py)
# ---------------------------------------------------------------------------
# The container only needs to (a) validate the tone byte on build and (b) write
# it at header offset 5. We replicate the canonical v1 tone set so build is
# byte-identical to the monolith for any valid tone, and rejects the same bytes.

TONE_ORDINARY  = 0x00
TONE_AFFECTION = 0x01
TONE_SEEKING   = 0x02
TONE_PLAY      = 0x03
TONE_LUST      = 0x04
TONE_RAGE      = 0x05
TONE_FEAR      = 0x06
TONE_GRIEF     = 0x07
TONE_DEMONIC   = 0x0D
TONE_NATURE    = 0x6E
TONE_AI        = 0xA1
TONE_HOPE      = 0xE5
TONE_REVERENCE = 0xFF

VALID_TONES = frozenset({
    TONE_ORDINARY, TONE_AFFECTION, TONE_SEEKING, TONE_PLAY, TONE_LUST,
    TONE_RAGE, TONE_FEAR, TONE_GRIEF, TONE_DEMONIC, TONE_NATURE, TONE_AI,
    TONE_HOPE, TONE_REVERENCE,
})


def validate_tone(tone):
    """Raise ValueError if tone is not a recognized v1 byte. Inlined so the
    container build path matches the monolith without importing tone.py."""
    if tone not in VALID_TONES:
        raise ValueError(
            f"tone must be one of the canonical v1 tones; got {tone:#04x}"
        )


# ---------------------------------------------------------------------------
# Codec enum + name maps (mirrors canonical/sound.py)
# ---------------------------------------------------------------------------
# Speech vocoders (quipu-native bodies, decoded by the numpy DSP below):
CODEC_STFT   = 0x00   # band-limited 8-bit STFT-magnitude vocoder
CODEC_LPC    = 0x01   # LPC-10-style vocoder
CODEC_CODEC2 = 0x02   # Codec2-700C (needs libcodec2)
# Opaque standard formats (the body is a real container/bitstream; meta empty):
CODEC_OPUS   = 0x10   # ogg/opus
CODEC_MP3    = 0x11   # mp3
CODEC_WAV    = 0x12   # WAV / PCM
CODEC_FLAC   = 0x13   # flac
# Composed-music recipe — a 'QM' module (instruments + a note timeline) rendered
# by the music DSP, not a waveform. See colegio/music.py for the keyless codec.
CODEC_MUSIC  = 0x20   # music

# Canonical byte -> name map. THE dictionary; everything else is derived.
CODEC_NAMES = {
    CODEC_STFT:   "stft",
    CODEC_LPC:    "lpc",
    CODEC_CODEC2: "codec2",
    CODEC_OPUS:   "opus",
    CODEC_MP3:    "mp3",
    CODEC_WAV:    "wav",
    CODEC_FLAC:   "flac",
    CODEC_MUSIC:  "music",
}

# Reverse lookup: name -> byte.
CODEC_BY_NAME = {v: k for k, v in CODEC_NAMES.items()}

# The three speech vocoders whose bodies need the numpy DSP to decode.
VOCODER_CODECS = frozenset({CODEC_STFT, CODEC_LPC, CODEC_CODEC2})

# Opaque standard codecs -> a MIME type, for renderers emitting <audio>.
CODEC_MIME = {
    CODEC_OPUS: "audio/ogg",
    CODEC_MP3:  "audio/mpeg",
    CODEC_WAV:  "audio/wav",
    CODEC_FLAC: "audio/flac",
}

TYPE_SOUND = 0x07
_MAGIC = b"\xc1\xdd\x00\x01"


def codec_name(codec):
    """Return the canonical name for a codec byte, or 'unknown_0xNN' for an
    unrecognized value (readers pass unknown codecs through without failing)."""
    return CODEC_NAMES.get(codec, f"unknown_0x{codec:02x}")


# ---------------------------------------------------------------------------
# Vocoder codec_meta (un)packers (pure struct)
# ---------------------------------------------------------------------------

def pack_stft_meta(n_frames, g_min, g_max):
    """Pack the STFT vocoder's codec_meta: n_frames:u16 + g_min:f32 +
    g_max:f32 = 10 bytes, big-endian ('>Hff')."""
    if not (0 <= int(n_frames) <= 0xFFFF):
        raise ValueError(f"n_frames must fit in u16 [0, 65535]; got {n_frames}")
    return struct.pack(">Hff", int(n_frames), float(g_min), float(g_max))


def unpack_stft_meta(codec_meta):
    """Unpack STFT codec_meta -> (n_frames, g_min, g_max). Expects exactly
    10 bytes ('>Hff')."""
    codec_meta = bytes(codec_meta)
    if len(codec_meta) < 10:
        raise ValueError(
            f"stft codec_meta must be >= 10 bytes; got {len(codec_meta)}"
        )
    n_frames, g_min, g_max = struct.unpack(">Hff", codec_meta[:10])
    return n_frames, g_min, g_max


def pack_frames_meta(n_frames):
    """Pack the LPC / Codec2 codec_meta: just n_frames:u16 = 2 bytes,
    big-endian ('>H')."""
    if not (0 <= int(n_frames) <= 0xFFFF):
        raise ValueError(f"n_frames must fit in u16 [0, 65535]; got {n_frames}")
    return struct.pack(">H", int(n_frames))


def unpack_frames_meta(codec_meta):
    """Unpack LPC / Codec2 codec_meta -> n_frames (int). Expects >= 2
    bytes ('>H')."""
    codec_meta = bytes(codec_meta)
    if len(codec_meta) < 2:
        raise ValueError(
            f"frames codec_meta must be >= 2 bytes; got {len(codec_meta)}"
        )
    return struct.unpack(">H", codec_meta[:2])[0]


# ---------------------------------------------------------------------------
# Build / read the 0x07 sound container (pure struct; byte-identical to monolith)
# ---------------------------------------------------------------------------

def build_sound_quipu(codec, body, *, sample_rate=0, channels=0,
                      duration_ms=0, codec_meta=b"", title="",
                      tone=TONE_ORDINARY):
    """Build a 0x07 sound quipu's (header_bytes, body_bytes) pair.

    Header layout (16 + cmlen + titlelen bytes):
        c1dd0001 magic + 0x07 type + tone + codec
        + sample_rate:u16 BE + channels:u8 + duration_ms:u32 BE
        + cmlen:u8 + codec_meta + titlelen:u8 + title (UTF-8).
    Body is returned unchanged.
    """
    validate_tone(tone)

    if not isinstance(codec, int) or not (0 <= codec <= 0xFF):
        raise ValueError(f"codec must be a byte in [0, 255]; got {codec!r}")
    if not (0 <= sample_rate <= 0xFFFF):
        raise ValueError(f"sample_rate must be in [0, 65535]; got {sample_rate}")
    if not (0 <= channels <= 0xFF):
        raise ValueError(f"channels must be in [0, 255]; got {channels}")
    if not (0 <= duration_ms <= 0xFFFFFFFF):
        raise ValueError(
            f"duration_ms must be in [0, {0xFFFFFFFF}]; got {duration_ms}"
        )

    if not isinstance(codec_meta, (bytes, bytearray)):
        raise TypeError(
            f"codec_meta must be bytes, got {type(codec_meta).__name__}"
        )
    codec_meta = bytes(codec_meta)
    if len(codec_meta) > 255:
        raise ValueError(f"codec_meta is {len(codec_meta)} bytes; max is 255")

    if not isinstance(title, str):
        raise TypeError(f"title must be str, got {type(title).__name__}")
    title_raw = title.encode("utf-8")
    if len(title_raw) > 255:
        raise ValueError(
            f"title encodes to {len(title_raw)} UTF-8 bytes; max is 255"
        )

    if not isinstance(body, (bytes, bytearray)):
        raise TypeError(f"body must be bytes, got {type(body).__name__}")
    body_bytes = bytes(body)

    header = (
        _MAGIC
        + bytes([TYPE_SOUND, tone, codec])
        + struct.pack(">H", sample_rate)
        + bytes([channels])
        + struct.pack(">I", duration_ms)
        + bytes([len(codec_meta)]) + codec_meta
        + bytes([len(title_raw)]) + title_raw
    )
    return header, body_bytes


def read_sound_quipu(header_bytes, body_bytes):
    """Parse a 0x07 sound quipu. Keyless. Returns the same dict shape as the
    monolith canonical/sound.read_sound_quipu."""
    header_bytes = bytes(header_bytes)
    body_bytes = bytes(body_bytes)

    if header_bytes[:4] != _MAGIC:
        raise ValueError("not a quipu (c1dd0001 magic missing)")
    if len(header_bytes) < 16:
        raise ValueError(
            f"header too short: {len(header_bytes)} bytes (need >= 16)"
        )
    if header_bytes[4] != TYPE_SOUND:
        raise ValueError(
            f"not a sound quipu (type byte = {header_bytes[4]:#04x}, "
            f"expected 0x07)"
        )

    tone        = header_bytes[5]
    codec       = header_bytes[6]
    sample_rate = struct.unpack(">H", header_bytes[7:9])[0]
    channels    = header_bytes[9]
    duration_ms = struct.unpack(">I", header_bytes[10:14])[0]
    cmlen       = header_bytes[14]

    cm_start = 15
    cm_end = cm_start + cmlen
    if cm_end > len(header_bytes):
        raise ValueError(
            f"header truncated: codec_meta claims {cmlen} bytes but only "
            f"{len(header_bytes) - cm_start} remain"
        )
    codec_meta = header_bytes[cm_start:cm_end]

    if cm_end >= len(header_bytes):
        raise ValueError("header truncated: missing title length byte")
    titlelen = header_bytes[cm_end]
    title_start = cm_end + 1
    title_end = title_start + titlelen
    if title_end > len(header_bytes):
        raise ValueError(
            f"header truncated: title claims {titlelen} bytes but only "
            f"{len(header_bytes) - title_start} remain"
        )
    title = header_bytes[title_start:title_end].decode("utf-8", errors="replace")

    return {
        "type":        "sound",
        "tone":        tone,
        "codec":       codec,
        "codec_name":  codec_name(codec),
        "sample_rate": sample_rate,
        "channels":    channels,
        "duration_ms": duration_ms,
        "codec_meta":  codec_meta,
        "title":       title,
        "body":        body_bytes,
        "size":        len(body_bytes),
    }


def sound_header_len(header_bytes):
    """Return the total header length computed purely from header fields:
    16 + cmlen + titlelen."""
    header_bytes = bytes(header_bytes)
    if len(header_bytes) < 16:
        raise ValueError("header too short to measure")
    cmlen = header_bytes[14]
    titlelen = header_bytes[15 + cmlen]
    return 16 + cmlen + titlelen


# ===========================================================================
# DSP — STFT-magnitude vocoder (codec 0x00)
# ===========================================================================

SR = 8000          # Hz
FRAME = 256        # samples per analysis window (32 ms at 8 kHz)
HOP = FRAME // 2   # 128 samples (16 ms, 50% overlap for constant-OLA)
K_BINS = 32        # number of low-frequency magnitude bins kept (0 - 1 kHz)
N_BINS_FULL = FRAME // 2 + 1   # 129
GRIFFIN_LIM_ITERS = 32


def _hann(n):
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / n))


def _frame_signal(x, frame=FRAME, hop=HOP):
    """Split x into overlapping frames. Returns (n_frames, frame)."""
    n_samples = len(x)
    if n_samples < frame:
        x = np.pad(x, (0, frame - n_samples))
        n_samples = frame
    n_frames = 1 + (n_samples - frame) // hop
    frames = np.zeros((n_frames, frame), dtype=np.float32)
    for i in range(n_frames):
        s = i * hop
        frames[i] = x[s:s + frame]
    return frames


def _overlap_add(frames, hop=HOP):
    """Reconstruct signal from analysis frames via overlap-add."""
    n_frames, frame = frames.shape
    out_len = (n_frames - 1) * hop + frame
    out = np.zeros(out_len, dtype=np.float32)
    norm = np.zeros(out_len, dtype=np.float32)
    w = _hann(frame).astype(np.float32)
    for i in range(n_frames):
        s = i * hop
        out[s:s + frame] += frames[i]
        norm[s:s + frame] += w * w   # accounting for analysis + synth window
    norm = np.maximum(norm, 1e-6)
    return out / norm


def _stft_encode_body(audio_samples):
    """Run the STFT-magnitude DSP and return (body_bytes, n_frames, g_min,
    g_max, duration_ms). The caller wraps the result in the sound container."""
    x = np.asarray(audio_samples, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError("audio must be mono (1-D)")

    # Peak normalize so quantization range is well-used
    peak = float(np.max(np.abs(x)))
    if peak > 0:
        x = x / peak

    # Analysis: window + STFT
    frames = _frame_signal(x)
    w = _hann(FRAME).astype(np.float32)
    windowed = frames * w
    spectrum = np.fft.rfft(windowed, axis=1)             # (n_frames, 129)
    magnitude = np.abs(spectrum).astype(np.float32)
    mag_kept = magnitude[:, :K_BINS]                      # (n_frames, 32)

    # Log-domain, normalize to [0, 1] per utterance, quantize to uint8
    log_mag = np.log(mag_kept + 1e-6)
    g_min = float(log_mag.min())
    g_max = float(log_mag.max())
    span = g_max - g_min if g_max > g_min else 1.0
    norm = (log_mag - g_min) / span
    quant = np.clip(np.round(norm * 255.0), 0, 255).astype(np.uint8)

    n_frames = int(quant.shape[0])
    if n_frames > 65535:
        raise ValueError("utterance too long: n_frames must fit in uint16")

    body = quant.tobytes()  # n_frames × 32 bytes, row-major
    duration_ms = int(round(len(x) / SR * 1000))
    return body, n_frames, g_min, g_max, duration_ms


def encode_sound_stft(audio_samples, title="", tone=TONE_ORDINARY):
    """Encode mono 8 kHz float audio into a sound container (type 0x07,
    codec 0x00 = STFT-magnitude vocoder). Returns (header, body)."""
    body, n_frames, g_min, g_max, duration_ms = _stft_encode_body(audio_samples)
    return build_sound_quipu(
        CODEC_STFT, body,
        sample_rate=SR, channels=1, duration_ms=duration_ms,
        codec_meta=pack_stft_meta(n_frames, g_min, g_max),
        title=title, tone=tone,
    )


def decode_voice(header_bytes, body_bytes, gl_iters=GRIFFIN_LIM_ITERS):
    """Decode a sound container (type 0x07, codec 0x00 = STFT) back to audio.
    Returns (audio_float32, meta_dict)."""
    rec = read_sound_quipu(header_bytes, body_bytes)
    if rec["codec"] != CODEC_STFT:
        raise ValueError(
            f"not the STFT codec (codec = {rec['codec']:#04x}, expected 0x00)"
        )
    tone = rec["tone"]
    title = rec["title"]
    n_frames, g_min, g_max = unpack_stft_meta(rec["codec_meta"])
    body_bytes = rec["body"]

    expected = n_frames * K_BINS
    if len(body_bytes) < expected:
        raise ValueError(f"body too short: {len(body_bytes)} < {expected}")
    quant = np.frombuffer(body_bytes[:expected], dtype=np.uint8)
    quant = quant.reshape((n_frames, K_BINS))

    # Un-quantize back to log-magnitude, then to linear magnitude
    span = g_max - g_min if g_max > g_min else 1.0
    log_mag = (quant.astype(np.float32) / 255.0) * span + g_min
    mag_kept = np.maximum(np.exp(log_mag) - 1e-6, 0.0)

    # Zero-pad up to full STFT bin count (drop high band → muted highs)
    mag = np.zeros((n_frames, N_BINS_FULL), dtype=np.float32)
    mag[:, :K_BINS] = mag_kept

    # Griffin-Lim phase recovery: random init, iterate
    rng = np.random.default_rng(0)
    phase = 2 * np.pi * rng.random((n_frames, N_BINS_FULL)).astype(np.float32)
    spectrum = mag * np.exp(1j * phase)
    w = _hann(FRAME).astype(np.float32)

    for _ in range(gl_iters):
        frames = np.fft.irfft(spectrum, n=FRAME, axis=1).astype(np.float32) * w
        audio = _overlap_add(frames)
        re_frames = _frame_signal(audio)[:n_frames] * w
        re_spectrum = np.fft.rfft(re_frames, axis=1)
        spectrum = mag * np.exp(1j * np.angle(re_spectrum))

    frames = np.fft.irfft(spectrum, n=FRAME, axis=1).astype(np.float32) * w
    audio = _overlap_add(frames)

    # Peak normalize output (Griffin-Lim doesn't preserve scale)
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak * 0.95

    return audio, {
        "title": title,
        "tone": tone,
        "codec": CODEC_STFT,
        "n_frames": n_frames,
        "duration_s": n_frames * HOP / SR,
        "sample_rate": SR,
        "k_bins": K_BINS,
    }


# ===========================================================================
# DSP — LPC-10-style vocoder (codec 0x01)
# ===========================================================================

LPC_FRAME = 200          # samples at 8 kHz = 25 ms
LPC_HOP = LPC_FRAME      # no overlap
LPC_ORDER = 10           # filter order — classic LPC-10
LPC_BYTES_PER_FRAME = LPC_ORDER + 2  # refl coefs + gain + pitch = 12
PRE_EMPHASIS = 0.97      # high-pass coefficient — flattens spectrum for LPC


def _autocorr(x, order):
    """Compute autocorrelation R[0..order] of a 1-D signal."""
    R = np.zeros(order + 1, dtype=np.float64)
    for k in range(order + 1):
        if k == 0:
            R[k] = np.dot(x, x)
        else:
            R[k] = np.dot(x[:-k], x[k:])
    return R


def _levinson_durbin(R, order):
    """Solve the symmetric Toeplitz system for LPC analysis. Returns
    (a, k) where a[0]=1 and k are the reflection (PARCOR) coefficients."""
    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0
    k = np.zeros(order, dtype=np.float64)
    E = R[0]
    if E <= 1e-12:
        return a, k

    for i in range(order):
        if abs(E) < 1e-12:
            break
        # Compute reflection coefficient k[i]
        s = R[i + 1]
        for j in range(i):
            s += a[j + 1] * R[i - j]
        k[i] = -s / E
        # Update a[1..i+1] in place using two passes
        a_new = a.copy()
        a_new[i + 1] = k[i]
        for j in range(i):
            a_new[j + 1] = a[j + 1] + k[i] * a[i - j]
        a = a_new
        E = E * (1.0 - k[i] ** 2)

    return a, k


def _refl_to_lpc(k_vec):
    """Convert reflection coefficients back to LPC filter coefficients."""
    order = len(k_vec)
    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0
    for i in range(order):
        a_new = a.copy()
        a_new[i + 1] = k_vec[i]
        for j in range(i):
            a_new[j + 1] = a[j + 1] + k_vec[i] * a[i - j]
        a = a_new
    return a


def _estimate_pitch(frame, sr=SR, min_hz=70.0, max_hz=400.0):
    """Center-clipped autocorrelation pitch detector. Returns pitch period
    in samples (0 if frame is judged unvoiced)."""
    n = len(frame)
    if n < 64:
        return 0
    # Center-clip to suppress formant carry-through (classic Sondhi)
    thresh = 0.3 * np.max(np.abs(frame))
    if thresh < 1e-6:
        return 0  # near-silence
    clipped = np.where(np.abs(frame) > thresh,
                       np.sign(frame) * (np.abs(frame) - thresh),
                       0.0).astype(np.float32)
    # Autocorrelation by full numpy correlate (small array, fast enough)
    R = np.correlate(clipped, clipped, mode="full")[n - 1:]
    min_lag = max(int(sr / max_hz), 1)
    max_lag = min(int(sr / min_hz), n - 1)
    if max_lag <= min_lag:
        return 0
    peak_lag = int(np.argmax(R[min_lag:max_lag + 1])) + min_lag
    # Voiced/unvoiced decision: R[peak]/R[0] threshold
    if R[0] <= 0:
        return 0
    if R[peak_lag] / R[0] < 0.30:
        return 0
    if peak_lag > 255:
        return 0  # can't fit in a byte → degrade to unvoiced
    return peak_lag


def encode_sound_lpc(audio_samples, title="", tone=TONE_ORDINARY):
    """Encode mono 8 kHz audio with the LPC vocoder into a sound container
    (type 0x07, codec 0x01 = LPC-10). Returns (header, body); codec_meta
    carries n_frames (u16)."""
    x = np.asarray(audio_samples, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError("audio must be mono (1-D)")
    peak = float(np.max(np.abs(x)))
    if peak > 0:
        x = x / peak

    # Pre-emphasis flattens the spectrum, which makes LPC analysis crisper
    x_pre = np.empty_like(x)
    x_pre[0] = x[0]
    x_pre[1:] = x[1:] - PRE_EMPHASIS * x[:-1]

    # Pad to whole frames
    n_frames = max(1, (len(x_pre) + LPC_FRAME - 1) // LPC_FRAME)
    padded = np.zeros(n_frames * LPC_FRAME, dtype=np.float32)
    padded[:len(x_pre)] = x_pre
    if n_frames > 65535:
        raise ValueError("utterance too long: n_frames must fit uint16")

    w = _hann(LPC_FRAME).astype(np.float32)
    quant = np.zeros((n_frames, LPC_BYTES_PER_FRAME), dtype=np.uint8)

    for i in range(n_frames):
        frame_raw = padded[i * LPC_FRAME:(i + 1) * LPC_FRAME]
        frame_w = frame_raw * w
        R = _autocorr(frame_w, LPC_ORDER)
        a, refl = _levinson_durbin(R, LPC_ORDER)
        # Residual energy: R[0] · ∏(1 − k²)
        e_res = R[0] * float(np.prod(1.0 - refl ** 2))
        e_res = max(e_res, 1e-12)
        gain = float(np.sqrt(e_res / LPC_FRAME))  # RMS-equiv
        pitch = _estimate_pitch(frame_raw)

        # Quantize reflection coefs (range [-1, 1]) to 8-bit
        refl_q = np.clip(np.round((refl + 1.0) * 127.5), 0, 255).astype(np.uint8)
        # Quantize gain: log domain, range [-60, 0] dB → [0, 255]
        gain_db = 20.0 * np.log10(max(gain, 1e-6))
        gain_q = int(np.clip(np.round((gain_db + 60.0) * (255.0 / 60.0)), 0, 255))

        quant[i, :LPC_ORDER] = refl_q
        quant[i, LPC_ORDER] = gain_q
        quant[i, LPC_ORDER + 1] = pitch  # 0 = unvoiced

    body = quant.tobytes()
    duration_ms = int(round(len(x) / SR * 1000))
    return build_sound_quipu(
        CODEC_LPC, body,
        sample_rate=SR, channels=1, duration_ms=duration_ms,
        codec_meta=pack_frames_meta(n_frames),
        title=title, tone=tone,
    )


def decode_voice_lpc(header_bytes, body_bytes):
    """Decode a sound container (type 0x07, codec 0x01 = LPC) back to audio.
    Returns (audio_float32, meta_dict)."""
    rec = read_sound_quipu(header_bytes, body_bytes)
    if rec["codec"] != CODEC_LPC:
        raise ValueError(
            f"not the LPC codec (codec = {rec['codec']:#04x}, expected 0x01)"
        )
    tone = rec["tone"]
    title = rec["title"]
    n_frames = unpack_frames_meta(rec["codec_meta"])
    body_bytes = rec["body"]

    expected = n_frames * LPC_BYTES_PER_FRAME
    if len(body_bytes) < expected:
        raise ValueError(f"body too short: {len(body_bytes)} < {expected}")
    quant = np.frombuffer(body_bytes[:expected], dtype=np.uint8)
    quant = quant.reshape((n_frames, LPC_BYTES_PER_FRAME))

    rng = np.random.default_rng(0)
    audio = np.zeros(n_frames * LPC_FRAME, dtype=np.float32)
    filter_state = np.zeros(LPC_ORDER, dtype=np.float64)
    pitch_phase = 0  # carry across voiced frames for continuity

    for i in range(n_frames):
        refl_q = quant[i, :LPC_ORDER]
        gain_q = int(quant[i, LPC_ORDER])
        pitch = int(quant[i, LPC_ORDER + 1])

        refl = (refl_q.astype(np.float64) - 127.5) / 127.5
        refl = np.clip(refl, -0.99, 0.99)  # keep filter strictly stable
        gain_db = (gain_q / 255.0) * 60.0 - 60.0
        gain = 10.0 ** (gain_db / 20.0)
        a = _refl_to_lpc(refl)

        # Excitation
        excitation = np.zeros(LPC_FRAME, dtype=np.float64)
        if pitch >= 20:  # voiced (≥ 50 ms period would be sub-audible)
            j = pitch_phase
            while j < LPC_FRAME:
                # √pitch scaling keeps perceived loudness independent of period
                excitation[j] = np.sqrt(pitch)
                j += pitch
            pitch_phase = j - LPC_FRAME
        else:  # unvoiced
            excitation = rng.standard_normal(LPC_FRAME) * 0.3
            pitch_phase = 0

        excitation *= gain

        # All-pole synthesis filter: y[n] = e[n] - Σ a[k] · y[n-k]
        y = np.zeros(LPC_FRAME, dtype=np.float64)
        state = filter_state.copy()
        for n in range(LPC_FRAME):
            v = excitation[n]
            for k_idx in range(LPC_ORDER):
                v -= a[k_idx + 1] * state[k_idx]
            y[n] = v
            # Shift state right
            state[1:] = state[:-1]
            state[0] = v
        filter_state = state

        audio[i * LPC_FRAME:(i + 1) * LPC_FRAME] = y.astype(np.float32)

    # Defend against the occasional filter blow-up: replace nan/inf with 0
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    # Soft-clip pathological excursions before de-emphasis to keep the
    # leaky integrator below from amplifying any one frame's transient
    audio = np.tanh(audio / 4.0) * 4.0

    # De-emphasis (inverse of pre-emphasis IIR)
    de = np.empty_like(audio)
    de[0] = audio[0]
    for n in range(1, len(audio)):
        de[n] = audio[n] + PRE_EMPHASIS * de[n - 1]
    de = np.nan_to_num(de, nan=0.0, posinf=0.0, neginf=0.0)

    peak = float(np.max(np.abs(de)))
    if peak > 0:
        de = de / peak * 0.95

    return de, {
        "title": title,
        "tone": tone,
        "codec": 0x01,
        "n_frames": n_frames,
        "duration_s": n_frames * LPC_FRAME / SR,
        "sample_rate": SR,
        "lpc_order": LPC_ORDER,
    }


# ===========================================================================
# DSP — Codec2 700C (codec 0x02) — lazy/optional (needs libcodec2 + pycodec2)
# ===========================================================================

C2_FRAME = 320      # samples per Codec2 700C frame (40 ms @ 8 kHz)
C2_BPF = 4          # bytes per frame
C2_MODE = 700       # pycodec2 mode integer for 700C


def _get_c2():
    """Lazy-import the Codec2 binding so audio.py still loads (and the 0x00 /
    0x01 codecs still work) on machines without libcodec2."""
    import pycodec2
    return pycodec2.Codec2(C2_MODE)


def encode_sound_codec2(audio_samples, title="", tone=TONE_ORDINARY):
    """Encode mono 8 kHz audio with Codec2 700C into a sound container
    (type 0x07, codec 0x02). Needs libcodec2 + pycodec2. Returns
    (header, body); codec_meta carries n_frames (u16)."""
    x = np.asarray(audio_samples, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError("audio must be mono")

    # Convert float [-1, 1] to int16
    peak = float(np.max(np.abs(x)))
    if peak > 0:
        x = x / peak
    x_i16 = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)

    # Pad to whole frames
    n_frames = max(1, (len(x_i16) + C2_FRAME - 1) // C2_FRAME)
    padded = np.zeros(n_frames * C2_FRAME, dtype=np.int16)
    padded[:len(x_i16)] = x_i16
    if n_frames > 65535:
        raise ValueError("utterance too long: n_frames must fit uint16")

    c2 = _get_c2()
    chunks = bytearray()
    for i in range(n_frames):
        frame = padded[i * C2_FRAME:(i + 1) * C2_FRAME]
        chunks.extend(c2.encode(frame))

    duration_ms = int(round(len(x_i16) / SR * 1000))
    return build_sound_quipu(
        CODEC_CODEC2, bytes(chunks),
        sample_rate=SR, channels=1, duration_ms=duration_ms,
        codec_meta=pack_frames_meta(n_frames),
        title=title, tone=tone,
    )


def decode_voice_c2(header_bytes, body_bytes):
    """Decode a sound container (type 0x07, codec 0x02 = Codec2 700C)."""
    rec = read_sound_quipu(header_bytes, body_bytes)
    if rec["codec"] != CODEC_CODEC2:
        raise ValueError(
            f"not the Codec2 codec (codec = {rec['codec']:#04x}, "
            f"expected 0x02)"
        )
    tone = rec["tone"]
    title = rec["title"]
    n_frames = unpack_frames_meta(rec["codec_meta"])
    body_bytes = rec["body"]

    expected = n_frames * C2_BPF
    if len(body_bytes) < expected:
        raise ValueError(f"body too short: {len(body_bytes)} < {expected}")

    c2 = _get_c2()
    audio_i16 = np.zeros(n_frames * C2_FRAME, dtype=np.int16)
    for i in range(n_frames):
        chunk = body_bytes[i * C2_BPF:(i + 1) * C2_BPF]
        audio_i16[i * C2_FRAME:(i + 1) * C2_FRAME] = c2.decode(chunk)

    audio_f = audio_i16.astype(np.float32) / 32768.0
    return audio_f, {
        "title": title,
        "tone": tone,
        "codec": 0x02,
        "n_frames": n_frames,
        "duration_s": n_frames * C2_FRAME / SR,
        "sample_rate": SR,
        "bitrate_bps": 700,
    }


# ---------------------------------------------------------------------------
# Unified container decode dispatcher
# ---------------------------------------------------------------------------

# Maps each vocoder codec byte to the DSP decoder that reconstructs audio.
_DECODE_DISPATCH = {
    CODEC_STFT:   decode_voice,
    CODEC_LPC:    decode_voice_lpc,
    CODEC_CODEC2: decode_voice_c2,
}


def decode_sound(header_bytes, body_bytes):
    """Decode a sound container (type 0x07) to audio samples.

    Reads the container (keyless), then routes to the matching numpy DSP for
    the vocoder codecs (0x00 STFT / 0x01 LPC / 0x02 Codec2). Opaque standard
    codecs (0x10 opus / 0x11 mp3 / 0x12 wav / 0x13 flac) are NOT decoded here —
    their bodies are real bitstreams for a browser/audio library, so this
    raises ValueError.

    Returns:
        (audio_samples, meta_dict) — same shape the per-codec decoders return.
    """
    rec = read_sound_quipu(header_bytes, body_bytes)
    codec = rec["codec"]
    fn = _DECODE_DISPATCH.get(codec)
    if fn is None:
        raise ValueError(
            f"codec 0x{codec:02x} ({rec['codec_name']}) is not a quipu "
            f"vocoder; its body is an opaque {rec['codec_name']} bitstream. "
            f"Use read_sound_quipu() and decode it with an audio "
            f"library / browser instead."
        )
    return fn(header_bytes, body_bytes)
