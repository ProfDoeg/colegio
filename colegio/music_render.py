"""music_render.py — the numpy DSP that renders a 0x20 music module to audio.

The keyless reader is colegio/music.py (pure struct); this is the render half:
a deterministic mixer (oscillators + ADSR, sample resampling, slice playback,
soft-limit + peak-normalize) plus the build-by-reference RESOLVER. Numpy is
imported at module top — this is the DSP layer; the keyless container path
(music.py, audio.py read/write) stays numpy-free, and callers (e.g. spectra)
lazy-import this module only when they actually render.

  render_music(body, out_rate=22050, resolver=None) -> float32 mono in [-1,1]
  build_audio_resolver(fetch) -> resolver(ref_txid_bytes) -> (pcm_float, rate)
      fetch(txid_hex) -> (header_hex, body_hex)   (spectra's source fetch)
  to_wav_bytes(pcm, rate) -> 16-bit PCM mono WAV bytes
"""
import io
import os
import struct
import subprocess
import tempfile
import wave
from itertools import groupby

import numpy as np

from colegio.music import (
    read_music_body, KIND_SYNTH, KIND_SAMPLE, KIND_SLICED,
    KIND_SAMPLE_REF, KIND_SLICED_REF, REF_NORMALIZE,
    WAVE_SQUARE, WAVE_TRIANGLE, WAVE_SAW, WAVE_SINE, WAVE_NOISE,
)

NOISE_SEED = 0x5150          # 'QP' — deterministic noise (matches the prototype)


# ---------------------------------------------------------------------------
# PCM helpers — colegio's reader carries pcm as RAW BYTES; resolved refs as int16
# ---------------------------------------------------------------------------
def _pcm_scale(bits):
    return 128.0 if bits == 8 else 32768.0


def _pcm_float(ins):
    """Instrument pcm -> float32 in [-1,1]. Handles raw bytes (embedded, per
    `bits`) or an already-numpy int array (a resolved reference)."""
    pcm = ins["pcm"]
    bits = int(ins.get("bits", 8))
    if isinstance(pcm, (bytes, bytearray)):
        dt = np.int8 if bits == 8 else ">i2"
        pcm = np.frombuffer(bytes(pcm), dtype=dt).astype(np.float32)
    else:
        pcm = np.asarray(pcm, dtype=np.float32)
    return pcm / _pcm_scale(bits)


def module_num_rows(module):
    pats = module["patterns"]
    return sum(pats[o]["num_rows"] for o in module["order"])


def _midi_to_freq(note):
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def _adsr_envelope(n_held, n_release, attack_ms, decay_ms, sustain_level,
                   release_ms, rate):
    total = n_held + n_release
    if total <= 0:
        return np.zeros(0, dtype=np.float32)
    env = np.zeros(total, dtype=np.float32)
    sus = float(sustain_level) / 255.0
    a = min(max(0, int(round(attack_ms * rate / 1000.0))), n_held)
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a, endpoint=False, dtype=np.float32)
    pos = a
    d = min(max(0, int(round(decay_ms * rate / 1000.0))), max(0, n_held - pos))
    if d > 0:
        env[pos:pos + d] = np.linspace(1.0, sus, d, endpoint=False, dtype=np.float32)
    pos += d
    if pos < n_held:
        env[pos:n_held] = sus
    level_at_off = env[n_held - 1] if n_held > 0 else sus
    if n_release > 0:
        env[n_held:total] = np.linspace(level_at_off, 0.0, n_release,
                                        endpoint=True, dtype=np.float32)
    return env


def _render_synth_voice(ins, note, n_samples, rate, rng):
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    freq = _midi_to_freq(note)
    waveform = ins.get("waveform", WAVE_SQUARE)
    if waveform == WAVE_NOISE:
        nz = (rng.random(n_samples, dtype=np.float32) * 2.0 - 1.0)
        duty = int(ins.get("duty", 128))
        if duty < 252 and n_samples >= 16:
            F = np.fft.rfft(nz)
            k = np.fft.rfftfreq(n_samples)
            cut = max(0.01, duty / 255.0) * 0.5
            mask = 1.0 / (1.0 + (k / cut) ** 6)
            nz = np.fft.irfft(F * mask, n=n_samples).astype(np.float32)
            mx = float(np.max(np.abs(nz))) or 1.0
            nz = (nz / mx).astype(np.float32)
        return nz
    t = np.arange(n_samples, dtype=np.float64)
    phase = (freq * t / rate) % 1.0
    if waveform == WAVE_SQUARE:
        duty = min(max(float(ins.get("duty", 128)) / 256.0, 0.01), 0.99)
        wave_ = np.where(phase < duty, 1.0, -1.0)
    elif waveform == WAVE_TRIANGLE:
        wave_ = 2.0 * np.abs(2.0 * phase - 1.0) - 1.0
    elif waveform == WAVE_SAW:
        wave_ = 2.0 * phase - 1.0
    elif waveform == WAVE_SINE:
        wave_ = np.sin(2.0 * np.pi * phase)
    else:
        wave_ = np.where(phase < 0.5, 1.0, -1.0)
    return wave_.astype(np.float32)


def _render_sample_voice(ins, note, n_samples, rate):
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    pcm = _pcm_float(ins)
    L = len(pcm)
    if L == 0:
        return np.zeros(n_samples, dtype=np.float32)
    srate = ins.get("srate", rate)
    base_note = ins.get("base_note", 60)
    loop_start = ins.get("loop_start", 0)
    loop_end = ins.get("loop_end", 0)
    has_loop = (loop_end > loop_start) and (loop_end <= L) and (loop_start >= 0)
    step = (2.0 ** ((note - base_note) / 12.0)) * (float(srate) / float(rate))
    pos = np.arange(n_samples, dtype=np.float64) * step
    if has_loop:
        loop_len = loop_end - loop_start
        wrapped = pos.copy()
        past = pos >= loop_start
        wrapped[past] = loop_start + np.mod(pos[past] - loop_start, loop_len)
        pos = wrapped
        valid = np.ones(n_samples, dtype=bool)
    else:
        valid = pos < (L - 1)
    out = np.zeros(n_samples, dtype=np.float32)
    if not np.any(valid):
        return out
    pv = pos[valid]
    i0 = np.floor(pv).astype(np.int64)
    frac = (pv - i0).astype(np.float32)
    i0 = np.clip(i0, 0, L - 1)
    i1 = np.clip(i0 + 1, 0, L - 1)
    out[valid] = pcm[i0] * (1.0 - frac) + pcm[i1] * frac
    return out


def _render_sliced_voice(ins, note, n_samples, rate):
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    pcm = _pcm_float(ins)
    slices = ins["slices"]
    si = note - 1
    if si < 0 or si >= len(slices):
        return np.zeros(n_samples, dtype=np.float32)
    start, length = slices[si]
    seg = pcm[start:start + length]
    if len(seg) < 2:
        return np.zeros(n_samples, dtype=np.float32)
    step = float(ins.get("srate", rate)) / float(rate)
    pos = np.arange(n_samples, dtype=np.float64) * step
    valid = pos < (len(seg) - 1)
    out = np.zeros(n_samples, dtype=np.float32)
    if not np.any(valid):
        return out
    pv = pos[valid]
    i0 = np.floor(pv).astype(np.int64)
    frac = (pv - i0).astype(np.float32)
    i0 = np.clip(i0, 0, len(seg) - 1)
    i1 = np.clip(i0 + 1, 0, len(seg) - 1)
    out[valid] = seg[i0] * (1.0 - frac) + seg[i1] * frac
    return out


def _resample(x, num):
    """Resample real signal x to `num` samples. Uses scipy.signal.resample when
    available (the proven path); otherwise a numpy Fourier resample with the same
    band-limited behaviour, so the DSP carries no hard scipy dependency."""
    num = max(1, int(num))
    try:
        import scipy.signal as _sig
        return _sig.resample(x, num)
    except Exception:
        x = np.asarray(x, dtype=np.float64)
        N = len(x)
        if N == 0 or num == N:
            return x.copy()
        X = np.fft.rfft(x)
        nyq = num // 2 + 1
        Y = np.zeros(nyq, dtype=complex)
        m = min(len(X), nyq)
        Y[:m] = X[:m]
        if num < N and num % 2 == 0 and m == nyq:   # downsample: fold Nyquist bin
            Y[-1] = Y[-1].real
        return np.fft.irfft(Y, n=num) * (float(num) / float(N))


def _resolve_ref_instrument(ins, resolver):
    """kind 3/4 -> embed-equivalent (kind 1/2): fetch+decode the referenced quipu,
    take [src_start_ms,+src_len_ms], resample to srate, optionally normalize,
    quantize to int16. resolver(ref_txid_bytes) -> (float_mono_pcm, source_rate)."""
    src_pcm, src_rate = resolver(bytes(ins["ref_txid"]))
    src = np.asarray(src_pcm, dtype=np.float64)
    a = max(0, int(round(ins["src_start_ms"] * src_rate / 1000.0)))
    b = min(len(src), a + int(round(ins["src_len_ms"] * src_rate / 1000.0)))
    seg = np.array(src[a:b], dtype=np.float64)
    tgt = int(ins["srate"])
    if int(src_rate) != tgt and len(seg) > 1:
        seg = _resample(seg, max(1, int(round(len(seg) * tgt / src_rate))))
    if ins.get("flags", 0) & REF_NORMALIZE:
        mx = float(np.max(np.abs(seg))) or 1.0
        seg = seg / mx
    pcm16 = np.clip(np.round(seg * 30000.0), -32768, 32767).astype(np.int16)
    out = {"name": ins["name"], "volume": ins["volume"], "attack_ms": ins["attack_ms"],
           "decay_ms": ins["decay_ms"], "sustain_level": ins["sustain_level"],
           "release_ms": ins["release_ms"], "srate": tgt,
           "base_note": ins["base_note"], "bits": 16, "pcm": pcm16}
    if ins["kind"] == KIND_SAMPLE_REF:
        out["kind"] = KIND_SAMPLE
        out["loop_start"] = ins.get("loop_start", 0)
        out["loop_end"] = ins.get("loop_end", 0)
    else:
        out["kind"] = KIND_SLICED
        out["slices"] = ins["slices"]
    return out


def render_music(module_bytes, out_rate=22050, resolver=None):
    """Render a 0x20 music body to float32 mono [-1,1]. Reference instruments
    (kind 3/4) require a resolver(ref_txid)->(pcm,rate)."""
    module = module_bytes if isinstance(module_bytes, dict) else read_music_body(module_bytes)
    instruments = module["instruments"]
    patterns = module["patterns"]
    order = module["order"]
    if any(i["kind"] in (KIND_SAMPLE_REF, KIND_SLICED_REF) for i in instruments):
        if resolver is None:
            raise ValueError("module has reference instruments but no resolver supplied")
        instruments = [_resolve_ref_instrument(i, resolver)
                       if i["kind"] in (KIND_SAMPLE_REF, KIND_SLICED_REF) else i
                       for i in instruments]
    samples_per_row = out_rate * 60.0 / (module["tempo_bpm"] * module["rows_per_beat"])
    total_rows = module_num_rows(module)
    total_samples = int(round(total_rows * samples_per_row))
    max_release_ms = max([i["release_ms"] for i in instruments], default=0)
    tail = int(round(max_release_ms * out_rate / 1000.0)) + 2
    buf = np.zeros(total_samples + tail, dtype=np.float32)
    rng = np.random.default_rng(NOISE_SEED)

    abs_events = []
    row_cursor = 0
    for pat_idx in order:
        pat = patterns[pat_idx]
        for ev in pat["events"]:
            if ev["row"] >= pat["num_rows"]:
                continue
            abs_events.append((row_cursor + ev["row"], ev["channel"],
                               ev["note"], ev["instrument"], ev["volume"]))
        row_cursor += pat["num_rows"]
    abs_events.sort(key=lambda e: (e[1], e[0]))

    for _ch, group in groupby(abs_events, key=lambda e: e[1]):
        chan_events = list(group)
        for i, (arow, _c, note, inst, vol) in enumerate(chan_events):
            if note == 0 or inst >= len(instruments):
                continue
            end_row = chan_events[i + 1][0] if i + 1 < len(chan_events) else total_rows
            if end_row <= arow:
                end_row = arow + 1
            start_s = int(round(arow * samples_per_row))
            n_held = max(1, int(round(end_row * samples_per_row)) - start_s)
            ins = instruments[inst]
            n_release = int(round(ins["release_ms"] * out_rate / 1000.0))
            n_total = n_held + n_release
            if ins["kind"] == KIND_SYNTH:
                raw = _render_synth_voice(ins, note, n_total, out_rate, rng)
            elif ins["kind"] == KIND_SLICED:
                raw = _render_sliced_voice(ins, note, n_total, out_rate)
            else:
                raw = _render_sample_voice(ins, note, n_total, out_rate)
            env = _adsr_envelope(n_held, n_release, ins["attack_ms"], ins["decay_ms"],
                                 ins["sustain_level"], ins["release_ms"], out_rate)
            m = min(len(raw), len(env))
            voice = raw[:m] * env[:m] * (float(vol) / 255.0) * (float(ins["volume"]) / 255.0)
            dst1 = min(start_s + m, len(buf))
            if start_s < len(buf) and dst1 > start_s:
                buf[start_s:dst1] += voice[:dst1 - start_s]

    buf *= float(module["master_volume"]) / 255.0
    buf = np.tanh(buf).astype(np.float32)
    peak = float(np.max(np.abs(buf))) if buf.size else 0.0
    if peak > 1e-9:
        buf *= (0.9 / peak)
    return buf[:total_samples].astype(np.float32)


def to_wav_bytes(pcm, rate=22050):
    """float32 mono [-1,1] -> 16-bit PCM mono WAV bytes."""
    pcm16 = np.clip(np.asarray(pcm, dtype=np.float32), -1.0, 1.0)
    pcm16 = np.round(pcm16 * 32767.0).astype("<i2")
    bio = io.BytesIO()
    with wave.open(bio, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(rate))
        w.writeframes(pcm16.tobytes())
    return bio.getvalue()


def to_opus_bytes(pcm, rate=22050, bitrate=48):
    """float32 mono -> Ogg/Opus bytes via opusenc (≈5x smaller than WAV for an
    inline data-URI). Returns None if opusenc is unavailable, so callers fall
    back to to_wav_bytes."""
    wav = tempfile.mktemp(suffix=".wav")
    opus = wav[:-4] + ".opus"
    try:
        open(wav, "wb").write(to_wav_bytes(pcm, rate))
        subprocess.run(["opusenc", "--quiet", "--bitrate", str(bitrate), wav, opus],
                       check=True)
        return open(opus, "rb").read()
    except (OSError, subprocess.CalledProcessError):
        return None
    finally:
        for f in (wav, opus):
            try:
                os.unlink(f)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# The chain resolver — decode a referenced SOUND quipu's audio to PCM
# ---------------------------------------------------------------------------
def _decode_opus(body):
    op = tempfile.mktemp(suffix=".opus")
    wv = op[:-5] + ".wav"
    try:
        open(op, "wb").write(body)
        subprocess.run(["opusdec", "--quiet", op, wv], check=True)
        w = wave.open(wv)
        n, ch, sr = w.getnframes(), w.getnchannels(), w.getframerate()
        pcm = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float64) / 32768.0
        w.close()
        if ch > 1:
            pcm = pcm.reshape(-1, ch).mean(1)
        return pcm, sr
    finally:
        for f in (op, wv):
            try:
                os.unlink(f)
            except OSError:
                pass


def _decode_wav(body):
    w = wave.open(io.BytesIO(body))
    n, ch, sr, sw = w.getnframes(), w.getnchannels(), w.getframerate(), w.getsampwidth()
    raw = w.readframes(n)
    w.close()
    dt = {1: np.int8, 2: "<i2", 4: "<i4"}.get(sw, "<i2")
    pcm = np.frombuffer(raw, dtype=dt).astype(np.float64) / float(1 << (8 * sw - 1))
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(1)
    return pcm, sr


def decode_sound_to_pcm(header_bytes, body_bytes, fetch=None):
    """Decode a 0x07 sound quipu's audio to (float mono pcm, rate). Routes by
    codec: vocoders via colegio.audio, opus/wav directly, music recursively."""
    from colegio import audio
    rec = audio.read_sound_quipu(header_bytes, body_bytes)
    codec, body = rec["codec"], rec["body"]
    if codec in (0x00, 0x01, 0x02):                       # quipu vocoders
        pcm, _meta = audio.decode_sound(header_bytes, body_bytes)
        return np.asarray(pcm, dtype=np.float64), (rec.get("sample_rate") or 8000)
    if codec == 0x10:                                     # opus
        return _decode_opus(body)
    if codec == 0x12:                                     # wav
        return _decode_wav(body)
    if codec == 0x20:                                     # music — render recursively
        res = build_audio_resolver(fetch) if fetch else None
        return render_music(body, 22050, resolver=res), 22050
    raise ValueError(f"music resolver cannot decode codec 0x{codec:02x}")


def build_audio_resolver(fetch):
    """fetch(txid_hex) -> (header_hex, body_hex). Returns a resolver
    (ref_txid_bytes) -> (float mono pcm, rate) for render_music."""
    def resolver(ref_txid_bytes):
        txid = bytes(ref_txid_bytes).hex()
        hb = fetch(txid)
        if not hb:
            raise KeyError("unresolved reference " + txid)
        header, body = bytes.fromhex(hb[0]), bytes.fromhex(hb[1])
        return decode_sound_to_pcm(header, body, fetch=fetch)
    return resolver
