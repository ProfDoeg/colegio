"""
music.py — the MUSIC codec for the 0x07 SOUND type (codec byte 0x20), ported
into the clean colegio package. Byte-identical to canonical/music.py (the
Colegio_Invisible oracle); the differential gate in tests/test_music.py asserts
it. Keyless, pure-stdlib (struct only) — no monolith import.

Music is NOT a separate quipu type: it is codec 0x20 *under* the 0x07 sound
type, beside the voice vocoders (0x00-0x02) and the standard audio formats
(0x10-0x13). Whatever the codec, what a sound quipu gives you is audio to play;
the codec byte says how it is represented. Voice and audio are recordings
(waveforms); MUSIC is a *recipe* — a score (instruments + a note timeline) that a
renderer turns into audio. Same shape as MIDI living under audio/*: the output is
sound, the encoding is symbolic.

This module is the keyless, PURE-STDLIB (struct only) reader/writer of the music
module — the body of a 0x20 sound quipu. PCM is carried as raw bytes; the numpy
mixer that renders a module to a waveform lives in the DSP layer (the working
prototype working/music/music_codec.py), not here.

Three instrument kinds = the three ways music is made:
  kind 0  SYNTH   — oscillators (square/triangle/saw/sine/noise) + ADSR; generated
  kind 1  SAMPLE  — a whole recording, pitched across the keyboard, looped
  kind 2  SLICED  — one long recording + a slice table; chop it, re-sequence

Wire format (the 0x07 body when codec == 0x20). All multi-byte ints BIG-ENDIAN:

  'QM'                      magic (0x51 0x4d)
  version:u8 = 2            (v1: u8 num_rows & event.row; v2: u16 each, so a
                            pattern can be a long, finely-subdivided grid)
  tempo_bpm:u16
  rows_per_beat:u8          rows per quarter-note (4 = a 16th-note grid)
  num_channels:u8           polyphony channels (a chord = several channels/row)
  num_instruments:u8
  num_patterns:u8
  order_len:u8
  master_volume:u8
  INSTRUMENTS (num_instruments):
    kind:u8 (0/1/2) · namelen:u8 · name · volume:u8 ·
    attack_ms:u16 · decay_ms:u16 · sustain_level:u8 · release_ms:u16
    if kind 0 SYNTH:  waveform:u8 (0 square,1 tri,2 saw,3 sine,4 noise) · duty:u8
                      (for noise, duty is a lowpass tone/cutoff)
    if kind 1 SAMPLE: srate:u16 · base_note:u8 · bits:u8 (8 or 16) ·
                      loop_start:u16 · loop_end:u16 (0,0 = one-shot) ·
                      pcm_len:u32 (SAMPLE count) · pcm (pcm_len samples,
                      int8 if bits==8 else int16 big-endian)
    if kind 2 SLICED: srate:u16 · base_note:u8 · bits:u8 (8 or 16) ·
                      num_slices:u16 · num_slices x (start:u32, length:u32)
                      (slice regions, in SAMPLES) · pcm_len:u32 (SAMPLE count) ·
                      pcm (as above)
  PATTERNS (num_patterns):
    num_rows:u16 · num_events:u16 ·
    events (num_events x [row:u16, channel:u8, note:u8, instrument:u8, volume:u8])
      note 0 = key-off; 1..127 = MIDI note; for a SLICED instrument note =
      slice_index + 1
      (v1 modules carry num_rows and row as u8; the reader accepts both versions)
  ORDER: order_len x pattern_index:u8

bits (per SAMPLE / SLICED instrument): 8 = dusty/lean (~-48 dB floor, SP-1200
character), 16 = clean/hi-fi (~-96 dB). Each instrument chooses independently, so
a dusty 8-bit drum chop and a clean 16-bit lead can share one track.
"""

from __future__ import annotations

import struct

MUSIC_MAGIC   = b"QM"          # 0x51 0x4d
MUSIC_VERSION = 2              # v2: u16 num_rows & event.row (v1 = u8; reader accepts both)
CODEC_MUSIC   = 0x20           # the 0x07 sound container's codec byte for music

# Instrument kinds (the per-element discriminator).
KIND_SYNTH      = 0
KIND_SAMPLE     = 1
KIND_SLICED     = 2
KIND_SAMPLE_REF = 3     # like SAMPLE, but PCM is RESOLVED from a referenced quipu
KIND_SLICED_REF = 4     # like SLICED, but the clip is RESOLVED from a referenced quipu
KIND_NAMES  = {KIND_SYNTH: "synth", KIND_SAMPLE: "sample", KIND_SLICED: "sliced",
               KIND_SAMPLE_REF: "sample_ref", KIND_SLICED_REF: "sliced_ref"}

# A reference instrument names another sound quipu by its 32-byte ROOT txid (the
# book.py convention) instead of embedding PCM. Upload a sound once; any future
# composition references it. The referenced audio is decoded, the region
# [src_start_ms, +src_len_ms] is taken, resampled to `srate`, optionally
# peak-normalized (REF_NORMALIZE), and then used exactly like an embedded sample.
REF_NORMALIZE = 0x01     # flags bit: peak-normalize the extracted region

# Synth waveforms.
WAVE_SQUARE, WAVE_TRIANGLE, WAVE_SAW, WAVE_SINE, WAVE_NOISE = 0, 1, 2, 3, 4
WAVE_NAMES = {0: "square", 1: "triangle", 2: "saw", 3: "sine", 4: "noise"}


def _u8(v, f):
    v = int(v)
    if not 0 <= v <= 0xFF:
        raise ValueError(f"{f} out of u8 range [0,255]: {v}")
    return v


def _u16(v, f):
    v = int(v)
    if not 0 <= v <= 0xFFFF:
        raise ValueError(f"{f} out of u16 range [0,65535]: {v}")
    return v


def _u32(v, f):
    v = int(v)
    if not 0 <= v <= 0xFFFFFFFF:
        raise ValueError(f"{f} out of u32 range: {v}")
    return v


def _samples_in(pcm_bytes, bits):
    """Sample (frame) count of a PCM byte string at the given bit depth."""
    if bits not in (8, 16):
        raise ValueError(f"bits must be 8 or 16; got {bits}")
    step = bits // 8
    if len(pcm_bytes) % step:
        raise ValueError(f"pcm byte length {len(pcm_bytes)} not a multiple of {step}")
    return len(pcm_bytes) // step


# ---------------------------------------------------------------------------
# Instrument / event / pattern dict builders (pure-stdlib; pcm carried as bytes)
# ---------------------------------------------------------------------------

def synth_instrument(name, waveform=WAVE_SQUARE, *, duty=128, volume=255,
                     attack_ms=2, decay_ms=40, sustain_level=200, release_ms=60):
    return {"kind": KIND_SYNTH, "name": str(name), "volume": int(volume),
            "attack_ms": int(attack_ms), "decay_ms": int(decay_ms),
            "sustain_level": int(sustain_level), "release_ms": int(release_ms),
            "waveform": int(waveform), "duty": int(duty)}


def sample_instrument(name, pcm_bytes, *, srate=22050, base_note=60, bits=8,
                      loop_start=0, loop_end=0, volume=255, attack_ms=0,
                      decay_ms=0, sustain_level=255, release_ms=20):
    """A whole-recording instrument (kind 1). pcm_bytes is the raw PCM already at
    `bits` depth (int8, or int16 big-endian)."""
    return {"kind": KIND_SAMPLE, "name": str(name), "volume": int(volume),
            "attack_ms": int(attack_ms), "decay_ms": int(decay_ms),
            "sustain_level": int(sustain_level), "release_ms": int(release_ms),
            "srate": int(srate), "base_note": int(base_note), "bits": int(bits),
            "loop_start": int(loop_start), "loop_end": int(loop_end),
            "pcm": bytes(pcm_bytes)}


def sliced_instrument(name, pcm_bytes, slices, *, srate=22050, base_note=60,
                      bits=8, volume=255, attack_ms=1, decay_ms=0,
                      sustain_level=255, release_ms=8):
    """A sliced-recording instrument (kind 2): one long clip + (start,length)
    slice regions in SAMPLES. Event note = slice_index + 1."""
    return {"kind": KIND_SLICED, "name": str(name), "volume": int(volume),
            "attack_ms": int(attack_ms), "decay_ms": int(decay_ms),
            "sustain_level": int(sustain_level), "release_ms": int(release_ms),
            "srate": int(srate), "base_note": int(base_note), "bits": int(bits),
            "slices": [(int(s), int(l)) for (s, l) in slices],
            "pcm": bytes(pcm_bytes)}


def _coerce_txid(ref):
    """Accept raw 32 bytes or a 64-char hex string; return raw 32 bytes.
    None/'' -> 32 zero bytes (a structural placeholder, never a real root) — the
    same sentinel book.py uses for unresolved references."""
    if ref is None or ref == "":
        return b"\x00" * 32
    if isinstance(ref, (bytes, bytearray)):
        if len(ref) != 32:
            raise ValueError(f"ref_txid raw must be 32 bytes (got {len(ref)})")
        return bytes(ref)
    if isinstance(ref, str):
        s = ref.strip()
        if len(s) != 64 or any(c not in "0123456789abcdefABCDEF" for c in s):
            raise ValueError("ref_txid hex must be exactly 64 hex chars")
        return bytes.fromhex(s)
    raise ValueError(f"ref_txid must be bytes(32) or hex(64); got {type(ref).__name__}")


def sample_ref_instrument(name, ref_txid, *, src_start_ms, src_len_ms,
                          srate=22050, base_note=60, normalize=True,
                          loop_start=0, loop_end=0, volume=255, attack_ms=0,
                          decay_ms=0, sustain_level=255, release_ms=20):
    """A whole-sample instrument (kind 3) whose audio is RESOLVED from another
    sound quipu: decode ref_txid, take [src_start_ms, +src_len_ms], resample to
    `srate`, optionally peak-normalize. No PCM is stored — only the reference."""
    return {"kind": KIND_SAMPLE_REF, "name": str(name), "volume": int(volume),
            "attack_ms": int(attack_ms), "decay_ms": int(decay_ms),
            "sustain_level": int(sustain_level), "release_ms": int(release_ms),
            "ref_txid": _coerce_txid(ref_txid), "srate": int(srate),
            "base_note": int(base_note),
            "flags": (REF_NORMALIZE if normalize else 0),
            "src_start_ms": int(src_start_ms), "src_len_ms": int(src_len_ms),
            "loop_start": int(loop_start), "loop_end": int(loop_end)}


def sliced_ref_instrument(name, ref_txid, slices, *, src_start_ms, src_len_ms,
                          srate=22050, base_note=60, normalize=True,
                          volume=255, attack_ms=1, decay_ms=0,
                          sustain_level=255, release_ms=8):
    """A sliced instrument (kind 4) whose clip is RESOLVED from another sound
    quipu: decode ref_txid, take [src_start_ms, +src_len_ms], resample to `srate`,
    optionally normalize, then index `slices` (start,length in RESOLVED samples)."""
    return {"kind": KIND_SLICED_REF, "name": str(name), "volume": int(volume),
            "attack_ms": int(attack_ms), "decay_ms": int(decay_ms),
            "sustain_level": int(sustain_level), "release_ms": int(release_ms),
            "ref_txid": _coerce_txid(ref_txid), "srate": int(srate),
            "base_note": int(base_note),
            "flags": (REF_NORMALIZE if normalize else 0),
            "src_start_ms": int(src_start_ms), "src_len_ms": int(src_len_ms),
            "slices": [(int(s), int(l)) for (s, l) in slices]}


def event(row, channel, note, instrument, volume=255):
    return {"row": int(row), "channel": int(channel), "note": int(note),
            "instrument": int(instrument), "volume": int(volume)}


def pattern(num_rows, events):
    return {"num_rows": int(num_rows), "events": list(events)}


# ---------------------------------------------------------------------------
# build / read — the 0x20 body wire format (keyless, struct only)
# ---------------------------------------------------------------------------

def build_music_body(*, tempo_bpm, rows_per_beat, num_channels, instruments,
                     patterns, order, master_volume=255):
    out = bytearray(MUSIC_MAGIC)
    out += bytes([_u8(MUSIC_VERSION, "version")])
    out += struct.pack(">H", _u16(tempo_bpm, "tempo_bpm"))
    out += bytes([
        _u8(rows_per_beat, "rows_per_beat"), _u8(num_channels, "num_channels"),
        _u8(len(instruments), "num_instruments"), _u8(len(patterns), "num_patterns"),
        _u8(len(order), "order_len"), _u8(master_volume, "master_volume"),
    ])
    for idx, ins in enumerate(instruments):
        kind = _u8(ins["kind"], f"instrument[{idx}].kind")
        nm = str(ins["name"]).encode("utf-8")
        if len(nm) > 255:
            raise ValueError(f"instrument[{idx}] name > 255 UTF-8 bytes")
        out += bytes([kind, len(nm)]) + nm
        out += bytes([_u8(ins["volume"], "volume")])
        out += struct.pack(">H", _u16(ins["attack_ms"], "attack_ms"))
        out += struct.pack(">H", _u16(ins["decay_ms"], "decay_ms"))
        out += bytes([_u8(ins["sustain_level"], "sustain_level")])
        out += struct.pack(">H", _u16(ins["release_ms"], "release_ms"))
        if kind == KIND_SYNTH:
            out += bytes([_u8(ins["waveform"], "waveform"), _u8(ins["duty"], "duty")])
        elif kind == KIND_SAMPLE:
            bits = _u8(ins.get("bits", 8), "bits")
            pcm = bytes(ins["pcm"])
            out += struct.pack(">H", _u16(ins["srate"], "srate"))
            out += bytes([_u8(ins["base_note"], "base_note"), bits])
            out += struct.pack(">H", _u16(ins["loop_start"], "loop_start"))
            out += struct.pack(">H", _u16(ins["loop_end"], "loop_end"))
            out += struct.pack(">I", _u32(_samples_in(pcm, bits), "pcm_len"))
            out += pcm
        elif kind == KIND_SLICED:
            bits = _u8(ins.get("bits", 8), "bits")
            pcm = bytes(ins["pcm"])
            slices = list(ins["slices"])
            out += struct.pack(">H", _u16(ins["srate"], "srate"))
            out += bytes([_u8(ins.get("base_note", 60), "base_note"), bits])
            out += struct.pack(">H", _u16(len(slices), "num_slices"))
            for s, l in slices:
                out += struct.pack(">II", _u32(s, "slice.start"), _u32(l, "slice.length"))
            out += struct.pack(">I", _u32(_samples_in(pcm, bits), "pcm_len"))
            out += pcm
        elif kind == KIND_SAMPLE_REF:
            out += _coerce_txid(ins["ref_txid"])
            out += struct.pack(">H", _u16(ins["srate"], "srate"))
            out += bytes([_u8(ins["base_note"], "base_note"),
                          _u8(ins.get("flags", 0), "flags")])
            out += struct.pack(">II", _u32(ins["src_start_ms"], "src_start_ms"),
                               _u32(ins["src_len_ms"], "src_len_ms"))
            out += struct.pack(">II", _u32(ins.get("loop_start", 0), "loop_start"),
                               _u32(ins.get("loop_end", 0), "loop_end"))
        elif kind == KIND_SLICED_REF:
            out += _coerce_txid(ins["ref_txid"])
            out += struct.pack(">H", _u16(ins["srate"], "srate"))
            out += bytes([_u8(ins["base_note"], "base_note"),
                          _u8(ins.get("flags", 0), "flags")])
            out += struct.pack(">II", _u32(ins["src_start_ms"], "src_start_ms"),
                               _u32(ins["src_len_ms"], "src_len_ms"))
            slices = list(ins["slices"])
            out += struct.pack(">H", _u16(len(slices), "num_slices"))
            for s, l in slices:
                out += struct.pack(">II", _u32(s, "slice.start"), _u32(l, "slice.length"))
        else:
            raise ValueError(f"instrument[{idx}] unknown kind {kind}")
    for pi, pat in enumerate(patterns):
        evs = list(pat["events"])
        out += struct.pack(">H", _u16(pat["num_rows"], f"pattern[{pi}].num_rows"))
        out += struct.pack(">H", _u16(len(evs), f"pattern[{pi}].num_events"))
        for e in evs:
            out += struct.pack(">H", _u16(e["row"], "row"))
            out += bytes([
                _u8(e["channel"], "channel"),
                _u8(e["note"], "note"), _u8(e["instrument"], "instrument"),
                _u8(e["volume"], "volume"),
            ])
    for o in order:
        out += bytes([_u8(o, "order")])
    return bytes(out)


def read_music_body(body):
    """Parse a music module body. Returns a dict; pcm is RAW BYTES (the DSP layer
    decodes per `bits`). Keyless, pure-stdlib."""
    body = bytes(body)
    if body[:2] != MUSIC_MAGIC:
        raise ValueError("not a music module ('QM' magic missing)")
    p = 2

    def need(n):
        if p + n > len(body):
            raise ValueError(f"music body truncated at offset {p} (need {n})")

    need(1); version = body[p]; p += 1
    if version not in (1, 2):
        raise ValueError(f"unsupported music version {version}")
    row_wide = version >= 2            # v2 carries num_rows & event.row as u16
    need(2); tempo_bpm = struct.unpack(">H", body[p:p + 2])[0]; p += 2
    need(6)
    rows_per_beat, num_channels, num_instruments, num_patterns, order_len, master_volume = body[p:p + 6]
    p += 6

    instruments = []
    for _ in range(num_instruments):
        need(2); kind = body[p]; nl = body[p + 1]; p += 2
        need(nl); name = body[p:p + nl].decode("utf-8", "replace"); p += nl
        need(1); volume = body[p]; p += 1
        need(2); attack_ms = struct.unpack(">H", body[p:p + 2])[0]; p += 2
        need(2); decay_ms = struct.unpack(">H", body[p:p + 2])[0]; p += 2
        need(1); sustain_level = body[p]; p += 1
        need(2); release_ms = struct.unpack(">H", body[p:p + 2])[0]; p += 2
        ins = {"kind": kind, "name": name, "volume": volume, "attack_ms": attack_ms,
               "decay_ms": decay_ms, "sustain_level": sustain_level, "release_ms": release_ms}
        if kind == KIND_SYNTH:
            need(2); ins["waveform"] = body[p]; ins["duty"] = body[p + 1]; p += 2
        elif kind == KIND_SAMPLE:
            need(2); ins["srate"] = struct.unpack(">H", body[p:p + 2])[0]; p += 2
            need(1); ins["base_note"] = body[p]; p += 1
            need(1); bits = body[p]; ins["bits"] = bits; p += 1
            need(2); ins["loop_start"] = struct.unpack(">H", body[p:p + 2])[0]; p += 2
            need(2); ins["loop_end"] = struct.unpack(">H", body[p:p + 2])[0]; p += 2
            need(4); pcm_len = struct.unpack(">I", body[p:p + 4])[0]; p += 4
            nb = pcm_len * (bits // 8); need(nb)
            ins["pcm"] = body[p:p + nb]; p += nb
        elif kind == KIND_SLICED:
            need(2); ins["srate"] = struct.unpack(">H", body[p:p + 2])[0]; p += 2
            need(1); ins["base_note"] = body[p]; p += 1
            need(1); bits = body[p]; ins["bits"] = bits; p += 1
            need(2); num_slices = struct.unpack(">H", body[p:p + 2])[0]; p += 2
            sl = []
            for _ in range(num_slices):
                need(8); s, l = struct.unpack(">II", body[p:p + 8]); p += 8; sl.append((s, l))
            ins["slices"] = sl
            need(4); pcm_len = struct.unpack(">I", body[p:p + 4])[0]; p += 4
            nb = pcm_len * (bits // 8); need(nb)
            ins["pcm"] = body[p:p + nb]; p += nb
        elif kind == KIND_SAMPLE_REF:
            need(32); ins["ref_txid"] = body[p:p + 32]; p += 32
            need(2); ins["srate"] = struct.unpack(">H", body[p:p + 2])[0]; p += 2
            need(2); ins["base_note"] = body[p]; ins["flags"] = body[p + 1]; p += 2
            need(8); ins["src_start_ms"], ins["src_len_ms"] = struct.unpack(">II", body[p:p + 8]); p += 8
            need(8); ins["loop_start"], ins["loop_end"] = struct.unpack(">II", body[p:p + 8]); p += 8
        elif kind == KIND_SLICED_REF:
            need(32); ins["ref_txid"] = body[p:p + 32]; p += 32
            need(2); ins["srate"] = struct.unpack(">H", body[p:p + 2])[0]; p += 2
            need(2); ins["base_note"] = body[p]; ins["flags"] = body[p + 1]; p += 2
            need(8); ins["src_start_ms"], ins["src_len_ms"] = struct.unpack(">II", body[p:p + 8]); p += 8
            need(2); num_slices = struct.unpack(">H", body[p:p + 2])[0]; p += 2
            sl = []
            for _ in range(num_slices):
                need(8); s, l = struct.unpack(">II", body[p:p + 8]); p += 8; sl.append((s, l))
            ins["slices"] = sl
        else:
            raise ValueError(f"unknown instrument kind {kind}")
        instruments.append(ins)

    patterns = []
    for _ in range(num_patterns):
        if row_wide:
            need(2); num_rows = struct.unpack(">H", body[p:p + 2])[0]; p += 2
        else:
            need(1); num_rows = body[p]; p += 1
        need(2); num_events = struct.unpack(">H", body[p:p + 2])[0]; p += 2
        evs = []
        for _ in range(num_events):
            if row_wide:
                need(6); row = struct.unpack(">H", body[p:p + 2])[0]
                ch, note, inst, vol = body[p + 2:p + 6]; p += 6
            else:
                need(5); row, ch, note, inst, vol = body[p:p + 5]; p += 5
            evs.append({"row": row, "channel": ch, "note": note,
                        "instrument": inst, "volume": vol})
        patterns.append({"num_rows": num_rows, "events": evs})

    need(order_len); order = list(body[p:p + order_len]); p += order_len
    return {"version": version, "tempo_bpm": tempo_bpm, "rows_per_beat": rows_per_beat,
            "num_channels": num_channels, "master_volume": master_volume,
            "instruments": instruments, "patterns": patterns, "order": order}
