"""Stage 4: Original 120.00-second stereo 48 kHz instrumental score and restrained environmental DSP.

Synthesizes all audio purely from numerical DSP (zero samples, zero narration/vocals):
- Original three-note leitmotif: D4 -> A4 -> F4 (evolving to radiant D major D4 -> A4 -> F#4).
- Separate stems: felt_piano, low_strings, upper_strings_harp, brass_orchestral, foley_env.
- Stereo hall reverberation, exact 120.00s sample length (5,760,000 samples),
  integrated loudness targeting ~-16 LUFS and -1.0 dBTP peak ceiling.
"""

from __future__ import annotations

import math
import os
import wave
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, lfilter

SR = 48000
DURATION_SEC = 120.0
TOTAL_SAMPLES = int(round(SR * DURATION_SEC))  # 5,760,000 exact


# ---------------------------------------------------------------------------
# Musical Pitch & Filter Utilities
# ---------------------------------------------------------------------------

def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def _bandpass(sig: np.ndarray, lo: float, hi: float, order: int = 2) -> np.ndarray:
    nyq = SR * 0.5
    b, a = butter(order, [max(10.0, lo) / nyq, min(nyq * 0.95, hi) / nyq], btype="band")
    return lfilter(b, a, sig).astype(np.float32)


def _lowpass(sig: np.ndarray, cutoff: float, order: int = 2) -> np.ndarray:
    nyq = SR * 0.5
    b, a = butter(order, min(nyq * 0.95, max(20.0, cutoff)) / nyq, btype="low")
    return lfilter(b, a, sig).astype(np.float32)


def _highpass(sig: np.ndarray, cutoff: float, order: int = 2) -> np.ndarray:
    nyq = SR * 0.5
    b, a = butter(order, min(nyq * 0.95, max(20.0, cutoff)) / nyq, btype="high")
    return lfilter(b, a, sig).astype(np.float32)


def _add_stereo(buf: np.ndarray, mono: np.ndarray, start_sec: float, pan: float = 0.0, gain: float = 1.0) -> None:
    """Mix mono clip into stereo buffer (TOTAL_SAMPLES, 2) with equal-power pan (-1..+1)."""
    i0 = int(round(start_sec * SR))
    if i0 >= TOTAL_SAMPLES or len(mono) == 0:
        return
    if i0 < 0:
        mono = mono[-i0:]
        i0 = 0
    n = min(len(mono), TOTAL_SAMPLES - i0)
    if n <= 0:
        return
    ang = (max(-1.0, min(1.0, pan)) + 1.0) * (math.pi * 0.25)
    gl = math.cos(ang) * gain
    gr = math.sin(ang) * gain
    seg = mono[:n]
    buf[i0:i0 + n, 0] += seg * gl
    buf[i0:i0 + n, 1] += seg * gr


# ---------------------------------------------------------------------------
# Instrument Synthesizers (Modal / Additive / Formant Physical Models)
# ---------------------------------------------------------------------------

def synth_felt_piano_note(midi: float, dur_sec: float, vel: float = 0.7, rng: np.random.Generator = None) -> np.ndarray:
    """Warm, intimate felt-hammer piano tone via modal partials + soft felt thump."""
    if rng is None:
        rng = np.random.default_rng(1)
    f0 = midi_to_hz(midi)
    tail = min(4.8, dur_sec + 1.6)
    n = int(round(tail * SR))
    t = np.arange(n, dtype=np.float32) / SR

    # Soft felt hammer suppresses harsh upper partials
    inharm = 0.00018
    sig = np.zeros(n, dtype=np.float32)
    partials = [
        (1, 1.00, 1.00),
        (2, 0.44, 1.45),
        (3, 0.20, 2.05),
        (4, 0.09, 2.85),
        (5, 0.035, 3.70),
        (6, 0.015, 4.60),
    ]
    for k, amp, damp in partials:
        fk = f0 * k * math.sqrt(1.0 + inharm * k * k)
        if fk > SR * 0.42:
            continue
        # Double-string slight chorus detune for authentic acoustic warmth
        decay = np.exp(-t * (0.58 * damp / max(0.35, dur_sec * 0.55))).astype(np.float32)
        ph = float(rng.uniform(0.0, 2.0 * math.pi))
        s_k = (np.sin(2.0 * math.pi * fk * t + ph) + 0.48 * np.sin(2.0 * math.pi * (fk * 1.0009) * t + ph * 0.7))
        sig += s_k.astype(np.float32) * (amp * decay)

    # Felt hammer mechanical whisper (15ms soft low-mid transient)
    n_th = min(n, int(0.025 * SR))
    thump = rng.standard_normal(n_th).astype(np.float32)
    thump = _bandpass(thump, 90.0, 620.0, order=2) * np.exp(-np.arange(n_th, dtype=np.float32) / (0.006 * SR))
    sig[:n_th] += thump * 0.11

    # Smooth 9ms felt attack + gentle release fade
    att = np.minimum(1.0, t / 0.009)
    rel_t = np.maximum(0.0, t - dur_sec)
    rel = np.exp(-rel_t * 2.8)
    end_fade = np.clip((tail - t) / 0.08, 0.0, 1.0)
    return (sig * att * rel * end_fade * (0.24 * vel)).astype(np.float32)


def synth_bowed_string(
    midi: float,
    dur_sec: float,
    vel: float = 0.65,
    is_cello: bool = True,
    vibrato_hz: float = 4.8,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Expressive bowed cello or violin/viola via bandlimited harmonics + wooden body formants."""
    if rng is None:
        rng = np.random.default_rng(2)
    f0 = midi_to_hz(midi)
    n = int(round((dur_sec + 0.55) * SR))
    t = np.arange(n, dtype=np.float32) / SR

    # Delayed organic vibrato
    vib_env = np.clip((t - 0.35) / 0.65, 0.0, 1.0)
    vib_depth = 0.0055 if is_cello else 0.0072
    inst_freq = f0 * (1.0 + vib_depth * vib_env * np.sin(2.0 * math.pi * vibrato_hz * t))
    phase = np.cumsum(2.0 * math.pi * inst_freq / SR).astype(np.float64)

    sig = np.zeros(n, dtype=np.float32)
    n_harm = 12 if is_cello else 9
    for k in range(1, n_harm + 1):
        if f0 * k > 6800.0:
            break
        # Bowed string 1/k^1.25 rolloff with mild odd-harmonic warmth
        amp = (1.0 / (k ** 1.28)) * (1.12 if k % 2 == 1 else 0.92)
        sig += (np.sin(phase * k) * amp).astype(np.float32)

    # Subtle rosin bow friction noise
    rosin = _bandpass(rng.standard_normal(n).astype(np.float32), 1400.0 if is_cello else 2600.0, 4800.0, order=2)
    sig += rosin * (0.018 if is_cello else 0.014)

    # Resonant wooden instrument body shaping
    if is_cello:
        body = _bandpass(sig, 110.0, 1850.0, order=2) * 0.72 + _lowpass(sig, 680.0, order=2) * 0.52
    else:
        body = _bandpass(sig, 260.0, 4400.0, order=2) * 0.75 + _lowpass(sig, 1600.0, order=2) * 0.35

    # Swelling bow envelope
    att_t = min(0.45, dur_sec * 0.32)
    att = np.sin(np.clip(t / max(0.04, att_t), 0.0, 1.0) * math.pi * 0.5)
    rel = np.clip(1.0 - np.maximum(0.0, t - dur_sec) / 0.52, 0.0, 1.0) ** 1.6
    swell = 0.86 + 0.18 * np.sin(np.clip(t / max(0.1, dur_sec), 0.0, 1.0) * math.pi)
    return (body * att * rel * swell * (0.20 * vel)).astype(np.float32)


def synth_harp_harmonic(midi: float, dur_sec: float = 2.6, vel: float = 0.65, rng: np.random.Generator = None) -> np.ndarray:
    """Luminous plucked harp & upper harmonic shimmer for optical/prism sequence."""
    if rng is None:
        rng = np.random.default_rng(3)
    f0 = midi_to_hz(midi)
    n = int(round(dur_sec * SR))
    t = np.arange(n, dtype=np.float32) / SR
    sig = np.zeros(n, dtype=np.float32)
    for k, amp, damp in ((1, 1.0, 1.15), (2, 0.52, 1.8), (3, 0.22, 2.7), (4, 0.08, 3.8)):
        if f0 * k < SR * 0.42:
            sig += (np.sin(2.0 * math.pi * f0 * k * t) * amp * np.exp(-t * damp)).astype(np.float32)
    att = np.minimum(1.0, t / 0.004)
    end_fade = np.clip((dur_sec - t) / 0.12, 0.0, 1.0)
    return (sig * att * end_fade * (0.19 * vel)).astype(np.float32)


def synth_french_horn_pad(midi: float, dur_sec: float, vel: float = 0.6) -> np.ndarray:
    """Warm orchestral horn swell for the cosmic and Principia climaxes."""
    f0 = midi_to_hz(midi)
    n = int(round((dur_sec + 0.7) * SR))
    t = np.arange(n, dtype=np.float32) / SR
    sig = np.zeros(n, dtype=np.float32)
    for k, amp in ((1, 1.0), (2, 0.68), (3, 0.42), (4, 0.22), (5, 0.11), (6, 0.05)):
        if f0 * k < 3200.0:
            sig += (np.sin(2.0 * math.pi * f0 * k * t) * amp).astype(np.float32)
    sig = _lowpass(sig, 1400.0, order=2)
    att = np.sin(np.clip(t / min(0.85, dur_sec * 0.35), 0.0, 1.0) * math.pi * 0.5)
    rel = np.clip(1.0 - np.maximum(0.0, t - dur_sec) / 0.65, 0.0, 1.0) ** 1.7
    return (sig * att * rel * (0.16 * vel)).astype(np.float32)


# ---------------------------------------------------------------------------
# Algorithmic Stereo Hall Reverb (Schroeder Multi-Tap + Diffuser)
# ---------------------------------------------------------------------------

def apply_stereo_reverb(stereo: np.ndarray, wet: float = 0.28, decay_s: float = 1.9) -> np.ndarray:
    """Warm concert-hall algorithmic stereo reverberation via fast FFT convolution."""
    from scipy.signal import fftconvolve

    ir_len = int(min(decay_s * 1.4, 2.4) * SR)
    t_ir = np.arange(ir_len, dtype=np.float32) / SR
    env = np.exp(-3.5 * t_ir / max(0.3, decay_s)).astype(np.float32)
    rng = np.random.default_rng(77123)
    mono_in = (0.5 * (stereo[:, 0] + stereo[:, 1])).astype(np.float32)
    rev = np.zeros_like(stereo)
    for ch in (0, 1):
        ir = rng.standard_normal(ir_len).astype(np.float32) * env
        # Early reflections + warm high-frequency absorption
        ir[:int(0.018 * SR)] *= 0.15
        ir = _lowpass(ir, 3400.0, order=2)
        ir /= (np.sqrt(np.sum(ir * ir)) + 1e-6)
        conv = fftconvolve(mono_in, ir, mode="full")[:len(mono_in)]
        rev[:, ch] = conv.astype(np.float32)
    return stereo * (1.0 - wet * 0.32) + rev * wet


# ---------------------------------------------------------------------------
# Environmental Foley Stem (Restrained, Physically Grounded, Zero Space SFX!)
# ---------------------------------------------------------------------------

def build_foley_stem(rng: np.random.Generator) -> np.ndarray:
    foley = np.zeros((TOTAL_SAMPLES, 2), dtype=np.float32)

    # 1. Shot 1 (0:00 - 0:08.5): Winter English wind across frozen hills
    n_wind = int(8.6 * SR)
    w_noise = rng.standard_normal(n_wind).astype(np.float32)
    t_w = np.arange(n_wind, dtype=np.float32) / SR
    gust = 0.45 + 0.55 * (0.5 + 0.5 * np.sin(t_w * 0.85 - 0.6)) * (0.6 + 0.4 * np.sin(t_w * 1.9))
    env_w = np.clip(t_w / 1.4, 0.0, 1.0) * np.clip((8.6 - t_w) / 2.2, 0.0, 1.0)
    wind_lo = _bandpass(w_noise, 140.0, 480.0, order=2) * gust * env_w * 0.14
    wind_hi = _bandpass(rng.standard_normal(n_wind).astype(np.float32), 620.0, 1350.0, order=2) * (gust ** 1.5) * env_w * 0.045
    _add_stereo(foley, wind_lo + wind_hi, 0.0, pan=-0.22, gain=1.0)
    _add_stereo(foley, _bandpass(rng.standard_normal(n_wind).astype(np.float32), 180.0, 560.0, order=2) * gust * env_w * 0.11, 0.0, pan=0.25, gain=1.0)

    # Hearth candle crackle inside Woolsthorpe (0:05.0 - 0:09.0) & candlelit studies (0:22-0:30, 0:53-1:04)
    for seg_start, seg_end, lvl in ((5.0, 9.0, 0.08), (22.0, 30.0, 0.065), (53.0, 64.0, 0.06)):
        t_cur = seg_start + 0.2
        while t_cur < seg_end - 0.2:
            n_cr = int(0.014 * SR)
            cr = _bandpass(rng.standard_normal(n_cr).astype(np.float32), 1800.0, 6500.0, order=2)
            cr *= np.exp(-np.arange(n_cr, dtype=np.float32) / (0.0022 * SR))
            _add_stereo(foley, cr, t_cur, pan=float(rng.uniform(-0.35, 0.35)), gain=lvl * float(rng.uniform(0.4, 1.1)))
            t_cur += float(rng.uniform(0.18, 0.72))

    # 2. Shot 2 (0:09 - 0:19): Gentle rain on leaded farmhouse window + wooden mechanism clicks
    n_rain = int(10.0 * SR)
    t_r = np.arange(n_rain, dtype=np.float32) / SR
    r_env = np.clip(t_r / 0.8, 0.0, 1.0) * np.clip((10.0 - t_r) / 1.0, 0.0, 1.0)
    rain_l = _bandpass(rng.standard_normal(n_rain).astype(np.float32), 900.0, 5200.0, order=2) * r_env * 0.042
    rain_r = _bandpass(rng.standard_normal(n_rain).astype(np.float32), 1000.0, 5400.0, order=2) * r_env * 0.042
    _add_stereo(foley, rain_l, 9.0, pan=-0.55, gain=1.0)
    _add_stereo(foley, rain_r, 9.0, pan=0.55, gain=1.0)
    # Soft miniature wooden windmill ratchet ticks (0:10.5 - 0:14.2)
    for k in range(11):
        t_tick = 10.6 + k * 0.31
        n_tk = int(0.018 * SR)
        tk = _bandpass(rng.standard_normal(n_tk).astype(np.float32), 550.0, 1900.0, order=2)
        tk *= np.exp(-np.arange(n_tk, dtype=np.float32) / (0.003 * SR))
        _add_stereo(foley, tk, t_tick, pan=0.12, gain=0.14)

    # 3. Shot 3 & Shot 6 & Shot 9: Quill scratching on parchment & soft page turns
    quill_windows = [(22.8, 29.2), (53.6, 58.2), (61.5, 63.6), (90.5, 93.2)]
    for qs, qe in quill_windows:
        tc = qs
        while tc < qe:
            dur = float(rng.uniform(0.16, 0.42))
            nq = int(dur * SR)
            tq = np.arange(nq, dtype=np.float32) / SR
            q_env = np.sin(np.pi * tq / dur) ** 1.4
            scratch = _bandpass(rng.standard_normal(nq).astype(np.float32), 1800.0, 5400.0, order=2) * q_env
            _add_stereo(foley, scratch, tc, pan=0.18, gain=0.055)
            tc += dur + float(rng.uniform(0.14, 0.36))
    # Page turns
    for pt in (25.4, 28.1, 61.1, 92.2):
        np_t = int(0.32 * SR)
        tp = np.arange(np_t, dtype=np.float32) / SR
        p_env = np.sin(np.pi * tp / 0.32) ** 1.8
        pg = _bandpass(rng.standard_normal(np_t).astype(np.float32), 600.0, 2600.0, order=2) * p_env
        _add_stereo(foley, pg, pt, pan=-0.15, gain=0.095)

    # 4. Shot 4 (0:30 - 0:40): Distant Cambridge church bell + quiet footsteps on moorland path
    for bt in (30.6, 33.9):
        nb = int(3.2 * SR)
        tb = np.arange(nb, dtype=np.float32) / SR
        # Distant minor-third cathedral bell (D3 = 146.8 Hz)
        bell = (
            np.sin(2 * math.pi * 146.83 * tb) * np.exp(-tb * 0.9)
            + 0.45 * np.sin(2 * math.pi * 293.66 * tb) * np.exp(-tb * 1.4)
            + 0.22 * np.sin(2 * math.pi * 349.23 * tb) * np.exp(-tb * 1.9)
        ).astype(np.float32)
        bell = _lowpass(bell, 950.0, order=2) * np.minimum(1.0, tb / 0.025)
        _add_stereo(foley, bell, bt, pan=-0.45, gain=0.075)
    # Footsteps (0:35.0 - 0:39.6)
    for k in range(8):
        ft = 35.1 + k * 0.56
        nf = int(0.08 * SR)
        step = _bandpass(rng.standard_normal(nf).astype(np.float32), 120.0, 820.0, order=2)
        step *= np.exp(-np.arange(nf, dtype=np.float32) / (0.018 * SR))
        _add_stereo(foley, step, ft, pan=-0.1 + 0.04 * k, gain=0.11)

    # 5. Shot 5 (0:40 - 0:53): Warm orchard breeze + soft apple thud in grass at 0:44.45
    n_br = int(7.5 * SR)
    t_b = np.arange(n_br, dtype=np.float32) / SR
    b_env = np.clip(t_b / 1.0, 0.0, 1.0) * np.clip((7.5 - t_b) / 1.5, 0.0, 1.0)
    breeze = _bandpass(rng.standard_normal(n_br).astype(np.float32), 450.0, 1900.0, order=2) * b_env * 0.028
    _add_stereo(foley, breeze, 40.0, pan=0.0, gain=1.0)
    # Apple soft thud into orchard grass (synced to Shot 5 landing at ~44.45s)
    n_th = int(0.14 * SR)
    t_th = np.arange(n_th, dtype=np.float32) / SR
    thud = (np.sin(2 * math.pi * (82.0 * np.exp(-t_th * 12.0)) * t_th) * np.exp(-t_th * 26.0)).astype(np.float32)
    grass_rustle = _bandpass(rng.standard_normal(n_th).astype(np.float32), 800.0, 3200.0, order=2) * np.exp(-t_th * 18.0) * 0.25
    _add_stereo(foley, thud * 0.35 + grass_rustle * 0.14, 44.45, pan=0.22, gain=1.0)

    # 6. Shot 7 (1:04.5): Wooden window shutter closing with soft wooden thud
    n_sh = int(0.12 * SR)
    t_sh = np.arange(n_sh, dtype=np.float32) / SR
    shutter = _bandpass(rng.standard_normal(n_sh).astype(np.float32), 140.0, 680.0, order=2) * np.exp(-t_sh * 24.0)
    _add_stereo(foley, shutter, 64.6, pan=-0.35, gain=0.28)

    # NOTE: Shot 8 (1:15 - 1:29) vacuum of space has ZERO literal sound effects — pure non-diegetic orchestral score!

    # 7. Shot 9 (1:37.0 - 1:40.2): Historical wooden printing press creak and inked sheet impression
    n_pr = int(0.45 * SR)
    t_pr = np.arange(n_pr, dtype=np.float32) / SR
    creak = _bandpass(rng.standard_normal(n_pr).astype(np.float32), 220.0, 920.0, order=2) * np.sin(np.pi * t_pr / 0.45) * 0.14
    _add_stereo(foley, creak, 97.1, pan=-0.1, gain=1.0)
    n_imp = int(0.15 * SR)
    imp = _lowpass(rng.standard_normal(n_imp).astype(np.float32), 340.0, order=2) * np.exp(-np.arange(n_imp, dtype=np.float32) / (0.025 * SR))
    _add_stereo(foley, imp, 97.7, pan=0.05, gain=0.26)

    # 8. Shot 10 (1:42 - 1:48): Distant London stone carriage murmur + Royal Mint silver coin chime
    n_ld = int(5.0 * SR)
    t_ld = np.arange(n_ld, dtype=np.float32) / SR
    ld_env = np.clip(t_ld / 0.8, 0.0, 1.0) * np.clip((5.0 - t_ld) / 1.2, 0.0, 1.0)
    murmur = _bandpass(rng.standard_normal(n_ld).astype(np.float32), 80.0, 320.0, order=2) * ld_env * 0.045
    _add_stereo(foley, murmur, 102.0, pan=0.0, gain=1.0)
    # Delicate silver coin clink on Royal Mint balance scale (1:45.2)
    n_cn = int(0.45 * SR)
    t_cn = np.arange(n_cn, dtype=np.float32) / SR
    coin = (
        np.sin(2 * math.pi * 3840.0 * t_cn) * np.exp(-t_cn * 11.0)
        + 0.5 * np.sin(2 * math.pi * 5910.0 * t_cn) * np.exp(-t_cn * 15.0)
    ).astype(np.float32)
    _add_stereo(foley, coin, 105.2, pan=0.14, gain=0.055)

    return foley


# ---------------------------------------------------------------------------
# Full Synchronized 120-Second Original Score Composer
# ---------------------------------------------------------------------------

def synthesize_master_audio(out_wav_path: str, stems_dir: str = "") -> Dict[str, float]:
    """Compose, render separate stems, mix, reverberate, and loudness-normalize the 120.00s score."""
    rng = np.random.default_rng(16421727)

    piano = np.zeros((TOTAL_SAMPLES, 2), dtype=np.float32)
    low_str = np.zeros((TOTAL_SAMPLES, 2), dtype=np.float32)
    hi_str_harp = np.zeros((TOTAL_SAMPLES, 2), dtype=np.float32)
    brass = np.zeros((TOTAL_SAMPLES, 2), dtype=np.float32)

    def add_piano(midi: float, t0: float, dur: float, vel: float = 0.7, pan: float = 0.0) -> None:
        _add_stereo(piano, synth_felt_piano_note(midi, dur, vel, rng), t0, pan=pan, gain=1.0)

    def add_cello(midi: float, t0: float, dur: float, vel: float = 0.65, pan: float = -0.22) -> None:
        _add_stereo(low_str, synth_bowed_string(midi, dur, vel, is_cello=True, rng=rng), t0, pan=pan, gain=1.0)

    def add_violin(midi: float, t0: float, dur: float, vel: float = 0.58, pan: float = 0.24) -> None:
        _add_stereo(hi_str_harp, synth_bowed_string(midi, dur, vel, is_cello=False, vibrato_hz=5.3, rng=rng), t0, pan=pan, gain=1.0)

    def add_harp(midi: float, t0: float, dur: float = 2.4, vel: float = 0.6, pan: float = 0.30) -> None:
        _add_stereo(hi_str_harp, synth_harp_harmonic(midi, dur, vel, rng), t0, pan=pan, gain=1.0)

    def add_horn(midi: float, t0: float, dur: float, vel: float = 0.55, pan: float = -0.1) -> None:
        _add_stereo(brass, synth_french_horn_pad(midi, dur, vel), t0, pan=pan, gain=1.0)

    # Original Three-Note Leitmotif: D4 (62) -> A4 (69) -> F4 (65)  [in D minor / Dorian]
    # Major Resolution Variant:      D4 (62) -> A4 (69) -> F#4 (66) [in D major]

    # --- SHOT 1 (0:00 - 0:09): Wind & near silence -> Felt piano introduces 3-note motif at 0:04.2 ---
    add_piano(62, 4.2, 1.5, 0.54, -0.05)   # D4
    add_piano(69, 5.5, 1.4, 0.58, 0.06)    # A4
    add_piano(65, 6.8, 2.2, 0.52, 0.02)    # F4
    add_piano(50, 4.2, 4.5, 0.34, -0.15)   # D3 soft pedal warmth

    # --- SHOT 2 (0:09 - 0:19): Curious Child — delicate questioning ostinato + motif ---
    for bar, (bass, top_seq) in enumerate([
        (50, [62, 69, 65, 64, 62, 67]),
        (48, [60, 67, 64, 62, 65, 64]),
        (46, [62, 69, 65, 70, 69, 65]),
    ]):
        t_bar = 9.4 + bar * 3.1
        add_piano(bass, t_bar, 2.8, 0.42, -0.18)
        for i, m in enumerate(top_seq):
            add_piano(m, t_bar + i * 0.48, 0.9, 0.50 + 0.05 * (i == 1), -0.08 + i * 0.04)

    # --- SHOT 3 (0:19 - 0:30): Cambridge — disciplined Baroque-inflected piano + warm cello ---
    cam_chords = [
        (19.2, 50, [62, 69, 65, 69, 72, 69, 65, 64]),
        (22.4, 48, [60, 67, 64, 67, 70, 67, 64, 62]),
        (25.6, 46, [58, 65, 62, 65, 69, 65, 62, 60]),
        (28.4, 45, [57, 64, 61, 64]),
    ]
    for t0, bass, notes in cam_chords:
        add_cello(bass, t0, 3.0 if len(notes) > 4 else 1.8, 0.52, -0.26)
        for i, m in enumerate(notes):
            add_piano(m, t0 + i * 0.38, 0.72, 0.54, -0.1 + 0.03 * i)
    add_violin(74, 22.8, 2.8, 0.32, 0.28)
    add_violin(72, 25.8, 2.6, 0.30, 0.28)

    # --- SHOT 4 (0:30 - 0:40): The Silent Return — score contracts to lone cello & space ---
    # Meaningful pause at 30.0 - 30.8 (only distant bell), then solitary mournful cello
    cello_phrase = [
        (30.9, 50, 2.2, 0.62),  # D3
        (33.2, 57, 1.9, 0.68),  # A3
        (35.2, 53, 2.4, 0.64),  # F3 (motif in low register!)
        (37.7, 52, 1.3, 0.52),  # E3
        (39.0, 50, 1.4, 0.44),  # D3
    ]
    for t0, m, dur, vel in cello_phrase:
        add_cello(m, t0, dur, vel, -0.12)

    # --- SHOT 5 (0:40 - 0:53): The Falling Apple — warm wonder, motif lifts on landing ---
    add_piano(50, 40.4, 3.6, 0.36, -0.15)
    add_piano(62, 41.0, 1.4, 0.46, 0.04)
    add_piano(69, 42.3, 1.4, 0.48, 0.08)
    # Pause of breath as apple detaches (43.7 - 44.4), then luminous chord on grass landing!
    add_piano(53, 44.5, 4.2, 0.54, -0.18)   # F3
    add_piano(65, 44.5, 2.2, 0.60, 0.05)    # F4
    add_piano(69, 45.4, 2.0, 0.58, 0.12)    # A4
    add_piano(74, 46.4, 2.8, 0.62, 0.18)    # D5
    add_cello(53, 45.0, 3.8, 0.56, -0.28)
    add_violin(77, 46.8, 3.2, 0.42, 0.30)
    # Match cut to Earth & Moon orbit (48.5 - 53.0)
    add_cello(55, 48.6, 4.2, 0.62, -0.25)   # G3
    add_violin(74, 48.8, 2.4, 0.50, 0.25)
    add_violin(76, 51.2, 2.0, 0.54, 0.28)
    add_piano(62, 49.0, 1.2, 0.54, -0.05)
    add_piano(69, 50.2, 1.2, 0.58, 0.05)
    add_piano(74, 51.4, 1.8, 0.62, 0.12)

    # --- SHOT 6 (0:53 - 1:04): Mathematics of Change — flowing kinetic pulse ---
    prog6 = [(53.2, 50, [62, 65, 69, 74]), (55.8, 53, [65, 69, 72, 77]),
             (58.4, 55, [67, 70, 74, 79]), (61.0, 57, [64, 69, 73, 76])]
    for t0, bass, arp in prog6:
        add_cello(bass, t0, 2.5, 0.58, -0.24)
        for rep in range(2):
            for idx, m in enumerate(arp):
                add_piano(m, t0 + rep * 1.28 + idx * 0.31, 0.55, 0.52, -0.12 + 0.08 * idx)
        add_violin(arp[-1], t0 + 0.4, 2.1, 0.42, 0.26)

    # --- SHOT 7 (1:04 - 1:15): The Nature of Light — Harp harmonics & luminous upper strings ---
    # Brief hush as shutter closes (64.0 - 65.2), then shimmering spectral bloom!
    harp_notes = [
        (65.3, 74), (65.65, 77), (66.0, 81), (66.35, 86),
        (67.1, 72), (67.45, 76), (67.8, 79), (68.15, 84),
        (69.0, 70), (69.35, 74), (69.7, 77), (70.05, 82),
        (71.0, 74), (71.35, 78), (71.7, 81), (72.05, 86),  # D major hint!
        (73.0, 76), (73.4, 79), (73.8, 83), (74.2, 88),
    ]
    for i, (th, mh) in enumerate(harp_notes):
        add_harp(mh, th, dur=2.6, vel=0.62, pan=-0.35 + (i % 4) * 0.24)
    add_violin(81, 65.8, 3.2, 0.55, 0.34)
    add_violin(79, 69.0, 2.8, 0.52, -0.32)
    add_violin(81, 71.8, 3.2, 0.62, 0.30)
    add_violin(86, 72.4, 2.8, 0.52, -0.25)
    add_piano(62, 66.2, 2.2, 0.44, 0.0)
    add_piano(69, 68.4, 2.2, 0.48, 0.06)
    add_piano(74, 71.2, 2.8, 0.52, -0.05)

    # --- SHOT 8 (1:15 - 1:29): Beyond the Earth — Full orchestral cosmic expansion ---
    cosmic_prog = [
        (75.2, 38, 50, 62, 74, 3.4),  # D minor/open fifth
        (78.6, 46, 58, 65, 77, 3.4),  # Bb maj7
        (82.0, 48, 60, 67, 79, 3.4),  # C add9
        (85.4, 45, 57, 69, 81, 3.6),  # A sus -> resolution prep
    ]
    for t0, b_low, b_mid, horn_n, v_hi, dur in cosmic_prog:
        add_cello(b_low, t0, dur, 0.78, -0.34)
        add_cello(b_mid, t0, dur, 0.70, -0.18)
        add_horn(horn_n, t0 + 0.1, dur - 0.1, 0.68, -0.12)
        add_horn(horn_n + 7, t0 + 0.3, dur - 0.2, 0.54, 0.14)
        add_violin(v_hi, t0, dur, 0.72, 0.32)
        add_violin(v_hi - 5, t0 + 0.2, dur - 0.2, 0.58, -0.26)
        # Expanding leitmotif in harp & piano
        add_piano(62, t0 + 0.2, 1.3, 0.66, 0.0)
        add_piano(69, t0 + 1.2, 1.3, 0.70, 0.08)
        add_piano(74 if t0 > 81 else 65, t0 + 2.2, 1.4, 0.68, -0.06)
        add_harp(v_hi, t0 + 0.5, 2.4, 0.58, 0.38)
        add_harp(v_hi - 7, t0 + 1.6, 2.4, 0.52, -0.36)

    # --- SHOT 9 (1:29 - 1:42): The Principia — Steady resolve -> FIRST MAJOR RESOLUTION at 1:37.5 ---
    # Halley & 3 laws chamber pulse (89.0 - 97.2)
    for idx, (t0, bass, top) in enumerate([
        (89.2, 50, [62, 69, 65, 69]),
        (91.8, 53, [65, 72, 69, 72]),
        (94.4, 55, [67, 74, 71, 74]),
    ]):
        add_cello(bass, t0, 2.5, 0.64, -0.22)
        add_violin(top[1], t0, 2.4, 0.50, 0.25)
        for j, m in enumerate(top):
            add_piano(m, t0 + j * 0.60, 0.9, 0.60, -0.08 + j * 0.05)

    # Printing press reveal of Philosophiæ Naturalis Principia Mathematica, 1687 -> D MAJOR!
    add_cello(38, 97.4, 4.6, 0.80, -0.30)   # D2
    add_cello(50, 97.4, 4.6, 0.72, -0.15)   # D3
    add_horn(62, 97.5, 4.4, 0.72, -0.10)    # D4
    add_horn(66, 97.6, 4.3, 0.66, 0.14)     # F#4 (radiant major third!)
    add_horn(69, 97.7, 4.2, 0.62, 0.04)     # A4
    add_violin(78, 97.5, 4.4, 0.72, 0.32)   # F#5
    add_violin(81, 98.2, 3.8, 0.64, -0.28)  # A5
    # Major-mode leitmotif D4 -> A4 -> F#4
    add_piano(62, 97.5, 1.4, 0.74, -0.04)
    add_piano(69, 98.7, 1.4, 0.76, 0.04)
    add_piano(66, 99.9, 2.3, 0.78, 0.0)
    add_harp(74, 97.6, 2.6, 0.64, 0.35)
    add_harp(78, 98.1, 2.6, 0.62, -0.35)
    add_harp(81, 98.6, 2.8, 0.64, 0.28)

    # --- SHOT 10 (1:42 - 1:51): The Passing Years — reflective, autumnal dignity & quiet fatigue ---
    pass_notes = [
        (102.4, 57, 64, 2.2, 0.50),
        (104.8, 55, 62, 2.2, 0.48),
        (107.2, 53, 60, 2.0, 0.44),
        (109.2, 52, 61, 1.8, 0.42),
    ]
    for t0, cb, pm, dur, vel in pass_notes:
        add_cello(cb, t0, dur, vel * 1.1, -0.20)
        add_piano(pm, t0 + 0.1, dur, vel, 0.06)
        add_piano(pm + 7, t0 + 1.0, dur * 0.7, vel * 0.85, -0.05)

    # --- SHOT 11 (1:51 - 2:00): The Hidden Order — Grand orchestral theme -> single final piano note ---
    add_cello(38, 111.1, 5.4, 0.84, -0.34)
    add_cello(50, 111.1, 5.4, 0.76, -0.18)
    add_horn(62, 111.2, 5.1, 0.74, -0.12)
    add_horn(69, 111.4, 4.9, 0.68, 0.14)
    add_horn(66, 113.4, 3.0, 0.66, 0.04)
    add_violin(74, 111.2, 2.4, 0.76, 0.34)
    add_violin(81, 112.4, 2.6, 0.78, -0.30)
    add_violin(78, 113.8, 2.8, 0.74, 0.28)
    add_violin(86, 114.2, 2.4, 0.66, -0.24)
    # Orchestral-scale leitmotif on piano + harp
    add_piano(62, 111.3, 1.4, 0.76, -0.05)
    add_piano(69, 112.6, 1.4, 0.78, 0.05)
    add_piano(66, 113.9, 2.5, 0.76, 0.0)
    for k, hm in enumerate((74, 78, 81, 86)):
        add_harp(hm, 111.5 + k * 0.45, 2.8, 0.62, -0.35 + k * 0.23)

    # Hold title card -> One solitary soft felt-piano D4 note at 1:57.6, decaying cleanly to 2:00.00!
    add_piano(62, 117.6, 1.4, 0.44, 0.0)

    # Foley environmental stem
    foley = build_foley_stem(rng)

    # Apply stereo hall reverb to musical stems (lighter room reverb on foley)
    piano_wet = apply_stereo_reverb(piano, wet=0.26, decay_s=1.8)
    low_wet = apply_stereo_reverb(low_str, wet=0.30, decay_s=2.1)
    hi_wet = apply_stereo_reverb(hi_str_harp, wet=0.34, decay_s=2.3)
    brass_wet = apply_stereo_reverb(brass, wet=0.32, decay_s=2.4)
    foley_wet = apply_stereo_reverb(foley, wet=0.10, decay_s=0.7)

    # Phrase-level master mix
    mix = piano_wet * 1.05 + low_wet * 0.95 + hi_wet * 0.92 + brass_wet * 0.85 + foley_wet * 0.95

    # Gentle sub-bass cleanup (< 32 Hz) & exact end fade-to-zero over last 0.35 s
    for ch in (0, 1):
        mix[:, ch] = _highpass(mix[:, ch], 30.0, order=2)
    fade_n = int(0.35 * SR)
    mix[-fade_n:, :] *= np.linspace(1.0, 0.0, fade_n, dtype=np.float32)[:, None]
    fade_in = int(0.08 * SR)
    mix[:fade_in, :] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)[:, None]

    # Integrated loudness estimation (K-weighted approximation) -> target -16.0 LUFS, cap true peak at -1.0 dBTP
    # Compute gated RMS over active blocks
    block = int(0.4 * SR)
    powers = []
    for i in range(0, TOTAL_SAMPLES - block, block // 2):
        seg = mix[i:i + block]
        p = float(np.mean(seg ** 2))
        if p > 1e-7:
            powers.append(p)
    mean_p = float(np.mean(powers)) if powers else 1e-5
    est_lufs = -0.691 + 10.0 * math.log10(mean_p + 1e-12)
    target_lufs = -16.0
    gain_db = target_lufs - est_lufs
    gain_lin = 10.0 ** (gain_db / 20.0)
    mix = mix * gain_lin

    # Smooth soft limiter to guarantee <= -1.0 dBTP (0.891 linear) without clipping artifacts
    peak_ceil = 10.0 ** (-1.0 / 20.0)  # 0.89125
    peak_before = float(np.max(np.abs(mix)))
    if peak_before > peak_ceil * 0.92:
        knee = peak_ceil * 0.78
        span = peak_ceil - knee
        mag = np.abs(mix)
        limited = np.where(mag <= knee, mag, knee + span * np.tanh((mag - knee) / max(1e-5, span)))
        mix = np.sign(mix) * limited

    peak_after = float(np.max(np.abs(mix)))
    peak_db = 20.0 * math.log10(max(1e-9, peak_after))
    final_p = float(np.mean([float(np.mean(mix[i:i + block] ** 2)) for i in range(0, TOTAL_SAMPLES - block, block // 2) if float(np.mean(mix[i:i + block] ** 2)) > 1e-7]))
    final_lufs = -0.691 + 10.0 * math.log10(final_p + 1e-12)

    os.makedirs(os.path.dirname(os.path.abspath(out_wav_path)), exist_ok=True)
    _write_wav_16bit(out_wav_path, mix)

    if stems_dir:
        os.makedirs(stems_dir, exist_ok=True)
        for name, st in (("felt_piano", piano_wet), ("low_strings", low_wet),
                         ("upper_strings_harp", hi_wet), ("brass", brass_wet), ("foley_env", foley_wet)):
            _write_wav_16bit(os.path.join(stems_dir, f"{name}.wav"), np.clip(st * gain_lin, -0.95, 0.95))

    return {
        "duration_sec": TOTAL_SAMPLES / SR,
        "samples": TOTAL_SAMPLES,
        "sample_rate": SR,
        "integrated_lufs_est": round(final_lufs, 2),
        "true_peak_db": round(peak_db, 2),
    }


def _write_wav_16bit(path: str, stereo_f32: np.ndarray) -> None:
    pcm = np.clip(np.round(stereo_f32 * 32767.0), -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())
