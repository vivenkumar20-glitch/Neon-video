"""Stage 4: Foreground circular disc bokeh, 35mm optical finishing, halation,
chromatic aberration, C1-continuous tanh shoulder, silver-halide grain, and subtle analog patina.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .painter import linear_lum, linear_to_srgb


# ---------------------------------------------------------------------------
# Circular Aperture Disc Bokeh (with subtle spherical-aberration rim)
# ---------------------------------------------------------------------------

def make_disc_kernel(radius: float = 7.8, rim_boost: float = 0.24) -> np.ndarray:
    """Normalized circular aperture disc kernel with subtle spherical-aberration rim."""
    r = max(1.0, float(radius))
    half = int(math.ceil(r + 1.5))
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
    d = np.sqrt(x * x + y * y)
    # Soft antialias edge
    disc = np.clip(r + 0.6 - d, 0.0, 1.0)
    # Subtle bright ring near r * 0.85 for vintage cinema lens character
    ring = np.exp(-((d - r * 0.84) / max(0.65, r * 0.14)) ** 2) * rim_boost * disc
    kernel = (disc + ring).astype(np.float32)
    s = float(np.sum(kernel))
    return kernel / max(1e-8, s)


def apply_disc_bokeh_premul(premul_rgba: np.ndarray, radius: float = 7.8) -> np.ndarray:
    """Convolve foreground premultiplied RGBA silhouettes with a circular aperture disc kernel.

    Scales radius proportionally when rendering low-res animatics.
    """
    h, w = premul_rgba.shape[:2]
    scaled_r = radius * (w / 2048.0)
    if scaled_r < 0.8:
        return premul_rgba
    k = make_disc_kernel(scaled_r)
    # Fast separable/FFT convolution via OpenCV filter2D (or downscale-convolve for large radii)
    if k.shape[0] > 19:
        sh, sw = max(16, h // 2), max(16, w // 2)
        small = cv2.resize(premul_rgba, (sw, sh), interpolation=cv2.INTER_AREA)
        k2 = make_disc_kernel(scaled_r * 0.5)
        blurred = cv2.filter2D(small, cv2.CV_32F, k2, borderType=cv2.BORDER_CONSTANT)
        return np.clip(cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR), 0.0, 1.5)
    out = cv2.filter2D(premul_rgba.astype(np.float32), cv2.CV_32F, k, borderType=cv2.BORDER_CONSTANT)
    return np.clip(out, 0.0, 1.5)


# ---------------------------------------------------------------------------
# 35mm Optical Finishing & Analog Transfer Patina
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OpticsProfile:
    halation_strength: float = 0.22
    halation_tint: Tuple[float, float, float] = (1.05, 0.82, 0.64)
    ca_px: float = 0.85          # radial corner chromatic aberration at 2048w (0.0 for text/diagrams)
    vignette: float = 0.24
    black_lift: float = 0.011
    knee: float = 0.80
    ceiling: float = 0.965
    grain_strength: float = 0.018
    yiq_chroma_soft_px: float = 0.85
    scanline_mod: float = 0.004


_RADIAL_CACHE: dict = {}
_GRAIN_BANK: dict = {}


def _get_radial_maps(h: int, w: int, ca_px: float, vignette: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    key = (h, w, round(ca_px, 3), round(vignette, 3))
    if key in _RADIAL_CACHE:
        return _RADIAL_CACHE[key]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    nx = (xx - cx) / max(1.0, cx)
    ny = (yy - cy) / max(1.0, cy)
    r2 = nx * nx + ny * ny
    scale = ca_px * (w / 2048.0)
    dx = nx * r2 * scale
    dy = ny * r2 * scale
    map_r_x = (xx + dx).astype(np.float32)
    map_r_y = (yy + dy).astype(np.float32)
    map_b_x = (xx - dx * 0.85).astype(np.float32)
    map_b_y = (yy - dy * 0.85).astype(np.float32)
    vig = np.clip(1.0 - vignette * (r2 ** 1.15) * 0.52, 0.58, 1.0).astype(np.float32)[..., None]
    _RADIAL_CACHE[key] = (map_r_x, map_r_y, map_b_x, map_b_y, vig)
    return _RADIAL_CACHE[key]


def _get_grain_field(h: int, w: int, frame_idx: int) -> np.ndarray:
    """Fast zero-copy deterministic silver-halide grain field from oversized multi-tile bank."""
    key = (h, w)
    pad = 256
    if key not in _GRAIN_BANK:
        rng = np.random.default_rng(900101)
        gh, gw = max(16, int(round((h + pad) / 1.4))), max(16, int(round((w + pad) / 1.4)))
        bank = []
        for _ in range(6):
            g = rng.standard_normal((gh, gw)).astype(np.float32)
            bank.append(cv2.resize(g, (w + pad, h + pad), interpolation=cv2.INTER_LINEAR))
        _GRAIN_BANK[key] = bank
    bank = _GRAIN_BANK[key]
    g = bank[frame_idx % len(bank)]
    sy = (frame_idx * 37) % pad
    sx = (frame_idx * 71) % pad
    out = g[sy:sy + h, sx:sx + w]
    if (frame_idx // len(bank)) % 2 == 1:
        out = out[:, ::-1]
    return out


def _c1_tanh_shoulder(x: np.ndarray, knee: float = 0.80, ceiling: float = 0.965) -> np.ndarray:
    """C1-continuous highlight roll-off: linear below knee, smooth tanh approaching ceiling above knee."""
    span = max(1e-4, ceiling - knee)
    mask = x > knee
    if not np.any(mask):
        return x
    out = x.copy()
    out[mask] = knee + span * np.tanh((x[mask] - knee) / span)
    return out


def finish_35mm(
    linear_rgb: np.ndarray,
    frame_idx: int,
    profile: OpticsProfile = OpticsProfile(),
    glints_linear: Optional[np.ndarray] = None,
    exposure_mult: float = 1.0,
) -> np.ndarray:
    """Apply Stage 4 35mm theatrical optics in float32 linear RGB and convert once to uint8 RGB."""
    h, w = linear_rgb.shape[:2]
    res_scale = w / 2048.0
    if exposure_mult != 1.0:
        img = np.maximum(0.0, linear_rgb.astype(np.float32) * float(exposure_mult))
    else:
        img = np.maximum(0.0, linear_rgb.astype(np.float32))

    # 1. Multi-scale highlight halation (sigma = 6.5, 24, 72 px computed cleanly on 1/2, 1/4, 1/8 pyramid)
    if profile.halation_strength > 0.0:
        hw, hh = max(8, w // 2), max(8, h // 2)
        half_img = cv2.resize(img, (hw, hh), interpolation=cv2.INTER_AREA)
        lum_h = 0.2126 * half_img[..., 0] + 0.7152 * half_img[..., 1] + 0.0722 * half_img[..., 2]
        hi_h = half_img * np.clip((lum_h - 0.72) * (1.0 / 0.35), 0.0, 2.5)[..., None]
        if glints_linear is not None:
            hi_h = hi_h + cv2.resize(np.maximum(0.0, glints_linear), (hw, hh), interpolation=cv2.INTER_AREA)

        s1 = max(1.0, 6.5 * res_scale * 0.5)
        s2 = max(1.2, 24.0 * res_scale * 0.25)
        s3 = max(1.5, 72.0 * res_scale * 0.125)

        b1_h = cv2.GaussianBlur(hi_h, (0, 0), sigmaX=s1, borderType=cv2.BORDER_REFLECT_101)
        qtr = cv2.resize(hi_h, (max(8, hw // 2), max(8, hh // 2)), interpolation=cv2.INTER_AREA)
        b2_q = cv2.GaussianBlur(qtr, (0, 0), sigmaX=s2, borderType=cv2.BORDER_REFLECT_101)
        octv = cv2.resize(qtr, (max(8, hw // 4), max(8, hh // 4)), interpolation=cv2.INTER_AREA)
        b3_o = cv2.GaussianBlur(octv, (0, 0), sigmaX=s3, borderType=cv2.BORDER_REFLECT_101)

        halo_h = (
            0.48 * b1_h
            + 0.34 * cv2.resize(b2_q, (hw, hh), interpolation=cv2.INTER_LINEAR)
            + 0.18 * cv2.resize(b3_o, (hw, hh), interpolation=cv2.INTER_LINEAR)
        )
        tint = np.asarray(profile.halation_tint, dtype=np.float32).reshape(1, 1, 3)
        halo = cv2.resize(halo_h * (tint * profile.halation_strength), (w, h), interpolation=cv2.INTER_LINEAR)
        img = img + halo * np.clip(1.25 - img * 0.55, 0.15, 1.0)

    # 2. Radial corner chromatic aberration + gentle optical vignette
    map_rx, map_ry, map_bx, map_by, vig = _get_radial_maps(h, w, profile.ca_px, profile.vignette)
    if profile.ca_px > 0.05:
        r_ch = cv2.remap(img[..., 0], map_rx, map_ry, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        b_ch = cv2.remap(img[..., 2], map_bx, map_by, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        img[..., 0] = r_ch
        img[..., 2] = b_ch

    if profile.vignette > 0.0:
        img *= vig

    # 3. Linear shadow black lift + C1-continuous tanh highlight shoulder -> single LUT sRGB display transfer
    img = img * (1.0 - profile.black_lift) + profile.black_lift
    img = _c1_tanh_shoulder(img, knee=profile.knee, ceiling=profile.ceiling)
    srgb = linear_to_srgb(img)

    # 4. Luminance-weighted silver-halide grain
    if profile.grain_strength > 0.0:
        g_luma = _get_grain_field(h, w, frame_idx)
        lum_s = 0.299 * srgb[..., 0] + 0.587 * srgb[..., 1] + 0.114 * srgb[..., 2]
        env = np.clip(1.12 - 0.95 * (lum_s * np.sqrt(np.maximum(lum_s, 0.0))), 0.12, 1.12)
        srgb = srgb + (g_luma * env * profile.grain_strength)[..., None]

    # 5. Restrained analog transfer patina: horizontal YIQ chroma softening (~0.85 px) + micro scanline
    if profile.yiq_chroma_soft_px > 0.2:
        r, g, b = srgb[..., 0], srgb[..., 1], srgb[..., 2]
        y_ch = 0.299 * r + 0.587 * g + 0.114 * b
        i_ch = 0.596 * r - 0.274 * g - 0.322 * b
        q_ch = 0.211 * r - 0.523 * g + 0.312 * b
        # Fast 3-tap horizontal smoothing matching ~0.85px sigma
        i_s = i_ch.copy()
        i_s[:, 1:-1] = 0.23 * i_ch[:, :-2] + 0.54 * i_ch[:, 1:-1] + 0.23 * i_ch[:, 2:]
        q_s = q_ch.copy()
        q_s[:, 1:-1] = 0.23 * q_ch[:, :-2] + 0.54 * q_ch[:, 1:-1] + 0.23 * q_ch[:, 2:]
        if profile.scanline_mod > 0.0:
            y_ch[0::2, :] *= (1.0 - profile.scanline_mod * 0.5)
            y_ch[1::2, :] *= (1.0 + profile.scanline_mod * 0.5)
        srgb[..., 0] = y_ch + 0.956 * i_s + 0.621 * q_s
        srgb[..., 1] = y_ch - 0.272 * i_s - 0.647 * q_s
        srgb[..., 2] = y_ch - 1.106 * i_s + 1.703 * q_s

    return np.clip(srgb * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
