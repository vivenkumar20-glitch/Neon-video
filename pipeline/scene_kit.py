"""Shared painted environment primitives, architectural props, celestial bodies,
and multi-plane depth helpers used across all 11 shots.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import cairo
import cv2
import numpy as np

from .core_3d import Camera3D, over_premul, smoothstep, smootherstep, voronoi_masonry, warp_plane_premul
from .painter import InkCanvas2x, apply_gouache_finish, fbm_gouache, hex_to_linear, linear_to_srgb, srgb_to_linear


# ---------------------------------------------------------------------------
# Painterly Skies, Clouds, Rolling Hills & Atmospheric Gradients
# ---------------------------------------------------------------------------

def paint_sky_gradient(
    h: int,
    w: int,
    top_hex: str,
    mid_hex: str,
    horizon_hex: str,
    sun_xy_norm: Optional[Tuple[float, float]] = None,
    sun_glow_rgb: Tuple[float, float, float] = (0.95, 0.72, 0.44),
    sun_radius: float = 0.28,
    horizon_y: float = 0.68,
    seed: int = 10,
) -> np.ndarray:
    """Linear RGB gouache sky with smooth multi-stop atmospheric gradient and optional sun bloom."""
    rng = np.random.default_rng(seed)
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]

    c_top = np.array(hex_to_linear(top_hex), dtype=np.float32).reshape(1, 1, 3)
    c_mid = np.array(hex_to_linear(mid_hex), dtype=np.float32).reshape(1, 1, 3)
    c_hor = np.array(hex_to_linear(horizon_hex), dtype=np.float32).reshape(1, 1, 3)

    t = np.clip(y / max(0.05, horizon_y), 0.0, 1.0)
    t1 = np.clip(t * 2.0, 0.0, 1.0)[..., None]
    t2 = np.clip((t - 0.5) * 2.0, 0.0, 1.0)[..., None]
    sky = np.where(t[..., None] < 0.5, c_top * (1.0 - t1) + c_mid * t1, c_mid * (1.0 - t2) + c_hor * t2)
    sky = np.broadcast_to(sky, (h, w, 3)).copy()

    if sun_xy_norm is not None:
        sx, sy = sun_xy_norm
        dx = (x - sx) * (w / max(1.0, float(h)))
        dy = y - sy
        d = np.sqrt(dx * dx + dy * dy)
        glow = np.exp(-((d / max(0.02, sun_radius)) ** 1.65)).astype(np.float32)[..., None]
        sky = sky + glow * np.asarray(sun_glow_rgb, dtype=np.float32).reshape(1, 1, 3)

    return apply_gouache_finish(
        sky, rng, mottle_strength=0.042, n_dabs=84, dab_strength=0.075,
        paper_tooth=0.014, shadow_bounce_rgb=(0.018, 0.016, 0.022),
    )


def paint_clouds_layer(
    h: int,
    w: int,
    y_band: Tuple[float, float] = (0.08, 0.52),
    lit_srgb: Tuple[float, float, float] = (0.94, 0.88, 0.80),
    shadow_srgb: Tuple[float, float, float] = (0.48, 0.54, 0.64),
    density: float = 0.55,
    seed: int = 20,
) -> np.ndarray:
    """Hand-painted volumetric gouache cloud strata in linear premultiplied RGBA."""
    rng = np.random.default_rng(seed)
    n_base = fbm_gouache(h, w, rng, ((140, 88), (52, 38), (22, 18), (9, 8)), (0.48, 0.28, 0.16, 0.08))
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    band = np.clip((y - y_band[0]) / 0.12, 0.0, 1.0) * np.clip((y_band[1] - y) / 0.14, 0.0, 1.0)
    thresh = 1.05 - density * 1.45
    alpha = np.clip((n_base - thresh) * 0.65, 0.0, 0.88) * band

    # Top-lit cloud shading (light from above)
    grad_y = np.gradient(n_base, axis=0)
    lit_f = np.clip(0.55 - grad_y * 6.0, 0.0, 1.0)[..., None]
    c_lit = srgb_to_linear(np.array(lit_srgb, dtype=np.float32)).reshape(1, 1, 3)
    c_shd = srgb_to_linear(np.array(shadow_srgb, dtype=np.float32)).reshape(1, 1, 3)
    rgb = c_shd * (1.0 - lit_f) + c_lit * lit_f
    a = alpha[..., None].astype(np.float32)
    return np.concatenate([rgb * a, a], axis=-1).astype(np.float32)


def paint_rolling_hills(
    base_rgb: np.ndarray,
    layers: Sequence[Tuple[float, float, Tuple[float, float, float], float, int]],
) -> np.ndarray:
    """Paint layered atmospheric English countryside hills onto linear RGB image.

    Each tuple in layers: (base_y_norm, amp_norm, hex_or_linear_rgb, mist_blend, seed).
    """
    h, w = base_rgb.shape[:2]
    out = base_rgb.copy()
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]

    for y0, amp, col_lin, mist, seed in layers:
        rng = np.random.default_rng(seed)
        ph1, ph2, ph3 = rng.uniform(0.0, 6.28, 3)
        ridge = (
            y0
            + amp * 0.55 * np.sin(x * 3.8 + ph1)
            + amp * 0.30 * np.sin(x * 8.4 + ph2)
            + amp * 0.15 * np.sin(x * 16.5 + ph3)
        )
        mask = np.clip((yy - ridge[None, :]) * h * 0.35, 0.0, 1.0).astype(np.float32)
        mottle = fbm_gouache(h, w, rng, ((68, 64), (24, 22), (9, 9)), (0.5, 0.32, 0.18))
        # Subtle snow/sunlight slope shading
        slope = np.gradient(ridge)
        slope_shade = np.clip(1.0 - slope[None, :] * 14.0, 0.72, 1.24)
        depth_mist = np.clip(1.0 - (yy - ridge[None, :]) * 3.2, 0.0, 1.0) * mist

        c = np.asarray(col_lin, dtype=np.float32).reshape(1, 1, 3)
        hill_rgb = c * slope_shade[..., None] * (1.0 + mottle[..., None] * 0.08)
        # Mist lifts distant hill top toward sky horizon tone
        hor_col = np.mean(out[max(0, int(y0 * h) - 8):max(1, int(y0 * h)), :, :], axis=(0, 1)).reshape(1, 1, 3)
        hill_rgb = hill_rgb * (1.0 - depth_mist[..., None]) + hor_col * depth_mist[..., None]
        out = out * (1.0 - mask[..., None]) + hill_rgb * mask[..., None]
    return out


# ---------------------------------------------------------------------------
# Period Scientific & Architectural Props (Books, Candles, Leaded Windows)
# ---------------------------------------------------------------------------

def draw_painted_bookshelf(
    canvas: InkCanvas2x,
    x0: float,
    y0: float,
    width: float,
    height: float,
    n_shelves: int = 3,
    seed: int = 404,
    warm_tint: Tuple[float, float, float] = (1.0, 0.90, 0.76),
) -> None:
    """Individually painted 17th-century leather-bound books (varied heights, tilts, tooling, never identical)."""
    rng = np.random.default_rng(seed)
    ctx = canvas.ctx
    shelf_h = height / n_shelves
    ink = (0.09, 0.07, 0.06, 0.90)

    # Dark oak bookcase back & side uprights
    ctx.set_source_rgba(0.08 * warm_tint[0], 0.055 * warm_tint[1], 0.04 * warm_tint[2], 0.96)
    ctx.rectangle(x0, y0, width, height)
    ctx.fill()

    book_palettes = [
        (0.38, 0.16, 0.12), (0.28, 0.22, 0.14), (0.18, 0.24, 0.18),
        (0.45, 0.34, 0.21), (0.24, 0.17, 0.22), (0.56, 0.48, 0.34),
        (0.31, 0.14, 0.10), (0.20, 0.21, 0.24),
    ]

    for s in range(n_shelves):
        sy_bot = y0 + (s + 1) * shelf_h - shelf_h * 0.10
        cx = x0 + width * 0.03
        while cx < x0 + width * 0.93:
            bw = float(rng.uniform(width * 0.032, width * 0.072))
            if cx + bw > x0 + width * 0.96:
                break
            if rng.random() < 0.08:
                cx += bw * 0.75  # occasional gap
                continue
            bh = shelf_h * float(rng.uniform(0.56, 0.84))
            tilt = float(rng.uniform(-0.07, 0.07))
            base = book_palettes[int(rng.integers(0, len(book_palettes)))]
            var = float(rng.uniform(0.84, 1.18))
            b_lit = (min(0.95, base[0] * var * warm_tint[0]),
                     min(0.95, base[1] * var * warm_tint[1]),
                     min(0.95, base[2] * var * warm_tint[2]))
            b_shd = (b_lit[0] * 0.52, b_lit[1] * 0.52, b_lit[2] * 0.55)
            dx_top = tilt * bh
            pts = [
                (cx + dx_top, sy_bot - bh),
                (cx + bw + dx_top, sy_bot - bh),
                (cx + bw, sy_bot),
                (cx, sy_bot),
            ]
            canvas.fill_cel_region(pts, b_lit, b_shd, (-0.8, -0.4), outline_srgb=ink, outline_width=1.1,
                                   shadow_shift_px=bw * 0.38, seed=seed + int(cx) + s * 100)
            # Raised spine cords & faint gold leaf bands
            for frac in (0.18, 0.32, 0.76):
                ly = sy_bot - bh * frac
                lx0 = cx + dx_top * frac + 2.0
                lx1 = cx + bw + dx_top * frac - 2.0
                ctx.set_source_rgba(0.76 * warm_tint[0], 0.62 * warm_tint[1], 0.32 * warm_tint[2], 0.48)
                ctx.set_line_width(1.2)
                ctx.move_to(lx0, ly)
                ctx.line_to(lx1, ly)
                ctx.stroke()
            cx += bw + float(rng.uniform(0.5, 2.5))

        # Thick oak shelf plank
        plank = [
            (x0 - 4, sy_bot),
            (x0 + width + 4, sy_bot),
            (x0 + width + 4, sy_bot + shelf_h * 0.11),
            (x0 - 4, sy_bot + shelf_h * 0.11),
        ]
        canvas.fill_cel_region(
            plank,
            (0.24 * warm_tint[0], 0.16 * warm_tint[1], 0.10 * warm_tint[2]),
            (0.11 * warm_tint[0], 0.07 * warm_tint[1], 0.05 * warm_tint[2]),
            (-0.5, -0.8), outline_srgb=ink, outline_width=1.4, seed=seed + 50 + s,
        )


def draw_candle_and_flame(
    canvas: InkCanvas2x,
    base_x: float,
    base_y: float,
    candle_h: float,
    candle_w: float,
    frame_idx: int,
    seed: int = 17,
) -> Tuple[float, float]:
    """Draw tallow candle, brass holder, wax drips, and smoothly animated flame. Returns flame center."""
    t = frame_idx / 24.0
    sway = math.sin(t * 6.8 + seed) * candle_w * 0.14 + math.sin(t * 12.3 + seed * 2) * candle_w * 0.06
    stretch = 1.0 + 0.08 * math.sin(t * 9.4 + seed)
    ink = (0.12, 0.09, 0.07, 0.90)

    # Brass candlestick saucer & stem
    dish = [
        (base_x - candle_w * 1.45, base_y),
        (base_x + candle_w * 1.45, base_y),
        (base_x + candle_w * 1.15, base_y + candle_w * 0.36),
        (base_x - candle_w * 1.15, base_y + candle_w * 0.36),
    ]
    canvas.fill_cel_region(dish, (0.64, 0.48, 0.24), (0.32, 0.22, 0.11), (-0.6, -0.7),
                           outline_srgb=ink, outline_width=1.3, seed=seed)

    # Melted wax column
    wax = [
        (base_x - candle_w * 0.48, base_y),
        (base_x - candle_w * 0.50, base_y - candle_h),
        (base_x - candle_w * 0.18, base_y - candle_h + candle_w * 0.14),
        (base_x + candle_w * 0.20, base_y - candle_h + candle_w * 0.10),
        (base_x + candle_w * 0.48, base_y - candle_h + candle_w * 0.04),
        (base_x + candle_w * 0.50, base_y),
    ]
    canvas.fill_cel_region(wax, (0.93, 0.85, 0.68), (0.62, 0.48, 0.34), (-0.7, -0.5),
                           outline_srgb=ink, outline_width=1.2, seed=seed + 1)

    # Wick
    fx = base_x + sway * 0.2
    fy = base_y - candle_h - candle_w * 0.22
    canvas.stroke_catmull_rom([(base_x, base_y - candle_h + 2), (fx, fy + candle_w * 0.15)],
                              base_width=max(1.2, candle_w * 0.08), color_srgb=(0.10, 0.08, 0.07, 0.95), seed=seed + 2)

    # Layered translucent flame (outer amber -> core warm white)
    fh = candle_w * 1.45 * stretch
    fw = candle_w * 0.42
    outer = [
        (fx, fy + fh * 0.18),
        (fx - fw, fy - fh * 0.18),
        (fx - fw * 0.42 + sway * 0.6, fy - fh * 0.72),
        (fx + sway, fy - fh),
        (fx + fw * 0.48 + sway * 0.6, fy - fh * 0.70),
        (fx + fw, fy - fh * 0.18),
    ]
    canvas.fill_smooth_poly(outer, (1.0, 0.72, 0.24, 0.92))
    inner = [
        (fx, fy + fh * 0.06),
        (fx - fw * 0.52, fy - fh * 0.18),
        (fx + sway * 0.5, fy - fh * 0.64),
        (fx + fw * 0.52, fy - fh * 0.18),
    ]
    canvas.fill_smooth_poly(inner, (1.0, 0.97, 0.84, 0.98))
    return fx, fy - fh * 0.42


def add_radial_light_glow(
    linear_rgb: np.ndarray,
    cx: float,
    cy: float,
    radius_px: float,
    color_linear: Tuple[float, float, float],
    strength: float = 0.65,
    power: float = 1.9,
) -> np.ndarray:
    """Add smooth physical inverse-power volumetric glow in linear RGB space."""
    h, w = linear_rgb.shape[:2]
    # Compute in bounded ROI for speed
    r_pad = int(math.ceil(radius_px * 2.4))
    x0, x1 = max(0, int(cx) - r_pad), min(w, int(cx) + r_pad)
    y0, y1 = max(0, int(cy) - r_pad), min(h, int(cy) + r_pad)
    if x1 <= x0 or y1 <= y0:
        return linear_rgb
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(1.0, float(radius_px))
    falloff = np.exp(-(d ** power)) * (1.0 / (1.0 + d * d * 0.55))
    col = np.asarray(color_linear, dtype=np.float32).reshape(1, 1, 3) * float(strength)
    out = linear_rgb.copy()
    out[y0:y1, x0:x1] += falloff[..., None] * col
    return out


def draw_leaded_window_frame(
    canvas: InkCanvas2x,
    x0: float,
    y0: float,
    ww: float,
    wh: float,
    cols: int = 4,
    rows: int = 5,
    frame_srgb: Tuple[float, float, float] = (0.14, 0.11, 0.09),
) -> None:
    """17th-century English casement window with diamond/rectangular lead cames & timber mullions."""
    ctx = canvas.ctx
    # Outer heavy oak frame
    t_out = max(6.0, ww * 0.045)
    ctx.set_source_rgba(*frame_srgb, 0.98)
    ctx.rectangle(x0, y0, ww, t_out)
    ctx.rectangle(x0, y0 + wh - t_out, ww, t_out)
    ctx.rectangle(x0, y0, t_out, wh)
    ctx.rectangle(x0 + ww - t_out, y0, t_out, wh)
    # Central vertical mullion & horizontal transom
    ctx.rectangle(x0 + ww * 0.5 - t_out * 0.45, y0, t_out * 0.9, wh)
    ctx.rectangle(x0, y0 + wh * 0.48 - t_out * 0.35, ww, t_out * 0.7)
    ctx.fill()

    # Fine dark lead cames
    ctx.set_source_rgba(frame_srgb[0] * 0.7, frame_srgb[1] * 0.7, frame_srgb[2] * 0.7, 0.82)
    ctx.set_line_width(max(1.8, ww * 0.008))
    for c in range(1, cols):
        lx = x0 + (ww * c) / cols
        ctx.move_to(lx, y0)
        ctx.line_to(lx, y0 + wh)
    for r in range(1, rows):
        ly = y0 + (wh * r) / rows
        ctx.move_to(x0, ly)
        ctx.line_to(x0 + ww, ly)
    ctx.stroke()


# ---------------------------------------------------------------------------
# Consistent Hand-Painted Near-Side Moon & Deep Astronomical Starfield
# ---------------------------------------------------------------------------

def paint_starfield_cosmos(h: int, w: int, seed: int = 808) -> np.ndarray:
    """Deep astronomical indigo painted cosmos with subtle Milky Way gouache nebula & crisp stars."""
    rng = np.random.default_rng(seed)
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]

    base = np.array([0.009, 0.013, 0.032], dtype=np.float32).reshape(1, 1, 3)
    deep = np.array([0.003, 0.005, 0.014], dtype=np.float32).reshape(1, 1, 3)
    sky = np.broadcast_to(deep * (1.0 - y[..., None] * 0.5) + base * (0.5 + 0.5 * y[..., None]), (h, w, 3)).copy()

    # Diagonal Milky Way painterly dust band
    diag = np.exp(-(((y - (0.78 - 0.55 * x)) / 0.22) ** 2)).astype(np.float32)
    neb = np.clip(fbm_gouache(h, w, rng, ((96, 92), (34, 36), (14, 14)), (0.5, 0.32, 0.18)) * 0.5 + 0.45, 0.0, 1.0)
    sky += (diag * neb)[..., None] * np.array([0.018, 0.024, 0.048], dtype=np.float32).reshape(1, 1, 3)

    # Crisp seeded stars of varied magnitude and Planckian tints (blue-white, warm gold, ivory)
    n_stars = int(950 * (w / 2048.0) ** 0.7 + 250)
    star_tints = [
        (0.85, 0.90, 1.00), (1.00, 0.96, 0.88), (1.00, 0.84, 0.62), (0.74, 0.84, 1.00)
    ]
    for _ in range(n_stars):
        sx = int(rng.integers(2, w - 2))
        sy = int(rng.integers(2, h - 2))
        mag = float(rng.exponential(0.28))
        br = min(1.8, 0.14 + mag * 0.85)
        tint = np.array(star_tints[int(rng.integers(0, len(star_tints)))], dtype=np.float32)
        if mag > 0.75:
            # Bright star with tiny 3px soft core
            r = max(1, int(round(2.2 * (w / 2048.0))))
            yy, xx = np.mgrid[max(0, sy - r):min(h, sy + r + 1), max(0, sx - r):min(w, sx + r + 1)]
            g = np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / max(0.6, (r * 0.65) ** 2)).astype(np.float32)
            sky[max(0, sy - r):min(h, sy + r + 1), max(0, sx - r):min(w, sx + r + 1)] += g[..., None] * tint * br
        else:
            sky[sy, sx] += tint * br

    return apply_gouache_finish(sky, rng, mottle_strength=0.03, n_dabs=60, dab_strength=0.05,
                                paper_tooth=0.008, shadow_bounce_rgb=(0.004, 0.006, 0.012))


def paint_consistent_moon(
    dst_linear: np.ndarray,
    cx: float,
    cy: float,
    radius_px: float,
    sun_dir_2d: Tuple[float, float] = (-0.78, -0.35),
    alpha_scale: float = 1.0,
    daytime_blend: bool = False,
) -> np.ndarray:
    """Paint the Moon with consistent historical near-side Maria (Imbrium, Serenitatis, Tranquillitatis,
    Crisium, Tycho) so its face stays recognizable across Shots 5, 8, and 11.
    """
    h, w = dst_linear.shape[:2]
    r = max(4.0, float(radius_px))
    pad = int(math.ceil(r * 1.15))
    x0, x1 = max(0, int(cx) - pad), min(w, int(cx) + pad)
    y0, y1 = max(0, int(cy) - pad), min(h, int(cy) + pad)
    if x1 <= x0 or y1 <= y0:
        return dst_linear

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    nx = (xx - cx) / r
    ny = (yy - cy) / r
    rr2 = nx * nx + ny * ny
    inside = np.clip((1.0 - np.sqrt(np.maximum(0.0, rr2))) * r * 0.8, 0.0, 1.0)
    nz = np.sqrt(np.maximum(0.0, 1.0 - np.minimum(1.0, rr2)))

    # Consistent near-side lunar maria coordinates (normalized disc coords)
    maria = [
        (-0.24, -0.28, 0.28, 0.24, 0.34),  # Mare Imbrium
        (0.16, -0.22, 0.20, 0.19, 0.30),   # Mare Serenitatis
        (0.32, 0.05, 0.24, 0.19, 0.32),    # Mare Tranquillitatis
        (0.56, -0.10, 0.11, 0.10, 0.36),   # Mare Crisium
        (-0.42, 0.08, 0.24, 0.28, 0.26),   # Oceanus Procellarum
        (0.28, 0.30, 0.16, 0.13, 0.24),    # Mare Fecunditatis
    ]
    albedo = np.full_like(nx, 0.78, dtype=np.float32)
    for mx, my, rx, ry, dark in maria:
        d2 = ((nx - mx) / rx) ** 2 + ((ny - my) / ry) ** 2
        albedo -= np.exp(-d2 * 1.65) * dark

    # Crater texture & bright Tycho southern ray highlight
    rng = np.random.default_rng(1668)
    cr_noise = fbm_gouache(y1 - y0, x1 - x0, rng, ((18, 18), (7, 7)), (0.6, 0.4))
    albedo = np.clip(albedo + cr_noise * 0.055, 0.24, 0.94)
    tycho = np.exp(-(((nx + 0.08) ** 2 + (ny - 0.56) ** 2) / 0.012)) * 0.22
    albedo = np.clip(albedo + tycho, 0.24, 0.98)

    # Lambertian + gentle Lommel-Seeliger lunar phase lighting
    lx, ly = sun_dir_2d
    lz = math.sqrt(max(0.05, 1.0 - min(0.95, lx * lx + ly * ly)))
    lvec = np.array([-lx, -ly, lz], dtype=np.float32)
    lvec /= np.linalg.norm(lvec)
    ndotl = np.clip(nx * lvec[0] + ny * lvec[1] + nz * lvec[2], 0.0, 1.0)
    lit = (ndotl ** 0.72) * 0.92 + (0.04 if not daytime_blend else 0.0)

    moon_col = np.array([0.92, 0.89, 0.82], dtype=np.float32).reshape(1, 1, 3)
    moon_rgb = albedo[..., None] * lit[..., None] * moon_col

    out = dst_linear.copy()
    if daytime_blend:
        # Daytime pale Moon adds gently over blue sky (unlit side vanishes into sky)
        a = (inside * np.clip(ndotl * 1.4, 0.0, 1.0) * alpha_scale)[..., None]
        out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1.0 - a * 0.35) + moon_rgb * a * 0.88
    else:
        a = (inside * alpha_scale)[..., None]
        out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1.0 - a) + moon_rgb * a
    return out


# ---------------------------------------------------------------------------
# Subtle Camera Drift / Pan Crop Utility for Oversampled Painted Backdrops
# ---------------------------------------------------------------------------

def sample_overscan(bg_overscan: np.ndarray, out_w: int, out_h: int, dx_frac: float, dy_frac: float, zoom: float = 1.0) -> np.ndarray:
    """Sample (out_h, out_w) viewport from an overscanned painting with smooth subpixel camera drift."""
    oh, ow = bg_overscan.shape[:2]
    crop_w = ow / max(1.0, float(zoom))
    crop_h = oh / max(1.0, float(zoom))
    cx = ow * 0.5 + dx_frac * (ow - crop_w) * 0.5
    cy = oh * 0.5 + dy_frac * (oh - crop_h) * 0.5
    x0 = max(0.0, min(ow - crop_w, cx - crop_w * 0.5))
    y0 = max(0.0, min(oh - crop_h, cy - crop_h * 0.5))
    # Affine transform is ~4x faster than warpPerspective with identical subpixel linear sampling
    sx = out_w / max(1.0, crop_w)
    sy = out_h / max(1.0, crop_h)
    M = np.array([[sx, 0.0, -x0 * sx], [0.0, sy, -y0 * sy]], dtype=np.float32)
    return cv2.warpAffine(bg_overscan, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
