"""Stage 3: Painterly gouache surface engine and 2x supersampled PyCairo ink canvas.

All lighting and compositing occurs in declared linear RGB space; sRGB conversion happens
once at final display encoding in Stage 4 to prevent double gamma correction.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import cairo
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Color Space Conversions (declared linear <-> sRGB)
# ---------------------------------------------------------------------------

_SRGB_TO_LIN_LUT = np.where(
    np.linspace(0.0, 1.0, 4096, dtype=np.float32) <= 0.04045,
    np.linspace(0.0, 1.0, 4096, dtype=np.float32) / 12.92,
    ((np.linspace(0.0, 1.0, 4096, dtype=np.float32) + 0.055) / 1.055) ** 2.4,
).astype(np.float32)

_LIN_TO_SRGB_LUT = np.where(
    np.linspace(0.0, 1.0, 4096, dtype=np.float32) <= 0.0031308,
    np.linspace(0.0, 1.0, 4096, dtype=np.float32) * 12.92,
    1.055 * (np.linspace(0.0, 1.0, 4096, dtype=np.float32) ** (1.0 / 2.4)) - 0.055,
).astype(np.float32)


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    arr = np.asarray(c, dtype=np.float32)
    idx = np.clip((arr * 4095.0).astype(np.int32), 0, 4095)
    return _SRGB_TO_LIN_LUT[idx]


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    arr = np.asarray(c, dtype=np.float32)
    idx = np.clip((arr * 4095.0).astype(np.int32), 0, 4095)
    return _LIN_TO_SRGB_LUT[idx]


def hex_to_linear(hex_str: str) -> Tuple[float, float, float]:
    h = hex_str.lstrip("#")
    rgb = np.array([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)], dtype=np.float32) / 255.0
    lin = srgb_to_linear(rgb)
    return float(lin[0]), float(lin[1]), float(lin[2])


def linear_lum(rgb: np.ndarray) -> np.ndarray:
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)


# ---------------------------------------------------------------------------
# Isotropic Gouache Mottling, Flat-Brush Dabs, Paper Tooth & Shadow Bounce
# ---------------------------------------------------------------------------

def fbm_gouache(
    h: int,
    w: int,
    rng: np.random.Generator,
    scales: Sequence[Tuple[float, float]] = ((72.0, 68.0), (26.0, 29.0), (10.5, 9.5), (4.5, 4.8)),
    weights: Sequence[float] = (0.44, 0.28, 0.18, 0.10),
) -> np.ndarray:
    """Multi-octave bicubic-upsampled zero-mean unit-variance Gaussian mottling at balanced square scales.

    Avoids high-aspect-ratio row/column noise so 1:1 macro crops never show synthetic striping.
    """
    acc = np.zeros((h, w), dtype=np.float32)
    ref_scale = max(0.35, min(1.5, w / 2048.0))
    for (sx, sy), wt in zip(scales, weights):
        gh = max(3, int(round(h / max(2.0, sy * ref_scale))))
        gw = max(3, int(round(w / max(2.0, sx * ref_scale))))
        grid = rng.standard_normal((gh, gw)).astype(np.float32)
        up = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        std = float(np.std(up)) + 1e-6
        acc += (up - float(np.mean(up))) / std * float(wt)
    acc /= (float(np.std(acc)) + 1e-6)
    return acc


def apply_gouache_finish(
    linear_rgb: np.ndarray,
    rng: np.random.Generator,
    mottle_strength: float = 0.055,
    n_dabs: int = 92,
    dab_strength: float = 0.085,
    paper_tooth: float = 0.018,
    shadow_bounce_rgb: Tuple[float, float, float] = (0.042, 0.028, 0.018),
    region_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Enrich painted background fills with isotropic fbm gouache mottling, 80-100 angled
    soft convex-polygon flat-brush dabs at 1/4 resolution, cold-press paper tooth,
    pigment pooling, and warm bounce light inside deep shadows (lum < 0.28).
    """
    h, w = linear_rgb.shape[:2]
    out = linear_rgb.astype(np.float32).copy()

    # 1. Isotropic multi-octave gouache mottling
    mottle = fbm_gouache(h, w, rng)
    # Subtle warm/cool pigment shift between channels
    mottle_c = np.stack([
        mottle * (mottle_strength * 1.05),
        mottle * (mottle_strength * 0.96),
        -mottle * (mottle_strength * 0.45) + fbm_gouache(h, w, rng, ((54, 52), (18, 19)), (0.65, 0.35)) * (mottle_strength * 0.65),
    ], axis=-1)
    out = out * (1.0 + mottle_c)

    # 2. 80-100 randomly angled soft convex-polygon flat-brush dabs at 1/4 resolution
    qh, qw = max(16, h // 4), max(16, w // 4)
    dab_accum = np.zeros((qh, qw, 3), dtype=np.float32)
    dab_weight = np.zeros((qh, qw, 1), dtype=np.float32)
    small_src = cv2.resize(out, (qw, qh), interpolation=cv2.INTER_AREA)

    for _ in range(n_dabs):
        cx = float(rng.uniform(0.0, qw))
        cy = float(rng.uniform(0.0, qh))
        angle = float(rng.uniform(0.0, math.pi))
        rx = float(rng.uniform(qw * 0.035, qw * 0.11))
        ry = float(rng.uniform(qh * 0.018, qh * 0.055))
        n_verts = int(rng.integers(5, 8))
        thetas = np.linspace(0.0, 2.0 * math.pi, n_verts, endpoint=False) + rng.uniform(-0.18, 0.18, n_verts)
        radii_j = rng.uniform(0.82, 1.16, n_verts)
        ca, sa = math.cos(angle), math.sin(angle)
        lx = np.cos(thetas) * rx * radii_j
        ly = np.sin(thetas) * ry * radii_j
        px = np.round(cx + lx * ca - ly * sa).astype(np.int32)
        py = np.round(cy + lx * sa + ly * ca).astype(np.int32)
        poly = np.stack([px, py], axis=-1)

        sx = int(np.clip(round(cx), 0, qw - 1))
        sy = int(np.clip(round(cy), 0, qh - 1))
        col = small_src[sy, sx] * (1.0 + rng.normal(0.0, 0.09, 3).astype(np.float32))
        col = np.clip(col, 0.005, 1.2)

        patch = np.zeros((qh, qw), dtype=np.float32)
        cv2.fillConvexPoly(patch, poly, 1.0, lineType=cv2.LINE_AA)
        dab_accum += patch[..., None] * col.reshape(1, 1, 3)
        dab_weight += patch[..., None]

    # Soften brush dab edges & add subtle pigment edge pooling
    dab_accum = cv2.GaussianBlur(dab_accum, (0, 0), sigmaX=2.2, borderType=cv2.BORDER_REFLECT_101)
    dab_weight = cv2.GaussianBlur(dab_weight, (0, 0), sigmaX=2.2, borderType=cv2.BORDER_REFLECT_101)
    if dab_weight.ndim == 2:
        dab_weight = dab_weight[..., None]
    dab_rgb_q = dab_accum / np.maximum(1e-4, dab_weight)
    dab_alpha_q = np.clip(dab_weight * dab_strength, 0.0, 0.35)

    dab_rgb = cv2.resize(dab_rgb_q, (w, h), interpolation=cv2.INTER_CUBIC)
    dab_alpha = cv2.resize(dab_alpha_q, (w, h), interpolation=cv2.INTER_CUBIC)[..., None]
    # Pigment pooling slightly darkens the rim of dab weight transitions
    pool_edge = np.abs(dab_alpha - cv2.GaussianBlur(dab_alpha, (0, 0), sigmaX=4.0)[..., None])
    out = out * (1.0 - dab_alpha) + dab_rgb * dab_alpha - pool_edge * 0.12

    # 3. Fine isotropic cold-press paper tooth
    if paper_tooth > 0.0:
        gh = max(16, int(round(h / 2.2)))
        gw = max(16, int(round(w / 2.2)))
        tooth = rng.standard_normal((gh, gw)).astype(np.float32)
        tooth = cv2.resize(tooth, (w, h), interpolation=cv2.INTER_CUBIC)
        tooth = (tooth - float(np.mean(tooth))) / (float(np.std(tooth)) + 1e-6)
        out *= (1.0 + tooth[..., None] * paper_tooth)

    # 4. Warm bounce light inside deep shadows (lum < 0.28)
    lum = linear_lum(out)
    shadow_w = np.clip((0.28 - lum) / 0.28, 0.0, 1.0) ** 1.6
    bounce = np.asarray(shadow_bounce_rgb, dtype=np.float32).reshape(1, 1, 3)
    out = out + shadow_w[..., None] * bounce

    out = np.clip(out, 0.0, 2.5)
    if region_mask is not None:
        m = np.clip(region_mask.astype(np.float32), 0.0, 1.0)
        if m.ndim == 2:
            m = m[..., None]
        out = linear_rgb * (1.0 - m) + out * m
    return out


# ---------------------------------------------------------------------------
# 2x Supersampled PyCairo Ink & Cel Canvas
# ---------------------------------------------------------------------------

def _catmull_rom_samples(pts: Sequence[Tuple[float, float]], samples_per_seg: int = 8) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64)
    if len(p) < 2:
        return p
    if len(p) == 2:
        t = np.linspace(0.0, 1.0, samples_per_seg + 1)[:, None]
        return p[0] * (1.0 - t) + p[1] * t
    ext = np.vstack([p[0] + (p[0] - p[1]), p, p[-1] + (p[-1] - p[-2])])
    out = []
    for i in range(len(p) - 1):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        ts = np.linspace(0.0, 1.0, samples_per_seg, endpoint=(i == len(p) - 2))
        t2 = ts * ts
        t3 = t2 * ts
        seg = 0.5 * (
            (2.0 * p1)[None, :]
            + (-p0 + p2)[None, :] * ts[:, None]
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3)[None, :] * t2[:, None]
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3)[None, :] * t3[:, None]
        )
        out.append(seg)
    return np.vstack(out)


class InkCanvas2x:
    """2x supersampled Cairo ARGB32 drawing surface for hand-inked contours and cel forms.

    Colors passed to drawing methods are in sRGB [0..1] (converted properly to linear
    premultiplied RGBA upon Lanczos downsampling to target resolution).
    """

    def __init__(self, width: int, height: int, supersample: int = 2) -> None:
        self.w = int(width)
        self.h = int(height)
        self.ss = int(supersample)
        self.surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.w * self.ss, self.h * self.ss)
        self.ctx = cairo.Context(self.surf)
        self.ctx.scale(self.ss, self.ss)
        self.ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        self.ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        self.ctx.set_antialias(cairo.ANTIALIAS_BEST)

    def set_color_linear(self, r: float, g: float, b: float, a: float = 1.0) -> None:
        """Set Cairo color from linear RGB by encoding to sRGB for 8-bit surface storage."""
        sr, sg, sb = linear_to_srgb(np.array([r, g, b], dtype=np.float32))
        self.ctx.set_source_rgba(float(sr), float(sg), float(sb), float(a))

    def set_color_srgb(self, r: float, g: float, b: float, a: float = 1.0) -> None:
        self.ctx.set_source_rgba(float(r), float(g), float(b), float(a))

    def stroke_catmull_rom(
        self,
        pts: Sequence[Tuple[float, float]],
        base_width: float = 2.0,
        color_srgb: Tuple[float, float, float, float] = (0.11, 0.09, 0.08, 0.94),
        taper: bool = True,
        tremor: float = 0.28,
        seed: int = 1,
        z_depth: Optional[float] = None,
        fade_dist: float = 14.0,
        closed: bool = False,
    ) -> None:
        """Stroke a tapered Catmull-Rom spline with subtle seeded hand tremor and perspective width decay."""
        if len(pts) < 2:
            return
        pts_in = list(pts)
        if closed:
            pts_in = pts_in + [pts_in[0]]
        samples = _catmull_rom_samples(pts_in, samples_per_seg=6)
        n = len(samples)
        if n < 2:
            return

        # Perspective width decay: clip(1 / (1 + z / (fade_dist * 0.45)), 0.25, 1.0)
        scale_w = 1.0
        if z_depth is not None:
            scale_w = float(np.clip(1.0 / (1.0 + max(0.0, z_depth) / (fade_dist * 0.45)), 0.25, 1.0))
        eff_w = base_width * scale_w

        # Subtle deterministic hand tremor along normal
        if tremor > 0.0 and n > 3:
            rng = np.random.default_rng(seed)
            t_param = np.linspace(0.0, 1.0, n)
            freq1 = rng.uniform(2.5, 5.5)
            freq2 = rng.uniform(7.0, 13.0)
            ph1 = rng.uniform(0.0, 6.28)
            ph2 = rng.uniform(0.0, 6.28)
            jitter = (0.65 * np.sin(t_param * freq1 * math.pi + ph1) + 0.35 * np.sin(t_param * freq2 * math.pi + ph2)) * tremor
            tang = np.gradient(samples, axis=0)
            norms = np.stack([-tang[:, 1], tang[:, 0]], axis=-1)
            nlen = np.linalg.norm(norms, axis=-1, keepdims=True) + 1e-6
            samples = samples + (norms / nlen) * jitter[:, None]

        ctx = self.ctx
        ctx.set_source_rgba(*color_srgb)

        if not taper:
            ctx.set_line_width(eff_w)
            ctx.move_to(float(samples[0, 0]), float(samples[0, 1]))
            for x, y in samples[1:]:
                ctx.line_to(float(x), float(y))
            ctx.stroke()
            return

        # Tapered stroke via short segments with smooth pressure variation
        for i in range(n - 1):
            u = (i + 0.5) / max(1, n - 1)
            env = 1.0 if closed else math.sin(max(0.06, min(0.94, u)) * math.pi) ** 0.42
            pressure = 0.86 + 0.22 * math.sin(u * 5.2 + seed * 0.7)
            w_seg = max(0.45, eff_w * env * pressure)
            ctx.set_line_width(w_seg)
            ctx.move_to(float(samples[i, 0]), float(samples[i, 1]))
            ctx.line_to(float(samples[i + 1, 0]), float(samples[i + 1, 1]))
            ctx.stroke()

    def fill_smooth_poly(
        self,
        pts: Sequence[Tuple[float, float]],
        color_srgb: Tuple[float, float, float, float],
        smooth: bool = True,
    ) -> None:
        if len(pts) < 3:
            return
        if smooth:
            p = list(pts) + [pts[0]]
            samples = _catmull_rom_samples(p, samples_per_seg=5)
        else:
            samples = np.asarray(pts, dtype=np.float64)
        ctx = self.ctx
        ctx.set_source_rgba(*color_srgb)
        ctx.move_to(float(samples[0, 0]), float(samples[0, 1]))
        for x, y in samples[1:]:
            ctx.line_to(float(x), float(y))
        ctx.close_path()
        ctx.fill()

    def fill_cel_region(
        self,
        pts: Sequence[Tuple[float, float]],
        base_srgb: Tuple[float, float, float],
        shadow_srgb: Tuple[float, float, float],
        light_dir_2d: Tuple[float, float] = (-0.7, -0.7),
        rim_srgb: Optional[Tuple[float, float, float]] = None,
        outline_srgb: Tuple[float, float, float, float] = (0.12, 0.09, 0.08, 0.92),
        outline_width: float = 1.85,
        shadow_shift_px: float = 9.0,
        seed: int = 1,
        smooth: bool = True,
    ) -> None:
        """Traditional cel-colored form with crisp hard-edged shadow shape, optional rim, and tapered contour."""
        if len(pts) < 3:
            return
        if smooth:
            p = list(pts) + [pts[0]]
            samples = _catmull_rom_samples(p, samples_per_seg=5)
        else:
            samples = np.asarray(pts, dtype=np.float64)
        ctx = self.ctx
        ctx.save()

        # Clip to shape silhouette
        ctx.move_to(float(samples[0, 0]), float(samples[0, 1]))
        for x, y in samples[1:]:
            ctx.line_to(float(x), float(y))
        ctx.close_path()
        ctx.clip_preserve()

        # Base cel fill
        ctx.set_source_rgba(*base_srgb, 1.0)
        ctx.fill()

        # Crisp hard-edged cel shadow offset opposite to light direction
        lx, ly = light_dir_2d
        ln = math.hypot(lx, ly) + 1e-6
        dx, dy = (lx / ln) * shadow_shift_px, (ly / ln) * shadow_shift_px
        ctx.set_source_rgba(*shadow_srgb, 1.0)
        ctx.paint()
        ctx.move_to(float(samples[0, 0] + dx), float(samples[0, 1] + dy))
        for x, y in samples[1:]:
            ctx.line_to(float(x + dx), float(y + dy))
        ctx.close_path()
        ctx.set_source_rgba(*base_srgb, 1.0)
        ctx.fill()

        # Optional crisp light rim along lit silhouette edge
        if rim_srgb is not None:
            ctx.move_to(float(samples[0, 0] - dx * 0.25), float(samples[0, 1] - dy * 0.25))
            for x, y in samples[1:]:
                ctx.line_to(float(x - dx * 0.25), float(y - dy * 0.25))
            ctx.close_path()
            ctx.set_source_rgba(*rim_srgb, 0.55)
            ctx.set_line_width(max(1.2, outline_width * 1.1))
            ctx.stroke()

        ctx.restore()

        # Crisp hand-inked tapered contour
        if outline_width > 0.0:
            if smooth:
                self.stroke_catmull_rom(pts, base_width=outline_width, color_srgb=outline_srgb,
                                        taper=False, tremor=0.20, seed=seed, closed=True)
            else:
                ctx.set_source_rgba(*outline_srgb)
                ctx.set_line_width(outline_width)
                ctx.move_to(float(samples[0, 0]), float(samples[0, 1]))
                for x, y in samples[1:]:
                    ctx.line_to(float(x), float(y))
                ctx.close_path()
                ctx.stroke()

    def draw_bezier_tree(
        self,
        root_x: float,
        root_y: float,
        height_px: float,
        trunk_w: float,
        seed: int,
        bark_srgb: Tuple[float, float, float] = (0.14, 0.11, 0.09),
        foliage_srgb: Optional[Tuple[float, float, float]] = None,
        foliage_lit_srgb: Optional[Tuple[float, float, float]] = None,
        wind_phase: float = 0.0,
        wind_amp: float = 0.0,
        depth_lim: int = 5,
    ) -> None:
        """Seeded recursive cubic Bézier oak/apple tree with root flare, sweeping gnarls, and filigree twigs."""
        rng = np.random.default_rng(seed)
        ctx = self.ctx

        # Soft feathered ground shadow beneath trunk root flare
        ctx.save()
        ctx.translate(root_x + trunk_w * 0.4, root_y + trunk_w * 0.12)
        ctx.scale(2.8, 0.45)
        g_sh = cairo.RadialGradient(0.0, 0.0, trunk_w * 0.1, 0.0, 0.0, trunk_w * 1.15)
        g_sh.add_color_stop_rgba(0.0, bark_srgb[0] * 0.65, bark_srgb[1] * 0.70, bark_srgb[2] * 0.85, 0.24)
        g_sh.add_color_stop_rgba(1.0, bark_srgb[0] * 0.65, bark_srgb[1] * 0.70, bark_srgb[2] * 0.85, 0.0)
        ctx.set_source(g_sh)
        ctx.arc(0.0, 0.0, trunk_w * 1.15, 0.0, 2.0 * math.pi)
        ctx.fill()
        ctx.restore()

        def branch(x0: float, y0: float, length: float, angle: float, width: float, depth: int) -> None:
            if depth <= 0 or length < 4.0:
                return
            sway = wind_amp * (1.0 / max(1, depth)) * math.sin(wind_phase + x0 * 0.015 + depth)
            ang = angle + sway
            bend = float(rng.uniform(-0.34, 0.34))
            x3 = x0 + math.cos(ang) * length
            y3 = y0 + math.sin(ang) * length
            cx1 = x0 + math.cos(ang + bend * 0.75) * length * 0.38
            cy1 = y0 + math.sin(ang + bend * 0.75) * length * 0.38
            cx2 = x0 + math.cos(ang - bend * 0.55) * length * 0.74
            cy2 = y0 + math.sin(ang - bend * 0.55) * length * 0.74

            steps = 7
            left_pts, right_pts = [], []
            for i in range(steps + 1):
                t = i / steps
                mt = 1.0 - t
                px = mt**3 * x0 + 3 * mt**2 * t * cx1 + 3 * mt * t**2 * cx2 + t**3 * x3
                py = mt**3 * y0 + 3 * mt**2 * t * cy1 + 3 * mt * t**2 * cy2 + t**3 * y3
                tx = 3 * mt**2 * (cx1 - x0) + 6 * mt * t * (cx2 - cx1) + 3 * t**2 * (x3 - cx2)
                ty = 3 * mt**2 * (cy1 - y0) + 6 * mt * t * (cy2 - cy1) + 3 * t**2 * (y3 - cy2)
                tn = math.hypot(tx, ty) + 1e-6
                nx, ny = -ty / tn, tx / tn
                flare = (1.55 if (depth == depth_lim and t < 0.25) else 1.0)
                taper_end = 0.25 if depth == 1 else 0.56
                w_cur = max(0.38, width * flare * (1.0 - (1.0 - taper_end) * t)) * 0.5
                left_pts.append((px + nx * w_cur, py + ny * w_cur))
                right_pts.append((px - nx * w_cur, py - ny * w_cur))
            poly = left_pts + right_pts[::-1]
            ctx.set_source_rgba(*bark_srgb, 0.96)
            ctx.move_to(*poly[0])
            for px, py in poly[1:]:
                ctx.line_to(px, py)
            ctx.close_path()
            ctx.fill()

            # Foliage clusters on outer twigs
            if foliage_srgb is not None and depth <= 2:
                n_cl = int(rng.integers(3, 6))
                for _ in range(n_cl):
                    fx = x3 + float(rng.uniform(-length * 0.50, length * 0.50))
                    fy = y3 + float(rng.uniform(-length * 0.42, length * 0.34))
                    fr = float(rng.uniform(length * 0.28, length * 0.52))
                    nv = 7
                    lpts = []
                    for k in range(nv):
                        th = k * (2.0 * math.pi / nv) + float(rng.uniform(-0.22, 0.22))
                        rr = fr * float(rng.uniform(0.72, 1.18))
                        lpts.append((fx + math.cos(th) * rr * 1.18, fy + math.sin(th) * rr * 0.85))
                    col = foliage_lit_srgb if (foliage_lit_srgb is not None and rng.random() < 0.38) else foliage_srgb
                    shade_var = float(rng.uniform(0.86, 1.12))
                    self.fill_smooth_poly(
                        lpts,
                        (min(1.0, col[0] * shade_var), min(1.0, col[1] * shade_var), min(1.0, col[2] * shade_var), 0.93),
                    )

            # Spawn organic asymmetric child branches + fine side twigs
            n_ch = 3 if depth >= 3 else int(rng.integers(2, 4))
            spread = float(rng.uniform(0.42, 0.68))
            for c in range(n_ch):
                frac = -0.5 + c / max(1, n_ch - 1) if n_ch > 1 else 0.0
                child_ang = ang + frac * spread * 1.95 + float(rng.uniform(-0.18, 0.18))
                child_len = length * float(rng.uniform(0.58, 0.78))
                # Start some side branches slightly below tip for natural gnarled oak silhouette
                start_t = 1.0 if c != 1 else float(rng.uniform(0.68, 0.92))
                sx = x0 + (x3 - x0) * start_t
                sy = y0 + (y3 - y0) * start_t
                branch(sx, sy, child_len, child_ang, width * float(rng.uniform(0.52, 0.64)), depth - 1)

        branch(root_x, root_y, height_px * 0.32, -math.pi * 0.5 + float(rng.uniform(-0.08, 0.08)), trunk_w, depth_lim)

    def draw_calligraphy(
        self,
        text: str,
        x: float,
        y: float,
        size_px: float,
        color_srgb: Tuple[float, float, float, float] = (0.90, 0.85, 0.74, 0.94),
        italic: bool = True,
        bold: bool = False,
        align: str = "center",
        letter_spacing: float = 0.0,
    ) -> None:
        """Crisp 17th-century serif/italic ink lettering rendered at 2x supersampling."""
        ctx = self.ctx
        ctx.save()
        slant = cairo.FONT_SLANT_ITALIC if italic else cairo.FONT_SLANT_NORMAL
        weight = cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL
        ctx.select_font_face("Liberation Serif", slant, weight)
        ctx.set_font_size(size_px)

        if letter_spacing == 0.0:
            ext = ctx.text_extents(text)
            tx = x
            if align == "center":
                tx = x - (ext.width * 0.5 + ext.x_bearing)
            elif align == "right":
                tx = x - (ext.width + ext.x_bearing)
            # Subtle ink drop/paper halo for legibility without modern shadow
            ctx.set_source_rgba(0.03, 0.025, 0.02, color_srgb[3] * 0.32)
            ctx.move_to(tx + 1.2, y + 1.4)
            ctx.show_text(text)
            ctx.set_source_rgba(*color_srgb)
            ctx.move_to(tx, y)
            ctx.show_text(text)
        else:
            advs = [ctx.text_extents(ch).x_advance + letter_spacing for ch in text]
            total_w = sum(advs) - letter_spacing
            cx = x - total_w * 0.5 if align == "center" else (x - total_w if align == "right" else x)
            for ch, adv in zip(text, advs):
                ctx.set_source_rgba(0.03, 0.025, 0.02, color_srgb[3] * 0.32)
                ctx.move_to(cx + 1.2, y + 1.4)
                ctx.show_text(ch)
                ctx.set_source_rgba(*color_srgb)
                ctx.move_to(cx, y)
                ctx.show_text(ch)
                cx += adv
        ctx.restore()

    def to_premul_linear_rgba(self) -> np.ndarray:
        """Downsample 2x Cairo premultiplied ARGB32 surface via uint8 SIMD Lanczos4 and convert to linear premul RGBA."""
        self.surf.flush()
        buf = np.frombuffer(self.surf.get_data(), dtype=np.uint8)
        bgra = buf.reshape((self.h * self.ss, self.w * self.ss, 4))
        # Fast SIMD uint8 2x2 box SSAA downsample directly on premultiplied BGRA (crisp, zero ringing)
        if self.ss > 1:
            bgra_ds = cv2.resize(bgra, (self.w, self.h), interpolation=cv2.INTER_AREA)
        else:
            bgra_ds = bgra

        premul_lin = np.zeros((self.h, self.w, 4), dtype=np.float32)
        a_u8 = bgra_ds[..., 3]
        nz = a_u8 > 0
        if not np.any(nz):
            return premul_lin

        a = a_u8[nz].astype(np.float32) * (1.0 / 255.0)
        inv_a = 1.0 / np.maximum(a, 1e-4)
        # Convert B, G, R premul -> straight sRGB -> LUT linear -> re-premultiply (only on drawn pixels!)
        r_s = np.clip((bgra_ds[..., 2][nz].astype(np.float32) * (1.0 / 255.0)) * inv_a, 0.0, 1.0)
        g_s = np.clip((bgra_ds[..., 1][nz].astype(np.float32) * (1.0 / 255.0)) * inv_a, 0.0, 1.0)
        b_s = np.clip((bgra_ds[..., 0][nz].astype(np.float32) * (1.0 / 255.0)) * inv_a, 0.0, 1.0)

        premul_lin[..., 0][nz] = _SRGB_TO_LIN_LUT[(r_s * 4095.0).astype(np.int32)] * a
        premul_lin[..., 1][nz] = _SRGB_TO_LIN_LUT[(g_s * 4095.0).astype(np.int32)] * a
        premul_lin[..., 2][nz] = _SRGB_TO_LIN_LUT[(b_s * 4095.0).astype(np.int32)] * a
        premul_lin[..., 3][nz] = a
        return premul_lin
