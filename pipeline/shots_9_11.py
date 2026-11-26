"""Shots 9 to 11 (1:29 - 2:00, frames 2136 - 2879):
- Shot 9: The Principia (1:29-1:42, f 2136-2447)
- Shot 10: The Passing Years (1:42-1:51, f 2448-2663)
- Shot 11: The Hidden Order (1:51-2:00, f 2664-2879)
"""

from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np

from .characters import (
    blink_envelope,
    draw_closeup_hand,
    draw_halley_bust,
    draw_newton_bust,
    quantize_exposure,
)
from .core_3d import Camera3D, over_premul, parchment_uv, smootherstep, smoothstep, voronoi_masonry, warp_plane_premul
from .painter import InkCanvas2x, apply_gouache_finish, hex_to_linear
from .physics_optics import integrate_verlet_orbit, simulate_three_laws
from .post_optics import OpticsProfile, apply_disc_bokeh_premul
from .scene_kit import (
    add_radial_light_glow,
    draw_candle_and_flame,
    draw_leaded_window_frame,
    draw_painted_bookshelf,
    paint_clouds_layer,
    paint_consistent_moon,
    paint_rolling_hills,
    paint_sky_gradient,
    paint_starfield_cosmos,
    sample_overscan,
)


# ===========================================================================
# SHOT 9 — The Principia (Frames 2136 - 2447, 1:29 - 1:42)
# ===========================================================================

class Shot9Renderer:
    """1680s mature Newton & Edmund Halley -> Three physically exact painted demonstrations of Laws I-III ->
    1687 wooden printing workshop & leather-bound Philosophiæ Naturalis Principia Mathematica reveal.
    """

    def __init__(self, w: int, h: int) -> None:
        self.w, self.h = w, h
        self.prof_study = OpticsProfile(halation_strength=0.28, halation_tint=(1.12, 0.84, 0.50), ca_px=0.25, vignette=0.28)
        self.prof_diag = OpticsProfile(halation_strength=0.18, halation_tint=(1.04, 0.90, 0.68), ca_px=0.0, vignette=0.26)
        self.prof_book = OpticsProfile(halation_strength=0.24, halation_tint=(1.10, 0.88, 0.58), ca_px=0.0, vignette=0.26)
        self._build_caches()

    def _build_caches(self) -> None:
        w, h = self.w, self.h
        # 1. 1684 study room with Halley
        rng = np.random.default_rng(901)
        sbg = np.full((h, w, 3), np.array([0.045, 0.030, 0.020], dtype=np.float32))
        sbg = apply_gouache_finish(sbg, rng, mottle_strength=0.05, n_dabs=86, dab_strength=0.08,
                                   shadow_bounce_rgb=(0.045, 0.026, 0.012))
        sc = InkCanvas2x(w, h)
        draw_painted_bookshelf(sc, w * 0.04, h * 0.05, w * 0.34, h * 0.64, 3, seed=902)
        draw_painted_bookshelf(sc, w * 0.64, h * 0.05, w * 0.32, h * 0.64, 3, seed=903)
        sc.fill_cel_region([(0, h * 0.72), (w, h * 0.72), (w, h), (0, h)],
                           (0.24, 0.15, 0.09), (0.11, 0.07, 0.04), (0.0, -0.9), outline_width=2.0, seed=904)
        # Tall stacks of accumulated manuscript pages on center table
        for sx_f, sh_f, sd in ((0.41, 0.09, 905), (0.50, 0.13, 906), (0.58, 0.07, 907)):
            sx, sh = w * sx_f, h * sh_f
            sc.fill_cel_region([(sx - w * 0.05, h * 0.75 - sh), (sx + w * 0.05, h * 0.75 - sh),
                                (sx + w * 0.055, h * 0.76), (sx - w * 0.055, h * 0.76)],
                               (0.88, 0.81, 0.64), (0.62, 0.54, 0.40), (-0.4, -0.8), outline_width=1.5, seed=sd)
        self.halley_bg = over_premul(sbg, sc.to_premul_linear_rgba())

        # 2. Parchment plate for the 3 physical demonstrations of Newton's Laws of Motion
        p_uv = parchment_uv(h, w, base_rgb=(0.56, 0.47, 0.34), seed=910)[..., :3]
        self.laws_bg = apply_gouache_finish(p_uv, np.random.default_rng(911), mottle_strength=0.04, n_dabs=80,
                                            dab_strength=0.06, paper_tooth=0.02, shadow_bounce_rgb=(0.04, 0.025, 0.012))

        # 3. 1687 Printing Press Workshop & Leather-Bound Principia reveal backdrop
        wbg = np.full((h, w, 3), np.array([0.050, 0.036, 0.024], dtype=np.float32))
        self.press_bg = apply_gouache_finish(wbg, np.random.default_rng(915), mottle_strength=0.05, n_dabs=84,
                                             dab_strength=0.08, shadow_bounce_rgb=(0.04, 0.024, 0.012))

    def _draw_brass_sphere(self, canvas: InkCanvas2x, cx: float, cy: float, r: float, seed: int) -> None:
        pts = [(cx + math.cos(a) * r, cy + math.sin(a) * r) for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)]
        canvas.fill_cel_region(pts, (0.82, 0.65, 0.30), (0.38, 0.27, 0.11), (-0.7, -0.6),
                               rim_srgb=(0.98, 0.90, 0.62), outline_srgb=(0.12, 0.09, 0.06, 0.94),
                               outline_width=2.0, shadow_shift_px=r * 0.32, seed=seed)

    def _draw_force_arrow(self, canvas: InkCanvas2x, x0: float, y0: float, dx: float, col=(0.64, 0.16, 0.12, 0.92)) -> None:
        if abs(dx) < 3.0:
            return
        x1 = x0 + dx
        sgn = 1.0 if dx > 0 else -1.0
        canvas.stroke_catmull_rom([(x0, y0), (x1, y0)], 3.8 * (self.w / 2048.0), col, taper=False, seed=940)
        ah = 14.0 * (self.w / 2048.0)
        canvas.fill_smooth_poly([(x1, y0), (x1 - sgn * ah, y0 - ah * 0.55), (x1 - sgn * ah, y0 + ah * 0.55)], col, False)

    def render(self, local_f: int, global_f: int) -> Tuple[np.ndarray, OpticsProfile]:
        w, h = self.w, self.h
        # 0..75: Silent exchange between mature Newton (left) and Edmund Halley (right) over manuscripts
        if local_f < 76:
            exp_f = quantize_exposure(local_f, cadence=2)
            img = self.halley_bg.copy()
            dyn = InkCanvas2x(w, h)
            draw_newton_bust(
                dyn, w * 0.30, h * 0.44, scale=h * 0.26, age="mature", view="three_quarter_right",
                light_dir_2d=(0.75, -0.35), light_tint=(1.0, 0.88, 0.72),
                blink=blink_envelope(local_f, 68, 12), gaze=(0.38, 0.05), breath_phase=exp_f * 0.09, seed=920,
            )
            draw_halley_bust(
                dyn, w * 0.71, h * 0.45, scale=h * 0.25, light_dir_2d=(-0.75, -0.35),
                light_tint=(1.0, 0.88, 0.72), blink=blink_envelope(local_f, 74, 38), seed=921,
            )
            fx, fy = draw_candle_and_flame(dyn, w * 0.50, h * 0.72, h * 0.11, w * 0.020, global_f, seed=922)
            img = over_premul(img, dyn.to_premul_linear_rgba())
            mod = 1.0 + 0.04 * math.sin(global_f * 0.34)
            img = add_radial_light_glow(img, fx, fy, w * 0.36, (1.0, 0.58, 0.22), 0.44 * mod)
            return img, self.prof_study

        # 76..203: Three elegant, physically exact painted demonstrations of Newton's Laws of Motion
        if local_f < 204:
            sub = local_f - 76  # 128 frames -> ~42 frames per law
            img = self.laws_bg.copy()
            dyn = InkCanvas2x(w, h)
            ink = (0.14, 0.10, 0.08, 0.85)
            xl, xr = w * 0.14, w * 0.86

            if sub < 42:
                # LAW I: Constant velocity under zero net force (uniform spacing strobe marks!)
                u = sub / 41.0
                sim = simulate_three_laws(u)["law1"]
                y_tr = h * 0.54
                dyn.stroke_catmull_rom([(xl, y_tr + h * 0.042), (xr, y_tr + h * 0.042)], 3.2 * (w / 2048.0), ink, taper=False, seed=930)
                # Equal-interval tick marks proving uniform motion!
                for k in range(7):
                    xk = xl + (xr - xl) * (0.10 + 0.12 * k)
                    if xk <= xl + (xr - xl) * sim["x"]:
                        dyn.ctx.set_source_rgba(0.32, 0.22, 0.14, 0.45)
                        dyn.ctx.arc(xk, y_tr, h * 0.012, 0, 2 * math.pi)
                        dyn.ctx.fill()
                sx = xl + (xr - xl) * sim["x"]
                self._draw_brass_sphere(dyn, sx, y_tr, h * 0.040, seed=931)
                dyn.draw_calligraphy("Lex I", w * 0.50, h * 0.24, 34.0 * (w / 2048.0), (0.18, 0.13, 0.09, 0.78), italic=True)
            elif sub < 85:
                # LAW II: Equal-mass spheres (m1 = m2) under different applied forces F1 vs F2 -> a2 = (F2/F1) * a1
                u = (sub - 42) / 42.0
                sim = simulate_three_laws(u)["law2"]
                for row_y, x_pos, f_val, sd in ((h * 0.42, sim["x1"], sim["f1"], 932), (h * 0.66, sim["x2"], sim["f2"], 933)):
                    dyn.stroke_catmull_rom([(xl, row_y + h * 0.040), (xr, row_y + h * 0.040)], 3.0 * (w / 2048.0), ink, taper=False, seed=sd)
                    sx = xl + (xr - xl) * min(0.95, x_pos)
                    self._draw_brass_sphere(dyn, sx, row_y, h * 0.038, seed=sd + 10)
                    # Applied force arrow proportional to F_net
                    self._draw_force_arrow(dyn, sx + h * 0.045, row_y, f_val * w * 0.42)
                dyn.draw_calligraphy("Lex II", w * 0.50, h * 0.22, 34.0 * (w / 2048.0), (0.18, 0.13, 0.09, 0.78), italic=True)
            else:
                # LAW III: Two distinct interacting bodies (mA < mB) exerting equal & opposite forces F_AB = -F_BA!
                u = (sub - 85) / 42.0
                sim = simulate_three_laws(u)["law3"]
                y_tr = h * 0.54
                dyn.stroke_catmull_rom([(xl, y_tr + h * 0.052), (xr, y_tr + h * 0.052)], 3.2 * (w / 2048.0), ink, taper=False, seed=935)
                xA = xl + (xr - xl) * sim["xA"]
                xB = xl + (xr - xl) * sim["xB"]
                rA = h * 0.036
                rB = h * 0.048  # Larger mass mB = 1.6*mA accelerates proportionally less under identical force magnitude!
                self._draw_brass_sphere(dyn, xA, y_tr + (h * 0.052 - rA), rA, seed=936)
                self._draw_brass_sphere(dyn, xB, y_tr + (h * 0.052 - rB), rB, seed=937)
                # Paired equal-and-opposite force arrows acting on DIFFERENT bodies during spring push!
                if abs(sim["fA"]) > 0.01:
                    self._draw_force_arrow(dyn, xA - rA * 1.15, y_tr, -w * 0.12)
                    self._draw_force_arrow(dyn, xB + rB * 1.15, y_tr, +w * 0.12)
                dyn.draw_calligraphy("Lex III", w * 0.50, h * 0.24, 34.0 * (w / 2048.0), (0.18, 0.13, 0.09, 0.78), italic=True)

            return over_premul(img, dyn.to_premul_linear_rgba()), self.prof_diag

        # 204..311: Historical wooden screw printing press impression -> Leather-bound Principia title reveal (1687)!
        u = (local_f - 204) / 107.0
        img = self.press_bg.copy()
        dyn = InkCanvas2x(w, h)
        ink = (0.10, 0.08, 0.06, 0.94)

        if u < 0.34:
            # Wooden screw press descending onto freshly inked rag paper sheet
            pu = smootherstep(u / 0.34)
            press_y = h * (0.12 + 0.30 * math.sin(pu * math.pi))
            # Heavy oak upright frame
            dyn.fill_cel_region([(w * 0.22, h * 0.05), (w * 0.30, h * 0.05), (w * 0.30, h * 0.95), (w * 0.22, h * 0.95)],
                                (0.28, 0.18, 0.11), (0.13, 0.08, 0.05), (-0.7, -0.4), outline_width=2.2, smooth=False, seed=950)
            dyn.fill_cel_region([(w * 0.70, h * 0.05), (w * 0.78, h * 0.05), (w * 0.78, h * 0.95), (w * 0.70, h * 0.95)],
                                (0.28, 0.18, 0.11), (0.13, 0.08, 0.05), (-0.7, -0.4), outline_width=2.2, smooth=False, seed=951)
            # Central wooden screw & descending platen
            dyn.fill_cel_region([(w * 0.46, h * 0.06), (w * 0.54, h * 0.06), (w * 0.54, press_y + h * 0.14), (w * 0.46, press_y + h * 0.14)],
                                (0.36, 0.24, 0.14), (0.18, 0.11, 0.07), (-0.7, -0.4), outline_width=2.0, smooth=False, seed=952)
            dyn.fill_cel_region([(w * 0.31, press_y + h * 0.12), (w * 0.69, press_y + h * 0.12),
                                 (w * 0.69, press_y + h * 0.22), (w * 0.31, press_y + h * 0.22)],
                                (0.32, 0.22, 0.13), (0.16, 0.10, 0.06), (-0.6, -0.7), outline_width=2.2, smooth=False, seed=953)
            # Type bed & rag sheet below
            dyn.fill_cel_region([(w * 0.28, h * 0.68), (w * 0.72, h * 0.68), (w * 0.74, h * 0.80), (w * 0.26, h * 0.80)],
                                (0.88, 0.83, 0.70), (0.62, 0.56, 0.44), (-0.6, -0.7), outline_width=2.0, smooth=False, seed=954)
        else:
            # Hero reveal of the bound 1687 title page: Philosophiæ Naturalis Principia Mathematica, 1687
            bu = smootherstep((u - 0.34) / 0.28)
            bx0, by0, bw, bh = w * 0.27, h * 0.10, w * 0.46, h * 0.80
            # Leather cover border peeking around title page
            dyn.fill_cel_region([(bx0 - 18, by0 - 14), (bx0 + bw + 18, by0 - 14),
                                 (bx0 + bw + 22, by0 + bh + 18), (bx0 - 14, by0 + bh + 18)],
                                (0.28, 0.14, 0.09), (0.14, 0.07, 0.05), (-0.6, -0.7), outline_width=2.4, smooth=False, seed=955)
            # Aged rag-paper title page
            dyn.fill_cel_region([(bx0, by0), (bx0 + bw, by0), (bx0 + bw, by0 + bh), (bx0, by0 + bh)],
                                (0.91, 0.85, 0.71), (0.76, 0.68, 0.54), (-0.5, -0.7), outline_width=1.8, smooth=False, seed=956)
            # Double-rule 17th-century Royal Society title border
            pad = w * 0.024
            dyn.ctx.set_source_rgba(0.16, 0.12, 0.09, 0.75 * bu)
            dyn.ctx.set_line_width(2.0 * (w / 2048.0))
            dyn.ctx.rectangle(bx0 + pad, by0 + pad, bw - 2 * pad, bh - 2 * pad)
            dyn.ctx.stroke()

            sc_f = w / 2048.0
            ink_t = (0.12, 0.09, 0.07, 0.96 * bu)
            dyn.draw_calligraphy("PHILOSOPHIÆ", w * 0.50, h * 0.28, 54.0 * sc_f, ink_t, italic=False, bold=True, letter_spacing=3.0 * sc_f)
            dyn.draw_calligraphy("NATURALIS", w * 0.50, h * 0.36, 46.0 * sc_f, ink_t, italic=False, letter_spacing=4.0 * sc_f)
            dyn.draw_calligraphy("PRINCIPIA", w * 0.50, h * 0.49, 76.0 * sc_f, (0.48, 0.12, 0.09, 0.96 * bu), italic=False, bold=True, letter_spacing=5.0 * sc_f)
            dyn.draw_calligraphy("MATHEMATICA", w * 0.50, h * 0.58, 48.0 * sc_f, ink_t, italic=False, letter_spacing=3.5 * sc_f)
            # Ornamental divider line
            dyn.stroke_catmull_rom([(w * 0.37, h * 0.65), (w * 0.63, h * 0.65)], 2.0 * sc_f, (0.18, 0.13, 0.09, 0.80 * bu), seed=958)
            dyn.draw_calligraphy("Autore Is. Newton", w * 0.50, h * 0.72, 30.0 * sc_f, ink_t, italic=True)
            dyn.draw_calligraphy("LONDINI, 1687", w * 0.50, h * 0.81, 32.0 * sc_f, ink_t, italic=False, letter_spacing=3.0 * sc_f)

        img = over_premul(img, dyn.to_premul_linear_rgba())
        img = add_radial_light_glow(img, w * 0.50, h * 0.48, w * 0.42, (1.0, 0.76, 0.42), 0.24)
        return img, self.prof_book


# ===========================================================================
# SHOT 10 — The Passing Years (Frames 2448 - 2663, 1:42 - 1:51)
# ===========================================================================

class Shot10Renderer:
    """Layered misty London -> Royal Mint balance scales & coins -> 1705 Knighthood ->
    Hand match cut (child mechanism -> elderly brass instrument) -> Elderly Newton beside afternoon window.
    """

    def __init__(self, w: int, h: int) -> None:
        self.w, self.h = w, h
        self.prof_lon = OpticsProfile(halation_strength=0.22, halation_tint=(0.98, 0.86, 0.70), vignette=0.28)
        self.prof_win = OpticsProfile(halation_strength=0.30, halation_tint=(1.14, 0.88, 0.54), vignette=0.30)
        self._build_caches()

    def _build_caches(self) -> None:
        w, h = self.w, self.h
        # 1. Layered 1700s London skyline & misty stone street with St. Paul's dome rising in mist
        sky = paint_sky_gradient(h, w, "#3B4652", "#787D7E", "#C6B69C", sun_xy_norm=(0.74, 0.44),
                                 sun_glow_rgb=(0.38, 0.28, 0.18), horizon_y=0.68, seed=1001)
        sky = over_premul(sky, paint_clouds_layer(h, w, (0.04, 0.50), (0.82, 0.76, 0.68), (0.32, 0.35, 0.39), 0.66, seed=1002))
        lc = InkCanvas2x(w, h)
        # Distant Baroque dome (St. Paul's Cathedral under construction / completion c. 1700)
        dx, dy = w * 0.66, h * 0.56
        lc.fill_smooth_poly([(dx - w * 0.06, dy), (dx - w * 0.05, dy - h * 0.14), (dx, dy - h * 0.22),
                             (dx + w * 0.05, dy - h * 0.14), (dx + w * 0.06, dy)], (0.36, 0.38, 0.41, 0.78))
        # Layered London gables, chimneys & stone facades
        rng = np.random.default_rng(1003)
        for bx in np.linspace(0.0, 0.94, 14):
            bw = w * rng.uniform(0.06, 0.095)
            bh = h * rng.uniform(0.24, 0.46)
            col_v = rng.uniform(0.14, 0.24)
            lc.fill_cel_region([(w * bx, h * 0.78), (w * bx, h * 0.78 - bh), (w * bx + bw, h * 0.78 - bh * 0.95),
                                (w * bx + bw, h * 0.78)],
                               (col_v * 1.1, col_v, col_v * 0.92), (col_v * 0.55, col_v * 0.52, col_v * 0.52),
                               (0.7, -0.4), outline_width=1.6, smooth=False, seed=1010 + int(bx * 100))
        # Wet cobblestone street
        lc.fill_cel_region([(0, h * 0.75), (w, h * 0.75), (w, h), (0, h)],
                           (0.18, 0.17, 0.16), (0.09, 0.09, 0.10), (0.5, -0.8), outline_width=0.0, smooth=False, seed=1015)
        self.london_bg = over_premul(sky, lc.to_premul_linear_rgba())

        # 2. Royal Mint dark oak workshop backdrop
        self.mint_bg = apply_gouache_finish(
            np.full((h, w, 3), np.array([0.055, 0.042, 0.030], dtype=np.float32)),
            np.random.default_rng(1020), mottle_strength=0.05, n_dabs=84, dab_strength=0.08,
            shadow_bounce_rgb=(0.042, 0.028, 0.015),
        )

        # 3. Late afternoon sunlit window interior for elderly Newton
        w_sky = paint_sky_gradient(h, w, "#5A6B7A", "#B89A72", "#E8C082", sun_xy_norm=(0.82, 0.46),
                                   sun_glow_rgb=(0.68, 0.44, 0.20), horizon_y=0.76, seed=1030)
        wc = InkCanvas2x(w, h)
        # Warm shadowed interior plaster walls framing tall leaded window (left, right, top lintel, bottom sill)
        wc.fill_cel_region([(0, 0), (w * 0.58, 0), (w * 0.58, h), (0, h)],
                           (0.22, 0.17, 0.13), (0.10, 0.08, 0.07), (0.8, -0.3), outline_width=0.0, smooth=False, seed=1031)
        wc.fill_cel_region([(w * 0.94, 0), (w, 0), (w, h), (w * 0.94, h)],
                           (0.15, 0.12, 0.09), (0.08, 0.06, 0.05), (0.8, -0.3), outline_width=0.0, smooth=False, seed=1032)
        wc.fill_cel_region([(w * 0.57, 0), (w * 0.95, 0), (w * 0.95, h * 0.06), (w * 0.57, h * 0.06)],
                           (0.16, 0.13, 0.10), (0.09, 0.07, 0.06), (0.4, 0.8), outline_width=0.0, smooth=False, seed=1033)
        wc.fill_cel_region([(w * 0.57, h * 0.80), (w * 0.95, h * 0.80), (w * 0.95, h), (w * 0.57, h)],
                           (0.19, 0.14, 0.10), (0.09, 0.07, 0.05), (0.6, -0.6), outline_width=0.0, smooth=False, seed=1034)
        # Warm oak window sill ledge catching afternoon sunlight
        wc.fill_cel_region([(w * 0.56, h * 0.79), (w * 0.95, h * 0.79), (w * 0.95, h * 0.825), (w * 0.56, h * 0.825)],
                           (0.38, 0.26, 0.15), (0.16, 0.11, 0.07), (0.6, -0.8), outline_width=2.0, smooth=False, seed=1035)
        draw_leaded_window_frame(wc, w * 0.57, h * 0.05, w * 0.38, h * 0.75, 3, 5, (0.12, 0.09, 0.07))
        self.elder_bg = over_premul(w_sky, wc.to_premul_linear_rgba())

    def render(self, local_f: int, global_f: int) -> Tuple[np.ndarray, OpticsProfile]:
        w, h = self.w, self.h
        # 0..53: Misty London streets with moving wooden carriage & rising coal smoke
        if local_f < 54:
            u = local_f / 53.0
            img = self.london_bg.copy()
            dyn = InkCanvas2x(w, h)
            # Horse carriage rolling steadily left-to-right across midground street
            cx = w * (0.20 + 0.46 * u)
            cy = h * 0.80
            coach = [(cx - w * 0.08, cy - h * 0.11), (cx + w * 0.08, cy - h * 0.11),
                     (cx + w * 0.07, cy), (cx - w * 0.07, cy)]
            dyn.fill_cel_region(coach, (0.16, 0.12, 0.10), (0.08, 0.06, 0.06), (0.7, -0.5), outline_width=1.8, seed=1040)
            for wx in (cx - w * 0.05, cx + w * 0.05):
                dyn.stroke_catmull_rom([(wx + math.cos(a) * h * 0.042, cy + h * 0.015 + math.sin(a) * h * 0.042)
                                       for a in np.linspace(0, 2 * math.pi, 14, endpoint=False)],
                                      3.5 * (w / 2048.0), (0.14, 0.11, 0.09, 0.94), taper=False, closed=True, seed=1041)
            dyn.draw_calligraphy("London", w * 0.16, h * 0.15, 32.0 * (w / 2048.0),
                                 (0.92, 0.88, 0.80, 0.82 * smoothstep(local_f / 14.0)), italic=True)
            return over_premul(img, dyn.to_premul_linear_rgba()), self.prof_lon

        # 54..107: Royal Mint precision brass balance scale weighing struck silver coins + brief 1705 knighthood emblem
        if local_f < 108:
            u = (local_f - 54) / 53.0
            img = self.mint_bg.copy()
            dyn = InkCanvas2x(w, h)
            cx, cy = w * 0.50, h * 0.44
            # Balance beam settling smoothly into exact equilibrium!
            tilt = 0.14 * math.exp(-u * 4.2) * math.cos(u * 10.5)
            # Central brass pillar
            dyn.fill_cel_region([(cx - w * 0.014, cy - h * 0.04), (cx + w * 0.014, cy - h * 0.04),
                                 (cx + w * 0.04, h * 0.82), (cx - w * 0.04, h * 0.82)],
                                (0.76, 0.60, 0.28), (0.36, 0.26, 0.11), (-0.7, -0.6), outline_width=2.0, smooth=False, seed=1050)
            # Cross-beam
            arm = w * 0.24
            lx, ly = cx - math.cos(tilt) * arm, cy - math.sin(tilt) * arm
            rx, ry = cx + math.cos(tilt) * arm, cy + math.sin(tilt) * arm
            dyn.stroke_catmull_rom([(lx, ly), (rx, ry)], 8.0 * (w / 2048.0), (0.82, 0.66, 0.30, 0.98), taper=False, seed=1051)
            # Suspended brass pans + silver coins & standard weight
            for px, py, is_coins in ((lx, ly, True), (rx, ry, False)):
                pan_y = py + h * 0.22
                dyn.stroke_catmull_rom([(px, py), (px - w * 0.06, pan_y)], 2.2 * (w / 2048.0), (0.70, 0.56, 0.26, 0.90), seed=1052)
                dyn.stroke_catmull_rom([(px, py), (px + w * 0.06, pan_y)], 2.2 * (w / 2048.0), (0.70, 0.56, 0.26, 0.90), seed=1053)
                pan = [(px - w * 0.075, pan_y), (px + w * 0.075, pan_y), (px + w * 0.055, pan_y + h * 0.032), (px - w * 0.055, pan_y + h * 0.032)]
                dyn.fill_cel_region(pan, (0.78, 0.62, 0.28), (0.38, 0.28, 0.12), (-0.6, -0.7), outline_width=1.8, seed=1054)
                if is_coins:
                    for k in range(3):
                        dyn.fill_cel_region([(px - w * 0.035 + k * 6, pan_y - 10 - k * 9), (px + w * 0.035 + k * 6, pan_y - 10 - k * 9),
                                             (px + w * 0.035 + k * 6, pan_y - 2 - k * 9), (px - w * 0.035 + k * 6, pan_y - 2 - k * 9)],
                                            (0.88, 0.90, 0.92), (0.50, 0.54, 0.58), (-0.7, -0.6), outline_width=1.2, seed=1055 + k)
                else:
                    dyn.fill_cel_region([(px - w * 0.025, pan_y - h * 0.055), (px + w * 0.025, pan_y - h * 0.055),
                                         (px + w * 0.032, pan_y), (px - w * 0.032, pan_y)],
                                        (0.68, 0.52, 0.22), (0.34, 0.24, 0.10), (-0.7, -0.6), outline_width=1.5, seed=1058)
            dyn.draw_calligraphy("Royal Mint  ·  Knighted 1705", w * 0.50, h * 0.90, 28.0 * (w / 2048.0),
                                 (0.88, 0.80, 0.64, 0.78 * smoothstep(u / 0.3)), italic=True)
            img = over_premul(img, dyn.to_premul_linear_rgba())
            img = add_radial_light_glow(img, cx, cy, w * 0.35, (1.0, 0.72, 0.34), 0.28)
            return img, self.prof_lon

        # 108..149: Match cut — Child's hand turning wooden gear (108..128) -> Elderly hand holding brass divider (129..149)
        if local_f < 150:
            is_child = local_f < 129
            u = ((local_f - 108) / 20.0) if is_child else ((local_f - 129) / 20.0)
            img = self.mint_bg.copy()
            dyn = InkCanvas2x(w, h)
            # Matching circular mechanism center for an unmistakable graphic match cut!
            gx, gy = w * 0.58, h * 0.52
            gr = h * 0.15
            ang = u * 0.85
            if is_child:
                # Wooden windmill cogwheel
                pts = [(gx + math.cos(a) * gr, gy + math.sin(a) * gr) for a in np.linspace(ang, ang + 2 * math.pi, 16, endpoint=False)]
                dyn.fill_cel_region(pts, (0.58, 0.42, 0.24), (0.30, 0.20, 0.11), (-0.6, -0.6), outline_width=2.0, seed=1060)
                draw_closeup_hand(dyn, w * 0.28, h * 0.58, scale=h * 0.32, age="child", pose="hold",
                                  angle_deg=-10.0, light_tint=(0.96, 0.94, 0.90), seed=1061)
            else:
                # Engraved brass astronomical astrolabe / compass ring
                pts = [(gx + math.cos(a) * gr, gy + math.sin(a) * gr) for a in np.linspace(ang, ang + 2 * math.pi, 24, endpoint=False)]
                dyn.stroke_catmull_rom(pts, 10.0 * (w / 2048.0), (0.82, 0.66, 0.30, 0.96), taper=False, closed=True, seed=1062)
                dyn.stroke_catmull_rom([(gx - gr, gy), (gx + gr, gy)], 3.0 * (w / 2048.0), (0.76, 0.60, 0.26, 0.92), seed=1063)
                draw_closeup_hand(dyn, w * 0.28, h * 0.58, scale=h * 0.34, age="elderly", pose="hold",
                                  angle_deg=-10.0, light_tint=(1.0, 0.90, 0.74), seed=1064)
            return over_premul(img, dyn.to_premul_linear_rgba()), self.prof_lon

        # 150..215: Elderly Newton seated quietly beside window, golden late-afternoon light crossing his weathered face
        u = (local_f - 150) / 65.0
        exp_f = quantize_exposure(local_f, cadence=3)  # Quiet contemplative cadence on threes
        img = self.elder_bg.copy()
        dyn = InkCanvas2x(w, h)
        draw_newton_bust(
            dyn, w * 0.41, h * 0.53, scale=h * 0.33, age="elderly", view="three_quarter_right",
            light_dir_2d=(0.85, -0.30), light_tint=(1.0, 0.88, 0.68),
            blink=blink_envelope(local_f - 150, 58, 22), head_tilt_deg=2.0,
            gaze=(0.35, 0.08), breath_phase=exp_f * 0.08, seed=1070,
        )
        img = over_premul(img, dyn.to_premul_linear_rgba())
        # Warm afternoon sunlight beam streaming softly through leaded window across his face
        img = add_radial_light_glow(img, w * 0.78, h * 0.42, w * 0.44, (1.05, 0.72, 0.32), 0.38)
        return img, self.prof_win


# ===========================================================================
# SHOT 11 — The Hidden Order (Frames 2664 - 2879, 1:51 - 2:00)
# ===========================================================================

class Shot11Renderer:
    """Star-filled Woolsthorpe apple tree connected by fine ink geometries (apple, pendulum, spectrum, Moon) ->
    Camera rises into wide awe-inspiring solar system -> Textured dark manuscript card & final fade to black.
    """

    def __init__(self, w: int, h: int) -> None:
        self.w, self.h = w, h
        self.prof_sky = OpticsProfile(halation_strength=0.30, halation_tint=(0.96, 0.94, 1.08), ca_px=0.18, vignette=0.25)
        self.prof_end = OpticsProfile(halation_strength=0.14, halation_tint=(1.0, 0.90, 0.72), ca_px=0.0, vignette=0.30)
        self._build_caches()

    def _build_caches(self) -> None:
        w, h = self.w, self.h
        # 1. Immense star-filled sky over the familiar Woolsthorpe apple tree silhouette
        self.cosmos = paint_starfield_cosmos(h, w, seed=1101)
        # Tall vertical panorama (2x height) so camera smoothly rises from tree -> clouds -> Earth -> Solar System!
        pan_h = int(h * 1.95)
        self.cosmos_pan = paint_starfield_cosmos(pan_h, w, seed=1102)
        tree_layer = InkCanvas2x(w, h)
        # Rolling dark nocturnal hillside + iconic gnarled Woolsthorpe apple tree
        tree_layer.fill_smooth_poly(
            [(0, h * 0.84), (w * 0.35, h * 0.80), (w * 0.72, h * 0.83), (w, h * 0.81), (w, h), (0, h)],
            (0.035, 0.048, 0.042, 0.98),
        )
        tree_layer.draw_bezier_tree(
            w * 0.30, h * 0.88, h * 0.64, w * 0.026, seed=506,  # Same seed as Shot 5 apple tree!
            bark_srgb=(0.05, 0.05, 0.06), foliage_srgb=(0.05, 0.08, 0.07),
            foliage_lit_srgb=(0.10, 0.14, 0.16), depth_lim=4,
        )
        self.tree_premul = tree_layer.to_premul_linear_rgba()

        # Precompute 5 Verlet planetary orbits for grand ascending solar system finale
        self.orbits = []
        for a, ecc, col, r_sc in (
            (0.18, 0.18, (0.78, 0.74, 0.68), 5.0),
            (0.29, 0.02, (0.34, 0.62, 0.95), 8.2),
            (0.42, 0.09, (0.88, 0.44, 0.28), 6.5),
            (0.62, 0.05, (0.88, 0.74, 0.56), 14.0),
            (0.84, 0.055, (0.92, 0.84, 0.62), 12.0),
        ):
            rp = a * (1.0 - ecc)
            vp = math.sqrt((1.0 / a) * (1.0 + ecc) / (1.0 - ecc))
            tr, _, _ = integrate_verlet_orbit(np.array([rp, 0.0]), np.array([0.0, vp]), mu=1.0, dt=0.008, steps=1200)
            self.orbits.append((tr, col, r_sc))

        # 2. Textured dark aged manuscript backdrop for final title card
        dark_p = parchment_uv(h, w, base_rgb=(0.042, 0.036, 0.029), seed=1110)[..., :3]
        self.end_manuscript = apply_gouache_finish(dark_p, np.random.default_rng(1111), mottle_strength=0.05,
                                                   n_dabs=86, dab_strength=0.07, paper_tooth=0.025,
                                                   shadow_bounce_rgb=(0.012, 0.010, 0.008))

    def render(self, local_f: int, global_f: int) -> Tuple[np.ndarray, OpticsProfile]:
        w, h = self.w, self.h
        # 0..121: Apple tree under starry sky -> fine golden-ink geometries connecting apple, pendulum, prism, Moon ->
        # Smooth camera rise past tree & clouds into grand solar system!
        if local_f < 122:
            u = local_f / 121.0
            rise = smootherstep(max(0.0, (u - 0.34) / 0.66))
            # Sample rising star panorama from bottom (tree) to top (solar system)
            pan_h = self.cosmos_pan.shape[0]
            y_off = int(round((1.0 - rise) * (pan_h - h)))
            img = self.cosmos_pan[y_off:y_off + h, :, :].copy()

            # Moon visible above tree & shifting downward as camera ascends
            moon_y = h * 0.24 + rise * h * 0.75
            if moon_y < h * 1.15:
                img = paint_consistent_moon(img, w * 0.74, moon_y, h * 0.054, (-0.78, -0.35), 1.0)

            # Composite apple tree moving down out of frame as camera ascends into cosmos
            shift_y = int(round(rise * h * 1.15))
            if shift_y < h:
                tree_shifted = np.zeros_like(self.tree_premul)
                tree_shifted[shift_y:, :, :] = self.tree_premul[:h - shift_y, :, :]
                img = over_premul(img, tree_shifted)

            dyn = InkCanvas2x(w, h)
            # Fine golden-ink unifying geometries weaving apple, pendulum arc, prism ray & Moon orbit!
            geom_a = smoothstep(u / 0.22) * (1.0 - smoothstep((u - 0.56) / 0.22))
            if geom_a > 0.01:
                gcol = (0.95, 0.84, 0.58, 0.72 * geom_a)
                sc = w / 2048.0
                # 1) Falling apple vertical drop curve from branch (x=0.44)
                dyn.stroke_catmull_rom([(w * 0.44, h * 0.40 + shift_y), (w * 0.44, h * 0.78 + shift_y)], 2.2 * sc, gcol, seed=1120)
                # 2) Pendulum harmonic arc sweeping below
                pend = [(w * 0.44 + math.sin(a) * h * 0.26, h * 0.40 + shift_y + math.cos(a) * h * 0.26)
                        for a in np.linspace(-0.65, 0.65, 22)]
                dyn.stroke_catmull_rom(pend, 2.0 * sc, gcol, seed=1121)
                # 3) Subtle spectral refraction fan branching rightward toward Moon
                spec_c = [(0.92, 0.28, 0.22), (0.96, 0.72, 0.24), (0.34, 0.86, 0.42), (0.28, 0.64, 0.96)]
                for si, sc_rgb in enumerate(spec_c):
                    dyn.stroke_catmull_rom([(w * 0.44, h * 0.48 + shift_y), (w * 0.74, h * (0.32 + si * 0.032) + shift_y)],
                                           1.8 * sc, (*sc_rgb, 0.55 * geom_a), seed=1122 + si)
                # 4) Grand circular Moon orbital arc connecting tree to Moon
                orb = [(w * 0.44 + math.cos(a) * w * 0.36, h * 0.52 + shift_y + math.sin(a) * h * 0.34)
                       for a in np.linspace(-math.pi * 0.85, -math.pi * 0.08, 32)]
                dyn.stroke_catmull_rom(orb, 2.2 * sc, gcol, seed=1126)

            # Grand wide solar system emerging in upper cosmos as camera rises!
            sol_a = smootherstep((u - 0.42) / 0.38)
            if sol_a > 0.01:
                sx, sy = w * 0.50, h * (0.48 - (1.0 - rise) * 0.55)
                img = add_radial_light_glow(img, sx, sy, h * 0.32, (1.05, 0.76, 0.36), 0.72 * sol_a, power=1.5)
                img = add_radial_light_glow(img, sx, sy, h * 0.06, (1.35, 1.22, 0.90), 1.05 * sol_a, power=2.0)
                for idx, (tr, col, r_sc) in enumerate(self.orbits):
                    pts = [(sx + float(p[0]) * w * 0.52, sy - float(p[1]) * h * 0.27) for p in tr[::22]]
                    dyn.stroke_catmull_rom(pts, 1.6 * (w / 2048.0), (0.92, 0.84, 0.64, 0.42 * sol_a),
                                           taper=False, closed=True, seed=1130 + idx)
                    kp = int((idx * 170 + u * 260) % len(tr))
                    px = sx + float(tr[kp, 0]) * w * 0.52
                    py = sy - float(tr[kp, 1]) * h * 0.27
                    dyn.ctx.set_source_rgba(*col, 0.95 * sol_a)
                    dyn.ctx.arc(px, py, r_sc * (w / 2048.0) * 1.15, 0, 2 * math.pi)
                    dyn.ctx.fill()

            img = over_premul(img, dyn.to_premul_linear_rgba())
            # Smooth dissolve into manuscript end card in final 14 frames of this segment
            if local_f > 107:
                t_d = smoothstep((local_f - 107) / 14.0)
                img = img * (1.0 - t_d) + self.end_manuscript * t_d
            return img, self.prof_sky

        # 122..215: Textured dark manuscript page -> Handwritten title & original caption -> Hold -> Fade completely to black!
        # Holds clearly readable for ~3 seconds (f 128..190), fades cleanly to pure black by f 215 (2:00.00)
        fade_in = smoothstep((local_f - 122) / 14.0)
        fade_out = 1.0 - smootherstep((local_f - 186) / 29.0)
        vis = fade_in * fade_out
        if vis <= 1e-4:
            return np.zeros((h, w, 3), dtype=np.float32), self.prof_end

        img = self.end_manuscript.copy()
        dyn = InkCanvas2x(w, h)
        sc = w / 2048.0
        # Subtle ruled manuscript flourish
        dyn.stroke_catmull_rom([(w * 0.36, h * 0.525), (w * 0.64, h * 0.525)], 1.6 * sc,
                               (0.84, 0.76, 0.60, 0.45 * fade_in), seed=1150)
        dyn.draw_calligraphy(
            "Sir Isaac Newton — 1642–1727",
            w * 0.50, h * 0.47, 56.0 * sc,
            color_srgb=(0.94, 0.89, 0.78, 0.96 * fade_in),
            italic=False, bold=False, letter_spacing=1.8 * sc,
        )
        dyn.draw_calligraphy(
            "In the ordinary, he found the laws of the extraordinary.",
            w * 0.50, h * 0.59, 34.0 * sc,
            color_srgb=(0.86, 0.80, 0.68, 0.90 * fade_in),
            italic=True, bold=False,
        )
        img = over_premul(img, dyn.to_premul_linear_rgba())
        return img * fade_out, self.prof_end
