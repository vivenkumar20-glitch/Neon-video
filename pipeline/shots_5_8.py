"""Shots 5 to 8 (0:40 - 1:29, frames 960 - 2135):
- Shot 5: The Falling Apple (0:40-0:53, f 960-1271)
- Shot 6: The Mathematics of Change (0:53-1:04, f 1272-1535)
- Shot 7: The Nature of Light (1:04-1:15, f 1536-1799)
- Shot 8: Beyond the Earth (1:15-1:29, f 1800-2135)
"""

from __future__ import annotations

import math
from typing import List, Tuple

import cairo
import cv2
import numpy as np

from .characters import (
    blink_envelope,
    draw_closeup_eye,
    draw_closeup_hand,
    draw_newton_bust,
    draw_newton_figure,
    quantize_exposure,
)
from .core_3d import Camera3D, over_premul, parchment_uv, smootherstep, smoothstep, warp_plane_premul, wood_planks_uv
from .painter import InkCanvas2x, apply_gouache_finish, fbm_gouache, hex_to_linear, linear_to_srgb, srgb_to_linear
from .physics_optics import (
    fluxion_curve_and_tangent,
    integrate_verlet_orbit,
    newton_cannonball_trajectories,
    riemann_strips,
    simulate_falling_apple,
    trace_prism_ray,
    trace_prism_recombination,
    wavelength_to_linear_rgb,
)
from .post_optics import OpticsProfile, apply_disc_bokeh_premul
from .scene_kit import (
    add_radial_light_glow,
    draw_candle_and_flame,
    paint_clouds_layer,
    paint_consistent_moon,
    paint_rolling_hills,
    paint_sky_gradient,
    paint_starfield_cosmos,
    sample_overscan,
)


# ===========================================================================
# SHOT 5 — The Falling Apple (Frames 960 - 1271, 0:40 - 0:53)
# ===========================================================================

class Shot5Renderer:
    """Sunlit orchard beneath apple tree -> Apple falls naturally (never hits head!) ->
    Newton looks at apple then daytime Moon -> Match cut apple curve to Earth horizon & Moon orbit.
    """

    def __init__(self, w: int, h: int) -> None:
        self.w, self.h = w, h
        self.prof_garden = OpticsProfile(halation_strength=0.28, halation_tint=(1.10, 0.96, 0.72), vignette=0.22)
        self.prof_orbit = OpticsProfile(halation_strength=0.26, halation_tint=(0.86, 0.94, 1.12), ca_px=0.15, vignette=0.24)
        self._build_caches()

    def _build_caches(self) -> None:
        w, h = self.w, self.h
        # 1. Warm afternoon Woolsthorpe orchard with pale daytime Moon visible high right
        sky = paint_sky_gradient(
            h, w, "#4874A8", "#88B0CE", "#E4DEC8",
            sun_xy_norm=(0.18, 0.28), sun_glow_rgb=(0.55, 0.44, 0.25), sun_radius=0.34,
            horizon_y=0.66, seed=501,
        )
        sky = over_premul(sky, paint_clouds_layer(h, w, (0.06, 0.45), (0.98, 0.95, 0.88), (0.62, 0.70, 0.80), 0.46, seed=502))
        sky = paint_consistent_moon(sky, w * 0.80, h * 0.20, h * 0.048, sun_dir_2d=(-0.82, -0.30),
                                    alpha_scale=0.85, daytime_blend=True)

        hills = paint_rolling_hills(sky, [
            (0.60, 0.04, hex_to_linear("#688656"), 0.40, 503),
            (0.70, 0.04, hex_to_linear("#4D723B"), 0.16, 504),
            (0.79, 0.025, hex_to_linear("#3C622B"), 0.04, 505),
        ])

        # Gnarled Woolsthorpe apple tree anchored on left-midground (x ~ 24%) arching overhead to 65%
        tc = InkCanvas2x(w, h)
        tc.draw_bezier_tree(
            w * 0.24, h * 0.90, h * 0.86, w * 0.036, seed=506,
            bark_srgb=(0.24, 0.17, 0.12), foliage_srgb=(0.22, 0.40, 0.18),
            foliage_lit_srgb=(0.42, 0.58, 0.24), depth_lim=4,
        )
        # Overhanging fruiting bough reaching across upper right where the hero apple hangs
        bough = [(w * 0.24, h * 0.44), (w * 0.46, h * 0.33), (w * 0.64, h * 0.285), (w * 0.76, h * 0.30)]
        tc.stroke_catmull_rom(bough, 16.0 * (w / 2048.0), (0.24, 0.17, 0.12, 0.98), taper=True, tremor=0.8, seed=507)
        # Static red-gold apples nestled cleanly in canopy & hanging directly beneath bough
        for ax, ay in ((0.18, 0.36), (0.31, 0.31), (0.36, 0.38), (0.53, 0.325)):
            self._draw_apple(tc, w * ax, h * ay, h * 0.019, 0.1, seed=510)

        self.orchard_bg = over_premul(hills, tc.to_premul_linear_rgba())

        # Foreground out-of-focus sunlit grass blades at bottom corners (Stage 4 circular disc bokeh)
        fg_c = InkCanvas2x(w, h)
        rng = np.random.default_rng(512)
        for _ in range(38):
            gx = rng.uniform(0.0, w)
            gh = rng.uniform(h * 0.08, h * 0.19)
            lean = rng.uniform(-35, 35)
            fg_c.stroke_catmull_rom([(gx, h + 10), (gx + lean * 0.5, h - gh * 0.5), (gx + lean, h - gh)],
                                   7.5 * (w / 2048.0), (0.14, 0.26, 0.10, 0.92), taper=True, seed=513)
        self.orchard_fg = apply_disc_bokeh_premul(fg_c.to_premul_linear_rgba(), radius=7.8)

        # 2. Match-cut orbital diagram backdrop (painted deep astronomical sky + precomputed curved Earth limb)
        self.cosmos_bg = paint_starfield_cosmos(h, w, seed=520)
        cx_e, cy_e, r_e = w * 0.42, h * 1.48, h * 1.02
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = np.sqrt((xx - cx_e) ** 2 + (yy - cy_e) ** 2)
        self._e_body_m = np.clip((r_e - d) * 0.65, 0.0, 1.0)[..., None]
        self._e_atmo = (np.exp(-((np.abs(d - r_e) / (h * 0.028)) ** 1.6)) * (d >= r_e - 4.0))[..., None]
        rng_e = np.random.default_rng(540)
        e_tex = cv2.resize(fbm_gouache(h // 4, w // 4, rng_e, ((30, 30), (11, 11)), (0.65, 0.35)), (w, h), interpolation=cv2.INTER_CUBIC)
        land = np.clip((e_tex - 0.35) * 1.5, 0.0, 1.0)[..., None]
        lit_h = np.clip(1.05 - (np.abs(xx - w * 0.25) / (w * 0.95)) ** 1.6, 0.18, 1.05)[..., None]
        apple_col = np.array([0.46, 0.07, 0.04], dtype=np.float32).reshape(1, 1, 3)
        earth_col = np.array([0.07, 0.18, 0.38], dtype=np.float32).reshape(1, 1, 3)
        earth_surf = earth_col * (1.0 - land * 0.55) + np.array([0.14, 0.24, 0.13], np.float32) * (land * 0.55)
        self._apple_lit = (apple_col * lit_h).astype(np.float32)
        self._earth_lit = (earth_surf * lit_h).astype(np.float32)

    def _draw_apple(self, canvas: InkCanvas2x, cx: float, cy: float, rad: float, rot: float, seed: int = 515) -> None:
        ctx = canvas.ctx
        ctx.save()
        ctx.translate(cx, cy)
        ctx.rotate(rot)
        pts = [
            (0.0, -rad * 0.85),
            (rad * 0.72, -rad * 0.92),
            (rad * 1.02, -rad * 0.10),
            (rad * 0.78, rad * 0.88),
            (0.0, rad * 0.98),
            (-rad * 0.78, rad * 0.88),
            (-rad * 1.02, -rad * 0.10),
            (-rad * 0.72, -rad * 0.92),
        ]
        canvas.fill_cel_region(pts, (0.76, 0.22, 0.14), (0.42, 0.10, 0.08), (-0.7, -0.6),
                               rim_srgb=(0.94, 0.72, 0.34), outline_width=max(1.2, rad * 0.09),
                               shadow_shift_px=rad * 0.32, seed=seed)
        # Stem & small leaf
        canvas.stroke_catmull_rom([(0.0, -rad * 0.78), (rad * 0.12, -rad * 1.38)], max(1.2, rad * 0.12),
                                  (0.24, 0.16, 0.10, 0.95), seed=seed + 1)
        ctx.restore()

    def render(self, local_f: int, global_f: int) -> Tuple[np.ndarray, OpticsProfile]:
        w, h = self.w, self.h
        # 0..135: Orchard scene — Newton studies beneath tree; apple detaches at local_f=86, falls naturally (g=9.81 m/s^2)
        # Landing occurs at local_f ~ 86 + 0.75*24 = 104 (t = 44.33s, synced to soft grass thud SFX!)
        if local_f < 136:
            img = self.orchard_bg.copy()
            exp_f = quantize_exposure(local_f, cadence=2)
            dyn = InkCanvas2x(w, h)
            # Newton seated quietly resting his back against the apple tree trunk (well away from apple landing spot!)
            draw_newton_figure(
                dyn, w * 0.295, h * 0.88, height_px=h * 0.36, age="young", action="sit_right",
                phase=exp_f * 0.08, light_dir_2d=(-0.75, -0.60), light_tint=(1.0, 0.96, 0.86), seed=525,
            )
            # Open leather-backed notebook resting naturally on his lap
            dyn.fill_cel_region([(w * 0.32, h * 0.72), (w * 0.375, h * 0.705), (w * 0.37, h * 0.755), (w * 0.315, h * 0.765)],
                                (0.90, 0.86, 0.76), (0.66, 0.62, 0.54), (-0.6, -0.7), outline_width=1.4, smooth=False, seed=526)

            # Drifting summer insects & swaying dappled leaves
            t_s = local_f / 24.0
            for k in range(14):
                ix = w * (0.20 + 0.65 * ((k * 0.17 + t_s * 0.03) % 1.0))
                iy = h * (0.42 + 0.32 * math.sin(t_s * 1.1 + k * 1.3) * 0.5)
                dyn.ctx.set_source_rgba(1.0, 0.94, 0.70, 0.58)
                dyn.ctx.arc(ix, iy, 2.2 * (w / 2048.0), 0, 2 * math.pi)
                dyn.ctx.fill()

            # Hero Apple physics: gentle stem sway before f=86, then exact g=9.81 m/s^2 fall onto grass at x=0.64
            t_fall = max(0.0, (local_f - 86) / 24.0)
            dx_m, y_m, rot, landed = simulate_falling_apple(t_fall, y_start_m=2.85, y_ground_m=0.08)
            # Map physical meters y_m in [0..2.85] to screen y in [0.85h .. 0.29h]
            ax = w * 0.64 + dx_m * (h * 0.18)
            ay = h * (0.855 - (y_m / 2.85) * 0.565)
            if t_fall == 0.0:
                sway_a = math.sin(t_s * 3.2) * 0.14
                ax += math.sin(t_s * 3.2) * 4.0
                rot = sway_a
            self._draw_apple(dyn, ax, ay, h * 0.024, rot, seed=530)

            img = over_premul(img, dyn.to_premul_linear_rgba())
            img = over_premul(img, self.orchard_fg)
            return img, self.prof_garden

        # 136..209: Hold on Newton looking thoughtfully at the fallen apple, then lifting his gaze to the Moon
        if local_f < 210:
            u = (local_f - 136) / 73.0
            exp_f = quantize_exposure(local_f, cadence=2)
            # Warm sunlit orchard sky & sunlit canopy backdrop behind his bust
            img = sample_overscan(self.orchard_bg, w, h, 0.18, -0.22, zoom=1.35)
            dyn = InkCanvas2x(w, h)
            # Slow contemplative head lift: looking down (u < 0.35) -> looking up toward daytime Moon (u > 0.45)
            lift = smootherstep((u - 0.28) / 0.42)
            view_mode = "looking_up_right" if lift > 0.52 else "profile_right"
            tilt = 12.0 * (1.0 - lift) - 10.0 * lift
            draw_newton_bust(
                dyn, w * 0.38, h * 0.52, scale=h * 0.34, age="young", view=view_mode,
                light_dir_2d=(-0.72, -0.65), light_tint=(1.0, 0.95, 0.85),
                blink=blink_envelope(local_f - 136, 70, 28), head_tilt_deg=tilt,
                gaze=(0.45, 0.45 - 0.90 * lift), breath_phase=exp_f * 0.10, seed=535,
            )
            return over_premul(img, dyn.to_premul_linear_rgba()), self.prof_garden

        # 210..311: Match cut — Apple's curved silhouette dissolves into curved Earth limb;
        # falling trajectory gradually bends into the Moon's orbital arc!
        u = (local_f - 210) / 101.0
        earth_blend = smootherstep(min(1.0, u / 0.32))
        cx_e, cy_e, r_e = w * 0.42, h * 1.48, h * 1.02
        surf_lit = self._apple_lit * (1.0 - earth_blend) + self._earth_lit * earth_blend
        img = self.cosmos_bg * (1.0 - self._e_body_m) + surf_lit * self._e_body_m
        img = img + self._e_atmo * np.array([0.24, 0.50, 0.92], np.float32) * (0.42 * earth_blend)

        # Consistent painted Moon moving along its orbital arc
        orbit_r = r_e + h * 0.32
        moon_ang = -math.pi * 0.62 + 0.36 * smootherstep(max(0.0, (u - 0.15) / 0.85))
        mx = cx_e + math.cos(moon_ang) * orbit_r
        my = cy_e + math.sin(moon_ang) * orbit_r
        img = paint_consistent_moon(img, mx, my, h * 0.052, sun_dir_2d=(-0.80, -0.35), alpha_scale=earth_blend)

        # Delicate fine ink-drawn curves showing falling apple arc morphing into curved orbital path
        dyn = InkCanvas2x(w, h)
        morph = smootherstep(max(0.0, (u - 0.22) / 0.70))
        # Draw 3 progressive hand-inked arcs from vertical-ish fall to orbital curve + inward acceleration ticks
        for k, frac in enumerate((0.25, 0.58, 1.0)):
            if morph < frac * 0.35:
                continue
            a_arc = min(1.0, (morph - frac * 0.35) / 0.35)
            ang0 = -math.pi * 0.58
            ang_span = 0.08 + 0.58 * frac * morph
            pts = []
            for i in range(28):
                s = i / 27.0
                ang = ang0 + ang_span * s
                drop = (1.0 - frac) * (s * s) * h * 0.34
                r_cur = orbit_r - drop
                if r_cur >= r_e - 2:
                    pts.append((cx_e + math.cos(ang) * r_cur, cy_e + math.sin(ang) * r_cur))
            if len(pts) >= 2:
                dyn.stroke_catmull_rom(pts, 2.2 * (w / 2048.0), (0.94, 0.86, 0.66, 0.72 * a_arc),
                                       taper=True, tremor=0.22, seed=550 + k)
        # Subtle inward radial acceleration arrows toward Earth's center (same force bends both!)
        if morph > 0.55:
            ar_a = smoothstep((morph - 0.55) / 0.30)
            for a_i in (-math.pi * 0.52, -math.pi * 0.42, -math.pi * 0.32):
                p_out = (cx_e + math.cos(a_i) * orbit_r, cy_e + math.sin(a_i) * orbit_r)
                p_in = (cx_e + math.cos(a_i) * (orbit_r - h * 0.075), cy_e + math.sin(a_i) * (orbit_r - h * 0.075))
                dyn.stroke_catmull_rom([p_out, p_in], 1.8 * (w / 2048.0), (0.96, 0.82, 0.50, 0.65 * ar_a), seed=560)

        return over_premul(img, dyn.to_premul_linear_rgba()), self.prof_orbit


# ===========================================================================
# SHOT 6 — The Mathematics of Change (Frames 1272 - 1535, 0:53 - 1:04)
# ===========================================================================

class Shot6Renderer:
    """Candlelit parchment curve with live tangent & narrowing Riemann strips ->
    Visual rhymes (stream, pendulum, turning wheel) -> Crossing out mistakes & beginning again.
    """

    def __init__(self, w: int, h: int) -> None:
        self.w, self.h = w, h
        self.prof_desk = OpticsProfile(halation_strength=0.28, halation_tint=(1.14, 0.82, 0.46), ca_px=0.0, vignette=0.28)
        self.prof_nat = OpticsProfile(halation_strength=0.22, halation_tint=(0.94, 0.96, 1.0), vignette=0.24)
        self._build_caches()

    def _build_caches(self) -> None:
        w, h = self.w, self.h
        p_uv = parchment_uv(h, w, base_rgb=(0.56, 0.46, 0.32), seed=601)[..., :3]
        self.parch_bg = apply_gouache_finish(p_uv, np.random.default_rng(602), mottle_strength=0.045, n_dabs=86,
                                             dab_strength=0.07, paper_tooth=0.022, shadow_bounce_rgb=(0.042, 0.024, 0.01))
        # Painted stream / millwheel dusk nature background for visual match-cut triptych
        sky = paint_sky_gradient(h, w, "#2C4256", "#5E7B8A", "#C4B496", horizon_y=0.58, seed=603)
        self.nature_bg = paint_rolling_hills(sky, [
            (0.55, 0.04, hex_to_linear("#3A523B"), 0.32, 604),
            (0.70, 0.03, hex_to_linear("#263D2B"), 0.12, 605),
        ])

    def render(self, local_f: int, global_f: int) -> Tuple[np.ndarray, OpticsProfile]:
        w, h = self.w, self.h
        # 0..131: Parchment desk — curve drawn, moving point with exact changing tangent, narrowing Riemann strips!
        if local_f < 132:
            u = local_f / 131.0
            img = self.parch_bg.copy()
            dyn = InkCanvas2x(w, h)
            x0, x1 = w * 0.14, w * 0.84
            y_base, y_span = h * 0.78, h * 0.56

            def to_px(xn: float, yn: float) -> Tuple[float, float]:
                return x0 + xn * (x1 - x0), y_base - yn * y_span

            # Hand-drawn axes
            dyn.stroke_catmull_rom([(x0 - 20, y_base), (x1 + 30, y_base)], 2.4 * (w / 2048.0), (0.16, 0.12, 0.09, 0.72), seed=610)
            dyn.stroke_catmull_rom([(x0, y_base + 20), (x0, h * 0.15)], 2.4 * (w / 2048.0), (0.16, 0.12, 0.09, 0.72), seed=611)

            # Narrowing Riemann accumulation strips (N = 6 -> 12 -> 24 -> 48 -> 96) in second half
            if u > 0.36:
                phase_r = (u - 0.36) / 0.64
                counts = [6, 12, 24, 48, 96]
                n_s = counts[min(len(counts) - 1, int(phase_r * len(counts)))]
                strips, _ = riemann_strips(n_s)
                for xl, dx, ym in strips:
                    p_bl = to_px(xl, 0.0)
                    p_tr = to_px(xl + dx, ym)
                    dyn.ctx.set_source_rgba(0.68, 0.44, 0.18, 0.24)
                    dyn.ctx.rectangle(p_bl[0], p_tr[1], max(1.0, p_tr[0] - p_bl[0]), p_bl[1] - p_tr[1])
                    dyn.ctx.fill()
                    dyn.ctx.set_source_rgba(0.24, 0.16, 0.10, 0.50 if n_s <= 24 else 0.28)
                    dyn.ctx.set_line_width(max(0.7, 1.6 * (w / 2048.0)))
                    dyn.ctx.rectangle(p_bl[0], p_tr[1], max(1.0, p_tr[0] - p_bl[0]), p_bl[1] - p_tr[1])
                    dyn.ctx.stroke()

            # Smooth analytical fluxion curve y(x)
            xs = np.linspace(0.04, 0.94, 48)
            curve_pts = [to_px(float(xv), fluxion_curve_and_tangent(float(xv))[0]) for xv in xs]
            dyn.stroke_catmull_rom(curve_pts, 3.6 * (w / 2048.0), (0.12, 0.08, 0.06, 0.95), tremor=0.22, seed=612)

            # Moving point along curve & exact analytical tangent dy/dx!
            xp = 0.10 + 0.78 * smootherstep(u)
            yp, dydx = fluxion_curve_and_tangent(xp)
            pt_c = to_px(xp, yp)
            dt_x = 0.18
            t_a = to_px(xp - dt_x, yp - dydx * dt_x)
            t_b = to_px(xp + dt_x, yp + dydx * dt_x)
            dyn.stroke_catmull_rom([t_a, t_b], 2.8 * (w / 2048.0), (0.58, 0.14, 0.10, 0.90), taper=True, seed=613)
            dyn.ctx.set_source_rgba(0.58, 0.14, 0.10, 0.96)
            dyn.ctx.arc(pt_c[0], pt_c[1], 7.0 * (w / 2048.0), 0, 2 * math.pi)
            dyn.ctx.fill()

            # Discreet historical nod on corner of sheet (Newton's fluxion dot & Leibniz's independent notation)
            dyn.draw_calligraphy("ẋ, ẏ", w * 0.22, h * 0.21, 28.0 * (w / 2048.0), (0.18, 0.13, 0.09, 0.65), italic=True)
            if u > 0.62:
                la = smoothstep((u - 0.62) / 0.25)
                dyn.draw_calligraphy("dx, ∫  —  Leipzig", w * 0.80, h * 0.90, 22.0 * (w / 2048.0),
                                     (0.24, 0.18, 0.13, 0.48 * la), italic=True)

            img = over_premul(img, dyn.to_premul_linear_rgba())
            mod = 1.0 + 0.04 * math.sin(global_f * 0.33)
            img = add_radial_light_glow(img, w * 0.88, h * 0.15, w * 0.45, (1.0, 0.60, 0.22), 0.34 * mod)
            return img, self.prof_desk

        # 132..203: Living visual rhymes — Flowing stream + swinging brass pendulum + turning mill wheel
        if local_f < 204:
            t = (local_f - 132) / 24.0
            img = self.nature_bg.copy()
            dyn = InkCanvas2x(w, h)
            # Turning wooden waterwheel on left-midground (x = 30%)
            wx, wy, wr = w * 0.30, h * 0.62, h * 0.22
            ang = t * 1.15
            ink = (0.11, 0.08, 0.06, 0.94)
            # Wheel rim
            ring_pts = [(wx + math.cos(a) * wr, wy + math.sin(a) * wr) for a in np.linspace(0, 2 * math.pi, 28, endpoint=False)]
            dyn.stroke_catmull_rom(ring_pts, 12.0 * (w / 2048.0), (0.28, 0.18, 0.10, 0.95), taper=False, closed=True, seed=620)
            for k in range(8):
                ak = ang + k * (math.pi * 0.25)
                dyn.stroke_catmull_rom([(wx, wy), (wx + math.cos(ak) * wr * 1.08, wy + math.sin(ak) * wr * 1.08)],
                                       7.0 * (w / 2048.0), (0.24, 0.15, 0.09, 0.95), taper=False, seed=621 + k)
            # Flowing stream streamlines curving past bottom of wheel
            for s_i in range(7):
                sy0 = h * (0.76 + s_i * 0.032)
                spts = [(w * (x_f), sy0 + math.sin(x_f * 8.0 - t * 3.8 + s_i) * 9.0) for x_f in np.linspace(0.0, 1.0, 22)]
                dyn.stroke_catmull_rom(spts, 3.2 * (w / 2048.0), (0.74, 0.84, 0.88, 0.56), taper=False, seed=630 + s_i)

            # Swinging brass pendulum suspended in foreground right (x = 70%), harmonically oscillating!
            px0, py0 = w * 0.72, h * 0.06
            plen = h * 0.58
            theta = 0.42 * math.sin(t * 2.55)
            bx = px0 + math.sin(theta) * plen
            by = py0 + math.cos(theta) * plen
            # Faint geometric ink arc tracing pendulum path
            arc_p = [(px0 + math.sin(a) * plen, py0 + math.cos(a) * plen) for a in np.linspace(-0.44, 0.44, 24)]
            dyn.stroke_catmull_rom(arc_p, 2.0 * (w / 2048.0), (0.94, 0.88, 0.72, 0.52), taper=True, seed=640)
            dyn.stroke_catmull_rom([(px0, py0), (bx, by)], 3.2 * (w / 2048.0), (0.68, 0.54, 0.26, 0.96), taper=False, seed=641)
            # Brass pendulum bob
            r_b = h * 0.038
            bob_pts = [(bx + math.cos(a) * r_b, by + math.sin(a) * r_b) for a in np.linspace(0, 2 * math.pi, 14, endpoint=False)]
            dyn.fill_cel_region(bob_pts, (0.82, 0.65, 0.30), (0.42, 0.30, 0.12), (-0.7, -0.6), outline_srgb=ink, outline_width=2.0, seed=642)
            return over_premul(img, dyn.to_premul_linear_rgba()), self.prof_nat

        # 204..263: Newton crossing out a mistaken ink draft and beginning again (labor & uncertainty!)
        u = (local_f - 204) / 59.0
        img = self.parch_bg.copy()
        dyn = InkCanvas2x(w, h)
        # Earlier tentative geometric curve & equations
        old_pts = [(w * (0.20 + 0.55 * s), h * (0.58 - 0.25 * math.sin(s * 4.5))) for s in np.linspace(0, 1, 20)]
        dyn.stroke_catmull_rom(old_pts, 2.8 * (w / 2048.0), (0.18, 0.13, 0.10, 0.72), seed=650)
        # Bold frustrated diagonal strike-through stroke drawn live by quill hand
        s_prog = smootherstep(min(1.0, u / 0.55))
        p_start = (w * 0.22, h * 0.62)
        p_end = (w * 0.22 + w * 0.52 * s_prog, h * 0.62 - h * 0.28 * s_prog)
        if s_prog > 0.02:
            dyn.stroke_catmull_rom([p_start, p_end], 4.5 * (w / 2048.0), (0.11, 0.08, 0.06, 0.92), tremor=0.5, seed=651)
        # Fresh clean curve beginning cleanly below after strike-through
        if u > 0.60:
            n_prog = smootherstep((u - 0.60) / 0.40)
            fresh = [(w * (0.22 + 0.45 * s * n_prog), h * (0.76 - 0.16 * (s * s))) for s in np.linspace(0, 1, 14)]
            dyn.stroke_catmull_rom(fresh, 3.0 * (w / 2048.0), (0.10, 0.07, 0.05, 0.94), seed=652)
            tip_pos = fresh[-1]
        else:
            tip_pos = p_end
        draw_closeup_hand(dyn, tip_pos[0] - w * 0.16, tip_pos[1] - h * 0.02, scale=h * 0.34, age="young",
                          pose="quill", angle_deg=-6.0, light_tint=(1.0, 0.88, 0.70), ink_stained=True, seed=653)
        img = over_premul(img, dyn.to_premul_linear_rgba())
        img = add_radial_light_glow(img, w * 0.86, h * 0.14, w * 0.40, (1.0, 0.58, 0.22), 0.30)
        return img, self.prof_desk


# ===========================================================================
# SHOT 7 — The Nature of Light (Frames 1536 - 1799, 1:04 - 1:15)
# ===========================================================================

class Shot7Renderer:
    """Darkened shuttered chamber -> Exact Snell's law flint prism dispersion ->
    Validated converging lens + inverted prism recombination -> Eye reflecting spectrum -> Abstract painted spectrum.
    """

    def __init__(self, w: int, h: int) -> None:
        self.w, self.h = w, h
        self.prof_prism = OpticsProfile(halation_strength=0.34, halation_tint=(1.02, 0.98, 1.06), ca_px=0.0, vignette=0.30)
        self.prof_abs = OpticsProfile(halation_strength=0.30, halation_tint=(1.0, 0.96, 1.0), ca_px=0.15, vignette=0.20)
        self._build_caches()

    def _build_caches(self) -> None:
        w, h = self.w, self.h
        # 1. Darkened 17th-century oak-shuttered room backdrop
        rng = np.random.default_rng(701)
        dark = np.full((h, w, 3), np.array([0.011, 0.010, 0.014], dtype=np.float32))
        self.dark_room = apply_gouache_finish(dark, rng, mottle_strength=0.05, n_dabs=80, dab_strength=0.07,
                                              shadow_bounce_rgb=(0.010, 0.008, 0.007))
        # Subtle wooden shutter on far left + plaster opposite wall on far right
        rc = InkCanvas2x(w, h)
        rc.fill_cel_region([(0, 0), (w * 0.09, 0), (w * 0.09, h), (0, h)],
                           (0.11, 0.08, 0.06), (0.05, 0.04, 0.03), (0.8, 0.0), outline_width=2.0, smooth=False, seed=702)
        rc.fill_cel_region([(w * 0.89, 0), (w, 0), (w, h), (w * 0.89, h)],
                           (0.15, 0.14, 0.13), (0.07, 0.07, 0.07), (-0.8, 0.0), outline_width=1.8, smooth=False, seed=703)
        self.dark_room = over_premul(self.dark_room, rc.to_premul_linear_rgba())

        # Precompute exact Snell's law single-prism rays (21 wavelengths across 405..685 nm)
        self.wls = np.linspace(405.0, 685.0, 21)
        self.p1_verts = [np.array([0.0, -0.14]), np.array([0.16, -0.14]), np.array([0.08, 0.14])]
        self.ray_orig = np.array([-0.55, -0.05])
        self.ray_dir = np.array([1.0, 0.12])
        self.single_rays = [trace_prism_ray(self.ray_orig, self.ray_dir, self.p1_verts, float(wl), x_stop=0.92)
                            for wl in self.wls]

        # Precompute exact validated recombination setup (Prism 1 -> Converging Lens -> Inverted Prism 2 -> White beam)
        p_mid = trace_prism_ray(self.ray_orig, self.ray_dir, self.p1_verts, 535.0, x_stop=0.45)
        self.lens_x = 0.45
        self.lens_y = float(p_mid.points[-1][1])
        d_g = p_mid.points[-1] - p_mid.points[-2]
        d_g /= np.linalg.norm(d_g)
        y_at_p2 = self.lens_y + (d_g[1] / d_g[0]) * (0.82 - 0.45)
        self.p2_verts = [
            np.array([0.82 - 0.08, y_at_p2 - 0.12]),
            np.array([0.82 + 0.08, y_at_p2 - 0.12]),
            np.array([0.82, y_at_p2 + 0.16]),
        ]
        self.recomb_rays = trace_prism_recombination(
            self.ray_orig, self.ray_dir, self.p1_verts,
            lens_x=self.lens_x, lens_y=self.lens_y, lens_focal=0.185,
            prism2_verts=self.p2_verts, x_screen=1.22, wavelengths_nm=self.wls,
        )

        # Abstract hand-painted spectral gouache landscape for the lyrical transcendence moment
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        xn = xx / max(1.0, float(w))
        yn = yy / max(1.0, float(h))
        rng_a = np.random.default_rng(710)
        flow = fbm_gouache(h, w, rng_a, ((120, 72), (44, 28), (18, 12)), (0.55, 0.30, 0.15))
        wl_map = np.clip(410.0 + (xn * 0.72 + yn * 0.24 + flow * 0.08) * 280.0, 400.0, 690.0)
        # Vectorized spectral brush painting
        abs_img = np.zeros((h, w, 3), dtype=np.float32)
        for i in range(0, 21):
            wl = 405.0 + i * 14.0
            wt = np.exp(-((wl_map - wl) / 24.0) ** 2).astype(np.float32)
            abs_img += wt[..., None] * wavelength_to_linear_rgb(wl).reshape(1, 1, 3)
        abs_img = abs_img / (np.max(abs_img) + 1e-5) * 0.72
        self.abstract_spec = apply_gouache_finish(abs_img, rng_a, mottle_strength=0.09, n_dabs=100, dab_strength=0.14,
                                                  paper_tooth=0.02, shadow_bounce_rgb=(0.02, 0.015, 0.03))

    def _phys_to_px(self, pt_2d: np.ndarray, x_min: float, x_max: float) -> Tuple[float, float]:
        u = (float(pt_2d[0]) - x_min) / (x_max - x_min)
        # Flip Y so +Y physical is upward on screen
        v = 0.50 - float(pt_2d[1]) * 1.12
        return self.w * (0.09 + 0.80 * u), self.h * v

    def _draw_beam_rays(self, base_linear: np.ndarray, rays, x_min: float, x_max: float, reveal: float,
                        dust_phase: float, is_recomb: bool = False) -> np.ndarray:
        h, w = base_linear.shape[:2]
        # Draw volumetric optical beams at half resolution then upscale cleanly (zero 21x full-res allocs!)
        bh, bw = max(32, h // 2), max(32, w // 2)
        sx, sy = bw / float(w), bh / float(h)
        beam_acc = np.zeros((bh, bw, 3), dtype=np.float32)
        layer = np.zeros((bh, bw), dtype=np.float32)

        # Incoming white sunbeam from shutter aperture to Prism 1 entry
        p0 = self._phys_to_px(rays[0].points[0], x_min, x_max)
        p1 = self._phys_to_px(rays[len(rays) // 2].points[1], x_min, x_max)
        p1_cur = (p0[0] + (p1[0] - p0[0]) * min(1.0, reveal * 2.4), p0[1] + (p1[1] - p0[1]) * min(1.0, reveal * 2.4))
        cv2.line(layer, (int(p0[0] * sx), int(p0[1] * sy)), (int(p1_cur[0] * sx), int(p1_cur[1] * sy)), 1.0,
                 thickness=max(2, int(round(6 * (w / 2048.0)))), lineType=cv2.LINE_AA)
        beam_acc += layer[..., None] * np.array([1.10, 1.03, 0.90], dtype=np.float32)

        # Dispersed (and optionally recombined) exact Snell-traced rays
        if reveal > 0.35:
            disp_u = smoothstep((reveal - 0.35) / 0.55)
            thick = max(2, int(round(4 * (w / 2048.0))))
            gain = 0.24 if not is_recomb else 0.22
            for r in rays:
                pts_px = [self._phys_to_px(p, x_min, x_max) for p in r.points]
                layer.fill(0.0)
                for seg_i in range(1, len(pts_px) - 1):
                    a = pts_px[seg_i]
                    b = pts_px[seg_i + 1]
                    if seg_i == len(pts_px) - 2:
                        b = (a[0] + (b[0] - a[0]) * disp_u, a[1] + (b[1] - a[1]) * disp_u)
                    cv2.line(layer, (int(a[0] * sx), int(a[1] * sy)), (int(b[0] * sx), int(b[1] * sy)),
                             1.0, thickness=thick, lineType=cv2.LINE_AA)
                beam_acc += layer[..., None] * (r.rgb.reshape(1, 1, 3) * gain)

        soft_h = cv2.GaussianBlur(beam_acc, (0, 0), sigmaX=max(1.5, 7.0 * (w / 2048.0)))
        combined_h = beam_acc * 0.88 + soft_h * 0.72
        combined = cv2.resize(combined_h, (w, h), interpolation=cv2.INTER_LINEAR)
        out = base_linear + combined

        # Floating sunlit dust motes illuminated ONLY inside the beam envelope!
        beam_lum_h = np.mean(soft_h, axis=-1)
        rng = np.random.default_rng(740)
        for k in range(44):
            mx = int(rng.uniform(w * 0.10, w * 0.88))
            my = int((rng.uniform(h * 0.25, h * 0.78) + math.sin(dust_phase + k) * 14.0) % h)
            hx, hy = int(mx * sx), int(my * sy)
            if 1 < hx < bw - 1 and 1 < hy < bh - 1 and beam_lum_h[hy, hx] > 0.04:
                cv2.circle(out, (mx, my), max(1, int(round(2.2 * (w / 2048.0)))),
                           (0.95, 0.92, 0.84), -1, lineType=cv2.LINE_AA)
        return out

    def _draw_glass_prism(self, canvas: InkCanvas2x, verts_phys, x_min: float, x_max: float) -> None:
        """Crisp straight-edged triangular flint glass prism with specular bevel (no Catmull-Rom bowing!)."""
        pts = [self._phys_to_px(v, x_min, x_max) for v in verts_phys]
        canvas.fill_smooth_poly(pts, (0.76, 0.88, 0.97, 0.20), smooth=False)
        ctx = canvas.ctx
        sc = self.w / 2048.0
        # Subtle inner glass reflection facet along left illuminated edge
        cx = sum(p[0] for p in pts) / 3.0
        cy = sum(p[1] for p in pts) / 3.0
        inner = [(p[0] * 0.88 + cx * 0.12, p[1] * 0.88 + cy * 0.12) for p in pts]
        ctx.set_source_rgba(0.95, 0.98, 1.0, 0.22)
        ctx.set_line_width(1.8 * sc)
        ctx.move_to(*inner[0])
        ctx.line_to(*inner[2])
        ctx.stroke()
        # Razor-crisp straight triangular glass outline
        ctx.set_source_rgba(0.88, 0.95, 1.0, 0.84)
        ctx.set_line_width(2.8 * sc)
        ctx.move_to(*pts[0])
        ctx.line_to(*pts[1])
        ctx.line_to(*pts[2])
        ctx.close_path()
        ctx.stroke()

    def render(self, local_f: int, global_f: int) -> Tuple[np.ndarray, OpticsProfile]:
        w, h = self.w, self.h
        t = local_f / 24.0
        # 0..104: Single prism dispersion — shutter closes, sunbeam enters triangular glass prism -> wall spectrum
        if local_f < 105:
            u = local_f / 104.0
            img = self.dark_room.copy()
            reveal = smootherstep(max(0.0, (u - 0.08) / 0.65))
            img = self._draw_beam_rays(img, self.single_rays, -0.55, 0.92, reveal, dust_phase=t * 1.2)
            # Draw crisp glass prism + aperture highlight
            c = InkCanvas2x(w, h)
            self._draw_glass_prism(c, self.p1_verts, -0.55, 0.92)
            # Shutter aperture bright pinhole on left
            ap = self._phys_to_px(self.ray_orig, -0.55, 0.92)
            c.ctx.set_source_rgba(1.0, 0.98, 0.90, 0.96)
            c.ctx.arc(ap[0], ap[1], 7.0 * (w / 2048.0), 0, 2 * math.pi)
            c.ctx.fill()
            return over_premul(img, c.to_premul_linear_rgba()), self.prof_prism

        # 105..170: Validated recombination optical setup (Prism 1 -> Converging Lens -> Inverted Prism 2 -> White!)
        if local_f < 171:
            u = (local_f - 105) / 65.0
            img = self.dark_room.copy()
            reveal = smootherstep(min(1.0, 0.38 + u * 0.78))
            img = self._draw_beam_rays(img, self.recomb_rays, -0.55, 1.22, reveal, dust_phase=t * 1.2, is_recomb=True)
            c = InkCanvas2x(w, h)
            self._draw_glass_prism(c, self.p1_verts, -0.55, 1.22)
            self._draw_glass_prism(c, self.p2_verts, -0.55, 1.22)
            # Biconvex converging glass lens at lens_x
            lt = self._phys_to_px(np.array([self.lens_x, self.lens_y + 0.19]), -0.55, 1.22)
            lb = self._phys_to_px(np.array([self.lens_x, self.lens_y - 0.19]), -0.55, 1.22)
            lw_px = 14.0 * (w / 2048.0)
            lens_poly = [lt, (lt[0] + lw_px, (lt[1] + lb[1]) * 0.5), lb, (lt[0] - lw_px, (lt[1] + lb[1]) * 0.5)]
            c.fill_smooth_poly(lens_poly, (0.78, 0.90, 0.98, 0.24))
            c.stroke_catmull_rom(lens_poly, 2.4 * (w / 2048.0), (0.88, 0.95, 1.0, 0.78), taper=False, closed=True, seed=755)
            # Recombined white beam patch on right screen
            if reveal > 0.78:
                p_end = self._phys_to_px(self.recomb_rays[len(self.recomb_rays) // 2].points[-1], -0.55, 1.22)
                img = add_radial_light_glow(img, p_end[0], p_end[1], w * 0.04, (1.05, 1.02, 0.96), 0.75 * smoothstep((reveal - 0.78) / 0.22))
            return over_premul(img, c.to_premul_linear_rgba()), self.prof_prism

        # 171..218: Close-up of Newton's eye reflecting the full physically ordered spectrum
        if local_f < 219:
            u = (local_f - 171) / 47.0
            bg = np.full((h, w, 3), np.array([0.045, 0.036, 0.032], dtype=np.float32))
            dyn = InkCanvas2x(w, h)
            draw_closeup_eye(
                dyn, w * 0.50, h * 0.50, scale=h * 0.58, age="young",
                gaze=(0.08 * math.sin(u * 2.5), -0.04), blink=blink_envelope(local_f - 171, 60, 34),
                spectrum_reflect=0.95, light_tint=(0.76, 0.72, 0.74), seed=760,
            )
            return over_premul(bg, dyn.to_premul_linear_rgba()), self.prof_prism

        # 219..263: Lyrical painted abstract landscape of flowing spectral color -> return quietly to dark room
        u = (local_f - 219) / 44.0
        drift = sample_overscan(self.abstract_spec, w, h, -0.25 + 0.50 * u, 0.08 * math.sin(u * 3.14), zoom=1.10)
        # Gentle return dissolve to dark room in final 12 frames
        ret = smoothstep((u - 0.72) / 0.28)
        out = drift * (1.0 - ret) + self.dark_room * ret
        return out, self.prof_abs


# ===========================================================================
# SHOT 8 — Beyond the Earth (Frames 1800 - 2135, 1:15 - 1:29)
# ===========================================================================

class Shot8Renderer:
    """1668 reflecting telescope construction & look -> Painted lunar landscape ->
    Earth & Moon -> Symbolic Velocity Verlet cannonball thought experiment -> Solar system orbits.
    """

    def __init__(self, w: int, h: int) -> None:
        self.w, self.h = w, h
        self.prof_work = OpticsProfile(halation_strength=0.25, halation_tint=(1.10, 0.86, 0.56), vignette=0.28)
        self.prof_space = OpticsProfile(halation_strength=0.32, halation_tint=(0.90, 0.95, 1.12), ca_px=0.25, vignette=0.24)
        self._build_caches()

    def _build_caches(self) -> None:
        w, h = self.w, self.h
        # 1. Workshop backdrop for 1668 reflecting telescope
        rng = np.random.default_rng(801)
        wb = np.full((h, w, 3), np.array([0.045, 0.036, 0.030], dtype=np.float32))
        wb = apply_gouache_finish(wb, rng, mottle_strength=0.05, n_dabs=84, dab_strength=0.08,
                                  shadow_bounce_rgb=(0.035, 0.025, 0.018))
        wc = InkCanvas2x(w, h)
        # Starry blue night sky through open casement window on upper right
        wc.fill_smooth_poly([(w * 0.58, h * 0.04), (w * 0.96, h * 0.04), (w * 0.96, h * 0.62), (w * 0.58, h * 0.62)],
                            (0.08, 0.13, 0.24, 0.96), False)
        # Workbench table
        wc.fill_cel_region([(0, h * 0.72), (w, h * 0.72), (w, h), (0, h)],
                           (0.24, 0.16, 0.10), (0.11, 0.07, 0.05), (0.6, -0.7), outline_width=2.0, seed=802)
        self.work_bg = over_premul(wb, wc.to_premul_linear_rgba())

        # 2. Vast hand-painted lunar crater landscape (horizon + shadowed mountains + irregular craters)
        self.cosmos_bg = paint_starfield_cosmos(h, w, seed=805)
        lunar = self.cosmos_bg.copy()
        lunar = paint_rolling_hills(lunar, [
            (0.56, 0.065, hex_to_linear("#78756E"), 0.08, 806),
            (0.66, 0.055, hex_to_linear("#5C5A55"), 0.04, 807),
            (0.78, 0.035, hex_to_linear("#454340"), 0.0, 808),
        ])
        lc = InkCanvas2x(w, h)
        # Irregular hand-drawn lunar craters with crisp sunlit inner rims & deep ink shadows
        for cx_f, cy_f, rx_f, ry_f, sd in (
            (0.26, 0.76, 0.12, 0.036, 810), (0.64, 0.84, 0.18, 0.052, 811),
            (0.84, 0.72, 0.09, 0.026, 812), (0.44, 0.92, 0.14, 0.040, 813),
        ):
            cx, cy, rx, ry = w * cx_f, h * cy_f, w * rx_f, h * ry_f
            rim = [(cx + math.cos(a) * rx, cy + math.sin(a) * ry) for a in np.linspace(0, 2 * math.pi, 18, endpoint=False)]
            lc.fill_cel_region(rim, (0.46, 0.44, 0.41), (0.16, 0.16, 0.17), (-0.85, -0.35),
                               rim_srgb=(0.84, 0.82, 0.76), outline_srgb=(0.12, 0.12, 0.13, 0.85),
                               outline_width=1.8, shadow_shift_px=rx * 0.22, seed=sd)
        self.lunar_bg = over_premul(lunar, lc.to_premul_linear_rgba())

        # 3. Precompute exact Velocity Verlet trajectories + cached Earth globe backdrop for Cannonball shot!
        self.cannon_trajs = newton_cannonball_trajectories(r_earth=1.0, mountain_h=0.14, mu=1.0)
        cx_c, cy_c, r_px_c = w * 0.50, h * 0.52, h * 0.23
        yy_c, xx_c = np.mgrid[0:h, 0:w].astype(np.float32)
        d_c = np.sqrt((xx_c - cx_c) ** 2 + (yy_c - cy_c) ** 2)
        em_c = np.clip((r_px_c - d_c) * 0.65, 0.0, 1.0)[..., None]
        nx_c, ny_c = (xx_c - cx_c) / r_px_c, (yy_c - cy_c) / r_px_c
        nz_c = np.sqrt(np.maximum(0.0, 1.0 - np.minimum(1.0, nx_c * nx_c + ny_c * ny_c)))
        lit_c = np.clip(-nx_c * 0.65 - ny_c * 0.40 + nz_c * 0.64, 0.08, 1.0)[..., None]
        earth_rgb_c = np.array([0.08, 0.22, 0.46], np.float32) * lit_c
        atmo_c = np.exp(-((np.abs(d_c - r_px_c) / (h * 0.015)) ** 1.6)) * (d_c >= r_px_c - 2.0)
        self.cannon_earth_bg = (
            self.cosmos_bg * (1.0 - em_c)
            + earth_rgb_c * em_c
            + atmo_c[..., None] * np.array([0.22, 0.48, 0.90], np.float32) * 0.35
        ).astype(np.float32)

        # 4. Precompute exact Velocity Verlet elliptical planetary orbits (Mercury, Venus, Earth, Mars, Jupiter, Saturn)
        # (name, semi_major_a, eccentricity_e, color_srgb, radius_px_scale, tilt_deg, phase0)
        self.planets_spec = [
            ("Mercury", 0.18, 0.205, (0.72, 0.68, 0.64), 4.5, 5.0, 0.4),
            ("Venus",   0.26, 0.007, (0.94, 0.84, 0.60), 7.0, 2.5, 2.1),
            ("Earth",   0.36, 0.017, (0.34, 0.62, 0.94), 7.8, 0.0, 4.0),
            ("Mars",    0.47, 0.093, (0.86, 0.42, 0.28), 6.2, 2.0, 1.2),
            ("Jupiter", 0.66, 0.049, (0.86, 0.72, 0.56), 14.5, 1.2, 5.2),
            ("Saturn",  0.86, 0.056, (0.90, 0.82, 0.60), 12.2, 2.2, 3.0),
        ]
        self.planet_orbits = []
        for name, a, ecc, col, r_sc, tilt, ph0 in self.planets_spec:
            # Perihelion launch for exact Keplerian ellipse via Velocity Verlet
            r_peri = a * (1.0 - ecc)
            v_peri = math.sqrt((1.0 / a) * (1.0 + ecc) / (1.0 - ecc))
            traj, _, _ = integrate_verlet_orbit(
                np.array([r_peri, 0.0]), np.array([0.0, v_peri]), mu=1.0, dt=0.006, steps=1600,
            )
            self.planet_orbits.append((name, traj, col, r_sc, ph0))

    def _draw_reflecting_telescope(self, canvas: InkCanvas2x, cx: float, cy: float, scale: float) -> None:
        """Plausible circa-1668 Newtonian reflecting telescope (ball mount, brass tube, side eyepiece, speculum mirror)."""
        S = scale
        ink = (0.10, 0.08, 0.07, 0.94)
        # Turned oak wooden pedestal & ball-and-socket cradle
        base = [(cx - S * 0.28, cy + S * 0.36), (cx + S * 0.28, cy + S * 0.36),
                (cx + S * 0.20, cy + S * 0.14), (cx - S * 0.20, cy + S * 0.14)]
        canvas.fill_cel_region(base, (0.38, 0.24, 0.14), (0.19, 0.11, 0.07), (-0.6, -0.7), outline_srgb=ink, outline_width=2.0, smooth=False, seed=820)
        ball = [(cx + math.cos(a) * S * 0.14, cy + S * 0.08 + math.sin(a) * S * 0.14) for a in np.linspace(0, 2 * math.pi, 14, endpoint=False)]
        canvas.fill_cel_region(ball, (0.44, 0.28, 0.16), (0.20, 0.12, 0.07), (-0.6, -0.7), outline_srgb=ink, outline_width=2.0, seed=821)

        # Tilted leather-wrapped & brass-ringed telescope tube pointing up-right toward night window
        ang = math.radians(-26.0)
        ca, sa = math.cos(ang), math.sin(ang)
        nx, ny = -sa, ca
        L0, L1, R = -S * 0.42, S * 0.56, S * 0.115
        tube = [
            (cx + ca * L0 + nx * R, cy + sa * L0 + ny * R),
            (cx + ca * L1 + nx * R, cy + sa * L1 + ny * R),
            (cx + ca * L1 - nx * R, cy + sa * L1 - ny * R),
            (cx + ca * L0 - nx * R, cy + sa * L0 - ny * R),
        ]
        canvas.fill_cel_region(tube, (0.58, 0.44, 0.22), (0.28, 0.19, 0.09), (-0.5, -0.8),
                               rim_srgb=(0.94, 0.82, 0.48), outline_srgb=ink, outline_width=2.2, smooth=False, seed=822)
        # Concave speculum mirror glint inside open aperture & side-mounted Newtonian eyepiece!
        ep_x = cx + ca * (L1 * 0.65) - nx * (R * 1.15)
        ep_y = cy + sa * (L1 * 0.65) - ny * (R * 1.15)
        ep_pts = [
            (ep_x - ca * S * 0.04, ep_y - sa * S * 0.04),
            (ep_x + ca * S * 0.04, ep_y + sa * S * 0.04),
            (ep_x + ca * S * 0.04 - nx * S * 0.12, ep_y + sa * S * 0.04 - ny * S * 0.12),
            (ep_x - ca * S * 0.04 - nx * S * 0.12, ep_y - sa * S * 0.04 - ny * S * 0.12),
        ]
        canvas.fill_cel_region(ep_pts, (0.82, 0.66, 0.30), (0.42, 0.30, 0.12), (-0.5, -0.8), outline_srgb=ink, outline_width=1.8, smooth=False, seed=823)

    def render(self, local_f: int, global_f: int) -> Tuple[np.ndarray, OpticsProfile]:
        w, h = self.w, self.h
        # 0..79: Newton adjusting & looking through his 1668 reflecting telescope
        if local_f < 80:
            u = local_f / 79.0
            exp_f = quantize_exposure(local_f, cadence=2)
            img = self.work_bg.copy()
            img = paint_consistent_moon(img, w * 0.84, h * 0.22, h * 0.055, (-0.75, -0.35), 0.95)
            dyn = InkCanvas2x(w, h)
            draw_newton_bust(
                dyn, w * 0.35, h * 0.46, scale=h * 0.26, age="young", view="profile_right",
                light_dir_2d=(0.75, -0.55), light_tint=(0.94, 0.90, 0.84),
                blink=blink_envelope(local_f, 70, 20), head_tilt_deg=-12.0 + 4.0 * smootherstep(u),
                gaze=(0.40, -0.15), breath_phase=exp_f * 0.10, seed=825,
            )
            self._draw_reflecting_telescope(dyn, w * 0.56, h * 0.56, scale=h * 0.46)
            dyn.draw_calligraphy("c. 1668", w * 0.12, h * 0.88, 28.0 * (w / 2048.0),
                                 (0.88, 0.82, 0.68, 0.78 * smoothstep(local_f / 18.0)), italic=True)
            img = over_premul(img, dyn.to_premul_linear_rgba())
            img = add_radial_light_glow(img, w * 0.16, h * 0.72, w * 0.34, (1.0, 0.62, 0.24), 0.28)
            return img, self.prof_work

        # 80..145: Vast hand-painted lunar crater landscape with distant Earth rising above ridge
        if local_f < 146:
            u = (local_f - 80) / 65.0
            img = sample_overscan(self.lunar_bg, w, h, -0.18 + 0.36 * smootherstep(u), 0.0, zoom=1.06)
            # Distant pale blue Earth hanging silently in the black lunar sky
            ex, ey, er = w * 0.30, h * 0.26, h * 0.072
            yy, xx = np.mgrid[int(ey - er - 10):int(ey + er + 10), int(ex - er - 10):int(ex + er + 10)].astype(np.float32)
            nx, ny = (xx - ex) / er, (yy - ey) / er
            r2 = nx * nx + ny * ny
            m = np.clip((1.0 - np.sqrt(np.maximum(0.0, r2))) * er * 0.7, 0.0, 1.0)
            nz = np.sqrt(np.maximum(0.0, 1.0 - np.minimum(1.0, r2)))
            lit = np.clip(-nx * 0.75 - ny * 0.25 + nz * 0.60, 0.0, 1.0)
            ecol = np.array([0.16, 0.42, 0.88], np.float32) * (lit ** 0.8)[..., None]
            sl_y = slice(int(ey - er - 10), int(ey + er + 10))
            sl_x = slice(int(ex - er - 10), int(ex + er + 10))
            img[sl_y, sl_x] = img[sl_y, sl_x] * (1.0 - m[..., None]) + ecol * m[..., None]
            return img, self.prof_space

        # 146..245: Explicitly symbolic visualization of Newton's Cannonball Thought Experiment (Velocity Verlet!)
        if local_f < 246:
            u = (local_f - 146) / 99.0
            img = self.cannon_earth_bg.copy()
            cx, cy = w * 0.50, h * 0.52
            r_px = h * 0.23

            dyn = InkCanvas2x(w, h)
            # Symbolic mountain peak atop Earth (explicitly stylized as Newton's De Mundi Systemate engraving!)
            mtn = [(cx - r_px * 0.14, cy - r_px * 0.96), (cx, cy - r_px * 1.15), (cx + r_px * 0.14, cy - r_px * 0.96)]
            dyn.fill_cel_region(mtn, (0.48, 0.42, 0.34), (0.22, 0.19, 0.16), (-0.7, -0.5),
                                outline_srgb=(0.88, 0.80, 0.62, 0.85), outline_width=1.8, smooth=False, seed=830)

            # Plot progressive Velocity Verlet trajectories (short fall -> long fall -> circular orbit -> ellipse)
            traj_styles = [
                (0.00, 0.28, (0.86, 0.72, 0.54, 0.78)),
                (0.22, 0.54, (0.92, 0.78, 0.52, 0.86)),
                (0.45, 0.85, (0.98, 0.88, 0.58, 0.96)),
                (0.66, 1.00, (0.72, 0.88, 1.00, 0.84)),
            ]
            for (name, traj, _), (t0, t1, col) in zip(self.cannon_trajs, traj_styles):
                if u < t0:
                    continue
                prog = smootherstep((u - t0) / max(0.05, t1 - t0))
                k_end = max(2, int(len(traj) * prog))
                step = max(1, k_end // 64)
                pts = [(cx + float(p[0]) * r_px, cy - float(p[1]) * r_px) for p in traj[:k_end:step]]
                pts.append((cx + float(traj[k_end - 1, 0]) * r_px, cy - float(traj[k_end - 1, 1]) * r_px))
                dyn.stroke_catmull_rom(pts, 2.6 * (w / 2048.0), col, taper=False, tremor=0.18, seed=835)
                # Glowing brass projectile tip
                tip = pts[-1]
                dyn.ctx.set_source_rgba(1.0, 0.94, 0.74, 0.98)
                dyn.ctx.arc(tip[0], tip[1], 5.5 * (w / 2048.0), 0, 2 * math.pi)
                dyn.ctx.fill()

            return over_premul(img, dyn.to_premul_linear_rgba()), self.prof_space

        # 246..335: Grand elliptical planetary orbits around the Sun (Velocity Verlet Keplerian system)
        u = (local_f - 246) / 89.0
        img = self.cosmos_bg.copy()
        sx, sy = w * 0.44, h * 0.52  # Off-axis Sun position for sweeping depth composition
        # Radiant Sun core + multi-layer golden corona
        img = add_radial_light_glow(img, sx, sy, h * 0.34, (1.05, 0.72, 0.30), 0.72, power=1.45)
        img = add_radial_light_glow(img, sx, sy, h * 0.07, (1.35, 1.18, 0.82), 1.05, power=2.0)

        dyn = InkCanvas2x(w, h)
        scale_x = w * 0.54 * (0.94 + 0.14 * smootherstep(u))
        scale_y = h * 0.30 * (0.94 + 0.14 * smootherstep(u))  # Inclined ecliptic foreshortening

        for idx, (name, traj, col, r_sc, ph0) in enumerate(self.planet_orbits):
            # Delicate golden-parchment elliptical orbital path
            pts = [(sx + float(p[0]) * scale_x, sy - float(p[1]) * scale_y) for p in traj[::20]]
            dyn.stroke_catmull_rom(pts, 1.5 * (w / 2048.0), (0.86, 0.80, 0.66, 0.36), taper=False, closed=True, seed=850 + idx)
            # Planet position advancing coherently along Keplerian orbit (inner planets faster!)
            speed = 1.0 / (0.25 + idx * 0.32)
            k_pos = int((ph0 * 180.0 + u * 520.0 * speed) % len(traj))
            px = sx + float(traj[k_pos, 0]) * scale_x
            py = sy - float(traj[k_pos, 1]) * scale_y
            pr = r_sc * (w / 2048.0) * 1.35
            # Saturn rings!
            if name == "Saturn":
                ring = [(px + math.cos(a) * pr * 2.2, py + math.sin(a) * pr * 0.72) for a in np.linspace(0, 2 * math.pi, 20, endpoint=False)]
                dyn.stroke_catmull_rom(ring, 2.8 * (w / 2048.0), (0.88, 0.80, 0.62, 0.76), taper=False, closed=True, seed=860)
            p_poly = [(px + math.cos(a) * pr, py + math.sin(a) * pr) for a in np.linspace(0, 2 * math.pi, 12, endpoint=False)]
            dyn.fill_cel_region(p_poly, col, (col[0] * 0.30, col[1] * 0.30, col[2] * 0.36),
                                (sx - px, sy - py), outline_srgb=(0.08, 0.08, 0.10, 0.65), outline_width=1.0, seed=870 + idx)

        return over_premul(img, dyn.to_premul_linear_rgba()), self.prof_space
