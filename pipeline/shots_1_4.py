"""Shots 1 to 4 (0:00 - 0:40, frames 0 - 959):
- Shot 1: A World Before Him (0:00-0:09, f 0-215)
- Shot 2: The Curious Child (0:09-0:19, f 216-455)
- Shot 3: Cambridge (0:19-0:30, f 456-719)
- Shot 4: The Silent Return (0:30-0:40, f 720-959)
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import cv2
import numpy as np

from .characters import (
    blink_envelope,
    draw_closeup_eye,
    draw_closeup_hand,
    draw_newton_bust,
    draw_newton_figure,
    draw_supporting_scholar,
    quantize_exposure,
)
from .core_3d import (
    Camera3D,
    composite_depth_premul,
    over_premul,
    parchment_uv,
    smootherstep,
    smoothstep,
    voronoi_masonry,
    warp_plane_premul,
    wood_planks_uv,
)
from .painter import InkCanvas2x, apply_gouache_finish, fbm_gouache, hex_to_linear, srgb_to_linear
from .physics_optics import candle_illumination, make_sun_dir, schlick_fresnel
from .post_optics import OpticsProfile, apply_disc_bokeh_premul
from .scene_kit import (
    add_radial_light_glow,
    draw_candle_and_flame,
    draw_leaded_window_frame,
    draw_painted_bookshelf,
    paint_clouds_layer,
    paint_rolling_hills,
    paint_sky_gradient,
    sample_overscan,
)


# ===========================================================================
# SHOT 1 — A World Before Him (Frames 0 - 215, 0:00 - 0:09)
# ===========================================================================

class Shot1Renderer:
    """Winter dawn English countryside -> Woolsthorpe Manor amber window -> Interior cradle & candle."""

    def __init__(self, w: int, h: int) -> None:
        self.w, self.h = w, h
        self.profile_ext = OpticsProfile(halation_strength=0.26, halation_tint=(1.08, 0.86, 0.66), vignette=0.26)
        self.profile_int = OpticsProfile(halation_strength=0.32, halation_tint=(1.15, 0.80, 0.45), ca_px=0.25, vignette=0.32)
        self._build_ext_cache()
        self._build_int_cache()

    def _build_ext_cache(self) -> None:
        ow, oh = int(self.w * 1.12), int(self.h * 1.12)
        # Pale blue winter dawn sky with low amber sunrise glow on right horizon (off-axis 64%)
        sky = paint_sky_gradient(
            oh, ow, "#17253C", "#4B688A", "#C4B8AA",
            sun_xy_norm=(0.70, 0.56), sun_glow_rgb=(0.55, 0.36, 0.22), sun_radius=0.30,
            horizon_y=0.64, seed=101,
        )
        clouds = paint_clouds_layer(oh, ow, (0.06, 0.46), (0.86, 0.80, 0.74), (0.34, 0.44, 0.58), 0.58, seed=102)
        sky = over_premul(sky, clouds)

        # Snow-covered rolling Lincolnshire hills
        hills = paint_rolling_hills(sky, [
            (0.56, 0.045, hex_to_linear("#8D9DB2"), 0.55, 103),
            (0.64, 0.055, hex_to_linear("#ABB8C8"), 0.38, 104),
            (0.73, 0.050, hex_to_linear("#C5D0DC"), 0.20, 105),
        ])

        # Metric 3D Woolsthorpe Manor nestled into the winter valley (off-axis right-midground)
        cam = Camera3D(eye=[0.0, -34.0, 3.2], target=[1.6, 0.0, 2.2], width=ow, height=oh, fov_deg=34.0,
                       principal_shift=(0.06, 0.0))
        fog_col = np.array(hex_to_linear("#9EB0C4"), dtype=np.float32)

        # Authentic English Lincolnshire limestone masonry (warm buff-grey stone, narrow mortar)
        front_uv = voronoi_masonry(320, 480, 9.6, 5.2, stone_w_m=0.56, stone_h_m=0.28, mortar_m=0.015,
                                   base_rgb=(0.24, 0.22, 0.19), mortar_rgb=(0.11, 0.10, 0.09), weathering=0.42, seed=106)
        side_uv = voronoi_masonry(320, 320, 6.0, 5.2, stone_w_m=0.56, stone_h_m=0.28, mortar_m=0.015,
                                  base_rgb=(0.15, 0.16, 0.18), mortar_rgb=(0.07, 0.07, 0.08), weathering=0.45, seed=107)
        roof_uv = voronoi_masonry(220, 480, 10.2, 4.0, stone_w_m=0.42, stone_h_m=0.26, mortar_m=0.012,
                                  base_rgb=(0.64, 0.70, 0.78), mortar_rgb=(0.38, 0.43, 0.50), weathering=0.22, seed=108)

        zbuf = np.full((oh, ow), 1e6, dtype=np.float32)
        p0_m = np.array([1.2, 0.0, -0.6])  # Foundation embedded cleanly into snow hill!
        e_front = np.array([1.0, 0.16, 0.0])
        e_front /= np.linalg.norm(e_front)
        e_side = np.array([-e_front[1], e_front[0], 0.0])
        e_up = np.array([0.0, 0.0, 1.0])

        p_front, z_f = warp_plane_premul(cam, front_uv, p0_m, e_front, e_up, 9.6, 5.2, fog_col, 16.0, 0.022)
        p_side, z_s = warp_plane_premul(cam, side_uv, p0_m, e_side, e_up, 6.0, 5.2, fog_col, 16.0, 0.022)
        # Roof slopes cleanly from front eave (z=4.55) up to central ridge (z=7.4, depth=3.0 along e_side)
        e_roof_slope = e_side * 3.15 + e_up * 2.95
        roof_len = float(np.linalg.norm(e_roof_slope))
        p_roof, z_r = warp_plane_premul(
            cam, roof_uv,
            p0_m - e_front * 0.35 - e_side * 0.25 + e_up * 4.95,
            e_front, e_roof_slope / roof_len, 10.3, roof_len, fog_col, 16.0, 0.022,
        )

        bg = hills
        bg, zbuf = composite_depth_premul(bg, zbuf, p_side, z_s)
        bg, zbuf = composite_depth_premul(bg, zbuf, p_front, z_f)
        bg, zbuf = composite_depth_premul(bg, zbuf, p_roof, z_r)

        # Compute exact 3D projections for gable triangle, eaves, stone windows, doorway & chimney
        gable_uv = cam.project_clipped_polygon([
            p0_m + e_up * 5.05,
            p0_m + e_side * 6.0 + e_up * 5.05,
            p0_m + e_side * 3.0 + e_up * 7.85,
        ])
        eave_uv, _ = cam.project(np.array([
            p0_m - e_front * 0.35 - e_side * 0.25 + e_up * 4.95,
            p0_m + e_front * 9.95 - e_side * 0.25 + e_up * 4.95,
        ]))
        chim_base = p0_m + e_front * 7.8 + e_side * 2.6 + e_up * 7.2
        chim_uv, _ = cam.project(np.array([
            chim_base, chim_base + e_front * 0.95,
            chim_base + e_front * 0.95 + e_up * 2.3, chim_base + e_up * 2.3,
        ]))
        self.chimney_top_ov = (float(np.mean(chim_uv[2:4, 0])), float(np.mean(chim_uv[2:4, 1])))

        def proj_win(u0: float, u1: float, z0: float, z1: float) -> np.ndarray:
            pts, _ = cam.project(np.array([
                p0_m + e_front * u0 + e_up * z0,
                p0_m + e_front * u1 + e_up * z0,
                p0_m + e_front * u1 + e_up * z1,
                p0_m + e_front * u0 + e_up * z1,
            ]))
            return pts

        w_amber = proj_win(4.2, 5.4, 2.7, 4.1)
        w_left = proj_win(1.5, 2.6, 2.7, 4.0)
        w_right = proj_win(7.0, 8.1, 2.7, 4.0)
        w_low_l = proj_win(1.5, 2.6, 0.9, 2.1)
        w_low_r = proj_win(7.0, 8.1, 0.9, 2.1)
        door_uv = proj_win(4.3, 5.3, 0.5, 2.2)
        self.amber_win_center = (float(np.mean(w_amber[:, 0])), float(np.mean(w_amber[:, 1])))

        # Hand-inked manor architectural accents + grounded snow drift + gnarled winter oaks
        ink = InkCanvas2x(ow, oh)
        # Triangular stone gable sealing side wall seamlessly to pitched roof!
        if len(gable_uv) >= 3:
            ink.fill_cel_region([tuple(p) for p in gable_uv], (0.28, 0.30, 0.34), (0.20, 0.22, 0.26), (-0.7, -0.5),
                                outline_srgb=(0.11, 0.11, 0.13, 0.85), outline_width=1.8, smooth=False, seed=109)
        # Crisp timber roof eave shadow line
        ink.stroke_catmull_rom([tuple(eave_uv[0]), tuple(eave_uv[1])], 5.5, (0.14, 0.13, 0.14, 0.90), taper=False, tremor=0.25)
        # Crisp rectangular stone chimney (smooth=False!)
        ink.fill_cel_region([tuple(p) for p in chim_uv], (0.34, 0.30, 0.27), (0.18, 0.17, 0.18), (-0.7, -0.5),
                            outline_srgb=(0.10, 0.09, 0.10, 0.92), outline_width=1.8, smooth=False, seed=110)

        # Windows with crisp stone lintels & sills + ONE glowing amber upstairs window!
        for w_pts, lit in ((w_left, False), (w_right, False), (w_low_l, False), (w_low_r, False), (w_amber, True)):
            pts = [tuple(p) for p in w_pts]
            if lit:
                ink.fill_smooth_poly(pts, (1.0, 0.78, 0.34, 0.98), smooth=False)
                cx_w, cy_w = np.mean(w_pts, axis=0)
                ink.stroke_catmull_rom([(cx_w, w_pts[0, 1]), (cx_w, w_pts[2, 1])], 2.0, (0.24, 0.14, 0.07, 0.92), taper=False)
                ink.stroke_catmull_rom([(w_pts[0, 0], cy_w), (w_pts[1, 0], cy_w)], 2.0, (0.24, 0.14, 0.07, 0.92), taper=False)
            else:
                ink.fill_smooth_poly(pts, (0.13, 0.17, 0.23, 0.92), smooth=False)
            # Crisp unrounded rectangular casing & stone sill
            ink.ctx.set_source_rgba(0.11, 0.10, 0.11, 0.88)
            ink.ctx.set_line_width(2.0)
            ink.ctx.move_to(*pts[0])
            for p in pts[1:]:
                ink.ctx.line_to(*p)
            ink.ctx.close_path()
            ink.ctx.stroke()
        # Dark oak manor front door
        ink.fill_smooth_poly([tuple(p) for p in door_uv], (0.15, 0.11, 0.09, 0.96), smooth=False)

        # Gentle rolling snowdrift hugging the manor foundation naturally (blends seamlessly into hillside)
        base_l, _ = cam.project(p0_m + e_side * 5.8 + e_up * 0.55)
        base_m, _ = cam.project(p0_m + e_up * 0.45)
        base_r, _ = cam.project(p0_m + e_front * 9.8 + e_up * 0.55)
        drift_pts = [
            (0.0, oh * 0.80),
            (ow * 0.25, oh * 0.77),
            (base_l[0] - ow * 0.06, base_l[1] + 10),
            (base_m[0], base_m[1] - 2),
            (base_r[0] + ow * 0.05, base_r[1] + 8),
            (ow, oh * 0.78),
            (ow, oh),
            (0.0, oh),
        ]
        ink.fill_smooth_poly(drift_pts, (0.76, 0.82, 0.89, 0.88))
        # Soft blue cast snow shadow right under the manor walls
        ink.stroke_catmull_rom([(base_l[0], base_l[1] + 5), (base_m[0], base_m[1] + 3), (base_r[0], base_r[1] + 5)],
                               11.0, (0.42, 0.50, 0.62, 0.36), taper=True, tremor=0.5)

        # Gnarled bare winter English oaks (depth_lim=5 with sweeping organic twigs)
        ink.draw_bezier_tree(ow * 0.17, oh * 0.88, oh * 0.62, ow * 0.020, seed=110, bark_srgb=(0.14, 0.13, 0.15), depth_lim=5)
        ink.draw_bezier_tree(ow * 0.30, oh * 0.78, oh * 0.34, ow * 0.010, seed=111, bark_srgb=(0.24, 0.26, 0.31), depth_lim=4)
        ink.draw_bezier_tree(ow * 0.89, oh * 0.78, oh * 0.30, ow * 0.009, seed=112, bark_srgb=(0.26, 0.28, 0.33), depth_lim=4)

        # Grounded snow-draped dry-stone field wall tucked into foreground meadow
        wall_top = [(-20, oh * 0.855), (ow * 0.25, oh * 0.825), (ow * 0.56, oh * 0.865), (ow + 20, oh * 0.835)]
        wall_bot = [(ow + 20, oh * 0.895), (ow * 0.56, oh * 0.925), (ow * 0.25, oh * 0.885), (-20, oh * 0.915)]
        ink.fill_cel_region(wall_top + wall_bot, (0.36, 0.38, 0.42), (0.22, 0.25, 0.30), (-0.4, -0.8),
                            outline_srgb=(0.18, 0.20, 0.24, 0.62), outline_width=1.4, seed=113)
        ink.stroke_catmull_rom(wall_top, 7.0, (0.90, 0.93, 0.97, 0.92), tremor=0.7, seed=114, taper=False)

        bg = over_premul(bg, ink.to_premul_linear_rgba())
        bg = add_radial_light_glow(bg, self.amber_win_center[0], self.amber_win_center[1], ow * 0.075,
                                   (1.0, 0.55, 0.16), strength=0.72)

        # Foreground bokeh winter oak branch silhouette on far left corner
        fg_c = InkCanvas2x(ow, oh)
        fg_c.draw_bezier_tree(ow * 0.01, oh * 1.05, oh * 0.72, ow * 0.036, seed=115, bark_srgb=(0.06, 0.07, 0.09), depth_lim=3)
        self.ext_fg = apply_disc_bokeh_premul(fg_c.to_premul_linear_rgba(), radius=8.2)
        self.ext_bg = bg
        self.ext_size = (ow, oh)

    def _build_int_cache(self) -> None:
        w, h = self.w, self.h
        rng = np.random.default_rng(120)
        # Warm candlelit plaster & timber interior wall
        cam = Camera3D(eye=[0.0, -3.2, 1.25], target=[0.25, 0.0, 0.95], width=w, height=h, fov_deg=40.0,
                       principal_shift=(0.06, 0.0))
        wall_uv = voronoi_masonry(320, 512, 5.2, 3.2, stone_w_m=0.65, stone_h_m=0.40, mortar_m=0.012,
                                  base_rgb=(0.14, 0.10, 0.07), mortar_rgb=(0.07, 0.05, 0.04), weathering=0.25, seed=121)
        floor_uv = wood_planks_uv(320, 480, n_planks=7, base_rgb=(0.13, 0.08, 0.045), seed=122)
        bg = np.full((h, w, 3), np.array([0.035, 0.022, 0.014], dtype=np.float32))
        zbuf = np.full((h, w), 1e6, dtype=np.float32)

        pw, zw = warp_plane_premul(cam, wall_uv, np.array([-2.6, 1.4, -0.2]), np.array([1.0, 0.0, 0.0]),
                                   np.array([0.0, 0.0, 1.0]), 5.2, 3.2)
        pf, zf = warp_plane_premul(cam, floor_uv, np.array([-2.6, -2.5, 0.0]), np.array([1.0, 0.0, 0.0]),
                                   np.array([0.0, 1.0, 0.0]), 5.2, 4.2)
        bg, zbuf = composite_depth_premul(bg, zbuf, pw, zw)
        bg, zbuf = composite_depth_premul(bg, zbuf, pf, zf)
        bg = apply_gouache_finish(bg, rng, mottle_strength=0.05, n_dabs=88, dab_strength=0.08,
                                  shadow_bounce_rgb=(0.048, 0.026, 0.012))

        # Cold winter blue frost window at upper left contrasting warm interior
        win_c = InkCanvas2x(w, h)
        wx, wy, ww, wh = w * 0.08, h * 0.09, w * 0.22, h * 0.36
        win_c.fill_smooth_poly([(wx, wy), (wx + ww, wy), (wx + ww, wy + wh), (wx, wy + wh)], (0.36, 0.50, 0.68, 0.92), False)
        draw_leaded_window_frame(win_c, wx, wy, ww, wh, 3, 4, (0.10, 0.07, 0.05))

        # Hand-crafted oak wooden cradle & swaddled newborn Isaac on right-of-center (x ~ 57%)
        cx, cy = w * 0.56, h * 0.69
        cw, ch = w * 0.30, h * 0.24
        ink = (0.10, 0.07, 0.05, 0.94)
        # Cradle rockers
        win_c.stroke_catmull_rom(
            [(cx - cw * 0.55, cy + ch * 0.58), (cx, cy + ch * 0.72), (cx + cw * 0.55, cy + ch * 0.58)],
            base_width=14.0, color_srgb=(0.20, 0.12, 0.07, 0.96), tremor=0.4, seed=123, taper=False,
        )
        # Cradle body
        cradle = [
            (cx - cw * 0.54, cy - ch * 0.32),
            (cx + cw * 0.52, cy - ch * 0.26),
            (cx + cw * 0.44, cy + ch * 0.52),
            (cx - cw * 0.46, cy + ch * 0.52),
        ]
        win_c.fill_cel_region(cradle, (0.36, 0.22, 0.12), (0.18, 0.10, 0.06), (0.7, -0.6),
                              outline_srgb=ink, outline_width=2.6, shadow_shift_px=16.0, seed=124)
        # Hood over baby's head
        hood = [
            (cx - cw * 0.56, cy - ch * 0.30),
            (cx - cw * 0.44, cy - ch * 0.92),
            (cx - cw * 0.08, cy - ch * 0.90),
            (cx - cw * 0.16, cy - ch * 0.28),
        ]
        win_c.fill_cel_region(hood, (0.33, 0.20, 0.11), (0.16, 0.09, 0.05), (0.7, -0.5),
                              outline_srgb=ink, outline_width=2.4, seed=125)
        # Soft linen blanket & tiny sleeping newborn face nestled peacefully
        blanket = [
            (cx - cw * 0.26, cy - ch * 0.32),
            (cx + cw * 0.38, cy - ch * 0.28),
            (cx + cw * 0.34, cy - ch * 0.04),
            (cx - cw * 0.30, cy - ch * 0.04),
        ]
        win_c.fill_cel_region(blanket, (0.88, 0.80, 0.66), (0.58, 0.46, 0.36), (0.7, -0.7),
                              outline_srgb=ink, outline_width=2.0, seed=126)
        # Newborn head & peaceful closed eyes
        bx, by, br = cx - cw * 0.22, cy - ch * 0.36, w * 0.032
        baby_head = [
            (bx - br, by - br * 0.7), (bx + br * 0.9, by - br * 0.8),
            (bx + br * 1.05, by + br * 0.5), (bx - br * 0.8, by + br * 0.65),
        ]
        win_c.fill_cel_region(baby_head, (0.94, 0.79, 0.65), (0.68, 0.50, 0.39), (0.8, -0.5),
                              outline_srgb=ink, outline_width=1.8, seed=127)
        win_c.stroke_catmull_rom([(bx - br * 0.35, by - br * 0.05), (bx - br * 0.05, by + br * 0.04)], 1.6, ink, seed=128)
        win_c.stroke_catmull_rom([(bx + br * 0.25, by - br * 0.08), (bx + br * 0.52, by + br * 0.01)], 1.6, ink, seed=129)

        # Wooden stool on right holding candle
        sx, sy = w * 0.81, h * 0.62
        stool = [(sx - w * 0.07, sy), (sx + w * 0.07, sy), (sx + w * 0.06, h * 0.96), (sx - w * 0.06, h * 0.96)]
        win_c.fill_cel_region(stool, (0.26, 0.16, 0.09), (0.12, 0.07, 0.04), (0.4, -0.8), outline_srgb=ink, outline_width=2.0, seed=130)

        self.int_bg = over_premul(bg, win_c.to_premul_linear_rgba())
        self.candle_base = (sx, sy)

    def render(self, local_f: int, global_f: int) -> Tuple[np.ndarray, OpticsProfile]:
        # Frames 0..124: Winter dawn exterior + rising smoke + snow drift (fade in from black 0..34)
        # Frames 125..215: Warm candlelit interior cradle + discreet ink lettering "Woolsthorpe, England, 1642"
        if local_f < 125:
            u = local_f / 124.0
            ow, oh = self.ext_size
            # Gentle push-in toward Woolsthorpe Manor's amber window
            zoom = 1.0 + 0.09 * smootherstep(u)
            dx = 0.25 * smootherstep(u)
            dy = -0.04 * smootherstep(u)
            img = sample_overscan(self.ext_bg, self.w, self.h, dx, dy, zoom)

            # Animated chimney smoke & delicate winter snow flurries
            dyn = InkCanvas2x(self.w, self.h)
            t = local_f / 24.0
            sx0 = (self.chimney_top_ov[0] / ow) * self.w
            sy0 = (self.chimney_top_ov[1] / oh) * self.h
            import cairo as _cairo
            for k in range(10):
                age = (t * 0.35 + k * 0.11) % 1.0
                px = sx0 + age * self.w * 0.075 + math.sin(age * 6.0 + k) * 10.0
                py = sy0 - age * self.h * 0.16
                rad = (9.0 + age * 34.0) * (self.w / 2048.0)
                a_sm = math.sin(age * math.pi) * 0.22
                grad = _cairo.RadialGradient(px, py, rad * 0.1, px, py, rad)
                grad.add_color_stop_rgba(0.0, 0.86, 0.88, 0.92, a_sm)
                grad.add_color_stop_rgba(1.0, 0.86, 0.88, 0.92, 0.0)
                dyn.ctx.set_source(grad)
                dyn.ctx.arc(px, py, rad, 0, 2 * math.pi)
                dyn.ctx.fill()
            # Gentle sparse snow drifting in winter wind
            rng = np.random.default_rng(199)
            for k in range(55):
                x0 = rng.uniform(0.0, self.w)
                y0 = rng.uniform(0.0, self.h)
                spd = rng.uniform(0.4, 1.0)
                fx = (x0 + t * 55.0 * spd + math.sin(t * 1.8 + k) * 14.0) % self.w
                fy = (y0 + t * 34.0 * spd) % self.h
                r_f = rng.uniform(1.1, 2.6) * (self.w / 2048.0)
                dyn.ctx.set_source_rgba(0.92, 0.95, 0.98, 0.42 * spd)
                dyn.ctx.arc(fx, fy, r_f, 0, 2 * math.pi)
                dyn.ctx.fill()

            img = over_premul(img, dyn.to_premul_linear_rgba())
            fg = sample_overscan(self.ext_fg, self.w, self.h, dx * 1.45, dy * 1.4, zoom * 1.04)
            img = over_premul(img, fg)

            # Fade from black over first 34 frames
            fade_in = smoothstep(local_f / 34.0)
            return img * fade_in, self.profile_ext

        # Interior cradle scene (frames 125..215)
        t_in = (local_f - 125) / 90.0
        img = self.int_bg.copy()
        dyn = InkCanvas2x(self.w, self.h)
        fx, fy = draw_candle_and_flame(dyn, self.candle_base[0], self.candle_base[1],
                                       self.h * 0.14, self.w * 0.022, global_f, seed=131)
        # Discreet historical ink lettering: Woolsthorpe, England, 1642
        title_a = smoothstep((local_f - 142) / 22.0) * (1.0 - smoothstep((local_f - 196) / 18.0))
        if title_a > 0.01:
            dyn.draw_calligraphy(
                "Woolsthorpe, England, 1642",
                self.w * 0.25, self.h * 0.85,
                size_px=34.0 * (self.w / 2048.0),
                color_srgb=(0.92, 0.84, 0.68, 0.88 * title_a),
                italic=True,
            )
        img = over_premul(img, dyn.to_premul_linear_rgba())
        # Temporally coherent warm candle illumination
        mod = 1.0 + 0.045 * math.sin(global_f * 0.31) + 0.02 * math.sin(global_f * 0.67)
        img = add_radial_light_glow(img, fx, fy, self.w * 0.38, (1.0, 0.56, 0.20), strength=0.46 * mod, power=1.55)
        img = add_radial_light_glow(img, fx, fy, self.w * 0.06, (1.0, 0.85, 0.52), strength=0.68 * mod, power=2.0)
        return img, self.profile_int


# ===========================================================================
# SHOT 2 — The Curious Child (Frames 216 - 455, 0:09 - 0:19)
# ===========================================================================

class Shot2Renderer:
    """Young Isaac at rainy window building wooden windmill -> Full countryside mill -> Merging raindrops & eye."""

    def __init__(self, w: int, h: int) -> None:
        self.w, self.h = w, h
        self.prof_room = OpticsProfile(halation_strength=0.22, halation_tint=(0.82, 0.90, 1.02), vignette=0.25)
        self.prof_mill = OpticsProfile(halation_strength=0.20, halation_tint=(0.86, 0.92, 0.98), vignette=0.24)
        self.prof_eye = OpticsProfile(halation_strength=0.24, halation_tint=(0.86, 0.93, 1.02), ca_px=0.2, vignette=0.28)
        self._build_caches()

    def _build_caches(self) -> None:
        w, h = self.w, self.h
        # 1. Farmhouse window room backdrop
        sky = paint_sky_gradient(h, w, "#485C6E", "#7B8E9B", "#B8C0C2", horizon_y=0.72, seed=201)
        hills = paint_rolling_hills(sky, [
            (0.54, 0.04, hex_to_linear("#4E634D"), 0.45, 202),
            (0.66, 0.045, hex_to_linear("#3C523B"), 0.25, 203),
        ])
        # Interior plaster wall & oak sill framing window on right (55%..95%)
        wall_c = InkCanvas2x(w, h)
        ink = (0.10, 0.08, 0.07, 0.92)
        # Left interior wall
        wall_c.fill_cel_region([(0, 0), (w * 0.44, 0), (w * 0.44, h), (0, h)],
                               (0.24, 0.21, 0.17), (0.13, 0.11, 0.09), (0.8, -0.3), outline_srgb=ink, outline_width=0.0, seed=204)
        wall_c.fill_cel_region([(w * 0.92, 0), (w, 0), (w, h), (w * 0.92, h)],
                               (0.18, 0.16, 0.13), (0.10, 0.09, 0.08), (0.8, -0.3), outline_srgb=ink, outline_width=0.0, seed=205)
        draw_leaded_window_frame(wall_c, w * 0.43, h * 0.04, w * 0.50, h * 0.66, 4, 4, (0.14, 0.11, 0.09))
        # Oak workbench table at bottom
        desk_uv = wood_planks_uv(240, 640, n_planks=5, base_rgb=(0.22, 0.14, 0.08), seed=206)
        cam = Camera3D([0.0, -2.2, 1.35], [0.0, 0.0, 0.72], w, h, 44.0)
        desk_p, _ = warp_plane_premul(cam, desk_uv, np.array([-2.2, -1.2, 0.62]), np.array([1.0, 0.0, 0.0]),
                                      np.array([0.0, 1.0, 0.0]), 4.4, 2.2)
        bg = over_premul(hills, wall_c.to_premul_linear_rgba())
        bg = over_premul(bg, desk_p)

        # Handmade sketches, wooden ruler & brass dividers resting on workbench
        props = InkCanvas2x(w, h)
        # Sketch sheet on table
        sheet = [(w * 0.07, h * 0.78), (w * 0.30, h * 0.74), (w * 0.34, h * 0.94), (w * 0.09, h * 0.98)]
        props.fill_cel_region(sheet, (0.84, 0.78, 0.65), (0.62, 0.56, 0.46), (0.7, -0.6), outline_srgb=ink, outline_width=1.6, seed=207)
        # Ink sketch of windmill wheel on sheet
        props.stroke_catmull_rom([(w * 0.16, h * 0.85), (w * 0.24, h * 0.81)], 1.4, (0.24, 0.18, 0.14, 0.75), seed=208)
        props.stroke_catmull_rom([(w * 0.19, h * 0.80), (w * 0.21, h * 0.90)], 1.4, (0.24, 0.18, 0.14, 0.75), seed=209)
        # Wooden calipers & ruler
        props.stroke_catmull_rom([(w * 0.74, h * 0.82), (w * 0.93, h * 0.88)], 9.0, (0.48, 0.34, 0.18, 0.95), taper=False, seed=210)
        self.room_bg = over_premul(bg, props.to_premul_linear_rgba())

        # 2. Full-size English countryside windmill in rain-mist
        sky2 = paint_sky_gradient(h, w, "#3A4E62", "#6A7E8D", "#AEB9BD",
                                  sun_xy_norm=(0.24, 0.36), sun_glow_rgb=(0.26, 0.26, 0.24), sun_radius=0.35, seed=211)
        sky2 = over_premul(sky2, paint_clouds_layer(h, w, (0.04, 0.54), (0.78, 0.80, 0.80), (0.30, 0.38, 0.46), 0.72, seed=212))
        hills2 = paint_rolling_hills(sky2, [
            (0.55, 0.05, hex_to_linear("#4A5D4D"), 0.52, 213),
            (0.67, 0.06, hex_to_linear("#354A36"), 0.30, 214),
            (0.79, 0.04, hex_to_linear("#263A27"), 0.12, 215),
        ])
        # Painted stone post-mill tower off-axis at x = 62%
        mc = InkCanvas2x(w, h)
        mx, my = w * 0.62, h * 0.72
        mw, mh = w * 0.11, h * 0.36
        tower = [(mx - mw * 0.56, my), (mx - mw * 0.36, my - mh), (mx + mw * 0.36, my - mh), (mx + mw * 0.56, my)]
        mc.fill_cel_region(tower, (0.44, 0.41, 0.36), (0.24, 0.24, 0.23), (-0.7, -0.5), outline_srgb=ink, outline_width=2.2, seed=216)
        cap = [(mx - mw * 0.42, my - mh), (mx, my - mh * 1.24), (mx + mw * 0.42, my - mh)]
        mc.fill_cel_region(cap, (0.26, 0.21, 0.18), (0.14, 0.12, 0.12), (-0.7, -0.5), outline_srgb=ink, outline_width=2.0, seed=217)
        mc.draw_bezier_tree(w * 0.18, h * 0.84, h * 0.36, w * 0.012, seed=218, bark_srgb=(0.14, 0.12, 0.10),
                            foliage_srgb=(0.18, 0.28, 0.19), foliage_lit_srgb=(0.28, 0.38, 0.25), depth_lim=3)
        self.mill_bg = over_premul(hills2, mc.to_premul_linear_rgba())
        self.mill_hub = (mx, my - mh * 1.04)

        # 3. Eye close-up background (warm interior skin & dark bokeh shadow)
        rng3 = np.random.default_rng(220)
        eyebg = np.full((h, w, 3), np.array(hex_to_linear("#6F5746"), dtype=np.float32))
        self.eye_bg = apply_gouache_finish(eyebg, rng3, mottle_strength=0.04, n_dabs=75, dab_strength=0.06,
                                           shadow_bounce_rgb=(0.03, 0.02, 0.015))

    def _draw_windmill_sails(self, canvas: InkCanvas2x, cx: float, cy: float, radius: float, angle: float,
                             col_lit=(0.62, 0.48, 0.32), col_shd=(0.34, 0.25, 0.16)) -> None:
        ink = (0.11, 0.09, 0.08, 0.94)
        for k in range(4):
            a = angle + k * (math.pi * 0.5)
            ca, sa = math.cos(a), math.sin(a)
            nx, ny = -sa, ca
            # Wooden spar
            p0 = (cx, cy)
            p1 = (cx + ca * radius, cy + sa * radius)
            canvas.stroke_catmull_rom([p0, p1], max(2.2, radius * 0.038), (0.24, 0.16, 0.10, 0.96), taper=False, seed=230 + k)
            # Lattice sailcloth panel
            r_in = radius * 0.22
            sw = radius * 0.20
            sail = [
                (cx + ca * r_in, cy + sa * r_in),
                (cx + ca * radius * 0.96, cy + sa * radius * 0.96),
                (cx + ca * radius * 0.96 + nx * sw, cy + sa * radius * 0.96 + ny * sw),
                (cx + ca * r_in + nx * sw, cy + sa * r_in + ny * sw),
            ]
            canvas.fill_cel_region(sail, col_lit, col_shd, (-0.6, -0.7), outline_srgb=ink, outline_width=1.4, seed=240 + k)
        canvas.ctx.set_source_rgba(0.22, 0.15, 0.10, 1.0)
        canvas.ctx.arc(cx, cy, max(4.0, radius * 0.07), 0, 2 * math.pi)
        canvas.ctx.fill()

    def render(self, local_f: int, global_f: int) -> Tuple[np.ndarray, OpticsProfile]:
        w, h = self.w, self.h
        t = local_f / 24.0
        # 0..107: Child Isaac at rainy window turning miniature wooden windmill
        if local_f < 108:
            img = self.room_bg.copy()
            exp_f = quantize_exposure(local_f, cadence=2)
            dyn = InkCanvas2x(w, h)
            # Rain streaks outside window pane
            rng = np.random.default_rng(250 + local_f // 2)
            dyn.ctx.set_source_rgba(0.76, 0.84, 0.90, 0.30)
            dyn.ctx.set_line_width(1.5 * (w / 2048.0))
            for _ in range(42):
                rx = rng.uniform(w * 0.45, w * 0.90)
                ry = rng.uniform(h * 0.06, h * 0.66)
                dyn.ctx.move_to(rx, ry)
                dyn.ctx.line_to(rx - 7.0, ry + 36.0)
            dyn.ctx.stroke()

            # Young child Isaac seated left-center looking intently at miniature windmill in his hands
            draw_newton_bust(
                dyn, w * 0.34, h * 0.44, scale=h * 0.25, age="child", view="three_quarter_right",
                light_dir_2d=(0.75, -0.45), light_tint=(0.94, 0.96, 1.0),
                blink=blink_envelope(local_f, 76, 15), head_tilt_deg=4.5, gaze=(0.35, 0.35),
                breath_phase=exp_f * 0.11, seed=255,
            )
            # Miniature wooden windmill turning smoothly on table right in front of him
            mx, my = w * 0.56, h * 0.78
            # Small wooden base stand
            dyn.fill_cel_region([(mx - w * 0.04, my + h * 0.09), (mx - w * 0.025, my - h * 0.08),
                                 (mx + w * 0.025, my - h * 0.08), (mx + w * 0.04, my + h * 0.09)],
                                (0.50, 0.36, 0.22), (0.28, 0.19, 0.12), (0.6, -0.6), outline_width=1.8, seed=256)
            self._draw_windmill_sails(dyn, mx, my - h * 0.07, h * 0.145, t * 1.45,
                                      col_lit=(0.76, 0.64, 0.46), col_shd=(0.46, 0.37, 0.25))
            # Child's right hand gently spinning/testing the wooden mechanism
            draw_closeup_hand(dyn, w * 0.39, h * 0.80, scale=h * 0.19, age="child", pose="hold",
                              angle_deg=-10.0 + math.sin(exp_f * 0.22) * 4.0, light_dir_2d=(0.7, -0.5),
                              light_tint=(0.95, 0.96, 1.0), seed=257)
            return over_premul(img, dyn.to_premul_linear_rgba()), self.prof_room

        # 108..173: Cut to full-size countryside windmill turning in misty rain (matching sail angle!)
        if local_f < 174:
            img = self.mill_bg.copy()
            dyn = InkCanvas2x(w, h)
            # Match rotation phase smoothly from miniature windmill
            ang = t * 0.72 + 2.4
            self._draw_windmill_sails(dyn, self.mill_hub[0], self.mill_hub[1], h * 0.26, ang,
                                      col_lit=(0.74, 0.70, 0.62), col_shd=(0.40, 0.40, 0.39))
            # Rain falling softly across countryside
            rng = np.random.default_rng(270 + local_f)
            dyn.ctx.set_source_rgba(0.80, 0.86, 0.92, 0.28)
            dyn.ctx.set_line_width(1.4 * (w / 2048.0))
            for _ in range(95):
                rx = rng.uniform(0, w)
                ry = rng.uniform(0, h)
                dyn.ctx.move_to(rx, ry)
                dyn.ctx.line_to(rx - 10.0, ry + 48.0)
            dyn.ctx.stroke()
            return over_premul(img, dyn.to_premul_linear_rgba()), self.prof_mill

        # 174..239: Raindrop sliding down glass, merging with another, watched by young Isaac's eye
        u = (local_f - 174) / 65.0
        img = self.eye_bg.copy()
        dyn = InkCanvas2x(w, h)
        # Eye gaze tracks the descending droplet!
        drop_y1 = h * (0.20 + 0.56 * smootherstep(u))
        drop_y2 = h * 0.54
        merged = drop_y1 >= drop_y2 - h * 0.03
        gaze_y = (drop_y1 / h - 0.50) * 1.15
        draw_closeup_eye(
            dyn, w * 0.52, h * 0.50, scale=h * 0.56, age="child",
            gaze=(0.12, gaze_y), blink=blink_envelope(local_f - 174, 60, 42),
            rain_reflect=0.92, light_tint=(0.88, 0.92, 0.98), seed=280,
        )
        # Foreground window glass veil + glistening merging raindrops
        dx = w * 0.66
        # Wet droplet trail
        dyn.stroke_catmull_rom([(dx, h * 0.08), (dx + 4, drop_y1)], 4.0 * (w / 2048.0),
                               (0.82, 0.90, 0.98, 0.25), taper=False, seed=281)
        if not merged:
            for ry, rad in ((drop_y1, 14.0), (drop_y2, 16.0)):
                r_sc = rad * (w / 2048.0)
                dyn.ctx.set_source_rgba(0.75, 0.85, 0.96, 0.48)
                dyn.ctx.arc(dx, ry, r_sc, 0, 2 * math.pi)
                dyn.ctx.fill()
                dyn.ctx.set_source_rgba(0.98, 0.99, 1.0, 0.85)
                dyn.ctx.arc(dx - r_sc * 0.3, ry - r_sc * 0.35, r_sc * 0.26, 0, 2 * math.pi)
                dyn.ctx.fill()
        else:
            # Combined heavier droplet accelerates slightly downward
            r_sc = 22.0 * (w / 2048.0)
            y_comb = drop_y2 + (drop_y1 - drop_y2 + h * 0.03) * 1.35
            dyn.ctx.set_source_rgba(0.78, 0.88, 0.98, 0.55)
            dyn.ctx.arc(dx, y_comb, r_sc, 0, 2 * math.pi)
            dyn.ctx.fill()
            dyn.ctx.set_source_rgba(0.98, 0.99, 1.0, 0.90)
            dyn.ctx.arc(dx - r_sc * 0.3, y_comb - r_sc * 0.35, r_sc * 0.28, 0, 2 * math.pi)
            dyn.ctx.fill()
        return over_premul(img, dyn.to_premul_linear_rgba()), self.prof_eye


# ===========================================================================
# SHOT 3 — Cambridge (Frames 456 - 719, 0:19 - 0:30)
# ===========================================================================

class Shot3Renderer:
    """Trinity College 1661 weathered stone & wet Fresnel cobblestones -> Candlelit study & geometry montage."""

    def __init__(self, w: int, h: int) -> None:
        self.w, self.h = w, h
        self.prof_court = OpticsProfile(halation_strength=0.22, halation_tint=(0.92, 0.94, 1.0), vignette=0.26)
        self.prof_study = OpticsProfile(halation_strength=0.30, halation_tint=(1.14, 0.82, 0.48), ca_px=0.35, vignette=0.30)
        self._build_caches()

    def _build_caches(self) -> None:
        w, h = self.w, self.h
        # 1. Trinity College Great Court 1661 (Metric Camera3D, off-axis vanishing point at ~62% width)
        cam = Camera3D(eye=[-1.2, -16.0, 1.75], target=[2.6, 8.0, 2.8], width=w, height=h, fov_deg=40.0,
                       principal_shift=(0.08, 0.0))
        self.court_cam = cam
        fog_col = np.array(hex_to_linear("#96A4B0"), dtype=np.float32)

        sky = paint_sky_gradient(h, w, "#4F647B", "#889AA8", "#C8CED0", horizon_y=0.62, seed=301)
        sky = over_premul(sky, paint_clouds_layer(h, w, (0.04, 0.45), (0.88, 0.89, 0.88), (0.44, 0.50, 0.58), 0.60, seed=302))

        # Left Gothic stone college facade (Voronoi irregular ashlar limestone)
        wall_l_uv = voronoi_masonry(420, 720, 26.0, 9.5, stone_w_m=0.58, stone_h_m=0.32, mortar_m=0.020,
                                    base_rgb=(0.31, 0.28, 0.23), mortar_rgb=(0.12, 0.11, 0.10), weathering=0.45, seed=303)
        # Distant Great Gate back wall spanning full courtyard width + right enclosing cloister wall
        wall_b_uv = voronoi_masonry(360, 680, 28.0, 10.5, stone_w_m=0.60, stone_h_m=0.34, mortar_m=0.020,
                                    base_rgb=(0.27, 0.25, 0.21), mortar_rgb=(0.11, 0.10, 0.09), weathering=0.40, seed=304)
        wall_r_uv = voronoi_masonry(340, 560, 24.0, 8.2, stone_w_m=0.58, stone_h_m=0.32, mortar_m=0.020,
                                    base_rgb=(0.22, 0.21, 0.19), mortar_rgb=(0.09, 0.09, 0.08), weathering=0.44, seed=308)
        # Wet cobblestone courtyard floor spanning full width
        cobble_uv = voronoi_masonry(480, 640, 24.0, 30.0, stone_w_m=0.28, stone_h_m=0.22, mortar_m=0.024,
                                    base_rgb=(0.15, 0.16, 0.17), mortar_rgb=(0.06, 0.06, 0.07), weathering=0.50, seed=305)

        zbuf = np.full((h, w), 1e6, dtype=np.float32)
        bg = sky
        wl, zl = warp_plane_premul(cam, wall_l_uv, np.array([-4.5, -10.0, 0.0]), np.array([0.0, 1.0, 0.0]),
                                   np.array([0.0, 0.0, 1.0]), 26.0, 9.5, fog_col, 9.0, 0.032)
        wb, zb = warp_plane_premul(cam, wall_b_uv, np.array([-4.5, 16.0, 0.0]), np.array([1.0, 0.0, 0.0]),
                                   np.array([0.0, 0.0, 1.0]), 28.0, 10.5, fog_col, 9.0, 0.032)
        wr, zr = warp_plane_premul(cam, wall_r_uv, np.array([13.0, -8.0, 0.0]), np.array([0.0, 1.0, 0.0]),
                                   np.array([0.0, 0.0, 1.0]), 24.0, 8.2, fog_col, 9.0, 0.032)
        fl, zf = warp_plane_premul(cam, cobble_uv, np.array([-5.0, -14.0, 0.0]), np.array([1.0, 0.0, 0.0]),
                                   np.array([0.0, 1.0, 0.0]), 24.0, 30.0, fog_col, 9.0, 0.032)

        # Fresnel wet cobblestone puddles reflecting sky + Gothic facade (Stage 2 view-angle Fresnel!)
        yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        cos_th = np.clip((yy - 0.56) * 1.4, 0.05, 0.95)
        fres = schlick_fresnel(cos_th, f0=0.035)
        rng_p = np.random.default_rng(306)
        puddle_mask = np.clip((fbm_gouache(h, w, rng_p, ((95, 42), (32, 18)), (0.65, 0.35)) + 0.15) * 0.9, 0.0, 1.0)
        puddle_mask *= (fl[..., 3] > 0.5).astype(np.float32)
        refl_src = cv2.flip(over_premul(sky, wl), 0)
        refl_src = cv2.GaussianBlur(refl_src, (0, 0), sigmaX=2.2)
        refl_w = (fres * puddle_mask * 0.62)[..., None]
        fl_rgb = fl[..., :3] * (1.0 - refl_w) + refl_src * refl_w * fl[..., 3:4]
        fl_wet = np.concatenate([fl_rgb, fl[..., 3:4]], axis=-1)

        bg, zbuf = composite_depth_premul(bg, zbuf, wb, zb)
        bg, zbuf = composite_depth_premul(bg, zbuf, wr, zr)
        bg, zbuf = composite_depth_premul(bg, zbuf, wl, zl)
        bg, zbuf = composite_depth_premul(bg, zbuf, fl_wet, zf)

        # Draw crisp Gothic pointed lancet windows & archway projected directly via Camera3D (no bowed loops!)
        arch_c = InkCanvas2x(w, h)

        def _draw_crisp_gothic(pts3d: list, fill_col, stroke_col, lw: float = 2.4, mullion: bool = True) -> None:
            uv_poly = cam.project_clipped_polygon(pts3d)
            if len(uv_poly) < 3:
                return
            pts = [tuple(p) for p in uv_poly]
            arch_c.fill_smooth_poly(pts, fill_col, smooth=False)
            ctx = arch_c.ctx
            ctx.set_source_rgba(*stroke_col)
            ctx.set_line_width(lw)
            ctx.move_to(*pts[0])
            for p in pts[1:]:
                ctx.line_to(*p)
            ctx.close_path()
            ctx.stroke()
            # Exact 3D-projected vertical Gothic stone mullion & horizontal transom bar
            if mullion and len(pts3d) == 5:
                m_bot, _ = cam.project(0.5 * (pts3d[0] + pts3d[1]))
                m_top, _ = cam.project(pts3d[3])
                t_l, _ = cam.project(0.5 * (pts3d[0] + pts3d[4]))
                t_r, _ = cam.project(0.5 * (pts3d[1] + pts3d[2]))
                ctx.set_line_width(lw * 0.72)
                ctx.move_to(float(m_bot[0]), float(m_bot[1]))
                ctx.line_to(float(m_top[0]), float(m_top[1]))
                ctx.move_to(float(t_l[0]), float(t_l[1]))
                ctx.line_to(float(t_r[0]), float(t_r[1]))
                ctx.stroke()

        for y_pos in (-4.0, 0.5, 5.0, 9.5):
            w_pts3 = [
                np.array([-4.45, y_pos, 2.2]),
                np.array([-4.45, y_pos + 1.8, 2.2]),
                np.array([-4.45, y_pos + 1.8, 5.6]),
                np.array([-4.45, y_pos + 0.9, 6.7]),
                np.array([-4.45, y_pos, 5.6]),
            ]
            _draw_crisp_gothic(w_pts3, (0.12, 0.16, 0.22, 0.92), (0.14, 0.13, 0.12, 0.92), 2.5, True)

        # Distant Great Gate arched doorway + flanking lancet windows on back wall
        gate3 = [
            np.array([1.2, 15.9, 0.0]), np.array([5.4, 15.9, 0.0]),
            np.array([5.4, 15.9, 4.4]), np.array([3.3, 15.9, 5.8]), np.array([1.2, 15.9, 4.4]),
        ]
        _draw_crisp_gothic(gate3, (0.08, 0.09, 0.11, 0.94), (0.15, 0.14, 0.13, 0.90), 2.4, False)
        for bx_w in (-2.0, 7.8, 11.4):
            bw3 = [
                np.array([bx_w, 15.9, 2.8]), np.array([bx_w + 1.6, 15.9, 2.8]),
                np.array([bx_w + 1.6, 15.9, 6.0]), np.array([bx_w + 0.8, 15.9, 7.0]), np.array([bx_w, 15.9, 6.0]),
            ]
            _draw_crisp_gothic(bw3, (0.13, 0.16, 0.22, 0.88), (0.16, 0.15, 0.14, 0.85), 2.0, True)

        self.court_bg = over_premul(bg, arch_c.to_premul_linear_rgba())

        # 2. Candlelit Cambridge Study backdrop (individually painted books & oak desk)
        rng_s = np.random.default_rng(310)
        sbg = np.full((h, w, 3), np.array([0.042, 0.028, 0.018], dtype=np.float32))
        sbg = apply_gouache_finish(sbg, rng_s, mottle_strength=0.05, n_dabs=84, dab_strength=0.08,
                                   shadow_bounce_rgb=(0.045, 0.025, 0.012))
        sc = InkCanvas2x(w, h)
        draw_painted_bookshelf(sc, w * 0.04, h * 0.05, w * 0.44, h * 0.68, n_shelves=3, seed=311)
        # Oak study desk in foreground
        desk = [(0, h * 0.72), (w, h * 0.72), (w, h), (0, h)]
        sc.fill_cel_region(desk, (0.24, 0.15, 0.09), (0.11, 0.07, 0.04), (0.6, -0.7), outline_width=2.2, seed=312)
        # Open manuscript & parchment scroll on desk
        pg = [(w * 0.44, h * 0.75), (w * 0.84, h * 0.74), (w * 0.88, h * 0.96), (w * 0.40, h * 0.97)]
        sc.fill_cel_region(pg, (0.85, 0.76, 0.58), (0.60, 0.51, 0.37), (0.7, -0.6), outline_width=1.8, seed=313)
        self.study_bg = over_premul(sbg, sc.to_premul_linear_rgba())

        # 3. Close-up Parchment Desk for Quill Geometry Montage
        p_uv = parchment_uv(h, w, base_rgb=(0.58, 0.48, 0.33), seed=315)[..., :3]
        self.parch_bg = apply_gouache_finish(p_uv, np.random.default_rng(316), mottle_strength=0.04, n_dabs=80,
                                             dab_strength=0.06, paper_tooth=0.022, shadow_bounce_rgb=(0.04, 0.024, 0.01))

    def render(self, local_f: int, global_f: int) -> Tuple[np.ndarray, OpticsProfile]:
        w, h = self.w, self.h
        # 0..107: Trinity College 1661 courtyard — moving scholars & young Newton entering carrying books
        if local_f < 108:
            img = self.court_bg.copy()
            u = local_f / 107.0
            exp_f = quantize_exposure(local_f, cadence=2)
            dyn = InkCanvas2x(w, h)
            # Supporting scholars in distant & midground depth bands (individually designed)
            for idx, (wx0, wy0, vx, vy, sc_s) in enumerate([
                (4.2, 9.5, -0.4, -0.2, 320), (1.8, 6.2, 0.3, -0.35, 321), (-1.2, 4.5, 0.25, 0.2, 322),
            ]):
                pos = np.array([wx0 + vx * u * 3.2, wy0 + vy * u * 3.2, 0.0])
                uv_foot, zc = self.court_cam.project(pos)
                uv_head, _ = self.court_cam.project(pos + np.array([0.0, 0.0, 1.72]))
                fig_h = max(30.0, float(uv_foot[1] - uv_head[1]))
                draw_supporting_scholar(dyn, float(uv_foot[0]), float(uv_foot[1]), fig_h, seed=sc_s, phase=exp_f * 0.22)

            # Young Newton walking forward into courtyard carrying books
            n_pos = np.array([-0.9 + u * 2.1, -6.2 + u * 4.4, 0.0])
            n_foot, _ = self.court_cam.project(n_pos)
            n_head, _ = self.court_cam.project(n_pos + np.array([0.0, 0.0, 1.72]))
            nh = max(60.0, float(n_foot[1] - n_head[1]))
            draw_newton_figure(dyn, float(n_foot[0]), float(n_foot[1]), nh, age="young", action="walk_right_books",
                               phase=exp_f * 0.32, light_dir_2d=(-0.6, -0.7), light_tint=(0.94, 0.96, 1.0), seed=325)

            # Discreet ink date establishing Trinity College, Cambridge, 1661
            ta = smoothstep((local_f - 10) / 18.0) * (1.0 - smoothstep((local_f - 84) / 18.0))
            if ta > 0.01:
                dyn.draw_calligraphy("Trinity College, Cambridge, 1661", w * 0.72, h * 0.14,
                                     size_px=30.0 * (w / 2048.0), color_srgb=(0.14, 0.13, 0.12, 0.85 * ta), italic=True)
            return over_premul(img, dyn.to_premul_linear_rgba()), self.prof_court

        # 108..185: Candlelit Study — Newton leaning in deep concentration, melting candle
        if local_f < 186:
            u = (local_f - 108) / 77.0
            exp_f = quantize_exposure(local_f, cadence=2)
            img = self.study_bg.copy()
            dyn = InkCanvas2x(w, h)
            draw_newton_bust(
                dyn, w * 0.62, h * 0.44, scale=h * 0.27, age="young", view="three_quarter_left",
                light_dir_2d=(0.78, 0.35), light_tint=(1.0, 0.86, 0.66),
                blink=blink_envelope(local_f, 70, 18), head_tilt_deg=-6.0 - 4.0 * smootherstep(u),
                gaze=(-0.35, 0.45), breath_phase=exp_f * 0.10, seed=330,
            )
            # Candle gradually melting shorter to evoke passage of nights
            ch = h * (0.16 - 0.055 * u)
            fx, fy = draw_candle_and_flame(dyn, w * 0.86, h * 0.73, ch, w * 0.022, global_f, seed=331)
            img = over_premul(img, dyn.to_premul_linear_rgba())
            mod = 1.0 + 0.04 * math.sin(global_f * 0.35)
            img = add_radial_light_glow(img, fx, fy, w * 0.42, (1.0, 0.56, 0.20), strength=0.45 * mod, power=1.6)
            img = add_radial_light_glow(img, fx, fy, w * 0.065, (1.0, 0.88, 0.55), strength=0.65 * mod, power=2.0)
            return img, self.prof_study

        # 186..263: Ink-stained hand drawing elegant Euclidean/conic geometric construction on parchment
        u = (local_f - 186) / 77.0
        img = self.parch_bg.copy()
        dyn = InkCanvas2x(w, h)
        cx, cy = w * 0.46, h * 0.50
        rad = h * 0.30
        ink_col = (0.13, 0.09, 0.07, 0.90)

        # Circle & inscribed chord/tangent geometry drawn progressively by the quill nib!
        n_arc = max(2, int(44 * min(1.0, u * 1.45)))
        th = np.linspace(-math.pi * 0.75, -math.pi * 0.75 + 2.0 * math.pi * min(1.0, u * 1.45), n_arc)
        arc_pts = [(cx + math.cos(a) * rad, cy + math.sin(a) * rad) for a in th]
        dyn.stroke_catmull_rom(arc_pts, 3.0 * (w / 2048.0), ink_col, taper=False, tremor=0.35, seed=340)

        # Inscribed geometric chords & secant emerging after circle completes
        if u > 0.55:
            p2 = smoothstep((u - 0.55) / 0.40)
            p_a = (cx - rad * 0.86, cy + rad * 0.50)
            p_b = (cx + rad * 0.86, cy + rad * 0.50)
            p_c = (cx, cy - rad)
            dyn.stroke_catmull_rom([p_a, (p_a[0] + (p_c[0] - p_a[0]) * p2, p_a[1] + (p_c[1] - p_a[1]) * p2)],
                                   2.4 * (w / 2048.0), (0.18, 0.12, 0.09, 0.84), seed=341)
            dyn.stroke_catmull_rom([p_c, (p_c[0] + (p_b[0] - p_c[0]) * p2, p_c[1] + (p_b[1] - p_c[1]) * p2)],
                                   2.4 * (w / 2048.0), (0.18, 0.12, 0.09, 0.84), seed=342)
            dyn.stroke_catmull_rom([(cx - rad * 1.15, cy - rad), (cx + rad * 1.15 * p2, cy - rad)],
                                   2.0 * (w / 2048.0), (0.45, 0.16, 0.12, 0.75), seed=343)

        # Quill hand tracks active drawing tip smoothly
        tip_target = arc_pts[-1] if u <= 0.68 else (cx + rad * 0.86 * smoothstep((u - 0.68) / 0.32), cy + rad * 0.50)
        draw_closeup_hand(
            dyn, tip_target[0] - w * 0.17, tip_target[1] - h * 0.03, scale=h * 0.34,
            age="young", pose="quill", angle_deg=-4.0, light_dir_2d=(0.7, -0.5),
            light_tint=(1.0, 0.88, 0.70), ink_stained=True, seed=345,
        )
        img = over_premul(img, dyn.to_premul_linear_rgba())
        img = add_radial_light_glow(img, w * 0.86, h * 0.18, w * 0.45, (1.0, 0.64, 0.28), strength=0.32)
        return img, self.prof_study


# ===========================================================================
# SHOT 4 — The Silent Return (Frames 720 - 959, 0:30 - 0:40)
# ===========================================================================

class Shot4Renderer:
    """1665 empty Cambridge courtyard & closing door -> Match cut opening Woolsthorpe door -> Vast moorland walk."""

    def __init__(self, w: int, h: int) -> None:
        self.w, self.h = w, h
        self.prof_grey = OpticsProfile(halation_strength=0.14, halation_tint=(0.80, 0.84, 0.90), vignette=0.32)
        self._build_caches()

    def _build_caches(self) -> None:
        w, h = self.w, self.h
        # 1. Silent grey 1665 Cambridge stone doorway
        cam = Camera3D([0.0, -4.5, 1.4], [0.2, 0.0, 1.3], w, h, 38.0)
        stone = voronoi_masonry(420, 620, 6.2, 4.2, stone_w_m=0.48, stone_h_m=0.28, mortar_m=0.022,
                                base_rgb=(0.24, 0.24, 0.25), mortar_rgb=(0.09, 0.09, 0.10), weathering=0.55, seed=401)
        bg_c = np.full((h, w, 3), np.array([0.06, 0.065, 0.075], dtype=np.float32))
        wp, _ = warp_plane_premul(cam, stone, np.array([-3.0, 0.0, -0.4]), np.array([1.0, 0.0, 0.0]),
                                  np.array([0.0, 0.0, 1.0]), 6.2, 4.2)
        self.door_cam_bg = over_premul(bg_c, wp)

        # 2. Woolsthorpe farmhouse timber interior doorway
        timber = wood_planks_uv(420, 620, n_planks=9, base_rgb=(0.18, 0.13, 0.09), seed=402)
        wp2, _ = warp_plane_premul(cam, timber, np.array([-3.0, 0.0, -0.4]), np.array([1.0, 0.0, 0.0]),
                                   np.array([0.0, 0.0, 1.0]), 6.2, 4.2)
        self.door_farm_bg = over_premul(bg_c, wp2)

        # 3. Vast brooding English countryside beneath dark slate clouds
        ow, oh = int(w * 1.08), int(h * 1.08)
        sky = paint_sky_gradient(oh, ow, "#1E2630", "#45515E", "#8B959C", horizon_y=0.66, seed=405)
        sky = over_premul(sky, paint_clouds_layer(oh, ow, (0.04, 0.56), (0.64, 0.68, 0.72), (0.18, 0.22, 0.28), 0.78, seed=406))
        moor = paint_rolling_hills(sky, [
            (0.56, 0.045, hex_to_linear("#3C4742"), 0.52, 407),
            (0.67, 0.055, hex_to_linear("#2E3A30"), 0.30, 408),
            (0.79, 0.038, hex_to_linear("#222C22"), 0.12, 409),
        ])
        # Winding dirt path across the vast hills (making Newton feel small against nature)
        pc = InkCanvas2x(ow, oh)
        pc.fill_smooth_poly(
            [(ow * 0.0, oh * 0.90), (ow * 0.44, oh * 0.83), (ow * 0.85, oh * 0.78),
             (ow * 0.85, oh * 0.80), (ow * 0.45, oh * 0.88), (ow * 0.0, oh * 0.98)],
            (0.30, 0.27, 0.22, 0.75),
        )
        # Single wind-bent hawthorn tree on distant hill
        pc.draw_bezier_tree(ow * 0.78, oh * 0.75, oh * 0.16, ow * 0.005, seed=410, bark_srgb=(0.12, 0.13, 0.13), depth_lim=3)
        self.moor_bg = over_premul(moor, pc.to_premul_linear_rgba())

    def render(self, local_f: int, global_f: int) -> Tuple[np.ndarray, OpticsProfile]:
        w, h = self.w, self.h
        # 0..64: Empty Cambridge stone doorway closing shut (1665)
        if local_f < 65:
            u = smootherstep(local_f / 64.0)
            img = self.door_cam_bg.copy()
            c = InkCanvas2x(w, h)
            dx0, dy0, dw, dh = w * 0.36, h * 0.12, w * 0.28, h * 0.82
            # Dark interior recess behind doorway
            c.fill_smooth_poly([(dx0, dy0), (dx0 + dw, dy0), (dx0 + dw, dy0 + dh), (dx0, dy0 + dh)],
                               (0.04, 0.05, 0.06, 0.98), False)
            # Heavy oak door leaf swinging closed from left to right
            leaf_w = dw * (0.28 + 0.72 * u)
            c.fill_cel_region(
                [(dx0, dy0), (dx0 + leaf_w, dy0 + (1.0 - u) * h * 0.03),
                 (dx0 + leaf_w, dy0 + dh - (1.0 - u) * h * 0.03), (dx0, dy0 + dh)],
                (0.22, 0.16, 0.12), (0.11, 0.08, 0.06), (-0.7, -0.4), outline_width=2.6, smooth=False, seed=415,
            )
            c.draw_calligraphy("1665", w * 0.78, h * 0.86, 32.0 * (w / 2048.0),
                               (0.76, 0.78, 0.80, 0.78 * smoothstep(local_f / 20.0)), italic=True)
            return over_premul(img, c.to_premul_linear_rgba()), self.prof_grey

        # 65..119: Match cut — Woolsthorpe farmhouse door opening onto cool countryside light
        if local_f < 120:
            u = smootherstep((local_f - 65) / 54.0)
            img = self.door_farm_bg.copy()
            c = InkCanvas2x(w, h)
            dx0, dy0, dw, dh = w * 0.36, h * 0.12, w * 0.28, h * 0.82
            # Misty open countryside daylight revealed as door swings open
            c.fill_smooth_poly([(dx0, dy0), (dx0 + dw, dy0), (dx0 + dw, dy0 + dh), (dx0, dy0 + dh)],
                               (0.68, 0.74, 0.77, 0.98), False)
            leaf_w = dw * (1.0 - 0.76 * u)
            c.fill_cel_region(
                [(dx0, dy0), (dx0 + leaf_w, dy0 + u * h * 0.035),
                 (dx0 + leaf_w, dy0 + dh - u * h * 0.035), (dx0, dy0 + dh)],
                (0.28, 0.20, 0.13), (0.14, 0.10, 0.07), (0.7, -0.4), outline_width=2.6, smooth=False, seed=416,
            )
            img = over_premul(img, c.to_premul_linear_rgba())
            img = add_radial_light_glow(img, dx0 + dw * 0.65, dy0 + dh * 0.5, w * 0.24, (0.65, 0.72, 0.78), 0.26 * u)
            return img, self.prof_grey

        # 120..239: Solitary Newton walking across vast overcast English countryside
        u = (local_f - 120) / 119.0
        exp_f = quantize_exposure(local_f, cadence=2)
        img = sample_overscan(self.moor_bg, w, h, -0.35 + 0.70 * u, 0.0, zoom=1.03)
        dyn = InkCanvas2x(w, h)
        # Small solitary figure against immense rolling moorland
        nx = w * (0.28 + 0.30 * u)
        ny = h * (0.87 - 0.045 * u)
        draw_newton_figure(
            dyn, nx, ny, height_px=h * 0.135, age="young", action="walk_right",
            phase=exp_f * 0.28, light_dir_2d=(-0.5, -0.8), light_tint=(0.82, 0.86, 0.90), seed=420,
        )
        return over_premul(img, dyn.to_premul_linear_rgba()), self.prof_grey
