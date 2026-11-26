"""Stage 3: Historically informed character model sheets, shared facial landmarks,
pose-specific 2x PyCairo drawings, close-up hands/eyes, and exposure-sheet timing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cairo
import numpy as np

from .painter import InkCanvas2x


# ---------------------------------------------------------------------------
# Exposure-Sheet Cadence & Blink Timing
# ---------------------------------------------------------------------------

def quantize_exposure(frame_idx: int, cadence: int = 2, hold_frames: Sequence[Tuple[int, int]] = ()) -> int:
    """Traditional animation exposure sheet timing (twos, threes, and explicit holds)."""
    for start, end in hold_frames:
        if start <= frame_idx <= end:
            return start
    return (frame_idx // max(1, cadence)) * max(1, cadence)


def blink_envelope(frame_idx: int, period_frames: int = 86, offset: int = 23) -> float:
    """Return 0.0 (open) to 1.0 (closed) natural 4-frame hand-drawn eye blink."""
    m = (frame_idx + offset) % period_frames
    if m == 0:
        return 0.45
    if m in (1, 2):
        return 1.0
    if m == 3:
        return 0.35
    return 0.0


# ---------------------------------------------------------------------------
# Shared Landmark & Palette Definitions (Zero Identity Drift Across Ages)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgePalette:
    skin_lit: Tuple[float, float, float]
    skin_shadow: Tuple[float, float, float]
    hair_base: Tuple[float, float, float]
    hair_shadow: Tuple[float, float, float]
    hair_highlight: Tuple[float, float, float]
    coat_lit: Tuple[float, float, float]
    coat_shadow: Tuple[float, float, float]
    collar_lit: Tuple[float, float, float]
    collar_shadow: Tuple[float, float, float]
    ink: Tuple[float, float, float, float]
    eye_iris: Tuple[float, float, float]
    jaw_fullness: float
    brow_crease: float
    hair_length: float


NEWTON_AGES: Dict[str, AgePalette] = {
    "child": AgePalette(
        skin_lit=(0.89, 0.76, 0.65),
        skin_shadow=(0.66, 0.51, 0.43),
        hair_base=(0.24, 0.16, 0.11),
        hair_shadow=(0.12, 0.08, 0.06),
        hair_highlight=(0.38, 0.27, 0.19),
        coat_lit=(0.25, 0.28, 0.23),
        coat_shadow=(0.13, 0.15, 0.12),
        collar_lit=(0.86, 0.83, 0.76),
        collar_shadow=(0.62, 0.59, 0.54),
        ink=(0.12, 0.09, 0.08, 0.94),
        eye_iris=(0.32, 0.37, 0.34),
        jaw_fullness=0.92,
        brow_crease=0.0,
        hair_length=0.58,
    ),
    "young": AgePalette(
        skin_lit=(0.87, 0.75, 0.64),
        skin_shadow=(0.62, 0.49, 0.41),
        hair_base=(0.23, 0.15, 0.10),
        hair_shadow=(0.11, 0.07, 0.05),
        hair_highlight=(0.39, 0.27, 0.18),
        coat_lit=(0.17, 0.16, 0.18),
        coat_shadow=(0.08, 0.08, 0.10),
        collar_lit=(0.90, 0.87, 0.81),
        collar_shadow=(0.64, 0.61, 0.57),
        ink=(0.11, 0.08, 0.07, 0.94),
        eye_iris=(0.30, 0.36, 0.34),
        jaw_fullness=0.98,
        brow_crease=0.35,
        hair_length=1.08,
    ),
    "mature": AgePalette(
        skin_lit=(0.84, 0.72, 0.62),
        skin_shadow=(0.58, 0.46, 0.39),
        hair_base=(0.42, 0.38, 0.35),
        hair_shadow=(0.23, 0.20, 0.18),
        hair_highlight=(0.64, 0.60, 0.56),
        coat_lit=(0.22, 0.14, 0.15),
        coat_shadow=(0.11, 0.07, 0.08),
        collar_lit=(0.91, 0.88, 0.82),
        collar_shadow=(0.66, 0.63, 0.58),
        ink=(0.11, 0.09, 0.08, 0.94),
        eye_iris=(0.30, 0.35, 0.34),
        jaw_fullness=1.05,
        brow_crease=0.72,
        hair_length=1.18,
    ),
    "elderly": AgePalette(
        skin_lit=(0.82, 0.71, 0.62),
        skin_shadow=(0.56, 0.45, 0.39),
        hair_base=(0.76, 0.74, 0.70),
        hair_shadow=(0.50, 0.48, 0.45),
        hair_highlight=(0.92, 0.90, 0.86),
        coat_lit=(0.15, 0.14, 0.16),
        coat_shadow=(0.07, 0.07, 0.09),
        collar_lit=(0.90, 0.88, 0.83),
        collar_shadow=(0.65, 0.62, 0.58),
        ink=(0.12, 0.10, 0.09, 0.92),
        eye_iris=(0.34, 0.38, 0.38),
        jaw_fullness=1.08,
        brow_crease=1.0,
        hair_length=1.14,
    ),
}


def _tint_srgb(col: Tuple[float, float, float], light_tint: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (
        float(np.clip(col[0] * light_tint[0], 0.02, 0.98)),
        float(np.clip(col[1] * light_tint[1], 0.02, 0.98)),
        float(np.clip(col[2] * light_tint[2], 0.02, 0.98)),
    )


def _rot_pt(x: float, y: float, cx: float, cy: float, ang_rad: float) -> Tuple[float, float]:
    dx, dy = x - cx, y - cy
    ca, sa = math.cos(ang_rad), math.sin(ang_rad)
    return cx + dx * ca - dy * sa, cy + dx * sa + dy * ca


# ---------------------------------------------------------------------------
# Pose-Specific Head & Bust Drawings (Front-3/4, Profile, Looking Up/Down)
# ---------------------------------------------------------------------------

def draw_newton_bust(
    canvas: InkCanvas2x,
    cx: float,
    cy: float,
    scale: float,
    age: str = "young",
    view: str = "three_quarter_right",
    light_dir_2d: Tuple[float, float] = (-0.75, -0.55),
    light_tint: Tuple[float, float, float] = (1.0, 0.97, 0.93),
    blink: float = 0.0,
    head_tilt_deg: float = 0.0,
    gaze: Tuple[float, float] = (0.0, 0.0),
    breath_phase: float = 0.0,
    seed: int = 11,
) -> None:
    """Draw Newton head & shoulders in pose-specific 2D traditional cel-animated anatomy.

    view: 'three_quarter_right', 'three_quarter_left', 'profile_right', 'profile_left', 'looking_up_right'
    scale: head height in pixels (~120 to 320 for close-up/medium shots).
    """
    pal = NEWTON_AGES[age]
    ctx = canvas.ctx
    flip = -1.0 if view in ("three_quarter_left", "profile_left") else 1.0
    is_profile = view in ("profile_right", "profile_left", "looking_up_right")
    pitch_up = -0.22 if view == "looking_up_right" else 0.0

    skin_l = _tint_srgb(pal.skin_lit, light_tint)
    skin_s = _tint_srgb(pal.skin_shadow, light_tint)
    hair_b = _tint_srgb(pal.hair_base, light_tint)
    hair_s = _tint_srgb(pal.hair_shadow, light_tint)
    hair_h = _tint_srgb(pal.hair_highlight, light_tint)
    coat_l = _tint_srgb(pal.coat_lit, light_tint)
    coat_s = _tint_srgb(pal.coat_shadow, light_tint)
    col_l = _tint_srgb(pal.collar_lit, light_tint)
    col_s = _tint_srgb(pal.collar_shadow, light_tint)

    S = scale
    breath_y = math.sin(breath_phase) * S * 0.012
    ld = (light_dir_2d[0], light_dir_2d[1])
    lw = max(1.1, S * 0.0115)

    # 1. Shoulders & Period Coat (17th-century wool cloak/coat + falling band collar or cravat)
    sh_y = cy + S * 0.68 + breath_y
    sh_w = S * (0.98 if age != "child" else 0.86)
    coat_pts = [
        (cx - sh_w * 1.18, sh_y + S * 0.92),
        (cx - sh_w * 0.95, sh_y + S * 0.14),
        (cx - sh_w * 0.38, sh_y - S * 0.10),
        (cx + sh_w * 0.38, sh_y - S * 0.10),
        (cx + sh_w * 0.95, sh_y + S * 0.16),
        (cx + sh_w * 1.18, sh_y + S * 0.92),
    ]
    canvas.fill_cel_region(
        coat_pts, coat_l, coat_s, light_dir_2d=ld,
        outline_srgb=pal.ink, outline_width=lw * 1.15, shadow_shift_px=S * 0.09, seed=seed + 1,
    )
    # Coat lapel / fold lines
    canvas.stroke_catmull_rom(
        [(cx - sh_w * 0.25, sh_y - S * 0.02), (cx - sh_w * 0.14, sh_y + S * 0.42), (cx - sh_w * 0.18, sh_y + S * 0.85)],
        base_width=lw * 0.9, color_srgb=pal.ink, tremor=0.25, seed=seed + 2,
    )
    canvas.stroke_catmull_rom(
        [(cx + sh_w * 0.28, sh_y + S * 0.02), (cx + sh_w * 0.36, sh_y + S * 0.48), (cx + sh_w * 0.44, sh_y + S * 0.85)],
        base_width=lw * 0.8, color_srgb=pal.ink, tremor=0.25, seed=seed + 3,
    )

    # Falling-band white linen collar (young/child) or ruffled cravat (mature/elderly)
    collar_pts = [
        (cx - S * 0.28, sh_y - S * 0.09),
        (cx - S * 0.36, sh_y + S * 0.20),
        (cx - S * 0.05, sh_y + S * 0.26),
        (cx + S * 0.06, sh_y + S * 0.18),
        (cx + S * 0.28, sh_y + S * 0.24),
        (cx + S * 0.30, sh_y - S * 0.08),
    ]
    canvas.fill_cel_region(
        collar_pts, col_l, col_s, light_dir_2d=ld,
        outline_srgb=pal.ink, outline_width=lw * 0.88, shadow_shift_px=S * 0.04, seed=seed + 4,
    )

    # 2. Back hair cascades BEHIND neck & shoulders (never wrapping under chin like a hood!)
    hl = pal.hair_length
    ctx.save()
    ctx.translate(cx, cy)
    ctx.rotate(math.radians(head_tilt_deg) + pitch_up * flip)
    ctx.translate(-cx, -cy)

    # Left and right wavy cascading shoulder-length locks drawn BEHIND neck & face
    lock_back = [
        (cx - S * 0.44 * flip, cy - S * 0.34),
        (cx - S * 0.56 * flip, cy + S * 0.08),
        (cx - S * 0.58 * flip, cy + S * (0.48 * hl)),
        (cx - S * 0.42 * flip, cy + S * (0.64 * hl)),
        (cx - S * 0.24 * flip, cy + S * (0.52 * hl)),
        (cx - S * 0.26 * flip, cy + S * 0.05),
        (cx - S * 0.02 * flip, cy - S * 0.52),
        (cx + S * 0.34 * flip, cy - S * 0.38),
        (cx + S * 0.40 * flip, cy + S * (0.28 * hl)),
        (cx + S * 0.34 * flip, cy + S * (0.50 * hl)),
        (cx + S * 0.20 * flip, cy + S * (0.46 * hl)),
        (cx + S * 0.22 * flip, cy - S * 0.22),
    ]
    canvas.fill_cel_region(
        lock_back, hair_b, hair_s, light_dir_2d=ld,
        outline_srgb=pal.ink, outline_width=lw * 0.95, shadow_shift_px=S * 0.07, seed=seed + 6,
    )
    ctx.restore()

    # Neck drawn OVER back hair so jaw & neck stay completely open and natural!
    neck_pts = [
        (cx - S * 0.16, cy + S * 0.32),
        (cx - S * 0.19, sh_y + S * 0.04),
        (cx + S * 0.19, sh_y + S * 0.04),
        (cx + S * 0.15, cy + S * 0.32),
    ]
    canvas.fill_cel_region(
        neck_pts, skin_l, skin_s, light_dir_2d=(ld[0], ld[1] - 0.55),
        outline_srgb=pal.ink, outline_width=lw * 0.85, shadow_shift_px=S * 0.06, seed=seed + 5,
    )

    # Head rotation transform for face, features, and front wavy side-locks
    ctx.save()
    ctx.translate(cx, cy)
    ctx.rotate(math.radians(head_tilt_deg) + pitch_up * flip)
    ctx.translate(-cx, -cy)

    # 3. Face silhouette (refined slender aristocratic cheekbones, high brow, quiet resolute chin)
    jf = pal.jaw_fullness
    if is_profile:
        face_pts = [
            (cx - S * 0.20 * flip, cy - S * 0.42),
            (cx + S * 0.21 * flip, cy - S * 0.41),   # high forehead
            (cx + S * 0.27 * flip, cy - S * 0.15),   # brow ridge
            (cx + S * 0.24 * flip, cy - S * 0.05),   # eye socket
            (cx + S * 0.39 * flip, cy + S * 0.12),   # aquiline nose tip
            (cx + S * 0.27 * flip, cy + S * 0.16),   # philtrum
            (cx + S * 0.30 * flip, cy + S * 0.23),   # upper lip
            (cx + S * 0.27 * flip, cy + S * 0.29),   # lower lip
            (cx + S * 0.28 * flip, cy + S * 0.39),   # sculpted chin
            (cx + S * 0.04 * flip, cy + S * 0.43 * jf),
            (cx - S * 0.20 * flip, cy + S * 0.22),
        ]
    else:
        face_pts = [
            (cx - S * 0.30 * flip, cy - S * 0.38),
            (cx + S * 0.22 * flip, cy - S * 0.40),
            (cx + S * 0.32 * flip, cy - S * 0.12),
            (cx + S * 0.31 * flip, cy + S * 0.09),   # cheekbone
            (cx + S * 0.22 * flip, cy + S * 0.32),
            (cx + S * 0.07 * flip, cy + S * 0.44),   # chin
            (cx - S * 0.17 * flip, cy + S * 0.34 * jf),
            (cx - S * 0.30 * flip, cy + S * 0.05),
        ]

    canvas.fill_cel_region(
        face_pts, skin_l, skin_s, light_dir_2d=ld,
        rim_srgb=(min(1.0, skin_l[0] * 1.12), min(1.0, skin_l[1] * 1.09), min(1.0, skin_l[2] * 1.04)),
        outline_srgb=pal.ink, outline_width=lw * 0.92, shadow_shift_px=S * 0.07, seed=seed + 7,
    )

    # 4. Facial Features (Eyes, Aquiline Nose, Brow, Quiet Mouth, Age Lines)
    gx, gy = gaze[0] * S * 0.025, gaze[1] * S * 0.02
    if is_profile:
        ex, ey = cx + S * 0.16 * flip, cy - S * 0.02
        _draw_eye_feature(canvas, ex, ey, S * 0.088, flip, pal, blink, gx, gy, lw, seed + 10)
        canvas.stroke_catmull_rom(
            [(ex - S * 0.06 * flip, ey - S * 0.075), (ex + S * 0.05 * flip, ey - S * 0.085), (ex + S * 0.10 * flip, ey - S * 0.055)],
            base_width=lw * 1.12, color_srgb=pal.ink, tremor=0.14, seed=seed + 11,
        )
        canvas.stroke_catmull_rom(
            [(cx + S * 0.27 * flip, cy + S * 0.25), (cx + S * 0.18 * flip, cy + S * 0.258)],
            base_width=lw * 0.85, color_srgb=pal.ink, tremor=0.12, seed=seed + 12,
        )
        canvas.stroke_catmull_rom(
            [(cx - S * 0.12 * flip, cy - S * 0.01), (cx - S * 0.17 * flip, cy + S * 0.08), (cx - S * 0.11 * flip, cy + S * 0.16)],
            base_width=lw * 0.78, color_srgb=pal.ink, tremor=0.14, seed=seed + 13,
        )
    else:
        ex_near, ey_near = cx - S * 0.05 * flip, cy - S * 0.01
        ex_far, ey_far = cx + S * 0.18 * flip, cy - S * 0.015
        _draw_eye_feature(canvas, ex_near, ey_near, S * 0.095, flip, pal, blink, gx, gy, lw, seed + 14)
        _draw_eye_feature(canvas, ex_far, ey_far, S * 0.078, flip, pal, blink, gx, gy, lw, seed + 15)

        canvas.stroke_catmull_rom(
            [(ex_near - S * 0.08 * flip, ey_near - S * 0.075), (ex_near, ey_near - S * 0.092), (ex_near + S * 0.07 * flip, ey_near - S * 0.068)],
            base_width=lw * 1.12, color_srgb=pal.ink, tremor=0.14, seed=seed + 16,
        )
        canvas.stroke_catmull_rom(
            [(ex_far - S * 0.05 * flip, ey_far - S * 0.072), (ex_far + S * 0.04 * flip, ey_far - S * 0.082), (ex_far + S * 0.09 * flip, ey_far - S * 0.060)],
            base_width=lw * 1.0, color_srgb=pal.ink, tremor=0.14, seed=seed + 17,
        )

        # Delicate nose bridge & refined nostril shadow
        nose_pts = [
            (cx + S * 0.08 * flip, cy - S * 0.03),
            (cx + S * 0.12 * flip, cy + S * 0.11),
            (cx + S * 0.06 * flip, cy + S * 0.15),
        ]
        canvas.fill_smooth_poly(
            [nose_pts[0], nose_pts[1], (cx + S * 0.04 * flip, cy + S * 0.14), (cx + S * 0.05 * flip, cy + S * 0.03)],
            (*skin_s, 0.55), smooth=False,
        )
        canvas.stroke_catmull_rom(nose_pts, base_width=lw * 0.88, color_srgb=pal.ink, tremor=0.12, seed=seed + 18)

        # Quiet sculpted mouth
        canvas.stroke_catmull_rom(
            [(cx - S * 0.02 * flip, cy + S * 0.26), (cx + S * 0.08 * flip, cy + S * 0.255), (cx + S * 0.17 * flip, cy + S * 0.262)],
            base_width=lw * 0.85, color_srgb=pal.ink, tremor=0.12, seed=seed + 19,
        )
        canvas.stroke_catmull_rom(
            [(cx + S * 0.04 * flip, cy + S * 0.295), (cx + S * 0.11 * flip, cy + S * 0.295)],
            base_width=lw * 0.55, color_srgb=(*skin_s, 0.85), tremor=0.1, seed=seed + 20,
        )

    # Age-specific restrained lines (mature / elderly)
    if pal.brow_crease > 0.2:
        a_l = min(0.50, pal.brow_crease * 0.42)
        canvas.stroke_catmull_rom(
            [(cx - S * 0.02 * flip, cy - S * 0.14), (cx + S * 0.13 * flip, cy - S * 0.145)],
            base_width=lw * 0.50, color_srgb=(*pal.ink[:3], a_l), tremor=0.1, seed=seed + 21,
        )
        if age == "elderly":
            canvas.stroke_catmull_rom(
                [(cx - S * 0.01 * flip, cy + S * 0.14), (cx - S * 0.05 * flip, cy + S * 0.27)],
                base_width=lw * 0.55, color_srgb=(*pal.ink[:3], 0.42), tremor=0.12, seed=seed + 22,
            )
            canvas.stroke_catmull_rom(
                [(cx - S * 0.10 * flip, cy + S * 0.045), (cx - S * 0.01 * flip, cy + S * 0.052)],
                base_width=lw * 0.48, color_srgb=(*skin_s, 0.72), tremor=0.1, seed=seed + 23,
            )

    # 5. Front Hair Crown & Wavy Side Lock (draping gracefully down shoulder, never crossing under chin)
    front_lock_l = [
        (cx - S * 0.36 * flip, cy - S * 0.42),
        (cx - S * 0.02 * flip, cy - S * 0.54),
        (cx + S * 0.28 * flip, cy - S * 0.42),
        (cx + S * 0.22 * flip, cy - S * 0.28),
        (cx - S * 0.04 * flip, cy - S * 0.34),
        (cx - S * 0.24 * flip, cy - S * 0.10),
        (cx - S * 0.27 * flip, cy + S * (0.28 * hl)),
        (cx - S * 0.36 * flip, cy + S * (0.58 * hl)),
        (cx - S * 0.52 * flip, cy + S * (0.44 * hl)),
        (cx - S * 0.48 * flip, cy - S * 0.10),
    ]
    canvas.fill_cel_region(
        front_lock_l, hair_b, hair_s, light_dir_2d=ld,
        rim_srgb=hair_h, outline_srgb=pal.ink, outline_width=lw * 0.92, shadow_shift_px=S * 0.06, seed=seed + 24,
    )
    # Soft wavy hair flow lines
    for i in range(4):
        frac = -0.34 + i * 0.08
        canvas.stroke_catmull_rom(
            [
                (cx + S * (frac * 0.5) * flip, cy - S * 0.44),
                (cx + S * (frac - 0.16) * flip, cy - S * 0.02),
                (cx + S * (frac - 0.14) * flip, cy + S * (0.42 * hl)),
            ],
            base_width=lw * 0.62,
            color_srgb=(*hair_h, 0.52) if i % 2 == 0 else (*pal.ink[:3], 0.50),
            tremor=0.24,
            seed=seed + 30 + i,
        )

    ctx.restore()


def _draw_eye_feature(
    canvas: InkCanvas2x,
    ex: float,
    ey: float,
    ew: float,
    flip: float,
    pal: AgePalette,
    blink: float,
    gx: float,
    gy: float,
    lw: float,
    seed: int,
) -> None:
    ctx = canvas.ctx
    open_h = ew * 0.44 * max(0.04, 1.0 - blink)
    if blink >= 0.85:
        canvas.stroke_catmull_rom(
            [(ex - ew * 0.55, ey + ew * 0.06), (ex, ey + ew * 0.14), (ex + ew * 0.55, ey + ew * 0.04)],
            base_width=lw * 1.05, color_srgb=pal.ink, tremor=0.1, seed=seed,
        )
        return

    # Sclera almond
    top_pts = [(ex - ew * 0.55, ey + open_h * 0.15), (ex, ey - open_h), (ex + ew * 0.55, ey + open_h * 0.1)]
    bot_pts = [(ex + ew * 0.55, ey + open_h * 0.1), (ex, ey + open_h * 0.72), (ex - ew * 0.55, ey + open_h * 0.15)]
    canvas.fill_smooth_poly(top_pts + bot_pts[1:], (0.93, 0.91, 0.87, 0.98))

    # Iris & pupil
    ir = ew * 0.30
    ix, iy = ex + gx, ey + gy - open_h * 0.05
    ctx.save()
    ctx.arc(ix, iy, ir, 0, 2 * math.pi)
    ctx.set_source_rgba(*pal.eye_iris, 1.0)
    ctx.fill()
    ctx.arc(ix, iy, ir * 0.52, 0, 2 * math.pi)
    ctx.set_source_rgba(0.06, 0.05, 0.05, 1.0)
    ctx.fill()
    # Quiet restrained specular catchlight
    ctx.arc(ix - ir * 0.32, iy - ir * 0.32, max(1.0, ir * 0.22), 0, 2 * math.pi)
    ctx.set_source_rgba(0.97, 0.96, 0.92, 0.88)
    ctx.fill()
    ctx.restore()

    # Crisp upper lash line & subtle eyelid crease
    canvas.stroke_catmull_rom(top_pts, base_width=lw * 1.15, color_srgb=pal.ink, tremor=0.08, seed=seed + 1)
    canvas.stroke_catmull_rom(
        [(ex - ew * 0.42, ey - open_h * 1.25), (ex, ey - open_h * 1.52), (ex + ew * 0.40, ey - open_h * 1.15)],
        base_width=lw * 0.52, color_srgb=(*pal.ink[:3], 0.55), tremor=0.08, seed=seed + 2,
    )


# ---------------------------------------------------------------------------
# Close-up Expressive Eye (Raindrop Reflection in Shot 2 / Prism Spectrum in Shot 7)
# ---------------------------------------------------------------------------

def draw_closeup_eye(
    canvas: InkCanvas2x,
    cx: float,
    cy: float,
    scale: float,
    age: str = "young",
    gaze: Tuple[float, float] = (0.0, 0.0),
    blink: float = 0.0,
    spectrum_reflect: float = 0.0,
    rain_reflect: float = 0.0,
    light_tint: Tuple[float, float, float] = (0.95, 0.92, 0.88),
    seed: int = 55,
) -> None:
    """Theatrical extreme close-up of Newton's eye reflecting raindrops or the refracted spectrum."""
    pal = NEWTON_AGES[age]
    ctx = canvas.ctx
    S = scale
    skin_l = _tint_srgb(pal.skin_lit, light_tint)
    skin_s = _tint_srgb(pal.skin_shadow, light_tint)

    # Surrounding eyelid & brow bridge skin plane
    ctx.save()
    open_f = max(0.05, 1.0 - blink)
    ew = S * 0.92
    eh = S * 0.34 * open_f

    # Eye aperture clip
    upper = [(cx - ew * 0.56, cy + eh * 0.14), (cx - ew * 0.18, cy - eh * 0.92),
             (cx + ew * 0.22, cy - eh * 0.82), (cx + ew * 0.56, cy + eh * 0.12)]
    lower = [(cx + ew * 0.56, cy + eh * 0.12), (cx + ew * 0.16, cy + eh * 0.72),
             (cx - ew * 0.24, cy + eh * 0.74), (cx - ew * 0.56, cy + eh * 0.14)]

    # Soft radial eyelid socket shading (no hard polygonal skin cutoff!)
    sock = cairo.RadialGradient(cx, cy - eh * 0.1, ew * 0.25, cx, cy, ew * 0.85)
    sock.add_color_stop_rgba(0.0, *skin_s, 0.36)
    sock.add_color_stop_rgba(1.0, *skin_s, 0.0)
    ctx.set_source(sock)
    ctx.arc(cx, cy, ew * 0.88, 0, 2 * math.pi)
    ctx.fill()

    # Sclera
    canvas.fill_smooth_poly(upper + lower[1:], (0.91, 0.89, 0.85, 1.0))

    # Clip inside eye opening for iris + reflections
    ctx.move_to(*upper[0])
    for p in upper[1:] + lower[1:]:
        ctx.line_to(*p)
    ctx.close_path()
    ctx.clip()

    ix = cx + gaze[0] * S * 0.16
    iy = cy + gaze[1] * S * 0.10
    ir = S * 0.255

    # Iris radial painterly gradient
    grad = cairo.RadialGradient(ix, iy, ir * 0.2, ix, iy, ir)
    grad.add_color_stop_rgba(0.0, pal.eye_iris[0] * 1.35, pal.eye_iris[1] * 1.35, pal.eye_iris[2] * 1.25, 1.0)
    grad.add_color_stop_rgba(0.75, *pal.eye_iris, 1.0)
    grad.add_color_stop_rgba(1.0, pal.eye_iris[0] * 0.42, pal.eye_iris[1] * 0.42, pal.eye_iris[2] * 0.45, 1.0)
    ctx.set_source(grad)
    ctx.arc(ix, iy, ir, 0, 2 * math.pi)
    ctx.fill()

    # Subtle radial iris fibers
    rng = np.random.default_rng(seed)
    for k in range(32):
        ang = k * (2.0 * math.pi / 32.0) + float(rng.uniform(-0.04, 0.04))
        r0 = ir * float(rng.uniform(0.45, 0.58))
        r1 = ir * float(rng.uniform(0.84, 0.96))
        ctx.set_source_rgba(pal.eye_iris[0] * 1.45, pal.eye_iris[1] * 1.45, pal.eye_iris[2] * 1.35, 0.26)
        ctx.set_line_width(max(1.0, S * 0.005))
        ctx.move_to(ix + math.cos(ang) * r0, iy + math.sin(ang) * r0)
        ctx.line_to(ix + math.cos(ang) * r1, iy + math.sin(ang) * r1)
        ctx.stroke()

    # Pupil
    ctx.set_source_rgba(0.04, 0.035, 0.04, 1.0)
    ctx.arc(ix, iy, ir * 0.42, 0, 2 * math.pi)
    ctx.fill()

    # Upper eyelid cast shadow across top of sclera & iris
    ctx.set_source_rgba(0.10, 0.08, 0.10, 0.34)
    ctx.rectangle(cx - ew, cy - eh * 1.2, ew * 2.0, eh * 0.85)
    ctx.fill()

    # Spectral rainbow reflection across cornea (Shot 7)
    if spectrum_reflect > 0.01:
        spec_cols = [
            (0.88, 0.14, 0.12), (0.96, 0.48, 0.12), (0.96, 0.86, 0.18),
            (0.24, 0.84, 0.32), (0.18, 0.62, 0.94), (0.42, 0.22, 0.88),
        ]
        bw = ir * 0.24
        for idx, sc in enumerate(spec_cols):
            rx = ix - ir * 0.68 + idx * bw * 0.92
            ctx.set_source_rgba(*sc, 0.72 * spectrum_reflect)
            ctx.rectangle(rx, iy - ir * 0.32, bw * 0.95, ir * 0.56)
            ctx.fill()

    # Leaded window + raindrop catchlight reflection (Shot 2)
    if rain_reflect > 0.01:
        ctx.set_source_rgba(0.86, 0.92, 0.98, 0.68 * rain_reflect)
        wx, wy = ix - ir * 0.48, iy - ir * 0.42
        ww, wh = ir * 0.42, ir * 0.34
        ctx.rectangle(wx, wy, ww * 0.44, wh * 0.44)
        ctx.rectangle(wx + ww * 0.52, wy, ww * 0.44, wh * 0.44)
        ctx.rectangle(wx, wy + wh * 0.52, ww * 0.44, wh * 0.44)
        ctx.rectangle(wx + ww * 0.52, wy + wh * 0.52, ww * 0.44, wh * 0.44)
        ctx.fill()
    else:
        ctx.set_source_rgba(0.98, 0.96, 0.91, 0.88)
        ctx.arc(ix - ir * 0.34, iy - ir * 0.34, ir * 0.15, 0, 2 * math.pi)
        ctx.fill()

    ctx.restore()

    # Bold tapered upper lash contour, gentle lower lash, and eyelid fold
    lw = max(1.6, S * 0.014)
    canvas.stroke_catmull_rom(upper, base_width=lw * 1.45, color_srgb=pal.ink, tremor=0.18, seed=seed + 3)
    canvas.stroke_catmull_rom(lower, base_width=lw * 0.65, color_srgb=(*pal.ink[:3], 0.72), tremor=0.16, seed=seed + 4)
    canvas.stroke_catmull_rom(
        [(cx - ew * 0.50, cy - eh * 1.22), (cx - ew * 0.05, cy - eh * 1.48), (cx + ew * 0.48, cy - eh * 1.12)],
        base_width=lw * 0.62, color_srgb=(*pal.ink[:3], 0.58), tremor=0.16, seed=seed + 5,
    )
    # Eyebrow above
    canvas.stroke_catmull_rom(
        [(cx - ew * 0.68, cy - S * 0.48), (cx - ew * 0.06, cy - S * 0.56), (cx + ew * 0.62, cy - S * 0.44)],
        base_width=lw * 2.2, color_srgb=(*pal.hair_shadow, 0.92), tremor=0.35, seed=seed + 6,
    )


# ---------------------------------------------------------------------------
# Expressive Close-up Hand Rig (Quill, Mechanism, Brass Instrument, Coin)
# ---------------------------------------------------------------------------

def draw_closeup_hand(
    canvas: InkCanvas2x,
    wrist_x: float,
    wrist_y: float,
    scale: float,
    age: str = "young",
    pose: str = "quill",
    angle_deg: float = -18.0,
    light_dir_2d: Tuple[float, float] = (-0.7, -0.6),
    light_tint: Tuple[float, float, float] = (1.0, 0.94, 0.84),
    ink_stained: bool = False,
    seed: int = 77,
) -> Tuple[float, float]:
    """Draw an anatomically poised hand in cel style. Returns (fingertip_x, fingertip_y) for prop contact."""
    pal = NEWTON_AGES[age]
    skin_l = _tint_srgb(pal.skin_lit, light_tint)
    skin_s = _tint_srgb(pal.skin_shadow, light_tint)
    coat_l = _tint_srgb(pal.coat_lit, light_tint)
    coat_s = _tint_srgb(pal.coat_shadow, light_tint)
    cuff_l = _tint_srgb(pal.collar_lit, light_tint)
    cuff_s = _tint_srgb(pal.collar_shadow, light_tint)

    S = scale
    ang = math.radians(angle_deg)
    lw = max(1.3, S * 0.0085)

    def xform(lx: float, ly: float) -> Tuple[float, float]:
        ca, sa = math.cos(ang), math.sin(ang)
        return wrist_x + (lx * ca - ly * sa) * S, wrist_y + (lx * sa + ly * ca) * S

    # Sleeve & linen cuff
    sleeve = [xform(-0.75, -0.24), xform(-0.08, -0.17), xform(-0.08, 0.21), xform(-0.75, 0.28)]
    cuff = [xform(-0.15, -0.18), xform(0.03, -0.15), xform(0.02, 0.18), xform(-0.15, 0.20)]
    canvas.fill_cel_region(sleeve, coat_l, coat_s, light_dir_2d, outline_srgb=pal.ink, outline_width=lw * 1.1, seed=seed)
    canvas.fill_cel_region(cuff, cuff_l, cuff_s, light_dir_2d, outline_srgb=pal.ink, outline_width=lw * 0.95, seed=seed + 1)

    # Palm & fingers (slender for young/elderly, slightly shorter for child)
    flen = 0.86 if age == "child" else 1.0
    palm_pts = [
        xform(0.0, -0.14),
        xform(0.32, -0.16),
        xform(0.56 * flen, -0.08),
        xform(0.50 * flen, 0.18),
        xform(0.24, 0.22),
        xform(0.0, 0.15),
    ]
    canvas.fill_cel_region(palm_pts, skin_l, skin_s, light_dir_2d, outline_srgb=pal.ink, outline_width=lw, seed=seed + 2)

    # Index finger & thumb pinch
    idx_finger = [
        xform(0.36, -0.14),
        xform(0.68 * flen, -0.16),
        xform(0.92 * flen, -0.04),
        xform(0.89 * flen, 0.04),
        xform(0.63 * flen, -0.05),
        xform(0.36, -0.03),
    ]
    mid_finger = [
        xform(0.38, -0.04),
        xform(0.70 * flen, 0.00),
        xform(0.84 * flen, 0.10),
        xform(0.78 * flen, 0.16),
        xform(0.42, 0.09),
    ]
    thumb = [
        xform(0.18, 0.12),
        xform(0.46 * flen, 0.18),
        xform(0.72 * flen, 0.08),
        xform(0.70 * flen, 0.01),
        xform(0.44 * flen, 0.06),
        xform(0.22, 0.03),
    ]
    canvas.fill_cel_region(mid_finger, skin_s, (skin_s[0] * 0.85, skin_s[1] * 0.85, skin_s[2] * 0.85),
                           light_dir_2d, outline_srgb=pal.ink, outline_width=lw * 0.9, seed=seed + 3)
    canvas.fill_cel_region(idx_finger, skin_l, skin_s, light_dir_2d, outline_srgb=pal.ink, outline_width=lw, seed=seed + 4)
    canvas.fill_cel_region(thumb, skin_l, skin_s, light_dir_2d, outline_srgb=pal.ink, outline_width=lw, seed=seed + 5)

    # Subtle knuckle creases & aged hand tendons
    if age in ("mature", "elderly"):
        for k in range(2):
            p0 = xform(0.12, -0.07 + k * 0.09)
            p1 = xform(0.34, -0.09 + k * 0.09)
            canvas.stroke_catmull_rom([p0, p1], base_width=lw * 0.55, color_srgb=(*skin_s, 0.65), tremor=0.1, seed=seed + 10 + k)

    tip = xform(0.90 * flen, -0.01)
    if ink_stained:
        ctx = canvas.ctx
        ctx.set_source_rgba(0.12, 0.10, 0.16, 0.45)
        ctx.arc(tip[0] - S * 0.04, tip[1] + S * 0.02, S * 0.042, 0, 2 * math.pi)
        ctx.fill()

    if pose == "quill":
        # Draw goose-feather quill with visible sepia shaft and dark ink-dipped nib anchored in fingers
        nib = xform(0.98 * flen, 0.12)
        top = xform(0.58 * flen, -0.80)
        canvas.stroke_catmull_rom([nib, xform(0.80 * flen, -0.18), top], base_width=lw * 1.8,
                                  color_srgb=(0.32, 0.24, 0.18, 0.96), tremor=0.12, seed=seed + 15)
        # Ink-dipped quill nib
        canvas.stroke_catmull_rom([nib, xform(0.92 * flen, 0.04)], base_width=lw * 2.0,
                                  color_srgb=(0.08, 0.07, 0.08, 0.98), tremor=0.06, seed=seed + 17)
        # Feather vane
        vane = [
            xform(0.72 * flen, -0.34),
            xform(0.46 * flen, -0.58),
            xform(0.54 * flen, -0.84),
            top,
            xform(0.74 * flen, -0.64),
            xform(0.80 * flen, -0.38),
        ]
        canvas.fill_cel_region(vane, (0.93, 0.90, 0.83), (0.72, 0.68, 0.61), light_dir_2d,
                               outline_srgb=pal.ink, outline_width=lw * 0.85, seed=seed + 16)
        return nib

    return tip


# ---------------------------------------------------------------------------
# Full / Three-Quarter Staged Figure Drawing (Walking, Sitting, Observing)
# ---------------------------------------------------------------------------

def draw_newton_figure(
    canvas: InkCanvas2x,
    foot_x: float,
    foot_y: float,
    height_px: float,
    age: str = "young",
    action: str = "walk_right",
    phase: float = 0.0,
    light_dir_2d: Tuple[float, float] = (-0.7, -0.6),
    light_tint: Tuple[float, float, float] = (1.0, 0.97, 0.93),
    seed: int = 33,
) -> None:
    """Staged full-body figure of Newton for wide & medium compositions (never a static cutout)."""
    pal = NEWTON_AGES[age]
    H = height_px
    coat_l = _tint_srgb(pal.coat_lit, light_tint)
    coat_s = _tint_srgb(pal.coat_shadow, light_tint)
    skin_l = _tint_srgb(pal.skin_lit, light_tint)
    skin_s = _tint_srgb(pal.skin_shadow, light_tint)
    hair_b = _tint_srgb(pal.hair_base, light_tint)
    hair_s = _tint_srgb(pal.hair_shadow, light_tint)
    col_l = _tint_srgb(pal.collar_lit, light_tint)
    col_s = _tint_srgb(pal.collar_shadow, light_tint)
    lw = max(0.85, H * 0.0048)

    flip = -1.0 if "left" in action else 1.0
    stride = math.sin(phase) if "walk" in action else 0.0
    bob = abs(math.cos(phase)) * H * 0.015 if "walk" in action else 0.0

    # Legs & buckled 17th-century shoes
    hip_y = foot_y - H * 0.44 - bob
    if action.startswith("sit"):
        # Seated under tree or desk
        leg1 = [
            (foot_x - H * 0.08 * flip, hip_y),
            (foot_x + H * 0.24 * flip, hip_y + H * 0.06),
            (foot_x + H * 0.34 * flip, foot_y - H * 0.03),
            (foot_x + H * 0.24 * flip, foot_y),
            (foot_x + H * 0.14 * flip, hip_y + H * 0.15),
            (foot_x - H * 0.12 * flip, hip_y + H * 0.12),
        ]
        canvas.fill_cel_region(leg1, coat_s, (coat_s[0] * 0.7, coat_s[1] * 0.7, coat_s[2] * 0.7),
                               light_dir_2d, outline_srgb=pal.ink, outline_width=lw, seed=seed)
    else:
        for side, sgn in ((-1.0, -stride), (1.0, stride)):
            fx = foot_x + sgn * H * 0.11 * flip
            fy = foot_y - max(0.0, sgn * side) * H * 0.02
            knee_x = foot_x + sgn * H * 0.06 * flip + H * 0.02 * flip
            knee_y = hip_y + H * 0.22
            leg = [
                (foot_x - H * 0.035, hip_y),
                (knee_x + H * 0.032, knee_y),
                (fx + H * 0.024, fy - H * 0.02),
                (fx + H * 0.055 * flip, fy),
                (fx - H * 0.022 * flip, fy),
                (knee_x - H * 0.032, knee_y),
                (foot_x + H * 0.035, hip_y),
            ]
            col_use = coat_l if side > 0 else coat_s
            canvas.fill_cel_region(leg, col_use, coat_s, light_dir_2d, outline_srgb=pal.ink, outline_width=lw, seed=seed + int(side + 2))

    # Torso / Scholar's Gown or Long 17th-century Coat (with subtle wind/motion follow-through)
    sh_y = foot_y - H * 0.78 - bob
    hem_sway = math.sin(phase - 0.5) * H * 0.035
    torso = [
        (foot_x - H * 0.10 * flip, sh_y),
        (foot_x + H * 0.10 * flip, sh_y + H * 0.01),
        (foot_x + H * 0.15 * flip + hem_sway * 0.4, hip_y + H * 0.16),
        (foot_x - H * 0.18 * flip + hem_sway, hip_y + H * 0.18),
    ]
    canvas.fill_cel_region(torso, coat_l, coat_s, light_dir_2d, outline_srgb=pal.ink, outline_width=lw * 1.1, seed=seed + 5)

    # White collar flash
    collar = [
        (foot_x - H * 0.05, sh_y - H * 0.01),
        (foot_x + H * 0.06 * flip, sh_y + H * 0.07),
        (foot_x + H * 0.01 * flip, sh_y + H * 0.09),
        (foot_x - H * 0.05 * flip, sh_y + H * 0.05),
    ]
    canvas.fill_cel_region(collar, col_l, col_s, light_dir_2d, outline_srgb=pal.ink, outline_width=lw * 0.85, seed=seed + 6)

    # Carrying books or holding notebook when applicable
    if "books" in action:
        bx = foot_x + H * 0.08 * flip
        by = sh_y + H * 0.19
        canvas.fill_cel_region(
            [(bx - H * 0.06, by - H * 0.04), (bx + H * 0.08, by - H * 0.05),
             (bx + H * 0.09, by + H * 0.05), (bx - H * 0.05, by + H * 0.06)],
            (0.42, 0.24, 0.14), (0.24, 0.13, 0.08), light_dir_2d, outline_srgb=pal.ink, outline_width=lw, seed=seed + 7,
        )

    # Arm
    arm_swing = -stride * H * 0.06 if "walk" in action else 0.0
    hand_x = foot_x + (H * 0.14 if "books" in action or "sit" in action else arm_swing) * flip
    hand_y = sh_y + (H * 0.21 if "books" in action else H * 0.28)
    arm = [
        (foot_x - H * 0.03 * flip, sh_y + H * 0.03),
        (foot_x + H * 0.05 * flip, sh_y + H * 0.05),
        (hand_x + H * 0.025, hand_y),
        (hand_x - H * 0.025, hand_y + H * 0.01),
    ]
    canvas.fill_cel_region(arm, coat_l, coat_s, light_dir_2d, outline_srgb=pal.ink, outline_width=lw, seed=seed + 8)

    # Head & iconic wavy hair
    head_y = sh_y - H * 0.09
    head_r = H * 0.072
    back_h = [
        (foot_x - head_r * 1.15 * flip, head_y - head_r * 0.9),
        (foot_x - head_r * 1.35 * flip, head_y + head_r * 1.15),
        (foot_x + head_r * 0.45 * flip, head_y + head_r * 1.05),
        (foot_x + head_r * 0.85 * flip, head_y - head_r * 0.85),
    ]
    canvas.fill_cel_region(back_h, hair_b, hair_s, light_dir_2d, outline_srgb=pal.ink, outline_width=lw, seed=seed + 9)
    face = [
        (foot_x - head_r * 0.45 * flip, head_y - head_r * 0.78),
        (foot_x + head_r * 0.85 * flip, head_y - head_r * 0.65),
        (foot_x + head_r * 1.02 * flip, head_y + head_r * 0.15),
        (foot_x + head_r * 0.62 * flip, head_y + head_r * 0.92),
        (foot_x - head_r * 0.42 * flip, head_y + head_r * 0.75),
    ]
    canvas.fill_cel_region(face, skin_l, skin_s, light_dir_2d, outline_srgb=pal.ink, outline_width=lw, seed=seed + 10)


# ---------------------------------------------------------------------------
# Supporting Characters (Individually Designed — Never Cloned)
# ---------------------------------------------------------------------------

def draw_halley_bust(
    canvas: InkCanvas2x,
    cx: float,
    cy: float,
    scale: float,
    light_dir_2d: Tuple[float, float] = (0.7, -0.5),
    light_tint: Tuple[float, float, float] = (1.0, 0.90, 0.76),
    blink: float = 0.0,
    seed: int = 88,
) -> None:
    """Edmund Halley (1684): rounder warm face, curled periwig, russet-ochre 1680s coat."""
    S = scale
    skin_l = _tint_srgb((0.89, 0.74, 0.62), light_tint)
    skin_s = _tint_srgb((0.65, 0.50, 0.40), light_tint)
    wig_l = _tint_srgb((0.36, 0.25, 0.17), light_tint)
    wig_s = _tint_srgb((0.18, 0.12, 0.08), light_tint)
    coat_l = _tint_srgb((0.46, 0.26, 0.16), light_tint)
    coat_s = _tint_srgb((0.25, 0.13, 0.08), light_tint)
    ink = (0.12, 0.09, 0.07, 0.94)
    lw = max(1.1, S * 0.0115)

    sh_y = cy + S * 0.66
    coat = [
        (cx - S * 1.08, sh_y + S * 0.90),
        (cx - S * 0.88, sh_y + S * 0.10),
        (cx - S * 0.34, sh_y - S * 0.10),
        (cx + S * 0.36, sh_y - S * 0.10),
        (cx + S * 0.90, sh_y + S * 0.14),
        (cx + S * 1.08, sh_y + S * 0.90),
    ]
    canvas.fill_cel_region(coat, coat_l, coat_s, light_dir_2d, outline_srgb=ink, outline_width=lw * 1.1, seed=seed)

    # Full curled 1680s periwig framing rounder cheeks & cascading behind white cravat
    wig = [
        (cx - S * 0.56, cy - S * 0.32),
        (cx - S * 0.05, cy - S * 0.58),
        (cx + S * 0.50, cy - S * 0.38),
        (cx + S * 0.64, cy + S * 0.18),
        (cx + S * 0.56, cy + S * 0.68),
        (cx + S * 0.24, cy + S * 0.64),
        (cx + S * 0.24, cy + S * 0.15),
        (cx - S * 0.24, cy + S * 0.15),
        (cx - S * 0.24, cy + S * 0.64),
        (cx - S * 0.50, cy + S * 0.66),
        (cx - S * 0.60, cy + S * 0.16),
    ]
    canvas.fill_cel_region(wig, wig_l, wig_s, light_dir_2d, outline_srgb=ink, outline_width=lw, seed=seed + 1)

    # White 1680s lace cravat & neck
    cravat = [
        (cx - S * 0.20, sh_y - S * 0.10),
        (cx + S * 0.20, sh_y - S * 0.10),
        (cx + S * 0.16, sh_y + S * 0.32),
        (cx - S * 0.16, sh_y + S * 0.32),
    ]
    canvas.fill_cel_region(cravat, (0.91, 0.88, 0.81), (0.65, 0.62, 0.56), light_dir_2d, outline_srgb=ink, outline_width=lw * 0.85, seed=seed + 9)

    # Facing left toward Newton
    face = [
        (cx - S * 0.36, cy - S * 0.24),
        (cx + S * 0.22, cy - S * 0.36),
        (cx + S * 0.32, cy + S * 0.14),
        (cx + S * 0.12, cy + S * 0.44),
        (cx - S * 0.16, cy + S * 0.44),
        (cx - S * 0.35, cy + S * 0.18),
    ]
    canvas.fill_cel_region(face, skin_l, skin_s, light_dir_2d, outline_srgb=ink, outline_width=lw * 0.95, seed=seed + 2)

    # Warm eyes & gentle encouraging smile
    halley_pal = AgePalette(skin_l, skin_s, wig_l, wig_s, wig_l, coat_l, coat_s, (0.9, 0.88, 0.8),
                            (0.65, 0.62, 0.55), ink, (0.28, 0.22, 0.16), 1.1, 0.2, 1.0)
    _draw_eye_feature(canvas, cx - S * 0.17, cy - S * 0.02, S * 0.082, -1.0, halley_pal, blink, -S * 0.01, 0.0, lw, seed + 3)
    _draw_eye_feature(canvas, cx + S * 0.07, cy - S * 0.01, S * 0.095, -1.0, halley_pal, blink, -S * 0.01, 0.0, lw, seed + 4)
    canvas.stroke_catmull_rom(
        [(cx - S * 0.07, cy - S * 0.04), (cx - S * 0.13, cy + S * 0.11), (cx - S * 0.05, cy + S * 0.15)],
        base_width=lw * 0.9, color_srgb=ink, tremor=0.14, seed=seed + 5,
    )
    canvas.stroke_catmull_rom(
        [(cx - S * 0.16, cy + S * 0.26), (cx - S * 0.05, cy + S * 0.28), (cx + S * 0.05, cy + S * 0.25)],
        base_width=lw * 0.85, color_srgb=ink, tremor=0.12, seed=seed + 6,
    )


def draw_supporting_scholar(
    canvas: InkCanvas2x,
    foot_x: float,
    foot_y: float,
    height_px: float,
    seed: int,
    phase: float = 0.0,
    facing: float = 1.0,
    light_dir_2d: Tuple[float, float] = (-0.6, -0.7),
) -> None:
    """Individually proportioned Cambridge scholar in academic gown & cap."""
    rng = np.random.default_rng(seed)
    H = height_px * float(rng.uniform(0.92, 1.08))
    w_scale = float(rng.uniform(0.88, 1.16))
    gown_tone = float(rng.uniform(0.09, 0.17))
    gown_l = (gown_tone * 1.15, gown_tone * 1.12, gown_tone * 1.28)
    gown_s = (gown_tone * 0.55, gown_tone * 0.54, gown_tone * 0.65)
    ink = (0.08, 0.07, 0.08, 0.88)
    lw = max(0.7, H * 0.005)
    sway = math.sin(phase + seed) * H * 0.025

    # Striding legs & buckled shoes peeking below academic gown hem
    stride = math.sin(phase + seed) * H * 0.055
    for sx in (-stride, stride):
        leg = [
            (foot_x + sx - H * 0.024, foot_y - H * 0.14),
            (foot_x + sx + H * 0.024, foot_y - H * 0.14),
            (foot_x + sx + H * 0.032, foot_y),
            (foot_x + sx - H * 0.026, foot_y),
        ]
        canvas.fill_cel_region(leg, (0.07, 0.07, 0.09), (0.04, 0.04, 0.05), light_dir_2d,
                               outline_srgb=ink, outline_width=lw * 0.8, smooth=False, seed=seed + 1)

    # Academic gown with structured shoulders, draped side sleeves, and flowing hem
    pts = [
        (foot_x - H * 0.08 * w_scale, foot_y - H * 0.78),
        (foot_x + H * 0.08 * w_scale, foot_y - H * 0.78),
        (foot_x + H * 0.15 * w_scale, foot_y - H * 0.68),
        (foot_x + H * 0.18 * w_scale + sway * 0.5, foot_y - H * 0.42),
        (foot_x + H * 0.16 * w_scale + sway, foot_y - H * 0.10),
        (foot_x - H * 0.16 * w_scale + sway, foot_y - H * 0.09),
        (foot_x - H * 0.18 * w_scale + sway * 0.5, foot_y - H * 0.42),
        (foot_x - H * 0.15 * w_scale, foot_y - H * 0.68),
    ]
    canvas.fill_cel_region(pts, gown_l, gown_s, light_dir_2d, outline_srgb=ink, outline_width=lw, seed=seed)
    # White scholar's falling-band collar
    canvas.fill_smooth_poly(
        [(foot_x - H * 0.04, foot_y - H * 0.79), (foot_x + H * 0.04, foot_y - H * 0.79),
         (foot_x + H * 0.03, foot_y - H * 0.72), (foot_x - H * 0.03, foot_y - H * 0.72)],
        (0.84, 0.82, 0.77, 0.94), smooth=False,
    )
    # Head, dark hair & mortarboard/academic cap
    hr = H * 0.062
    hy = foot_y - H * 0.865
    canvas.fill_smooth_poly(
        [(foot_x - hr * 1.15, hy - hr * 0.6), (foot_x + hr * 1.15, hy - hr * 0.6),
         (foot_x + hr * 1.1, hy + hr * 1.1), (foot_x - hr * 1.1, hy + hr * 1.1)],
        (0.16, 0.12, 0.09, 0.95),
    )
    canvas.fill_smooth_poly(
        [(foot_x - hr * 0.82, hy - hr * 0.85), (foot_x + hr * 0.82, hy - hr * 0.85),
         (foot_x + hr * 0.72, hy + hr * 0.85), (foot_x - hr * 0.72, hy + hr * 0.85)],
        (0.76, 0.65, 0.56, 0.96),
    )
    canvas.fill_smooth_poly(
        [(foot_x - hr * 1.65, hy - hr * 0.98), (foot_x + hr * 1.65, hy - hr * 0.98),
         (foot_x + hr * 1.15, hy - hr * 0.58), (foot_x - hr * 1.15, hy - hr * 0.58)],
        (0.09, 0.09, 0.11, 0.96), smooth=False,
    )


def render_model_sheet(out_path: str) -> None:
    """Export Stage 5 verification character model sheet of Newton's 4 ages + Halley + eye/hand."""
    import cv2
    from .painter import linear_to_srgb

    w, h = 1600, 900
    bg = np.full((h, w, 3), np.array([0.86, 0.81, 0.71], dtype=np.float32))
    canvas = InkCanvas2x(w, h)
    canvas.draw_calligraphy("SIR ISAAC NEWTON — CHARACTER MODEL SHEET (1642–1727)", w * 0.5, 58, 30,
                            (0.14, 0.11, 0.09, 0.95), italic=False, bold=True)

    ages = [("child", "Child (c. 1652)"), ("young", "Young Adult (1661–1668)"),
            ("mature", "Mature (1687)"), ("elderly", "Elderly (c. 1715)")]
    for idx, (age_key, label) in enumerate(ages):
        cx = 210 + idx * 380
        draw_newton_bust(canvas, cx, 232, 152, age=age_key, view="three_quarter_right", seed=100 + idx * 7)
        canvas.draw_calligraphy(label, cx, 526, 22, (0.14, 0.10, 0.08, 0.95), italic=True)

    # Bottom row: Profile looking up, Edmund Halley, Close-up Eye w/ Spectrum, Close-up Quill Hand
    draw_newton_bust(canvas, 220, 650, 130, age="young", view="looking_up_right", seed=201)
    canvas.draw_calligraphy("Profile — Looking Up", 220, 882, 20, (0.14, 0.10, 0.08, 0.95))

    draw_halley_bust(canvas, 590, 650, 130, seed=202)
    canvas.draw_calligraphy("Edmund Halley (1684)", 590, 882, 20, (0.14, 0.10, 0.08, 0.95))

    draw_closeup_eye(canvas, 980, 695, 185, age="young", spectrum_reflect=0.85, seed=203)
    canvas.draw_calligraphy("Eye Study (Spectral Reflection)", 980, 882, 20, (0.14, 0.10, 0.08, 0.95))

    draw_closeup_hand(canvas, 1265, 735, 210, age="young", pose="quill", ink_stained=True, seed=204)
    canvas.draw_calligraphy("Hand Rig (Quill & Ink)", 1380, 882, 20, (0.14, 0.10, 0.08, 0.95))

    layer = canvas.to_premul_linear_rgba()
    a = layer[..., 3:4]
    bg_lin = np.power(bg, 2.2)
    comp = layer[..., :3] + bg_lin * (1.0 - a)
    srgb = np.clip(np.round(linear_to_srgb(comp) * 255.0), 0, 255).astype(np.uint8)
    cv2.imwrite(out_path, cv2.cvtColor(srgb, cv2.COLOR_RGB2BGR))
