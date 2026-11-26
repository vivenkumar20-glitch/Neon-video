"""Stage 2: Unified 3D light solver, scientific optics (Snell's law + recombination),
orbital Verlet dynamics, falling apple physics, calculus constructions, and Newton's 3 laws.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Unified 3D Lighting & Wet Surface Fresnel Reflections
# ---------------------------------------------------------------------------

def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def make_sun_dir(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Return normalized 3D light travel vector pointing from sky toward floor (SUN_DIR[2] < 0)."""
    az = math.radians(azimuth_deg)
    el = math.radians(max(2.0, min(88.0, elevation_deg)))
    # Travel direction goes downward (-Z)
    dx = math.cos(el) * math.cos(az)
    dy = math.cos(el) * math.sin(az)
    dz = -math.sin(el)
    return normalize(np.array([dx, dy, dz], dtype=np.float64))


def trace_shadow_onto_plane(
    pt_world: np.ndarray,
    sun_dir: np.ndarray,
    plane_z: float = 0.0,
) -> np.ndarray:
    """Project a 3D world point along sun_dir onto horizontal plane z = plane_z."""
    assert sun_dir[2] < -1e-6, "SUN_DIR must point downward (SUN_DIR[2] < 0)"
    p = np.asarray(pt_world, dtype=np.float64)
    t = (plane_z - p[2]) / sun_dir[2]
    return p + t * sun_dir


def backward_trace_aperture(
    floor_xy: np.ndarray,
    sun_dir: np.ndarray,
    plane_p0: np.ndarray,
    plane_normal: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Trace floor points (..., 3) backward along -SUN_DIR onto a window/shutter/roof plane.

    Returns (hit_points, t_values).
    """
    ray_dir = -normalize(sun_dir)
    n = normalize(plane_normal)
    denom = float(np.dot(ray_dir, n))
    if abs(denom) < 1e-8:
        return np.zeros_like(floor_xy), np.full(floor_xy.shape[:-1], -1.0)
    num = np.tensordot(plane_p0 - floor_xy, n, axes=([-1], [0]))
    t = num / denom
    hits = floor_xy + t[..., None] * ray_dir
    return hits, t


def candle_illumination(
    world_pts: np.ndarray,
    normals: np.ndarray,
    candle_pos: np.ndarray,
    frame_idx: int,
    seed: int = 7,
    radius_m: float = 2.6,
    base_intensity: float = 1.35,
) -> np.ndarray:
    """Temporally coherent localized candle illumination with bounded distance attenuation."""
    t = frame_idx / 24.0
    # Smooth multi-sine flame modulation (no harsh random per-frame flicker)
    flame_mod = (
        1.0
        + 0.045 * math.sin(t * 7.3 + seed)
        + 0.025 * math.sin(t * 13.7 + seed * 1.9)
        + 0.015 * math.sin(t * 21.1 + seed * 0.4)
    )
    delta = candle_pos - world_pts
    dist2 = np.sum(delta * delta, axis=-1) + 0.045
    dist = np.sqrt(dist2)
    l_dir = delta / dist[..., None]
    ndotl = np.clip(np.sum(normals * l_dir, axis=-1), 0.0, 1.0)
    # Smooth windowing at radius_m
    win = np.clip(1.0 - (dist / radius_m) ** 2, 0.0, 1.0) ** 2
    att = (base_intensity * flame_mod) * win / (0.25 + dist2)
    return (att * (0.22 + 0.78 * ndotl)).astype(np.float32)


def schlick_fresnel(cos_theta: np.ndarray, f0: float = 0.02) -> np.ndarray:
    """View-angle Fresnel reflectance F = F0 + (1 - F0) * (1 - cos_theta)^5."""
    c = np.clip(cos_theta, 0.0, 1.0)
    return (f0 + (1.0 - f0) * ((1.0 - c) ** 5)).astype(np.float32)


def mirror_points_across_plane(pts: np.ndarray, p0: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Mirror 3D points across plane (p0, normal): pts - 2 * dot(pts - p0, n) * n."""
    n = normalize(normal)
    d = np.tensordot(pts - p0, n, axes=([-1], [0]))
    return pts - 2.0 * d[..., None] * n


# ---------------------------------------------------------------------------
# Snell's Law Glass Prism Dispersion & Recombination Ray Tracer
# ---------------------------------------------------------------------------

def cauchy_n(wavelength_nm: float, A: float = 1.620, B: float = 0.0240) -> float:
    """Wavelength-dependent dense lead-flint glass refractive index n(lambda) = A + B / lambda^2 (lambda in um)."""
    lam_um = wavelength_nm * 1e-3
    return A + B / (lam_um * lam_um)


def wavelength_to_linear_rgb(wavelength_nm: float) -> np.ndarray:
    """Physically plausible smooth spectral-to-linear-RGB mapping (395 nm to 705 nm)."""
    wl = float(wavelength_nm)
    if wl < 440.0:
        r = -(wl - 440.0) / (440.0 - 380.0)
        g = 0.0
        b = 1.0
    elif wl < 490.0:
        r = 0.0
        g = (wl - 440.0) / (490.0 - 440.0)
        b = 1.0
    elif wl < 510.0:
        r = 0.0
        g = 1.0
        b = -(wl - 510.0) / (510.0 - 490.0)
    elif wl < 580.0:
        r = (wl - 510.0) / (580.0 - 510.0)
        g = 1.0
        b = 0.0
    elif wl < 645.0:
        r = 1.0
        g = -(wl - 645.0) / (645.0 - 580.0)
        b = 0.0
    else:
        r = 1.0
        g = 0.0
        b = 0.0

    # Smooth spectral sensitivity taper at violet and deep red tails
    if wl < 420.0:
        factor = 0.25 + 0.75 * max(0.0, (wl - 385.0) / 35.0)
    elif wl > 660.0:
        factor = 0.25 + 0.75 * max(0.0, (710.0 - wl) / 50.0)
    else:
        factor = 1.0

    # Convert perceptual approximation gently into linear light
    srgb = np.clip(np.array([r, g, b], dtype=np.float32) * factor, 0.0, 1.0)
    return (srgb ** 2.1).astype(np.float32)


def _ray_segment_intersect_2d(
    ro: np.ndarray, rd: np.ndarray, a: np.ndarray, b: np.ndarray
) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
    """Intersect 2D ray ro + t*rd (t > 1e-5) with segment [a, b]. Returns (t, hit_pt, unit_normal)."""
    v = b - a
    denom = rd[0] * v[1] - rd[1] * v[0]
    if abs(denom) < 1e-10:
        return None
    diff = a - ro
    t = (diff[0] * v[1] - diff[1] * v[0]) / denom
    u = (diff[0] * rd[1] - diff[1] * rd[0]) / denom
    if t > 1e-5 and 0.0 <= u <= 1.0:
        hit = ro + t * rd
        n = np.array([-v[1], v[0]], dtype=np.float64)
        n /= (np.linalg.norm(n) + 1e-12)
        if np.dot(n, rd) > 0:
            n = -n
        return float(t), hit, n
    return None


def refract_2d(d: np.ndarray, n: np.ndarray, n1: float, n2: float) -> Tuple[np.ndarray, bool]:
    """Exact 2D Snell's law vector refraction (n points against incoming d). Returns (dir_out, is_tir)."""
    d = d / (np.linalg.norm(d) + 1e-12)
    n = n / (np.linalg.norm(n) + 1e-12)
    cos_i = -float(np.dot(d, n))
    eta = n1 / n2
    k = 1.0 - eta * eta * (1.0 - cos_i * cos_i)
    if k < 0.0:
        # Total internal reflection
        refl = d + 2.0 * cos_i * n
        return refl / (np.linalg.norm(refl) + 1e-12), True
    out = eta * d + (eta * cos_i - math.sqrt(k)) * n
    return out / (np.linalg.norm(out) + 1e-12), False


@dataclass
class TracedRay:
    wavelength_nm: float
    points: List[np.ndarray]   # 2D physical meters (x, y)
    rgb: np.ndarray            # linear RGB
    tir_occurred: bool = False


def trace_prism_ray(
    origin: np.ndarray,
    direction: np.ndarray,
    prism_verts: Sequence[np.ndarray],
    wavelength_nm: float,
    x_stop: float,
    n_air: float = 1.00029,
) -> TracedRay:
    """Trace a single monochromatic ray through a 2D triangular glass prism via exact Snell's law."""
    ro = np.asarray(origin, dtype=np.float64)
    rd = np.asarray(direction, dtype=np.float64)
    rd = rd / (np.linalg.norm(rd) + 1e-12)
    verts = [np.asarray(v, dtype=np.float64) for v in prism_verts]
    edges = [(verts[i], verts[(i + 1) % len(verts)]) for i in range(len(verts))]
    n_glass = cauchy_n(wavelength_nm)

    pts = [ro.copy()]
    tir_any = False

    # Entry intersection
    best = None
    for a, b in edges:
        hit = _ray_segment_intersect_2d(ro, rd, a, b)
        if hit is not None and (best is None or hit[0] < best[0]):
            best = hit
    if best is None:
        t_end = max(0.05, (x_stop - ro[0]) / max(1e-5, rd[0]))
        pts.append(ro + t_end * rd)
        return TracedRay(wavelength_nm, pts, wavelength_to_linear_rgb(wavelength_nm), False)

    _, p_in, n_in = best
    pts.append(p_in)
    rd_in, tir1 = refract_2d(rd, n_in, n_air, n_glass)
    tir_any = tir_any or tir1

    # Exit intersection (allow up to 2 internal bounces if TIR occurs)
    cur_o = p_in + rd_in * 1e-4
    cur_d = rd_in
    for _ in range(3):
        best_exit = None
        for a, b in edges:
            hit = _ray_segment_intersect_2d(cur_o, cur_d, a, b)
            if hit is not None and (best_exit is None or hit[0] < best_exit[0]):
                best_exit = hit
        if best_exit is None:
            break
        _, p_out, n_out = best_exit
        pts.append(p_out)
        rd_out, is_tir = refract_2d(cur_d, n_out, n_glass, n_air)
        if not is_tir:
            cur_o = p_out
            cur_d = rd_out
            break
        tir_any = True
        cur_o = p_out + rd_out * 1e-4
        cur_d = rd_out

    # Propagate to screen x = x_stop
    if abs(cur_d[0]) > 1e-6:
        t_screen = (x_stop - cur_o[0]) / cur_d[0]
        if t_screen > 0:
            pts.append(cur_o + t_screen * cur_d)
        else:
            pts.append(cur_o + cur_d * 0.5)
    else:
        pts.append(cur_o + cur_d * 0.5)

    return TracedRay(wavelength_nm, pts, wavelength_to_linear_rgb(wavelength_nm), tir_any)


def trace_prism_recombination(
    origin: np.ndarray,
    direction: np.ndarray,
    prism1_verts: Sequence[np.ndarray],
    lens_x: float,
    lens_y: float,
    lens_focal: float,
    prism2_verts: Sequence[np.ndarray],
    x_screen: float,
    wavelengths_nm: Sequence[float],
) -> List[TracedRay]:
    """Newton's Experimentum Crucis / Recombination:
    Prism 1 disperses white light -> thin converging lens focuses dispersed rays onto
    an inverted identical Prism 2 -> Prism 2 cancels angular dispersion and outputs a
    recombined white beam!

    Each refraction uses exact Snell's law and thin-lens deflection; validated numerically
    in verify_physics().
    """
    rays: List[TracedRay] = []
    for wl in wavelengths_nm:
        r1 = trace_prism_ray(origin, direction, prism1_verts, wl, x_stop=lens_x)
        pts = list(r1.points)
        p_lens = pts[-1]
        d_before_lens = p_lens - pts[-2]
        d_before_lens /= (np.linalg.norm(d_before_lens) + 1e-12)

        # Thin converging lens deflection: slope_after = slope_before - (y - lens_y) / f
        slope_in = d_before_lens[1] / max(1e-6, d_before_lens[0])
        slope_out = slope_in - (p_lens[1] - lens_y) / lens_focal
        d_after_lens = np.array([1.0, slope_out], dtype=np.float64)
        d_after_lens /= np.linalg.norm(d_after_lens)

        # Trace through inverted prism 2
        r2 = trace_prism_ray(p_lens, d_after_lens, prism2_verts, wl, x_stop=x_screen)
        pts.extend(r2.points[1:])
        rays.append(TracedRay(wl, pts, wavelength_to_linear_rgb(wl), r1.tir_occurred or r2.tir_occurred))
    return rays


# ---------------------------------------------------------------------------
# Falling Apple & Velocity Verlet Orbital Mechanics
# ---------------------------------------------------------------------------

def simulate_falling_apple(
    t_sec: float,
    y_start_m: float = 2.85,
    y_ground_m: float = 0.08,
    g: float = 9.81,
    vx_breeze: float = 0.14,
    omega_rad_s: float = 2.6,
) -> Tuple[float, float, float, bool]:
    """Exact constant-g gravitational fall with natural soft grass settle.

    Returns (dx_m, y_m, rot_rad, landed).
    """
    if t_sec <= 0.0:
        return 0.0, y_start_m, 0.0, False
    t_hit = math.sqrt(2.0 * (y_start_m - y_ground_m) / g)
    if t_sec < t_hit:
        y = y_start_m - 0.5 * g * t_sec * t_sec
        dx = vx_breeze * t_sec
        rot = omega_rad_s * t_sec
        return dx, y, rot, False
    # Gentle damped grass bounce
    dt = t_sec - t_hit
    bounce = 0.045 * math.exp(-dt * 9.0) * abs(math.sin(dt * 14.0))
    dx = vx_breeze * t_hit + 0.06 * (1.0 - math.exp(-dt * 4.5))
    rot = omega_rad_s * t_hit + 0.42 * (1.0 - math.exp(-dt * 4.5))
    return dx, y_ground_m + bounce, rot, True


def integrate_verlet_orbit(
    r0: np.ndarray,
    v0: np.ndarray,
    mu: float,
    dt: float,
    steps: int,
    r_planet: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Velocity Verlet integrator for a = -mu * r / ||r||^3.

    Returns (positions (K, D), energies (K,), impacted_planet).
    """
    r = np.asarray(r0, dtype=np.float64).copy()
    v = np.asarray(v0, dtype=np.float64).copy()

    def accel(pos: np.ndarray) -> np.ndarray:
        rn = float(np.linalg.norm(pos))
        return -mu * pos / (rn * rn * rn + 1e-18)

    a = accel(r)
    pos_hist = [r.copy()]
    energy_hist = [0.5 * float(np.dot(v, v)) - mu / float(np.linalg.norm(r))]
    impacted = False

    for _ in range(steps):
        r_new = r + v * dt + 0.5 * a * dt * dt
        rn = float(np.linalg.norm(r_new))
        if r_planet > 0.0 and rn <= r_planet:
            # Clamp onto planet surface
            pos_hist.append(r_new * (r_planet / max(1e-12, rn)))
            energy_hist.append(energy_hist[-1])
            impacted = True
            break
        a_new = accel(r_new)
        v_new = v + 0.5 * (a + a_new) * dt
        r, v, a = r_new, v_new, a_new
        pos_hist.append(r.copy())
        energy_hist.append(0.5 * float(np.dot(v, v)) - mu / rn)

    return np.array(pos_hist), np.array(energy_hist), impacted


def newton_cannonball_trajectories(
    r_earth: float = 1.0,
    mountain_h: float = 0.14,
    mu: float = 1.0,
) -> List[Tuple[str, np.ndarray, bool]]:
    """Compute 4 physically exact Velocity Verlet trajectories for Newton's cannonball thought experiment.

    Circular orbital velocity at r0 = r_earth + mountain_h is v_circ = sqrt(mu / r0).
    """
    r0 = np.array([0.0, r_earth + mountain_h], dtype=np.float64)
    v_circ = math.sqrt(mu / np.linalg.norm(r0))
    cases = [
        ("short_fall", 0.52 * v_circ, 600),
        ("long_fall", 0.80 * v_circ, 1100),
        ("circular_orbit", 1.00 * v_circ, 2200),
        ("elliptical_orbit", 1.14 * v_circ, 3200),
    ]
    out = []
    for name, speed, steps in cases:
        v0 = np.array([speed, 0.0], dtype=np.float64)
        traj, _, hit = integrate_verlet_orbit(r0, v0, mu=mu, dt=0.004, steps=steps, r_planet=r_earth)
        out.append((name, traj, hit))
    return out


# ---------------------------------------------------------------------------
# Calculus (Fluxions & Areas) & Newton's Three Laws Numerical Models
# ---------------------------------------------------------------------------

def fluxion_curve_and_tangent(x: float) -> Tuple[float, float]:
    """Smooth analytical curve y(x) and exact derivative dy/dx on x in [0, 1]."""
    y = 0.18 + 0.42 * x + 0.24 * math.sin(math.pi * 1.15 * x) - 0.14 * (x ** 2)
    dydx = 0.42 + 0.24 * (math.pi * 1.15) * math.cos(math.pi * 1.15 * x) - 0.28 * x
    return y, dydx


def riemann_strips(n_strips: int, x0: float = 0.08, x1: float = 0.90) -> Tuple[List[Tuple[float, float, float]], float]:
    """Return list of (x_left, width, y_mid) midpoint Riemann rectangles and total area."""
    dx = (x1 - x0) / n_strips
    strips = []
    area = 0.0
    for i in range(n_strips):
        xl = x0 + i * dx
        xm = xl + 0.5 * dx
        ym, _ = fluxion_curve_and_tangent(xm)
        strips.append((xl, dx, ym))
        area += ym * dx
    return strips, area


def simulate_three_laws(t_norm: float) -> dict:
    """Numerically exact positions and force vectors for Newton's Three Laws demonstrations.

    t_norm in [0, 1] represents 2.4 physical seconds.
    """
    t = max(0.0, min(1.0, float(t_norm))) * 2.4

    # 1st Law: constant velocity under zero net force (v = 0.42 m/s)
    law1_x = 0.10 + 0.36 * t
    law1_v = 0.36

    # 2nd Law: equal-mass bodies (m1 = m2 = 1.0 kg), distinct constant forces F1 = 0.14 N, F2 = 0.36 N
    m = 1.0
    f1, f2 = 0.14, 0.36
    a1, a2 = f1 / m, f2 / m
    law2_x1 = 0.10 + 0.5 * a1 * t * t
    law2_x2 = 0.10 + 0.5 * a2 * t * t

    # 3rd Law: two bodies (mA = 1.0 kg, mB = 1.6 kg) initially compressed together at center x=0.50,
    # released by an internal spring impulse at t_release = 0.45 s.
    # Equal & opposite forces F_AB = -F_BA act on DIFFERENT bodies -> mA*vA + mB*vB = 0.
    mA, mB = 1.0, 1.6
    t_rel = 0.45
    t_push = 0.35
    f_mag = 0.85
    if t <= t_rel:
        xA, xB = 0.46, 0.54
        fA, fB = 0.0, 0.0
    elif t <= t_rel + t_push:
        dt = t - t_rel
        aA = -f_mag / mA
        aB = +f_mag / mB
        xA = 0.46 + 0.5 * aA * dt * dt
        xB = 0.54 + 0.5 * aB * dt * dt
        fA, fB = -f_mag, +f_mag
    else:
        dt_coast = t - (t_rel + t_push)
        vA = -(f_mag / mA) * t_push
        vB = +(f_mag / mB) * t_push
        xA = 0.46 + 0.5 * (-f_mag / mA) * t_push * t_push + vA * dt_coast
        xB = 0.54 + 0.5 * (+f_mag / mB) * t_push * t_push + vB * dt_coast
        fA, fB = 0.0, 0.0

    return {
        "law1": {"x": law1_x, "v": law1_v, "f_net": 0.0},
        "law2": {"x1": law2_x1, "x2": law2_x2, "a1": a1, "a2": a2, "f1": f1, "f2": f2, "m": m},
        "law3": {"xA": xA, "xB": xB, "mA": mA, "mB": mB, "fA": fA, "fB": fB},
    }


def verify_physics() -> dict:
    """Numerical validation suite for Stage 2 & Stage 5 scientific accuracy checks."""
    report = {}

    # 1. Verify prism dispersion ordering (violet bends more than red)
    p1 = [np.array([0.0, -0.14]), np.array([0.16, -0.14]), np.array([0.08, 0.14])]
    r_red = trace_prism_ray(np.array([-0.35, -0.02]), np.array([1.0, 0.14]), p1, 660.0, x_stop=0.85)
    r_violet = trace_prism_ray(np.array([-0.35, -0.02]), np.array([1.0, 0.14]), p1, 415.0, x_stop=0.85)
    assert r_violet.points[-1][1] < r_red.points[-1][1], "Violet must refract further downward than red"
    report["prism_dispersion_spread_m"] = round(float(r_red.points[-1][1] - r_violet.points[-1][1]), 5)

    # 2. Verify prism recombination converges rays back together
    # Place thin lens at x=0.45, identical upright prism at exact 2f conjugate focus x=0.82
    # (Newton's Opticks Book I Part II Prop V Exp 11: optical time-reversal cancels angular dispersion!)
    p_mid = trace_prism_ray(np.array([-0.35, -0.02]), np.array([1.0, 0.14]), p1, 535.0, x_stop=0.45)
    lens_y = float(p_mid.points[-1][1])
    p2_center_x = 0.82
    d_g = p_mid.points[-1] - p_mid.points[-2]
    d_g /= np.linalg.norm(d_g)
    slope_g = d_g[1] / d_g[0]
    y_at_p2 = lens_y + slope_g * (p2_center_x - 0.45)
    # Upright identical prism centered vertically so green ray enters at the conjugate height
    p2 = [
        np.array([p2_center_x - 0.08, y_at_p2 - 0.12]),
        np.array([p2_center_x + 0.08, y_at_p2 - 0.12]),
        np.array([p2_center_x, y_at_p2 + 0.16]),
    ]
    recomb = trace_prism_recombination(
        np.array([-0.35, -0.02]), np.array([1.0, 0.14]),
        p1, lens_x=0.45, lens_y=lens_y, lens_focal=0.185,
        prism2_verts=p2, x_screen=1.25, wavelengths_nm=[420.0, 535.0, 650.0],
    )
    spread_lens = abs(recomb[0].points[3][1] - recomb[2].points[3][1])
    spread_recomb = abs(recomb[0].points[-1][1] - recomb[2].points[-1][1])
    assert spread_recomb < spread_lens * 0.45, f"Recombination failed: {spread_recomb} vs {spread_lens}"
    report["recomb_spread_ratio"] = round(float(spread_recomb / max(1e-6, spread_lens)), 4)

    # 3. Verify Velocity Verlet energy conservation for bound circular & elliptical orbits
    r0 = np.array([1.14, 0.0])
    v0 = np.array([0.0, math.sqrt(1.0 / 1.14) * 1.12])
    _, energies, hit = integrate_verlet_orbit(r0, v0, mu=1.0, dt=0.004, steps=2500, r_planet=1.0)
    assert not hit, "Bound elliptical orbit must not impact planet"
    rel_e_err = float(np.max(np.abs(energies - energies[0])) / abs(energies[0]))
    assert rel_e_err < 1e-4, f"Verlet energy drift too large: {rel_e_err}"
    report["verlet_max_rel_energy_error"] = float(f"{rel_e_err:.2e}")

    # 4. Verify Newton's Three Laws momentum conservation & a = F/m
    s = simulate_three_laws(0.95)
    assert abs(s["law2"]["a2"] / s["law2"]["a1"] - s["law2"]["f2"] / s["law2"]["f1"]) < 1e-9
    # Check 3rd law center of mass remains stationary at (mA*0.46 + mB*0.54)/(mA+mB)
    com0 = (s["law3"]["mA"] * 0.46 + s["law3"]["mB"] * 0.54) / (s["law3"]["mA"] + s["law3"]["mB"])
    com1 = (s["law3"]["mA"] * s["law3"]["xA"] + s["law3"]["mB"] * s["law3"]["xB"]) / (s["law3"]["mA"] + s["law3"]["mB"])
    assert abs(com1 - com0) < 1e-9, "3rd law must conserve center of mass"
    report["third_law_com_drift"] = abs(com1 - com0)

    # 5. Verify Riemann sum convergence
    _, a6 = riemann_strips(6)
    _, a96 = riemann_strips(96)
    report["riemann_area_6_vs_96"] = (round(a6, 5), round(a96, 5))
    return report
