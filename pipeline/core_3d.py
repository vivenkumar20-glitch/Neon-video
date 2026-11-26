"""Stage 1: Metric 3D camera rig, near-plane-safe UV homography, and historic material UV synthesis.

All scene coordinates use right-handed world meters (+X right, +Y forward, +Z up).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

NEAR_CLIP_Z = 0.05


@dataclass(frozen=True)
class Camera3D:
    """Metric pinhole camera in right-handed world meters (+X right, +Y forward, +Z up)."""

    eye: np.ndarray          # (3,) float64
    target: np.ndarray       # (3,) float64
    width: int
    height: int
    fov_deg: float = 42.0
    up: np.ndarray = None    # (3,) float64, default (0, 0, 1)
    principal_shift: Tuple[float, float] = (0.0, 0.0)  # fractional shift of optical center

    def __post_init__(self) -> None:
        eye = np.asarray(self.eye, dtype=np.float64)
        target = np.asarray(self.target, dtype=np.float64)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64) if self.up is None else np.asarray(self.up, dtype=np.float64)
        object.__setattr__(self, "eye", eye)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "up", up)

    @property
    def basis(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (right, cam_down, forward) orthonormal vectors in world space.

        In camera space: +x_c right, +y_c down (image rows), +z_c forward.
        """
        fwd = self.target - self.eye
        fwd_norm = np.linalg.norm(fwd)
        if fwd_norm < 1e-9:
            fwd = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            fwd = fwd / fwd_norm
        right = np.cross(fwd, self.up)
        r_norm = np.linalg.norm(right)
        if r_norm < 1e-9:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            right = right / r_norm
        cam_up = np.cross(right, fwd)
        cam_down = -cam_up
        return right, cam_down, fwd

    @property
    def R(self) -> np.ndarray:
        """3x3 world-to-camera rotation matrix mapping world delta to (x_c, y_c_down, z_c_fwd)."""
        right, cam_down, fwd = self.basis
        return np.vstack([right, cam_down, fwd])

    @property
    def K(self) -> np.ndarray:
        """3x3 camera intrinsic matrix."""
        f = 0.5 * self.height / math.tan(math.radians(self.fov_deg) * 0.5)
        cx = self.width * (0.5 + self.principal_shift[0])
        cy = self.height * (0.5 + self.principal_shift[1])
        return np.array(
            [[f, 0.0, cx],
             [0.0, f, cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def world_to_cam(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64)
        return (pts - self.eye) @ self.R.T

    def project(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project (N, 3) or (3,) world points -> (uv_px, z_cam)."""
        single = np.asarray(pts).ndim == 1
        p = np.atleast_2d(np.asarray(pts, dtype=np.float64))
        pc = self.world_to_cam(p)
        z = pc[:, 2]
        z_safe = np.where(np.abs(z) < 1e-6, 1e-6, z)
        K = self.K
        u = K[0, 0] * (pc[:, 0] / z_safe) + K[0, 2]
        v = K[1, 1] * (pc[:, 1] / z_safe) + K[1, 2]
        uv = np.stack([u, v], axis=-1)
        if single:
            return uv[0], float(z[0])
        return uv, z

    def clip_segment_near(
        self, a: np.ndarray, b: np.ndarray, z_near: float = NEAR_CLIP_Z
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Clip 3D world segment [a, b] against z_cam >= z_near before projection."""
        R = self.R
        pa = R @ (np.asarray(a, dtype=np.float64) - self.eye)
        pb = R @ (np.asarray(b, dtype=np.float64) - self.eye)
        za, zb = pa[2], pb[2]
        if za < z_near and zb < z_near:
            return None
        if za >= z_near and zb >= z_near:
            return np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        t = (z_near - za) / (zb - za)
        p_clip = np.asarray(a, dtype=np.float64) + t * (np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64))
        if za < z_near:
            return p_clip, np.asarray(b, dtype=np.float64)
        return np.asarray(a, dtype=np.float64), p_clip

    def clip_polygon_near(self, poly_world: Sequence[np.ndarray], z_near: float = NEAR_CLIP_Z) -> List[np.ndarray]:
        """Sutherland-Hodgman clip of a 3D world polygon against z_cam >= z_near."""
        pts = [np.asarray(p, dtype=np.float64) for p in poly_world]
        if len(pts) < 3:
            return []
        R = self.R
        out: List[np.ndarray] = []
        n = len(pts)
        for i in range(n):
            a = pts[i]
            b = pts[(i + 1) % n]
            za = float((R @ (a - self.eye))[2])
            zb = float((R @ (b - self.eye))[2])
            a_in = za >= z_near
            b_in = zb >= z_near
            if a_in and b_in:
                out.append(b)
            elif a_in and not b_in:
                t = (z_near - za) / (zb - za)
                out.append(a + t * (b - a))
            elif (not a_in) and b_in:
                t = (z_near - za) / (zb - za)
                out.append(a + t * (b - a))
                out.append(b)
        return out

    def project_clipped_polygon(self, poly_world: Sequence[np.ndarray], z_near: float = NEAR_CLIP_Z) -> np.ndarray:
        clipped = self.clip_polygon_near(poly_world, z_near=z_near)
        if len(clipped) < 3:
            return np.zeros((0, 2), dtype=np.float64)
        uv, _ = self.project(np.array(clipped))
        return uv

    def plane_homography(
        self,
        p0: np.ndarray,
        e1: np.ndarray,
        e2: np.ndarray,
        uv_w: int,
        uv_h: int,
        phys_w: float,
        phys_h: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute exact 3x3 UV-to-screen homography from 3D plane basis.

        H_uv_to_screen = K @ [R @ e1, R @ e2, R @ (p0 - eye)] @ H_uv_to_plane
        Returns (H_uv_to_screen, M_cam_from_uv) where M_cam_from_uv maps [u, v, 1]^T to camera-space 3D.
        """
        p0 = np.asarray(p0, dtype=np.float64)
        e1 = np.asarray(e1, dtype=np.float64)
        e2 = np.asarray(e2, dtype=np.float64)
        H_uv_to_plane = np.array(
            [[phys_w / max(1.0, float(uv_w)), 0.0, 0.0],
             [0.0, phys_h / max(1.0, float(uv_h)), 0.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        R = self.R
        basis_cam = np.column_stack([R @ e1, R @ e2, R @ (p0 - self.eye)])
        M_cam_from_uv = basis_cam @ H_uv_to_plane
        H_uv_to_screen = self.K @ M_cam_from_uv
        return H_uv_to_screen, M_cam_from_uv

    def ray_plane_depth_and_dist(
        self,
        p0: np.ndarray,
        e1: np.ndarray,
        e2: np.ndarray,
        phys_w: float,
        phys_h: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute per-pixel exact ray-plane depth z_cam, ray distance dist, and valid UV bounds mask.

        Evaluated efficiently on a 1/2-res grid or full-res grid as needed; returns full (H, W) arrays.
        """
        H, W = self.height, self.width
        K = self.K
        R = self.R
        p0 = np.asarray(p0, dtype=np.float64)
        e1 = np.asarray(e1, dtype=np.float64)
        e2 = np.asarray(e2, dtype=np.float64)
        normal_w = np.cross(e1, e2)
        n_norm = np.linalg.norm(normal_w)
        if n_norm < 1e-12:
            return np.zeros((H, W), np.float32), np.zeros((H, W), np.float32), np.zeros((H, W), bool)
        normal_w = normal_w / n_norm

        # Plane equation in camera space: n_c . X_c = d_c
        n_c = R @ normal_w
        p0_c = R @ (p0 - self.eye)
        d_c = float(np.dot(n_c, p0_c))

        # Build screen ray directions in camera space: [ (x - cx)/fx, (y - cy)/fy, 1 ]
        xs = (np.arange(W, dtype=np.float32) - float(K[0, 2])) / float(K[0, 0])
        ys = (np.arange(H, dtype=np.float32) - float(K[1, 2])) / float(K[1, 1])
        denom = (
            np.float32(n_c[0]) * xs[None, :]
            + np.float32(n_c[1]) * ys[:, None]
            + np.float32(n_c[2])
        )
        valid = np.abs(denom) > 1e-6
        denom_safe = np.where(valid, denom, 1.0)
        z_cam = np.float32(d_c) / denom_safe
        valid &= z_cam > np.float32(NEAR_CLIP_Z)

        # Ray distance = z_cam * sqrt(rx^2 + ry^2 + 1)
        ray_len = np.sqrt(xs[None, :] ** 2 + ys[:, None] ** 2 + 1.0, dtype=np.float32)
        dist = np.where(valid, z_cam * ray_len, 1e6).astype(np.float32)

        # Check plane physical UV bounds via inverse of basis_cam
        # X_c = z_cam * [rx, ry, 1] -> delta_w = R^T @ (X_c - p0_c)
        e1_c = (R @ e1).astype(np.float32)
        e2_c = (R @ e2).astype(np.float32)
        e1_len2 = float(np.dot(e1_c, e1_c)) + 1e-12
        e2_len2 = float(np.dot(e2_c, e2_c)) + 1e-12
        dx_c = z_cam * xs[None, :] - np.float32(p0_c[0])
        dy_c = z_cam * ys[:, None] - np.float32(p0_c[1])
        dz_c = z_cam - np.float32(p0_c[2])
        u_m = (dx_c * e1_c[0] + dy_c * e1_c[1] + dz_c * e1_c[2]) / e1_len2
        v_m = (dx_c * e2_c[0] + dy_c * e2_c[1] + dz_c * e2_c[2]) / e2_len2
        valid &= (u_m >= 0.0) & (u_m <= phys_w) & (v_m >= 0.0) & (v_m <= phys_h)
        z_cam = np.where(valid, z_cam, 1e6).astype(np.float32)
        return z_cam, dist, valid


def warp_plane_premul(
    cam: Camera3D,
    uv_rgba_linear: np.ndarray,
    p0: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    phys_w: float,
    phys_h: float,
    fog_color_linear: Optional[np.ndarray] = None,
    fog_start: float = 12.0,
    fog_density: float = 0.025,
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp a rectangular linear RGBA UV painting onto the camera frame via exact 3D homography.

    - Converts to premultiplied RGBA before warping.
    - Masks warped pixels by exact ray-plane depth z_cam > 0.05.
    - Blends exponential ray-distance fog: 1 - exp(-max(0, dist - fog_start) * fog_density).
    Returns (warped_premul_rgba_linear, z_cam_buffer).
    """
    uv_h, uv_w = uv_rgba_linear.shape[:2]
    H_mat, _ = cam.plane_homography(p0, e1, e2, uv_w, uv_h, phys_w, phys_h)

    premul = uv_rgba_linear.astype(np.float32).copy()
    alpha = premul[..., 3:4]
    premul[..., :3] *= alpha

    warped = cv2.warpPerspective(
        premul,
        H_mat,
        (cam.width, cam.height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0.0, 0.0, 0.0, 0.0),
    )

    z_cam, dist, valid = cam.ray_plane_depth_and_dist(p0, e1, e2, phys_w, phys_h)
    valid_f = valid.astype(np.float32)[..., None]
    warped *= valid_f

    if fog_color_linear is not None and fog_density > 0.0:
        fog_t = (1.0 - np.exp(-np.maximum(0.0, dist - fog_start) * fog_density)).astype(np.float32)[..., None]
        fog_rgb = np.asarray(fog_color_linear, dtype=np.float32).reshape(1, 1, 3)
        a_w = warped[..., 3:4]
        warped[..., :3] = warped[..., :3] * (1.0 - fog_t) + fog_rgb * a_w * fog_t

    return warped, z_cam


def composite_depth_premul(
    dst_rgb: np.ndarray,
    dst_z: np.ndarray,
    src_premul_rgba: np.ndarray,
    src_z: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Composite premultiplied linear RGBA layer into dst_rgb using per-pixel depth buffer."""
    a = np.clip(src_premul_rgba[..., 3], 0.0, 1.0)
    nearer = (a > 1e-4) & (src_z < dst_z)
    mask = nearer[..., None]
    a3 = a[..., None]
    out_rgb = np.where(mask, src_premul_rgba[..., :3] + dst_rgb * (1.0 - a3), dst_rgb)
    out_z = np.where(nearer & (a > 0.5), src_z, dst_z)
    return out_rgb, out_z


def over_premul(dst_rgb: np.ndarray, src_premul_rgba: np.ndarray) -> np.ndarray:
    """Standard Porter-Duff Over compositing for premultiplied linear RGBA over opaque linear RGB."""
    a = np.clip(src_premul_rgba[..., 3:4], 0.0, 1.0)
    return src_premul_rgba[..., :3] + dst_rgb * (1.0 - a)


# ---------------------------------------------------------------------------
# Historic Surface Generators (anchored in physical UV meters)
# ---------------------------------------------------------------------------

def _bicubic_noise(h: int, w: int, grid_h: int, grid_w: int, rng: np.random.Generator) -> np.ndarray:
    base = rng.standard_normal((max(2, grid_h), max(2, grid_w))).astype(np.float32)
    up = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    std = float(np.std(up))
    return up / (std + 1e-6)


def voronoi_masonry(
    h: int,
    w: int,
    phys_w: float,
    phys_h: float,
    stone_w_m: float = 0.42,
    stone_h_m: float = 0.24,
    mortar_m: float = 0.018,
    base_rgb: Tuple[float, float, float] = (0.28, 0.25, 0.21),
    mortar_rgb: Tuple[float, float, float] = (0.11, 0.10, 0.09),
    weathering: float = 0.35,
    seed: int = 101,
) -> np.ndarray:
    """Generate historic irregular stone/masonry in linear RGBA via vectorized 3x3 Voronoi (d2 - d1).

    Replaces uniform rectangular grids with jittered seed centers in physical meters,
    perturbs lookup coordinates with smooth bicubic noise, and varies stone pigmentation,
    erosion, and mortar crevices per cell.
    """
    rng = np.random.default_rng(seed)
    nx = max(3, int(math.ceil(phys_w / stone_w_m)) + 2)
    ny = max(3, int(math.ceil(phys_h / stone_h_m)) + 2)

    # Physical coordinates with smooth bicubic organic warp
    u = np.linspace(0.0, phys_w, w, dtype=np.float32)[None, :]
    v = np.linspace(0.0, phys_h, h, dtype=np.float32)[:, None]
    warp_x = _bicubic_noise(h, w, max(4, ny // 2), max(4, nx // 2), rng) * (stone_w_m * 0.09)
    warp_y = _bicubic_noise(h, w, max(4, ny // 2), max(4, nx // 2), rng) * (stone_h_m * 0.11)

    # Convert to cell coordinates (staggered courses + jitter)
    cy_f = (v + warp_y) / stone_h_m
    cx_f = (u + warp_x) / stone_w_m
    iy = np.floor(cy_f).astype(np.int32)
    ix = np.floor(cx_f).astype(np.int32)

    # Seed table for jittered coursed-ashlar cell centers & correlated natural stone pigmentation
    jitter_x = rng.uniform(0.28, 0.72, size=(ny + 4, nx + 4)).astype(np.float32)
    jitter_y = rng.uniform(0.40, 0.60, size=(ny + 4, nx + 4)).astype(np.float32)
    # Course stagger on alternating rows
    row_stagger = (np.arange(ny + 4) % 2)[:, None].astype(np.float32) * 0.50
    # Correlated luminance variation + tiny ochre/cool-slate warmth shift (never independent pink/green RGB!)
    cell_lum = rng.normal(0.0, 1.0, size=(ny + 4, nx + 4, 1)).astype(np.float32)
    cell_warm = rng.normal(0.0, 1.0, size=(ny + 4, nx + 4, 1)).astype(np.float32)
    cell_tone = (
        cell_lum * np.array([0.042, 0.039, 0.035], dtype=np.float32)
        + cell_warm * np.array([0.014, 0.006, -0.012], dtype=np.float32)
    )

    d1 = np.full((h, w), 1e9, dtype=np.float32)
    d2 = np.full((h, w), 1e9, dtype=np.float32)
    best_id_y = np.zeros((h, w), dtype=np.int32)
    best_id_x = np.zeros((h, w), dtype=np.int32)

    for dy in (-1, 0, 1):
        ny_idx = np.clip(iy + dy + 2, 0, ny + 3)
        stagger = row_stagger[ny_idx, 0]
        for dx in (-1, 0, 1):
            nx_idx = np.clip(ix + dx + 2, 0, nx + 3)
            px = (ix + dx).astype(np.float32) + jitter_x[ny_idx, nx_idx] + stagger * 0.45
            py = (iy + dy).astype(np.float32) + jitter_y[ny_idx, nx_idx]
            # Weighted Chebyshev/Euclidean metric gives authentic coursed historic ashlar blocks with organic edges
            adx = np.abs(cx_f - px) * stone_w_m
            ady = np.abs(cy_f - py) * stone_h_m
            eucl = np.sqrt(adx * adx + ady * ady, dtype=np.float32)
            cheb = np.maximum(adx * 0.72, ady * 1.18)
            dist = 0.35 * eucl + 0.65 * cheb
            closer1 = dist < d1
            d2 = np.where(closer1, d1, np.minimum(d2, dist))
            d1 = np.where(closer1, dist, d1)
            best_id_y = np.where(closer1, ny_idx, best_id_y)
            best_id_x = np.where(closer1, nx_idx, best_id_x)

    crevice = d2 - d1
    mortar_mask = np.clip((mortar_m - crevice) / (mortar_m * 0.65 + 1e-6), 0.0, 1.0)
    bevel = np.clip((crevice - mortar_m) / (mortar_m * 2.2 + 1e-6), 0.0, 1.0)

    # Per-stone color variation + surface erosion + mossy/damp vertical weathering
    base = np.asarray(base_rgb, dtype=np.float32).reshape(1, 1, 3)
    mortar = np.asarray(mortar_rgb, dtype=np.float32).reshape(1, 1, 3)
    stone_shift = cell_tone[best_id_y, best_id_x]
    fine_grain = _bicubic_noise(h, w, max(8, h // 12), max(8, w // 12), rng)[..., None] * 0.022
    broad_damp = _bicubic_noise(h, w, 6, 6, rng)[..., None] * (0.042 * weathering)

    # Subtle cracks using high-frequency ridge noise
    crack_n = np.abs(_bicubic_noise(h, w, max(10, h // 18), max(10, w // 18), rng))
    cracks = np.clip((0.045 - crack_n) / 0.045, 0.0, 1.0)[..., None] * (0.30 * weathering)

    # Rainwater staining from top and rising damp near bottom
    v_norm = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    damp_grad = (0.14 * weathering) * (v_norm ** 2.2)

    stone_col = base + stone_shift + fine_grain + broad_damp - damp_grad - cracks * 0.08
    stone_col *= (0.86 + 0.16 * bevel[..., None])

    rgb = stone_col * (1.0 - mortar_mask[..., None]) + mortar * mortar_mask[..., None]
    rgb = np.clip(rgb, 0.01, 0.98)
    rgba = np.concatenate([rgb, np.ones((h, w, 1), dtype=np.float32)], axis=-1)
    return rgba


def wood_planks_uv(
    h: int,
    w: int,
    n_planks: int = 8,
    base_rgb: Tuple[float, float, float] = (0.18, 0.11, 0.06),
    seed: int = 202,
) -> np.ndarray:
    """Painterly weathered timber planks in linear RGBA."""
    rng = np.random.default_rng(seed)
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    plank_idx = np.clip((y * n_planks).astype(np.int32), 0, n_planks - 1)
    plank_local = (y * n_planks) - plank_idx

    # Organic grain rings modulated by isotropic noise
    n_low = _bicubic_noise(h, w, 8, 12, rng)
    n_mid = _bicubic_noise(h, w, 24, 18, rng)
    grain = np.sin((plank_local * 18.0 + n_low * 2.2 + x * 1.4) * math.pi) * 0.5 + 0.5
    grain = (grain ** 1.6) * 0.14 + n_mid * 0.025

    plank_tones = rng.normal(0.0, 0.025, size=(n_planks, 3)).astype(np.float32)
    base = np.asarray(base_rgb, dtype=np.float32).reshape(1, 1, 3)
    rgb = base + plank_tones[plank_idx[:, 0]][:, None, :] + grain[..., None] * np.array([0.42, 0.28, 0.16], np.float32)

    # Plank seams
    seam = np.minimum(plank_local, 1.0 - plank_local)
    seam_dark = np.clip(seam / 0.045, 0.0, 1.0)[..., None]
    rgb *= (0.35 + 0.65 * seam_dark)

    rgb = np.clip(rgb, 0.01, 0.95)
    return np.concatenate([rgb, np.ones((h, w, 1), dtype=np.float32)], axis=-1)


def parchment_uv(
    h: int,
    w: int,
    base_rgb: Tuple[float, float, float] = (0.64, 0.55, 0.39),
    seed: int = 303,
) -> np.ndarray:
    """Warm 17th-century rag-paper / parchment UV in linear RGBA."""
    rng = np.random.default_rng(seed)
    n1 = _bicubic_noise(h, w, 6, 8, rng)
    n2 = _bicubic_noise(h, w, 20, 24, rng)
    n3 = _bicubic_noise(h, w, 64, 64, rng)
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    edge_age = np.clip((x ** 2 + y ** 2) * 0.22, 0.0, 0.45)

    base = np.asarray(base_rgb, dtype=np.float32).reshape(1, 1, 3)
    mod = (n1 * 0.045 + n2 * 0.022 + n3 * 0.012 - edge_age * 0.18)[..., None]
    rgb = np.clip(base + mod * np.array([1.0, 0.92, 0.78], np.float32), 0.03, 0.96)
    return np.concatenate([rgb, np.ones((h, w, 1), dtype=np.float32)], axis=-1)


# ---------------------------------------------------------------------------
# Deterministic Cubic Camera Path
# ---------------------------------------------------------------------------

def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


def smootherstep(t: float) -> float:
    t = max(0.0, min(1.0, float(t)))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def cubic_hermite(p0: np.ndarray, p1: np.ndarray, m0: np.ndarray, m1: np.ndarray, t: float) -> np.ndarray:
    t = max(0.0, min(1.0, float(t)))
    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1


class CubicCameraPath:
    """Keyframe cubic-spline camera trajectory with deliberate holds and smooth easing."""

    def __init__(
        self,
        keyframes: Sequence[Tuple[float, Sequence[float], Sequence[float], float]],
        width: int,
        height: int,
        principal_shift: Tuple[float, float] = (0.06, 0.0),
    ) -> None:
        # Each keyframe: (t_norm [0..1], eye(3), target(3), fov_deg)
        self.kfs = [
            (float(t), np.asarray(e, dtype=np.float64), np.asarray(tgt, dtype=np.float64), float(fov))
            for t, e, tgt, fov in keyframes
        ]
        self.width = width
        self.height = height
        self.principal_shift = principal_shift

    def camera_at(self, t_norm: float) -> Camera3D:
        t = max(0.0, min(1.0, float(t_norm)))
        if len(self.kfs) == 1 or t <= self.kfs[0][0]:
            _, e, tgt, fov = self.kfs[0]
            return Camera3D(e, tgt, self.width, self.height, fov, principal_shift=self.principal_shift)
        if t >= self.kfs[-1][0]:
            _, e, tgt, fov = self.kfs[-1]
            return Camera3D(e, tgt, self.width, self.height, fov, principal_shift=self.principal_shift)

        idx = 0
        for i in range(len(self.kfs) - 1):
            if self.kfs[i][0] <= t <= self.kfs[i + 1][0]:
                idx = i
                break
        t0, e0, g0, f0 = self.kfs[idx]
        t1, e1, g1, f1 = self.kfs[idx + 1]
        u = smootherstep((t - t0) / max(1e-6, t1 - t0))
        eye = e0 * (1.0 - u) + e1 * u
        tgt = g0 * (1.0 - u) + g1 * u
        fov = f0 * (1.0 - u) + f1 * u
        return Camera3D(eye, tgt, self.width, self.height, fov, principal_shift=self.principal_shift)
