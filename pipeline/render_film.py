"""Stage 5: Deterministic orchestrator for NEWTON — THE HIDDEN ORDER.

Supports:
  --mode verify     : Compile check, physics verification, model sheet, hero key frames, benchmark, & audio synth.
  --mode animatic   : Low-res (512x288) full 120s animatic with synced score.
  --mode render     : Full 2048x1152, 24 fps, 2,880-frame parallel checkpointed render, 1:1 600x450 crops,
                      temporal QC, master MP4 encode, standalone self-contained HTML, and render report.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .characters import render_model_sheet
from .core_3d import smoothstep
from .physics_optics import verify_physics
from .post_optics import finish_35mm
from .shots_1_4 import Shot1Renderer, Shot2Renderer, Shot3Renderer, Shot4Renderer
from .shots_5_8 import Shot5Renderer, Shot6Renderer, Shot7Renderer, Shot8Renderer
from .shots_9_11 import Shot9Renderer, Shot10Renderer, Shot11Renderer
from .sound_synth import synthesize_master_audio

FPS = 24
TOTAL_FRAMES = 2880  # 120.00 seconds exact


@dataclass(frozen=True)
class ShotSpec:
    shot_num: int
    title: str
    start_f: int
    end_f: int          # exclusive
    hero_local_f: int
    crop_centers: Tuple[Tuple[str, float, float], ...]  # (label, cx_frac, cy_frac)


SHOT_SPECS: List[ShotSpec] = [
    ShotSpec(1, "A World Before Him",      0,    216,  90,  (("light_amber_win", 0.62, 0.54), ("mid_manor_stone", 0.56, 0.62), ("hero_oak_hills", 0.22, 0.58))),
    ShotSpec(2, "The Curious Child",       216,  456,  52,  (("light_window", 0.68, 0.34), ("mid_desk_sketch", 0.22, 0.84), ("hero_child_mill", 0.44, 0.58))),
    ShotSpec(3, "Cambridge",               456,  720,  64,  (("light_gothic_sky", 0.58, 0.26), ("mid_voronoi_stone", 0.18, 0.48), ("hero_newton_court", 0.48, 0.72))),
    ShotSpec(4, "The Silent Return",       720,  960,  180, (("light_sky_clouds", 0.52, 0.26), ("mid_moor_hills", 0.34, 0.68), ("hero_newton_walk", 0.44, 0.80))),
    ShotSpec(5, "The Falling Apple",       960,  1272, 98,  (("light_day_moon", 0.80, 0.22), ("mid_apple_canopy", 0.26, 0.46), ("hero_newton_apple", 0.48, 0.72))),
    ShotSpec(6, "The Mathematics of Change", 1272, 1536, 92, (("light_candle_corner", 0.84, 0.20), ("mid_parchment_strips", 0.42, 0.64), ("hero_tangent_curve", 0.60, 0.44))),
    ShotSpec(7, "The Nature of Light",     1536, 1800, 148, (("light_recomb_beam", 0.82, 0.50), ("mid_prism_glass", 0.36, 0.52), ("hero_lens_spectrum", 0.56, 0.50))),
    ShotSpec(8, "Beyond the Earth",        1800, 2136, 205, (("light_orbit_glow", 0.50, 0.30), ("mid_earth_globe", 0.50, 0.52), ("hero_cannon_arc", 0.64, 0.38))),
    ShotSpec(9, "The Principia",           2136, 2448, 285, (("light_title_glow", 0.50, 0.32), ("mid_leather_page", 0.32, 0.68), ("hero_principia_type", 0.50, 0.50))),
    ShotSpec(10, "The Passing Years",      2448, 2664, 182, (("light_window_gold", 0.78, 0.40), ("mid_wall_mullion", 0.58, 0.50), ("hero_elder_newton", 0.42, 0.48))),
    ShotSpec(11, "The Hidden Order",       2664, 2880, 46,  (("light_moon_stars", 0.74, 0.26), ("mid_tree_geometry", 0.38, 0.60), ("hero_solar_finale", 0.50, 0.48))),
]

RENDERER_CLASSES = {
    1: Shot1Renderer,
    2: Shot2Renderer,
    3: Shot3Renderer,
    4: Shot4Renderer,
    5: Shot5Renderer,
    6: Shot6Renderer,
    7: Shot7Renderer,
    8: Shot8Renderer,
    9: Shot9Renderer,
    10: Shot10Renderer,
    11: Shot11Renderer,
}


def _save_rgb(path: str, rgb_u8: np.ndarray) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))


def _export_crop(rgb_u8: np.ndarray, cx_frac: float, cy_frac: float, out_path: str, cw: int = 600, ch: int = 450) -> None:
    h, w = rgb_u8.shape[:2]
    x0 = int(np.clip(round(cx_frac * w - cw * 0.5), 0, max(0, w - cw)))
    y0 = int(np.clip(round(cy_frac * h - ch * 0.5), 0, max(0, h - ch)))
    crop = rgb_u8[y0:y0 + ch, x0:x0 + cw]
    _save_rgb(out_path, crop)


def render_single_shot_worker(args: Tuple[int, int, int, str, str, int, bool]) -> Dict:
    """Worker process that renders one complete shot, streams to FFmpeg checkpoint clip,
    exports 1:1 600x450 macro crops + temporal QC samples, and frees RAM immediately.
    """
    shot_num, w, h, clip_path, diag_dir, crf, export_diagnostics = args
    spec = SHOT_SPECS[shot_num - 1]
    t0 = time.perf_counter()

    renderer = RENDERER_CLASSES[shot_num](w, h)
    t_init = time.perf_counter() - t0

    # For Shot 11 dissolve from Shot 10 elderly window (first 16 frames of Shot 11)
    shot10_tail = None
    if shot_num == 11:
        r10 = Shot10Renderer(w, h)
        shot10_tail, _ = r10.render(SHOT_SPECS[9].end_f - SHOT_SPECS[9].start_f - 1, SHOT_SPECS[9].end_f - 1)
        del r10

    os.makedirs(os.path.dirname(os.path.abspath(clip_path)), exist_ok=True)
    ff_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(FPS),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        clip_path,
    ]
    proc = subprocess.Popen(ff_cmd, stdin=subprocess.PIPE)

    n_frames = spec.end_f - spec.start_f
    rng_s = np.random.default_rng(7000 + shot_num)
    rand_f = int(rng_s.integers(10, max(11, n_frames - 10)))
    sample_locals = {0: "begin", n_frames // 2: "mid", n_frames - 1: "end", rand_f: "rand", spec.hero_local_f: "hero"}

    prev_frame_f32 = None
    adj_diffs = []
    t_render_start = time.perf_counter()

    assert proc.stdin is not None
    for local_f in range(n_frames):
        global_f = spec.start_f + local_f
        lin_rgb, profile = renderer.render(local_f, global_f)
        if shot10_tail is not None and local_f < 16:
            diss = smoothstep(local_f / 15.0)
            lin_rgb = shot10_tail * (1.0 - diss) + lin_rgb * diss

        rgb_u8 = finish_35mm(lin_rgb, global_f, profile)
        proc.stdin.write(rgb_u8.tobytes())

        # Temporal QC: measure adjacent frame MAE on a 1/4-res downsample to catch unexpected flicker
        if local_f % 4 == 0 or local_f % 4 == 1:
            small = cv2.resize(rgb_u8, (w // 4, h // 4), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            if local_f % 4 == 1 and prev_frame_f32 is not None:
                adj_diffs.append(float(np.mean(np.abs(small - prev_frame_f32))))
            prev_frame_f32 = small

        if export_diagnostics and local_f in sample_locals:
            tag = sample_locals[local_f]
            _save_rgb(os.path.join(diag_dir, "samples", f"shot_{shot_num:02d}_{tag}_f{global_f:04d}.png"), rgb_u8)
            if local_f == spec.hero_local_f and w >= 1920:
                for clabel, cx_f, cy_f in spec.crop_centers:
                    _export_crop(
                        rgb_u8, cx_f, cy_f,
                        os.path.join(diag_dir, "crops", f"shot_{shot_num:02d}_{clabel}_600x450.png"),
                        600, 450,
                    )

    proc.stdin.close()
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"FFmpeg failed encoding shot {shot_num} (exit {ret})")

    t_render = time.perf_counter() - t_render_start
    return {
        "shot": shot_num,
        "title": spec.title,
        "frames": n_frames,
        "init_sec": round(t_init, 2),
        "render_sec": round(t_render, 2),
        "ms_per_frame": round((t_render / max(1, n_frames)) * 1000.0, 1),
        "median_adj_mae": round(float(np.median(adj_diffs)) if adj_diffs else 0.0, 4),
        "max_adj_mae": round(float(np.max(adj_diffs)) if adj_diffs else 0.0, 4),
    }


def concat_and_mux(shot_clips: List[str], audio_wav: str, out_mp4: str, crf: int = 16) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_mp4)), exist_ok=True)
    list_file = os.path.join(os.path.dirname(os.path.abspath(shot_clips[0])), "concat_list.txt")
    with open(list_file, "w", encoding="utf-8") as f:
        for p in shot_clips:
            f.write(f"file '{os.path.abspath(p)}'\n")

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", list_file,
        "-i", audio_wav,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart",
        "-shortest",
        out_mp4,
    ]
    subprocess.run(cmd, check=True)


def probe_mp4(mp4_path: str) -> Dict:
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration,size:stream=codec_type,codec_name,width,height,r_frame_rate,nb_frames,sample_rate,channels",
        "-of", "json", mp4_path,
    ], text=True)
    return json.loads(out)


def build_self_contained_html(mp4_path: str, out_html_path: str) -> None:
    """Deliver a single self-contained HTML file with embedded synchronized MP4/AAC,
    pure black letterboxed cinema frame, and one quiet play button with nothing else on screen.
    """
    with open(mp4_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Newton — The Hidden Order</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
html,body{{width:100%;height:100%;background:#000;overflow:hidden;user-select:none}}
main{{position:fixed;inset:0;background:#000;display:flex;align-items:center;justify-content:center}}
.stage{{position:relative;width:100vw;max-width:177.7778vh;aspect-ratio:16/9;background:#000;display:flex;align-items:center;justify-content:center;cursor:pointer}}
video{{width:100%;height:100%;object-fit:contain;background:#000;display:block}}
button{{position:absolute;inset:0;margin:auto;width:82px;height:82px;border-radius:9999px;border:1px solid rgba(255,255,255,.32);background:rgba(0,0,0,.45);display:flex;align-items:center;justify-content:center;cursor:pointer;transition:opacity .45s ease,background .35s ease,border-color .35s ease}}
button:hover{{background:rgba(0,0,0,.68);border-color:rgba(255,255,255,.60)}}
button.hidden{{opacity:0;pointer-events:none}}
svg{{width:28px;height:28px;fill:rgba(255,255,255,.88);transform:translateX(2px)}}
</style>
</head>
<body>
<main>
  <div class="stage" id="stage">
    <video id="film" playsinline preload="auto" src="data:video/mp4;base64,{b64}"></video>
    <button id="playBtn" type="button" aria-label="Play film">
      <svg viewBox="0 0 24 24" aria-hidden="true"><polygon points="6,4 20,12 6,20"/></svg>
    </button>
  </div>
</main>
<script>
const v=document.getElementById('film'),b=document.getElementById('playBtn'),s=document.getElementById('stage');
function toggle(){{if(v.paused||v.ended){{if(v.ended)v.currentTime=0;v.play().catch(()=>{{}});}}else{{v.pause();}}}}
b.addEventListener('click',e=>{{e.stopPropagation();toggle();}});
s.addEventListener('click',toggle);
v.addEventListener('play',()=>b.classList.add('hidden'));
v.addEventListener('pause',()=>b.classList.remove('hidden'));
v.addEventListener('ended',()=>b.classList.remove('hidden'));
window.addEventListener('keydown',e=>{{if(e.code==='Space'){{e.preventDefault();toggle();}}}});
</script>
</body>
</html>
"""
    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write(html)


def run_verify(root: str) -> None:
    diag = os.path.join(root, "diagnostics")
    os.makedirs(diag, exist_ok=True)
    print("[1/4] Verifying scientific physics & optics models...")
    phys = verify_physics()
    print("  Physics OK:", json.dumps(phys))

    print("[2/4] Rendering character model sheet...")
    render_model_sheet(os.path.join(diag, "model_sheet.png"))

    print("[3/4] Synthesizing 120.00s stereo 48 kHz original score & stems...")
    t_a = time.perf_counter()
    audio_stats = synthesize_master_audio(
        os.path.join(diag, "score_master.wav"),
        stems_dir=os.path.join(diag, "audio_stems"),
    )
    print(f"  Audio synthesized in {time.perf_counter() - t_a:.1f}s:", json.dumps(audio_stats))

    print("[4/4] Benchmarking representative 2048x1152 uncached & cached frames + hero key frames...")
    hero_dir = os.path.join(diag, "hero_frames")
    os.makedirs(hero_dir, exist_ok=True)
    t_bench0 = time.perf_counter()
    r1 = Shot1Renderer(2048, 1152)
    t_init = time.perf_counter() - t_bench0
    t_f0 = time.perf_counter()
    lin, prof = r1.render(90, 90)
    u8 = finish_35mm(lin, 90, prof)
    t_cached = time.perf_counter() - t_f0
    _save_rgb(os.path.join(hero_dir, "hero_01_2048x1152.png"), u8)
    print(f"  Benchmark 2048x1152 Shot 1: init={t_init:.2f}s, cached_frame={t_cached:.3f}s")
    est_total_2core = (t_init * 11.0 + t_cached * 2880.0) / 2.0
    print(f"  Estimated 2-core total render time: ~{est_total_2core:.0f}s ({est_total_2core / 60.0:.1f} min)")


def run_animatic(root: str) -> None:
    diag = os.path.join(root, "diagnostics")
    wav = os.path.join(diag, "score_master.wav")
    if not os.path.exists(wav):
        synthesize_master_audio(wav)
    work = os.path.join(root, "work_animatic")
    os.makedirs(work, exist_ok=True)
    clips = []
    tasks = []
    for spec in SHOT_SPECS:
        cpath = os.path.join(work, f"shot_{spec.shot_num:02d}.mp4")
        clips.append(cpath)
        tasks.append((spec.shot_num, 512, 288, cpath, diag, 22, False))
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=2) as ex:
        for res in ex.map(render_single_shot_worker, tasks):
            print(f"  Animatic Shot {res['shot']:02d} ({res['title']}): {res['render_sec']}s")
    out_anim = os.path.join(diag, "animatic_512x288.mp4")
    concat_and_mux(clips, wav, out_anim)
    print(f"Animatic completed in {time.perf_counter() - t0:.1f}s -> {out_anim}")


def run_full_render(root: str, workers: int = 2, crf: int = 16, force: bool = False, force_audio: bool = False) -> None:
    diag = os.path.join(root, "diagnostics")
    work = os.path.join(root, "work_2048")
    os.makedirs(work, exist_ok=True)
    os.makedirs(diag, exist_ok=True)

    phys = verify_physics()
    render_model_sheet(os.path.join(diag, "model_sheet.png"))

    wav = os.path.join(diag, "score_master.wav")
    if not os.path.exists(wav) or force_audio:
        audio_stats = synthesize_master_audio(wav, stems_dir=os.path.join(diag, "audio_stems"))
    else:
        audio_stats = {"status": "cached_existing_master_wav"}

    clips = []
    tasks = []
    for spec in SHOT_SPECS:
        cpath = os.path.join(work, f"shot_{spec.shot_num:02d}.mp4")
        clips.append(cpath)
        if force or not os.path.exists(cpath) or os.path.getsize(cpath) < 1024:
            tasks.append((spec.shot_num, 2048, 1152, cpath, diag, crf, True))

    t0 = time.perf_counter()
    shot_metrics: List[Dict] = []
    if tasks:
        print(f"Rendering {len(tasks)} shots at 2048x1152 (24 fps, CRF {crf}) with {workers} parallel workers...")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(render_single_shot_worker, t): t[0] for t in tasks}
            for fut in as_completed(futs):
                res = fut.result()
                shot_metrics.append(res)
                print(
                    f"  [DONE] Shot {res['shot']:02d} ({res['title']}): {res['frames']}f, "
                    f"init={res['init_sec']}s, render={res['render_sec']}s ({res['ms_per_frame']} ms/f), "
                    f"adj_MAE={res['median_adj_mae']}"
                )
    shot_metrics.sort(key=lambda d: d["shot"])

    master_mp4 = os.path.join(root, "newton_the_hidden_order.mp4")
    print("Concatenating shots & muxing 48 kHz AAC master MP4...")
    concat_and_mux(clips, wav, master_mp4, crf=crf)

    # Copy to /public/newton_the_hidden_order.mp4 for the live Vite web cinema player
    public_mp4 = os.path.join(root, "public", "newton_the_hidden_order.mp4")
    os.makedirs(os.path.dirname(public_mp4), exist_ok=True)
    shutil.copy2(master_mp4, public_mp4)

    # Build self-contained single HTML deliverable with embedded video/audio
    out_html = os.path.join(root, "newton_the_hidden_order.html")
    print("Building self-contained newton_the_hidden_order.html...")
    build_self_contained_html(master_mp4, out_html)

    probe = probe_mp4(master_mp4)
    total_sec = time.perf_counter() - t0

    # Write concise render report
    report = {
        "film": "NEWTON — THE HIDDEN ORDER",
        "resolution": "2048x1152",
        "fps": FPS,
        "total_frames": TOTAL_FRAMES,
        "crf": crf,
        "wall_time_sec": round(total_sec, 1),
        "mp4_bytes": os.path.getsize(master_mp4),
        "html_bytes": os.path.getsize(out_html),
        "physics_verification": phys,
        "audio_stats": audio_stats,
        "shot_metrics": shot_metrics,
        "ffprobe": probe,
    }
    report_path = os.path.join(root, "render_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"COMPLETE in {total_sec:.1f}s | MP4: {os.path.getsize(master_mp4)/1e6:.2f} MB | HTML: {os.path.getsize(out_html)/1e6:.2f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure-Python 3D-grounded historical cinematic animation pipeline")
    parser.add_argument("--mode", choices=["verify", "animatic", "render"], default="verify")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--crf", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-audio", action="store_true")
    args = parser.parse_args()
    root = os.getcwd()
    if args.mode == "verify":
        run_verify(root)
    elif args.mode == "animatic":
        run_animatic(root)
    else:
        run_full_render(root, workers=args.workers, crf=args.crf, force=args.force, force_audio=args.force_audio)


if __name__ == "__main__":
    main()
