"""Assemble recorded segments into horizontal README GIFs with animated transitions and a progress bar.
Usage: reels.py [reel-name ...]   Output: docs/assets/readme/<reel>.gif"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from cdp import FRAMES, REPO
from scenes import ACCENT

OUT = REPO / "docs" / "assets" / "readme"
FPS = 25
DT = 1 / FPS
FRAME_MS = round(1000 / FPS)
BAR = 5  # progress bar height, px

# reel: (accent key, [(segment, speed)], output width, height)
REELS = {
    "hero": ("ALL", [("hero_title", 1), ("hero_trap", 1), ("hero_gap", 1), ("hero_why", 1)], 900, 432),
    "programme": ("ALL", [(f"prog_V{i}", 1) for i in range(1, 6)], 900, 432),
    "v1": ("V1", [("v1_title", 1), ("v1_steps", 1.35), ("v1_ladder", 1), ("v1_catch", 1), ("v1_sim", 1), ("v1_gap", 1), ("v1_next", 1)], 900, 432),
    "v2": ("V2", [("v2_title", 1), ("v2_steps", 1.35), ("v2_c", 1), ("v2_conformal", 1), ("v2_verdict", 1), ("v2_next", 1)], 900, 432),
    "v3": ("V3", [("v3_title", 1), ("v3_steps", 1.35), ("v3_leak", 1), ("v3_noise", 1), ("v3_age", 1), ("v3_verdict", 1), ("v3_next", 1)], 900, 432),
    "v4": ("V4", [("v4_title", 1), ("v4_steps", 1.35), ("v4_power", 1), ("v4_shift", 1), ("v4_probe", 1), ("v4_contract", 1), ("v4_survive", 1), ("v4_next", 1)], 900, 432),
    "v5": ("V5", [("v5_title", 1), ("v5_funnel", 1), ("v5_screens", 1), ("v5_confirm", 1), ("v5_gates", 1), ("v5_slope", 1), ("v5_q11", 1), ("v5_next", 1)], 900, 432),
    "v5_fold": ("V5", [("v5_fold", "auto")], 900, 576),
    "v5_milk10k": ("V5", [("m_title", 1), ("m_f1", 1), ("m_primary", 1), ("m_contract", 1), ("m_use", 1)], 900, 432),
    "system": ("ALL", [("sys", 1)], 900, 432),
}
# the fold machine flutters on every frame, so a slightly firmer pixel threshold saves bytes nobody can see
TUNE = {"v5_fold": {"hold": 1.25, "motion": 2.2, "keep": 8, "trim_end": 6.0}}
CUTS = {"programme"}  # the rail stays in place between steps, so a hard cut reads as one continuous animation


def hex_rgb(h: str) -> tuple[int, int, int]:
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def auto_times(d: Path, stamps: np.ndarray, start: float, end: float, hold: float = 1.25, motion: float = 2.2) -> list[float]:
    """Time-lapse that spends its time where the caption is readable: fast through motion, slow on holds."""
    probe = np.arange(start, end, DT)
    small, change = None, []
    for t in probe:
        i = max(int(np.searchsorted(stamps, t, side="right")) - 1, 0)
        im = np.asarray(Image.open(d / f"{i:05d}.png").convert("L").resize((200, 128), Image.BILINEAR), dtype=np.float32)
        change.append(0.0 if small is None else float(np.abs(im - small).mean()))
        small = im
    change = np.convolve(change, np.ones(9) / 9, mode="same")
    times, t = [], start
    while t < end:
        times.append(t)
        k = min(int((t - start) / DT), len(change) - 1)
        t += DT * (motion if change[k] > 0.05 else hold)
    return times


def load_segment(name: str, speed, size: tuple[int, int], tune: dict | None = None) -> list[np.ndarray]:
    """Resample a recording to constant FPS using the frame timestamps."""
    d = FRAMES / name
    meta = json.loads((d / "stamps.json").read_text())
    stamps = np.array(meta["stamps"])
    start, end = max(meta["start"], stamps[0]), meta["end"]
    out, cache_i, cache = [], None, None
    tune = tune or {}
    end -= tune.get("trim_end", 0)  # the recording's last seconds only hold the closing caption
    times = auto_times(d, stamps, start, end, tune.get("hold", 1.25), tune.get("motion", 2.2)) if speed == "auto" else np.arange(start, end, DT * speed)
    for t in times:
        i = max(int(np.searchsorted(stamps, t, side="right")) - 1, 0)
        if i != cache_i:
            im = Image.open(d / f"{i:05d}.png").convert("RGB")
            if im.size != size:
                im = im.resize(size, Image.LANCZOS)
            cache, cache_i = np.asarray(im), i
        out.append(cache)
    return out


def ease(x: np.ndarray | float):
    return x * x * x * (x * (6 * x - 15) + 10)


def sweep(a, b, colour, n=15):
    """A solid colour block crosses the frame: it covers A, then uncovers B."""
    h, w, _ = a.shape
    rgb = np.array(colour, dtype=np.uint8)
    out = []
    for i in range(1, n + 1):
        k = i / n
        lead = int(ease(min(1, k * 1.6)) * w)
        trail = int(ease(max(0, (k - 0.38) / 0.62)) * w)
        f = a.copy()
        f[:, :trail] = b[:, :trail]
        f[:, trail:lead] = rgb
        out.append(f)
    return out


def diagonal(a, b, colour, n=15):
    """A slanted edge with a colour stripe wipes B over A."""
    h, w, _ = a.shape
    yy, xx = np.mgrid[0:h, 0:w]
    pos = xx + yy * 0.45
    span = pos.max() + 90
    rgb = np.array(colour, dtype=np.uint8)
    out = []
    for i in range(1, n + 1):
        edge = ease(i / n) * span
        f = a.copy()
        f[pos < edge] = rgb
        f[pos < edge - 70] = b[pos < edge - 70]
        out.append(f)
    return out


def iris(a, b, colour, n=15):
    """B opens from the centre in a growing circle with a colour ring."""
    h, w, _ = a.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - w / 2, yy - h / 2)
    rmax = r.max() + 30
    rgb = np.array(colour, dtype=np.uint8)
    out = []
    for i in range(1, n + 1):
        rad = ease(i / n) * rmax
        f = a.copy()
        f[r < rad] = rgb
        f[r < rad - 26] = b[r < rad - 26]
        out.append(f)
    return out


def push(a, b, colour, n=15):
    """B slides in from the right and pushes A out."""
    h, w, _ = a.shape
    out = []
    for i in range(1, n + 1):
        x = int(ease(i / n) * w)
        f = np.empty_like(a)
        f[:, :w - x] = a[:, x:]
        f[:, w - x:] = b[:, :x] if x else b[:, :0]
        out.append(f)
    return out


TRANSITIONS = [sweep, diagonal, push, iris]


def build(name: str) -> None:
    accent, segments, w, h = REELS[name]
    colour = hex_rgb(ACCENT[accent])
    frames: list[np.ndarray] = []
    refresh: set[int] = set()
    for k, (seg, speed) in enumerate(segments):
        part = load_segment(seg, speed, (w, h), TUNE.get(name))
        if k and name not in CUTS:
            frames += TRANSITIONS[(k - 1) % len(TRANSITIONS)](frames[-1], part[0], colour)
        refresh.add(len(frames))  # every scene starts from clean pixels, so nothing of the last one lingers
        frames += part
    total = len(frames)

    # progress bar: keeps long holds alive and tells the viewer where the loop is
    out = []
    for i, f in enumerate(frames):
        g = f.copy()
        g[h - BAR:, :] = (g[h - BAR:, :].astype(np.int16) * 0.55).astype(np.uint8)
        # the bar moves in 3 px steps: smooth to the eye, and long holds merge into fewer frames
        g[h - BAR:, : int((i + 1) / total * w / 3) * 3] = colour
        out.append(g)

    # one palette for the whole reel, from a spread of frames plus the accent colours
    sample_idx = np.linspace(0, total - 1, min(48, total)).astype(int)
    tile_w = 500
    tile_h = int(tile_w * h / w)
    montage = Image.new("RGB", (tile_w * 8, tile_h * ((len(sample_idx) + 7) // 8)))
    for j, idx in enumerate(sample_idx):
        montage.paste(Image.fromarray(out[idx]).resize((tile_w, tile_h), Image.BILINEAR), ((j % 8) * tile_w, (j // 8) * tile_h))
    palette = montage.quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)

    # map to the palette; pixels that barely changed keep the previous index, so the delta stays small
    images, durations = [], []
    prev_idx = None
    table = np.array(palette.getpalette() + [0] * 768, dtype=np.int16)[:768].reshape(-1, 3)
    for i, f in enumerate(out):
        idx = np.asarray(Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE)).copy()
        if prev_idx is not None and i not in refresh:
            # compared with what is on screen, not with the previous source frame, so slow fades cannot leave ghosts
            same = np.abs(f.astype(np.int16) - table[prev_idx]).max(axis=2) < TUNE.get(name, {}).get("keep", 5)
            idx[same] = prev_idx[same]
            if np.array_equal(idx, prev_idx):
                durations[-1] += FRAME_MS
                continue
        im = Image.fromarray(idx.astype(np.uint8), "P")
        im.putpalette(palette.getpalette())
        images.append(im)
        durations.append(FRAME_MS)
        prev_idx = idx
    durations[-1] += 1200  # a beat before the loop restarts

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.gif"
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations, loop=0, disposal=1, optimize=True, transparency=255)
    print(f"{name}: {len(images)} frames, {sum(durations) / 1000:.1f} s, {w}x{h}, {path.stat().st_size / 1e6:.2f} MB", flush=True)


if __name__ == "__main__":
    for reel in sys.argv[1:] or list(REELS):
        build(reel)
