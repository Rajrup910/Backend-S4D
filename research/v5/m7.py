"""M7 -- acquisition randomisation in OD space, plus the SWAD secondary (Amendment 02 M7; runsheet 7).

Written 30 Sep as a NEW file while the night queue ran; wired in after it ends. LOAO runs only
(`train_v5 --arm m7 --loao-holdout <archive>`; the LOAO `control` uses the plain V4 transform):
  * `train_v5.run`: for the `m7` arm, `build_train_transform(recipe)` is replaced by
    `build_m7_train_transform(image_size, basis)`, where `basis` is the fold's frozen
    `ChromophoreBasis` (results/v5/chromophore/<split>.json; needed only for the chromophore-scaling
    step, which is skipped and recorded if it is None);
  * `--swad-window 10 30` (LOAO control and m7): `SwadAverager` is updated after every epoch in the
    window and its weights are saved as `_swad.pt`; it is scored on the held-out archive as a
    pre-registered secondary.
Order (runsheet): geometric -> OD -> Planckian -> aperture -> CLAHE -> blur/noise -> re-JPEG.
Each step is applied with p = 0.5 independently unless stated. The CPU-heavy steps (CLAHE, the
distortions) must be timed in the smoke epoch: if the epoch grows by more than 25% they move to the
GPU or are dropped and noted (audit AU9).
"""

from __future__ import annotations

import io
import math
import os
import random

import numpy as np
import torch
from PIL import Image

EXPOSURE_RANGE = 0.08  # OD += delta * (1,1,1), delta ~ U[-0.08, 0.08]
CHROMOPHORE_SCALE = 0.10  # c_mel, c_hb *= 1 + u, u ~ U[-0.10, 0.10]
ILLUMINANT_K = (5000.0, 8000.0)  # 6,500 K +/- 1,500 K
APERTURE_RADIUS = (0.45, 0.60)  # of the short side
VIGNETTE = (0.0, 0.3)
CLAHE_P, CLAHE_CLIP, CLAHE_TILES = 0.4, 4.0, (8, 8)
BLUR_MAX, NOISE_SIGMA = 5, (0.02, 0.11)
RESAMPLE_SCALE, JPEG_QUALITY = (0.4, 1.0), (70, 95)
RRC_SCALE = (0.6, 1.0)
P_STEP = 0.5
OD_CLAMP = (1.0 / 255.0, 254.0 / 255.0)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def planckian_gain(kelvin: float) -> np.ndarray:
    """RGB gain of a blackbody at `kelvin`, relative to 6,500 K (Tanner Helland approximation,
    valid 1,000-40,000 K). An illuminant change, not a hue rotation."""
    def rgb(k: float) -> np.ndarray:
        t = k / 100.0
        r = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
        g = (99.4708025861 * math.log(t) - 161.1195681661 if t <= 66
             else 288.1221695283 * (t - 60) ** -0.0755148492)
        b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307)
        return np.clip(np.array([r, g, b]), 1.0, 255.0)
    return rgb(kelvin) / rgb(6500.0)


def _od(img: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(img, *OD_CLAMP))


def _from_od(od: np.ndarray) -> np.ndarray:
    return np.clip(np.exp(-od), 0.0, 1.0)


def _smooth_field(h: int, w: int, amplitude: float, cells: int, rng: random.Random) -> np.ndarray:
    import cv2

    seed = np.random.default_rng(rng.randrange(2 ** 31))
    field = seed.uniform(-1, 1, size=(cells, cells, 2)).astype(np.float32) * amplitude
    return cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC)


class M7Transform:
    """PIL RGB image -> normalised (3, S, S) tensor, with the M7 augmentations."""

    def __init__(self, image_size: int, basis=None, seed: int | None = None) -> None:
        self.size = image_size
        self.basis = basis  # ChromophoreBasis or None
        self.rng = random.Random(seed)
        self._pid = os.getpid()

    def _reseed_in_worker(self) -> None:
        """Each DataLoader worker receives a pickled copy of this object, so without a reseed
        every worker would replay the same augmentation stream. torch gives each worker a distinct
        `initial_seed()` derived from the loader's generator, so this stays reproducible."""
        if os.getpid() != self._pid:
            self._pid = os.getpid()
            self.rng = random.Random(torch.initial_seed() % (2 ** 63))

    # ---- geometric
    def _geometric(self, im: Image.Image) -> Image.Image:
        w, h = im.size
        scale = self.rng.uniform(*RRC_SCALE)
        cw, ch = int(round(w * math.sqrt(scale))), int(round(h * math.sqrt(scale)))
        x0, y0 = self.rng.randint(0, max(w - cw, 0)), self.rng.randint(0, max(h - ch, 0))
        im = im.crop((x0, y0, x0 + cw, y0 + ch)).resize((self.size, self.size), Image.BILINEAR)
        im = im.rotate(self.rng.uniform(0, 360), resample=Image.BILINEAR, fillcolor=(0, 0, 0))
        if self.rng.random() < 0.5:
            im = im.transpose(Image.TRANSPOSE)
        if self.rng.random() < P_STEP:  # OneOf(optical distortion, grid distortion)
            import cv2

            a = np.asarray(im)
            h2, w2 = a.shape[:2]
            if self.rng.random() < 0.5:  # optical: radial barrel/pincushion
                k = self.rng.uniform(-0.3, 0.3)
                ys, xs = np.mgrid[0:h2, 0:w2].astype(np.float32)
                cx, cy = (w2 - 1) / 2, (h2 - 1) / 2
                r2 = ((xs - cx) / cx) ** 2 + ((ys - cy) / cy) ** 2
                mapx, mapy = cx + (xs - cx) * (1 + k * r2), cy + (ys - cy) * (1 + k * r2)
            else:  # grid: smooth low-frequency displacement
                d = _smooth_field(h2, w2, 0.03 * w2, 5, self.rng)
                ys, xs = np.mgrid[0:h2, 0:w2].astype(np.float32)
                mapx, mapy = xs + d[..., 0], ys + d[..., 1]
            a = cv2.remap(a, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            im = Image.fromarray(a)
        return im

    # ---- optical density
    def _od_steps(self, a: np.ndarray) -> np.ndarray:
        if self.rng.random() < P_STEP:  # exposure jitter along the shading axis
            a = _from_od(_od(a) + self.rng.uniform(-EXPOSURE_RANGE, EXPOSURE_RANGE))
        if self.basis is not None and self.rng.random() < P_STEP:  # chromophore scaling
            od = torch.from_numpy(_od(a)).permute(2, 0, 1).unsqueeze(0).float()
            conc = self.basis.concentrations(od)
            conc[:, 0] *= 1 + self.rng.uniform(-CHROMOPHORE_SCALE, CHROMOPHORE_SCALE)
            conc[:, 1] *= 1 + self.rng.uniform(-CHROMOPHORE_SCALE, CHROMOPHORE_SCALE)
            a = _from_od(self.basis.reconstruct(conc).squeeze(0).permute(1, 2, 0).numpy())
        if self.rng.random() < P_STEP:  # Planckian illuminant (linear-light gain)
            gain = planckian_gain(self.rng.uniform(*ILLUMINANT_K)).astype(np.float32)
            a = np.clip((a.astype(np.float32) ** 2.2 * gain) ** (1 / 2.2), 0, 1)
        return a

    # ---- aperture / vignette
    def _aperture(self, a: np.ndarray) -> np.ndarray:
        if self.rng.random() >= P_STEP:
            return a
        h, w = a.shape[:2]
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xs - w / 2) / min(h, w)) ** 2 + ((ys - h / 2) / min(h, w)) ** 2)
        radius = self.rng.uniform(*APERTURE_RADIUS)
        vign = 1.0 - self.rng.uniform(*VIGNETTE) * np.clip(r / radius, 0, 1) ** 2
        return a * (r <= radius)[..., None] * vign[..., None]

    def __call__(self, image: Image.Image) -> torch.Tensor:
        import cv2

        self._reseed_in_worker()
        im = self._geometric(image.convert("RGB"))
        a = np.asarray(im, dtype=np.float32) / 255.0
        a = self._aperture(self._od_steps(a))
        u8 = (np.clip(a, 0, 1) * 255).astype(np.uint8)
        if self.rng.random() < CLAHE_P:  # CLAHE on luminance only
            lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
            lab[..., 0] = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILES).apply(lab[..., 0])
            u8 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        if self.rng.random() < P_STEP:  # OneOf(gaussian blur, median blur, gaussian noise)
            choice = self.rng.randrange(3)
            k = self.rng.choice([3, 5][: 1 + (BLUR_MAX >= 5)])
            if choice == 0:
                u8 = cv2.GaussianBlur(u8, (k, k), 0)
            elif choice == 1:
                u8 = cv2.medianBlur(u8, k)
            else:
                sigma = self.rng.uniform(*NOISE_SIGMA)
                noise = np.random.default_rng(self.rng.randrange(2 ** 31)).normal(0, sigma * 255, u8.shape)
                u8 = np.clip(u8.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if self.rng.random() < P_STEP:  # resample then re-JPEG
            scale = self.rng.uniform(*RESAMPLE_SCALE)
            small = Image.fromarray(u8).resize((max(int(self.size * scale), 16),) * 2, Image.BILINEAR)
            buf = io.BytesIO()
            small.resize((self.size, self.size), Image.BILINEAR).save(
                buf, "JPEG", quality=self.rng.randint(*JPEG_QUALITY))
            u8 = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
        t = torch.from_numpy(np.ascontiguousarray(u8)).permute(2, 0, 1).float() / 255.0
        return (t - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)


def build_m7_train_transform(image_size: int, basis=None, seed: int | None = None) -> M7Transform:
    return M7Transform(image_size, basis, seed)


class SwadAverager:
    """Dense weight average over epochs [start, end] (1-indexed, inclusive); one extra model copy.
    Float tensors are averaged; integer buffers (e.g. counters) are copied from the last update."""

    def __init__(self, start: int = 10, end: int = 30) -> None:
        self.start, self.end, self.n = start, end, 0
        self.avg: dict[str, torch.Tensor] = {}

    def update(self, model: torch.nn.Module, epoch: int) -> bool:
        if not (self.start <= epoch <= self.end):
            return False
        self.n += 1
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                # clone(): for a float32 CPU tensor .float().cpu() is the live parameter itself,
                # and the first average would alias it (caught by the unit test, 30 Sep).
                cur = v.detach().float().cpu().clone()
                self.avg[k] = cur if k not in self.avg else self.avg[k] + (cur - self.avg[k]) / self.n
            else:
                self.avg[k] = v.detach().cpu().clone()
        return True

    def state_dict(self) -> dict[str, torch.Tensor]:
        return dict(self.avg)
