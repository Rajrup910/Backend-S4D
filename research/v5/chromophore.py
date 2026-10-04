"""DRE-0 / M1 chromophore unmixing, DRE-1 palette and DRE-2 mask-free lesion geometry.

One torch implementation serves both the GPU arms (`look`, `structure`, `geometry`) and the CPU
pre-checks (M1-QC, Q1, Q2/D6, Q5), so what the pre-checks certify is exactly what the arms compute.
Every function takes batched tensors (B, C, H, W) with sRGB in [0, 1] and works on any device.

Values fixed by docs/V5_RUNSHEET.md section 7 are named constants. Values the runsheet leaves open
are marked `[impl]` below and listed by `implementation_declarations()`, so they can be declared
before the E1 hash (audit AU15) instead of being chosen after a result is seen.

Physics (Amendment 02 M1):
  sRGB -> linear (inverse sRGB gamma) -> clamp [1/255, 254/255] -> OD = -ln(I).
  OD = c_mel * v_mel + c_hb * v_hb + c_shade * (1, 1, 1). v_mel and v_hb are found by ICA on the
  OD plane orthogonal to the shading axis (Tsumura), from one fold's training pixels only. Because
  c_mel and c_hb are invariant to any shading component of v_mel / v_hb, fitting in the plane loses
  nothing.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ------------------------------------------------------------------ fixed values (runsheet §7)
OD_CLAMP = (1.0 / 255.0, 254.0 / 255.0)
GLARE_SRGB = 250.0 / 255.0
DEPTH_EPS = 0.02  # OD
DEPTH_GATE_W = 0.1  # w = 0.1 * t_mel
PALETTE_K = 6
PALETTE_TAU = 0.05  # [declared here] in the runsheet
PALETTE_COUNT_AREA = 0.05
PALETTE_COUNT_SLOPE = 0.01
GEOMETRY_BETA = 1.0  # [declared here]
GEOMETRY_W = 0.1  # w = 0.1 * t  [declared here]
GEOMETRY_MIN_COVER = 0.03
GEOMETRY_MAX_COVER = 0.95
ICA_PIXELS = 1_000_000

# ------------------------------------------------------------------ implementation choices [impl]
#: Aperture / vignette pixels (the black surround of BCN / MSKCC images) carry no skin signal and
#: saturate OD. They are excluded from the skin ring, the lesion density and the fits, like glare.
APERTURE_SRGB_MAX = 20.0 / 255.0  # [impl] a pixel is aperture if max(R, G, B) < this
#: "Outer 10% ring": pixels in the outer 10% of the half-extent on any side (Chebyshev frame).
RING_FRACTION = 0.10  # [impl]
#: Reference spectral directions used only to NAME the two ICA components (which is melanin,
#: which is haemoglobin) and fix their sign. Projected camera-band absorption shapes: melanin
#: absorbs B > G > R; haemoglobin (Q-band) absorbs G most. They never enter the unmixing.
REF_MEL = (0.30, 0.60, 1.00)  # [impl] relative OD in (R, G, B)
REF_HB = (0.10, 1.00, 0.60)  # [impl]
#: Otsu threshold is computed on a 256-bin histogram of the lesion density per image.
OTSU_BINS = 256  # [impl]
LESION_PIXEL_THRESHOLD = 0.5  # [impl] soft mask s > 0.5 counts as a lesion pixel in fold stats
#: An Otsu threshold below this (OD units above skin) means the image has no lesion contrast at
#: all (a uniform image): s would be 0.5 everywhere, so the geometry falls back instead.
MIN_OTSU_T = 1e-3  # [impl]
# NaN fix (2026-10-02, CHANGELOG): a crop that is almost all aperture/glare (one valid pixel was seen)
# gave a point mask, radius 0 and palette eccentricity ~1e8, which overflowed fp16 in the classifier.
# Coverage is measured over valid pixels, so the coverage fallback could not see it. These two make the
# geometry fall back (image centre, image axes, r = half the short side) instead.
GEOMETRY_MIN_VALID = 0.03  # [impl, NaN fix] valid pixels / frame; same value as GEOMETRY_MIN_COVER
GEOMETRY_MIN_RADIUS_PX = 2.0  # [impl, NaN fix] a lesion radius under 2 px is a point, not a lesion

SHADING = torch.tensor([1.0, 1.0, 1.0]) / math.sqrt(3.0)


def implementation_declarations() -> dict[str, object]:
    """The [impl] values, for the adoption record (AU15)."""
    return {
        "APERTURE_SRGB_MAX": APERTURE_SRGB_MAX, "RING_FRACTION": RING_FRACTION,
        "REF_MEL_RGB": REF_MEL, "REF_HB_RGB": REF_HB, "OTSU_BINS": OTSU_BINS,
        "LESION_PIXEL_THRESHOLD": LESION_PIXEL_THRESHOLD, "MIN_OTSU_T": MIN_OTSU_T,
        "GEOMETRY_MIN_VALID": GEOMETRY_MIN_VALID, "GEOMETRY_MIN_RADIUS_PX": GEOMETRY_MIN_RADIUS_PX,
        "ring_definition": "Chebyshev frame: |x-cx|/(W/2) > 0.9 or |y-cy|/(H/2) > 0.9",
        "aperture_definition": "max(R,G,B) < 20/255 in sRGB; excluded like glare",
        "component_naming": "ICA mixing vectors in the plane orthogonal to (1,1,1) are assigned "
                            "melanin/haemoglobin (and sign-fixed) by maximum total cosine to the "
                            "projected reference spectra; the references never enter unmixing",
    }


# ------------------------------------------------------------------ colour -> OD
def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp_min(1e-12) ** (1 / 2.4) - 0.055)


def glare_mask(srgb: torch.Tensor) -> torch.Tensor:
    """(B,1,H,W) bool: any sRGB channel >= 250/255 (specular reflection)."""
    return (srgb >= GLARE_SRGB).any(dim=1, keepdim=True)


def aperture_mask(srgb: torch.Tensor) -> torch.Tensor:
    """(B,1,H,W) bool: black aperture / vignette surround [impl]."""
    return srgb.amax(dim=1, keepdim=True) < APERTURE_SRGB_MAX


def optical_density(srgb: torch.Tensor) -> torch.Tensor:
    """OD = -ln(clamp(linear(sRGB))), natural log, strictly positive and finite."""
    return -torch.log(srgb_to_linear(srgb.float()).clamp(*OD_CLAMP))


def unnormalise(x: torch.Tensor, mean, std) -> torch.Tensor:
    """ImageNet-normalised tensor -> sRGB in [0, 1] (step 1 of M1, inside the model)."""
    m = torch.as_tensor(mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    s = torch.as_tensor(std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return (x * s + m).clamp(0.0, 1.0)


def ring_mask(h: int, w: int, device=None) -> torch.Tensor:
    """(1,1,H,W) bool: the outer RING_FRACTION of the half-extent on any side [impl]."""
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h * 2 - 1
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w * 2 - 1
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return ((yy.abs() > 1 - RING_FRACTION) | (xx.abs() > 1 - RING_FRACTION)).view(1, 1, h, w)


def masked_median(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-image median of x over mask; (B,). Vectorised (one nanmedian over the batch, no
    per-image loop, so it runs batched on the GPU). An empty mask falls back to the whole image."""
    flat = x.flatten(1).float()
    m = mask.expand_as(x).flatten(1)
    empty = ~m.any(1)
    m = m | empty.unsqueeze(1)
    return torch.where(m, flat, torch.full_like(flat, float("nan"))).nanmedian(dim=1).values.to(x.dtype)


# ------------------------------------------------------------------ the basis
def _reference_plane_vector(ref) -> np.ndarray:
    v = np.asarray(ref, dtype=np.float64)
    v = v - v.mean()
    return v / np.linalg.norm(v)


@dataclass
class ChromophoreBasis:
    """Per-fold unmixing, fitted on that fold's training pixels only, then frozen."""

    v_mel: list[float]
    v_hb: list[float]
    fold: int | None = None
    n_pixels: int = 0
    #: Fold statistics for the depth gate and the stem standardisation (set by `fit_stats`).
    t_mel: float | None = None
    channel_mean: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    channel_std: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    naming_cosines: dict[str, float] = field(default_factory=dict)

    # --- fit
    @classmethod
    def fit_ica(cls, od_pixels: np.ndarray, seed: int = 0, fold: int | None = None
                ) -> ChromophoreBasis:
        """ICA on the OD plane orthogonal to the shading axis (N x 3 non-glare pixels)."""
        from sklearn.decomposition import FastICA

        od = np.asarray(od_pixels, dtype=np.float64)
        if od.ndim != 2 or od.shape[1] != 3:
            raise ValueError("od_pixels must be N x 3")
        shading = np.ones(3) / math.sqrt(3.0)
        # Orthonormal basis (e1, e2) of the plane orthogonal to the shading axis.
        e1 = np.array([1.0, -1.0, 0.0]) / math.sqrt(2.0)
        e2 = np.cross(shading, e1)
        plane = np.stack([e1, e2], axis=1)  # 3 x 2
        coords = od @ plane  # N x 2
        ica = FastICA(n_components=2, whiten="unit-variance", random_state=seed, max_iter=1000)
        ica.fit(coords)
        mixing_2d = ica.mixing_  # 2 x 2, columns = component directions in plane coordinates
        cands = [plane @ mixing_2d[:, i] for i in range(2)]
        cands = [c / np.linalg.norm(c) for c in cands]
        ref_m, ref_h = _reference_plane_vector(REF_MEL), _reference_plane_vector(REF_HB)
        best = None
        for i_mel in (0, 1):
            for s_mel in (1.0, -1.0):
                for s_hb in (1.0, -1.0):
                    vm = s_mel * cands[i_mel]
                    vh = s_hb * cands[1 - i_mel]
                    score = float(vm @ ref_m + vh @ ref_h)
                    if best is None or score > best[0]:
                        best = (score, vm, vh)
        _, vm, vh = best
        return cls(v_mel=vm.tolist(), v_hb=vh.tolist(), fold=fold, n_pixels=len(od),
                   naming_cosines={"mel_vs_ref": float(vm @ ref_m),
                                   "hb_vs_ref": float(vh @ ref_h),
                                   "mel_vs_hb": float(vm @ vh)})

    # --- apply
    def unmix_matrix(self, device=None, dtype=torch.float32) -> torch.Tensor:
        """3x3 W with [c_mel, c_hb, c_shade] = W @ OD."""
        mixing = torch.tensor([self.v_mel, self.v_hb, [1.0, 1.0, 1.0]], dtype=torch.float64).t()
        return torch.linalg.inv(mixing).to(device=device, dtype=dtype)

    def concentrations(self, od: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) OD -> (B,3,H,W) [c_mel, c_hb, c_shade]."""
        w = self.unmix_matrix(od.device, od.dtype)
        return torch.einsum("ij,bjhw->bihw", w, od)

    def reconstruct(self, conc: torch.Tensor) -> torch.Tensor:
        """Inverse of `concentrations` (used by M7's chromophore scaling)."""
        mixing = torch.tensor([self.v_mel, self.v_hb, [1.0, 1.0, 1.0]],
                              dtype=conc.dtype, device=conc.device).t()
        return torch.einsum("ij,bjhw->bihw", mixing, conc)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ChromophoreBasis:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


# ------------------------------------------------------------------ maps
@dataclass
class ChromophoreMaps:
    od: torch.Tensor  # (B,3,H,W)
    c_mel: torch.Tensor  # (B,1,H,W)
    c_hb: torch.Tensor
    c_depth: torch.Tensor
    valid: torch.Tensor  # (B,1,H,W) bool: not glare, not aperture
    m_skin: torch.Tensor  # (B,)
    h_skin: torch.Tensor


def depth_channel(od: torch.Tensor, c_mel: torch.Tensor, valid: torch.Tensor,
                  t_mel: float) -> torch.Tensor:
    """c_depth = tanh(ln((OD_B+e)/(OD_R+e)) - mu_skin) * sigmoid((c_mel - t_mel) / (0.1 t_mel)).
    Lower means deeper. mu_skin = median log-ratio over the image's outer ring (valid pixels)."""
    log_ratio = torch.log((od[:, 2:3] + DEPTH_EPS) / (od[:, 0:1] + DEPTH_EPS))
    ring = ring_mask(od.shape[-2], od.shape[-1], od.device) & valid
    mu_skin = masked_median(log_ratio, ring).view(-1, 1, 1, 1)
    gate_w = max(DEPTH_GATE_W * abs(t_mel), 1e-6)
    gate = torch.sigmoid((c_mel - t_mel) / gate_w)
    return torch.tanh(log_ratio - mu_skin) * gate


def compute_maps(srgb: torch.Tensor, basis: ChromophoreBasis) -> ChromophoreMaps:
    """All chromophore maps for a batch of sRGB images in [0, 1]."""
    if basis.t_mel is None:
        raise RuntimeError("basis has no fold statistics; run fit_stats on training images first")
    srgb = srgb.float()
    od = optical_density(srgb)
    conc = basis.concentrations(od)
    c_mel, c_hb = conc[:, 0:1], conc[:, 1:2]
    valid = ~(glare_mask(srgb) | aperture_mask(srgb))
    ring = ring_mask(srgb.shape[-2], srgb.shape[-1], srgb.device) & valid
    m_skin = masked_median(c_mel, ring)
    h_skin = masked_median(c_hb, ring)
    c_depth = depth_channel(od, c_mel, valid, basis.t_mel)
    return ChromophoreMaps(od=od, c_mel=c_mel, c_hb=c_hb, c_depth=c_depth, valid=valid,
                           m_skin=m_skin, h_skin=h_skin)


def stem_channels(maps: ChromophoreMaps, basis: ChromophoreBasis) -> torch.Tensor:
    """[c_mel, c_hb, c_depth] standardised with the fold's training mean / SD -> (B,3,H,W)."""
    x = torch.cat([maps.c_mel, maps.c_hb, maps.c_depth], dim=1)
    mean = torch.tensor(basis.channel_mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(basis.channel_std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std.clamp_min(1e-6)


# ------------------------------------------------------------------ DRE-2 geometry
def otsu_threshold(x: torch.Tensor, valid: torch.Tensor, bins: int = OTSU_BINS) -> torch.Tensor:
    """Per-image Otsu threshold of x over valid pixels; (B,)."""
    out = []
    for b in range(x.shape[0]):
        v = x[b][valid[b]]
        if v.numel() < 2 or float(v.max() - v.min()) < 1e-8:
            out.append(v.max() if v.numel() else x.new_tensor(0.0))
            continue
        lo, hi = float(v.min()), float(v.max())
        hist = torch.histc(v, bins=bins, min=lo, max=hi)
        p = hist / hist.sum()
        edges = torch.linspace(lo, hi, bins + 1, device=x.device)
        centers = edges[:-1] + (hi - lo) / bins / 2
        w0 = torch.cumsum(p, 0)
        mu0 = torch.cumsum(p * centers, 0)
        mu_t = mu0[-1]
        between = (mu_t * w0 - mu0) ** 2 / (w0 * (1 - w0)).clamp_min(1e-12)
        # Split after bin i sits at edges[i + 1]. Empty bins make the criterion flat, and a plain
        # argmax would return the first bin of that plateau (the edge of the darker mode); take
        # the middle of the plateau instead, the split a human would draw.
        top = torch.nonzero(between >= between.max() * (1 - 1e-6)).flatten()
        out.append((edges[top.min() + 1] + edges[top.max() + 1]) / 2)
    return torch.stack(out)


@dataclass
class LesionGeometry:
    s: torch.Tensor  # (B,1,H,W) soft mask
    centroid: torch.Tensor  # (B,2) (y, x) in pixels
    theta: torch.Tensor  # (B,) principal-axis angle, radians, from +x toward +y
    radius: torch.Tensor  # (B,) r = 2 sqrt(lambda_1), pixels
    axes: torch.Tensor  # (B,2,2) rows = unit major, minor axis (y, x)
    fallback: torch.Tensor  # (B,) bool
    coverage: torch.Tensor  # (B,) mean of s over valid pixels


def lesion_density(maps: ChromophoreMaps) -> torch.Tensor:
    m = (maps.c_mel - maps.m_skin.view(-1, 1, 1, 1)).clamp_min(0)
    h = (maps.c_hb - maps.h_skin.view(-1, 1, 1, 1)).clamp_min(0)
    return (m + GEOMETRY_BETA * h) * maps.valid


def lesion_geometry(maps: ChromophoreMaps, force_fallback: bool = False) -> LesionGeometry:
    """DRE-2: chromophore soft mask, moments, principal axes and radius. No learning."""
    ell = lesion_density(maps)
    t = otsu_threshold(ell, maps.valid)
    width = (GEOMETRY_W * t).clamp_min(1e-6).view(-1, 1, 1, 1)
    s = torch.sigmoid((ell - t.view(-1, 1, 1, 1)) / width) * maps.valid
    return geometry_from_mask(s, maps.valid, degenerate=t < MIN_OTSU_T,
                              force_fallback=force_fallback)


def geometry_from_mask(s: torch.Tensor, valid: torch.Tensor,
                       degenerate: torch.Tensor | None = None,
                       force_fallback: bool = False) -> LesionGeometry:
    """Moments, principal axes and radius of any soft mask s (B,1,H,W) -- the DRE-2 soft mask, or
    an expert mask for the pre-checks that separate "is the signal in the pixels" from "does the
    chromophore mask find the lesion" (Q2 primary, Q1)."""
    b, _, h, w = s.shape
    valid_n = valid.flatten(1).sum(1).clamp_min(1)
    coverage = s.flatten(1).sum(1) / valid_n
    ys = torch.arange(h, device=s.device, dtype=s.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
    xs = torch.arange(w, device=s.device, dtype=s.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
    mass = s.flatten(1).sum(1).clamp_min(1e-6)
    cy = (s * ys).flatten(1).sum(1) / mass
    cx = (s * xs).flatten(1).sum(1) / mass
    dy, dx = ys - cy.view(-1, 1, 1, 1), xs - cx.view(-1, 1, 1, 1)
    cyy = (s * dy * dy).flatten(1).sum(1) / mass
    cxx = (s * dx * dx).flatten(1).sum(1) / mass
    cxy = (s * dx * dy).flatten(1).sum(1) / mass
    cov = torch.stack([torch.stack([cyy, cxy], -1), torch.stack([cxy, cxx], -1)], -2)
    evals, evecs = torch.linalg.eigh(cov)  # ascending
    major = evecs[..., :, 1]  # (y, x)
    minor = evecs[..., :, 0]
    radius = 2.0 * evals[:, 1].clamp_min(0).sqrt()
    theta = torch.atan2(major[:, 0], major[:, 1])
    fallback = (coverage < GEOMETRY_MIN_COVER) | (coverage > GEOMETRY_MAX_COVER)
    fallback = fallback | (valid.flatten(1).sum(1) < GEOMETRY_MIN_VALID * h * w) \
        | (radius < GEOMETRY_MIN_RADIUS_PX)
    if degenerate is not None:
        fallback = fallback | degenerate
    if force_fallback:
        fallback = torch.ones_like(fallback)
    # Fallback: image centre, image axes, r = half the short side.
    centre = torch.tensor([(h - 1) / 2, (w - 1) / 2], device=s.device, dtype=s.dtype)
    centroid = torch.where(fallback.view(-1, 1), centre.expand(b, 2), torch.stack([cy, cx], -1))
    eye = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=s.device, dtype=s.dtype)  # x-axis major
    axes = torch.where(fallback.view(-1, 1, 1), eye.expand(b, 2, 2), torch.stack([major, minor], 1))
    theta = torch.where(fallback, torch.zeros_like(theta), theta)
    radius = torch.where(fallback, torch.full_like(radius, min(h, w) / 2), radius)
    return LesionGeometry(s=s, centroid=centroid, theta=theta, radius=radius, axes=axes,
                          fallback=fallback, coverage=coverage)


# ------------------------------------------------------------------ DRE-1 palette
@dataclass
class Palette:
    centers: list[list[float]]  # K x 3 OD prototypes, frozen per fold
    fold: int | None = None

    @classmethod
    def fit(cls, od_lesion_pixels: np.ndarray, seed: int = 0, fold: int | None = None) -> Palette:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=PALETTE_K, n_init=10, random_state=seed)
        km.fit(np.asarray(od_lesion_pixels, dtype=np.float64))
        order = np.argsort(km.cluster_centers_.sum(1))  # light -> dark, for a stable naming order
        return cls(centers=km.cluster_centers_[order].tolist(), fold=fold)

    def assign(self, od: torch.Tensor) -> torch.Tensor:
        """Soft assignment a_k = softmax(-||x - c_k||^2 / tau); (B,K,H,W)."""
        c = torch.tensor(self.centers, device=od.device, dtype=od.dtype)  # K x 3
        d2 = ((od.unsqueeze(1) - c.view(1, -1, 3, 1, 1)) ** 2).sum(2)
        return torch.softmax(-d2 / PALETTE_TAU, dim=1)


def palette_tokens(a: torch.Tensor, s: torch.Tensor, geometry: LesionGeometry) -> dict[str, torch.Tensor]:
    """Area fraction per colour inside s, the soft colour count and per-colour eccentricity."""
    b, k, h, w = a.shape
    mass = s.flatten(1).sum(1).clamp_min(1e-6)
    weighted = a * s
    area = weighted.flatten(2).sum(2) / mass.view(-1, 1)  # (B,K)
    count = torch.sigmoid((area - PALETTE_COUNT_AREA) / PALETTE_COUNT_SLOPE).sum(1)
    ys = torch.arange(h, device=a.device, dtype=a.dtype).view(1, 1, h, 1)
    xs = torch.arange(w, device=a.device, dtype=a.dtype).view(1, 1, 1, w)
    wsum = weighted.flatten(2).sum(2).clamp_min(1e-6)
    cy = (weighted * ys).flatten(2).sum(2) / wsum
    cx = (weighted * xs).flatten(2).sum(2) / wsum
    mu = geometry.centroid
    ecc = torch.sqrt((cy - mu[:, 0:1]) ** 2 + (cx - mu[:, 1:2]) ** 2) / geometry.radius.clamp_min(1e-6).view(-1, 1)
    return {"area": area, "count": count, "eccentricity": ecc}


# ------------------------------------------------------------------ DRE-3 reflections
def reflect_about_axes(x: torch.Tensor, geometry: LesionGeometry) -> list[torch.Tensor]:
    """Mirror images of x about the lesion's major and minor axes through its centroid.

    Implemented with `grid_sample`: for an output pixel p, sample x at the reflection of p about
    the axis line. Pixels whose reflection falls outside the image read 0 (padding_mode='zeros').
    """
    b, _, h, w = x.shape
    ys = torch.arange(h, device=x.device, dtype=x.dtype).view(1, h, 1).expand(b, h, w)
    xs = torch.arange(w, device=x.device, dtype=x.dtype).view(1, 1, w).expand(b, h, w)
    cy = geometry.centroid[:, 0].view(-1, 1, 1)
    cx = geometry.centroid[:, 1].view(-1, 1, 1)
    out = []
    for axis_index in (0, 1):
        u = geometry.axes[:, axis_index]  # (B,2) unit (y, x) along the mirror line
        uy, ux = u[:, 0].view(-1, 1, 1), u[:, 1].view(-1, 1, 1)
        dy, dx = ys - cy, xs - cx
        along = dy * uy + dx * ux
        ry = cy + 2 * along * uy - dy
        rx = cx + 2 * along * ux - dx
        grid = torch.stack([rx / (w - 1) * 2 - 1, ry / (h - 1) * 2 - 1], dim=-1)
        out.append(F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True))
    return out


def d4_reflections(x: torch.Tensor) -> list[torch.Tensor]:
    """Fallback axes: g_k = rot_{90k} o flip_h, k = 1..4 (four reflections of D4, image centre)."""
    flipped = torch.flip(x, dims=[-1])
    return [torch.rot90(flipped, k, dims=(-2, -1)) for k in range(1, 5)]


def asymmetry(x: torch.Tensor, s: torch.Tensor, mirrors: list[torch.Tensor],
              s_mirrors: list[torch.Tensor] | None = None) -> torch.Tensor:
    """Normalised asymmetry per axis: sum_p s ||x - g(x)||_1 / sum_p s (||x||_1 + ||g(x)||_1).
    Returns (B, n_axes) in [0, 1]."""
    vals = []
    for g in mirrors:
        num = (s * (x - g).abs().sum(1, keepdim=True)).flatten(1).sum(1)
        den = (s * (x.abs().sum(1, keepdim=True) + g.abs().sum(1, keepdim=True))).flatten(1).sum(1)
        vals.append(num / den.clamp_min(1e-6))
    return torch.stack(vals, 1)


def chromophore_asymmetry(maps: ChromophoreMaps, geometry: LesionGeometry,
                          use_d4: bool = False) -> torch.Tensor:
    """A_chrom per axis on skin-subtracted, non-negative c_mel and c_hb; (B, n_axes).

    The runsheet formula uses c_mel + c_hb in the denominator; they are taken above the skin
    baseline (clamped at 0) so the ratio lies in [0, 1] as the design states [impl].
    """
    m = (maps.c_mel - maps.m_skin.view(-1, 1, 1, 1)).clamp_min(0)
    hb = (maps.c_hb - maps.h_skin.view(-1, 1, 1, 1)).clamp_min(0)
    x = torch.cat([m, hb], 1) * maps.valid
    mirrors = d4_reflections(x) if use_d4 else reflect_about_axes(x, geometry)
    s = torch.ones_like(geometry.s) if use_d4 else geometry.s
    return asymmetry(x, s, mirrors)


# ------------------------------------------------------------------ fold statistics
@torch.no_grad()
def fold_stats_from_batches(batches, basis: ChromophoreBasis) -> dict[str, float]:
    """t_mel (median lesion melanin) and the stem channel mean / SD from training batches.

    Two-pass by design: t_mel needs the lesion mask, the depth gate needs t_mel. `batches` is an
    iterable of sRGB tensors; it is consumed twice, so pass a re-iterable (a list of paths loader).
    """
    lesion_mel: list[torch.Tensor] = []
    provisional = ChromophoreBasis(**{**asdict(basis), "t_mel": 1.0})
    for x in batches:
        maps = compute_maps(x, provisional)
        geo = lesion_geometry(maps)
        keep = (geo.s > LESION_PIXEL_THRESHOLD) & maps.valid
        lesion_mel.append(maps.c_mel[keep].flatten()[::7])
    t_mel = float(torch.cat(lesion_mel).median())
    stats_basis = ChromophoreBasis(**{**asdict(basis), "t_mel": t_mel})
    sums = torch.zeros(3, dtype=torch.float64)
    sq = torch.zeros(3, dtype=torch.float64)
    n = 0
    for x in batches:
        maps = compute_maps(x, stats_basis)
        stack = torch.cat([maps.c_mel, maps.c_hb, maps.c_depth], 1)
        v = stack.permute(1, 0, 2, 3)[:, maps.valid.permute(1, 0, 2, 3)[0]]
        sums += v.double().sum(1)
        sq += (v.double() ** 2).sum(1)
        n += v.shape[1]
    mean = (sums / max(n, 1))
    std = (sq / max(n, 1) - mean ** 2).clamp_min(1e-12).sqrt()
    return {"t_mel": t_mel, "channel_mean": mean.tolist(), "channel_std": std.tolist()}
