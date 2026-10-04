"""DRE-10 Dermoscopic Structure Primitives and DRE-4 periphery -- fixed filters, torch, any device.

Maps (all (B,1,H,W)):
  tubularity  Frangi-type vesselness of c_hb, max over sigma in {1, 2, 4} px
  blobness    bright-blob measure |l1|/|l2| of c_hb (both eigenvalues negative), max over sigma
  network     Gabor energy of c_mel, 4 orientations x wavelengths {6, 10} px, sigma = 0.56 lambda
  dots        LoG response of c_mel at sigma in {1.5, 3}, max over sigma
  veil        c_depth < fold P25 AND c_mel > fold median AND network energy < fold P25

Pixel sizes are quoted at 224 px and scale with the short side (runsheet §7). Tokens are read
inside the DRE-2 soft mask and the DRE-2 lesion frame. Values fixed by the runsheet are constants;
`[impl]` marks what the runsheet leaves open (see `implementation_declarations`).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from research.v5.chromophore import ChromophoreMaps, LesionGeometry

REF_SIZE = 224
HESSIAN_SIGMAS = (1.0, 2.0, 4.0)
FRANGI_BETA = 0.5
GABOR_THETAS = (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4)
GABOR_WAVELENGTHS = (6.0, 10.0)
GABOR_SIGMA_RATIO = 0.56
LOG_SIGMAS = (1.5, 3.0)
DOT_THRESHOLD_SD = 2.0
PERIPHERY_RHO = (0.6, 1.2)
BORDER_RHO = (0.85, 1.15)
N_SECTORS = 16
N_RADIAL = 3

# [impl] values
GABOR_GAMMA = 0.5  # [impl] spatial aspect ratio of the Gabor envelope
COVERAGE_PERCENTILE = 75.0  # [impl] a pixel "has" a structure above the fold's lesion-pixel P75
THIN_SIGMAS = (1.0,)  # [impl] tubular pixels at sigma=1 are thin; sigma in {2, 4} are thick
DENSITY_SCALE = 1000.0  # [impl] dot density = dots per 1000 lesion px at 224-px scale
BORDER_SAMPLES = 5  # [impl] radial samples across [0.85r, 1.15r] per sector for abruptness


# NaN fix (2026-10-02, CHANGELOG): abruptness is normalised by the lesion median c_mel, which can be
# ~0 (0 and 6e-08 seen), and the old 1e-6 floor turned that into variances up to ~7e7 that overflowed
# fp16. Below this floor the ratio is undefined: the rows are NaN, which the front maps to the fold mean
# and the fold statistics skip. 0.1 * t_mel is the melanin scale the depth gate already uses.
ABRUPT_MIN_MEDIAN_FRAC = 0.1  # [impl, NaN fix] times the fold's t_mel


def implementation_declarations() -> dict[str, object]:
    return {"GABOR_GAMMA": GABOR_GAMMA, "ABRUPT_MIN_MEDIAN_FRAC": ABRUPT_MIN_MEDIAN_FRAC, "COVERAGE_PERCENTILE": COVERAGE_PERCENTILE,
            "THIN_SIGMAS": THIN_SIGMAS, "DENSITY_SCALE": DENSITY_SCALE,
            "BORDER_SAMPLES": BORDER_SAMPLES,
            "blobness_definition": "Rb * (1 - exp(-S^2/2c^2)) where both Hessian eigenvalues < 0",
            "network_regularity": "1 - CV of the dominant Gabor wavelength over network pixels",
            "periphery_minus_centre": "mean c_mel on network pixels at rho in [0.6r,1.2r] minus "
                                      "at rho < 0.6r",
            "polymorphism_classes": "thin-tubular (sigma 1), thick-tubular (sigma 2,4), blob",
            "q2_direction": "every Q2/Q5 young-differential feature is tested one-sided, higher "
                            "in <40 mel than in <40 histo-nv, except where the runsheet states "
                            "the opposite (network regularity, Clark-Evans R: higher in nv)"}


def scale_factor(h: int, w: int) -> float:
    return min(h, w) / REF_SIZE


# ------------------------------------------------------------------ kernels
def _gauss_1d(sigma: float, order: int, device, dtype) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    if order == 0:
        return g
    # Sampled derivative-of-Gaussian kernels are renormalised so they are exact on polynomials:
    # sum = 0, and they return f' = 1 on f = x and f'' = 2 on f = x^2. Without this a small sigma
    # gives a second-derivative kernel with a non-zero sum, and a straight line picks up a spurious
    # curvature along its own length (seen as blobness on a vessel).
    if order == 1:
        k = -x / sigma ** 2 * g
        return k / (-(x * k).sum())
    k = (x ** 2 / sigma ** 4 - 1 / sigma ** 2) * g
    k = k - k.sum() * g
    return k * (2.0 / (x ** 2 * k).sum())


def _sep_conv(x: torch.Tensor, ky: torch.Tensor, kx: torch.Tensor) -> torch.Tensor:
    ry, rx = ky.numel() // 2, kx.numel() // 2
    x = F.conv2d(F.pad(x, (0, 0, ry, ry), mode="replicate"), ky.view(1, 1, -1, 1))
    return F.conv2d(F.pad(x, (rx, rx, 0, 0), mode="replicate"), kx.view(1, 1, 1, -1))


def hessian(x: torch.Tensor, sigma: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scale-normalised (sigma^2) Hessian components of a (B,1,H,W) map."""
    d, t = x.device, x.dtype
    g0, g1, g2 = (_gauss_1d(sigma, o, d, t) for o in (0, 1, 2))
    s2 = sigma ** 2
    return (s2 * _sep_conv(x, g2, g0), s2 * _sep_conv(x, g0, g2), s2 * _sep_conv(x, g1, g1))


def eigen2(ixx: torch.Tensor, iyy: torch.Tensor, ixy: torch.Tensor):
    """Eigenvalues ordered by magnitude: |l1| <= |l2|."""
    tr = ixx + iyy
    root = torch.sqrt((ixx - iyy) ** 2 + 4 * ixy ** 2)
    a, b = (tr + root) / 2, (tr - root) / 2
    swap = a.abs() > b.abs()
    return torch.where(swap, b, a), torch.where(swap, a, b)


def gabor_bank(scale: float, device, dtype) -> tuple[torch.Tensor, torch.Tensor, list]:
    """Even (zero-mean cosine) and odd (sine) kernels, one per (theta, lambda); padded to one size."""
    specs = [(th, lam * scale) for lam in GABOR_WAVELENGTHS for th in GABOR_THETAS]
    radius = int(math.ceil(3 * GABOR_SIGMA_RATIO * max(lam for _, lam in specs)))
    y, x = torch.meshgrid(torch.arange(-radius, radius + 1, device=device, dtype=dtype),
                          torch.arange(-radius, radius + 1, device=device, dtype=dtype),
                          indexing="ij")
    even, odd = [], []
    for theta, lam in specs:
        sigma = GABOR_SIGMA_RATIO * lam
        xr = x * math.cos(theta) + y * math.sin(theta)
        yr = -x * math.sin(theta) + y * math.cos(theta)
        env = torch.exp(-(xr ** 2 + (GABOR_GAMMA * yr) ** 2) / (2 * sigma ** 2))
        e = env * torch.cos(2 * math.pi * xr / lam)
        e = e - e.sum() / env.sum() * env  # zero DC, so a flat region gives 0
        o = env * torch.sin(2 * math.pi * xr / lam)
        norm = torch.sqrt((e ** 2).sum() + (o ** 2).sum()) / math.sqrt(2)
        even.append(e / norm)
        odd.append(o / norm)
    return torch.stack(even).unsqueeze(1), torch.stack(odd).unsqueeze(1), specs


# ------------------------------------------------------------------ maps
@dataclass
class DSPMaps:
    tubularity: torch.Tensor
    blobness: torch.Tensor
    tub_sigma_index: torch.Tensor  # argmax sigma index for tubularity
    network: torch.Tensor
    net_wavelength: torch.Tensor  # dominant wavelength (px at 224 scale)
    net_theta: torch.Tensor  # dominant orientation
    dots: torch.Tensor
    dot_sigma: torch.Tensor


def vessel_maps(c_hb: torch.Tensor, scale: float):
    tubs, blobs, norms = [], [], []
    comps = []
    for sigma in HESSIAN_SIGMAS:
        l1, l2 = eigen2(*hessian(c_hb, sigma * scale))
        comps.append((l1, l2))
        norms.append(torch.sqrt(l1 ** 2 + l2 ** 2))
    c = 0.5 * torch.stack(norms).flatten(2).amax(dim=(0, 2)).clamp_min(1e-8).view(-1, 1, 1, 1)
    for (l1, l2), s_norm in zip(comps, norms, strict=True):
        rb = l1.abs() / l2.abs().clamp_min(1e-8)
        structure = 1 - torch.exp(-s_norm ** 2 / (2 * c ** 2))
        tub = torch.exp(-rb ** 2 / (2 * FRANGI_BETA ** 2)) * structure * (l2 < 0)
        blob = rb * structure * (l1 < 0) * (l2 < 0)
        tubs.append(tub)
        blobs.append(blob)
    tub_stack = torch.stack(tubs)
    tubularity, index = tub_stack.max(0)
    return tubularity, torch.stack(blobs).amax(0), index


def network_maps(c_mel: torch.Tensor, scale: float):
    even, odd, specs = gabor_bank(scale, c_mel.device, c_mel.dtype)
    pad = even.shape[-1] // 2
    xp = F.pad(c_mel, (pad, pad, pad, pad), mode="replicate")
    energy = torch.sqrt(F.conv2d(xp, even) ** 2 + F.conv2d(xp, odd) ** 2)  # (B, n_filters, H, W)
    best, idx = energy.max(1, keepdim=True)
    lam = torch.tensor([s[1] / scale for s in specs], device=c_mel.device, dtype=c_mel.dtype)
    th = torch.tensor([s[0] for s in specs], device=c_mel.device, dtype=c_mel.dtype)
    return best, lam[idx], th[idx]


def dot_maps(c_mel: torch.Tensor, scale: float):
    responses = []
    for sigma in LOG_SIGMAS:
        ixx, iyy, _ = hessian(c_mel, sigma * scale)
        responses.append(-(ixx + iyy))  # bright blob in c_mel (a dark dot) -> positive
    stack = torch.stack(responses)
    best, idx = stack.max(0)
    sig = torch.tensor(LOG_SIGMAS, device=c_mel.device, dtype=c_mel.dtype)[idx]
    return best, sig


@torch.no_grad()
def compute_dsp(maps: ChromophoreMaps) -> DSPMaps:
    h, w = maps.c_mel.shape[-2:]
    scale = scale_factor(h, w)
    tub, blob, tidx = vessel_maps(maps.c_hb * maps.valid, scale)
    net, lam, th = network_maps(maps.c_mel * maps.valid, scale)
    dots, dsig = dot_maps(maps.c_mel * maps.valid, scale)
    return DSPMaps(tubularity=tub, blobness=blob, tub_sigma_index=tidx, network=net,
                   net_wavelength=lam, net_theta=th, dots=dots, dot_sigma=dsig)


@dataclass
class DSPThresholds:
    """Fold statistics over lesion pixels (s > 0.5) of the fold's training images."""

    tubularity_p75: float
    blobness_p75: float
    network_p75: float
    network_p25: float
    depth_p25: float
    mel_median: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def veil_map(maps: ChromophoreMaps, dsp: DSPMaps, thr: DSPThresholds) -> torch.Tensor:
    return ((maps.c_depth < thr.depth_p25) & (maps.c_mel > thr.mel_median)
            & (dsp.network < thr.network_p25) & maps.valid).float()


# ------------------------------------------------------------------ lesion-frame helpers
def radial_coords(geometry: LesionGeometry, h: int, w: int, device, dtype):
    """rho (in units of r) and phi (angle from the major axis) for every pixel; (B,1,H,W) each."""
    ys = torch.arange(h, device=device, dtype=dtype).view(1, 1, h, 1)
    xs = torch.arange(w, device=device, dtype=dtype).view(1, 1, 1, w)
    dy = ys - geometry.centroid[:, 0].view(-1, 1, 1, 1)
    dx = xs - geometry.centroid[:, 1].view(-1, 1, 1, 1)
    rho = torch.sqrt(dy ** 2 + dx ** 2) / geometry.radius.clamp_min(1e-6).view(-1, 1, 1, 1)
    phi = torch.atan2(dy, dx) - geometry.theta.view(-1, 1, 1, 1)
    return rho, torch.remainder(phi, 2 * math.pi)


def polar_grid(x: torch.Tensor, geometry: LesionGeometry, rho_range, n_sectors: int,
               n_radial: int) -> torch.Tensor:
    """Sample x on a polar grid about the lesion centroid, sectors measured from the major axis.
    Returns (B, C, n_radial, n_sectors)."""
    b, _, h, w = x.shape
    rhos = torch.linspace(rho_range[0], rho_range[1], n_radial, device=x.device, dtype=x.dtype)
    phis = (torch.arange(n_sectors, device=x.device, dtype=x.dtype) + 0.5) * 2 * math.pi / n_sectors
    rr, pp = torch.meshgrid(rhos, phis, indexing="ij")
    ang = pp.unsqueeze(0) + geometry.theta.view(-1, 1, 1)
    dist = rr.unsqueeze(0) * geometry.radius.view(-1, 1, 1)
    py = geometry.centroid[:, 0].view(-1, 1, 1) + dist * torch.sin(ang)
    px = geometry.centroid[:, 1].view(-1, 1, 1) + dist * torch.cos(ang)
    grid = torch.stack([px / (w - 1) * 2 - 1, py / (h - 1) * 2 - 1], dim=-1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)


def segmental_index(sector_values: torch.Tensor) -> torch.Tensor:
    """S = 1 - H(c / sum c) / ln(n): 0 = even around the rim, 1 = one sector. (B, n) -> (B,)."""
    c = sector_values.clamp_min(0)
    p = c / c.sum(1, keepdim=True).clamp_min(1e-12)
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum(1)
    return 1 - ent / math.log(c.shape[1])


def periphery_tokens(structure: torch.Tensor, geometry: LesionGeometry) -> dict[str, torch.Tensor]:
    """Handcrafted DRE-4 on a structure-energy map: segmental index, rim coverage, radial gradient."""
    grid = polar_grid(structure, geometry, PERIPHERY_RHO, N_SECTORS, N_RADIAL)[:, 0]  # (B,R,S)
    sectors = grid.mean(1)
    return {"segmental_index": segmental_index(sectors),
            "rim_coverage": (sectors > sectors.mean(1, keepdim=True)).float().mean(1),
            "radial_gradient": grid[:, -1].mean(1) - grid[:, 0].mean(1)}


def border_abruptness(c_mel: torch.Tensor, geometry: LesionGeometry, lesion_median: torch.Tensor,
                      t_mel: float | None = None) -> torch.Tensor:
    """Per-sector radial derivative of c_mel across [0.85r, 1.15r], normalised by the lesion
    median melanin; positive = pigment drops outward. (B, 16). With `t_mel`, rows whose
    |lesion median| < ABRUPT_MIN_MEDIAN_FRAC * t_mel are NaN (ratio undefined)."""
    grid = polar_grid(c_mel, geometry, BORDER_RHO, N_SECTORS, BORDER_SAMPLES)[:, 0]  # (B,R,S)
    span = (BORDER_RHO[1] - BORDER_RHO[0]) * geometry.radius.view(-1, 1)
    drop = (grid[:, 0] - grid[:, -1]) / span.clamp_min(1e-6)
    out = drop / lesion_median.abs().clamp_min(1e-6).view(-1, 1)
    if t_mel is not None:
        undefined = lesion_median.abs().view(-1, 1) < ABRUPT_MIN_MEDIAN_FRAC * t_mel
        out = torch.where(undefined, torch.full_like(out, float("nan")), out)
    return out


# ------------------------------------------------------------------ tokens
def _entropy(counts: torch.Tensor) -> torch.Tensor:
    p = counts / counts.sum().clamp_min(1e-12)
    return -(p * torch.log(p.clamp_min(1e-12))).sum()


def clark_evans(points: torch.Tensor, area: float) -> float:
    """R = mean nearest-neighbour distance / (0.5 sqrt(A / N)). >1 regular, <1 clustered."""
    n = points.shape[0]
    if n < 2 or area <= 0:
        return float("nan")
    d = torch.cdist(points, points)
    d.fill_diagonal_(float("inf"))
    return float(d.min(1).values.mean() / (0.5 * math.sqrt(area / n)))


@torch.no_grad()
def dsp_tokens(maps: ChromophoreMaps, dsp: DSPMaps, geometry: LesionGeometry,
               thr: DSPThresholds) -> list[dict[str, float]]:
    """About 16 scalar tokens per image, read inside the lesion. One dict per image."""
    b, _, h, w = maps.c_mel.shape
    scale = scale_factor(h, w)
    veil = veil_map(maps, dsp, thr)
    rho, phi = radial_coords(geometry, h, w, maps.c_mel.device, maps.c_mel.dtype)
    peak = (dsp.dots == F.max_pool2d(dsp.dots, 3, stride=1, padding=1))
    out = []
    for i in range(b):
        s = geometry.s[i, 0]
        lesion = (s > 0.5) & maps.valid[i, 0]
        area = float(lesion.sum())
        tok: dict[str, float] = {}
        if area < 4:
            out.append({k: float("nan") for k in TOKEN_NAMES})
            continue
        tub, blob = dsp.tubularity[i, 0], dsp.blobness[i, 0]
        net = dsp.network[i, 0]
        # vessels
        vessel = lesion & ((tub > thr.tubularity_p75) | (blob > thr.blobness_p75))
        tok["vessel_coverage"] = float(vessel.sum()) / area
        tok["tubular_blob_ratio"] = float((s * tub).sum() / (s * blob).sum().clamp_min(1e-8))
        tub_px = lesion & (tub > thr.tubularity_p75)
        sig_idx = dsp.tub_sigma_index[i, 0][tub_px]
        tok["calibre_entropy"] = float(_entropy(torch.bincount(sig_idx, minlength=3).float())) \
            if sig_idx.numel() else 0.0
        thin_idx = [HESSIAN_SIGMAS.index(sg) for sg in THIN_SIGMAS]
        is_blob = vessel & (blob > tub)
        is_tub = vessel & ~is_blob
        thin = is_tub & torch.isin(dsp.tub_sigma_index[i, 0],
                                   torch.tensor(thin_idx, device=tub.device))
        thick = is_tub & ~thin
        tok["vessel_polymorphism"] = float(_entropy(torch.stack(
            [thin.sum(), thick.sum(), is_blob.sum()]).float()))
        # network
        net_px = lesion & (net > thr.network_p75)
        tok["network_coverage"] = float(net_px.sum()) / area
        lam = dsp.net_wavelength[i, 0][net_px]
        tok["network_regularity"] = float(1 - lam.std(unbiased=False) / lam.mean().clamp_min(1e-8)) \
            if lam.numel() > 1 else float("nan")
        th = dsp.net_theta[i, 0][net_px]
        wts = net[net_px]
        tok["network_orientation_coherence"] = float(torch.sqrt(
            (wts * torch.cos(2 * th)).sum() ** 2 + (wts * torch.sin(2 * th)).sum() ** 2)
            / wts.sum().clamp_min(1e-8)) if th.numel() else float("nan")
        r_i = rho[i, 0]
        periph = net_px & (r_i >= PERIPHERY_RHO[0]) & (r_i <= PERIPHERY_RHO[1])
        centre = net_px & (r_i < PERIPHERY_RHO[0])
        cm = maps.c_mel[i, 0]
        tok["network_periphery_minus_centre"] = (float(cm[periph].mean() - cm[centre].mean())
                                                 if periph.any() and centre.any() else float("nan"))
        # dots / globules
        resp = dsp.dots[i, 0]
        thr_dot = resp[lesion].mean() + DOT_THRESHOLD_SD * resp[lesion].std(unbiased=False)
        found = peak[i, 0] & lesion & (resp > thr_dot)
        pts = torch.nonzero(found).float()
        n = pts.shape[0]
        tok["dot_density"] = n / (area / scale ** 2) * DENSITY_SCALE
        tok["dot_clark_evans"] = clark_evans(pts, area)
        rr = r_i[found]
        tok["dot_fraction_peripheral"] = float(((rr >= PERIPHERY_RHO[0]) & (rr <= PERIPHERY_RHO[1]))
                                               .float().mean()) if n else float("nan")
        sectors = torch.floor(phi[i, 0][found] / (2 * math.pi) * N_SECTORS).long().clamp(0, 15)
        tok["dot_sectors_occupied"] = float(torch.unique(sectors).numel()) / N_SECTORS if n else 0.0
        sz = dsp.dot_sigma[i, 0][found]
        tok["dot_size_cv"] = float(sz.std(unbiased=False) / sz.mean()) if n > 1 else float("nan")
        tok["dot_mean_depth"] = float(maps.c_depth[i, 0][found].mean()) if n else float("nan")
        # veil
        v = veil[i, 0]
        mass = float((s * v).sum())
        tok["veil_fraction"] = mass / float(s.sum())
        if mass > 0:
            ys = torch.arange(h, device=v.device, dtype=v.dtype).view(h, 1)
            xs = torch.arange(w, device=v.device, dtype=v.dtype).view(1, w)
            vy = float((s * v * ys).sum()) / mass
            vx = float((s * v * xs).sum()) / mass
            c = geometry.centroid[i]
            tok["veil_eccentricity"] = math.hypot(vy - float(c[0]), vx - float(c[1])) \
                / max(float(geometry.radius[i]), 1e-6)
        else:
            tok["veil_eccentricity"] = float("nan")
        out.append(tok)
    return out


TOKEN_NAMES = (
    "vessel_coverage", "tubular_blob_ratio", "calibre_entropy", "vessel_polymorphism",
    "network_coverage", "network_regularity", "network_orientation_coherence",
    "network_periphery_minus_centre", "dot_density", "dot_clark_evans",
    "dot_fraction_peripheral", "dot_sectors_occupied", "dot_size_cv", "dot_mean_depth",
    "veil_fraction", "veil_eccentricity",
)
