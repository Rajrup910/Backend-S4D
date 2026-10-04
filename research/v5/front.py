"""Arm front-end for `look`, `structure` and `geometry` (runsheet sections 6-7).

`ChromophoreFront` turns the ImageNet-normalised input batch into, with no gradient:
  * extra stem channels -- look: [c_mel, c_hb, c_depth] (6-channel stem); structure: + the five
    DSP maps (11-channel stem). Each standardised with the fold's training statistics;
  * handcrafted tokens  -- look: DRE-1 palette (13); structure: + DSP (16); geometry: DRE-3 colour
    and chromophore asymmetry, DRE-4 border abruptness and the fallback flag (7). Standardised
    with the fold's training mean / SD; a NaN token (e.g. no dots found) becomes 0, the fold mean;
  * the lesion geometry and palette maps that `GeometryHead` needs.

`GeometryHead` holds the LEARNED parts of DRE-3/4, which read the stride-16 map F3:
  structure asymmetry A_s (min, max over the lesion axes), pattern-count entropy over P = 8 learned
  prototypes (orthogonality 0.01), and the periphery channel c(phi) from a 1x1 conv on the polar-
  warped annulus -> segmental index S, rim coverage, radial gradient. 6 tokens, bounded, used raw.

Tokens are concatenated to the pooled feature, and the classifier columns that read them are
zero-initialised [impl], so every arm starts as its comparator and earns its extra input.
All statistics come from results/v5/chromophore/fold{k}.json (fit_fold_artefacts.py).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from research.v4.recipe import IMAGENET_MEAN, IMAGENET_STD
from research.v5 import chromophore as ch
from research.v5 import dsp
from research.v5.arms import ArmSpec

PALETTE_TOKENS = ([f"palette_area_{k}" for k in range(ch.PALETTE_K)] + ["palette_count"]
                  + [f"palette_ecc_{k}" for k in range(ch.PALETTE_K)])
DSP_MAP_NAMES = ("tubularity", "blobness", "network", "dots", "veil")
GEOMETRY_TOKENS = ("colour_asym_min", "colour_asym_max", "chrom_asym_min", "chrom_asym_max",
                   "abrupt_fraction", "abrupt_angular_variance", "geometry_fallback")
LEARNED_GEOMETRY_TOKENS = ("structure_asym_min", "structure_asym_max", "pattern_entropy",
                           "segmental_index", "rim_coverage", "radial_gradient")
PATTERN_PROTOTYPES = 8
TOKEN_Z_MAX = 50.0  # [impl, NaN fix] backstop bound on a standardised token; fp16 max is 65,504
PATTERN_TAU = 0.1  # [impl] same temperature as the DRE-6 memory head
PATTERN_ORTHO_WEIGHT = 0.01  # runsheet: [declared here]
F3_CHANNELS = 384


def token_names(spec: ArmSpec) -> list[str]:
    names: list[str] = []
    if spec.has("palette"):
        names += PALETTE_TOKENS
    if spec.has("dsp"):
        names += list(dsp.TOKEN_NAMES)
    if spec.has("geometry"):
        names += list(GEOMETRY_TOKENS)
    return names


def extra_channels(spec: ArmSpec) -> int:
    return (3 if spec.has("chromophore") else 0) + (len(DSP_MAP_NAMES) if spec.has("dsp") else 0)


def needs_front(spec: ArmSpec) -> bool:
    return bool({"chromophore", "palette", "dsp", "geometry"} & set(spec.modules))


class ChromophoreFront(nn.Module):
    def __init__(self, spec: ArmSpec, artefacts: dict, standardise: bool = True) -> None:
        super().__init__()
        self.spec = spec
        self.basis = ch.ChromophoreBasis(**artefacts["basis"])
        self.palette = ch.Palette(**artefacts["palette"])
        self.thresholds = dsp.DSPThresholds(**artefacts["dsp_thresholds"])
        self.abrupt_p75 = float(artefacts.get("abruptness_p75", 0.0))
        self.geometry_fallback_forced = bool(artefacts.get("force_d4_fallback", False))
        self.standardise = standardise
        self.backstop_hits = 0
        self.names = token_names(spec)
        stats = artefacts.get("token_stats", {})
        mean = [stats.get(n, {}).get("mean", 0.0) for n in self.names]
        std = [stats.get(n, {}).get("std", 1.0) or 1.0 for n in self.names]
        self.register_buffer("token_mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("token_std", torch.tensor(std, dtype=torch.float32))
        dmaps = artefacts.get("dsp_map_stats", {})
        self.register_buffer("dsp_mean", torch.tensor(
            [dmaps.get(n, {}).get("mean", 0.0) for n in DSP_MAP_NAMES], dtype=torch.float32))
        self.register_buffer("dsp_std", torch.tensor(
            [dmaps.get(n, {}).get("std", 1.0) or 1.0 for n in DSP_MAP_NAMES], dtype=torch.float32))

    @property
    def n_tokens(self) -> int:
        return len(self.names)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # fp32 regardless of AMP: the unmixing, Hessians and Gabor energies are fixed physics, and
        # fp16 would quantise OD differences that the depth and abruptness tokens are built on.
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self._forward(x)

    def _forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        srgb = ch.unnormalise(x.float(), IMAGENET_MEAN, IMAGENET_STD)
        maps = ch.compute_maps(srgb, self.basis)
        geo = ch.lesion_geometry(maps, force_fallback=self.geometry_fallback_forced)
        out: dict[str, torch.Tensor] = {}
        channels, tokens = [], []
        if self.spec.has("chromophore"):
            channels.append(ch.stem_channels(maps, self.basis))
        a = None
        if self.spec.has("palette") or self.spec.has("geometry"):
            a = self.palette.assign(maps.od)
        if self.spec.has("palette"):
            t = ch.palette_tokens(a, geo.s, geo)
            tokens.append(torch.cat([t["area"], t["count"].unsqueeze(1), t["eccentricity"]], 1))
        if self.spec.has("dsp"):
            d = dsp.compute_dsp(maps)
            veil = dsp.veil_map(maps, d, self.thresholds)
            stack = torch.cat([d.tubularity, d.blobness, d.network, d.dots, veil], 1)
            channels.append((stack - self.dsp_mean.view(1, -1, 1, 1))
                            / self.dsp_std.view(1, -1, 1, 1))
            rows = dsp.dsp_tokens(maps, d, geo, self.thresholds)
            tokens.append(torch.tensor([[r[n] for n in dsp.TOKEN_NAMES] for r in rows],
                                       device=x.device, dtype=torch.float32))
        if self.spec.has("geometry"):
            use_d4 = self.geometry_fallback_forced
            mirrors = ch.d4_reflections(a) if use_d4 else ch.reflect_about_axes(a, geo)
            s = torch.ones_like(geo.s) if use_d4 else geo.s
            colour = ch.asymmetry(a, s, mirrors)
            chrom = ch.chromophore_asymmetry(maps, geo, use_d4=use_d4)
            lesion = (geo.s > ch.LESION_PIXEL_THRESHOLD) & maps.valid
            med = torch.where(lesion, maps.c_mel, torch.full_like(maps.c_mel, float("nan"))
                              ).flatten(1).nanmedian(1).values.nan_to_num(1.0)
            abrupt = dsp.border_abruptness(maps.c_mel, geo, med, t_mel=self.basis.t_mel)  # (B,16)
            defined = torch.isfinite(abrupt).all(1)
            is_abrupt = torch.where(defined, (abrupt > self.abrupt_p75).float().mean(1),
                                    torch.full_like(abrupt[:, 0], float("nan")))
            tokens.append(torch.stack([colour.min(1).values, colour.max(1).values,
                                       chrom.min(1).values, chrom.max(1).values,
                                       is_abrupt, abrupt.var(1, unbiased=False),
                                       geo.fallback.float()], 1))
            out["geometry"] = geo  # type: ignore[assignment]
            out["palette_maps"] = a
        if channels:
            out["channels"] = torch.cat(channels, 1).to(x.dtype)
        if tokens:
            raw = torch.cat(tokens, 1)
            out["tokens_raw"] = raw
            z = (raw - self.token_mean) / self.token_std if self.standardise else raw
            z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            if self.standardise:
                # Backstop (NaN fix, 2026-10-02): a finite-but-huge token survives nan_to_num and
                # overflows fp16 in the classifier. The blow-ups are fixed at source (geometry fallback,
                # abruptness floor); what still exceeds TOKEN_Z_MAX is a real heavy tail (checked: raw
                # abrupt variance 0.56, z 98), so it saturates at +/-TOKEN_Z_MAX -- sign and 'extreme'
                # kept, fp16-safe -- and is counted (`backstop_hits`, in the run JSON).
                over = z.abs() > TOKEN_Z_MAX
                self.backstop_hits += int(over.sum())
                z = z.clamp(-TOKEN_Z_MAX, TOKEN_Z_MAX)
            out["tokens"] = z
        return out


class GeometryHead(nn.Module):
    """Learned DRE-3/4 parts on F3, in the frame of the DRE-2 lesion geometry."""

    def __init__(self) -> None:
        super().__init__()
        self.patterns = nn.Parameter(torch.randn(PATTERN_PROTOTYPES, F3_CHANNELS) * 0.02)
        self.periphery = nn.Conv2d(F3_CHANNELS, 1, kernel_size=1)

    def forward(self, f3: torch.Tensor, geo: ch.LesionGeometry, image_hw: tuple[int, int],
                use_d4: bool = False) -> torch.Tensor:
        f3 = f3.float()
        b, _, h, w = f3.shape
        sy, sx = h / image_hw[0], w / image_hw[1]
        s = F.interpolate(geo.s.float(), size=(h, w), mode="area")
        centroid = torch.stack([(geo.centroid[:, 0] + 0.5) * sy - 0.5,
                                (geo.centroid[:, 1] + 0.5) * sx - 0.5], 1)
        radius = geo.radius * (sy + sx) / 2
        small = ch.LesionGeometry(s=s, centroid=centroid, theta=geo.theta, radius=radius,
                                  axes=geo.axes, fallback=geo.fallback, coverage=geo.coverage)
        mirrors = ch.d4_reflections(f3) if use_d4 else ch.reflect_about_axes(f3, small)
        a_s = ch.asymmetry(f3, torch.ones_like(s) if use_d4 else s, mirrors)

        feat = F.normalize(f3.permute(0, 2, 3, 1).reshape(b, h * w, -1), dim=-1)
        protos = F.normalize(self.patterns, dim=-1)
        assign = torch.softmax(feat @ protos.t() / PATTERN_TAU, dim=-1)  # (B, HW, P)
        weights = s.flatten(1).unsqueeze(-1)
        hist = (assign * weights).sum(1) / weights.sum(1).clamp_min(1e-6)
        entropy = -(hist * torch.log(hist.clamp_min(1e-12))).sum(1) / math.log(PATTERN_PROTOTYPES)

        polar = dsp.polar_grid(f3, small, dsp.PERIPHERY_RHO, dsp.N_SECTORS, dsp.N_RADIAL)
        c = F.softplus(self.periphery(polar))[:, 0]  # (B, radial, sectors), non-negative [impl]
        sectors = c.mean(1)
        seg = dsp.segmental_index(sectors)
        rim = (sectors > sectors.mean(1, keepdim=True)).float().mean(1)
        radial = c[:, -1].mean(1) - c[:, 0].mean(1)
        return torch.stack([a_s.min(1).values, a_s.max(1).values, entropy, seg, rim, radial], 1)

    def orthogonality(self) -> torch.Tensor:
        p = F.normalize(self.patterns, dim=-1)
        gram = p @ p.t()
        off = gram - torch.eye(p.shape[0], device=p.device)
        return (off ** 2).sum() / (p.shape[0] * (p.shape[0] - 1))
