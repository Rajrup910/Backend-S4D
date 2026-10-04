"""Synthetic-image tests for research/v5/dsp.py. No dataset reads.

    $env:CUDA_VISIBLE_DEVICES=""; python -m pytest tests/test_v5_dsp.py -q
"""

from __future__ import annotations

import math

import torch

from research.v5 import chromophore as ch
from research.v5 import dsp

H = W = 96


def blank(value: float = 0.0) -> torch.Tensor:
    return torch.full((1, 1, H, W), value)


def centred_geometry(radius: float = 30.0) -> ch.LesionGeometry:
    s = blank()
    ys, xs = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    s[0, 0] = (((ys - 47.5) ** 2 + (xs - 47.5) ** 2) <= radius ** 2).float()
    return ch.LesionGeometry(s=s, centroid=torch.tensor([[47.5, 47.5]]), theta=torch.zeros(1),
                             radius=torch.tensor([radius]),
                             axes=torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]),
                             fallback=torch.zeros(1, dtype=torch.bool), coverage=s.mean().view(1))


def test_derivative_kernels_are_exact_on_polynomials():
    for sigma in (0.5, 1.0, 1.5, 4.0):
        k1 = dsp._gauss_1d(sigma, 1, None, torch.float64)
        k2 = dsp._gauss_1d(sigma, 2, None, torch.float64)
        r = k1.numel() // 2
        x = torch.arange(-r, r + 1, dtype=torch.float64)
        assert abs(float(k2.sum())) < 1e-12 and abs(float(k1.sum())) < 1e-12
        # convolution at 0 = sum f(-x) k(x)
        assert abs(float((-x * k1).sum()) - 1.0) < 1e-9
        assert abs(float((x ** 2 * k2).sum()) - 2.0) < 1e-9


def test_tubularity_fires_on_a_line_and_blobness_on_a_dot():
    line = blank()
    line[..., 47:50, 10:86] = 1.0
    line = torch.nn.functional.avg_pool2d(line, 3, 1, 1)
    tub, blob, _ = dsp.vessel_maps(line, 1.0)  # sigma in px as at 224-px scale
    assert float(tub[..., 48, 30:66].mean()) > 5 * float(tub[..., 10:20, 10:20].mean() + 1e-6)
    dot = blank()
    dot[..., 46:50, 46:50] = 1.0
    dot = torch.nn.functional.avg_pool2d(dot, 3, 1, 1)
    tub_d, blob_d, _ = dsp.vessel_maps(dot, 1.0)
    assert float(blob_d[..., 47, 47]) > float(tub_d[..., 47, 47])
    assert float(tub[..., 48, 48]) > float(blob[..., 48, 48])


def test_gabor_finds_grating_wavelength_and_orientation():
    xs = torch.arange(W).float().view(1, 1, 1, W).expand(1, 1, H, W)
    for lam in (6.0, 10.0):
        grating = 0.5 + 0.5 * torch.cos(2 * math.pi * xs / lam)
        energy, wl, th = dsp.network_maps(grating, 1.0)
        centre = (slice(None), slice(None), slice(30, 66), slice(30, 66))
        assert float((wl[centre] == lam).float().mean()) > 0.9
        assert float((th[centre].abs() < 1e-6).float().mean()) > 0.9  # varies along x -> theta 0
    flat_energy, _, _ = dsp.network_maps(blank(0.7), 1.0)
    assert float(flat_energy.abs().max()) < 1e-4  # zero-DC kernels ignore a flat region


def test_clark_evans_regular_vs_clustered():
    regular = torch.stack(torch.meshgrid(torch.arange(0, 60, 10.0), torch.arange(0, 60, 10.0),
                                         indexing="ij"), -1).view(-1, 2)
    g = torch.Generator().manual_seed(0)
    clustered = torch.cat([torch.randn(18, 2, generator=g) * 1.5 + c
                           for c in (torch.tensor([10.0, 10.0]), torch.tensor([45.0, 45.0]))])
    assert dsp.clark_evans(regular, 3600.0) > 1.5
    assert dsp.clark_evans(clustered, 3600.0) < 0.6
    assert math.isnan(dsp.clark_evans(regular[:1], 3600.0))


def test_segmental_index_extremes():
    assert abs(float(dsp.segmental_index(torch.ones(1, 16)))) < 1e-6
    one = torch.zeros(1, 16)
    one[0, 3] = 1.0
    assert abs(float(dsp.segmental_index(one)) - 1.0) < 1e-6


def test_periphery_distinguishes_full_rim_from_one_segment():
    geo = centred_geometry(30.0)
    ys, xs = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    rho = torch.sqrt((ys - 47.5) ** 2 + (xs - 47.5) ** 2) / 30.0
    phi = torch.remainder(torch.atan2(ys - 47.5, xs - 47.5), 2 * math.pi)
    rim = ((rho > 0.7) & (rho < 1.1)).float().view(1, 1, H, W)
    segment = (rim[0, 0] * (phi < math.pi / 4)).view(1, 1, H, W)
    full = dsp.periphery_tokens(rim, geo)
    seg = dsp.periphery_tokens(segment, geo)
    assert float(full["segmental_index"]) < 0.05
    assert float(seg["segmental_index"]) > 0.5
    assert float(full["rim_coverage"]) > float(seg["rim_coverage"])


def test_border_abruptness_sharp_vs_fading_edge():
    geo = centred_geometry(30.0)
    ys, xs = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    r = torch.sqrt((ys - 47.5) ** 2 + (xs - 47.5) ** 2)
    sharp = (r <= 30).float().view(1, 1, H, W)
    fading = torch.clamp(1 - (r - 15) / 30, 0, 1).view(1, 1, H, W)
    med = torch.ones(1)
    assert float(dsp.border_abruptness(sharp, geo, med).mean()) > \
        2 * float(dsp.border_abruptness(fading, geo, med).mean())


def test_dsp_tokens_run_and_return_every_token():
    geo = centred_geometry(30.0)
    g = torch.Generator().manual_seed(1)
    c_mel = 0.1 + geo.s * (0.6 + 0.1 * torch.rand(1, 1, H, W, generator=g))
    for y, x in ((30, 30), (60, 62), (40, 65), (55, 35)):
        c_mel[..., y - 1:y + 2, x - 1:x + 2] += 0.8  # dark dots
    maps = ch.ChromophoreMaps(od=torch.rand(1, 3, H, W) + 0.1, c_mel=c_mel,
                              c_hb=0.05 + 0.05 * torch.rand(1, 1, H, W, generator=g),
                              c_depth=torch.zeros(1, 1, H, W), valid=torch.ones(1, 1, H, W).bool(),
                              m_skin=torch.tensor([0.1]), h_skin=torch.tensor([0.05]))
    maps_dsp = dsp.compute_dsp(maps)
    thr = dsp.DSPThresholds(tubularity_p75=0.1, blobness_p75=0.1, network_p75=0.05,
                            network_p25=0.01, depth_p25=-0.2, mel_median=0.5)
    tokens = dsp.dsp_tokens(maps, maps_dsp, geo, thr)
    assert set(tokens[0]) == set(dsp.TOKEN_NAMES)
    assert tokens[0]["dot_density"] > 0
