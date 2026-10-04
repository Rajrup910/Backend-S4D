"""Synthetic-image tests for research/v5/chromophore.py. No dataset reads.

    $env:CUDA_VISIBLE_DEVICES=""; python -m pytest tests/test_v5_chromophore.py -q
"""

from __future__ import annotations

import math

import numpy as np
import torch

from research.v5 import chromophore as ch

# Realistic camera-band OD spectra (R, G, B) for synthesis.
V_MEL = np.array([0.30, 0.60, 1.00])
V_HB = np.array([0.10, 1.00, 0.60])


def synth(c_mel: torch.Tensor, c_hb: torch.Tensor, shade: float = 0.15) -> torch.Tensor:
    """Concentration maps (B,1,H,W) -> sRGB (B,3,H,W) through Beer-Lambert."""
    vm = torch.tensor(V_MEL, dtype=torch.float32).view(1, 3, 1, 1)
    vh = torch.tensor(V_HB, dtype=torch.float32).view(1, 3, 1, 1)
    od = c_mel * vm + c_hb * vh + shade
    return ch.linear_to_srgb(torch.exp(-od))


def fitted_basis_from_truth() -> ch.ChromophoreBasis:
    rng = np.random.default_rng(0)
    n = 40_000
    cm, chb, cs = rng.exponential(0.4, n), rng.exponential(0.3, n), rng.uniform(0.05, 0.3, n)
    od = cm[:, None] * V_MEL + chb[:, None] * V_HB + cs[:, None]
    basis = ch.ChromophoreBasis.fit_ica(od, seed=0)
    basis.t_mel = 0.3
    return basis


THIRTY_DEGREES = math.radians(30)


def ellipse(h=128, w=128, cy=60.0, cx=70.0, a=36.0, b=18.0, theta=THIRTY_DEGREES):
    ys, xs = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(), indexing="ij")
    dy, dx = ys - cy, xs - cx
    u = dx * math.cos(theta) + dy * math.sin(theta)
    v = -dx * math.sin(theta) + dy * math.cos(theta)
    return ((u / a) ** 2 + (v / b) ** 2 <= 1.0).float().view(1, 1, h, w)


# ------------------------------------------------------------------ colour -> OD
def test_srgb_linear_roundtrip_and_od_is_finite_at_extremes():
    x = torch.linspace(0, 1, 101).view(1, 1, 1, -1).expand(1, 3, 1, 101)
    assert torch.allclose(ch.linear_to_srgb(ch.srgb_to_linear(x)), x, atol=1e-5)
    od = ch.optical_density(torch.stack([torch.zeros(3), torch.ones(3)]).view(2, 3, 1, 1))
    assert torch.isfinite(od).all() and (od > 0).all()  # upper clamp keeps OD strictly positive
    assert abs(float(od.max()) - math.log(255.0)) < 1e-4


def test_glare_and_aperture_masks():
    x = torch.full((1, 3, 2, 2), 0.5)
    x[0, :, 0, 0] = 0.0  # aperture
    x[0, 1, 1, 1] = 0.99  # glare in one channel
    assert bool(ch.aperture_mask(x)[0, 0, 0, 0]) and not bool(ch.aperture_mask(x)[0, 0, 1, 1])
    assert bool(ch.glare_mask(x)[0, 0, 1, 1]) and not bool(ch.glare_mask(x)[0, 0, 0, 1])


# ------------------------------------------------------------------ basis
def test_ica_recovers_melanin_and_haemoglobin_and_names_them():
    rng = np.random.default_rng(1)
    n = 40_000
    cm, chb, cs = rng.exponential(0.4, n), rng.exponential(0.3, n), rng.uniform(0.05, 0.3, n)
    od = cm[:, None] * V_MEL + chb[:, None] * V_HB + cs[:, None]
    basis = ch.ChromophoreBasis.fit_ica(od, seed=0)
    conc = basis.concentrations(torch.tensor(od.T, dtype=torch.float64).view(1, 3, n, 1))
    got_m, got_h = conc[0, 0, :, 0].numpy(), conc[0, 1, :, 0].numpy()
    # Naming and sign: melanin tracks c_mel, haemoglobin tracks c_hb, both positively.
    assert np.corrcoef(got_m, cm)[0, 1] > 0.98
    assert np.corrcoef(got_h, chb)[0, 1] > 0.98
    assert basis.naming_cosines["mel_vs_ref"] > 0.9 and basis.naming_cosines["hb_vs_ref"] > 0.9


def test_unmix_reconstruct_roundtrip():
    basis = fitted_basis_from_truth()
    od = torch.rand(2, 3, 5, 5) + 0.1
    assert torch.allclose(basis.reconstruct(basis.concentrations(od)), od, atol=1e-4)


def test_basis_save_load(tmp_path):
    basis = fitted_basis_from_truth()
    basis.save(tmp_path / "b.json")
    again = ch.ChromophoreBasis.load(tmp_path / "b.json")
    assert again.v_mel == basis.v_mel and again.t_mel == basis.t_mel


# ------------------------------------------------------------------ depth
def test_blue_grey_region_is_deeper_than_brown():
    """Dermal melanin: OD_B falls relative to OD_R. Same melanin load, lower B:R -> lower depth."""
    basis = fitted_basis_from_truth()
    h = w = 64
    c_mel = torch.full((1, 1, h, w), 0.02)
    c_mel[..., 20:44, 8:28] = 1.0  # brown (epidermal) patch
    c_mel[..., 20:44, 36:56] = 1.0  # same load, made blue-grey below
    srgb = synth(c_mel, torch.full_like(c_mel, 0.05))
    lin = ch.srgb_to_linear(srgb)
    od = -torch.log(lin.clamp(*ch.OD_CLAMP))
    od[:, 2, 20:44, 36:56] *= 0.55  # less blue absorption
    od[:, 0, 20:44, 36:56] *= 1.60  # more red absorption (Tyndall + deep melanin)
    srgb_bg = ch.linear_to_srgb(torch.exp(-od))
    maps = ch.compute_maps(srgb_bg, basis)
    brown = float(maps.c_depth[..., 24:40, 12:24].mean())
    blue_grey = float(maps.c_depth[..., 24:40, 40:52].mean())
    assert blue_grey < brown
    skin = float(maps.c_depth[..., 0:6, 0:6].abs().mean())
    assert skin < 0.1  # melanin gate zeros the ratio on skin


# ------------------------------------------------------------------ geometry
def test_lesion_geometry_recovers_rotated_ellipse():
    basis = fitted_basis_from_truth()
    mask = ellipse()
    c_mel = 0.05 + 0.9 * mask
    srgb = synth(c_mel, torch.full_like(c_mel, 0.05))
    geo = ch.lesion_geometry(ch.compute_maps(srgb, basis))
    assert not bool(geo.fallback[0])
    cy, cx = geo.centroid[0].tolist()
    assert abs(cy - 60) < 1.5 and abs(cx - 70) < 1.5
    angle = (float(geo.theta[0]) - math.radians(30)) % math.pi
    assert min(angle, math.pi - angle) < math.radians(3)
    hard = (geo.s > 0.5).float()
    dice = 2 * (hard * mask).sum() / (hard.sum() + mask.sum())
    assert float(dice) > 0.95
    # r = 2 sqrt(lambda_1): for a uniform ellipse lambda_1 = a^2 / 4, so r = a.
    assert abs(float(geo.radius[0]) - 36) < 2.5


def test_geometry_fallback_on_empty_image():
    basis = fitted_basis_from_truth()
    c = torch.full((1, 1, 64, 64), 0.05)
    geo = ch.lesion_geometry(ch.compute_maps(synth(c, c), basis))
    assert bool(geo.fallback[0])
    assert geo.centroid[0].tolist() == [31.5, 31.5]


def test_aperture_is_excluded_from_skin_ring():
    basis = fitted_basis_from_truth()
    mask = ellipse(cy=64, cx=64, a=20, b=14)
    c_mel = 0.05 + 0.9 * mask
    srgb = synth(c_mel, torch.full_like(c_mel, 0.05))
    ys, xs = torch.meshgrid(torch.arange(128).float(), torch.arange(128).float(), indexing="ij")
    outside = ((ys - 63.5) ** 2 + (xs - 63.5) ** 2) > 62 ** 2
    srgb[:, :, outside] = 0.0  # black circular aperture
    maps = ch.compute_maps(srgb, basis)
    assert abs(float(maps.m_skin[0]) - 0.05) < 0.05  # skin, not the black aperture
    geo = ch.lesion_geometry(maps)
    assert not bool(geo.fallback[0]) and float(geo.coverage[0]) < 0.3


# ------------------------------------------------------------------ asymmetry
def test_symmetric_lesion_has_low_asymmetry_asymmetric_high():
    basis = fitted_basis_from_truth()
    sym = ellipse()
    geo_maps = ch.compute_maps(synth(0.05 + 0.9 * sym, torch.full_like(sym, 0.05)), basis)
    a_sym = ch.chromophore_asymmetry(geo_maps, ch.lesion_geometry(geo_maps))
    lop = sym.clone()
    # One half lighter, but still well above skin, so Otsu keeps both halves in the soft mask.
    # (Lighten it to near-skin and the mask drops that half: asymmetry is measured within s.)
    lop[..., :, 70:] *= 0.5
    lop_maps = ch.compute_maps(synth(0.05 + 0.9 * lop, torch.full_like(lop, 0.05)), basis)
    a_lop = ch.chromophore_asymmetry(lop_maps, ch.lesion_geometry(lop_maps))
    assert float(a_sym.max()) < 0.12
    assert float(a_lop.max()) > float(a_sym.max()) + 0.15


def test_d4_reflections_are_four_distinct_involutions():
    x = torch.arange(16.0).view(1, 1, 4, 4)
    refl = ch.d4_reflections(x)
    assert len(refl) == 4
    assert len({tuple(r.flatten().tolist()) for r in refl}) == 4
    for k, r in enumerate(refl, start=1):  # a reflection applied twice is the identity
        again = torch.rot90(torch.flip(r, dims=[-1]), k, dims=(-2, -1))
        assert torch.equal(again, x)


# ------------------------------------------------------------------ palette
def test_palette_count_and_eccentricity():
    rng = np.random.default_rng(0)
    # Pairwise squared distances >= 0.4, so at tau = 0.05 a pixel's off-colour weight is < e^-8.
    centres = np.array([[0.2, 0.3, 0.4], [1.0, 1.3, 1.6], [2.0, 2.4, 2.8],
                        [0.4, 1.6, 1.0], [1.4, 0.6, 0.7], [2.6, 1.2, 3.4]])
    pixels = np.concatenate([c + rng.normal(0, 0.02, (500, 3)) for c in centres])
    pal = ch.Palette.fit(pixels, seed=0)
    h = w = 32
    od = torch.tensor(centres[0], dtype=torch.float32).view(1, 3, 1, 1).expand(1, 3, h, w).clone()
    od[..., :, 24:] = torch.tensor(centres[2], dtype=torch.float32).view(3, 1, 1)  # eccentric colour
    a = pal.assign(od)
    s = torch.ones(1, 1, h, w)
    geo = ch.LesionGeometry(s=s, centroid=torch.tensor([[15.5, 15.5]]), theta=torch.zeros(1),
                            radius=torch.tensor([16.0]), axes=torch.eye(2).flip(1).unsqueeze(0),
                            fallback=torch.zeros(1, dtype=torch.bool), coverage=torch.ones(1))
    tok = ch.palette_tokens(a, s, geo)
    assert abs(float(tok["count"][0]) - 2.0) < 0.2
    ecc = tok["eccentricity"][0][tok["area"][0] > 0.1]
    assert float(ecc.max()) > 0.6  # the right-hand strip is far from the centre


def test_implementation_declarations_are_listed():
    d = ch.implementation_declarations()
    assert {"APERTURE_SRGB_MAX", "RING_FRACTION", "REF_MEL_RGB"} <= set(d)
