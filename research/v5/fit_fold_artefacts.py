# ruff: noqa: I001  -- precheck_common must be imported first: it hides the GPU before torch loads
"""Fit one fold's frozen chromophore artefacts from that fold's TRAINING images only.

    python -m research.v5.fit_fold_artefacts --fold 0

Writes results/v5/chromophore/fold{k}.json:
  * the ICA basis (Tsumura, on the OD plane orthogonal to shading), from ~10^6 non-glare,
    non-aperture pixels sampled evenly across the fold's training images;
  * t_mel (median lesion melanin) and the stem standardisation of [c_mel, c_hb, c_depth];
  * the DRE-1 palette (K = 6 k-means OD prototypes on lesion pixels);
  * the DRE-10 fold thresholds (tubularity / blobness / network P75, network P25, depth P25,
    melanin median over lesion pixels) and the DRE-4 abruptness P75.

Four streaming passes, one image in memory at a time (C3). Heavy CPU (C5): run it while a
384 px run is going or while the GPU is idle. It refuses to run before the E1 hash, because the
M1-QC / Q1 / Q2 / Q5 pre-checks read its output.
"""

from __future__ import annotations

from research.v5 import precheck_common as pc  # first: hides the GPU

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from research.v5 import chromophore as ch  # noqa: E402
from research.v5 import dsp  # noqa: E402

PER_IMAGE_CAP = 4096  # pixel reservoir cap per image and per quantity [impl]


def _sample(values: torch.Tensor, k: int, rng: np.random.Generator) -> np.ndarray:
    v = values.reshape(-1, values.shape[-1]) if values.dim() > 1 else values.reshape(-1)
    n = v.shape[0]
    if n == 0:
        return np.empty((0,) + tuple(v.shape[1:]), dtype=np.float32)
    idx = rng.choice(n, size=min(k, n), replace=False)
    return v[torch.from_numpy(idx)].numpy().astype(np.float32)


def training_rows(fold: int | None, loao_holdout: str | None):
    rows = pc.development_rows()
    if loao_holdout:
        return rows[rows["archive"] != loao_holdout].reset_index(drop=True)
    return pc.fold_train_rows(rows, fold)


@torch.no_grad()
def token_statistics(ids: list[str], artefacts: dict, image_size: int) -> dict:
    """Pass 4: token and DSP-map statistics through the SAME front the arms run, on the
    deterministic eval view (Resize + CenterCrop at the run resolution), unstandardised."""
    from PIL import Image

    from research.v4.recipe import Recipe, build_eval_transform
    from research.v5.arms import SCORE_ESC_MASS, ArmSpec
    from research.v5.front import DSP_MAP_NAMES, ChromophoreFront, token_names

    spec = ArmSpec(name="stats", modules=("chromophore", "palette", "dsp", "geometry"),
                   comparator=None, precondition=None, declared_score=SCORE_ESC_MASS,
                   hard_core="", falsifier="")
    front = ChromophoreFront(spec, artefacts, standardise=False)
    names = token_names(spec)
    transform = build_eval_transform(Recipe(image_size=image_size))
    t_sum = np.zeros(len(names))
    t_sq = np.zeros(len(names))
    t_n = np.zeros(len(names))
    m_sum = np.zeros(len(DSP_MAP_NAMES))
    m_sq = np.zeros(len(DSP_MAP_NAMES))
    m_n = 0
    started = time.time()
    for i, image_id in enumerate(ids):
        with Image.open(pc.IMAGE_DIR / f"{image_id}.jpg") as image:
            x = transform(image.convert("RGB")).unsqueeze(0)
        out = front(x)
        tok = out["tokens_raw"][0].double().numpy()
        ok = np.isfinite(tok)
        t_sum[ok] += tok[ok]
        t_sq[ok] += tok[ok] ** 2
        t_n[ok] += 1
        maps = out["channels"][0, 3:].double().flatten(1).numpy()  # the five DSP maps, raw
        m_sum += maps.sum(1)
        m_sq += (maps ** 2).sum(1)
        m_n += maps.shape[1]
        if (i + 1) % 500 == 0:
            rate = (i + 1) / (time.time() - started)
            print(f"  pass 4/4 tokens {i + 1:,}/{len(ids):,}  {rate:.1f} img/s", flush=True)
    t_mean = t_sum / np.maximum(t_n, 1)
    t_std = np.sqrt(np.maximum(t_sq / np.maximum(t_n, 1) - t_mean ** 2, 1e-12))
    m_mean = m_sum / max(m_n, 1)
    m_std = np.sqrt(np.maximum(m_sq / max(m_n, 1) - m_mean ** 2, 1e-12))
    return {"token_stats": {n: {"mean": float(t_mean[k]), "std": float(t_std[k]),
                                "n_finite": int(t_n[k])} for k, n in enumerate(names)},
            "dsp_map_stats": {n: {"mean": float(m_mean[k]), "std": float(m_std[k])}
                              for k, n in enumerate(DSP_MAP_NAMES)},
            "token_image_size": image_size}


@torch.no_grad()
def fit(fold: int | None, limit: int | None = None, seed: int = pc.SEED,
        loao_holdout: str | None = None, image_size: int = 224) -> dict:
    rows = training_rows(fold, loao_holdout)
    ids = rows["image_id"].astype(str).tolist()
    if limit:
        ids = ids[:limit]
    rng = np.random.default_rng(seed)
    per_image = max(1, int(np.ceil(ch.ICA_PIXELS / len(ids))))
    label = f"LOAO hold-out {loao_holdout}" if loao_holdout else f"fold {fold}"
    print(f"{label}: {len(ids):,} training images, {per_image} ICA pixels per image")

    # ---- pass 1: ICA pixels
    od_samples = []
    for _, _, x in pc.stream(ids, label="pass 1/4 ICA"):
        od = ch.optical_density(x)
        keep = (~(ch.glare_mask(x) | ch.aperture_mask(x)))[0, 0]
        od_samples.append(_sample(od[0].permute(1, 2, 0)[keep], per_image, rng))
    od_pixels = np.concatenate(od_samples)
    basis = ch.ChromophoreBasis.fit_ica(od_pixels, seed=seed, fold=fold)
    print(f"basis: v_mel={np.round(basis.v_mel, 3)} v_hb={np.round(basis.v_hb, 3)} "
          f"naming={basis.naming_cosines}")

    # ---- pass 2: lesion melanin (t_mel) and palette pixels. Geometry does not need t_mel.
    provisional = ch.ChromophoreBasis(**{**basis.__dict__, "t_mel": 1.0})
    mel_samples, pal_samples = [], []
    for _, _, x in pc.stream(ids, label="pass 2/4 lesion"):
        maps = ch.compute_maps(x, provisional)
        geo = ch.lesion_geometry(maps)
        lesion = ((geo.s > ch.LESION_PIXEL_THRESHOLD) & maps.valid)[0, 0]
        if bool(geo.fallback[0]) or int(lesion.sum()) == 0:
            continue
        mel_samples.append(_sample(maps.c_mel[0, 0][lesion], PER_IMAGE_CAP // 8, rng))
        pal_samples.append(_sample(maps.od[0].permute(1, 2, 0)[lesion], 64, rng))
    t_mel = float(np.median(np.concatenate(mel_samples)))
    basis.t_mel = t_mel
    pal_pixels = np.concatenate(pal_samples)
    if len(pal_pixels) > 200_000:
        pal_pixels = pal_pixels[rng.choice(len(pal_pixels), 200_000, replace=False)]
    palette = ch.Palette.fit(pal_pixels, seed=seed, fold=fold)
    print(f"t_mel={t_mel:.4f}; palette fitted on {len(pal_pixels):,} lesion pixels")

    # ---- pass 3: stem standardisation, DSP thresholds, abruptness percentile
    sums = np.zeros(3)
    sq = np.zeros(3)
    n = 0
    buckets: dict[str, list[np.ndarray]] = {k: [] for k in
                                            ("tub", "blob", "net", "depth", "mel", "abrupt")}
    for _, _, x in pc.stream(ids, label="pass 3/4 stats"):
        maps = ch.compute_maps(x, basis)
        stack = torch.cat([maps.c_mel, maps.c_hb, maps.c_depth], 1)[0]
        v = stack[:, maps.valid[0, 0]].double()
        sums += v.sum(1).numpy()
        sq += (v ** 2).sum(1).numpy()
        n += v.shape[1]
        geo = ch.lesion_geometry(maps)
        lesion = ((geo.s > ch.LESION_PIXEL_THRESHOLD) & maps.valid)[0, 0]
        if bool(geo.fallback[0]) or int(lesion.sum()) == 0:
            continue
        d = dsp.compute_dsp(maps)
        k = PER_IMAGE_CAP // 16
        buckets["tub"].append(_sample(d.tubularity[0, 0][lesion], k, rng))
        buckets["blob"].append(_sample(d.blobness[0, 0][lesion], k, rng))
        buckets["net"].append(_sample(d.network[0, 0][lesion], k, rng))
        buckets["depth"].append(_sample(maps.c_depth[0, 0][lesion], k, rng))
        buckets["mel"].append(_sample(maps.c_mel[0, 0][lesion], k, rng))
        med = maps.c_mel[0, 0][lesion].median().view(1)
        buckets["abrupt"].append(dsp.border_abruptness(maps.c_mel, geo, med, t_mel=basis.t_mel)[0].numpy())
    mean = sums / max(n, 1)
    std = np.sqrt(np.maximum(sq / max(n, 1) - mean ** 2, 1e-12))
    basis.channel_mean, basis.channel_std = mean.tolist(), std.tolist()
    cat = {k: np.concatenate(v) for k, v in buckets.items()}
    thresholds = dsp.DSPThresholds(
        tubularity_p75=float(np.percentile(cat["tub"], dsp.COVERAGE_PERCENTILE)),
        blobness_p75=float(np.percentile(cat["blob"], dsp.COVERAGE_PERCENTILE)),
        network_p75=float(np.percentile(cat["net"], dsp.COVERAGE_PERCENTILE)),
        network_p25=float(np.percentile(cat["net"], 25)),
        depth_p25=float(np.percentile(cat["depth"], 25)),
        mel_median=float(np.median(cat["mel"])))
    result = {"fold": fold, "loao_holdout": loao_holdout, "n_images": len(ids), "limit": limit,
              "seed": seed, "basis": basis.__dict__, "palette": palette.__dict__,
              "dsp_thresholds": thresholds.as_dict(),
              "abruptness_p75": float(np.nanpercentile(cat["abrupt"], 75)),
              "abruptness_undefined_values": int(np.isnan(cat["abrupt"]).sum()),
              "ica_pixels": int(len(od_pixels)),
              "declarations": {**ch.implementation_declarations(),
                               **dsp.implementation_declarations(),
                               "PER_IMAGE_CAP": PER_IMAGE_CAP,
                               "PRECHECK_SHORT_SIDE": pc.PRECHECK_SHORT_SIDE}}
    result.update(token_statistics(ids, result, image_size))
    return result


def load(fold: int) -> tuple[ch.ChromophoreBasis, ch.Palette, dsp.DSPThresholds, dict]:
    path = pc.FOLD_ARTEFACT_DIR / f"fold{fold}.json"
    if not path.is_file():
        raise SystemExit(f"{path} missing: run `python -m research.v5.fit_fold_artefacts "
                         f"--fold {fold}` first")
    data = json.loads(path.read_text(encoding="utf-8"))
    return (ch.ChromophoreBasis(**data["basis"]), ch.Palette(**data["palette"]),
            dsp.DSPThresholds(**data["dsp_thresholds"]), data)


def artefact_name(fold: int | None, loao_holdout: str | None, image_size: int) -> str:
    """fold<k>.json / loao-<archive>.json at 224 px (the screens); `_384` suffix at 384 px."""
    stem = f"loao-{loao_holdout}" if loao_holdout else f"fold{fold}"
    return f"{stem}{'' if image_size == 224 else f'_{image_size}'}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--fold", type=int)
    target.add_argument("--loao-holdout", choices=("ham", "bcn20000", "mskcc"))
    parser.add_argument("--image-size", type=int, choices=(224, 384), default=224,
                        help="resolution of the pass-4 token statistics (the run's resolution)")
    parser.add_argument("--limit", type=int, default=None,
                        help="first N training images only (timing trial; output marked)")
    args = parser.parse_args(argv)
    context = pc.start(f"fit_fold_artefacts {args.loao_holdout or f'fold {args.fold}'}")
    started = time.time()
    result = fit(args.fold, args.limit, loao_holdout=args.loao_holdout, image_size=args.image_size)
    result["context"] = context
    result["minutes"] = round((time.time() - started) / 60, 1)
    pc.FOLD_ARTEFACT_DIR.mkdir(parents=True, exist_ok=True)
    name = artefact_name(args.fold, args.loao_holdout, args.image_size)
    if args.limit:
        name = name.replace(".json", f"_limit{args.limit}.json")
    out = pc.FOLD_ARTEFACT_DIR / name
    out.write_text(json.dumps(result, indent=2, default=pc._json_default), encoding="utf-8")
    print(f"wrote {out.relative_to(pc.REPO_ROOT)} in {result['minutes']} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
