"""S51 / Phase R -- frozen-feature extraction for the backbone probe.

H2 certified that there is **no head-recoverable headroom on ConvNeXt-Tiny features**, over a
probe family that included the deployed head's own functional form. That is a statement about
one representation and nothing else. S51 asks the only question that follows from it: does a
better representation exist at all? This module produces the inputs to that test.

Three backbones, all **frozen** -- forward passes only, no weights are trained here:

    convnext_tiny     ml/checkpoints/convnext_tiny_best.HAM-only.pt   control, the deployed trunk
    panderm_vitb16    DermLIP PanDerm-base vision tower (ViT-B/16)    derm foundation model
    dinov2_vitb14     timm vit_base_patch14_dinov2.lvd142m            general-purpose control

## Why this checkpoint for PanDerm, and not the one the paper links

The PanDerm release ships its checkpoints as pickled `.pth` files on Google Drive. The only
Hugging Face copies of those exact files are unaffiliated third-party mirrors with no licence
declaration and no download history, and loading one means unpickling code from an untrusted
host. `redlessone/DermLIP_PanDerm-base-w-PubMed-256` is published by the PanDerm/DermLIP
authors' own account, ships **safetensors** (no code execution), and its `open_clip_config.json`
declares exactly the tower the plan names: ViT-B/16, 224 px, 12 layers, width 768. It is
PanDerm-base *after* DermLIP's CLIP alignment against PubMed text rather than the raw
masked-latent checkpoint, and every report this module feeds says so. That is the honest trade,
and it is recorded rather than hidden.

## The pooling question, settled before extraction and not after

The DermLIP tower is BEiT-shaped (`gamma_1`/`gamma_2` layer scale, split `q_bias`/`v_bias`, no
`k_bias`) with an absolute `pos_embed`, so it maps one-to-one onto timm's `vit_base_patch16_224`
with `init_values` set; the remap is in `_load_panderm` and is asserted exhaustive. What the
config does **not** state is whether its CLIP head consumed the CLS token or the mean of the
patch tokens. Rather than guess, and then discover which guess flattered the result, both are
extracted in the same pass at no extra GPU cost and stored side by side:

    features      CLS token after the final norm   -- the **pre-registered primary**
    features_alt  mean of patch tokens after norm  -- declared robustness arm

ConvNeXt-Tiny has no such ambiguity: its features come from `research.selective.features
.extract_features`' own forward hook, so the definition of "penultimate" is identical to every
cached feature file in this repository, and `features_alt` is absent for it.

Each backbone is normalised with **its own** declared preprocessing statistics -- ImageNet for
ConvNeXt, OpenAI-CLIP for the DermLIP tower, timm's config for DINOv2 -- because running a
frozen model under the wrong normalisation measures the mismatch, not the representation.

    $py -m research.v4.extract_backbone_features --selftest
    $py -m research.v4.extract_backbone_features --splits reserved val
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "ml" / "data" / "manifest_v4.csv"
IMAGE_DIR = REPO_ROOT / "data" / "external" / "isic2019_images" / "ISIC_2019_Training_Input"
OUT_DIR = REPO_ROOT / "research" / "v4" / "features"
CONVNEXT_CKPT = REPO_ROOT / "ml" / "checkpoints" / "convnext_tiny_best.HAM-only.pt"

PANDERM_REPO = "redlessone/DermLIP_PanDerm-base-w-PubMed-256"
PANDERM_FILE = "open_clip_model.safetensors"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RESIZE_RATIO = 256 / 224
DINOV2_IMAGE_SIZE = 224


# --------------------------------------------------------------------- dataset
class _ManifestDataset(Dataset):
    """Explicit manifest rows read from the ISIC-2019 input directory.

    Deliberately not `LesionDataset`: that class selects by a `split` column whose vocabulary is
    HAM's, and the V4 corpus uses its own five-way split. Rows are selected here by `image_id`.
    """

    def __init__(self, frame: pd.DataFrame,
                 transform: Callable[[Image.Image], torch.Tensor]) -> None:
        self._ids = frame["image_id"].astype(str).tolist()
        self._labels = frame["class_index_7"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        with Image.open(IMAGE_DIR / f"{self._ids[index]}.jpg") as image:
            image = image.convert("RGB")
        return self.transform(image), self._labels[index], self._ids[index]


def _eval_transform(size: int, mean, std) -> transforms.Compose:
    """Resize shortest side -> centre crop -> normalise.

    Structurally identical for every backbone; only the crop size and the normalisation
    statistics differ, and both come from the backbone's own declared config.
    """
    return transforms.Compose([
        transforms.Resize(int(round(size * RESIZE_RATIO)),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# --------------------------------------------------------------------- backbones
def _load_panderm() -> nn.Module:
    """DermLIP's PanDerm-base vision tower, remapped onto timm's ViT-B/16.

    Three key-level differences, all mechanical, all asserted:
      * `gamma_1` / `gamma_2`  -> `ls1.gamma` / `ls2.gamma`      (layer scale)
      * `q_bias` + `v_bias`    -> `qkv.bias` with a zero k block (BEiT gives k no bias)
      * the `visual.head` CLIP projection is dropped -- this module wants the 768-d
        pre-projection representation, which is what a linear probe should see.
    """
    import timm
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    raw = load_file(hf_hub_download(PANDERM_REPO, PANDERM_FILE))
    src = {k[len("visual."):]: v for k, v in raw.items() if k.startswith("visual.")}
    if not src:
        raise RuntimeError(f"{PANDERM_REPO}/{PANDERM_FILE} carries no `visual.*` tower")

    model = timm.create_model("vit_base_patch16_224", pretrained=False,
                              num_classes=0, init_values=1e-5)
    target = model.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()

    for key, value in src.items():
        if key.startswith("head."):
            consumed.add(key)
            continue
        if key.endswith(".attn.q_bias") or key.endswith(".attn.v_bias"):
            continue                                       # folded below
        mapped[key.replace(".gamma_1", ".ls1.gamma").replace(".gamma_2", ".ls2.gamma")] = value
        consumed.add(key)

    for block in range(len(model.blocks)):
        q = src[f"blocks.{block}.attn.q_bias"]
        v = src[f"blocks.{block}.attn.v_bias"]
        mapped[f"blocks.{block}.attn.qkv.bias"] = torch.cat([q, torch.zeros_like(q), v])
        consumed |= {f"blocks.{block}.attn.q_bias", f"blocks.{block}.attn.v_bias"}

    leftover = sorted(set(src) - consumed)
    if leftover:
        raise RuntimeError(f"unmapped PanDerm tower keys: {leftover[:8]}")
    extra = sorted(k for k in mapped if k not in target)
    if extra:
        raise RuntimeError(f"remapped keys absent from timm ViT-B/16: {extra[:8]}")
    uninitialised = sorted(k for k in target if k not in mapped and not k.startswith("head."))
    if uninitialised:
        raise RuntimeError(f"timm ViT-B/16 keys left uninitialised: {uninitialised[:8]}")
    bad = [(k, tuple(mapped[k].shape), tuple(target[k].shape))
           for k in mapped if mapped[k].shape != target[k].shape]
    if bad:
        raise RuntimeError(f"shape mismatch after remap: {bad[:4]}")

    model.load_state_dict(mapped, strict=False)
    return model


def _vit_embed(model: nn.Module, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(CLS after the final norm, mean of the patch tokens after the final norm)."""
    tokens = model.forward_features(batch)
    prefix = int(getattr(model, "num_prefix_tokens", 1))
    return tokens[:, 0], tokens[:, prefix:].mean(dim=1)


def _build(name: str, device: torch.device):
    """Return (model, transform, embed_fn, meta). `embed_fn` yields (primary, alt|None)."""
    if name == "convnext_tiny":
        from ml.training.common import build_model_from_checkpoint
        from research.selective.features import _final_linear
        model, payload = build_model_from_checkpoint(CONVNEXT_CKPT, device)
        captured: list[torch.Tensor] = []
        _final_linear(model).register_forward_hook(
            lambda _m, inputs, _o: captured.append(inputs[0].detach().float()))

        def embed(module: nn.Module, batch: torch.Tensor):
            captured.clear()
            module(batch)
            if len(captured) != 1:
                raise RuntimeError(f"expected 1 head activation per batch, got {len(captured)}")
            return captured[0], None

        meta = {"source": str(CONVNEXT_CKPT.relative_to(REPO_ROOT)), "licence": "in-repo",
                "image_size": 224, "pooling": "penultimate, research.selective.features hook",
                "arch": payload.get("arch", "convnext_tiny"),
                "pretraining": "ImageNet init, supervised fine-tune on the HAM10000 train split"}
        return (model.to(device).eval(),
                _eval_transform(224, IMAGENET_MEAN, IMAGENET_STD), embed, meta)

    if name == "panderm_vitb16":
        model = _load_panderm()
        meta = {"source": f"hf:{PANDERM_REPO}/{PANDERM_FILE}",
                "licence": "cc-by-4.0 per the model card; PanDerm itself is CC-BY-NC-4.0",
                "image_size": 224, "pooling": "primary=CLS after norm; alt=mean patch tokens",
                "arch": "vit_base_patch16_224 (DermLIP PanDerm-base vision tower)",
                "pretraining": "PanDerm masked latent modelling (2M derm images) "
                               "+ DermLIP CLIP alignment against PubMed text"}
        return model.to(device).eval(), _eval_transform(224, CLIP_MEAN, CLIP_STD), _vit_embed, meta

    if name == "dinov2_vitb14":
        import timm
        #  timm's default config for this checkpoint is 518 px. Running it there would confound
        #  the arm with input resolution -- which is rung R1's separate question in S52, not
        #  S51's -- and give DINOv2 a 5.3x token budget over the two 224 px arms. Pinned to 224
        #  (the standard DINOv2 linear-probe setting) so the only thing that varies between arms
        #  is the representation. Recorded as deviation D1 in the S51 report.
        model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True,
                                  num_classes=0, img_size=DINOV2_IMAGE_SIZE)
        cfg = timm.data.resolve_data_config({}, model=model)
        size = DINOV2_IMAGE_SIZE
        meta = {"source": "hf:timm/vit_base_patch14_dinov2.lvd142m", "licence": "apache-2.0",
                "image_size": size, "pooling": "primary=CLS after norm; alt=mean patch tokens",
                "arch": "vit_base_patch14_dinov2",
                "deviation_D1": "pinned to 224 px; timm default 518 px would confound the arm "
                                "with input resolution and is rung R1's question, not S51's",
                "pretraining": "LVD-142M, self-supervised, general-purpose"}
        return model.to(device).eval(), _eval_transform(size, cfg["mean"], cfg["std"]), _vit_embed, meta

    raise ValueError(f"unknown backbone {name!r}")


BACKBONES = ("convnext_tiny", "panderm_vitb16", "dinov2_vitb14")


def features_path(backbone: str, splits: list[str]) -> Path:
    return OUT_DIR / f"{backbone}_{'-'.join(sorted(splits))}.npz"


# --------------------------------------------------------------------- extraction
@torch.no_grad()
def extract(backbone: str, splits: list[str], batch_size: int, workers: int,
            device: torch.device) -> dict:
    manifest = pd.read_csv(MANIFEST, low_memory=False)
    frame = manifest[manifest["split"].isin(splits)].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"no manifest rows for splits {splits}")

    model, transform, embed, meta = _build(backbone, device)
    #  Windows spawns workers as full processes that each re-import torch; more than a
    #  handful exhausts the paging file before a single batch is decoded (WinError 1455).
    loader = DataLoader(_ManifestDataset(frame, transform), batch_size=batch_size,
                        shuffle=False, num_workers=workers, pin_memory=True,
                        persistent_workers=workers > 0, prefetch_factor=4 if workers else None)

    primary_chunks, alt_chunks, labels, ids = [], [], [], []
    started = time.time()
    use_amp = device.type == "cuda"
    for step, (images, batch_labels, batch_ids) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            primary, secondary = embed(model, images)
        primary_chunks.append(primary.float().cpu().numpy())
        if secondary is not None:
            alt_chunks.append(secondary.float().cpu().numpy())
        labels.append(np.asarray(batch_labels))
        ids.extend(batch_ids)
        if step % 25 == 0:
            rate = len(ids) / max(time.time() - started, 1e-9)
            print(f"    {len(ids):>6}/{len(frame)}  {rate:6.1f} img/s", flush=True)

    features = np.concatenate(primary_chunks, axis=0)
    payload = {"features": features, "labels": np.concatenate(labels, axis=0),
               "image_ids": np.asarray([str(i) for i in ids])}
    if alt_chunks:
        payload["features_alt"] = np.concatenate(alt_chunks, axis=0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = features_path(backbone, splits)
    np.savez_compressed(out, **payload)
    elapsed = time.time() - started
    meta |= {"backbone": backbone, "splits": sorted(splits), "n": int(len(frame)),
             "dim": int(features.shape[1]), "has_alt": bool(alt_chunks),
             "batch_size": batch_size, "seconds": round(elapsed, 1),
             "path": str(out.relative_to(REPO_ROOT))}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  wrote {out.relative_to(REPO_ROOT)}  {features.shape}  in {elapsed / 60:.1f} min")
    return meta


# --------------------------------------------------------------------- self-test
def selftest() -> int:
    print("extract_backbone_features.py self-test\n")
    ok = True

    manifest = pd.read_csv(MANIFEST, low_memory=False)
    reserved = manifest[manifest["split"] == "reserved"]
    present = sum((IMAGE_DIR / f"{i}.jpg").is_file() for i in reserved["image_id"].head(200))
    print(f"  1. reserved images resolve on disk: {present}/200 "
          f"-> {'PASS' if present == 200 else 'FAIL'}")
    ok &= present == 200

    #  The reserved cohort must contain no image the control checkpoint was trained on, or the
    #  control carries a supervised advantage the two foundation models do not have.
    v1 = pd.read_csv(REPO_ROOT / "ml" / "configs" / "splits" / "split_v1.csv")
    overlap = len(set(reserved["image_id"].astype(str))
                  & set(v1.loc[v1["split"] == "train", "image_id"].astype(str)))
    print(f"  2. reserved x HAM-train overlap = {overlap} -> {'PASS' if overlap == 0 else 'FAIL'}")
    ok &= overlap == 0

    batch = torch.randn(2, 3, 224, 224)
    model = _load_panderm().eval()
    with torch.no_grad():
        cls, mean = _vit_embed(model, batch)
    shapes_ok = tuple(cls.shape) == (2, 768) and tuple(mean.shape) == (2, 768)
    print(f"  3. PanDerm tower remaps and runs: cls{tuple(cls.shape)} mean{tuple(mean.shape)} "
          f"-> {'PASS' if shapes_ok else 'FAIL'}")
    ok &= shapes_ok

    #  A remap that silently dropped the pretrained weights would still produce the right
    #  shapes. Compare against a randomly initialised tower: the loaded one must differ.
    import timm
    fresh = timm.create_model("vit_base_patch16_224", pretrained=False,
                              num_classes=0, init_values=1e-5).eval()
    with torch.no_grad():
        fresh_cls, _ = _vit_embed(fresh, batch)
    gap = float(torch.abs(cls - fresh_cls).max())
    print(f"  4. loaded weights differ from random init: max|d|={gap:.4f} "
          f"-> {'PASS' if gap > 1e-3 else 'FAIL'}")
    ok &= gap > 1e-3

    #  The two poolings must not be the same tensor -- that would make the robustness arm a copy
    #  of the primary, certifying nothing.
    distinct = float(torch.abs(cls - mean).max()) > 1e-3
    print(f"  5. CLS and mean-pool are distinct readouts -> {'PASS' if distinct else 'FAIL'}")
    ok &= distinct

    print(f"\n{'ALL CHECKS PASS' if ok else 'SELF-TEST FAILED'}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S51 -- frozen backbone feature extraction")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--backbones", nargs="+", default=list(BACKBONES), choices=list(BACKBONES))
    parser.add_argument("--splits", nargs="+", default=["reserved", "val"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  splits={args.splits}  batch={args.batch_size}\n")
    for backbone in args.backbones:
        print(f"[{backbone}]")
        extract(backbone, args.splits, args.batch_size, args.workers, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
