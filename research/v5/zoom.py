"""DRE-8 -- evidence-driven second look (docs/v5_design/V5_DERM_REASONING_ENGINE.md; runsheet 7).

Written 30 Sep as a NEW file while the night queue ran; wired into train_v5 the same evening:
  * the dataset yields the 448 px view (the arm's transform at image_size 448) and
    `global_view` derives the aligned 224 px view from it on the GPU;
  * `ZoomNet` wraps an `ArmNet` that has `twostep` + `clues`; `train_v5` calls it as
    `out = zoom_net(global_view(x448), x448)`; `compute_loss` adds `zoom_loss(...)`;
  * `train_v5.declared_score` computes the noisy-OR score from `s_esc` (global) and `s_esc_zoom`.
  * modes: "evidence" (zoom), "lesion" (8b: DRE-2 box x 1.2) and "random" (8r).

Window: p* = argmax of the clue map on the 224 view (stop-grad); side = max(0.35 * 2r, 0.25 x the
short side), taken from the 448 copy and resized to 224; passed through the SHARED trunk. Noisy-OR:
P_esc = 1 - (1 - P_global)(1 - P_zoom); both logits receive the escalation loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from research.v5.modules import F3_BLOCKS, ArmNet

CROP_LESION_FRACTION = 0.35  # of the lesion diameter 2r
CROP_MIN_FRACTION = 0.25  # of the short side
CROP_SIZE = 224
LESION_BOX_SCALE = 1.2  # 8b


def window_side(radius: torch.Tensor | float, short_side: float = 1.0) -> torch.Tensor:
    """Side in units of the short side: max(0.35 * 2r, 0.25). `radius` is in the same units."""
    r = radius if isinstance(radius, torch.Tensor) else torch.tensor(float(radius))
    return torch.clamp(CROP_LESION_FRACTION * 2.0 * r, min=CROP_MIN_FRACTION) * short_side


def argmax_centre(clue_map: torch.Tensor) -> torch.Tensor:
    """(B, 2) centres (cy, cx) in [0, 1] of the strongest clue cell; no gradient flows through it."""
    with torch.no_grad():
        b, _, h, w = clue_map.shape
        idx = clue_map.float().flatten(1).argmax(1)
        cy = ((idx // w).float() + 0.5) / h
        cx = ((idx % w).float() + 0.5) / w
        return torch.stack([cy, cx], dim=1)


def crop_and_resize(image: torch.Tensor, centre: torch.Tensor, side: torch.Tensor,
                    out_size: int = CROP_SIZE) -> torch.Tensor:
    """Bilinear crop of `image` (B, C, H, W) around `centre` (B, 2, in [0,1]) with `side` (B,) as a
    fraction of the short side, resized to out_size, via grid_sample. The window is shifted, not
    shrunk, when it would leave the image."""
    b, _, h, w = image.shape
    short = min(h, w)
    half_y = (side * short / h).clamp(max=1.0)  # fraction of the height
    half_x = (side * short / w).clamp(max=1.0)
    cy = torch.minimum(torch.maximum(centre[:, 0], half_y / 2), 1 - half_y / 2)
    cx = torch.minimum(torch.maximum(centre[:, 1], half_x / 2), 1 - half_x / 2)
    ys = torch.linspace(-0.5, 0.5, out_size, device=image.device)
    xs = torch.linspace(-0.5, 0.5, out_size, device=image.device)
    gy = (cy.view(b, 1) + ys.view(1, -1) * half_y.view(b, 1)) * 2 - 1  # (B, S) in [-1, 1]
    gx = (cx.view(b, 1) + xs.view(1, -1) * half_x.view(b, 1)) * 2 - 1
    grid = torch.stack([gx.unsqueeze(1).expand(b, out_size, out_size),
                        gy.unsqueeze(2).expand(b, out_size, out_size)], dim=-1)
    return F.grid_sample(image.float(), grid, mode="bilinear", padding_mode="border",
                         align_corners=False).to(image.dtype)


def noisy_or_logit(g: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """logit of 1 - (1 - sigmoid(g)) (1 - sigmoid(z)), computed stably."""
    p = 1.0 - torch.sigmoid(-g.float()) * torch.sigmoid(-z.float())
    p = p.clamp(1e-6, 1 - 1e-6)
    return torch.log(p) - torch.log1p(-p)


class ZoomNet(nn.Module):
    """Wraps an ArmNet (twostep + clues). The second look reuses the SAME trunk and esc head."""

    def __init__(self, net: ArmNet, mode: str = "evidence") -> None:
        super().__init__()
        if mode not in ("evidence", "lesion", "random"):
            raise ValueError(mode)
        if not (net.spec.has("twostep") and net.spec.has("clues")):
            raise ValueError("zoom needs an arm with twostep + clues")
        self.net, self.mode = net, mode

    def _zoom_logit(self, crop: torch.Tensor) -> torch.Tensor:
        n = self.net
        if n.front is not None:  # a composite with a widened stem needs the same extra channels
            extra = n.front(crop)
            if "channels" in extra:
                crop = torch.cat([crop, extra["channels"]], dim=1)
        f3 = n.base.features[:F3_BLOCKS](crop)
        pooled = n.pool(n.base.features[F3_BLOCKS:](f3))
        z = n.head[1](n.head[0](pooled))
        return n.esc_head(z).squeeze(1)

    def forward(self, x224: torch.Tensor, x448: torch.Tensor, box: torch.Tensor | None = None,
                generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        out = self.net(x224)
        b = x224.shape[0]
        if self.mode == "evidence":
            centre = argmax_centre(out["clue_map"])
            side = window_side(0.5).expand(b).to(x448.device)  # no lesion frame: r = half the frame
        elif self.mode == "lesion":
            if box is None:
                raise ValueError("mode 'lesion' needs the DRE-2 box (cy, cx, radius)")
            centre, side = box[:, :2], window_side(box[:, 2] * LESION_BOX_SCALE)
        else:
            gen = generator
            centre = torch.rand(b, 2, generator=gen, device="cpu").to(x448.device)
            side = window_side(0.5).expand(b).to(x448.device)
        crop = crop_and_resize(x448, centre.to(x448.device), side.to(x448.device))
        out["s_esc_zoom"] = self._zoom_logit(crop)
        out["s_esc_global"] = out["s_esc"]
        out["s_esc"] = noisy_or_logit(out["s_esc_global"], out["s_esc_zoom"])
        out["zoom_centre"] = centre
        return out


def zoom_loss(out: dict[str, torch.Tensor], esc_target: torch.Tensor,
              esc_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Escalation BCE on the zoom logit (the global logit's BCE is already in compute_loss);
    weighted 0.5 like the other escalation heads."""
    per = F.binary_cross_entropy_with_logits(out["s_esc_zoom"].float(), esc_target.float(),
                                             reduction="none")
    if esc_weight is not None:
        per = per * esc_weight
    return 0.5 * per.sum() / max(per.numel(), 1)


ZOOM_SOURCE_SIZE = 448


def global_view(x448: torch.Tensor, size: int = CROP_SIZE) -> torch.Tensor:
    """The 224 px global view, area-downsampled from the 448 px view on the GPU.

    [impl, wiring 30 Sep] The dataset applies the arm's ordinary transform at 448 px (the V4
    train transform, or the V4 eval transform, at `image_size = 448`) and the global view is
    derived from that same tensor. The earlier sketch loaded a separate, un-augmented 448 copy,
    whose pixels do not line up with a cropped / flipped / rotated 224 view, so the clue-map
    argmax would have pointed at the wrong place. Deriving both views from one augmented tensor
    keeps the window exactly aligned; RRC / rotation / jitter are scale-relative, so the 224 view
    has the control's augmentation distribution up to resampling order."""
    return F.interpolate(x448.float(), size=(size, size), mode="bilinear", antialias=True,
                         align_corners=False).to(x448.dtype)
