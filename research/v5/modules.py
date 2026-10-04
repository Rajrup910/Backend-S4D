"""V5 model modules -- the parameter cards of docs/V5_RUNSHEET.md section 7, as code.

Every constant below is a value the runsheet fixes; nothing is tuned. The wrapped trunk is the V4
control's torchvision ConvNeXt-Tiny, built by the same `ml.training.common.build_model` call, so an
arm with no modules reproduces the control's forward pass exactly (checked by `parity.py`).

ConvNeXt-Tiny is split so the stride-16 stage-3 map F3 (384 channels) is reachable:
`features[:6]` -> F3 (14x14 at 224 px, 24x24 at 384 px), `features[6:]` -> F4 (768 channels).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from research.v5.arms import ArmSpec

# ------------------------------------------------------------------ fixed values (runsheet §7)
TWOSTEP_WEIGHT = 0.5  # M2: each BCE head
CLUE_R = 4.0  # M3 / DRE-5: log-sum-exp sharpness
GEM_P_INIT = 3.0  # GeM: learnable
RANK_MARGIN = 0.20  # M4: hinge margin on the logit scale
RANK_WEIGHT = 0.5  # M4
MEMORY_K = {"akiec": 2, "bcc": 4, "bkl": 4, "df": 2, "mel": 8, "nv": 8, "vasc": 2}  # DRE-6
MEMORY_SCALE = 16.0  # DRE-6 s
MEMORY_TAU = 0.1  # DRE-6 tau            [declared here]
MEMORY_ORTHO_WEIGHT = 0.01  # DRE-6      [declared here]
F3_BLOCKS = 6  # features[:6] ends at ConvNeXt stage 3
F3_CHANNELS = 384
F4_CHANNELS = 768

MELANOCYTIC_CODES = ("mel", "nv")
ESCALATING_CODES = ("mel", "bcc", "akiec")


# ------------------------------------------------------------------ small pieces
class GeM(nn.Module):
    """Generalised-mean pooling, p initialised at 3 and learnable. Output is (B, C, 1, 1)."""

    def __init__(self, p: float = GEM_P_INIT, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Computed in fp32: x ** p under AMP overflows fp16 for p = 3 on large activations.
        x32 = x.float().clamp(min=self.eps)
        pooled = F.avg_pool2d(x32.pow(self.p), kernel_size=x.shape[-2:]).pow(1.0 / self.p)
        return pooled.to(x.dtype)


def lse_pool(logit_map: torch.Tensor, r: float = CLUE_R) -> torch.Tensor:
    """Normalised log-sum-exp over space: s = (1/r) ln((1/K) sum exp(r p_i)). Shape (B,).

    Always lies between mean(p) and max(p) because of the 1/K, and approaches max as r grows.
    `torch.logsumexp` keeps it stable; the map is taken in fp32.
    """
    flat = logit_map.float().flatten(1)
    k = flat.shape[1]
    return (torch.logsumexp(r * flat, dim=1) - math.log(k)) / r


def eccentricity(logit_map: torch.Tensor, radius: torch.Tensor | float = 0.5) -> torch.Tensor:
    """DRE-5 token ||mu_e - mu|| / r, mu_e the softmax(e)-weighted centroid of the clue map.

    Coordinates are normalised to [0, 1] across the map, so `mu` is the lesion centre in those
    units (0.5, 0.5 by default -- the lesion frame of DRE-2 replaces it when `geometry` is on) and
    `radius` is the lesion radius in the same units.
    """
    b, _, h, w = logit_map.shape
    weights = torch.softmax(logit_map.float().flatten(1), dim=1).view(b, 1, h, w)
    ys = (torch.arange(h, device=logit_map.device, dtype=torch.float32) + 0.5) / h
    xs = (torch.arange(w, device=logit_map.device, dtype=torch.float32) + 0.5) / w
    cy = (weights.sum(3).squeeze(1) * ys).sum(1)
    cx = (weights.sum(2).squeeze(1) * xs).sum(1)
    centre = 0.5
    dist = torch.sqrt((cy - centre) ** 2 + (cx - centre) ** 2 + 1e-12)
    radius_t = radius if isinstance(radius, torch.Tensor) else torch.full_like(dist, float(radius))
    return dist / radius_t


def noisy_or_map_logit(logit_map: torch.Tensor) -> torch.Tensor:
    """DRE-5 CLUE: logit of 1 - prod_i (1 - sigmoid(p_i)) over the map, (B,), computed stably.
    With S = sum_i softplus(p_i), prod(1 - sigmoid) = exp(-S) and the logit is log(expm1(S))."""
    s = F.softplus(logit_map.float().flatten(1)).sum(1).clamp(min=1e-6)
    return s + torch.log(-torch.expm1(-s))


#: DRE-7 concept -> input tokens (runsheet section 7; DRE section DRE-7). A source that the arm does
#: not produce is dropped; a concept left with no source is absent and held at its neutral value
#: by `LogicHead`. [impl] The grouping of tokens per concept is declared here.
LOGIC_CONCEPT_SOURCES: dict[str, tuple[str, ...]] = {
    "SYM": ("colour_asym_min", "colour_asym_max", "chrom_asym_min", "chrom_asym_max",
            "structure_asym_min", "structure_asym_max"),
    "ONE_COLOUR": ("palette_count",),
    "BLUE_GREY": tuple(f"palette_area_{k}" for k in range(6)),
    "CLUE": ("clue",),
    "SEGMENTAL": ("segmental_index",),
    "MULTICOMPONENT": ("pattern_entropy",),
    "BORDER_ABRUPT": ("abrupt_fraction", "abrupt_angular_variance"),
    "NETWORK": ("m5:pigment_network",),
    "GLOBULES": ("m5:globules",),
    "STREAKS": ("m5:streaks",),
    "NEG_NETWORK": ("m5:negative_network",),
    "MILIA": ("m5:milia_like_cyst",),
    "NETWORK_ATYPICAL": ("network_coverage", "network_regularity",
                         "network_orientation_coherence", "network_periphery_minus_centre"),
    "GLOBULES_IRREGULAR": ("dot_clark_evans", "dot_fraction_peripheral", "dot_sectors_occupied",
                           "dot_size_cv"),
    "VEIL": ("veil_fraction", "veil_eccentricity"),
    "VESSEL_POLYMORPHOUS": ("vessel_coverage", "tubular_blob_ratio", "calibre_entropy",
                            "vessel_polymorphism"),
}


def logic_inputs(spec: ArmSpec, token_names: list[str]) -> dict[str, list[str]]:
    """The concepts this arm can supply, each with the sources it actually produces."""
    available = set(token_names)
    if spec.has("clues"):
        available.add("clue")
    if spec.has("m5"):
        from research.v5.m5 import ATTRIBUTES

        available |= {f"m5:{a}" for a in ATTRIBUTES}
    out = {}
    for concept, sources in LOGIC_CONCEPT_SOURCES.items():
        present = [s for s in sources if s in available]
        if present:
            out[concept] = present
    return out


class MemoryHead(nn.Module):
    """DRE-6 multi-prototype class head. Replaces the linear 7-class head.

    logit_c = s * tau * logsumexp_k( cos(f, w_ck) / tau ), f and w_ck L2-normalised. Class order
    follows `class_codes` (the frozen class mapping), with K_c prototypes per class.
    """

    def __init__(self, in_features: int, class_codes: tuple[str, ...]) -> None:
        super().__init__()
        self.class_codes = tuple(class_codes)
        self.k = [MEMORY_K[c] for c in self.class_codes]
        total = sum(self.k)
        self.prototypes = nn.Parameter(torch.randn(total, in_features) * 0.02)
        bounds = [0]
        for k in self.k:
            bounds.append(bounds[-1] + k)
        self.bounds = bounds

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits (B, C), similarities (B, total_prototypes))."""
        f = F.normalize(z.float(), dim=1)
        w = F.normalize(self.prototypes.float(), dim=1)
        sim = f @ w.t()
        logits = []
        for c in range(len(self.k)):
            block = sim[:, self.bounds[c]:self.bounds[c + 1]]
            logits.append(MEMORY_SCALE * MEMORY_TAU * torch.logsumexp(block / MEMORY_TAU, dim=1))
        return torch.stack(logits, dim=1), sim

    def orthogonality(self) -> torch.Tensor:
        """Mean squared off-diagonal cosine within each class's prototype set."""
        w = F.normalize(self.prototypes.float(), dim=1)
        penalties = []
        for c in range(len(self.k)):
            block = w[self.bounds[c]:self.bounds[c + 1]]
            if block.shape[0] < 2:
                continue
            gram = block @ block.t()
            off = gram - torch.eye(block.shape[0], device=gram.device)
            penalties.append((off ** 2).sum() / (block.shape[0] * (block.shape[0] - 1)))
        return torch.stack(penalties).mean() if penalties else w.new_zeros(())


# ------------------------------------------------------------------ the arm network
class ArmNet(nn.Module):
    """The control ConvNeXt-Tiny plus whichever heads the arm's spec declares.

    `forward` returns a dict: `logits` (B, 7) always; `s_esc` (B,) when the arm has an escalation
    head; `mel_logit` when it has the melanocytic head; `clue_map` (B, 1, H', W') for clues;
    `z` (the pooled, layer-normed feature) always; `memory_sim` for the memory head.
    """

    def __init__(self, spec: ArmSpec, base: nn.Module, class_codes: tuple[str, ...],
                 in_channels: int = 3, artefacts: dict | None = None) -> None:
        super().__init__()
        from research.v5 import front as front_mod

        self.spec = spec
        self.class_codes = tuple(class_codes)
        self.base = base  # torchvision ConvNeXt: .features, .avgpool (.classifier moved to .head)
        # The classifier is held once, as `head`, so no parameter appears under two state_dict
        # keys. Its layout is [LayerNorm2d, Flatten, Dropout, Linear] (ml.training.common).
        self.head = base.classifier
        base.classifier = nn.Identity()
        self.front: nn.Module | None = None
        self.geometry_head: nn.Module | None = None
        n_tokens = 0
        if front_mod.needs_front(spec):
            if artefacts is None:
                raise ValueError(f"arm {spec.name!r} needs the fold's chromophore artefacts "
                                 f"(results/v5/chromophore/fold<k>.json)")
            self.front = front_mod.ChromophoreFront(spec, artefacts)
            in_channels = 3 + front_mod.extra_channels(spec)
            n_tokens = self.front.n_tokens
            if spec.has("geometry"):
                self.geometry_head = front_mod.GeometryHead()
                n_tokens += len(front_mod.LEARNED_GEOMETRY_TOKENS)
        self.use_d4 = bool(artefacts.get("force_d4_fallback", False)) if artefacts else False
        if in_channels != 3:
            self._widen_stem(in_channels)
        if n_tokens:
            self._widen_classifier(n_tokens)
        self.n_tokens = n_tokens
        self.pool: nn.Module = GeM() if spec.has("gem") else base.avgpool

        if spec.has("twostep"):
            self.mel_head = nn.Linear(F4_CHANNELS, 1)
            self.esc_head = nn.Linear(F4_CHANNELS, 1)
        if spec.has("clues"):
            self.clue_conv = nn.Conv2d(F3_CHANNELS, 1, kernel_size=1)
        if spec.has("memory"):
            self.memory = MemoryHead(F4_CHANNELS, self.class_codes)
        if spec.has("m5"):
            from research.v5.m5 import M5Head

            self.m5_head = M5Head(F3_CHANNELS)
        # Token order of out["tokens"]: the front's tokens, then the learned geometry tokens.
        self.token_names: list[str] = (
            front_mod.token_names(spec)
            + (list(front_mod.LEARNED_GEOMETRY_TOKENS) if spec.has("geometry") else [])
        ) if self.front is not None else []
        self.logic: nn.Module | None = None
        if spec.has("logic"):
            if not ({"twostep", "clues"} & set(spec.modules)):
                raise ValueError("logic (DRE-7) is a residual on the escalation logit; the arm "
                                 "needs twostep or clues")
            from research.v5.logic import LogicHead

            self.logic_inputs = logic_inputs(spec, self.token_names)
            self.logic = LogicHead({c: len(v) for c, v in self.logic_inputs.items()})

    def _widen_stem(self, in_channels: int) -> None:
        """Stem conv 3 -> in_channels; the new input slices are zero-initialised.

        With zero weights on the extra channels the widened network computes exactly the original
        function at step 0 (unit-tested), so an arm starts as the control and only earns its extra
        input by training.
        """
        stem = self.base.features[0][0]
        new = nn.Conv2d(in_channels, stem.out_channels, stem.kernel_size, stem.stride,
                        stem.padding, bias=stem.bias is not None)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :3] = stem.weight
            if stem.bias is not None:
                new.bias.copy_(stem.bias)
        self.base.features[0][0] = new

    def _widen_classifier(self, n_tokens: int) -> None:
        """Linear(768 -> C) becomes Linear(768 + tokens -> C) with the token columns zero-initialised
        [impl], so the widened head computes the original logits at step 0."""
        old = self.head[3]
        new = nn.Linear(old.in_features + n_tokens, old.out_features)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :old.in_features] = old.weight
            new.bias.copy_(old.bias)
        self.head[3] = new

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        extra: dict = {}
        if self.front is not None:
            extra = self.front(x)
            if "channels" in extra:
                x = torch.cat([x, extra["channels"]], dim=1)
        f3 = self.base.features[:F3_BLOCKS](x)
        f4 = self.base.features[F3_BLOCKS:](f3)
        pooled = self.pool(f4)  # (B, 768, 1, 1)
        z = self.head[1](self.head[0](pooled))  # Flatten(LayerNorm2d(pooled)) -> (B, 768)
        out: dict[str, torch.Tensor] = {"z": z}
        head_in = z
        if self.n_tokens:
            tokens = [extra["tokens"]] if "tokens" in extra else []
            if self.geometry_head is not None:
                tokens.append(self.geometry_head(f3, extra["geometry"], tuple(x.shape[-2:]),
                                                 use_d4=self.use_d4))
            out["tokens"] = torch.cat(tokens, 1).to(z.dtype)
            head_in = torch.cat([z, out["tokens"]], dim=1)
        if self.spec.has("memory"):
            out["logits"], out["memory_sim"] = self.memory(z)
        else:
            out["logits"] = self.head[3](self.head[2](head_in))  # Linear(Dropout(.))
        if self.spec.has("twostep"):
            out["mel_logit"] = self.mel_head(z).squeeze(1)
            out["esc_logit_global"] = self.esc_head(z).squeeze(1)
            out["s_esc"] = out["esc_logit_global"]
        if self.spec.has("clues"):
            clue_map = self.clue_conv(f3)
            out["clue_map"] = clue_map
            # M3 replaces M2's globally pooled escalation logit with the pooled clue map.
            out["s_esc"] = lse_pool(clue_map)
            out["eccentricity"] = eccentricity(clue_map)
        if self.spec.has("m5"):
            out["m5_logits"] = self.m5_head(f3)
        if self.logic is not None:
            delta, rules = self.logic(self._logic_tokens(out), x.shape[0], x.device)
            # Kept so the falsifier "removing the logic head removes the gain" can be read.
            out["s_esc_nologic"] = out["s_esc"]
            out["s_esc"] = out["s_esc"] + delta.to(out["s_esc"].dtype)
            out["logic_rules"] = rules
        return out

    def _logic_tokens(self, out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Each DRE-7 concept's input tokens, (B, d). Sources: a named front/geometry token,
        "clue" (noisy-OR logit of the clue map) or "m5:<attribute>" (spatial max of that logit)."""
        from research.v5.m5 import ATTRIBUTES

        cols = {n: i for i, n in enumerate(self.token_names)}
        tokens: dict[str, torch.Tensor] = {}
        for concept, sources in self.logic_inputs.items():
            parts = []
            for src in sources:
                if src == "clue":
                    parts.append(noisy_or_map_logit(out["clue_map"]).unsqueeze(1))
                elif src.startswith("m5:"):
                    k = ATTRIBUTES.index(src[3:])
                    parts.append(out["m5_logits"][:, k].float().flatten(1).amax(1, keepdim=True))
                else:
                    parts.append(out["tokens"][:, cols[src]:cols[src] + 1].float())
            tokens[concept] = torch.cat(parts, 1)
        return tokens


# ------------------------------------------------------------------ losses
@dataclass
class LossParts:
    total: torch.Tensor
    parts: dict[str, float]


class ClassIndex:
    """Index tensors for the melanocytic / escalating / mel / benign groupings."""

    def __init__(self, class_codes: tuple[str, ...], device: torch.device) -> None:
        codes = list(class_codes)
        self.mel = codes.index("mel")
        self.melanocytic = torch.tensor([codes.index(c) for c in MELANOCYTIC_CODES], device=device)
        self.escalating = torch.tensor([codes.index(c) for c in ESCALATING_CODES], device=device)

    def is_in(self, labels: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
        return (labels.unsqueeze(1) == group.unsqueeze(0)).any(1)


def pairwise_hinge(s_esc: torch.Tensor, is_mel: torch.Tensor, is_benign: torch.Tensor,
                   margin: float = RANK_MARGIN) -> tuple[torch.Tensor, int]:
    """M4 in-batch ranking: mean over all (mel_i, confirmed-benign_j) pairs of
    max(0, m - (s_i - s_j)). Returns (loss, |P|). A micro-batch with no pair returns exactly 0
    (a real zero that still carries the graph, so the backward pass never sees a NaN)."""
    mel = s_esc[is_mel]
    ben = s_esc[is_benign]
    n_pairs = int(mel.numel() * ben.numel())
    if n_pairs == 0:
        return s_esc.sum() * 0.0, 0
    diff = mel.unsqueeze(1) - ben.unsqueeze(0)
    return F.relu(margin - diff).mean(), n_pairs


def compute_loss(spec: ArmSpec, out: dict[str, torch.Tensor], labels: torch.Tensor,
                 targets: torch.Tensor, criterion: nn.Module, index: ClassIndex,
                 esc_weight: torch.Tensor | None = None,
                 confirmed_benign: torch.Tensor | None = None,
                 memory: MemoryHead | None = None,
                 geometry_head: nn.Module | None = None,
                 m5_masks: torch.Tensor | None = None,
                 m5_has: torch.Tensor | None = None) -> LossParts:
    """The 7-class CE (unchanged from the control) plus each declared auxiliary term."""
    ce = criterion(out["logits"], targets)
    total = ce
    parts = {"ce": float(ce.detach())}

    if spec.has("twostep"):
        w = esc_weight if esc_weight is not None else torch.ones_like(labels, dtype=torch.float32)
        mel_target = index.is_in(labels, index.melanocytic).float()
        esc_target = index.is_in(labels, index.escalating).float()
        bce_mel = F.binary_cross_entropy_with_logits(out["mel_logit"].float(), mel_target)
        # DRE-8: the global logit keeps its own BCE; the zoom logit gets its own term below.
        s_global = out.get("s_esc_global", out["s_esc"])
        per_sample = F.binary_cross_entropy_with_logits(s_global.float(), esc_target,
                                                        reduction="none")
        # Weight-0 rows (M4) drop out of the numerator but not the denominator's batch size.
        bce_esc = (per_sample * w).sum() / w.new_tensor(float(max(labels.numel(), 1)))
        total = total + TWOSTEP_WEIGHT * bce_mel + TWOSTEP_WEIGHT * bce_esc
        parts["bce_mel"] = float(bce_mel.detach())
        parts["bce_esc"] = float(bce_esc.detach())
        if spec.has("zoom") and "s_esc_zoom" in out:
            from research.v5.zoom import zoom_loss

            bce_zoom = zoom_loss(out, esc_target, w)  # already x 0.5
            total = total + bce_zoom
            parts["bce_esc_zoom"] = float(bce_zoom.detach())

    if spec.has("m5") and m5_masks is not None and m5_has is not None:
        from research.v5.m5 import downsample_masks, m5_loss

        logits = out["m5_logits"]
        target = downsample_masks(m5_masks, tuple(logits.shape[-2:]))
        loss_m5 = m5_loss(logits, target, m5_has)  # already x M5_WEIGHT
        total = total + loss_m5
        parts["m5"] = float(loss_m5.detach())
        parts["m5_labelled"] = float(m5_has.sum().detach())

    if spec.has("m4"):
        benign = confirmed_benign if confirmed_benign is not None else torch.zeros_like(labels).bool()
        is_mel = labels == index.mel
        rank, n_pairs = pairwise_hinge(out["s_esc"].float(), is_mel, benign)
        total = total + RANK_WEIGHT * rank
        parts["rank"] = float(rank.detach())
        parts["rank_pairs"] = float(n_pairs)

    if spec.has("memory") and memory is not None:
        ortho = memory.orthogonality()
        total = total + MEMORY_ORTHO_WEIGHT * ortho
        parts["memory_ortho"] = float(ortho.detach())

    if spec.has("geometry") and geometry_head is not None:
        from research.v5.front import PATTERN_ORTHO_WEIGHT

        ortho = geometry_head.orthogonality()
        total = total + PATTERN_ORTHO_WEIGHT * ortho
        parts["pattern_ortho"] = float(ortho.detach())

    return LossParts(total=total, parts=parts)
