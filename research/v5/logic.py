"""DRE-7 -- clinical-logic residual head (docs/v5_design/V5_DERM_REASONING_ENGINE.md; runsheet 7).

Written 30 Sep as a NEW file while the night queue ran; wired in after it ends:
  * `ArmNet` (composite only) builds `LogicHead` and, after computing the escalation logit `s_esc`,
    replaces it with `out["s_esc"] = s_esc + delta` where `delta, rules = logic(concepts)`;
  * `concepts` come from the tokens the composite's modules already produce (see `CONCEPT_SOURCES`);
    a concept whose module is absent is simply not passed and gets the neutral value below;
  * `composite_nologic` is the same network with `LogicHead` absent -- the runsheet's comparison.

Rules (fixed structure, product t-norm, soft-OR = 1 - prod(1 - x)):
    R_benign          = SYM * ONE_COLOUR * (1 - CLUE) * (1 - NETWORK_ATYPICAL)
    R_chaos_clue      = (1 - SYM) * OR(CLUE, SEGMENTAL, BLUE_GREY, STREAKS, NEG_NETWORK,
                                       NETWORK_ATYPICAL, GLOBULES_IRREGULAR, VEIL,
                                       VESSEL_POLYMORPHOUS, BORDER_ABRUPT)
    R_sk              = MILIA * (1 - NETWORK)
    R_nonmelanocytic  = (1 - NETWORK) * (1 - GLOBULES)
Output: s_esc + sum_j alpha_j * logit(R_j), alpha initialised at 0 so the head starts as the plain
model. Missing concepts ("held at the neutral value", runsheet 7): 0 inside an OR list and inside
a (1 - x) factor (so the factor is 1); 0.5 for the positive factors SYM and ONE_COLOUR, which
keeps R_benign and (1 - SYM) uninformative rather than forcing them to 0 or 1.
Sign check (pre-declared, both seeds): alpha(R_chaos_clue) > 0, alpha(R_benign) < 0, alpha(R_sk) < 0.
"""

from __future__ import annotations

import torch
import torch.nn as nn

RULES = ("R_benign", "R_chaos_clue", "R_sk", "R_nonmelanocytic")
OR_LIST = ("CLUE", "SEGMENTAL", "BLUE_GREY", "STREAKS", "NEG_NETWORK", "NETWORK_ATYPICAL",
           "GLOBULES_IRREGULAR", "VEIL", "VESSEL_POLYMORPHOUS", "BORDER_ABRUPT")
CONCEPTS = ("SYM", "ONE_COLOUR", "BLUE_GREY", "CLUE", "SEGMENTAL", "MULTICOMPONENT", "NETWORK",
            "GLOBULES", "STREAKS", "NEG_NETWORK", "MILIA", "NETWORK_ATYPICAL",
            "GLOBULES_IRREGULAR", "VEIL", "VESSEL_POLYMORPHOUS", "BORDER_ABRUPT")
#: Which module supplies each concept (documentation; the caller passes the tokens).
CONCEPT_SOURCES = {
    "SYM": "geometry (DRE-3)", "ONE_COLOUR": "palette (DRE-1)", "BLUE_GREY": "palette (DRE-1)",
    "CLUE": "clues (DRE-5, noisy-OR of the clue map)", "SEGMENTAL": "geometry (DRE-4)",
    "MULTICOMPONENT": "geometry (DRE-3 pattern entropy)", "BORDER_ABRUPT": "geometry (DRE-4)",
    "NETWORK": "m5", "GLOBULES": "m5", "STREAKS": "m5", "NEG_NETWORK": "m5", "MILIA": "m5",
    "NETWORK_ATYPICAL": "dsp (DRE-10)", "GLOBULES_IRREGULAR": "dsp", "VEIL": "dsp",
    "VESSEL_POLYMORPHOUS": "dsp"}
NEUTRAL_POSITIVE = 0.5  # SYM, ONE_COLOUR
EPS = 1e-6


def soft_or(values: list[torch.Tensor]) -> torch.Tensor:
    out = torch.ones_like(values[0])
    for v in values:
        out = out * (1.0 - v)
    return 1.0 - out


class LogicHead(nn.Module):
    """Concepts (each a sigmoid of a small linear map of its tokens) -> 4 rules -> residual."""

    def __init__(self, token_dims: dict[str, int]) -> None:
        super().__init__()
        self.token_dims = dict(token_dims)  # concept -> number of input tokens
        self.maps = nn.ModuleDict({c: nn.Linear(d, 1) for c, d in self.token_dims.items()})
        self.alpha = nn.Parameter(torch.zeros(len(RULES)))  # initialised at 0

    def concepts(self, tokens: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {c: torch.sigmoid(self.maps[c](tokens[c].float())).squeeze(1)
                for c in self.token_dims if c in tokens}

    def rules(self, tokens: dict[str, torch.Tensor], batch: int, device) -> torch.Tensor:
        c = self.concepts(tokens)

        def val(name: str, missing: float) -> torch.Tensor:
            return c[name] if name in c else torch.full((batch,), missing, device=device)

        sym, one = val("SYM", NEUTRAL_POSITIVE), val("ONE_COLOUR", NEUTRAL_POSITIVE)
        clue, atyp = val("CLUE", 0.0), val("NETWORK_ATYPICAL", 0.0)
        net, glob, milia = val("NETWORK", 0.0), val("GLOBULES", 0.0), val("MILIA", 0.0)
        r_benign = sym * one * (1 - clue) * (1 - atyp)
        r_chaos = (1 - sym) * soft_or([val(n, 0.0) for n in OR_LIST])
        r_sk = milia * (1 - net)
        r_nonmel = (1 - net) * (1 - glob)
        return torch.stack([r_benign, r_chaos, r_sk, r_nonmel], dim=1)

    def forward(self, tokens: dict[str, torch.Tensor], batch: int, device=None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (delta (B,), rules (B, 4)); delta is added to the trunk escalation logit."""
        device = device or self.alpha.device
        r = self.rules(tokens, batch, device).clamp(EPS, 1.0 - EPS)
        logit_r = torch.log(r) - torch.log1p(-r)
        return (logit_r * self.alpha.unsqueeze(0)).sum(1), r

    def sign_check(self) -> dict[str, object]:
        a = dict(zip(RULES, self.alpha.detach().float().tolist()))
        checks = {"alpha(R_chaos_clue) > 0": a["R_chaos_clue"] > 0,
                  "alpha(R_benign) < 0": a["R_benign"] < 0, "alpha(R_sk) < 0": a["R_sk"] < 0}
        return {"alpha": a, "checks": checks, "pass": all(checks.values())}
