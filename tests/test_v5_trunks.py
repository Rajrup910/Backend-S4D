"""The timm -> torchvision trunk remap must reproduce timm's forward pass exactly (audit AU17).

Random timm weights (`pretrained=False`), so nothing is downloaded. If the mapping dropped or
misplaced one tensor, the features would differ by far more than float noise.
"""

from __future__ import annotations

import pytest
import torch
from torchvision import models

from research.v5.trunks import TRUNKS, load_trunk_weights, remap_timm_to_torchvision, timm_tag

timm = pytest.importorskip("timm")


@pytest.mark.parametrize("trunk", ["in22k", "dinov3"])
def test_remapped_torchvision_matches_timm(trunk: str) -> None:
    torch.manual_seed(0)
    tag = timm_tag(trunk, 224)
    source = timm.create_model(tag, pretrained=False, num_classes=0).eval()
    target = models.convnext_tiny(weights=None).eval()
    target.load_state_dict({**target.state_dict(),
                            **remap_timm_to_torchvision(source.state_dict())})
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        expected = source.forward_head(source.forward_features(x), pre_logits=True)
        got = target.classifier[1](target.classifier[0](target.avgpool(target.features(x))))
        f3_timm = source.stages[:3](source.stem(x))
        f3_tv = target.features[:6](x)
    assert torch.allclose(f3_tv, f3_timm, atol=1e-4), (f3_tv - f3_timm).abs().max()
    assert torch.allclose(got, expected, atol=1e-4), (got - expected).abs().max()


def test_load_trunk_covers_every_trunk_tensor() -> None:
    model = models.convnext_tiny(weights=None)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    load_trunk_weights(model, "dinov3", 224, pretrained=False)
    after = model.state_dict()
    changed = [k for k in before if k.startswith("features.") and not torch.equal(before[k], after[k])]
    # Every trunk tensor is overwritten; LayerNorm init (ones/zeros) may coincide, so count convs.
    convs = [k for k in before if k.startswith("features.") and before[k].dim() == 4]
    assert all(k in changed for k in convs)


def test_in1k_is_a_no_op_and_unknown_trunk_raises() -> None:
    model = models.convnext_tiny(weights=None)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    load_trunk_weights(model, "in1k", 224)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert TRUNKS["in1k"] is None
    with pytest.raises(ValueError):
        timm_tag("swin", 224)


def test_unknown_timm_key_raises() -> None:
    with pytest.raises(KeyError):
        remap_timm_to_torchvision({"stages.0.blocks.0.unexpected.weight": torch.zeros(1)})
