"""LR audit: relative weight change from pretraining, ||w_ft - w_pre|| / ||w_pre||, per depth bucket, on the fold-1 _last
checkpoints of the four V6-4 members (weight tensors with 2+ dims; the new head is skipped). CPU only; reads no image,
prediction or label.

    python scripts/lr_audit_weight_movement.py > results/v6/screens/lr_audit_weight_movement.txt
"""
import os, re, sys
from collections import defaultdict
from pathlib import Path
os.environ["HF_HUB_OFFLINE"] = "1"
REPO = Path(__file__).resolve().parents[1]; os.chdir(REPO); sys.path.insert(0, str(REPO))
import torch
torch.set_num_threads(4)
from ml.paths import load_class_mapping
from research.v5.arms import get_arm
from research.v5.train_v5 import build_arm
from research.v6 import members
codes = load_class_mapping().codes

def bucket(name):
    m = re.match(r"(?:.*\.)?blocks\.(\d+)\.", name)
    if m: return f"block{int(m.group(1)):02d}"
    m = re.search(r"features\.(\d+)\.", name) or re.search(r"stages\.(\d+)\.", name)
    if m: return f"feat{int(m.group(1))}"
    if any(t in name for t in ("patch_embed", "stem", "cls_token", "pos_embed", "register")): return "embed"
    if any(t in name for t in ("head", "classifier", "fc.")): return "head"
    return "other:" + name.split(".")[0]

def movement(pre, ft):
    num, den = defaultdict(float), defaultdict(float)
    for k, w0 in pre.items():
        if k not in ft or w0.dtype not in (torch.float32, torch.float16) or w0.dim() < 2: continue
        b = bucket(k)
        if b == "head": continue
        num[b] += (ft[k].float() - w0.float()).pow(2).sum().item(); den[b] += w0.float().pow(2).sum().item()
    return {b: (num[b] / den[b]) ** 0.5 for b in sorted(num)}

runs = [("anchor_h7", "anchor-v6_anchor_h7_f1_s42_in22k_v64_last.pt", 224, None),
        ("anchor384_h7", "anchor-v6_anchor384_h7_f1_s42_in22k_v64_last.pt", 384, None),
        ("dinov2_b_h7", "dinov2_b-v6_dinov2_b_h7_f1_s42_v64_last.pt", 224, "dinov2_b"),
        ("panderm_b_h7", "panderm_b-v6_panderm_b_h7_f1_s42_v64_last.pt", 224, "panderm_b")]
for label, ck, size, member in runs:
    if member is None:
        model = build_arm(get_arm("youngdata"), codes, None, "in22k", size)
    else:
        model = members.build_member(members.ROSTER[member], num_classes=7)
    pre = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ft = torch.load(REPO / "ml/checkpoints" / ck, map_location="cpu", weights_only=False)["state_dict"]
    common = len(set(pre) & set(ft)); print(f"\n{label}: {common}/{len(pre)} keys shared")
    mv = movement(pre, ft)
    print("  " + "  ".join(f"{b} {100*v:.2f}%" for b, v in mv.items()))
    del model, pre, ft
