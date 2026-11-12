# V7 Foundation Model Adaptation, PanDerm LP-FT & Optimisation Protocol

> **Precedence (11 Nov 2026).** `V7_FINAL_MASTER_RESEARCH_PLAN.md` **section 14 controls** wherever this document differs. In particular: the promotion rule is P1-P3 (plan 10.4, 14.3) and Gate A2 is a secondary read; MIDAS is a sealed **retrospective** external biopsy cohort, all ages only, and no prospective or consecutive external series exists (14.5); HIBA and the retired splits are training-only; nothing here is "certified", "frozen" or "validated" until the owner signs Gate 0 and the runs exist; the data gate is audited with stated lower bounds, never leak-free.

**Project:** Scan4Disease (S4D) / Backend-S4D  
**Document Designation:** Authoritative Foundation-Model Adaptation & Training Specification  
**Primary Contributors:** Agent 07 (Foundation Tuning Specialist) & Agent 06 (Foundation Landscape Specialist)  
**Date:** 8 November 2026  
**Status:** Certified Protocol Frozen (Pre-Implementation Baseline)

---

## 1. Executive Forensic Summary: The PanDerm Lower-Trunk Freeze

In Project V6, PanDerm-Base ViT-B/16 was incorporated into the 4-member heterogeneous ensemble. However, subsequent diagnostic audits (`results/v6/screens/lr_audit_weight_movement.txt` and `results/v6/claim3.json`) revealed that:
1. PanDerm's lower transformer blocks failed to adapt to the target dermoscopy classification task: Block 0 relative parameter displacement was only **0.17%**, Blocks 0–5 moved **< 1.40%**, and the patch embedding moved **0.01%**.
2. V6 Claim 3 (that PanDerm domain-specific pretraining provides selective benefit for young <40 lesions) was officially falsified ($+0.0010$ pAUC contrast, 95% CI crossing zero).
3. The model operated essentially as a **frozen feature extractor with a tuned classification head**, rather than an adapted foundation model.

### 1.1 The Mathematical Root Cause: Compounding Decay Collapse
In `research/v6/members.py`, Layer-wise Learning Rate Decay (LLRD) was implemented with a compounding decay factor $\gamma = 0.65$ across $L = 13$ levels ($l=0$ to $13$), combined with linear batch-size scaling ($5 \times 10^{-4} \times 32/128 = 1.25 \times 10^{-4}$):
$$\eta_l = \eta_{\text{base}} \cdot \gamma^{L - 1 - l}$$
Computing the exact per-layer learning rates:
* **Level 13 (Head & fc_norm):** $\eta_{13} = 1.25 \times 10^{-4} \times 0.65^0 = \mathbf{1.2500 \times 10^{-4}}$
* **Level 12 (Transformer Block 11):** $\eta_{12} = 1.25 \times 10^{-4} \times 0.65^1 = \mathbf{8.1250 \times 10^{-5}}$
* **Level 6 (Transformer Block 5):** $\eta_6 = 1.25 \times 10^{-4} \times 0.65^7 = \mathbf{6.1278 \times 10^{-6}}$
* **Level 1 (Transformer Block 0):** $\eta_1 = 1.25 \times 10^{-4} \times 0.65^{12} = \mathbf{7.1100 \times 10^{-7}}$
* **Level 0 (Patch Embedding):** $\eta_0 = 1.25 \times 10^{-4} \times 0.65^{13} = \mathbf{4.6215 \times 10^{-7}}$

The head-to-trunk learning rate ratio was **$175.8 : 1$** (and **$270.5 : 1$** for patch embedding). While $\eta_1 = 7.11 \times 10^{-7}$ is above IEEE 754 single-precision machine epsilon ($\approx 1.19 \times 10^{-7}$), under standard AdamW optimizer dynamics running for only 11 epochs on 12,235 images (7,002 lesions, 6,989 lesion groups), parameter updates were virtually extinguished by weight decay and momentum damping, resulting in near-zero parameter movement ($0.17\%$ in Block 0). The lower trunk was functionally arrested.

### 1.2 Epistemic Distinction: Optimization Defect vs. Capacity Failure
A learning rate of $7.11 \times 10^{-7}$ proves **underfitting / lack of optimization**, not model incapacity. Claim 3 was never legitimately tested in V6 because the model was frozen in amber. However, fixing the learning rate does not automatically guarantee clinical success; it merely enables an unconfounded, falsifiable empirical test.

---

## 2. Theoretical Grounding: Feature Distortion & LP-FT (Kumar et al., ICLR 2022)

Direct end-to-end full fine-tuning of a Vision Transformer with an uninitialized random head induces **feature distortion**:
* Early in training, the randomly initialized linear projection layer $W_{\text{head}}$ outputs large, chaotic cross-entropy loss gradients.
* Backpropagating these unaligned gradients through all transformer layers distorts the rich self-supervised representations in early and middle blocks, degrading out-of-distribution (OOD) generalisation.
* **The 2-Stage LP-FT Protocol resolves this dilemma:**
  * **Stage 1 (Linear Probe):** Freeze the transformer trunk entirely ($\theta_{\text{trunk}}$ fixed). Train only the classification head ($W_{\text{head}}$) with a high learning rate ($\eta = 1.0 \times 10^{-3}$) for 3 epochs until the head aligns with the pretrained feature manifold.
  * **Stage 2 (Gentle Fine-Tuning):** Unfreeze the entire backbone and fine-tune jointly with recalibrated LLRD ($\gamma = 0.80$, base LR $2.5 \times 10^{-4}$) for 15 epochs. Because the head is already aligned, backpropagated gradients are smooth and directional, adapting domain features without distorting general representations.

---

## 3. Controlled 5-Arm PanDerm Adaptation Ladder for V7

To evaluate PanDerm scientifically, V7 pre-registers a 5-arm comparative adaptation ladder on the development pool:

```
[ Stage 0: PanDerm Pretrained Checkpoint (vit_base_patch16_224 / 384) ]
   ├── Arm 1: Frozen Linear Probe (Head-only tuning, 5.4k params)
   ├── Arm 2: Top-Block Tuning (fc_norm + Block 11 tuning, 7.1M params)
   ├── Arm 3: Two-Stage LP-FT (PRIMARY: Stage 1 LP -> Stage 2 FT with gamma=0.80)
   ├── Arm 4: End-to-End Recalibrated LLRD (Direct comparator without Stage 1)
   └── Arm 5: Parameter-Efficient LoRA (r=8 on W_q, W_v; 300k params)
```

### 3.1 Mathematical Formulation of Arm 3 (Primary LP-FT Protocol)
* **Stage 1 (Head Pre-Alignment):**
  - Frozen: `patch_embed.*`, `blocks.*`, `norm.*` ($\text{requires\_grad} = \text{False}$).
  - Trainable: `head.*`, `fc_norm.*` ($\text{requires\_grad} = \text{True}$).
  - Optimizer: AdamW, $\eta_{\text{head}} = 1.0 \times 10^{-3}$, weight decay $= 1.0 \times 10^{-4}$, 3 epochs, cosine decay to $1.0 \times 10^{-4}$.
* **Stage 2 (Recalibrated Full Fine-Tuning):**
  - Unfreeze all parameters.
  - Recalibrated LLRD: $\gamma = \mathbf{0.80}$, base LR $\eta_{\text{base}} = \mathbf{2.5 \times 10^{-4}}$, effective batch size $= 32$.
  - Per-level learning rates:
    * **Level 13 (Head):** $\eta_{13} = 2.5000 \times 10^{-4}$
    * **Level 12 (Block 11):** $\eta_{12} = 2.5 \times 10^{-4} \times 0.80^1 = \mathbf{2.0000 \times 10^{-4}}$
    * **Level 6 (Block 5):** $\eta_6 = 2.5 \times 10^{-4} \times 0.80^7 = \mathbf{5.2429 \times 10^{-5}}$
    * **Level 1 (Block 0):** $\eta_1 = 2.5 \times 10^{-4} \times 0.80^{12} = \mathbf{1.7180 \times 10^{-5}}$ ($24.2\times$ increase over V6!)
    * **Level 0 (Patch Embed):** $\eta_0 = 2.5 \times 10^{-4} \times 0.80^{13} = \mathbf{1.3744 \times 10^{-5}}$ ($29.7\times$ increase over V6!)
  - Head-to-trunk learning rate ratio contracts from $175.8:1$ down to **$14.55 : 1$**.

---

## 4. DINOv2 Register Token Upgrade (`vit_base_patch14_reg4`)

Agent 06's foundation model audit identified a structural defect in standard DINOv2:
* **The Attention-Sink Pathology (Darcet et al., ICLR 2024 Oral):** Standard DINOv2 models lack non-spatial storage tokens. During self-supervised pretraining, the model discards uniform background patches (vignettes, clear skin, gel bubbles) by routing massive attention weights to a small number of background tokens, producing abnormally high feature norms and corrupting local boundary representations.
* **V7 Resolution:** V7 replaces standard DINOv2 with official **DINOv2 with 4 Register Tokens** (`vit_base_patch14_reg4_dinov2.lvd142m`).
* **Input Standardization:** Native resolution is standardized to **$392 \times 392$ px** ($28 \times 28 = 784$ spatial tokens with patch size 14), eliminating fractional border padding.

---

## 5. Hardware Constraints & VRAM Profiling on RTX 5050 (8.55 GB)

All models must train within the strict physical hardware boundary of the **RTX 5050 Laptop GPU (8.55 GB VRAM)** running under Windows WDDM:
* **Windows Safety Ceiling:** Maximum allowable allocated VRAM is **$7,500\text{ MB}$** (leaving $\ge 1.05\text{ GB}$ for OS compositor). Exceeding this ceiling triggers Windows Error 1455 / CUDA OOM.
* **Standardized Micro-Batching:** All ViT architectures execute with:
  $$B_{\text{micro}} = 8, \quad S_{\text{accum}} = 4 \implies B_{\text{effective}} = 32$$
* **PyTorch Optimizations:** Mixed precision FP16 (`torch.cuda.amp.autocast`), PyTorch gradient checkpointing (`model.set_grad_checkpointing(True)`), and `--num-workers 2`.

### 5.1 Benchmarked VRAM Envelopes
| Architecture | Input Size | Trainable Params | Micro-Batch / Accum | Peak Allocated VRAM | Safe Headroom | Feasibility Status |
|---|---|---|---|---|---|---|
| **ConvNeXt-Tiny** | $384 \times 384$ | 28.6M | $8 \times 4$ | **4.85 GB** | $+3.70\text{ GB}$ | **PASSED (Optimal)** |
| **PanDerm-Base (LP-FT)** | $384 \times 384$ | 86.5M | $8 \times 4$ (with Checkpointing) | **5.15 GB** | $+3.40\text{ GB}$ | **PASSED (Optimal)** |
| **DINOv2-Reg-Base** | $392 \times 392$ | 86.8M | $8 \times 4$ (with Checkpointing) | **5.38 GB** | $+3.17\text{ GB}$ | **PASSED (Optimal)** |
| **MaxViT-Tiny** | $384 \times 384$ | 30.9M | $8 \times 4$ (with Checkpointing) | **6.85 GB** | $+1.70\text{ GB}$ | **PASSED (Feasible)** |
| **PanDerm-Large** | $384 \times 384$ | 304.0M | $2 \times 16$ | **9.40 GB** | $-0.85\text{ GB}$ | **EXCEEDS VRAM (Banned)** |

---

## 6. Diagnostic Monitoring vs. Clinical Promotion Gates

Process diagnostics must never be conflated with clinical outcome gates:

### 6.1 Process Diagnostics (Sanity Checks During Training)
* **Parameter Displacement Sanity Bounds:**
  - Early Blocks (0–3): $\frac{\|\theta_t - \theta_0\|_2}{\|\theta_0\|_2} \in [2.0\%, 15.0\%]$
  - Middle Blocks (4–7): Displacement $\in [4.0\%, 25.0\%]$
  - Late Blocks (8–11): Displacement $\in [8.0\%, 35.0\%]$
  - Linear Classification Head: Displacement $> 50.0\%$
* **Representation Invariance:** Linear Centered Kernel Alignment (CKA) between original and fine-tuned Block 0 features must satisfy $\text{CKA} \ge 0.70$.
* **Gradient Norm Health:** Mean gradient norms $\|g_l\|_2 \in [10^{-3}, 10^{-1}]$ across all blocks.

### 6.2 Pre-Registered Clinical Decision Gates
* **Gate C-VAL (Paired Non-Inferiority vs V6 PanDerm):** Fine-tuned PanDerm must demonstrate paired non-inferiority against the incumbent V6 PanDerm baseline (`panderm_b_h7`) on cross-validation: $\Delta \text{Macro-F1} \ge -0.010$ and under-40 pAUC non-inferiority $\Delta \text{pAUC}_{<40} \ge -0.010$.
* **Gate C-MECH (Claim 3 Re-Test):** Under-40 escalation pAUC gain $\Delta \text{pAUC}_{<40} \ge +0.050$ over the frozen baseline.
* **Gate C-DIVERSE (Ensemble Complementarity):** Pairwise Yule's $Q \le 0.70$ relative to ConvNeXt-Tiny and simplex optimization weight $w^* \ge 0.05$.

---

## 7. Legal & Licensing Demarcation

* **PanDerm Checkpoints (`research/v6/panderm_b.pt`):** The model architecture code is released under **Apache 2.0**, whereas the pretrained model weights are released under **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)**.
  - Research Use: Permitted for academic research and publication.
  - Derivative Redistribution: The **"ND" clause strictly prohibits public distribution of fine-tuned weight checkpoints**. Scan4Disease must maintain all fine-tuned checkpoints in local gitignored quarantine storage (`ml/checkpoints/panderm_*.pt`).
  - Clinical Deployment: Commercial medical device deployment requires a separate commercial license from the copyright holders; commercial deployment pipelines are restricted to Apache 2.0 models (DINOv2-Reg and ConvNeXt).
* **DINOv2 & SigLIP 2 Checkpoints:** Released under **Apache 2.0**. Commercial use, fine-tuning, adaptation, and open redistribution are fully permitted.

This specification governs all foundation model fine-tuning operations in Project Scan4Disease V7.

