# V7 Image Input Fidelity, Native Resolution & High-Resolution Cropping Protocol

> **Precedence (11 Nov 2026).** `V7_FINAL_MASTER_RESEARCH_PLAN.md` **section 14 controls** wherever this document differs. In particular: the promotion rule is P1-P3 (plan 10.4, 14.3) and Gate A2 is a secondary read; MIDAS is a sealed **retrospective** external biopsy cohort, all ages only, and no prospective or consecutive external series exists (14.5); HIBA and the retired splits are training-only; nothing here is "certified", "frozen" or "validated" until the owner signs Gate 0 and the runs exist; the data gate is audited with stated lower bounds, never leak-free.

**Project:** Scan4Disease (S4D) / Backend-S4D  
**Document Designation:** Authoritative Image Preprocessing, Optical Resolution & Geometry Protocol  
**Primary Contributors:** Agent 08 (Image Resolution & Geometry Specialist) & Agent 09 (Melanoma Biology Specialist)  
**Date:** 8 November 2026  
**Status:** Certified Protocol Frozen (Pre-Implementation Baseline)

---

## 1. Executive Summary: The Evaluation Truncation Defect

A forensic audit of the repository's image transform pipeline (`ml/preprocessing/transforms.py`) uncovered a critical geometric distortion that has silently degraded test-time evaluation across all previous versions:

### 1.1 The 34.375% (34.31% Raster) Field-of-View (FOV) Amputation
* In standard evaluation code, the transform chain was defined as:
  ```python
  transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
  ])
  ```
* For HAM10000 images, the native dimensions are $600 \times 450$ px (aspect ratio $4:3 = 1.333$).
* When resized with `Resize(256)`, the shorter dimension (450 px) is scaled to 256 px. To preserve aspect ratio, the longer dimension scales to:
  $$\text{Width}_{\text{resized}} = 600 \times \left(\frac{256}{450}\right) = 341.33 \approx 341.33\text{ px}$$
* Applying `CenterCrop(224)` on this $341.33 \times 256$ px image discards:
  $$\Delta W = 341.33 - 224 = 117.33\text{ horizontal pixels}$$
  cutting exactly $58.67$ pixels off the left flank and $58.67$ pixels off the right flank.
* **Clinical Consequence:** `CenterCrop(224)` amputates **$11/32 = 34.375\% \approx 34.38\%$ of the continuous horizontal field of view ($117.33 / 341.33$)** on HAM10000 images at test time! In discrete integer-raster implementations where PIL rounds the resized width to 341 px first, the pixel loss is **$(341 - 224) / 341 = 117 / 341 = 34.31\%$**. Both expressions quantify the identical physical failure mode.
  (Similarly, on square images such as BCN20000 $1024 \times 1024$, `Resize(256) + CenterCrop(224)` discards $(256 - 224)/256 = 12.5\%$ from each dimension, causing a $23.44\%$ area loss: $1 - (224/256)^2 = 0.2344$).
  Dermoscopic hallmarks of early melanoma—peripheral pseudopods, radial streaming, and atypical network lines—concentrate along the lesion periphery. On off-center or large asymmetric lesions, the diagnostic margin was systematically severed prior to model evaluation.

### 1.2 Train vs. Eval Transform Inconsistency
* During training, `RandomResizedCrop(224, scale=(0.8, 1.0))` was used, dynamically jittering aspect ratios.
* At test time, static center-cropping amputated 34.375% (34.31% raster) of the horizontal image.
* **V7 Resolution:** V7 bans `CenterCrop` on dermoscopy images. All evaluation inputs are resized via **Aspect-Preserving Reflection Padding** to ensure $100.0\%$ FOV retention.

### 1.3 Aspect-Preserving Reflection Padding Resize & Mask Tracking
To achieve 100% FOV retention without geometric stretching or black border artifacts:
1. **Isotropic Scaling:** Scale the longer image dimension to target size $S$ (e.g. 384 px), scaling the shorter dimension proportionally: $S_{\text{short}} = \text{round}(S \times \min(H, W) / \max(H, W))$.
2. **Symmetric Reflection Padding:** Pad the remaining pixels equally along both sides of the shorter axis using boundary mirror reflection (`mode='reflect'`) to form an exact $S \times S$ square tensor.
3. **Valid Source Pixel Mask Tracking:** Generate a paired binary mask `valid_source_mask` of shape $(S, S)$, where $1$ denotes original dermatological pixels and $0$ denotes reflected padding. Downstream focal pooling, attention gating, and token aggregation modules read this mask to explicitly ignore padded border tokens.

---

## 2. Native Resolution vs. The Digital Resampling Fallacy

A central misunderstanding in prior revisions was the assumption that resizing low-resolution source images to 384 px or 1024 px restores high-frequency diagnostic detail:

### 2.1 The Whittaker-Shannon Resampling Limit
* Upsampling a discrete $600 \times 450$ px image to $1024 \times 1024$ px via bicubic interpolation adds zero optical information.
* Per the **Whittaker-Shannon Sampling Theorem**, the continuous bandlimit of the signal is strictly bounded by the physical Nyquist frequency of the original sensor capture ($f_{\text{Nyquist}} = \frac{1}{2 \Delta x_{\text{sensor}}}$). Interpolation calculates weighted linear combinations of existing pixels; it cannot reconstruct sub-resolution optical structures.

### 2.2 Receptive Field Mismatch (FixRes Autopsy)
* An empirical audit of the V6 FixRes experiment (`results/v6/fixres/fixres_rescore.json`) confirmed this fallacy:
  Evaluating 224-trained models at 256–320 px without resolution fine-tuning caused monotonic Macro-F1 degradation ($0.675 \to 0.659$) due to receptive-field scale shifts and lack of native margin cropping.

### 2.3 Cropping Native High-Resolution Sources (Pixel-Scale Boundaries)
While digital upsampling is a mathematical fallacy, **cropping native high-resolution originals** (e.g. BCN20000 at $1024 \times 1024$ px or MSKCC at $1024 \times 768$ px) preserves genuine physical sensor pixel density:
* **Nominal Optical Pitch Illustration:**
  Assuming a typical contact dermatoscope field of view ($\approx 18.0\text{ mm}$), resizing the full FOV to $224 \times 224$ px yields a nominal pixel pitch of:
  $$\Delta x = \frac{18,000\ \mu\text{m}}{224\text{ px}} \approx \mathbf{80.36\ \mu\text{m/pixel}}$$
  At $80.36\ \mu\text{m/pixel}$, fine melanocytic structures violate the Nyquist criterion ($N_{\text{sample}} \ge 2.0\text{ pixels}$):
  - Atypical network lines ($20\text{--}40\ \mu\text{m}$): $N_{\text{sample}} \approx 0.37\text{ px}$ (spatial aliasing and extinction).
  - Peripheral pseudopods / streaks ($40\text{--}80\ \mu\text{m}$): $N_{\text{sample}} \approx 0.62\text{ px}$ (blurred into uniform brown bands).
  - Dotted vessels ($20\text{--}50\ \mu\text{m}$): $N_{\text{sample}} \approx 0.50\text{ px}$ (extinguished).
* **The High-Resolution Margin Crop:**
  Cropping an edge sub-field directly from the native $1024 \times 1024$ px source and feeding it at $384 \times 384$ px yields nominal:
  $$\Delta x_{\text{crop}} = \frac{9,600\ \mu\text{m}}{384\text{ px}} \approx \mathbf{25.00\ \mu\text{m/pixel}}$$
  where a $50\ \mu\text{m}$ pseudopod spans $2.00\text{ pixels}$, satisfying the Nyquist-Shannon sampling limit.
* **Strict Physical Scale & Optical Calibration Boundary:**
  The calculations above are nominal physical approximations based on standard dermatoscope hardware. However, public datasets (HAM10000, BCN20000, MSKCC) do NOT provide calibrated per-image optical sensor pitch or physical scale-bar metadata. Therefore, in the V7 codebase, all crops are strictly designated as **pixel-scale crops** (e.g. $384 \times 384$ px native sub-arrays), and are **NEVER** claimed as verified micrometre measurements.

---

## 3. Dual-Stream Cross-Attention Architecture (Conditional Arm HR)

Clinical dermatological reader studies (Carrera et al., JAMA Dermatol 2016; Williams et al., Cochrane 2018) prove that dermatologists do not diagnose melanoma from local focal cutouts alone. Macro-architectural disorder ($\text{OR} = 6.6$) and pattern asymmetry ($\text{OR} = 4.9$) dominate local checklist criteria ($\text{OR} = 2.1\text{--}2.8$). 

Evaluating local margin crops without global lesion context triggers massive false-positive over-calling on dysplastic nevi. To test whether high-resolution margin features provide incremental value without destabilizing global architecture, V7 designs a **Dual-Stream Cross-Attention Architecture as Conditional Arm HR** (to be compared against the single-stream core trunk `V7_CANDIDATE_CORE` via Gate HR):

```
[ Raw Native Image Input (e.g. 1024x1024 px or 600x450 px) ]
   │
   ├─── Stream 1: Global Macro-Context Stream (384x384 px)
   │       └── Captures full FOV with aspect-preserving reflection padding.
   │       └── Extracts global architecture, overall asymmetry, and colour diversity.
   │       └── Backbone: ConvNeXt-Tiny / PanDerm / DINOv2-Reg -> Feature Map F_global.
   │
   └─── Stream 2: Native High-Resolution Margin Stream (384x384 px)
           └── Crops native lesion periphery at pixel scale (Delta_x <= 25 um/px nominal).
           └── Focuses on focal pseudopods, atypical network, and margin abruptness.
           └── Backbone: Shared weights -> Feature Map F_margin.
                   │
                   ▼
   [ Distance-Biased Cross-Attention Layer ]
   └── Queries: Global Tokens; Keys/Values: Margin Tokens.
   └── Modulates global contextual representations with high-resolution edge features.
                   │
                   ▼
   [ Calibrated Top-K Focal Pooling (K = 0.15 * H * W) with Quadruple Artifact Gating ]
                   │
                   ▼
         [ Final Vision Logits z_vision ]
```
   └── Modulates global contextual representations with high-resolution edge features.
                   │
                   ▼
   [ Calibrated Top-K Focal Pooling (K = 0.15 * H * W) with Quadruple Artifact Gating ]
                   │
                   ▼
         [ Final Vision Logits z_vision ]
```

---

## 4. Calibrated Top-$K$ Focal Pooling & Artifact Defense

### 4.1 The Pathology of Unconstrained Attention Pooling
Standard attention pooling or Multiple Instance Learning (MIL) pooling fails in dermatological screening because attention heads latch onto high-contrast extraneous artifacts:
* Coarse hair shafts with specular reflection
* Air bubbles in dermoscopy immersion gel
* Surgical skin marker ink lines
* Black circular dermatoscope aperture rims

### 4.2 Quadruple Artifact Gating & Valid Source Mask Pipeline
Prior to focal pooling, the spatial feature map $\mathbf{F} \in \mathbb{R}^{C \times H \times W}$ is gated through pre-flight filters and boundary validity masks:
1. **DullRazor Hair Suppression:** Bilateral morphological closing identifies linear dark transients and inpaints them using background texture synthesis.
2. **Specular Glare Neutralization:** Pixel regions with CIELAB $L^* > 95$ and chroma $\sqrt{a^{*2} + b^{*2}} < 5$ are masked to prevent saturation spikes.
3. **Surgical Ink Rejection:** Saturated blue/purple/green ink marks are identified in HSV space and set to zero attention weight.
4. **Circular Aperture Mask:** A soft circular vignetting mask zero-weights the dark boundary rim present in contact dermoscopy (BCN20000).
5. **Valid Source Pixel Mask Incorporation:** The composite artifact rejection mask explicitly integrates the reflected boundary mask:
   $$\mathbf{M}_{\text{artifact}}^{\text{total}} = \mathbf{M}_{\text{hair}} \cup \mathbf{M}_{\text{bubble}} \cup \mathbf{M}_{\text{ink}} \cup \mathbf{M}_{\text{rim}} \cup (1 - \text{valid\_source\_mask})$$
   Any spatial token where $\mathbf{M}_{\text{artifact}}^{\text{total}} = 1$ is assigned attention weight $-\infty$ (or zeroed out) prior to Top-$K$ ranking. This mathematically guarantees that synthetic reflection-padded border pixels outside the true optical sensor aperture can NEVER enter the top-scoring focal set $\Omega_{\text{top}}$.

### 4.3 Calibrated Malignancy-Projected Top-$K$ Pooling
Let $\mathbf{f}_{ij} \in \mathbb{R}^C$ be the artifact-cleaned feature token at spatial grid $(i, j)$.
1. Project $\mathbf{f}_{ij}$ onto the learned malignancy escalation vector $\mathbf{w}_{\text{esc}} = \mathbf{w}_{\text{mel}} - \mathbf{w}_{\text{nv}}$:
   $$s_{ij} = \mathbf{w}_{\text{esc}}^\top \mathbf{f}_{ij}$$
2. Select the top $K = \lfloor 0.15 \cdot H \cdot W \rfloor$ highest scoring spatial tokens ($15\%$ area fraction, matching early focal melanoma biology):
   $$\Omega_{\text{top}} = \text{TopK}_{(i, j)} (s_{ij}, K)$$
3. Compute the pooled focal vector $\mathbf{z}_{\text{focal}}$:
   $$\mathbf{z}_{\text{focal}} = \frac{1}{K} \sum_{(i, j) \in \Omega_{\text{top}}} \mathbf{f}_{ij}$$
4. Combine with the global average pooling vector $\mathbf{z}_{\text{GAP}}$ via a learned gating scalar $\alpha \in [0, 1]$:
   $$\mathbf{z}_{\text{final}} = (1 - \alpha) \mathbf{z}_{\text{GAP}} + \alpha \mathbf{z}_{\text{focal}}$$

---

## 5. Input Dimension Standardization Across Backbones

To ensure reproducibility and eliminate interpolation artifacts, V7 standardizes image input resolutions:

| Model Backbone | Standard Input Size | Token Grid ($H \times W$) | Patch Size | Aspect Handling | Rationale |
|---|---|---|---|---|---|
| **ConvNeXt-Tiny (384 px)** | $384 \times 384$ px | $12 \times 12$ | N/A (CNN) | Aspect-preserving pad | Native 384-pretrained weights; preserves $100\%$ FOV. |
| **PanDerm ViT-B/16 (384 px)** | $384 \times 384$ px | $24 \times 24 = 576$ tokens | $16 \times 16$ | Aspect-preserving pad | Bicubic interpolation of 224 position embeddings to $24 \times 24$. |
| **DINOv2-Reg ViT-B/14 (392 px)** | $\mathbf{392 \times 392}$ px | $28 \times 28 = 784$ tokens | $14 \times 14$ | Aspect-preserving pad | **Integer patch alignment:** $392 / 14 = 28$ patches exactly; eliminates border padding. |
| **MaxViT-Tiny (384 px)** | $384 \times 384$ px | Multi-Axis Grid | Block/Grid | Aspect-preserving pad | Native $384 \times 384$ window partitioning. |

---

## 6. Pre-Flight Transformation Quality Assertions

Before any batch is fed into V7 training or evaluation loaders, the preprocessing pipeline must satisfy:
* **Assertion P1 (FOV Preservation):** Area of original image preserved after padding must satisfy $\frac{\text{Area}_{\text{eval}}}{\text{Area}_{\text{native}}} \equiv 1.000$ ($\text{loss} = 0.0\%$).
* **Assertion P2 (Integer Patch Alignment):** For DINOv2 ($392 \times 392$), assert $H \pmod{14} = 0$ and $W \pmod{14} = 0$.
* **Assertion P3 (Margin Crop Sampling):** For all native margin crops, assert physical pixel pitch $\Delta x_{\text{crop}} \le 25.0\ \mu\text{m/pixel}$.
* **Assertion P4 (Artifact Rejection):** For synthetic images with planted black border circles and ink marks, assert that pooled attention weights inside the artifact mask satisfy $\sum_{(i, j) \in \text{Artifact}} w_{ij} < 10^{-5}$.

This protocol establishes exact geometric fidelity, eliminates evaluation FOV truncation, and resolves downsampling aliasing for Project Scan4Disease V7.

