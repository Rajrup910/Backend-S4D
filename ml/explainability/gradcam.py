"""Grad-CAM, implemented directly rather than pulled from a library.

Gradient-weighted Class Activation Mapping (Selvaraju et al., ICCV 2017) answers:
*which regions of the image pushed the model toward the class it predicted?*

The procedure, matching section 17 of the master specification:

    1. forward pass, capture the feature maps A^k of a late conv layer
    2. pick the predicted class c
    3. backward pass from the score y^c, capture dy^c/dA^k
    4. importance weight per channel: a^c_k = GAP(dy^c/dA^k)
    5. weighted sum over channels
    6. ReLU  -- keep only evidence *for* the class, discard evidence against it
    7. normalise to [0, 1]
    8. resize to the input resolution and overlay

Two implementation notes worth knowing:

* The input tensor is marked `requires_grad`. Without it, a model loaded purely for
  inference (all parameters frozen) builds no autograd graph at all, and the backward
  pass silently yields nothing.
* Hooks are registered and removed via a context manager. Leaving forward hooks attached
  to a model that the API server reuses across requests is a memory leak that only shows
  up under load.

This module has no dependency on the training code, so the backend can import it directly.
"""

from __future__ import annotations

from types import TracebackType
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Grad-CAM for a single model and target layer.

    Use as a context manager so the hooks are always cleaned up:

        with GradCAM(model, target_layer) as cam:
            heatmap, class_index, probabilities = cam(input_tensor)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles: list[Any] = []

    # --- hook plumbing ---

    def _save_activations(self, _module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
        self._activations = output.detach()

    def _save_gradients(self, _module: nn.Module, _grad_input: Any, grad_output: tuple) -> None:
        self._gradients = grad_output[0].detach()

    def __enter__(self) -> GradCAM:
        self._handles.append(self.target_layer.register_forward_hook(self._save_activations))
        self._handles.append(self.target_layer.register_full_backward_hook(self._save_gradients))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._activations = None
        self._gradients = None

    # --- the actual computation ---

    def __call__(
        self, input_tensor: torch.Tensor, class_index: int | None = None
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """Compute the class activation map for one image.

        Args:
            input_tensor: normalised image, shape (1, 3, H, W) or (3, H, W).
            class_index: which class to explain. Defaults to the predicted class.

        Returns:
            (cam, class_index, probabilities)
            `cam` is float32 in [0, 1] at the feature-map resolution (e.g. 7x7).
        """
        if not self._handles:
            raise RuntimeError(
                "GradCAM hooks are not registered. Use it as a context manager: "
                "`with GradCAM(model, layer) as cam: ...`"
            )

        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        if input_tensor.size(0) != 1:
            raise ValueError(f"Grad-CAM expects a single image, got batch of {input_tensor.size(0)}")

        was_training = self.model.training
        self.model.eval()

        # Without this, a fully-frozen inference model produces no autograd graph.
        input_tensor = input_tensor.clone().requires_grad_(True)

        with torch.enable_grad():
            self.model.zero_grad(set_to_none=True)
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

            if class_index is None:
                class_index = int(logits.argmax(dim=1).item())

            logits[0, class_index].backward()

        if was_training:
            self.model.train()

        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "No activations or gradients were captured. The target layer was probably not "
                "executed during the forward pass -- check that it belongs to this model."
            )

        activations = self._activations[0]  # (K, h, w)
        gradients = self._gradients[0]  # (K, h, w)

        # Step 4: channel importance = spatially averaged gradient.
        weights = gradients.mean(dim=(1, 2))  # (K,)

        # Steps 5-6: weighted sum, then ReLU to keep only positive evidence.
        cam = F.relu((weights[:, None, None] * activations).sum(dim=0))

        # Step 7: normalise. A uniformly-zero map means the class had no positive
        # spatial evidence; return zeros rather than dividing by zero.
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min) if cam_max > cam_min else torch.zeros_like(cam)

        return cam.cpu().numpy().astype(np.float32), class_index, probabilities


def resize_cam(cam: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a CAM to (width, height) with bilinear interpolation."""
    tensor = torch.from_numpy(cam)[None, None]
    resized = F.interpolate(
        tensor, size=(size[1], size[0]), mode="bilinear", align_corners=False
    )
    return resized[0, 0].numpy()


def apply_colormap(cam: np.ndarray, colormap: str = "jet") -> np.ndarray:
    """Map a [0, 1] CAM to an RGB uint8 image."""
    from matplotlib import colormaps

    coloured = colormaps[colormap](np.clip(cam, 0.0, 1.0))[..., :3]
    return (coloured * 255).astype(np.uint8)


def overlay_heatmap(
    image: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> Image.Image:
    """Blend a CAM over the original image.

    `alpha` is the heatmap's weight. Above roughly 0.5 the lesion itself becomes hard to
    see, which defeats the purpose -- the viewer needs to judge whether the highlighted
    region actually is the lesion.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    image = image.convert("RGB")
    resized = resize_cam(cam, image.size)
    heatmap = apply_colormap(resized, colormap)

    base = np.asarray(image, dtype=np.float32)
    blended = (1 - alpha) * base + alpha * heatmap.astype(np.float32)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def cam_statistics(cam: np.ndarray, border_fraction: float = 0.15) -> dict[str, Any]:
    """Quantitative sanity checks on a heatmap (master spec, section 18).

    Grad-CAM is only useful as evidence if it lands on the lesion. Lesions in HAM10000 are
    roughly centred, so a map whose mass sits on the border is a warning sign -- the model
    may be keying on a vignette, a ruler, a hair, or the frame of the dermatoscope rather
    than the lesion itself.

    These numbers do not prove the model reasons correctly. They flag images worth looking
    at by eye, which is the honest claim to make in the report.
    """
    cam = np.asarray(cam, dtype=np.float32)
    height, width = cam.shape
    total = float(cam.sum())

    band_h = max(1, int(round(height * border_fraction)))
    band_w = max(1, int(round(width * border_fraction)))

    interior = cam[band_h : height - band_h, band_w : width - band_w]
    border_mass = total - float(interior.sum())

    peak_index = int(np.argmax(cam))
    peak_y, peak_x = divmod(peak_index, width)

    # Distance of the peak from the centre, normalised so 1.0 is a corner.
    centre_y, centre_x = (height - 1) / 2, (width - 1) / 2
    max_distance = float(np.hypot(centre_y, centre_x)) or 1.0
    peak_offset = float(np.hypot(peak_y - centre_y, peak_x - centre_x)) / max_distance

    return {
        "resolution": [int(height), int(width)],
        "border_mass_fraction": float(border_mass / total) if total > 0 else 0.0,
        "interior_mass_fraction": float(interior.sum() / total) if total > 0 else 0.0,
        "peak_position": [int(peak_x), int(peak_y)],
        "peak_offset_from_centre": peak_offset,
        "mean_activation": float(cam.mean()),
        "activation_above_half": float((cam > 0.5).mean()),
        "is_degenerate": bool(total == 0.0),
    }


def describe_focus(stats: dict[str, Any]) -> str:
    """One short phrase describing where the model looked.

    Passed to the LLM as context. It is intentionally vague about anatomy -- the model has
    no notion of lesion borders or structures, and inventing that language would be exactly
    the overclaim the safety layer exists to prevent.
    """
    if stats["is_degenerate"]:
        return "no clear region of interest"
    if stats["border_mass_fraction"] > 0.5:
        return "spread toward the edges of the image rather than a central region"
    if stats["peak_offset_from_centre"] < 0.3:
        return "concentrated on the central region of the image"
    return "concentrated on an off-centre region of the image"
