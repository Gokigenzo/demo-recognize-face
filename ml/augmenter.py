"""Data augmentation (data preparation stage).

Generates extra training samples from each captured face so a handful of
photos becomes a richer dataset. Uses Albumentations when available, with a
pure-OpenCV fallback so the demo never hard-fails.

Each augmentation returns ``(label, image)`` tuples so the UI can build a
before/after gallery and explain *what* each transform does.
"""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

try:  # Albumentations is optional at runtime.
    import albumentations as A

    _ALBU = True
except Exception:  # noqa: BLE001
    _ALBU = False


def albumentations_available() -> bool:
    return _ALBU


def _albu_pipeline() -> "List[Tuple[str, object]]":
    """Named single-transform pipelines (one effect each, for clarity)."""
    return [
        ("Brightness", A.RandomBrightnessContrast(
            brightness_limit=0.4, contrast_limit=0.0, p=1.0)),
        ("Contrast", A.RandomBrightnessContrast(
            brightness_limit=0.0, contrast_limit=0.5, p=1.0)),
        ("Blur", A.GaussianBlur(blur_limit=(3, 7), p=1.0)),
        ("Flip", A.HorizontalFlip(p=1.0)),
    ]


def _opencv_augment(image: np.ndarray, effect: str) -> np.ndarray:
    """Fallback transforms implemented directly in OpenCV."""
    if effect == "Brightness":
        return cv2.convertScaleAbs(image, alpha=1.0, beta=40)
    if effect == "Contrast":
        return cv2.convertScaleAbs(image, alpha=1.5, beta=0)
    if effect == "Blur":
        return cv2.GaussianBlur(image, (7, 7), 0)
    if effect == "Flip":
        return cv2.flip(image, 1)
    return image


def augment_image(image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """Return a list of ``(effect_name, augmented_image)`` for one image."""
    if image is None or image.size == 0:
        return []
    effects = ["Brightness", "Contrast", "Blur", "Flip"]
    results: List[Tuple[str, np.ndarray]] = []
    if _ALBU:
        for name, transform in _albu_pipeline():
            try:
                out = transform(image=image)["image"]
            except Exception:  # noqa: BLE001 - fall back per-effect
                out = _opencv_augment(image, name)
            results.append((name, out))
    else:
        for name in effects:
            results.append((name, _opencv_augment(image, name)))
    return results


def expand_dataset(images: List[np.ndarray]) -> List[Tuple[str, np.ndarray]]:
    """Expand a list of base images into originals + all augmentations.

    Returns ``(label, image)`` tuples where the label is ``"Original"`` or the
    augmentation name. This is what powers the "10 -> 50 images" headline.
    """
    expanded: List[Tuple[str, np.ndarray]] = []
    for img in images:
        expanded.append(("Original", img))
        expanded.extend(augment_image(img))
    return expanded
