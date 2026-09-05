"""Repository path resolution and config loading.

Every script resolves paths through here so that relative paths in
`ml/configs/training_config.yaml` mean the same thing regardless of the working
directory a script was launched from.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# ml/paths.py -> ml/ -> repo root
REPO_ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR = REPO_ROOT / "ml" / "configs"
CLASS_MAPPING_PATH = CONFIG_DIR / "class_mapping.json"
TRAINING_CONFIG_PATH = CONFIG_DIR / "training_config.yaml"


def resolve(path: str | Path) -> Path:
    """Resolve a possibly-relative path against the repository root."""
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)


@lru_cache(maxsize=1)
def load_training_config() -> dict[str, Any]:
    with TRAINING_CONFIG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass(frozen=True)
class SkinClass:
    """One dermatological class, as declared in class_mapping.json."""

    index: int
    code: str
    name_en: str
    name_hi: str
    short_name_en: str
    short_name_hi: str
    plain_language_en: str
    plain_language_hi: str
    malignancy: str  # malignant | premalignant | benign

    @property
    def is_malignant(self) -> bool:
        return self.malignancy == "malignant"

    @property
    def needs_escalation(self) -> bool:
        """Malignant and pre-malignant classes both escalate in the triage layer."""
        return self.malignancy in ("malignant", "premalignant")


@dataclass(frozen=True)
class ClassMapping:
    """The version-controlled class mapping, loaded once and treated as read-only."""

    version: str
    classes: tuple[SkinClass, ...]

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(c.code for c in self.classes)

    @property
    def display_names(self) -> tuple[str, ...]:
        return tuple(c.short_name_en for c in self.classes)

    def by_code(self, code: str) -> SkinClass:
        for c in self.classes:
            if c.code == code:
                return c
        raise KeyError(f"unknown class code {code!r}; known codes are {self.codes}")

    def by_index(self, index: int) -> SkinClass:
        try:
            return self.classes[index]
        except IndexError as exc:
            raise KeyError(f"class index {index} out of range (0..{self.num_classes - 1})") from exc

    def index_of(self, code: str) -> int:
        return self.by_code(code).index

    def dx_to_index(self) -> dict[str, int]:
        """HAM10000 `dx` value -> class index."""
        return {code: c.index for c in self.classes for code in (c.code,)}


@lru_cache(maxsize=1)
def load_class_mapping() -> ClassMapping:
    with CLASS_MAPPING_PATH.open(encoding="utf-8") as fh:
        raw = json.load(fh)

    classes = []
    for entry in sorted(raw["classes"], key=lambda e: e["index"]):
        classes.append(
            SkinClass(
                index=entry["index"],
                code=entry["code"],
                name_en=entry["name"]["en"],
                name_hi=entry["name"]["hi"],
                short_name_en=entry["short_name"]["en"],
                short_name_hi=entry["short_name"]["hi"],
                plain_language_en=entry["plain_language"]["en"],
                plain_language_hi=entry["plain_language"]["hi"],
                malignancy=entry["malignancy"],
            )
        )

    expected = raw["num_classes"]
    if len(classes) != expected:
        raise ValueError(
            f"class_mapping.json declares num_classes={expected} but lists {len(classes)} classes"
        )
    indices = [c.index for c in classes]
    if indices != list(range(len(classes))):
        raise ValueError(f"class indices must be contiguous from 0; got {indices}")

    return ClassMapping(version=raw["version"], classes=tuple(classes))
