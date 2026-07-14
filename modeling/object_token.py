from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any

import numpy as np


@dataclass
class ObjectToken:
    object_id: int
    label: str
    center: np.ndarray
    size: np.ndarray
    yaw: float = 0.0
    confidence: float = 1.0
    embedding: np.ndarray | None = None
    point_stats: dict[str, Any] | None = None
    attributes: dict[str, str] = field(default_factory=dict)

    @property
    def loc6(self) -> np.ndarray:
        return np.concatenate([self.center.astype(float), self.size.astype(float)], axis=0)

    @property
    def volume(self) -> float:
        return float(np.prod(np.maximum(self.size, 1e-6)))

    def to_json(self) -> dict[str, Any]:
        item = asdict(self)
        item["center"] = self.center.tolist()
        item["size"] = self.size.tolist()
        if self.embedding is not None:
            item["embedding"] = self.embedding.tolist()
        return item


def from_spatiallm_box(box: dict, index: int) -> ObjectToken:
    center = np.asarray(box["center"], dtype=np.float32)
    size = np.asarray(box["scale"], dtype=np.float32)
    label = str(box.get("label") or box.get("class") or "unknown")
    attributes = box.get("attributes") or {}
    if not isinstance(attributes, dict):
        attributes = {}
    return ObjectToken(
        object_id=int(box.get("id", index)),
        label=label,
        center=center,
        size=np.abs(size),
        yaw=float(box.get("yaw", box.get("angle_z", 0.0))),
        confidence=float(box.get("score", 1.0)),
        attributes={str(k).lower(): str(v).lower() for k, v in attributes.items()},
    )
