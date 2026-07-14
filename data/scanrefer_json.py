from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class GroundingSample:
    scene_id: str
    description: str
    target_id: int | None = None


def load_grounding_json(path: str | Path) -> list[GroundingSample]:
    raw: list[dict[str, Any]] = json.loads(Path(path).read_text(encoding="utf-8"))
    samples: list[GroundingSample] = []
    for item in raw:
        scene_id = item.get("scene_id") or item.get("scan_id")
        description = item.get("description") or item.get("caption") or item.get("utterance")
        target = item.get("target_id", item.get("object_id"))
        if scene_id is None or description is None:
            continue
        samples.append(GroundingSample(scene_id=str(scene_id), description=str(description), target_id=None if target is None else int(target)))
    return samples

