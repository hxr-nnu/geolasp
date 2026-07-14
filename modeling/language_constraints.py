from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


VALID_RELATIONS = {
    "nearest",
    "near",
    "far",
    "left",
    "right",
    "front",
    "behind",
    "above",
    "below",
    "between",
    "corner",
    "against wall",
    "on the floor",
    "larger",
    "smaller",
    "taller",
    "lower",
    "supported by",
    "inside",
    "same room",
    "in corner of room",
    "touching",
    "adjacent",
    "occluding",
    "occluded by",
    "visible",
    "feature_similar",
}

RELATION_ALIASES = {
    "in corner of room": ["in the corner of the room", "in corner of room", "room corner"],
    "nearest": ["nearest to", "closest to", "nearest", "closest"],
    "on the floor": ["on the floor", "on floor", "floor"],
    "supported by": ["on top of", "sitting on", "resting on", "supported by", "on table", "on shelf", "on"],
    "inside": ["inside cabinet", "inside box", "inside", "within", "contained in", "in"],
    "same room": ["same room", "in the same room"],
    "touching": ["touching", "touches", "in contact with"],
    "adjacent": ["adjacent to", "next to", "beside"],
    "occluded by": ["occluded by", "blocked by", "hidden behind"],
    "occluding": ["occluding", "blocking"],
    "visible": ["visible", "seen", "clearly visible"],
    "feature_similar": ["feature similar", "feature_similar", "similar feature", "looks like", "visually similar"],
    "near": ["near", "next to", "beside", "close to", "nearby"],
    "far": ["far", "away from"],
    "left": ["left of", "on the left", "to the left"],
    "right": ["right of", "on the right", "to the right"],
    "front": ["in front of", "front"],
    "behind": ["behind", "back of"],
    "above": ["above", "on top of", "on the top", "on"],
    "below": ["below", "under", "beneath"],
    "between": ["between"],
    "corner": ["corner"],
    "against wall": ["against the wall", "against wall"],
    "larger": ["larger", "bigger"],
    "smaller": ["smaller", "shorter", "narrower"],
    "taller": ["taller", "higher"],
    "lower": ["lower", "bottom"],
}


def normalize_relation_name(name: str) -> str:
    name = name.strip().lower().replace("_", " ")
    alias_map = {
        "against the wall": "against wall",
        "on floor": "on the floor",
        "on table": "supported by",
        "on shelf": "supported by",
        "on": "supported by",
        "on top of": "supported by",
        "resting on": "supported by",
        "sitting on": "supported by",
        "contained in": "inside",
        "within": "inside",
        "inside cabinet": "inside",
        "inside box": "inside",
        "in": "inside",
        "in the same room": "same room",
        "room corner": "in corner of room",
        "in the corner of the room": "in corner of room",
        "adjacent to": "adjacent",
        "touches": "touching",
        "in contact with": "touching",
        "blocked by": "occluded by",
        "hidden behind": "occluded by",
        "blocking": "occluding",
        "feature similar": "feature_similar",
        "similar feature": "feature_similar",
        "looks like": "feature_similar",
        "visually similar": "feature_similar",
        "closest": "nearest",
        "closest to": "nearest",
        "nearest to": "nearest",
    }
    return alias_map.get(name, name)


COLOR_WORDS = {
    "black",
    "white",
    "red",
    "green",
    "blue",
    "yellow",
    "brown",
    "gray",
    "grey",
    "orange",
    "purple",
    "pink",
}
MATERIAL_WORDS = {
    "wood": "wooden",
    "wooden": "wooden",
    "metal": "metal",
    "metallic": "metal",
    "plastic": "plastic",
    "glass": "glass",
    "leather": "leather",
    "fabric": "fabric",
    "cloth": "fabric",
    "stone": "stone",
    "concrete": "concrete",
}
SHAPE_WORDS = {
    "round",
    "circular",
    "square",
    "rectangular",
    "long",
    "wide",
    "narrow",
    "thin",
    "flat",
    "tall",
    "low",
}


@dataclass
class RelationConstraint:
    relation: str
    anchor_label: str | None = None
    anchors: list[LanguageConstraint] = field(default_factory=list)
    negative: bool = False


@dataclass
class LanguageConstraint:
    target_label: str | None = None
    attributes: dict[str, str] = field(default_factory=dict)
    relations: list[RelationConstraint] = field(default_factory=list)
    raw_json: dict[str, Any] | None = None
    source: str = "rule"


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if "```json" in stripped:
        stripped = stripped.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in stripped:
        stripped = stripped.split("```", 1)[1].split("```", 1)[0].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end >= start:
        stripped = stripped[start : end + 1]
    return json.loads(stripped)


def normalize_constraint_json(obj: dict[str, Any]) -> dict[str, Any]:
    category = str(obj.get("category") or obj.get("target") or "object").strip().lower()
    attrs = obj.get("attributes") or {}
    if not isinstance(attrs, dict):
        attrs = {}

    relations = []
    for rel in obj.get("relations") or []:
        if not isinstance(rel, dict):
            continue
        name = normalize_relation_name(str(rel.get("relation_name") or rel.get("relation") or ""))
        if name not in VALID_RELATIONS:
            continue
        children = rel.get("objects") or rel.get("anchors") or []
        norm_children = [normalize_constraint_json(child) for child in children if isinstance(child, dict)]
        item = {"relation_name": name, "objects": norm_children}
        if rel.get("negative") is True:
            item["negative"] = True
        relations.append(item)

    return {
        "category": category,
        "attributes": {str(k).lower(): str(v).lower() for k, v in attrs.items()},
        "relations": relations,
    }


def constraint_from_json(obj: dict[str, Any], source: str = "llm") -> LanguageConstraint:
    norm = normalize_constraint_json(obj)
    constraint = _constraint_from_normalized(norm)
    constraint.source = source
    constraint.raw_json = norm
    return constraint


def _constraint_from_normalized(norm: dict[str, Any]) -> LanguageConstraint:
    relations: list[RelationConstraint] = []
    for rel in norm["relations"]:
        anchor_constraints = [_constraint_from_normalized(anchor) for anchor in rel.get("objects") or []]
        anchor_label = anchor_constraints[0].target_label if anchor_constraints else None
        relations.append(
            RelationConstraint(
                relation=rel["relation_name"],
                anchor_label=anchor_label,
                anchors=anchor_constraints,
                negative=bool(rel.get("negative")),
            )
        )
    return LanguageConstraint(
        target_label=norm["category"],
        attributes=norm["attributes"],
        relations=relations,
    )


def parse_constraints_rule(text: str, known_labels: list[str]) -> LanguageConstraint:
    lowered = text.lower()
    labels = sorted(set(known_labels), key=len, reverse=True)
    label_hits = []
    for label in labels:
        normalized = label.lower().replace("_", " ")
        pos = lowered.find(normalized)
        if pos >= 0:
            label_hits.append((pos, -len(normalized), label))
    target_label = sorted(label_hits)[0][2] if label_hits else None
    attributes = _extract_attributes_rule(lowered)

    relations: list[RelationConstraint] = []
    relation_hits = _find_relation_hits(lowered)
    for relation, hit, _start, end in relation_hits:
        anchor_label = None
        suffix = lowered[end:]
        anchor_hits = []
        for label in labels:
            normalized = label.lower().replace("_", " ")
            pos = suffix.find(normalized)
            if pos >= 0:
                anchor_hits.append((pos, -len(normalized), label))
        if anchor_hits:
            anchor_label = sorted(anchor_hits)[0][2]
        anchors = [LanguageConstraint(target_label=anchor_label)] if anchor_label else []
        relations.append(RelationConstraint(relation=relation, anchor_label=anchor_label, anchors=anchors))
    raw = {
        "category": target_label or "object",
        "attributes": attributes,
        "relations": [
            {"relation_name": rel.relation, "objects": ([{"category": rel.anchor_label, "relations": []}] if rel.anchor_label else [])}
            for rel in relations
        ],
    }
    return LanguageConstraint(target_label=target_label, attributes=attributes, relations=relations, raw_json=raw, source="rule")


def _extract_attributes_rule(text: str) -> dict[str, str]:
    attributes: dict[str, str] = {}
    color_hit = next((color for color in sorted(COLOR_WORDS, key=len, reverse=True) if _contains_word(text, color)), None)
    if color_hit:
        attributes["color"] = "gray" if color_hit == "grey" else color_hit

    material_hit = next((word for word in sorted(MATERIAL_WORDS, key=len, reverse=True) if _contains_word(text, word)), None)
    if material_hit:
        attributes["material"] = MATERIAL_WORDS[material_hit]

    shape_hit = next((shape for shape in sorted(SHAPE_WORDS, key=len, reverse=True) if _contains_word(text, shape)), None)
    if shape_hit:
        attributes["shape"] = "round" if shape_hit == "circular" else shape_hit
    return attributes


def _contains_word(text: str, word: str) -> bool:
    return re.search(rf"(?<![a-zA-Z0-9_]){re.escape(word)}(?![a-zA-Z0-9_])", text) is not None


def _find_relation_hits(text: str) -> list[tuple[str, str, int, int]]:
    candidates: list[tuple[int, int, str, str]] = []
    for relation, aliases in RELATION_ALIASES.items():
        for alias in aliases:
            pattern = rf"(?<![a-zA-Z0-9_]){re.escape(alias)}(?![a-zA-Z0-9_])"
            for match in re.finditer(pattern, text):
                candidates.append((match.start(), match.end(), relation, alias))

    occupied: list[tuple[int, int]] = []
    selected: list[tuple[int, int, str, str]] = []
    for start, end, relation, alias in sorted(candidates, key=lambda item: (-(item[1] - item[0]), item[0])):
        if any(start < used_end and end > used_start for used_start, used_end in occupied):
            continue
        occupied.append((start, end))
        selected.append((start, end, relation, alias))

    selected.sort(key=lambda item: item[0])
    return [(relation, alias, start, end) for start, end, relation, alias in selected]


class LLMConstraintParser:
    def __init__(
        self,
        model: str = "gpt-4o-2024-08-06",
        cache_dir: str | os.PathLike | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def parse(self, text: str, known_labels: list[str]) -> LanguageConstraint:
        cache_file = self._cache_file(text) if self.cache_dir else None
        if cache_file and cache_file.exists():
            return constraint_from_json(json.loads(cache_file.read_text(encoding="utf-8")), source="llm_cache")

        if not os.environ.get("OPENAI_API_KEY"):
            return parse_constraints_rule(text, known_labels)

        from openai import OpenAI

        labels = ", ".join(sorted(set(known_labels)))
        schema = {
            "category": "chair",
            "attributes": {"color": "brown", "material": "wooden", "shape": "rectangular"},
            "relations": [
                {"relation_name": "left", "objects": [{"category": "table", "relations": []}]},
                {"relation_name": "supported by", "objects": [{"category": "table", "relations": []}]},
                {"relation_name": "inside", "objects": [{"category": "cabinet", "relations": []}]},
            ],
        }
        system = (
            "You parse indoor 3D visual grounding descriptions into strict JSON. "
            "Return only valid JSON. Do not include markdown. "
            "Use category, attributes, and relations. Each relation uses relation_name and objects. "
            "Use attributes for visual properties such as color, material, and shape. "
            "Use empty arrays when there are no anchors."
        )
        user = (
            f"Known object labels in this scene: {labels}\n"
            f"Allowed relations: {', '.join(sorted(VALID_RELATIONS))}\n"
            "Allowed attribute keys: color, material, shape\n"
            f"JSON template: {json.dumps(schema, ensure_ascii=False)}\n"
            f"Description: {text}"
        )
        response = OpenAI(api_key=os.environ["OPENAI_API_KEY"]).chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self.temperature,
            top_p=0.3,
            max_tokens=512,
        )
        raw = response.choices[0].message.content or "{}"
        constraint_json = normalize_constraint_json(_extract_json(raw))
        if cache_file:
            cache_file.write_text(json.dumps(constraint_json, ensure_ascii=False, indent=2), encoding="utf-8")
        return constraint_from_json(constraint_json, source="llm")

    def _cache_file(self, text: str) -> Path:
        key = hashlib.sha1(f"{self.model}\n{text}".encode("utf-8")).hexdigest()
        assert self.cache_dir is not None
        return self.cache_dir / f"{key}.json"


def parse_constraints(
    text: str,
    known_labels: list[str],
    parser: LLMConstraintParser | None = None,
    mode: str = "auto",
) -> LanguageConstraint:
    if mode == "rule":
        return parse_constraints_rule(text, known_labels)
    if parser is None:
        return parse_constraints_rule(text, known_labels)
    try:
        return parser.parse(text, known_labels)
    except Exception:
        return parse_constraints_rule(text, known_labels)
