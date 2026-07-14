from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from geolasp.modeling.language_constraints import LanguageConstraint, RelationConstraint
from geolasp.modeling.scene_graph import GeometricSceneGraph


@dataclass
class AnchorExplanation:
    index: int
    object_id: int
    label: str
    score: float


@dataclass
class RelationExplanation:
    relation: str
    negative: bool
    scores: np.ndarray
    anchor_scores: np.ndarray | None = None
    top_anchors: list[AnchorExplanation] = field(default_factory=list)
    anchor_traces: list[ReasoningTrace] = field(default_factory=list)


@dataclass
class AttributeExplanation:
    name: str
    value: str
    scores: np.ndarray


@dataclass
class ReasoningTrace:
    constraint: LanguageConstraint
    category_scores: np.ndarray
    attribute_explanations: list[AttributeExplanation]
    attribute_scores: np.ndarray
    relation_explanations: list[RelationExplanation]
    raw_scores: np.ndarray
    final_scores: np.ndarray
    depth: int = 0


class GeometricReasoner:
    def __init__(self, graph: GeometricSceneGraph):
        self.graph = graph

    def score(self, constraint: LanguageConstraint, weights: dict[str, float] | None = None) -> np.ndarray:
        return self.evaluate(constraint, weights=weights).final_scores

    def evaluate(
        self,
        constraint: LanguageConstraint,
        weights: dict[str, float] | None = None,
        depth: int = 0,
        max_depth: int = 8,
    ) -> ReasoningTrace:
        n = len(self.graph.objects)
        if n == 0:
            empty = np.zeros((0,), dtype=np.float32)
            return ReasoningTrace(constraint, empty, [], empty, [], empty, empty, depth=depth)
        if depth > max_depth:
            uniform = np.ones((n,), dtype=np.float32)
            return ReasoningTrace(constraint, uniform, [], uniform, [], uniform, self._normalize(uniform), depth=depth)

        weights = weights or {}
        scores = np.ones((n,), dtype=np.float32)
        category_scores = self._category_scores(constraint)
        if constraint.target_label:
            scores *= self._blend(category_scores, weights.get("category", 1.0))

        attribute_explanations, attribute_scores = self._attribute_scores(constraint)
        if constraint.attributes:
            scores *= self._blend(attribute_scores, weights.get("attributes", 1.0))

        relation_explanations: list[RelationExplanation] = []
        for rel in constraint.relations:
            rel_scores, explanation = self._score_relation(rel, weights, depth, max_depth)
            relation_explanations.append(explanation)
            scores *= self._blend(rel_scores, weights.get(rel.relation, 1.0))

        final_scores = self._normalize(scores)
        return ReasoningTrace(
            constraint=constraint,
            category_scores=category_scores,
            attribute_explanations=attribute_explanations,
            attribute_scores=attribute_scores,
            relation_explanations=relation_explanations,
            raw_scores=scores,
            final_scores=final_scores,
            depth=depth,
        )

    def explain(
        self,
        constraint: LanguageConstraint,
        weights: dict[str, float] | None = None,
        top_k_anchors: int = 3,
    ) -> list[dict[str, Any]]:
        trace = self.evaluate(constraint, weights=weights)
        return [self.explain_candidate(trace, i, top_k_anchors=top_k_anchors) for i in range(len(self.graph.objects))]

    def explain_candidate(self, trace: ReasoningTrace, index: int, top_k_anchors: int = 3) -> dict[str, Any]:
        obj = self.graph.objects[index]
        attribute_items = [
            {
                "name": attr.name,
                "value": attr.value,
                "score": float(attr.scores[index]) if attr.scores.size else 0.0,
            }
            for attr in trace.attribute_explanations
        ]
        relation_items = []
        for rel in trace.relation_explanations:
            anchors = self._top_candidate_anchors(rel.relation, index, rel.anchor_scores, top_k_anchors)
            anchor_match_score = max((anchor.score for anchor in anchors), default=0.0)
            relation_items.append(
                {
                    "relation": rel.relation,
                    "negative": rel.negative,
                    "score": float(rel.scores[index]) if rel.scores.size else 0.0,
                    "relation_match_score": float(rel.scores[index]) if rel.scores.size else 0.0,
                    "anchor_match_score": float(anchor_match_score),
                    "anchor_constraints": [self._trace_summary(anchor_trace) for anchor_trace in rel.anchor_traces],
                    "anchors": [
                        {
                            "index": anchor.index,
                            "object_id": anchor.object_id,
                            "label": anchor.label,
                            "score": anchor.score,
                        }
                        for anchor in anchors
                    ],
                }
            )

        final_score = float(trace.final_scores[index]) if trace.final_scores.size else 0.0
        category_score = float(trace.category_scores[index]) if trace.category_scores.size else 0.0
        attribute_score = float(trace.attribute_scores[index]) if trace.attribute_scores.size else 1.0
        return {
            "index": index,
            "object_id": int(obj.object_id),
            "label": obj.label,
            "category_score": category_score,
            "category_match_score": category_score,
            "attribute_score": attribute_score,
            "attributes": attribute_items,
            "relations": relation_items,
            "raw_score": float(trace.raw_scores[index]) if trace.raw_scores.size else 0.0,
            "final_score": final_score,
            "explanation": self._candidate_sentence(obj.label, category_score, attribute_items, relation_items, final_score),
        }

    def _trace_summary(self, trace: ReasoningTrace) -> dict[str, Any]:
        if trace.final_scores.size == 0:
            return {"target_label": trace.constraint.target_label, "best_anchor": None, "relations": []}
        best_idx = int(trace.final_scores.argmax())
        obj = self.graph.objects[best_idx]
        return {
            "target_label": trace.constraint.target_label,
            "best_anchor": {
                "index": best_idx,
                "object_id": int(obj.object_id),
                "label": obj.label,
                "score": float(trace.final_scores[best_idx]),
            },
            "relations": [
                {
                    "relation": rel.relation,
                    "negative": rel.negative,
                    "score": float(rel.scores[best_idx]) if rel.scores.size else 0.0,
                    "anchor_constraints": [self._trace_summary(anchor_trace) for anchor_trace in rel.anchor_traces],
                }
                for rel in trace.relation_explanations
            ],
        }

    def _attribute_scores(self, constraint: LanguageConstraint) -> tuple[list[AttributeExplanation], np.ndarray]:
        n = len(self.graph.objects)
        if not constraint.attributes:
            return [], np.ones((n,), dtype=np.float32)

        explanations = []
        score_vectors = []
        for name, value in constraint.attributes.items():
            attr_scores = np.array(
                [self._object_attribute_match(obj, name, value) for obj in self.graph.objects],
                dtype=np.float32,
            )
            explanations.append(AttributeExplanation(name=name, value=value, scores=attr_scores))
            score_vectors.append(np.clip(attr_scores, 1e-4, 1.0))

        if not score_vectors:
            return explanations, np.ones((n,), dtype=np.float32)
        stacked = np.stack(score_vectors, axis=0)
        return explanations, np.exp(np.log(stacked).mean(axis=0)).astype(np.float32)

    def _score_relation(
        self,
        rel: RelationConstraint,
        weights: dict[str, float],
        depth: int,
        max_depth: int,
    ) -> tuple[np.ndarray, RelationExplanation]:
        n = len(self.graph.objects)
        mat = self.graph.relation_matrix(rel.relation)
        anchor_constraints = self._anchor_constraints(rel)
        anchor_traces = [
            self.evaluate(anchor, weights=weights, depth=depth + 1, max_depth=max_depth)
            for anchor in anchor_constraints
        ]

        if anchor_traces:
            anchor_vectors = [self._normalize(trace.final_scores) for trace in anchor_traces]
            anchor_scores = self._combine_anchor_scores(anchor_vectors)
            raw_rel_scores = self._combine_relation_scores([mat @ vec for vec in anchor_vectors])
        elif rel.anchor_label:
            anchor_scores = np.array(
                [self._label_match(obj.label, rel.anchor_label) for obj in self.graph.objects],
                dtype=np.float32,
            )
            raw_rel_scores = mat @ self._normalize(anchor_scores)
        else:
            anchor_scores = None
            raw_rel_scores = mat.max(axis=1) if mat.size else np.zeros((n,), dtype=np.float32)

        rel_scores = self._normalize(raw_rel_scores)
        if rel.negative:
            rel_scores = 1.0 - rel_scores

        explanation = RelationExplanation(
            relation=rel.relation,
            negative=rel.negative,
            scores=rel_scores,
            anchor_scores=None if anchor_scores is None else self._normalize(anchor_scores),
            top_anchors=[] if anchor_scores is None else self._top_anchors(anchor_scores),
            anchor_traces=anchor_traces,
        )
        return rel_scores, explanation

    def _anchor_constraints(self, rel: RelationConstraint) -> list[LanguageConstraint]:
        if rel.anchors:
            return rel.anchors
        if rel.anchor_label:
            return [LanguageConstraint(target_label=rel.anchor_label)]
        return []

    def _category_scores(self, constraint: LanguageConstraint) -> np.ndarray:
        if not constraint.target_label:
            return np.ones((len(self.graph.objects),), dtype=np.float32)
        return np.array(
            [self._label_match(obj.label, constraint.target_label) for obj in self.graph.objects],
            dtype=np.float32,
        )

    def _object_attribute_match(self, obj, name: str, value: str) -> float:
        name = name.lower().strip()
        value = self._canonical_attribute_value(value)
        raw_attributes = getattr(obj, "attributes", None) or {}
        attributes = {str(k).lower(): self._canonical_attribute_value(v) for k, v in raw_attributes.items()}
        label_text = obj.label.lower().replace("_", " ")

        explicit = attributes.get(name)
        if explicit:
            return 1.0 if self._text_match(explicit, value) else 0.05
        if self._text_match(label_text, value):
            return 1.0
        if name == "shape":
            return self._shape_match(obj, value)
        return 0.05

    def _top_anchors(self, anchor_scores: np.ndarray, k: int = 5) -> list[AnchorExplanation]:
        if anchor_scores.size == 0:
            return []
        scores = np.asarray(anchor_scores, dtype=np.float32)
        order = np.argsort(-scores)[:k]
        anchors = []
        for idx in order:
            obj = self.graph.objects[int(idx)]
            anchors.append(
                AnchorExplanation(
                    index=int(idx),
                    object_id=int(obj.object_id),
                    label=obj.label,
                    score=float(scores[idx]),
                )
            )
        return anchors

    def _top_candidate_anchors(
        self,
        relation: str,
        candidate_idx: int,
        anchor_scores: np.ndarray | None,
        k: int,
    ) -> list[AnchorExplanation]:
        if anchor_scores is None or anchor_scores.size == 0:
            return []
        mat = self.graph.relation_matrix(relation)
        if mat.size == 0:
            return []
        contributions = mat[candidate_idx] * self._normalize(anchor_scores)
        order = np.argsort(-contributions)[:k]
        anchors = []
        for idx in order:
            if contributions[idx] <= 0:
                continue
            obj = self.graph.objects[int(idx)]
            anchors.append(
                AnchorExplanation(
                    index=int(idx),
                    object_id=int(obj.object_id),
                    label=obj.label,
                    score=float(contributions[idx]),
                )
            )
        return anchors

    @staticmethod
    def _combine_anchor_scores(anchor_vectors: list[np.ndarray]) -> np.ndarray:
        if not anchor_vectors:
            return np.zeros((0,), dtype=np.float32)
        if len(anchor_vectors) == 1:
            return anchor_vectors[0]
        stacked = np.stack(anchor_vectors, axis=0)
        return stacked.max(axis=0)

    @staticmethod
    def _combine_relation_scores(score_vectors: list[np.ndarray]) -> np.ndarray:
        if not score_vectors:
            return np.zeros((0,), dtype=np.float32)
        if len(score_vectors) == 1:
            return score_vectors[0]
        stacked = np.stack([np.clip(vec, 1e-4, 1.0) for vec in score_vectors], axis=0)
        return np.exp(np.log(stacked).mean(axis=0)).astype(np.float32)

    @staticmethod
    def _candidate_sentence(
        label: str,
        category_score: float,
        attributes: list[dict[str, Any]],
        relations: list[dict[str, Any]],
        final_score: float,
    ) -> str:
        pieces = [f"label={label} category_match={category_score:.3f}"]
        for attr in attributes:
            pieces.append(f"{attr['name']}={attr['value']}:{attr['score']:.3f}")
        for rel in relations:
            anchor_text = ""
            if rel["anchors"]:
                anchor = rel["anchors"][0]
                anchor_text = f" anchor={anchor['label']}#{anchor['object_id']}:{rel['anchor_match_score']:.3f}"
            polarity = "not " if rel["negative"] else ""
            pieces.append(f"{polarity}relation_match({rel['relation']})={rel['score']:.3f}{anchor_text}")
        pieces.append(f"final={final_score:.3f}")
        return "; ".join(pieces)

    @staticmethod
    def _label_match(a: str, b: str) -> float:
        aa = a.lower().replace("_", " ")
        bb = b.lower().replace("_", " ")
        return 1.0 if aa == bb or aa in bb or bb in aa else 0.05

    @staticmethod
    def _text_match(text: str, value: str) -> bool:
        text = text.lower().replace("_", " ")
        value = value.lower().replace("_", " ")
        return value == text or value in text or text in value

    @staticmethod
    def _canonical_attribute_value(value: str) -> str:
        value = str(value).lower().replace("_", " ").strip()
        aliases = {
            "grey": "gray",
            "wood": "wooden",
            "metallic": "metal",
            "cloth": "fabric",
            "circular": "round",
        }
        return aliases.get(value, value)

    @staticmethod
    def _shape_match(obj, value: str) -> float:
        size = np.maximum(np.asarray(obj.size, dtype=np.float32), 1e-6)
        x, y, z = float(size[0]), float(size[1]), float(size[2])
        horizontal = max(x, y)
        small_horizontal = min(x, y)
        volume_ratio = max(x, y, z) / max(min(x, y, z), 1e-6)
        value = GeometricReasoner._canonical_attribute_value(value)

        if value == "tall":
            return float(np.clip((z / (horizontal + 1e-6) - 0.8) / 1.2, 0.05, 1.0))
        if value in {"flat", "low"}:
            return float(np.clip((horizontal / (z + 1e-6) - 1.0) / 3.0, 0.05, 1.0))
        if value in {"wide", "long"}:
            return float(np.clip((horizontal / (small_horizontal + 1e-6) - 1.0) / 2.0, 0.05, 1.0))
        if value in {"narrow", "thin"}:
            return float(np.clip((horizontal / (small_horizontal + 1e-6) - 1.0) / 3.0, 0.05, 1.0))
        if value == "square":
            ratio_xy = max(x, y) / max(min(x, y), 1e-6)
            return float(np.clip(1.0 - (ratio_xy - 1.0), 0.05, 1.0))
        if value == "rectangular":
            return float(np.clip((volume_ratio - 1.0) / 2.0, 0.05, 1.0))
        if value == "round":
            label_text = obj.label.lower().replace("_", " ")
            return 1.0 if "round" in label_text or "circular" in label_text else 0.05
        return 0.05

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0:
            return x
        min_value = float(x.min())
        max_value = float(x.max())
        if max_value - min_value < 1e-6:
            return np.ones_like(x) if max_value > 0 else np.zeros_like(x)
        return (x - min_value) / (max_value - min_value + 1e-6)

    @staticmethod
    def _blend(x: np.ndarray, weight: float) -> np.ndarray:
        return np.power(np.clip(x, 1e-4, 1.0), float(weight))
