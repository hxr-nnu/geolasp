from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from geolasp.adapters.spatiallm_adapter import (
    SpatialLMLayoutAdapter,
    SpatialLMPointFeatureExtractor,
    attach_scene_point_features,
)
from geolasp.config import (
    DEFAULT_CONSTRAINT_CACHE_DIR,
    DEFAULT_LAYOUT_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POINT_FEATURE_DIR,
    DEFAULT_SPATIALLM_ROOT,
)
from geolasp.data.scanrefer_json import load_grounding_json
from geolasp.modeling.alignment_model import (
    LanguageObjectInteractionModel,
    OBJECT_BASE_FEATURE_DIM,
    OBJECT_FEATURE_DIM,
    object_embedding_dim,
    object_feature_tensor,
    text_feature_tensor,
)
from geolasp.modeling.language_constraints import LLMConstraintParser, parse_constraints
from geolasp.modeling.reasoner import GeometricReasoner
from geolasp.modeling.scene_graph import GeometricSceneGraph


def load_interaction_model(checkpoint: str | None, device: str) -> LanguageObjectInteractionModel | None:
    if not checkpoint:
        return None
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        return None
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload.get("model", payload) if isinstance(payload, dict) else payload
    object_dim = OBJECT_FEATURE_DIM
    if isinstance(payload, dict):
        object_dim = int(payload.get("object_dim", object_dim))
    if isinstance(state_dict, dict) and "object_proj.0.weight" in state_dict:
        object_dim = int(state_dict["object_proj.0.weight"].shape[1])
    model = LanguageObjectInteractionModel(object_dim=object_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    scores = scores.astype(np.float32)
    scores = scores - scores.min()
    return scores / (scores.max() + 1e-6)


def attach_point_features(
    extractor: SpatialLMPointFeatureExtractor | None,
    point_cloud_dir: str | None,
    cache_dir: str | None,
    scene_id: str,
    objects,
) -> tuple[bool, list[str]]:
    if extractor is None:
        return False, []
    return attach_scene_point_features(
        objects,
        scene_id=scene_id,
        point_cloud_dir=point_cloud_dir,
        cache_dir=cache_dir,
        extractor=extractor,
        overwrite_cache=getattr(extractor, "overwrite_cache", False),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatiallm_root", default=str(DEFAULT_SPATIALLM_ROOT))
    parser.add_argument("--annotation_json", required=True)
    parser.add_argument("--layout_dir", default=str(DEFAULT_LAYOUT_DIR))
    parser.add_argument("--use_spatiallm_point_features", action="store_true")
    parser.add_argument("--point_cloud_dir")
    parser.add_argument("--point_feature_cache_dir", default=str(DEFAULT_POINT_FEATURE_DIR))
    parser.add_argument("--spatiallm_model_path", default="manycore-research/SpatialLM1.1-Qwen-0.5B")
    parser.add_argument("--spatiallm_device")
    parser.add_argument("--spatiallm_inference_dtype", default="auto")
    parser.add_argument("--no_point_cloud_cleanup", action="store_true")
    parser.add_argument("--overwrite_point_feature_cache", action="store_true")
    parser.add_argument("--constraint_cache_dir", default=str(DEFAULT_CONSTRAINT_CACHE_DIR))
    parser.add_argument("--constraint_parser", choices=["auto", "llm", "rule"], default="auto")
    parser.add_argument("--llm_model", default="gpt-4o-2024-08-06")
    parser.add_argument("--interaction_checkpoint")
    parser.add_argument("--fusion_alpha", type=float, default=0.5, help="final = alpha * neural + (1-alpha) * geometry")
    parser.add_argument("--scoring_mode", choices=["auto", "geometry", "neural", "fusion"], default="auto")
    parser.add_argument("--output_json", default=str(DEFAULT_OUTPUT_DIR / "predictions.json"))
    args = parser.parse_args()
    if args.use_spatiallm_point_features and not args.point_cloud_dir:
        print("warning: --point_cloud_dir is not set; eval_grounding.py will only use existing point feature caches.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter = SpatialLMLayoutAdapter(args.spatiallm_root)
    samples = load_grounding_json(args.annotation_json)
    llm_parser = LLMConstraintParser(model=args.llm_model, cache_dir=args.constraint_cache_dir)
    interaction_model = load_interaction_model(args.interaction_checkpoint, device)
    point_embedding_dim = 0
    if interaction_model is not None:
        point_embedding_dim = max(int(interaction_model.object_dim) - OBJECT_BASE_FEATURE_DIM, 0)
    point_extractor = (
        SpatialLMPointFeatureExtractor(
            args.spatiallm_root,
            model_path=args.spatiallm_model_path,
            device=args.spatiallm_device,
            inference_dtype=args.spatiallm_inference_dtype,
            cleanup=not args.no_point_cloud_cleanup,
        )
        if args.use_spatiallm_point_features
        else None
    )
    if point_extractor is not None:
        point_extractor.overwrite_cache = bool(args.overwrite_point_feature_cache)
    results = []
    correct = 0
    total = 0

    for sample in samples:
        layout_file = Path(args.layout_dir) / f"{sample.scene_id}.txt"
        if not layout_file.exists():
            continue
        objects = adapter.load_layout(layout_file)
        point_features_attached, point_warnings = attach_point_features(
            point_extractor,
            args.point_cloud_dir,
            args.point_feature_cache_dir,
            sample.scene_id,
            objects,
        )
        for warning in point_warnings:
            print(f"warning: {warning}")
        scene_point_feature_dim = object_embedding_dim(objects)
        graph = GeometricSceneGraph(objects)
        constraint = parse_constraints(
            sample.description,
            [o.label for o in objects],
            parser=llm_parser,
            mode=args.constraint_parser,
        )
        geometry_scores = GeometricReasoner(graph).score(constraint)
        final_scores = geometry_scores.copy()
        neural_scores = None
        item_warnings = list(point_warnings)
        effective_scoring_mode = args.scoring_mode

        neural_available = interaction_model is not None and objects
        if args.scoring_mode == "auto":
            effective_scoring_mode = "fusion" if neural_available else "geometry"
        if effective_scoring_mode in {"neural", "fusion"} and not neural_available:
            item_warnings.append("Neural scoring requested but no interaction checkpoint was loaded; using geometry scores.")
            effective_scoring_mode = "geometry"

        if neural_available and effective_scoring_mode != "geometry":
            with torch.inference_mode():
                text_features = text_feature_tensor(sample.description, device=device)
                object_features = object_feature_tensor(
                    objects,
                    device=device,
                    include_embeddings=point_embedding_dim > 0,
                    embedding_dim=point_embedding_dim,
                )
                logits = interaction_model(text_features, object_features)
                neural_scores = torch.softmax(logits, dim=0).detach().cpu().numpy()
            if effective_scoring_mode == "neural":
                final_scores = normalize_scores(neural_scores)
            else:
                alpha = min(max(args.fusion_alpha, 0.0), 1.0)
                final_scores = alpha * normalize_scores(neural_scores) + (1.0 - alpha) * normalize_scores(geometry_scores)

        pred_idx = int(final_scores.argmax()) if len(final_scores) else -1
        pred_obj_id = int(objects[pred_idx].object_id) if pred_idx >= 0 else -1
        reported_point_dim = scene_point_feature_dim if scene_point_feature_dim > 0 else point_embedding_dim
        item = {
            "scene_id": sample.scene_id,
            "description": sample.description,
            "constraint_source": constraint.source,
            "constraint_json": constraint.raw_json,
            "pred_object_id": pred_obj_id,
            "pred_index": pred_idx,
            "geometry_scores": geometry_scores.tolist(),
            "neural_scores": None if neural_scores is None else neural_scores.tolist(),
            "final_scores": final_scores.tolist(),
            "point_features_attached": point_features_attached,
            "point_features_requested": bool(args.use_spatiallm_point_features),
            "point_feature_dim": reported_point_dim,
            "scoring_mode": effective_scoring_mode,
            "warnings": item_warnings,
        }
        if sample.target_id is not None:
            item["target_id"] = sample.target_id
            correct += int(pred_obj_id == sample.target_id or pred_idx == sample.target_id)
            total += 1
        results.append(item)

    output = {"accuracy": None if total == 0 else correct / total, "num_eval": total, "num_pred": len(results), "results": results}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"accuracy": output["accuracy"], "num_eval": total, "num_pred": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
