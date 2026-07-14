from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

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
    object_embedding_dim,
    object_feature_tensor,
    text_feature_tensor,
)
from geolasp.modeling.language_constraints import LLMConstraintParser, parse_constraints


def find_target_index(objects, target_id: int | None) -> int | None:
    if target_id is None:
        return None
    for i, obj in enumerate(objects):
        if int(obj.object_id) == int(target_id) or i == int(target_id):
            return i
    return None


def attach_point_features(
    extractor: SpatialLMPointFeatureExtractor | None,
    point_cloud_dir: str | None,
    cache_dir: str | None,
    scene_id: str,
    objects,
) -> bool:
    if extractor is None:
        return False
    attached, warnings = attach_scene_point_features(
        objects,
        scene_id=scene_id,
        point_cloud_dir=point_cloud_dir,
        cache_dir=cache_dir,
        extractor=extractor,
        overwrite_cache=getattr(extractor, "overwrite_cache", False),
    )
    for warning in warnings:
        print(f"warning: {warning}")
    return attached


def main() -> None:
    default_ckpt = DEFAULT_OUTPUT_DIR / "language_object_interaction.pt"
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
    parser.add_argument("--checkpoint", default=str(default_ckpt))
    parser.add_argument("--constraint_cache_dir", default=str(DEFAULT_CONSTRAINT_CACHE_DIR))
    parser.add_argument("--constraint_parser", choices=["auto", "llm", "rule"], default="auto")
    parser.add_argument("--llm_model", default="gpt-4o-2024-08-06")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    if args.use_spatiallm_point_features and not args.point_cloud_dir:
        print("warning: --point_cloud_dir is not set; train_alignment.py will only use existing point feature caches.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter = SpatialLMLayoutAdapter(args.spatiallm_root)
    samples = [s for s in load_grounding_json(args.annotation_json) if s.target_id is not None]
    llm_parser = LLMConstraintParser(model=args.llm_model, cache_dir=args.constraint_cache_dir)
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

    cached = []
    point_feature_scenes = 0
    for sample in samples:
        layout_file = Path(args.layout_dir) / f"{sample.scene_id}.txt"
        if not layout_file.exists():
            continue
        objects = adapter.load_layout(layout_file)
        if attach_point_features(point_extractor, args.point_cloud_dir, args.point_feature_cache_dir, sample.scene_id, objects):
            point_feature_scenes += 1
        target_idx = find_target_index(objects, sample.target_id)
        if not objects or target_idx is None:
            continue
        constraint = parse_constraints(
            sample.description,
            [o.label for o in objects],
            parser=llm_parser,
            mode=args.constraint_parser,
        )
        cached.append((sample, objects, target_idx, constraint))

    point_embedding_dim = max((object_embedding_dim(objects) for _sample, objects, _target_idx, _constraint in cached), default=0)
    object_dim = OBJECT_BASE_FEATURE_DIM + point_embedding_dim
    model = LanguageObjectInteractionModel(object_dim=object_dim).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for sample, objects, target_idx, _constraint in cached:
            text_features = text_feature_tensor(sample.description, device=device)
            object_features = object_feature_tensor(
                objects,
                device=device,
                include_embeddings=point_embedding_dim > 0,
                embedding_dim=point_embedding_dim,
            )
            logits = model(text_features, object_features)
            if logits.numel() == 0:
                continue
            loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target_idx], device=device))
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = sum(losses) / max(len(losses), 1)
        print(f"epoch={epoch + 1} loss={mean_loss:.4f} samples={len(losses)}")

    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_type": "LanguageObjectInteractionModel",
            "object_dim": model.object_dim,
            "point_feature_dim": point_embedding_dim,
            "use_point_features": bool(point_embedding_dim > 0),
            "point_features_requested": bool(args.use_spatiallm_point_features),
            "point_features_attached_scenes": point_feature_scenes,
            "point_feature_cache_dir": args.point_feature_cache_dir if args.use_spatiallm_point_features else None,
            "point_feature_model": args.spatiallm_model_path if args.use_spatiallm_point_features else None,
            "spatiallm_model_path": args.spatiallm_model_path if args.use_spatiallm_point_features else None,
            "num_cached_samples": len(cached),
        },
        args.checkpoint,
    )
    print(f"saved checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
