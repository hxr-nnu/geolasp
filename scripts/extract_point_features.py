from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from geolasp.adapters.spatiallm_adapter import (
    SpatialLMLayoutAdapter,
    SpatialLMPointFeatureExtractor,
    attach_scene_point_features,
)
from geolasp.config import DEFAULT_LAYOUT_DIR, DEFAULT_POINT_FEATURE_DIR, DEFAULT_SPATIALLM_ROOT
from geolasp.data.scanrefer_json import load_grounding_json


def collect_scene_ids(layout_dir: Path, annotation_json: str | None, scene_ids: list[str] | None) -> list[str]:
    if scene_ids:
        return sorted(set(scene_ids))
    if annotation_json:
        return sorted({sample.scene_id for sample in load_grounding_json(annotation_json)})
    return sorted(path.stem for path in layout_dir.glob("*.txt"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen SpatialLM object-level point features for GeoLaSP.")
    parser.add_argument("--spatiallm_root", default=str(DEFAULT_SPATIALLM_ROOT))
    parser.add_argument("--spatiallm_model_path", default="manycore-research/SpatialLM1.1-Qwen-0.5B")
    parser.add_argument("--layout_dir", default=str(DEFAULT_LAYOUT_DIR))
    parser.add_argument("--point_cloud_dir", required=True)
    parser.add_argument("--point_feature_cache_dir", default=str(DEFAULT_POINT_FEATURE_DIR))
    parser.add_argument("--annotation_json")
    parser.add_argument("--scene_id", nargs="*")
    parser.add_argument("--spatiallm_device")
    parser.add_argument("--spatiallm_inference_dtype", default="auto")
    parser.add_argument("--no_point_cloud_cleanup", action="store_true")
    parser.add_argument("--overwrite_point_feature_cache", action="store_true")
    args = parser.parse_args()

    layout_dir = Path(args.layout_dir)
    scene_ids = collect_scene_ids(layout_dir, args.annotation_json, args.scene_id)
    adapter = SpatialLMLayoutAdapter(args.spatiallm_root)
    extractor = SpatialLMPointFeatureExtractor(
        args.spatiallm_root,
        model_path=args.spatiallm_model_path,
        device=args.spatiallm_device,
        inference_dtype=args.spatiallm_inference_dtype,
        cleanup=not args.no_point_cloud_cleanup,
    )

    attached_count = 0
    skipped_missing_layout = 0
    warnings: list[str] = []
    for scene_id in scene_ids:
        layout_file = layout_dir / f"{scene_id}.txt"
        if not layout_file.exists():
            skipped_missing_layout += 1
            warnings.append(f"Layout file not found for scene {scene_id}: {layout_file}")
            continue
        objects = adapter.load_layout(layout_file)
        attached, scene_warnings = attach_scene_point_features(
            objects,
            scene_id=scene_id,
            point_cloud_dir=args.point_cloud_dir,
            cache_dir=args.point_feature_cache_dir,
            extractor=extractor,
            overwrite_cache=args.overwrite_point_feature_cache,
        )
        warnings.extend(scene_warnings)
        attached_count += int(attached)
        status = "cached" if attached else "layout-only"
        print(f"{scene_id}: {status} objects={len(objects)}")

    summary = {
        "num_scenes": len(scene_ids),
        "num_attached": attached_count,
        "num_missing_layout": skipped_missing_layout,
        "point_feature_cache_dir": args.point_feature_cache_dir,
        "warnings": warnings,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
