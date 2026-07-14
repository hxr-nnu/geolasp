from __future__ import annotations

import argparse
import json
from pathlib import Path

from geolasp.adapters.spatiallm_adapter import SpatialLMLayoutAdapter
from geolasp.config import DEFAULT_CONSTRAINT_CACHE_DIR, DEFAULT_LAYOUT_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_SPATIALLM_ROOT
from geolasp.data.scanrefer_json import GroundingSample, load_grounding_json
from geolasp.modeling.language_constraints import LLMConstraintParser, parse_constraints
from geolasp.modeling.reasoner import GeometricReasoner
from geolasp.modeling.scene_graph import GeometricSceneGraph


def select_sample(args: argparse.Namespace) -> GroundingSample:
    if args.scene_id and args.description:
        return GroundingSample(scene_id=args.scene_id, description=args.description)
    if not args.annotation_json:
        raise SystemExit("Either provide --scene_id and --description, or provide --annotation_json.")

    samples = load_grounding_json(args.annotation_json)
    if args.scene_id:
        samples = [sample for sample in samples if sample.scene_id == args.scene_id]
    if not samples:
        raise SystemExit("No matching sample found.")
    if args.sample_index < 0 or args.sample_index >= len(samples):
        raise SystemExit(f"--sample_index must be in [0, {len(samples) - 1}].")
    return samples[args.sample_index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatiallm_root", default=str(DEFAULT_SPATIALLM_ROOT))
    parser.add_argument("--layout_dir", default=str(DEFAULT_LAYOUT_DIR))
    parser.add_argument("--annotation_json")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--scene_id")
    parser.add_argument("--description")
    parser.add_argument("--constraint_cache_dir", default=str(DEFAULT_CONSTRAINT_CACHE_DIR))
    parser.add_argument("--constraint_parser", choices=["auto", "llm", "rule"], default="auto")
    parser.add_argument("--llm_model", default="gpt-4o-2024-08-06")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_k_anchors", type=int, default=3)
    parser.add_argument("--output_json", default=str(DEFAULT_OUTPUT_DIR / "reasoner_explanation.json"))
    args = parser.parse_args()

    sample = select_sample(args)
    layout_file = Path(args.layout_dir) / f"{sample.scene_id}.txt"
    if not layout_file.exists():
        raise SystemExit(f"Layout file not found: {layout_file}")

    adapter = SpatialLMLayoutAdapter(args.spatiallm_root)
    objects = adapter.load_layout(layout_file)
    graph = GeometricSceneGraph(objects)
    llm_parser = LLMConstraintParser(model=args.llm_model, cache_dir=args.constraint_cache_dir)
    constraint = parse_constraints(
        sample.description,
        [obj.label for obj in objects],
        parser=llm_parser,
        mode=args.constraint_parser,
    )

    reasoner = GeometricReasoner(graph)
    trace = reasoner.evaluate(constraint)
    candidates = [reasoner.explain_candidate(trace, i, top_k_anchors=args.top_k_anchors) for i in range(len(objects))]
    ranked_candidates = sorted(candidates, key=lambda item: item["final_score"], reverse=True)
    for rank, item in enumerate(ranked_candidates, start=1):
        item["rank"] = rank

    output = {
        "scene_id": sample.scene_id,
        "description": sample.description,
        "constraint_source": constraint.source,
        "constraint_json": constraint.raw_json,
        "num_candidates": len(candidates),
        "top_k": args.top_k,
        "candidates": ranked_candidates,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "scene_id": sample.scene_id,
        "description": sample.description,
        "output_json": str(output_path),
        "top_candidates": [
            {
                "rank": item["rank"],
                "object_id": item["object_id"],
                "label": item["label"],
                "final_score": item["final_score"],
                "explanation": item["explanation"],
            }
            for item in ranked_candidates[: max(args.top_k, 0)]
        ],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
