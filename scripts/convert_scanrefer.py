from __future__ import annotations

import argparse
import json
from pathlib import Path

from geolasp.config import DEFAULT_PROCESSED_ROOT, DEFAULT_SCANREFER_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", default=str(DEFAULT_SCANREFER_ROOT / "ScanRefer_filtered_train.json"), help="Official ScanRefer json file.")
    parser.add_argument("--output_json", default=str(DEFAULT_PROCESSED_ROOT / "scanrefer_train.json"), help="GeoLaSP minimal annotation json.")
    args = parser.parse_args()

    data = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    converted = []
    for item in data:
        converted.append(
            {
                "scene_id": item["scene_id"],
                "description": item["description"],
                "target_id": int(item["object_id"]),
                "object_name": item.get("object_name"),
                "ann_id": item.get("ann_id"),
            }
        )
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"converted={len(converted)} output={args.output_json}")


if __name__ == "__main__":
    main()
