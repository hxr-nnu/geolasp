from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from geolasp.config import DEFAULT_LAYOUT_DIR, DEFAULT_SPATIALLM_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatiallm_root", default=str(DEFAULT_SPATIALLM_ROOT))
    parser.add_argument("--model_path", default="manycore-research/SpatialLM1.1-Qwen-0.5B")
    parser.add_argument("--point_cloud_dir", required=True)
    parser.add_argument("--output_dir", default=str(DEFAULT_LAYOUT_DIR))
    parser.add_argument("--detect_type", default="object", choices=["all", "arch", "object"])
    parser.add_argument("--category", nargs="*", default=[])
    args = parser.parse_args()

    root = Path(args.spatiallm_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(root / "inference.py"),
        "--point_cloud",
        args.point_cloud_dir,
        "--output",
        str(out),
        "--model_path",
        args.model_path,
        "--detect_type",
        args.detect_type,
    ]
    if args.category:
        cmd += ["--category", *args.category]
    subprocess.run(cmd, cwd=str(root), check=True)


if __name__ == "__main__":
    main()
