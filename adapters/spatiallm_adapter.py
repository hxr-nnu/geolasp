from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from geolasp.modeling.object_token import ObjectToken, from_spatiallm_box


def _ensure_import_root(root: Path) -> None:
    root_str = str(root)
    if root_str and root_str not in sys.path:
        sys.path.insert(0, root_str)


def _as_float_array(value: Any, length: int = 3) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.shape[0] < length:
        return None
    return arr[:length].astype(np.float32)


def _scalar_to_str(value: Any) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    values = arr.reshape(-1).tolist()
    return "" if not values else str(values[0])


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_loads_or_empty(value: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _split_call_args(text: str) -> list[str]:
    args: list[str] = []
    current: list[str] = []
    quote: str | None = None
    depth = 0
    for char in text:
        if quote is not None:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}" and depth > 0:
            depth -= 1
        if char == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        args.append("".join(current).strip())
    return args


def _clean_label(value: Any) -> str:
    label = str(value).strip()
    if len(label) >= 2 and label[0] == label[-1] and label[0] in {"'", '"'}:
        label = label[1:-1]
    return label or "unknown"


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


class SpatialLMLayoutAdapter:
    def __init__(self, spatiallm_root: str | os.PathLike | None = None):
        self.spatiallm_root = Path(spatiallm_root) if spatiallm_root else None
        self.Layout = None
        self._layout_import_error: Exception | None = None
        if self.spatiallm_root:
            _ensure_import_root(self.spatiallm_root)
        try:
            from spatiallm import Layout

            self.Layout = Layout
        except Exception as exc:
            self._layout_import_error = exc

    def load_layout(self, layout_file: str | os.PathLike) -> list[ObjectToken]:
        layout_text = Path(layout_file).read_text(encoding="utf-8")
        layout_error: Exception | None = None

        if self.Layout is not None:
            try:
                layout = self.Layout(layout_text)
                boxes = [box for box in layout.to_boxes() if box.get("class") == "bbox"]
                return [from_spatiallm_box(box, i) for i, box in enumerate(boxes)]
            except Exception as exc:
                layout_error = exc

        fallback_boxes = self._parse_layout_without_spatiallm(layout_text)
        if fallback_boxes is not None:
            return [from_spatiallm_box(box, i) for i, box in enumerate(fallback_boxes)]

        if layout_error is not None:
            raise RuntimeError(f"Failed to parse SpatialLM layout file {layout_file}: {layout_error}") from layout_error
        if self._layout_import_error is not None:
            raise RuntimeError(
                "Failed to import spatiallm.Layout and the layout file is not in a supported fallback format. "
                f"SpatialLM root: {self.spatiallm_root}. Original error: {self._layout_import_error}"
            ) from self._layout_import_error
        raise RuntimeError(f"Unsupported layout format: {layout_file}")

    @staticmethod
    def save_tokens_npz(tokens: list[ObjectToken], output_file: str | os.PathLike) -> None:
        save_object_feature_cache(tokens, output_file)

    @classmethod
    def _parse_layout_without_spatiallm(cls, layout_text: str) -> list[dict[str, Any]] | None:
        stripped = layout_text.strip()
        if not stripped:
            return []

        json_boxes = cls._parse_json_layout(stripped)
        if json_boxes is not None:
            return json_boxes

        boxes: list[dict[str, Any]] = []
        pattern = re.compile(r"^\s*bbox_(\d+)\s*=\s*Bbox\((.*)\)\s*$", re.IGNORECASE)
        for line in layout_text.splitlines():
            match = pattern.match(line)
            if not match:
                continue
            params = _split_call_args(match.group(2))
            if len(params) < 8:
                continue
            try:
                object_id = int(match.group(1)) + 3000
                label = _clean_label(params[0])
                center = [float(params[1]), float(params[2]), float(params[3])]
                yaw = float(params[4])
                scale = [abs(float(params[5])), abs(float(params[6])), abs(float(params[7]))]
            except ValueError:
                continue
            boxes.append(
                {
                    "id": object_id,
                    "class": "bbox",
                    "label": label,
                    "center": center,
                    "scale": scale,
                    "yaw": yaw,
                }
            )
        return boxes if boxes else None

    @classmethod
    def _parse_json_layout(cls, layout_text: str) -> list[dict[str, Any]] | None:
        try:
            raw = json.loads(layout_text)
        except Exception:
            return None

        candidates: Any = raw
        if isinstance(raw, dict):
            for key in ("objects", "boxes", "bboxes", "layout", "tokens"):
                if key in raw:
                    candidates = raw[key]
                    break
        if isinstance(candidates, dict):
            for key in ("objects", "boxes", "bboxes"):
                if key in candidates:
                    candidates = candidates[key]
                    break
        if not isinstance(candidates, list):
            return None

        boxes = []
        for index, item in enumerate(candidates):
            box = cls._box_from_mapping(item, index)
            if box is not None:
                boxes.append(box)
        return boxes

    @staticmethod
    def _box_from_mapping(item: Any, index: int) -> dict[str, Any] | None:
        if not isinstance(item, dict):
            return None
        center = _as_float_array(_first_present(item, "center", "position", "xyz"))
        size = _as_float_array(_first_present(item, "size", "scale", "extent"))
        bbox = _first_present(item, "bbox", "box")
        if (center is None or size is None) and bbox is not None:
            arr = _as_float_array(bbox, length=6)
            if arr is None:
                return None
            fmt = str(item.get("bbox_format") or item.get("box_format") or "").lower()
            if "xyzxyz" in fmt or "minmax" in fmt or "corner" in fmt:
                center = (arr[:3] + arr[3:6]) * 0.5
                size = np.abs(arr[3:6] - arr[:3])
            else:
                center = arr[:3]
                size = np.abs(arr[3:6])
        if center is None or size is None:
            return None

        label = _first_present(item, "label", "class_name", "category", "object_name", "class") or "unknown"
        object_id = item.get("id", item.get("object_id", index))
        try:
            object_id = int(object_id)
        except Exception:
            object_id = index
        return {
            "id": object_id,
            "class": "bbox",
            "label": _clean_label(label),
            "center": center,
            "scale": size,
            "yaw": float(item.get("yaw", item.get("angle_z", 0.0))),
            "score": float(item.get("score", item.get("confidence", 1.0))),
            "attributes": item.get("attributes") or {},
        }


@dataclass
class PointFeatureBundle:
    """SpatialLM point-token features aligned back to scene coordinates."""

    token_xyz: np.ndarray
    token_features: np.ndarray
    scene_embedding: np.ndarray
    min_extent: np.ndarray
    grid_size: float
    num_bins: int
    backbone: str
    source_file: str

    def save_npz(self, output_file: str | os.PathLike) -> None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            token_xyz=self.token_xyz.astype(np.float32),
            token_features=self.token_features.astype(np.float32),
            scene_embedding=self.scene_embedding.astype(np.float32),
            min_extent=self.min_extent.astype(np.float32),
            grid_size=np.asarray(self.grid_size, dtype=np.float32),
            num_bins=np.asarray(self.num_bins, dtype=np.int64),
            backbone=np.asarray(self.backbone),
            source_file=np.asarray(self.source_file),
        )

    @classmethod
    def load_npz(cls, input_file: str | os.PathLike) -> "PointFeatureBundle":
        data = np.load(input_file, allow_pickle=False)
        return cls(
            token_xyz=data["token_xyz"].astype(np.float32),
            token_features=data["token_features"].astype(np.float32),
            scene_embedding=data["scene_embedding"].astype(np.float32),
            min_extent=data["min_extent"].astype(np.float32),
            grid_size=float(data["grid_size"]),
            num_bins=int(data["num_bins"]),
            backbone=_scalar_to_str(data["backbone"]),
            source_file=_scalar_to_str(data["source_file"]),
        )


@dataclass
class ObjectFeatureCache:
    """Object-level GeoLaSP cache saved after bbox pooling."""

    object_ids: np.ndarray
    labels: np.ndarray
    centers: np.ndarray
    sizes: np.ndarray
    embeddings: np.ndarray
    point_stats: list[dict[str, Any]]
    scene_id: str = ""
    metadata: dict[str, Any] | None = None

    @property
    def feature_dim(self) -> int:
        if self.embeddings.ndim != 2:
            return 0
        return int(self.embeddings.shape[1])

    def save_npz(self, output_file: str | os.PathLike) -> None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        point_stats = np.asarray([_json_dumps(stats) for stats in self.point_stats], dtype=np.str_)
        np.savez_compressed(
            output_path,
            object_ids=self.object_ids.astype(np.int64),
            ids=self.object_ids.astype(np.int64),
            labels=self.labels.astype(np.str_),
            centers=self.centers.astype(np.float32),
            sizes=self.sizes.astype(np.float32),
            embeddings=self.embeddings.astype(np.float32),
            point_stats=point_stats,
            scene_id=np.asarray(self.scene_id),
            metadata_json=np.asarray(_json_dumps(self.metadata or {})),
        )

    @classmethod
    def load_npz(cls, input_file: str | os.PathLike) -> "ObjectFeatureCache":
        data = np.load(input_file, allow_pickle=False)
        files = set(data.files)
        id_key = "object_ids" if "object_ids" in files else "ids"
        required = {id_key, "labels", "centers", "sizes", "embeddings"}
        if not required.issubset(files):
            raise ValueError(f"{input_file} is not an object-level point feature cache")

        object_ids = data[id_key].astype(np.int64)
        embeddings = data["embeddings"].astype(np.float32)
        if embeddings.ndim == 1:
            if len(object_ids) == 0:
                embeddings = embeddings.reshape(0, 0)
            elif embeddings.size % len(object_ids) == 0:
                embeddings = embeddings.reshape(len(object_ids), -1)
            else:
                embeddings = embeddings.reshape(1, -1)

        raw_stats = data["point_stats"].astype(str).tolist() if "point_stats" in files else []
        if isinstance(raw_stats, str):
            raw_stats = [raw_stats]
        point_stats = [_json_loads_or_empty(item) for item in raw_stats]
        while len(point_stats) < len(object_ids):
            point_stats.append({})

        metadata = _json_loads_or_empty(data["metadata_json"]) if "metadata_json" in files else {}
        return cls(
            object_ids=object_ids,
            labels=data["labels"].astype(str),
            centers=data["centers"].astype(np.float32),
            sizes=data["sizes"].astype(np.float32),
            embeddings=embeddings,
            point_stats=point_stats,
            scene_id=_scalar_to_str(data["scene_id"]) if "scene_id" in files else "",
            metadata=metadata,
        )


def point_feature_cache_file(cache_dir: str | os.PathLike | None, scene_id: str) -> Path | None:
    if not cache_dir:
        return None
    return Path(cache_dir) / f"{scene_id}.npz"


def save_object_feature_cache(
    objects: list[ObjectToken],
    output_file: str | os.PathLike,
    scene_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> None:
    centers = np.stack([obj.center.astype(np.float32) for obj in objects], axis=0) if objects else np.zeros((0, 3), dtype=np.float32)
    sizes = np.stack([obj.size.astype(np.float32) for obj in objects], axis=0) if objects else np.zeros((0, 3), dtype=np.float32)
    labels = np.asarray([obj.label for obj in objects], dtype=np.str_)
    object_ids = np.asarray([obj.object_id for obj in objects], dtype=np.int64)
    feature_dim = _object_embedding_dim(objects)
    embeddings = np.zeros((len(objects), feature_dim), dtype=np.float32)
    point_stats: list[dict[str, Any]] = []

    for row, obj in enumerate(objects):
        embedding = getattr(obj, "embedding", None)
        if embedding is not None and feature_dim > 0:
            arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
            copy_dim = min(feature_dim, arr.shape[0])
            embeddings[row, :copy_dim] = arr[:copy_dim]
        stats = dict(getattr(obj, "point_stats", None) or {})
        stats.setdefault("point_feature_dim", int(feature_dim))
        stats.setdefault("point_feature_success", bool(embedding is not None and feature_dim > 0))
        point_stats.append(stats)

    ObjectFeatureCache(
        object_ids=object_ids,
        labels=labels,
        centers=centers,
        sizes=sizes,
        embeddings=embeddings,
        point_stats=point_stats,
        scene_id=scene_id,
        metadata=metadata or {},
    ).save_npz(output_file)


def attach_object_features_from_cache(objects: list[ObjectToken], cache_file: str | os.PathLike) -> bool:
    cache = ObjectFeatureCache.load_npz(cache_file)
    index_by_id = {int(object_id): idx for idx, object_id in enumerate(cache.object_ids.tolist())}
    attached = False

    for obj_index, obj in enumerate(objects):
        cache_index = index_by_id.get(int(obj.object_id))
        if cache_index is None and obj_index < len(cache.object_ids):
            cache_index = obj_index
        if cache_index is None or cache_index >= len(cache.embeddings):
            continue

        embedding = cache.embeddings[cache_index].astype(np.float32).reshape(-1)
        if cache.feature_dim > 0:
            obj.embedding = embedding.copy()
            attached = True
        stats = dict(cache.point_stats[cache_index] if cache_index < len(cache.point_stats) else {})
        stats["point_feature_cache_hit"] = True
        stats["point_feature_dim"] = int(embedding.shape[0])
        obj.point_stats = stats
    return attached


def attach_point_feature_bundle_to_objects(
    objects: list[ObjectToken],
    bundle: PointFeatureBundle,
    box_padding: float = 0.05,
) -> bool:
    if not objects:
        return False

    xyz = np.asarray(bundle.token_xyz, dtype=np.float32)
    features = np.asarray(bundle.token_features, dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    scene_embedding = np.asarray(bundle.scene_embedding, dtype=np.float32).reshape(-1)

    if xyz.size == 0 or features.size == 0:
        feature_dim = int(scene_embedding.shape[0])
        if feature_dim == 0 and features.ndim == 2:
            feature_dim = int(features.shape[1])
        for obj in objects:
            obj.embedding = np.zeros((feature_dim,), dtype=np.float32)
            obj.point_stats = {
                "point_feature_count": 0,
                "point_feature_bbox_count": 0,
                "point_feature_success": False,
                "point_feature_fallback": "empty_point_tokens",
                "point_feature_dim": feature_dim,
            }
        return feature_dim > 0

    attached = False
    for obj in objects:
        pooled, stats = _pool_single_object_feature(obj, xyz, features, scene_embedding, box_padding)
        obj.embedding = pooled.astype(np.float32)
        obj.point_stats = stats
        attached = attached or pooled.size > 0
    return attached


def attach_scene_point_features(
    objects: list[ObjectToken],
    scene_id: str,
    point_cloud_dir: str | os.PathLike | None,
    cache_dir: str | os.PathLike | None,
    extractor: "SpatialLMPointFeatureExtractor | None" = None,
    overwrite_cache: bool = False,
) -> tuple[bool, list[str]]:
    warnings: list[str] = []
    cache_file = point_feature_cache_file(cache_dir, scene_id)

    if cache_file is not None and cache_file.exists() and not overwrite_cache:
        try:
            attached = attach_object_features_from_cache(objects, cache_file)
            return attached, warnings
        except ValueError:
            try:
                bundle = PointFeatureBundle.load_npz(cache_file)
                attached = attach_point_feature_bundle_to_objects(objects, bundle, box_padding=getattr(extractor, "box_padding", 0.05))
                save_object_feature_cache(objects, cache_file, scene_id=scene_id, metadata={"converted_from": "point_token_cache"})
                return attached, warnings
            except Exception as exc:
                warnings.append(f"Existing cache is not readable for scene {scene_id}: {exc}")

    if extractor is None:
        warnings.append(f"Point feature cache miss for scene {scene_id}, but no SpatialLM extractor was provided.")
        return False, warnings

    point_cloud_file = find_point_cloud_file(point_cloud_dir, scene_id)
    if point_cloud_file is None:
        warnings.append(f"Point cloud not found for scene {scene_id}; using layout-only features.")
        return False, warnings

    try:
        bundle = extractor.extract(point_cloud_file, cache_file=None, use_cache=False)
        attached = extractor.attach_features_to_objects(objects, bundle)
    except Exception as exc:
        warnings.append(f"SpatialLM point feature extraction failed for scene {scene_id}: {exc}")
        return False, warnings

    if cache_file is not None:
        save_object_feature_cache(
            objects,
            cache_file,
            scene_id=scene_id,
            metadata={
                "spatiallm_model_path": extractor.model_path,
                "spatiallm_backbone": bundle.backbone,
                "source_file": bundle.source_file,
            },
        )
    return attached, warnings


def _object_embedding_dim(objects: list[ObjectToken]) -> int:
    dims = []
    for obj in objects:
        embedding = getattr(obj, "embedding", None)
        if embedding is not None:
            dims.append(int(np.asarray(embedding).reshape(-1).shape[0]))
    return max(dims, default=0)


def _pool_single_object_feature(
    obj: ObjectToken,
    xyz: np.ndarray,
    features: np.ndarray,
    scene_embedding: np.ndarray,
    box_padding: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    center = obj.center.astype(np.float32)
    half_size = np.maximum(obj.size.astype(np.float32) * 0.5 + float(box_padding), float(box_padding))
    lower = center - half_size
    upper = center + half_size
    bbox_mask = np.all((xyz >= lower) & (xyz <= upper), axis=1)
    mask = bbox_mask.copy()
    fallback_reason = "none"

    if not np.any(mask):
        distances = np.linalg.norm(xyz - center.reshape(1, 3), axis=1)
        radius = max(float(np.linalg.norm(half_size) * 1.5), float(box_padding))
        mask = distances <= radius
        fallback_reason = "expanded_radius"
        if not np.any(mask) and len(distances):
            nearest = np.argsort(distances)[: min(4, len(distances))]
            mask = np.zeros((len(distances),), dtype=bool)
            mask[nearest] = True
            fallback_reason = "nearest_tokens"
    else:
        distances = np.linalg.norm(xyz - center.reshape(1, 3), axis=1)

    if np.any(mask):
        selected = features[mask]
        pooled = selected.mean(axis=0)
        selected_xyz = xyz[mask]
        mean_distance = float(np.linalg.norm(selected_xyz - center.reshape(1, 3), axis=1).mean())
        count = int(selected.shape[0])
        success = True
    else:
        feature_dim = int(scene_embedding.shape[0])
        if feature_dim == 0 and features.ndim == 2:
            feature_dim = int(features.shape[1])
        pooled = scene_embedding if scene_embedding.shape[0] == feature_dim else np.zeros((feature_dim,), dtype=np.float32)
        mean_distance = 0.0
        count = 0
        success = False
        fallback_reason = "scene_embedding" if feature_dim > 0 else "zero_vector"

    return pooled.astype(np.float32), {
        "point_feature_count": count,
        "point_feature_bbox_count": int(bbox_mask.sum()),
        "point_feature_mean_distance": mean_distance,
        "point_feature_success": success,
        "point_feature_fallback": fallback_reason,
        "point_feature_dim": int(pooled.reshape(-1).shape[0]),
    }


def find_point_cloud_file(point_cloud_root: str | os.PathLike | None, scene_id: str) -> Path | None:
    if point_cloud_root is None:
        return None

    root = Path(point_cloud_root)
    if root.is_file():
        return root

    candidates = [
        root / f"{scene_id}.ply",
        root / "pcd" / f"{scene_id}.ply",
        root / scene_id / f"{scene_id}.ply",
        root / scene_id / f"{scene_id}_vh_clean_2.ply",
        root / scene_id / f"{scene_id}_vh_clean_2.labels.ply",
        root / "scans" / scene_id / f"{scene_id}.ply",
        root / "scans" / scene_id / f"{scene_id}_vh_clean_2.ply",
        root / "scans" / scene_id / f"{scene_id}_vh_clean_2.labels.ply",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


class SpatialLMPointFeatureExtractor:
    """Frozen SpatialLM point-token extractor for GeoLaSP object features."""

    def __init__(
        self,
        spatiallm_root: str | os.PathLike,
        model_path: str = "manycore-research/SpatialLM1.1-Qwen-0.5B",
        device: str | None = None,
        inference_dtype: str = "auto",
        cleanup: bool = True,
        box_padding: float = 0.05,
    ):
        self.spatiallm_root = Path(spatiallm_root)
        self.model_path = model_path
        self.device_name = device
        self.inference_dtype = inference_dtype
        self.cleanup = cleanup
        self.box_padding = float(box_padding)
        self._model = None
        self._torch = None

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def torch(self):
        if self._torch is None:
            import torch

            self._torch = torch
        return self._torch

    def extract(
        self,
        point_cloud_file: str | os.PathLike,
        cache_file: str | os.PathLike | None = None,
        use_cache: bool = True,
    ) -> PointFeatureBundle:
        if cache_file is not None and use_cache and Path(cache_file).exists():
            return PointFeatureBundle.load_npz(cache_file)

        point_cloud_tensor, min_extent, grid_size, num_bins = self._preprocess_point_cloud_file(point_cloud_file)
        with self.torch.inference_mode():
            token_features, token_xyz = self._forward_point_cloud_with_xyz(point_cloud_tensor, min_extent, grid_size, num_bins)

        if token_features.size:
            scene_embedding = token_features.mean(axis=0).astype(np.float32)
        else:
            scene_embedding = np.zeros((0,), dtype=np.float32)

        bundle = PointFeatureBundle(
            token_xyz=token_xyz.astype(np.float32),
            token_features=token_features.astype(np.float32),
            scene_embedding=scene_embedding,
            min_extent=min_extent.astype(np.float32),
            grid_size=float(grid_size),
            num_bins=int(num_bins),
            backbone=self._backbone_name(),
            source_file=str(point_cloud_file),
        )
        if cache_file is not None:
            bundle.save_npz(cache_file)
        return bundle

    def attach_features_to_objects(self, objects: list[ObjectToken], bundle: PointFeatureBundle) -> bool:
        return attach_point_feature_bundle_to_objects(objects, bundle, box_padding=self.box_padding)

    def _load_model(self) -> None:
        try:
            _ensure_import_root(self.spatiallm_root)
            import spatiallm  # noqa: F401 - registers SpatialLM model classes with transformers.
            from transformers import AutoModelForCausalLM

            torch = self.torch
            device = self.device_name or ("cuda" if torch.cuda.is_available() else "cpu")
            dtype_name = self.inference_dtype
            if dtype_name == "auto":
                dtype_name = "bfloat16" if device.startswith("cuda") else "float32"
            if not hasattr(torch, dtype_name):
                raise ValueError(f"Unsupported torch dtype '{dtype_name}'. Use auto, float32, float16, or bfloat16.")

            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=getattr(torch, dtype_name),
                trust_remote_code=True,
            )
            model.to(device)
            if hasattr(model, "set_point_backbone_dtype"):
                model.set_point_backbone_dtype(torch.float32)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self.device_name = device
            self._model = model
        except Exception as exc:
            raise RuntimeError(
                "Failed to load SpatialLM for frozen point feature extraction. "
                f"model_path={self.model_path!r}, spatiallm_root={str(self.spatiallm_root)!r}. "
                "Install SpatialLM and its point-backbone dependencies, or provide an existing "
                "--point_feature_cache_dir for layout-only/cache-only execution. "
                f"Original error: {exc}"
            ) from exc

    def _preprocess_point_cloud_file(self, point_cloud_file: str | os.PathLike):
        _ensure_import_root(self.spatiallm_root)
        from spatiallm import Layout
        from spatiallm.pcd import Compose, cleanup_pcd, get_points_and_colors, load_o3d_pcd

        num_bins = int(self.model.config.point_config["num_bins"])
        grid_size = float(Layout.get_grid_size(num_bins))
        pcd = load_o3d_pcd(str(point_cloud_file))
        if self.cleanup:
            pcd = cleanup_pcd(pcd, voxel_size=grid_size)
        points, colors = get_points_and_colors(pcd)
        if len(points) == 0:
            min_extent = np.zeros((3,), dtype=np.float32)
            point_cloud = np.zeros((0, 9), dtype=np.float32)
            return self.torch.as_tensor(point_cloud), min_extent, grid_size, num_bins

        min_extent = np.min(points, axis=0).astype(np.float32)
        transform = Compose(
            [
                dict(type="PositiveShift"),
                dict(type="NormalizeColor"),
                dict(
                    type="GridSample",
                    grid_size=grid_size,
                    hash_type="fnv",
                    mode="test",
                    keys=("coord", "color"),
                    return_grid_coord=True,
                    max_grid_coord=num_bins,
                ),
            ]
        )
        point_cloud = transform({"name": "pcd", "coord": points.copy(), "color": colors.copy()})
        point_cloud = np.concatenate(
            [point_cloud["grid_coord"], point_cloud["coord"], point_cloud["color"]],
            axis=1,
        )
        return self.torch.as_tensor(point_cloud), min_extent, grid_size, num_bins

    def _forward_point_cloud_with_xyz(
        self,
        point_cloud,
        min_extent: np.ndarray,
        grid_size: float,
        num_bins: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        model = self.model
        if point_cloud.numel() == 0:
            return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

        model.point_backbone.to(torch.float32)
        nan_mask = torch.isnan(point_cloud).any(dim=1)
        point_cloud = point_cloud[~nan_mask]
        coords = point_cloud[:, :3].int()
        feats = point_cloud[:, 3:].float()
        backbone = self._backbone_name()

        if backbone == "sonata":
            context, token_xyz = self._forward_sonata_with_xyz(coords, feats, num_bins)
        elif backbone == "scenescript":
            context, token_xyz = self._forward_scenescript_with_xyz(coords, feats, grid_size, num_bins)
        else:
            raise ValueError(f"Unknown SpatialLM point backbone: {backbone}")

        project_dtype = self._projector_dtype()
        context = model.point_proj(context.to(project_dtype))
        token_features = context.detach().float().cpu().numpy()
        token_xyz = token_xyz.detach().float().cpu().numpy() + min_extent.reshape(1, 3)
        return token_features, token_xyz

    def _forward_sonata_with_xyz(self, coords, feats, num_bins: int):
        from spatiallm.model.sonata_encoder import Point, fourier_encode_vector

        torch = self.torch
        device = self.model.device
        backbone = self.model.point_backbone
        input_dict = {
            "coord": feats[:, :3].to(device),
            "grid_coord": coords.to(device),
            "feat": feats.to(device),
            "batch": torch.zeros(coords.shape[0], dtype=torch.long, device=device),
        }
        point = Point(input_dict)
        point = backbone.embedding(point)
        point.serialization(order=backbone.order, shuffle_orders=backbone.shuffle_orders)
        point.sparsify()
        point = backbone.enc(point)
        context = point["sparse_conv_feat"].features
        token_xyz = point["coord"]

        if backbone.enable_fourier_encode:
            point_coords = point["grid_coord"]
            coords_normalised = point_coords / (backbone.reduced_grid_size - 1)
            encoded_coords = fourier_encode_vector(coords_normalised)
            context = torch.cat([context, encoded_coords], dim=-1)
            context = backbone.input_proj(context)

        return context, token_xyz

    def _forward_scenescript_with_xyz(self, coords, feats, grid_size: float, num_bins: int):
        import torchsparse
        from torchsparse.utils.collate import sparse_collate
        from spatiallm.model.scenescript_encoder import fourier_encode_vector, vox_to_sequence

        torch = self.torch
        device = self.model.device
        backbone = self.model.point_backbone
        pc_sparse_tensor = torchsparse.SparseTensor(coords=coords, feats=feats)
        pc_sparse_tensor = sparse_collate([pc_sparse_tensor]).to(device)
        outputs = backbone.sparse_resnet(pc_sparse_tensor)
        outputs = vox_to_sequence(outputs)
        context = outputs["seq"]
        context_mask = outputs["mask"]
        reduced_coords = outputs["coords"]
        coords_normalised = reduced_coords / (backbone.reduced_grid_size - 1)
        encoded_coords = fourier_encode_vector(coords_normalised)
        context = torch.cat([context, encoded_coords], dim=-1)
        context = backbone.input_proj(context)
        context = context + backbone.extra_embedding.view(1, 1, -1)

        valid = ~context_mask[0]
        context = context[0, valid]
        reduced_coords = reduced_coords[0, valid].float()
        res_reduction = max(int(round(float(num_bins) / float(backbone.reduced_grid_size))), 1)
        token_xyz = (reduced_coords + 0.5) * float(grid_size * res_reduction)
        return context, token_xyz

    def _backbone_name(self) -> str:
        backbone = getattr(self.model, "point_backbone_type", None)
        value = getattr(backbone, "value", backbone)
        return str(value).split(".")[-1].lower()

    def _projector_dtype(self):
        point_proj = self.model.point_proj
        if hasattr(point_proj, "parameters"):
            params = list(point_proj.parameters())
            if params:
                return params[0].dtype
        return self.torch.float32
