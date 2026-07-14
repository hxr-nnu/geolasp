from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geolasp.modeling.object_token import ObjectToken


WALL_LABEL_HINTS = ("wall",)
FLOOR_LABEL_HINTS = ("floor", "ground")
SUPPORT_LABEL_HINTS = ("table", "shelf", "desk", "counter", "cabinet", "stand")
CONTAINER_LABEL_HINTS = ("cabinet", "box", "drawer", "bin", "basket", "shelf", "closet", "wardrobe")


@dataclass
class Edge:
    src: int
    dst: int
    relation: str
    score: float


class GeometricSceneGraph:
    def __init__(self, objects: list[ObjectToken]):
        self.objects = objects
        self.edges: list[Edge] = []
        self._build_edges()

    def centers(self) -> np.ndarray:
        if not self.objects:
            return np.zeros((0, 3), dtype=np.float32)
        return np.stack([o.center for o in self.objects], axis=0)

    def sizes(self) -> np.ndarray:
        if not self.objects:
            return np.zeros((0, 3), dtype=np.float32)
        return np.stack([o.size for o in self.objects], axis=0)

    def box_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.objects:
            empty = np.zeros((0, 3), dtype=np.float32)
            return empty, empty
        centers = self.centers()
        half_sizes = np.maximum(self.sizes(), 1e-6) * 0.5
        return centers - half_sizes, centers + half_sizes

    def embeddings(self) -> np.ndarray:
        if not self.objects:
            return np.zeros((0, 0), dtype=np.float32)
        dims = [
            int(np.asarray(obj.embedding).reshape(-1).shape[0])
            for obj in self.objects
            if obj.embedding is not None
        ]
        embedding_dim = max(dims, default=0)
        if embedding_dim == 0:
            return np.zeros((len(self.objects), 0), dtype=np.float32)

        embeddings = np.zeros((len(self.objects), embedding_dim), dtype=np.float32)
        for i, obj in enumerate(self.objects):
            if obj.embedding is None:
                continue
            vec = np.asarray(obj.embedding, dtype=np.float32).reshape(-1)
            copy_dim = min(embedding_dim, vec.shape[0])
            embeddings[i, :copy_dim] = vec[:copy_dim]
        return embeddings

    def point_feature_counts(self) -> np.ndarray:
        counts = []
        for obj in self.objects:
            stats = obj.point_stats or {}
            counts.append(float(stats.get("point_feature_count", 0.0)))
        return np.asarray(counts, dtype=np.float32)

    def _build_edges(self) -> None:
        centers = self.centers()
        if len(centers) == 0:
            return
        sizes = self.sizes()
        avg_size = np.linalg.norm(sizes, axis=1).mean() + 1e-6
        vertical_scale = max(float(sizes[:, 2].mean()), 1e-6)
        for i, src in enumerate(self.objects):
            for j, dst in enumerate(self.objects):
                if i == j:
                    continue
                diff = dst.center - src.center
                dist = float(np.linalg.norm(diff))
                near = float(np.exp(-dist / avg_size))
                self.edges.append(Edge(i, j, "near", near))
                self.edges.append(Edge(i, j, "far", 1.0 - near))
                if abs(diff[0]) > abs(diff[1]):
                    self.edges.append(Edge(i, j, "left" if diff[0] > 0 else "right", min(abs(diff[0]) / avg_size, 1.0)))
                else:
                    self.edges.append(Edge(i, j, "front" if diff[1] > 0 else "behind", min(abs(diff[1]) / avg_size, 1.0)))
                dz = diff[2]
                if abs(dz) > 0.25 * avg_size:
                    self.edges.append(Edge(i, j, "below" if dz > 0 else "above", min(abs(dz) / avg_size, 1.0)))
                self._build_size_edges(i, j)
                self._build_vertical_order_edges(i, j, vertical_scale)
                self._build_wall_floor_pair_edges(i, j)
                self._build_support_containment_edges(i, j)
                self._build_contact_edges(i, j)
        self._build_between_edges()
        self._build_nearest_edges()
        self._build_unary_edges()
        self._build_room_edges()
        self._build_visibility_edges()
        self._build_feature_edges()

    def _add_edge(self, src: int, dst: int, relation: str, score: float) -> None:
        score = float(np.clip(score, 0.0, 1.0))
        if score > 0.0:
            self.edges.append(Edge(src, dst, relation, score))

    def _build_size_edges(self, i: int, j: int) -> None:
        sizes = np.maximum(self.sizes(), 1e-6)
        volumes = np.prod(sizes, axis=1)
        heights = sizes[:, 2]
        self._add_edge(i, j, "larger", self._ratio_score(volumes[i], volumes[j]))
        self._add_edge(i, j, "smaller", self._ratio_score(volumes[j], volumes[i]))
        self._add_edge(i, j, "taller", self._ratio_score(heights[i], heights[j]))

    def _build_vertical_order_edges(self, i: int, j: int, vertical_scale: float) -> None:
        centers = self.centers()
        delta = centers[j, 2] - centers[i, 2]
        self._add_edge(i, j, "lower", delta / (vertical_scale + 1e-6))

    def _build_wall_floor_pair_edges(self, i: int, j: int) -> None:
        mins, maxs = self.box_bounds()
        avg_size = max(float(np.linalg.norm(self.sizes(), axis=1).mean()), 1e-6)
        contact_scale = max(avg_size * 0.15, 0.05)
        dst_label = self.objects[j].label.lower().replace("_", " ")

        if self._label_has_any(dst_label, WALL_LABEL_HINTS):
            gap_xy = self._box_gap_xy(mins[i], maxs[i], mins[j], maxs[j])
            overlap_z = self._interval_overlap_ratio(mins[i, 2], maxs[i, 2], mins[j, 2], maxs[j, 2])
            self._add_edge(i, j, "against wall", np.exp(-((gap_xy / contact_scale) ** 2)) * overlap_z)

        if self._label_has_any(dst_label, FLOOR_LABEL_HINTS):
            vertical_gap = abs(mins[i, 2] - maxs[j, 2])
            overlap_xy = self._box_overlap_ratio_xy(mins[i], maxs[i], mins[j], maxs[j])
            self._add_edge(i, j, "on the floor", np.exp(-((vertical_gap / contact_scale) ** 2)) * overlap_xy)

    def _build_support_containment_edges(self, i: int, j: int) -> None:
        mins, maxs = self.box_bounds()
        sizes = self.sizes()
        avg_size = max(float(np.linalg.norm(sizes, axis=1).mean()), 1e-6)
        contact_scale = max(avg_size * 0.12, 0.05)
        dst_label = self.objects[j].label.lower().replace("_", " ")

        vertical_gap = abs(float(mins[i, 2] - maxs[j, 2]))
        above_anchor = 1.0 if mins[i, 2] >= mins[j, 2] else 0.25
        support_overlap = self._box_overlap_ratio_xy(mins[i], maxs[i], mins[j], maxs[j])
        support_label_boost = 1.0 if self._label_has_any(dst_label, SUPPORT_LABEL_HINTS) else 0.75
        support_score = np.exp(-((vertical_gap / contact_scale) ** 2)) * support_overlap * above_anchor * support_label_boost
        self._add_edge(i, j, "supported by", support_score)

        containment_score = self._containment_score(mins[i], maxs[i], mins[j], maxs[j])
        container_label_boost = 1.0 if self._label_has_any(dst_label, CONTAINER_LABEL_HINTS) else 0.7
        self._add_edge(i, j, "inside", containment_score * container_label_boost)

    def _build_contact_edges(self, i: int, j: int) -> None:
        mins, maxs = self.box_bounds()
        avg_size = max(float(np.linalg.norm(self.sizes(), axis=1).mean()), 1e-6)
        contact_scale = max(avg_size * 0.12, 0.05)
        gap_3d = self._box_gap_3d(mins[i], maxs[i], mins[j], maxs[j])
        overlap_3d = self._box_overlap_ratio_3d(mins[i], maxs[i], mins[j], maxs[j])
        touch_score = float(np.exp(-((gap_3d / contact_scale) ** 2))) * max(overlap_3d, 0.25)
        self._add_edge(i, j, "touching", touch_score)

        gap_xy = self._box_gap_xy(mins[i], maxs[i], mins[j], maxs[j])
        overlap_z = self._interval_overlap_ratio(mins[i, 2], maxs[i, 2], mins[j, 2], maxs[j, 2])
        adjacent_score = float(np.exp(-((gap_xy / (contact_scale * 2.0)) ** 2))) * max(overlap_z, 0.25)
        self._add_edge(i, j, "adjacent", adjacent_score)

    def _build_between_edges(self) -> None:
        centers = self.centers()
        sizes = self.sizes()
        n = len(centers)
        if n < 3:
            return

        horizontal_scale = max(float(np.linalg.norm(sizes[:, :2], axis=1).mean()), 1e-6)
        vertical_scale = max(float(sizes[:, 2].mean()), 1e-6)
        for i in range(n):
            point = centers[i, :2]
            for j in range(n):
                if i == j:
                    continue
                anchor = centers[j, :2]
                best = 0.0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    other = centers[k, :2]
                    segment = other - anchor
                    segment_len = float(np.linalg.norm(segment))
                    if segment_len < horizontal_scale * 0.5:
                        continue

                    t = float(np.dot(point - anchor, segment) / (segment_len**2 + 1e-6))
                    if t < 0.0 or t > 1.0:
                        continue

                    closest = anchor + t * segment
                    perpendicular_dist = float(np.linalg.norm(point - closest))
                    endpoint_dist = min(float(np.linalg.norm(point - anchor)), float(np.linalg.norm(point - other)))
                    expected_z = centers[j, 2] + t * (centers[k, 2] - centers[j, 2])
                    z_dist = abs(float(centers[i, 2] - expected_z))

                    line_score = float(np.exp(-((perpendicular_dist / horizontal_scale) ** 2)))
                    endpoint_score = min(endpoint_dist / (horizontal_scale * 0.5 + 1e-6), 1.0)
                    z_score = float(np.exp(-((z_dist / (vertical_scale * 2.0 + 1e-6)) ** 2)))
                    best = max(best, line_score * endpoint_score * (0.5 + 0.5 * z_score))
                self._add_edge(i, j, "between", best)

    def _build_nearest_edges(self) -> None:
        centers = self.centers()
        n = len(centers)
        if n < 2:
            return
        scale = max(float(np.linalg.norm(self.sizes(), axis=1).mean()), 1e-6)
        dists = np.linalg.norm(centers[:, None, :3] - centers[None, :, :3], axis=2)
        np.fill_diagonal(dists, np.inf)
        nearest = np.min(dists, axis=0)
        for anchor_idx in range(n):
            if not np.isfinite(nearest[anchor_idx]):
                continue
            for candidate_idx in range(n):
                if candidate_idx == anchor_idx:
                    continue
                margin = dists[candidate_idx, anchor_idx] - nearest[anchor_idx]
                self._add_edge(candidate_idx, anchor_idx, "nearest", np.exp(-(margin / scale)))

    def _build_unary_edges(self) -> None:
        centers = self.centers()
        mins, maxs = self.box_bounds()
        sizes = self.sizes()
        scene_min = mins.min(axis=0)
        scene_max = maxs.max(axis=0)
        scene_span = np.maximum(scene_max - scene_min, 1e-6)
        avg_size = max(float(np.linalg.norm(sizes, axis=1).mean()), 1e-6)
        wall_scale = max(avg_size * 0.2, float(np.linalg.norm(scene_span[:2])) * 0.03, 0.05)
        floor_scale = max(float(sizes[:, 2].mean()) * 0.08, 0.05)
        floor_z = self._floor_z(scene_min[2], mins, maxs)

        for i in range(len(self.objects)):
            dist_to_x_wall = min(abs(mins[i, 0] - scene_min[0]), abs(maxs[i, 0] - scene_max[0]))
            dist_to_y_wall = min(abs(mins[i, 1] - scene_min[1]), abs(maxs[i, 1] - scene_max[1]))
            boundary_dist = min(dist_to_x_wall, dist_to_y_wall)
            corner_dist = dist_to_x_wall + dist_to_y_wall
            floor_gap = abs(mins[i, 2] - floor_z)
            low_score = 1.0 - float(np.clip((centers[i, 2] - scene_min[2]) / scene_span[2], 0.0, 1.0))

            self._add_edge(i, i, "corner", np.exp(-((corner_dist / wall_scale) ** 2)))
            self._add_edge(i, i, "in corner of room", np.exp(-((corner_dist / wall_scale) ** 2)))
            self._add_edge(i, i, "against wall", np.exp(-((boundary_dist / wall_scale) ** 2)))
            self._add_edge(i, i, "on the floor", np.exp(-((floor_gap / floor_scale) ** 2)))
            self._add_edge(i, i, "lower", low_score)

    def _floor_z(self, default_floor_z: float, mins: np.ndarray, maxs: np.ndarray) -> float:
        floor_tops = []
        for i, obj in enumerate(self.objects):
            label = obj.label.lower().replace("_", " ")
            if self._label_has_any(label, FLOOR_LABEL_HINTS):
                floor_tops.append(float(maxs[i, 2]))
        return min(floor_tops) if floor_tops else float(default_floor_z)

    def _build_room_edges(self) -> None:
        n = len(self.objects)
        for i in range(n):
            self._add_edge(i, i, "same room", 1.0)
            for j in range(n):
                if i != j:
                    self._add_edge(i, j, "same room", 1.0)

    def _build_visibility_edges(self) -> None:
        mins, maxs = self.box_bounds()
        centers = self.centers()
        n = len(self.objects)
        avg_size = max(float(np.linalg.norm(self.sizes(), axis=1).mean()), 1e-6)
        for i, obj in enumerate(self.objects):
            self._add_edge(i, i, "visible", float(getattr(obj, "confidence", 1.0)))
            for j in range(n):
                if i == j:
                    continue
                projection_overlap = self._box_overlap_ratio_xz(mins[i], maxs[i], mins[j], maxs[j])
                if projection_overlap <= 0:
                    continue
                depth_delta = centers[j, 1] - centers[i, 1]
                front_score = 1.0 / (1.0 + np.exp(-depth_delta / (avg_size + 1e-6)))
                distance = float(np.linalg.norm(centers[j] - centers[i]))
                near_score = float(np.exp(-distance / (avg_size * 2.0 + 1e-6)))
                occlusion_score = projection_overlap * front_score * near_score
                self._add_edge(i, j, "occluded by", occlusion_score)
                self._add_edge(j, i, "occluding", occlusion_score)

    def _build_feature_edges(self) -> None:
        embeddings = self.embeddings()
        if embeddings.size == 0:
            return
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        valid = norms[:, 0] > 1e-6
        if valid.sum() < 2:
            return
        normalized = np.divide(embeddings, norms + 1e-6)
        sims = normalized @ normalized.T
        for i in range(len(self.objects)):
            if not valid[i]:
                continue
            for j in range(len(self.objects)):
                if i == j or not valid[j]:
                    continue
                self.edges.append(Edge(i, j, "feature_similar", float(np.clip(sims[i, j], 0.0, 1.0))))

    def relation_matrix(self, relation: str) -> np.ndarray:
        relation = self._normalize_relation_name(relation)
        n = len(self.objects)
        mat = np.zeros((n, n), dtype=np.float32)
        for edge in self.edges:
            if edge.relation == relation:
                mat[edge.src, edge.dst] = max(mat[edge.src, edge.dst], edge.score)
        return mat

    @staticmethod
    def _normalize_relation_name(relation: str) -> str:
        relation = relation.strip().lower().replace("_", " ")
        aliases = {
            "against the wall": "against wall",
            "on floor": "on the floor",
            "closest": "nearest",
            "closest to": "nearest",
            "nearest to": "nearest",
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
        }
        return aliases.get(relation, relation)

    @staticmethod
    def _ratio_score(a: float, b: float) -> float:
        if a <= b:
            return 0.0
        return float(np.clip(np.log((a + 1e-6) / (b + 1e-6)) / np.log(2.0), 0.0, 1.0))

    @staticmethod
    def _label_has_any(label: str, hints: tuple[str, ...]) -> bool:
        return any(hint in label for hint in hints)

    @staticmethod
    def _box_gap_xy(min_a: np.ndarray, max_a: np.ndarray, min_b: np.ndarray, max_b: np.ndarray) -> float:
        dx = max(float(min_b[0] - max_a[0]), float(min_a[0] - max_b[0]), 0.0)
        dy = max(float(min_b[1] - max_a[1]), float(min_a[1] - max_b[1]), 0.0)
        return float(np.hypot(dx, dy))

    @staticmethod
    def _interval_overlap_ratio(a0: float, a1: float, b0: float, b1: float) -> float:
        overlap = max(min(float(a1), float(b1)) - max(float(a0), float(b0)), 0.0)
        denom = max(min(float(a1 - a0), float(b1 - b0)), 1e-6)
        return float(np.clip(overlap / denom, 0.0, 1.0))

    @staticmethod
    def _box_overlap_ratio_xy(min_a: np.ndarray, max_a: np.ndarray, min_b: np.ndarray, max_b: np.ndarray) -> float:
        overlap_x = max(min(float(max_a[0]), float(max_b[0])) - max(float(min_a[0]), float(min_b[0])), 0.0)
        overlap_y = max(min(float(max_a[1]), float(max_b[1])) - max(float(min_a[1]), float(min_b[1])), 0.0)
        overlap = overlap_x * overlap_y
        area_a = max(float((max_a[0] - min_a[0]) * (max_a[1] - min_a[1])), 1e-6)
        return float(np.clip(overlap / area_a, 0.0, 1.0))

    @staticmethod
    def _box_overlap_ratio_xz(min_a: np.ndarray, max_a: np.ndarray, min_b: np.ndarray, max_b: np.ndarray) -> float:
        overlap_x = max(min(float(max_a[0]), float(max_b[0])) - max(float(min_a[0]), float(min_b[0])), 0.0)
        overlap_z = max(min(float(max_a[2]), float(max_b[2])) - max(float(min_a[2]), float(min_b[2])), 0.0)
        overlap = overlap_x * overlap_z
        area_a = max(float((max_a[0] - min_a[0]) * (max_a[2] - min_a[2])), 1e-6)
        return float(np.clip(overlap / area_a, 0.0, 1.0))

    @staticmethod
    def _box_gap_3d(min_a: np.ndarray, max_a: np.ndarray, min_b: np.ndarray, max_b: np.ndarray) -> float:
        gaps = [
            max(float(min_b[axis] - max_a[axis]), float(min_a[axis] - max_b[axis]), 0.0)
            for axis in range(3)
        ]
        return float(np.linalg.norm(gaps))

    @staticmethod
    def _box_overlap_ratio_3d(min_a: np.ndarray, max_a: np.ndarray, min_b: np.ndarray, max_b: np.ndarray) -> float:
        overlaps = [
            max(min(float(max_a[axis]), float(max_b[axis])) - max(float(min_a[axis]), float(min_b[axis])), 0.0)
            for axis in range(3)
        ]
        overlap = overlaps[0] * overlaps[1] * overlaps[2]
        volume_a = max(float(np.prod(np.maximum(max_a - min_a, 1e-6))), 1e-6)
        return float(np.clip(overlap / volume_a, 0.0, 1.0))

    @staticmethod
    def _containment_score(min_inner: np.ndarray, max_inner: np.ndarray, min_outer: np.ndarray, max_outer: np.ndarray) -> float:
        inner_size = np.maximum(max_inner - min_inner, 1e-6)
        lower_margin = (min_inner - min_outer) / inner_size
        upper_margin = (max_outer - max_inner) / inner_size
        margin = float(np.minimum(lower_margin, upper_margin).min())
        return float(1.0 / (1.0 + np.exp(-8.0 * margin)))
