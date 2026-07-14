from __future__ import annotations

import hashlib
import re

import numpy as np
import torch
from torch import nn


TEXT_FEATURE_DIM = 256
OBJECT_NUMERIC_DIM = 8
OBJECT_BASE_FEATURE_DIM = OBJECT_NUMERIC_DIM + TEXT_FEATURE_DIM
OBJECT_FEATURE_DIM = OBJECT_BASE_FEATURE_DIM


def _hash_index(token: str, dim: int = TEXT_FEATURE_DIM) -> int:
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % dim


def text_feature_tensor(text: str, device: str | torch.device = "cpu", dim: int = TEXT_FEATURE_DIM) -> torch.Tensor:
    vec = torch.zeros(dim, dtype=torch.float32, device=device)
    tokens = re.findall(r"[a-zA-Z0-9_\u4e00-\u9fff]+", text.lower())
    for token in tokens:
        vec[_hash_index(token, dim)] += 1.0
    norm = vec.norm(p=2)
    return vec / norm if norm > 0 else vec


def _label_feature(label: str, dim: int = TEXT_FEATURE_DIM) -> np.ndarray:
    vec = np.zeros((dim,), dtype=np.float32)
    tokens = re.findall(r"[a-zA-Z0-9_\u4e00-\u9fff]+", label.lower().replace("_", " "))
    for token in tokens:
        vec[_hash_index(token, dim)] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _embedding_array(obj, dim: int) -> np.ndarray:
    vec = np.zeros((dim,), dtype=np.float32)
    embedding = getattr(obj, "embedding", None)
    if embedding is None or dim == 0:
        return vec
    arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
    copy_dim = min(dim, arr.shape[0])
    vec[:copy_dim] = arr[:copy_dim]
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def object_embedding_dim(objects) -> int:
    dims = []
    for obj in objects:
        embedding = getattr(obj, "embedding", None)
        if embedding is not None:
            dims.append(int(np.asarray(embedding).reshape(-1).shape[0]))
    return max(dims, default=0)


def object_feature_tensor(
    objects,
    device: str | torch.device = "cpu",
    include_embeddings: bool = True,
    embedding_dim: int | None = None,
) -> torch.Tensor:
    if embedding_dim is None:
        embedding_dim = object_embedding_dim(objects) if include_embeddings else 0
    embedding_dim = int(max(embedding_dim, 0))
    feature_dim = OBJECT_BASE_FEATURE_DIM + embedding_dim

    if not objects:
        return torch.zeros((0, feature_dim), dtype=torch.float32, device=device)

    centers = np.stack([obj.center.astype(np.float32) for obj in objects], axis=0)
    sizes = np.stack([obj.size.astype(np.float32) for obj in objects], axis=0)
    center_mean = centers.mean(axis=0, keepdims=True)
    center_std = centers.std(axis=0, keepdims=True) + 1e-6
    size_scale = np.maximum(sizes.max(axis=0, keepdims=True), 1e-6)
    norm_centers = (centers - center_mean) / center_std
    norm_sizes = sizes / size_scale
    volumes = np.prod(np.maximum(sizes, 1e-6), axis=1, keepdims=True)
    volumes = volumes / (volumes.max() + 1e-6)
    confidences = np.asarray([[float(getattr(obj, "confidence", 1.0))] for obj in objects], dtype=np.float32)
    numeric = np.concatenate([norm_centers, norm_sizes, volumes, confidences], axis=1).astype(np.float32)
    labels = np.stack([_label_feature(obj.label) for obj in objects], axis=0)
    parts = [numeric, labels]
    if embedding_dim > 0:
        embeddings = np.stack([_embedding_array(obj, embedding_dim) for obj in objects], axis=0)
        parts.append(embeddings)
    features = np.concatenate(parts, axis=1)
    return torch.tensor(features, dtype=torch.float32, device=device)


class LanguageObjectInteractionModel(nn.Module):
    """Scores every object token by interacting language features with object features."""

    def __init__(self, text_dim: int = TEXT_FEATURE_DIM, object_dim: int = OBJECT_FEATURE_DIM, hidden_dim: int = 128):
        super().__init__()
        self.text_dim = text_dim
        self.object_dim = object_dim
        self.text_proj = nn.Sequential(nn.Linear(text_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.object_proj = nn.Sequential(nn.Linear(object_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, text_features: torch.Tensor, object_features: torch.Tensor) -> torch.Tensor:
        if text_features.dim() == 1:
            text_features = text_features.unsqueeze(0)
        if object_features.numel() == 0:
            return object_features.new_zeros((0,))

        text_hidden = self.text_proj(text_features).squeeze(0)
        object_hidden = self.object_proj(object_features)
        text_hidden = text_hidden.unsqueeze(0).expand_as(object_hidden)
        pair = torch.cat(
            [
                text_hidden,
                object_hidden,
                text_hidden * object_hidden,
                torch.abs(text_hidden - object_hidden),
            ],
            dim=1,
        )
        return self.pair_mlp(pair).squeeze(1)


def score_objects(model: LanguageObjectInteractionModel, text: str, objects, device: str | torch.device) -> torch.Tensor:
    text_features = text_feature_tensor(text, device=device)
    embedding_dim = max(int(model.object_dim) - OBJECT_BASE_FEATURE_DIM, 0)
    object_features = object_feature_tensor(
        objects,
        device=device,
        include_embeddings=embedding_dim > 0,
        embedding_dim=embedding_dim,
    )
    return model(text_features, object_features)


# Backward-compatible name for older checkpoints/scripts. New code should use
# LanguageObjectInteractionModel.
ConstraintWeightNet = LanguageObjectInteractionModel
