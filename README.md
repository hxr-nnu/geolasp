# GeoLaSP

GeoLaSP 是一个对象级三维视觉语言定位原型。它读取 SpatialLM 已生成的对象级 layout，构建显式几何场景图，解析英文文本描述为结构化空间约束，再用几何推理和可选的轻量语言-对象 alignment model 对候选对象排序和解释。

本项目不训练或微调 SpatialLM，也不训练或微调 Qwen、Llama 或任何大语言模型。SpatialLM 只作为预训练 layout 生成器和冻结点云特征提取器使用；所有 SpatialLM 点云特征调用都应处于 `eval()` 和 `torch.inference_mode()` 下。

## Modes

`layout-only` 是默认模式。它只读取 layout 中的对象类别、中心、尺寸、体积和置信度，不加载 SpatialLM 模型，不提取点云 embedding，可在 CPU 或低显存环境运行。

`point-feature` 是可选模式。它读取 layout，并从原始点云通过冻结的 SpatialLM 点云编码器提取 point token features，再按对象 bbox mean pooling 成 `ObjectToken.embedding`，同时写入 `ObjectToken.point_stats`。对象级 embedding 会保存为 `.npz` 缓存，后续训练和评估优先读缓存。

## Environment

基础 GeoLaSP 流程建议：

```powershell
conda create -n geolasp python=3.10 -y
conda activate geolasp
pip install numpy torch openai tqdm scipy scikit-learn
```

point-feature 模式还需要单独安装 SpatialLM 及其依赖，包括 PyTorch、CUDA、Transformers、SpatialLM 的 `spatiallm.pcd` 预处理依赖，以及 SpatialLM1.1 所需的 Sonata/Pointcept 相关依赖。layout-only 不需要加载 SpatialLM 模型。

推荐硬件：

- layout-only：CPU 可运行。
- point-feature：建议 CUDA GPU，24GB 显存更稳。

`OPENAI_API_KEY` 只用于英文文本约束解析。没有该环境变量时，`--constraint_parser auto` 会退回规则解析。

## Layout-Only Evaluation

```powershell
python scripts/eval_grounding.py `
  --annotation_json data/scanrefer_min.json `
  --layout_dir outputs/layouts `
  --output_json outputs/eval_layout_only.json
```

使用 LLM 解析文本，但不提取点云特征：

```powershell
python scripts/eval_grounding.py `
  --annotation_json data/scanrefer_min.json `
  --layout_dir outputs/layouts `
  --constraint_parser llm `
  --output_json outputs/eval_llm_geometry.json
```

## Point Feature Cache

离线提取并缓存 SpatialLM 对象级点云特征：

```powershell
python scripts/extract_point_features.py `
  --spatiallm_root /path/to/SpatialLM `
  --spatiallm_model_path manycore-research/SpatialLM1.1-Qwen-0.5B `
  --layout_dir outputs/layouts `
  --point_cloud_dir data/scannet_points `
  --point_feature_cache_dir outputs/point_features
```

每个 scene 会生成一个 `scene_id.npz`，至少包含 `object_ids`、`labels`、`centers`、`sizes`、`embeddings` 和 `point_stats`。默认优先读缓存；需要重算时加：

```powershell
--overwrite_point_feature_cache
```

## Train Alignment

不使用点云 embedding：

```powershell
python scripts/train_alignment.py `
  --annotation_json data/scanrefer_min.json `
  --layout_dir outputs/layouts `
  --checkpoint outputs/checkpoints/alignment_layout_only.pt
```

使用点云 embedding：

```powershell
python scripts/train_alignment.py `
  --annotation_json data/scanrefer_min.json `
  --layout_dir outputs/layouts `
  --point_cloud_dir data/scannet_points `
  --use_spatiallm_point_features `
  --point_feature_cache_dir outputs/point_features `
  --checkpoint outputs/checkpoints/alignment_with_point_features.pt
```

这里训练的是 GeoLaSP 自己的轻量 alignment model。checkpoint 会记录 `object_dim`、`point_feature_dim`、`use_point_features`、`point_feature_cache_dir` 和 `spatiallm_model_path` 等元信息。

## Evaluate With Fusion

几何推理 + 神经模型：

```powershell
python scripts/eval_grounding.py `
  --annotation_json data/scanrefer_min.json `
  --layout_dir outputs/layouts `
  --interaction_checkpoint outputs/checkpoints/alignment_layout_only.pt `
  --scoring_mode fusion `
  --fusion_alpha 0.6 `
  --output_json outputs/eval_fused.json
```

几何推理 + 神经模型 + 点云 embedding：

```powershell
python scripts/eval_grounding.py `
  --annotation_json data/scanrefer_min.json `
  --layout_dir outputs/layouts `
  --point_cloud_dir data/scannet_points `
  --use_spatiallm_point_features `
  --point_feature_cache_dir outputs/point_features `
  --interaction_checkpoint outputs/checkpoints/alignment_with_point_features.pt `
  --scoring_mode fusion `
  --fusion_alpha 0.6 `
  --output_json outputs/eval_full.json
```

`--scoring_mode geometry` 可只跑几何推理，`--scoring_mode neural` 可只跑 alignment model，`--scoring_mode fusion` 融合两者。输出 JSON 包含 `geometry_scores`、`neural_scores`、`final_scores`、`point_features_attached`、`point_feature_dim`、`constraint_source` 和 `constraint_json`。如果某个 scene 缺少点云特征，评估会记录 `warnings` 并降级到可用特征。
