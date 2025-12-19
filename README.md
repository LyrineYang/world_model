# HF 视频筛选上传管线

面向大体量视频数据（例：`ONE-Lab/HQ-video-data`，每包 50GB）的精简管线：下载 → 解压 → **PySceneDetect 切分** → 闪烁过滤 → 模型打分筛选 → 元数据输出 → 上传到新的 HF 数据集。默认按压缩包为处理单元，便于 7TB 级别吞吐和断点续跑。

## 流程细节（每步做什么、如何筛选）
- 下载/解压：从 `source_repo` 拉取 `shards` 列表中的压缩包到 `workdir/downloads/`，安全解压到 `workdir/extract/{shard}/`。
- 场景切分：PySceneDetect `AdaptiveDetector`，`threshold`/`min_scene_len` 控制敏感度；片段写入 `extract/{shard}/scenes/`，可选删除原视频。
- 闪烁过滤：亮度跳变检测（`brightness_delta`、`max_flash_ratio`、`sample_stride`），判定闪烁的片段标记 `reason=flash`，不保留。
- 模型打分：按 `models` 列表加载 scorer，对片段打分并共同筛选（异常标记 `scoring_error`，所有模型分数都达标才 `keep=True`）：
  - `dover`：视频质量（技术+美学）得分，默认 0~1。
  - `laion_aes`：CLIP+线性头美学得分，均匀采样 `num_frames` 帧取均值。
  - `unimatch_flow`：光流幅值均值，代表运动量，可过滤 PPT/静态内容。
  - `dummy`：恒 1.0，用于通路验证。
- 输出与上传：保留片段硬链/拷贝到 `workdir/output/{shard}/videos/`，元数据写 `metadata.jsonl`（source/output 路径、大小、scores、keep、reason）；未加 `--skip-upload` 时，将 `output/{shard}` 上传到 `target_repo`。OCR/字幕过滤可通过 `config.yaml` 中的 `ocr` 开关启用。

## 快速开始（统一环境流程）
### 获取代码
```bash
git clone <本仓库地址>
cd world_model
# 如未内置第三方源码，可在根目录拉取依赖
git clone https://github.com/QualityAssessment/DOVER.git DOVER
git clone https://github.com/autonomousvision/unimatch.git unimatch
git clone https://github.com/LAION-AI/aesthetic-predictor.git aesthetic-predictor
```
### 前置要求
- Python 3.10+，系统已安装 `ffmpeg`
- 写入 HF 的访问令牌：环境变量 `HF_TOKEN`
- 安装匹配平台的 `torch`/`torchvision`：GPU 场景选择对应 CUDA 轮子，CPU/ mac 场景使用纯 CPU 轮子（不要安装带 `+cuXXX` 的包）

### 环境安装（示例使用 uv，可替换为 conda/venv）
```bash
# 1) 安装匹配硬件的 torch/torchvision（示例 CUDA 12.1，如无 GPU 用 CPU 版）
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121
# mac/CPU 或无 CUDA: pip install torch==2.3.1 torchvision==0.18.1

# 2) 安装其余依赖
pip install uv
uv pip install -r requirements.txt
```

若使用 conda：
```bash
conda create -y -n world_model python=3.10
conda activate world_model
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121
# mac/CPU: pip install torch==2.3.1 torchvision==0.18.1
pip install -r requirements.txt
```
> 建议统一使用 torch 2.3.1 / torchvision 0.18.1，以避免多版本混用导致的加载/推理冲突。

### 模型与权重准备
```bash
# 复制配置并编辑
cp config.example.yaml config.yaml
# 可选：下载 UniMatch 光流权重（运动过滤）
bash scripts/download_unimatch_weights.sh
# 如需自带权重：将 DOVER 权重放入 DOVER/pretrained_weights/DOVER.pth（否则自动从 HF 拉取）
```

### 运行
```bash
python -m pipeline.pipeline --config config.yaml
# 仅处理前 N 个分片： --limit-shards N
# 仅本地处理不上传： --skip-upload

校准模式（采样分布，不上传，输出 calibration_meta.parquet 并打印分位数）：

```bash
python -m pipeline.pipeline --config config.yaml --calibration --sample-size 10000 --skip-upload
```

校准默认分位数为 0.4 / 0.7，可用 `--calibration-quantiles 0.4,0.7,0.9` 调整；输出路径可用 `--calibration-output /path/to/calibration_meta.parquet` 指定。
```

## 配置要点
- `source_repo`：源数据集（例：`ONE-Lab/HQ-video-data`）。
- `target_repo`：筛选后写入的新数据集。
- `workdir`：工作目录，内部会创建 `downloads/`、`extract/`、`output/`、`state/`。
- `shards`：待处理的压缩包文件名列表。
- `splitter`：Base Pool 场景切分，默认 `pyscenedetect`，推荐 `threshold=27.0`，`min_scene_len=16`（约 0.5s）。
- `splitter.cut`：是否物理切割视频。默认 false（保留原视频）；仍用 PySceneDetect 标出场景，并在元数据写入虚拟切片窗口（按 `window_len_frames`/`window_stride_frames`，示例 121/60 帧）。
- `flash_filter`：闪烁过滤（迪厅灯光等），默认开启，可调节亮度跳变阈值与采样步长。
- `models`：模型清单（如 `dover` / `laion_aes`）；`kind` 区分实现，`threshold` 用于筛选，`device` 绑定 GPU。可通过 `extra` 字段指定权重路径、采样策略等。
- `upload`：HF 上传分块大小和并发。
- `calibration`（可选）：`enabled`、`sample_size`、`output`、`quantiles` 用于采样评分并输出分布，帮助设定阈值；开启时默认不上传，可通过命令行 `--calibration`/`--sample-size`/`--calibration-output`/`--calibration-quantiles` 覆盖。

## 模型集成（Base Pool 已接入）
- DOVER：读取 `DOVER` 仓库与 `dover.yml`，默认使用 `pretrained_weights/DOVER.pth`（不存在则自动从 HF `teowu/DOVER` 下载）。可通过 `config.yaml` 中模型的 `extra` 指定 `repo_path/config_path/weight_path/data_key/output=fused|technical|aesthetic`。
- LAION-AES：使用 open_clip `ViT-L-14` 预训练 + 线性头（默认 `aesthetic-predictor/sa_0_4_vit_l_14_linear.pth`），均匀采样若干帧（`num_frames`）求平均。
- UniMatch 光流（可选）：用于运动过滤，默认加载 `unimatch` 仓库与 `pretrained/gmflow-scale1-mixdata-*.pth`（需手动下载到对应路径，见 `unimatch/MODEL_ZOO.md`），输出平均光流幅值作为运动分数。
- DummyScorer：连通性验证。
- 扩展方式：在 `pipeline/models.py` 注册新 scorer（如 VMAF/光流/质量/美学）。
- 多卡策略：在 `config.yaml` 为不同模型指定不同 `device`（如 `cuda:0,1`），避免频繁切换权重。
- Base Pool 默认流程：PySceneDetect 切分 → Flash Filter → （可选）OCR 文字过滤 → Dover/AES/UniMatch 评分 → 阈值筛选。

## 模型与资源准备（发布/部署建议）
- 如仓库未内置第三方源码，请在根目录拉取依赖（建议用官方仓库）：
  ```bash
  git clone https://github.com/QualityAssessment/DOVER.git DOVER
  git clone https://github.com/autonomousvision/unimatch.git unimatch
  git clone https://github.com/LAION-AI/aesthetic-predictor.git aesthetic-predictor
  ```
  若已有外部路径，可在 `config.yaml` 的模型 `extra.repo_path`/`weight_path` 指向。
- 随项目一起打包：`DOVER/`、`aesthetic-predictor/`、`unimatch/` 目录，避免用户额外克隆。
- 权重：
  - DOVER：默认自动从 HF `teowu/DOVER` 下载到 `DOVER/pretrained_weights/DOVER.pth`；可提前放好或使用 DOVER-Mobile 替代。
  - LAION-AES：线性头已包含在 `aesthetic-predictor/sa_0_4_vit_l_14_linear.pth`；open_clip 会自动下载主干权重。
  - UniMatch（运动过滤，需权重）：手动下载到 `unimatch/pretrained/`（示例脚本 `scripts/download_unimatch_weights.sh` 会下载 `gmflow-scale1-mixdata-train320x576-4c3a6e9a.pth`）。在 `config.yaml` 的模型 `extra.weight_path` 指定路径。
- OCR（文字过滤，可选）：使用 PaddleOCR（已在 requirements），需按平台安装 `paddlepaddle` 或 `paddlepaddle-gpu`（参考官方安装命令/版本）。配置 `ocr.enabled`、`text_area_threshold`、`sample_stride`、`lang`。
- 不打包源码时：在 `config.yaml` 中通过 `repo_path/weight_path` 指向外部路径，并保证可访问对应仓库/权重：
  - DOVER 源码 `https://github.com/QualityAssessment/DOVER.git`
  - UniMatch 源码 `https://github.com/haofeixu/unimatch.git`
  - 其余同上
- 必备依赖：`torch/torchvision`（匹配硬件）、`ffmpeg`、`HF_TOKEN`。

## 运行建议
- 多 GPU 时可在 `config.yaml` 为模型分配不同 `device`，避免权重频繁切换；如 DOVER `cuda:0`，AES `cuda:1`，UniMatch `cuda:2`。
- 磁盘紧张时可开启 `splitter.remove_source_after_split` 删除原视频，仅保留切分片段；处理完分片会清理解压目录。
- 先用 Dummy scorer 验证通路，再逐步启用 DOVER/AES/UniMatch 并调阈值、采样帧数和分辨率以平衡吞吐与质量。
- 如需阈值校准，可开启 `--calibration --sample-size N`：抽样 N 个切分片段跑过滤，输出 `calibration_meta.parquet` 与分位数日志，帮助设定各模型阈值（quantiles 可在 config.calibration.quantiles 设置，默认 0.4/0.7）。

## 产出
- 切分后的场景片段存放于 `extract/{shard}/scenes/`（处理完会在清理阶段删除，若开启 `remove_source_after_split=true` 会删除原视频）。
- 保留片段硬链接/拷贝到 `workdir/output/{shard}/videos/`（若 cut=false，则是原视频）。
- 元数据写入 `workdir/output/{shard}/metadata.jsonl`，包含：
  - `source_path`/`output_path`、文件大小
  - `scores`、`keep`、`reason`（含 split_failed/flash/ocr_text/scoring_error/score_below_threshold 等）
  - 场景与虚拟切片信息（当 cut=false 且 decord 可用）：`fps`、`total_frames`、`num_windows`、`windows`（帧范围列表）、`scenes`（场景起止帧及窗口数）
- 上传时将整个 `output/{shard}` 目录作为子目录推送到目标 HF 数据集，支持断点重试。

## 打包与部署
- 本地打包（含虚拟环境可选）：
  ```bash
  tar czf world_model_pipeline.tgz README.md requirements.txt config.yaml pipeline
  ```
  如需连同已下载的分片/状态一并迁移，在 `workdir` 外打包对应目录。
- 服务器端解压后：创建虚拟环境，安装依赖，设置 `HF_TOKEN`，运行与本地一致的命令。

## 注意
- 7TB 规模建议一次处理 1–2 个分片，确保 NVMe 空间充足；每个分片处理完即清理临时文件（默认清理解压目录，保留输出）。
- 若磁盘紧张可在 `splitter.remove_source_after_split=true` 时删除原视频，仅保留场景切分片段。
- 如果需要更复杂的流式或 GPU 光流/软转场模型（如 TransNet），可在现有 scorer 接口基础上扩展；UniMatch 运动过滤已提供示例配置。
