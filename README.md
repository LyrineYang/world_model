# HF 视频筛选管线

面向大体量视频数据的自动化筛选/上传工具：下载 → 解压 → 场景切分 → 闪烁过滤 →（可选 OCR）→ 多模型打分筛选 → 元数据输出/上传。默认按压缩包（分片）为处理单元，支持断点续跑和校准模式。

## 功能概览
- 场景切分：PySceneDetect `AdaptiveDetector`
- 闪烁过滤：亮度跳变检测
- OCR 文字过滤：PaddleOCR（可选 GPU）
- 多模型打分：DOVER（质量）、LAION-AES（美学）、UniMatch（运动），支持多卡并行
- 输出：保留片段/原视频、`metadata.jsonl`，可选上传到新 HF 数据集；校准模式输出分位数

## 环境要求
- Python 3.10+
- ffmpeg 已安装
- 服务器双卡 A800（80GB），或等效 CUDA GPU

### 依赖安装（GPU 服务器，CUDA 12.1 示例，一次性命令块）
```bash
# 1) 安装匹配硬件的 torch/torchvision
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# 2) 安装 Paddle（用于 OCR）
pip install paddlepaddle-gpu==2.6.1 -f https://www.paddlepaddle.org.cn/whl/cu121

# 3) 安装其余依赖
pip install -r requirements.txt

# 4) 拉取外部源码（若仓库未自带）
git clone https://github.com/QualityAssessment/DOVER.git DOVER
git clone https://github.com/autonomousvision/unimatch.git unimatch
git clone https://github.com/LAION-AI/aesthetic-predictor.git aesthetic-predictor
```

### conda 环境示例（GPU，CUDA 12.1）
```bash
conda create -y -n world_model python=3.10
conda activate world_model

# Torch/vision CUDA 12.1
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Paddle GPU
pip install paddlepaddle-gpu==2.6.1 -f https://www.paddlepaddle.org.cn/whl/cu121

# 其余依赖
pip install -r requirements.txt

# 外部源码（如未自带）
git clone https://github.com/QualityAssessment/DOVER.git DOVER
git clone https://github.com/autonomousvision/unimatch.git unimatch
git clone https://github.com/LAION-AI/aesthetic-predictor.git aesthetic-predictor
```

## 配置说明（`config.yaml` 关键字段）
示例详见 `config.example.yaml`。核心字段：
- `source_repo` / `target_repo`：源/目标 HF 数据集
- `workdir`：工作目录，内部创建 `downloads` / `extract` / `output` / `state`
- `shards`：待处理的分片文件名列表
- `runtime`：并行与流式
  - `stream_processing`: 是否边切分边打分（默认 true）
  - `scoring_workers`: scorer 并行线程数，0=按模型数自动
  - `queue_size`: 切分→打分队列长度
- `splitter`：场景切分参数（阈值/最小场景长/是否物理切割等）
- `flash_filter`：闪烁过滤（`brightness_delta`、`max_flash_ratio`、`sample_stride`、`record_only`）
- `ocr`：文字过滤（`enabled`、`text_area_threshold`、`sample_stride`、`lang`、`use_gpu`、`record_only`）
- `models`：评分模型列表（示例默认双卡 A800）
  - DOVER 质量：`kind: dover`, `device: cuda:0`
  - LAION-AES 美学：`kind: laion_aes`, `device: cuda:1`
  - UniMatch 运动：`kind: unimatch_flow`, `device: cuda:0`
  - 可按机器调整 `device`、`batch_size`、`threshold`、`extra`（权重/采样）
- `upload`：上传分块大小、并发、可选 720p 转码
- `calibration`：校准模式（`enabled`、`sample_size`、`output`、`quantiles`）

## 运行
基础命令：
```bash
python -m pipeline.pipeline --config config.yaml
```
常用选项：
- 仅处理前 N 个分片：`--limit-shards N`
- 不上传：`--skip-upload`
- 校准模式：`--calibration --sample-size 10000 --skip-upload`
- 自定义分位：`--calibration-quantiles 0.4,0.7,0.9`

## 性能与多卡建议（双 A800 80GB）
- 模型分配：DOVER/motion→`cuda:0`，AES→`cuda:1`（示例默认如此）
- 保持流式模式开启，减少 GPU 空转；`queue_size` 视内存调整
- `scoring_workers=0` 自动按模型数并行；如需限速可设为 1/2
- OCR：已安装 GPU 版 Paddle 时设 `ocr.use_gpu: true`；若解码占用高，可增大 `ocr.sample_stride`
- 闪烁过滤：误杀多可提高 `brightness_delta` 或增大 `max_flash_ratio`；漏检则反向调整

## 产出
- 处理后片段或原视频：`workdir/output/{shard}/videos/`
- 元数据：`workdir/output/{shard}/metadata.jsonl`
- 校准时输出 `calibration_meta.parquet` 并打印分位数

## 常见问题
- PaddleOCR 报 det/rec 参数：已兼容新旧版；若仍有问题，设置 `ocr.use_gpu`/`use_cpu` 对应安装版本
- 模型找不到权重：检查 `extra.weight_path`，按示例放置或让脚本自动下载（DOVER）
- GPU 利用率低：确认 `runtime.stream_processing=true`、`scoring_workers`>1、模型 `device` 分配合理
