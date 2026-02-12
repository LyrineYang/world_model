# HF 视频筛选管线

面向大体量视频数据的自动化筛选/上传工具：下载 → 解压 → 场景切分 → 闪烁过滤 →（可选 OCR）→ 多模型打分筛选 → 元数据输出/上传。默认按压缩包（分片）为处理单元，支持断点续跑和校准模式。

## 功能概览
- 场景切分：PySceneDetect `AdaptiveDetector`
- 闪烁过滤：亮度跳变检测
- OCR 文字过滤：RapidOCR（ONNXRuntime，默认 CPU，可选 GPU）
- Caption 生成：可配置 API（或未来替换开源模型）为保留片段生成描述
- 多模型打分：DOVER（质量）、LAION-AES（美学）、UniMatch（运动），支持多卡并行
- 输出：保留片段/原视频、`metadata.jsonl`，可选上传到新 HF 数据集；校准模式输出分位数
- 分片流水：按分片顺序处理，单个分片执行“下载 → 解压 → 切分/过滤 → 打分/Caption → 落盘 → 上传”；`runtime.stream_processing=true` 时切分、过滤、打分在分片内部是流式并行的，下载完成即开始处理，处理完成即上传下一个。

## 环境要求
- Python 3.10+
- ffmpeg 已安装
- 服务器双卡 A800（80GB），或等效 CUDA GPU

### 依赖安装（GPU 服务器，CUDA 12.x，OCR 默认 CPU）
```bash
# 0) 创建/激活环境（示例名 yjq_video_data）
conda create -y -n yjq_video_data python=3.10
conda activate yjq_video_data

# 1) 安装项目依赖，带约束文件锁定 numpy/opencv/huggingface_hub/rapidocr 版本
pip install -r requirements.txt -c constraints.txt

# 2) 安装匹配硬件的 torch/torchvision（cu121 轮子对 CUDA 12.x 驱动可用）
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# 2.1 可选：GPU 场景切分（TransNetV2 + DALI，提升切分吞吐）
pip install nvidia-dali-cuda120 transnetv2-pytorch huggingface_hub

# 3) OCR 默认走 CPU，无需安装 onnxruntime-gpu；config 中保持 ocr.use_gpu: false

# 4) 安装本地模型源码以避免导入失败（推荐）
pip install -e ./DOVER
pip install -e ./unimatch

# 5) 可选：清理旧 Paddle 以免干扰
# pip uninstall -y paddleocr paddlepaddle paddlepaddle-gpu

git clone https://github.com/QualityAssessment/DOVER.git DOVER
git clone https://github.com/autonomousvision/unimatch.git unimatch
git clone https://github.com/LAION-AI/aesthetic-predictor.git aesthetic-predictor

# 必备权重下载（请主动执行）
bash scripts/download_unimatch_weights.sh  # UniMatch 运动模型权重
```

权重下载
- DOVER：运行时自动通过 hf_hub_download 获取 `pretrained_weights/DOVER.pth`（也可手动放置同路径）。
- UniMatch：手动执行 `bash scripts/download_unimatch_weights.sh`，下载到 `unimatch/pretrained/gmflow-scale1-mixdata-train320x576-4c3a6e9a.pth`，或在 config 用 `weight_path` 覆盖。
- LAION AES：默认自动加载/下载 `aesthetic-predictor/sa_0_4_vit_l_14_linear.pth`，如需自定义可手动放置并在 config 覆盖。

> 如必须 GPU OCR，自行安装与驱动匹配的 onnxruntime-gpu + 对应 CUDA 运行库，并将配置改为 `ocr.use_gpu: true`。

## 运行
基础命令：
```bash
python -m pipeline.pipeline --config config.yaml
```
常用选项（追加在命令后）：
- 仅处理前 N 个分片：`python -m pipeline.pipeline --config config.yaml --limit-shards 2`
- 不上传：`python -m pipeline.pipeline --config config.yaml --skip-upload`
- 校准模式：`python -m pipeline.pipeline --config config.yaml --calibration --sample-size 10000 --skip-upload`
- 自定义分位：`python -m pipeline.pipeline --config config.yaml --calibration --calibration-quantiles 0.4,0.7,0.9`

### 监控进度（独立脚本，零侵入）
- 状态文件：`state/<shard>.json` 按阶段实时写入 `stage`（downloading/extracting/processing/materializing/uploading/processed）、`started_at`、`finished_at`，用于断点续跑/诊断。
- 监控脚本：`python scripts/monitor.py --config config.yaml --interval 2`。会汇总 downloaded/extracted/scored/uploaded 数、阶段分布、速率、平均分片耗时，以及已完成分片的 `summary.json`（time_total、clips_scored）统计。
- metadata 写入：每个分片处理完成后统一写 `output/<shard>/metadata.jsonl`（逐行记录 keep/reason/scores 等），视频文件仅对保留片段落盘；如全被过滤，`videos/` 可能为空但 metadata 仍存在。
- 运行示例（请从仓库根目录或使用绝对路径）：`python /scratch/ayuille1/jchen293/wcloong/video_dataset/scripts/monitor.py --config /scratch/ayuille1/jchen293/wcloong/video_dataset/config.yaml --interval 2`。

## 配置说明（`config.yaml` 关键字段）
示例详见 `config.example.yaml`，`config.test.yaml` 为最小跑通示例。

### 下载/输入
- `source_repo`：源 HF 数据集（dataset）。大规模时建议本地镜像或开启缓存。
- `workdir`：工作目录，内部自动创建 `downloads` / `extract` / `output` / `state`。
- `hf_token`（可选）：有权限账号的 HF token。建议通过环境变量 `HF_TOKEN` 注入，或在配置中显式填写（会覆盖环境变量）。用于访问 gated/private 数据集。
- `shards`：要处理的分片文件名列表。
- `shards_file`（可选）：分片文件名列表路径（每行一个文件名）。分片多时优先使用，生成方式示例：`python scripts/fetch_shards_list.py --repo ONE-Lab/HQ-video-data --output shards_ONE-Lab_HQ-video-data.txt`。

### 运行/切分/过滤
- `runtime`：流式与并行
  - `stream_processing` 是否边切分边打分（默认 true）
  - `scoring_workers` scorer 并行线程数，0=按模型数自动
  - `queue_size` 切分→打分队列长度
  - `prefetch_shards` 分片预取数量（默认 2，开启下载+解压预取，完成即入队处理）
  - `download_workers` 下载预取并发数（默认 2，与 `prefetch_shards` 配合）
  - 预取模式下会显示两条进度条：下载+解压（Prefetch）与处理/上传（Processed）
  - 断点续跑：每个分片有独立 state（downloaded/extracted/scored/uploaded）；若某分片 `uploaded=true` 且非校准/非 `--skip-upload`，会跳过该分片
- `splitter`：场景切分
  - `kind: transnet_dali`（默认）：DALI GPU 解码 + TransNetV2，`device`（如 cuda:0）、`batch_size`（推理批次）、`stride_frames`（抽帧步长，>1 更快但精度降）、`transnet_threshold`（边界阈值），可选 `weight_path`（本地权重）。依赖 nvidia-dali-cuda120 + TransNetV2。
  - `kind: transnet`（备选）：decord GPU 解码 + TransNetV2（需 decord CUDA 版），参数同上。
  - `kind: pyscenedetect`：`threshold`、`min_scene_len` 控制颗粒度；`cut` 控制是否物理切割，`window_len_frames`/`window_stride_frames` 控制虚拟窗口。
- `flash_filter`：闪烁过滤（`brightness_delta`、`max_flash_ratio`、`sample_stride`、`record_only`）
- `ocr`：文字过滤（`enabled`、`text_area_threshold`、`sample_stride`、`lang`、`use_gpu`、`record_only`）

### Caption（OpenRouter 默认）
- 位置：`config.yaml` 的 `caption` 段。
- 快速模板：`provider: openrouter`，`api_url: https://openrouter.ai/api/v1/chat/completions`，`model: gpt-4o`，`include_image: true`。
- 鉴权：`api_key`（建议用环境变量 `OPENROUTER_API_KEY` 注入），`api_key_header`（默认 Authorization）。可选：`openrouter_referer`、`openrouter_title`。
- 提示与输出：`system_prompt` / `user_prompt` 控制语气与格式；`max_tokens`、`temperature` 控制长度与多样性。
- 并发与鲁棒：`max_workers` 控制同时请求数，`timeout` / `retry` 控制超时与重试。
- 自建接口：将 `provider: api`，填写 `api_url`，`file_field`、`response_field`、`extra_fields` 对应文件上传式服务。

### 模型打分
- `models`：评分模型列表
  - DOVER 质量：`kind: dover`, `device: cuda:0`
  - LAION-AES 美学：`kind: laion_aes`, `device: cuda:1`
  - UniMatch 运动：`kind: unimatch_flow`, `device: cuda:0`
  - 可按机器调整 `device`、`batch_size`、`threshold`、`extra`（权重/采样）
  - 当前示例阈值基于 ~95 条标注样本的 P50 留存（视频级约 20%）：`dover≈0.57`、`aes=5.0`、`motion≈87`。留存率过高/过低可酌情调节。

### 上传/输出
- `upload`：HF 上传相关
  - `target_repo`：目标 HF 数据集（dataset）
  - `chunk_size_mb`：分块大小（MB）
  - `max_workers`：并发线程数
  - `resize_720p`：上传前是否转码 720p（占用更多时间但省流量）
  - `cleanup_after_upload`：上传成功后是否删除本地该分片的 download/extract/output（默认示例启用，节省磁盘；`--skip-upload` 或校准模式下不会清理）
- 产出目录：`workdir/output/{shard}`，含 `videos/`（保留片段/转码片段）与 `metadata.jsonl`（scores、过滤记录、可选 caption）。

### 校准
- `calibration`：`enabled` 控制校准模式，启用后只做下载/切分/过滤/打分，将打分结果写 `metadata.jsonl`，不会上传；`sample_size` 为抽样上限（到达后提前停），`output` 指定分布输出路径，`quantiles` 为分位点列表（用于报表）。

### Caption 配置（OpenRouter 示例）
- 配置文件：根目录 `config.yaml`（可由 `config.example.yaml` 复制）中的 `caption` 段。
- 推荐设置：`provider: openrouter`、`api_url: https://openrouter.ai/api/v1/chat/completions`、`model: gpt-4o`、`include_image: true`。API Key 使用环境变量更安全：`export OPENROUTER_API_KEY=xxxxx`，并在配置中留空 `api_key` 或直接填入。
- 提示词：`system_prompt`/`user_prompt` 默认用简洁中文描述，可按需求改写；`max_tokens`/`temperature` 控制长度与多样性。
- 并发/鲁棒：`max_workers` 控制同时请求数量，`timeout`/`retry` 控制超时与重试。
- 自建接口：若已有文件上传式 caption 服务，将 `provider` 设为 `api`，填充 `api_url`/`file_field`/`response_field`/`extra_fields`。

## 性能与多卡建议（双 A800 80GB）
- 模型分配：DOVER/motion→`cuda:0`，AES→`cuda:1`（示例默认如此）
- 保持流式模式开启，减少 GPU 空转；`queue_size` 视内存调整
- `scoring_workers=0` 自动按模型数并行；如需限速可设为 1/2
- OCR：RapidOCR 默认走 CPU；如安装了 onnxruntime-gpu 且想用 GPU，可设 `ocr.use_gpu: true`。解码占用高时可增大 `ocr.sample_stride`。
- 闪烁过滤：误杀多可提高 `brightness_delta` 或增大 `max_flash_ratio`；漏检则反向调整
- GPU 场景切分（TransNetV2，可选）：在 H100 80GB 上可设 `splitter.kind=transnet_dali`、`device=cuda:0`、`batch_size` 32–64、`stride_frames` 1–2，结合 `queue_size>=48` 保证 scorer 吃满；若切分仍成瓶颈，可将 `transnet_threshold` 提高或 `stride_frames` 增大以减少切片数。

## 产出
- 处理后片段或原视频：`workdir/output/{shard}/videos/`
- 元数据：`workdir/output/{shard}/metadata.jsonl`，包含 scores/过滤记录和可选 caption
- 校准时输出 `calibration_meta.parquet` 并打印分位数


## EgoDex 清洗（离线视频）

EgoDex 解压后的 `.mp4` 可直接清洗，无需转码。

```bash
# 在仓库根目录执行
export EGODEX_ROOT=../datasets/egodex300g

python scripts/run_workflow.py offline-filter \
  --config configs/config_filter_egodex300g.yaml \
  --input-dir "$EGODEX_ROOT" \
  --recursive \
  --copy-mode link \
  --output-dir ./workdir_egodex300g/output/offline_egodex300g
```
