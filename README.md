# 世界模型视频清洗管线

面向大规模视频清洗与标注的工程化流水线：下载 → 解压 → 切分/过滤 → 打分 → （可选）Caption → 元数据输出/上传。

## 目标与默认能力
- 默认过滤主线：`DOVER + UniMatch`（最小依赖闭环）
- Caption 主线：保留 API/OpenRouter 流程，预留 `qwen3_local` 本地模型接入
- third_party 统一由 `third_party/manifest.lock.yaml` 固定版本

## 目录约定
- 环境定义：`env/conda/`、`env/pip/`
- third_party 锁定：`third_party/manifest.lock.yaml`
- 环境脚本：`scripts/env/`
- 诊断脚本：`scripts/doctor.py`
- 过滤配置模板：`configs/profiles/filter_dover_unimatch.yaml`
- Caption 模板：`configs/profiles/caption_only_qwen3.yaml`

## 1) 环境管理（Conda + pip lock）

### 1.1 创建过滤环境（默认）
```bash
bash scripts/env/create_filter_env.sh --name world-model-filter
```

可选安装：
```bash
# TransNet DALI
bash scripts/env/install_optional.sh --name world-model-filter --transnet

# OCR
bash scripts/env/install_optional.sh --name world-model-filter --ocr

# AES (laion_aes)
bash scripts/env/install_optional.sh --name world-model-filter --aes
```

### 1.2 创建 Caption 环境（Qwen3 预留）
```bash
bash scripts/env/create_caption_env.sh --name world-model-caption
```

### 1.3 third_party 固定拉取
```bash
# 按 manifest.lock.yaml 固定 commit checkout
bash scripts/bootstrap_third_party.sh

# 若明确要跟踪远端最新（非可复现模式）
bash scripts/bootstrap_third_party.sh --floating
```

## 2) 运行前检查（Doctor）

```bash
# 过滤链路
python scripts/doctor.py filter

# 过滤链路 JSON 输出（失败返回非 0）
python scripts/doctor.py filter --json

# caption 链路（本地 qwen3）
python scripts/doctor.py caption --model-path /path/to/qwen3-local
```

## 3) 过滤流水线

### 3.1 推荐：从 profile 模板启动
```bash
python -m pipeline.pipeline --config configs/profiles/filter_dover_unimatch.yaml
```

### 3.2 常用参数
```bash
# 不上传
python -m pipeline.pipeline --config configs/config.yaml --skip-upload

# 仅处理前 N 个分片
python -m pipeline.pipeline --config configs/config.yaml --limit-shards 2

# 校准模式
python -m pipeline.pipeline --config configs/config.yaml --calibration --sample-size 10000 --skip-upload
```

## 4) Caption 流水线

### 4.1 对已产出的 metadata 做补充 caption
```bash
python -m pipeline.caption_only --config configs/profiles/caption_only_qwen3.yaml
```

### 4.2 `qwen3_local` 说明
- `caption.provider=qwen3_local` 时，需配置：
  - `caption.local_model_path` 或环境变量 `QWEN3_LOCAL_MODEL_PATH`
- 默认 `caption.enabled=false`，不会自动触发本地推理

## 5) 关键配置说明

### 5.1 模型 kinds
- 已实现：`dover`, `unimatch_flow`, `laion_aes`, `dummy`
- 预留（首版占位，不影响筛选，默认返回 `1.0`）：
  - `clip_consistency_stub`
  - `egovideo_consistency_stub`
- 兼容旧占位名（建议迁移到上面新命名）：
  - `clip_text_consistency` -> `clip_consistency_stub`
  - `egovideo_text_consistency` -> `egovideo_consistency_stub`

### 5.2 splitter 默认
- 默认 `pyscenedetect`
- `transnet` / `transnet_dali` 需要额外可选依赖

## 6) 兼容与迁移说明
- 旧配置仍可运行；新增字段都有默认值
- 不再使用 editable 方式安装 `third_party/unimatch`
- 旧 `bootstrap_third_party.sh --install-editable` 仅保留兼容参数，不再执行安装逻辑

## 7) Docker 预备模板
- `docker/filter.Dockerfile`
- `docker/caption.Dockerfile`

仅作为后续容器化基线模板，当前不作为默认运行入口。

## 8) EgoDex 300G 清洗（离线视频）

EgoDex 解压后视频可直接用于清洗（`.mp4`），无需额外转码。

```bash
# 0) 在仓库根目录执行
# cd <world_model_root>

# 1) 设置数据根目录（绝对或相对均可）
export EGODEX_ROOT=../datasets/egodex300g

# 2) 可选：环境检查
python scripts/check_env.py

# 3) 按 EgoDex 300G 配置做离线过滤（递归扫描所有 mp4）
python scripts/run_workflow.py offline-filter \
  --config configs/config_filter_egodex300g.yaml \
  --input-dir "$EGODEX_ROOT" \
  --recursive \
  --copy-mode link \
  --output-dir ./workdir_egodex300g/output/offline_egodex300g
```

如果要覆盖更严格/更宽松筛选逻辑，可在命令中追加：

```bash
--strategy "dover >= dover_thr and motion >= motion_thr"
```
