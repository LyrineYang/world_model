# EgoDex 清洗 Pipeline 优化计划（MVP）

## 目标
- 以最小代码改动提升 DOVER + UniMatch 离线清洗吞吐。
- 优先落地 4 个高 ROI 改动，避免首期进入重构工程。

## 原则
- 先做低侵入优化，再做结构性重构。
- 每项优化都可配置回退，确保可控上线。
- 首期不改动打分语义，不引入复杂并发流水线。

## 首期范围（仅 4 项）
1. `torch.inference_mode()` 替换推理路径中的 `torch.no_grad()`
2. CUDA 路径启用 BF16 autocast（`torch.autocast(..., dtype=torch.bfloat16)`）
3. 增加可选 `torch.compile`（默认关闭，H100 配置可开启）
4. 视频解码支持 `decode_device=auto`（先 GPU，失败回退 CPU）

## 不在首期（延后）
- 跨视频“真批推理”重构（DOVER/UniMatch 批量拼接）
- 额外 decode worker pool / 复杂生产者消费者流水线
- 大规模 runlog 结构改造
- Flash-attn / xFormers 深度接入

## 配置 Spec（MVP）

### 全局（保持最小）
- 不新增全局强耦合开关作为首期必需项。
- 如需全局后端开关（TF32/cuDNN），放到后续阶段统一引入。

### 模型级（`models[].extra`）
1. `precision`: `"fp32" | "bf16"`  
   - 默认：`"bf16"`（CUDA），非 CUDA 自动回退 `fp32`。
2. `compile`: `bool`  
   - 默认：`false`。
3. `compile_mode`: `"default" | "reduce-overhead" | "max-autotune"`  
   - 默认：`"reduce-overhead"`。
4. `decode_device`: `"auto" | "gpu" | "cpu"`  
   - 默认：`"auto"`。
5. `decode_gpu_index`: `int`  
   - 默认：`0`。

## 代码实现计划（MVP）

### 1) `pipeline/models.py`
- 增加轻量运行时 helper：
  - 精度策略解析（含 CPU 回退）
  - autocast 上下文
  - 可选 `torch.compile`
  - decode 设备策略与打开 `VideoReader` 的 fallback
- DOVER / UniMatch 推理路径改为：
  - `torch.inference_mode()`
  - BF16 autocast（按配置开启）
- DOVER / UniMatch 模型初始化支持可选 compile。
- 采样函数支持 `decode_device=auto`（GPU 失败回退 CPU）。

### 2) `configs/config_filter_egodex300g.yaml`
- 为 DOVER 与 UniMatch 添加上述 5 个模型级参数，给出 H100 推荐值。

### 3) 测试
- 至少执行现有模型相关单测，确保：
  - scorer 基本行为不变
  - 无新增回归

## 验收标准（首期）
1. 功能
- 4 个高 ROI 改动全部可由配置控制。
- decode 在 `auto` 模式下 GPU 失败可回退 CPU，不中断任务。

2. 质量
- keep/drop 逻辑与基线一致（允许极小浮点扰动）。
- `scoring_error` 占比不高于基线。

3. 性能
- 端到端吞吐目标：`>= 1.3x`（同机器、同数据、同阈值）。

## 后续阶段（Phase B）
- 若首期验证收益稳定，再推进：
1. 真批推理重构
2. 解码与推理流水线并发
3. 更细粒度 profiling 与自动调参
