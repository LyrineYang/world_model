# 工作记录（Base Pool 管线）

## 已完成
- 引入 PySceneDetect 自适应切分（默认 th=27，min_scene_len=16），替代 ffmpeg scene。
- 新增闪烁过滤（亮度跳变检测），标记为 reason=flash。
- 管线按分片：下载 → 解压 → 切分 → 闪烁过滤 → 模型评分 → 筛选 → 元数据 → 上传。
- 模型接口扩展：DOVER、LAION-AES 已接入真实推理（默认权重/配置），Dummy 用于连通性；元数据包含丢弃原因。
- 文档与配置更新：README、config.example.yaml、requirements.txt。
- 增加 WORKDIR 打包提示，说明 torch 需按 GPU 手动安装。
- README 增补模型准备：可将 DOVER/aesthetic-predictor 随项目打包或指定外部路径；DOVER 权重自动从 HF 拉取，AES 线性头已随仓库提供。
- 接入 UniMatch 光流 scorer（运动过滤）：需手动下载 `unimatch` 权重（如 gmflow-scale1-mixdata），配置示例已加入。
- 修复与安全/可观测性：
  - safe extract（zip/tar）阻止路径穿越
  - metadata 指向输出副本并保留 source_path
  - run_scorers 检测 NaN 评分并标记 `scoring_error`
  - 分片切分失败落地 reason=split_failed
  - limit_shards=0 时不处理任何分片
  - 模型路径默认相对 REPO_ROOT，避免 cwd 依赖

## 待办（需要模型/资源信息）
- 质量/运动类评分（可选）：VMAF 或轻量光流 scorer，TransNet V2 软转场（HQ 阶段）。
- 参数调优：闪烁阈值、分片并发、GPU 绑定策略；DOVER/AES 的采样帧数、阈值。

## 接下来计划
1) 验证 DOVER/AES/UniMatch 在小样本上的速度与显存，占用结果写入 log。
2) 评估是否加入 VMAF（libvmaf）或 CPU/Farneback 版光流，及 TransNet V2（HQ 阶段）。
3) 按磁盘预算调整 `remove_source_after_split` 默认值，并在 README 增补推荐配置。
