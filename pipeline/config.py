import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class ModelConfig:
    name: str
    kind: str
    threshold: float
    device: str
    batch_size: int = 1
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UploadConfig:
    chunk_size_mb: int = 512
    max_workers: int = 3
    resize_720p: bool = False
    cleanup_after_upload: bool = False  # 上传成功后是否清理本地 download/extract/output 以节省磁盘


@dataclass
class SplitterConfig:
    kind: str = "pyscenedetect"
    threshold: float = 27.0
    min_scene_len: int = 16
    remove_source_after_split: bool = False
    cut: bool = False  # 是否物理切割场景
    window_len_frames: int = 121  # 虚拟切片长度（帧）
    window_stride_frames: int = 60  # 虚拟切片步长（帧）


@dataclass
class FlashFilterConfig:
    enabled: bool = True
    brightness_delta: float = 60.0
    max_flash_ratio: float = 0.2
    sample_stride: int = 5
    record_only: bool = False  # 若为 True，则仅记录命中，不剔除


@dataclass
class OCRConfig:
    enabled: bool = False
    text_area_threshold: float = 0.02
    sample_stride: int = 10
    lang: str = "ch"
    use_gpu: bool = False
    record_only: bool = False  # 若为 True，则仅记录命中，不剔除


@dataclass
class CaptionConfig:
    enabled: bool = False
    provider: str = "api"  # api | openrouter（openai 兼容接口）
    api_url: str | None = None
    api_key: str | None = None
    api_key_header: str = "Authorization"
    model: str = "gpt-4o"  # 默认视觉多模态
    system_prompt: str = "你是视频内容描述助手，用简洁中文总结视频。"
    user_prompt: str = "为这段视频生成一句中文描述（不超过30字）。如有多场景请概括主要内容。"
    max_tokens: int = 120
    temperature: float = 0.2
    timeout: float = 60.0
    max_workers: int = 2
    retry: int = 1
    file_field: str = "file"
    response_field: str = "caption"
    extra_fields: Dict[str, Any] = field(default_factory=dict)
    include_image: bool = True  # openrouter: 若可读取视频帧则附带图片
    image_max_side: int = 512   # openrouter: 降采样最长边，降低 payload
    openrouter_referer: str | None = None  # 可选：OpenRouter 推荐传递
    openrouter_title: str | None = None


@dataclass
class RuntimeConfig:
    # 是否启用流水线流式处理（边切分/过滤边打分）
    stream_processing: bool = True
    # scorer 并行线程数，0 表示按模型数量自动
    scoring_workers: int = 0
    # 生产者-消费者队列长度，避免占用过多内存
    queue_size: int = 16
    # 分片预取数量（>0 时开启下载预取，下载完成即进入处理队列）
    prefetch_shards: int = 2
    # 下载预取并发数
    download_workers: int = 2


@dataclass
class FFmpegConfig:
    audio_sample_rate: int = 16000
    max_width: int = 480
    max_height: int = 360


@dataclass
class Config:
    source_repo: str
    target_repo: str
    workdir: Path
    hf_token: str | None
    shards: List[str]
    shards_file: Path | None
    models: List[ModelConfig]
    upload: UploadConfig
    splitter: SplitterConfig
    flash_filter: FlashFilterConfig
    ocr: OCRConfig
    caption: CaptionConfig
    runtime: RuntimeConfig
    ffmpeg: FFmpegConfig
    calibration: dict[str, Any] = field(default_factory=dict)
    limit_shards: int | None = None
    skip_upload: bool = False

    @property
    def downloads_dir(self) -> Path:
        return self.workdir / "downloads"

    @property
    def extract_dir(self) -> Path:
        return self.workdir / "extract"

    @property
    def output_dir(self) -> Path:
        return self.workdir / "output"

    @property
    def state_dir(self) -> Path:
        return self.workdir / "state"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HF video filtering pipeline")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument(
        "--limit-shards",
        type=int,
        default=None,
        help="Process only the first N shards",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Process without uploading to HF",
    )
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="Enable calibration mode (no upload, sample clips for score distribution)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Calibration sample size (number of clips to score)",
    )
    parser.add_argument(
        "--calibration-output",
        type=str,
        default=None,
        help="Output path for calibration parquet (optional)",
    )
    parser.add_argument(
        "--calibration-quantiles",
        type=str,
        default=None,
        help="Comma separated quantiles for calibration report, e.g. 0.4,0.7",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: Path, limit_shards: int | None = None, skip_upload: bool = False) -> Config:
    raw = _load_yaml(path)
    base_dir = path.parent

    models = [
        ModelConfig(
            name=m["name"],
            kind=m["kind"],
            threshold=float(m.get("threshold", 0.0)),
            device=m.get("device", "cpu"),
            batch_size=int(m.get("batch_size", 1)),
            extra={k: v for k, v in m.items() if k not in {"name", "kind", "threshold", "device", "batch_size"}},
        )
        for m in raw.get("models", [])
    ]

    shards_file = raw.get("shards_file")
    sf_path: Path | None = None
    shards: List[str] = []
    if shards_file:
        sf_path = Path(shards_file)
        if not sf_path.is_absolute():
            sf_path = base_dir / sf_path
        if sf_path.exists():
            with sf_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    shards.append(line)
        else:
            raise FileNotFoundError(f"shards_file not found: {sf_path}")
    else:
        shards = list(raw.get("shards", []))

    cfg = Config(
        source_repo=raw["source_repo"],
        target_repo=raw["target_repo"],
        workdir=Path(raw["workdir"]),
        hf_token=raw.get("hf_token"),
        shards=shards,
        shards_file=sf_path if sf_path else None,
        models=models,
        upload=UploadConfig(**raw.get("upload", {})),
        splitter=SplitterConfig(**raw.get("splitter", {})),
        flash_filter=FlashFilterConfig(**raw.get("flash_filter", {})),
        ocr=OCRConfig(**raw.get("ocr", {})),
        caption=CaptionConfig(**raw.get("caption", {})),
        runtime=RuntimeConfig(**raw.get("runtime", {})),
        ffmpeg=FFmpegConfig(**raw.get("ffmpeg", {})),
        calibration=raw.get("calibration", {}),
        limit_shards=limit_shards,
        skip_upload=skip_upload,
    )

    if cfg.limit_shards is not None:
        if cfg.limit_shards <= 0:
            cfg.shards = []
        else:
            cfg.shards = cfg.shards[: cfg.limit_shards]

    return cfg
