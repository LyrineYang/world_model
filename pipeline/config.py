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


@dataclass
class SplitterConfig:
    kind: str = "pyscenedetect"
    threshold: float = 27.0
    min_scene_len: int = 16
    remove_source_after_split: bool = False


@dataclass
class FlashFilterConfig:
    enabled: bool = True
    brightness_delta: float = 60.0
    max_flash_ratio: float = 0.2
    sample_stride: int = 5


@dataclass
class OCRConfig:
    enabled: bool = False
    text_area_threshold: float = 0.02
    sample_stride: int = 10
    lang: str = "ch"


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
    shards: List[str]
    models: List[ModelConfig]
    upload: UploadConfig
    splitter: SplitterConfig
    flash_filter: FlashFilterConfig
    ocr: OCRConfig
    ffmpeg: FFmpegConfig
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
    return parser.parse_args()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: Path, limit_shards: int | None = None, skip_upload: bool = False) -> Config:
    raw = _load_yaml(path)

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

    cfg = Config(
        source_repo=raw["source_repo"],
        target_repo=raw["target_repo"],
        workdir=Path(raw["workdir"]),
        shards=list(raw.get("shards", [])),
        models=models,
        upload=UploadConfig(**raw.get("upload", {})),
        splitter=SplitterConfig(**raw.get("splitter", {})),
        flash_filter=FlashFilterConfig(**raw.get("flash_filter", {})),
        ocr=OCRConfig(**raw.get("ocr", {})),
        ffmpeg=FFmpegConfig(**raw.get("ffmpeg", {})),
        limit_shards=limit_shards,
        skip_upload=skip_upload,
    )

    if cfg.limit_shards is not None:
        if cfg.limit_shards <= 0:
            cfg.shards = []
        else:
            cfg.shards = cfg.shards[: cfg.limit_shards]

    return cfg
