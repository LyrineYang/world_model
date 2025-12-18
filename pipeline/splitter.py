from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from scenedetect import SceneManager, open_video
from scenedetect.detectors import AdaptiveDetector
from scenedetect.frame_timecode import FrameTimecode
from tqdm import tqdm

from .config import SplitterConfig


@dataclass
class SceneSpan:
    start: float  # seconds
    end: float    # seconds


def detect_scenes(video_path: Path, cfg: SplitterConfig) -> List[SceneSpan]:
    video = open_video(str(video_path))
    manager = SceneManager()
    manager.add_detector(
        AdaptiveDetector(
            adaptive_threshold=cfg.threshold,
            min_scene_len=cfg.min_scene_len,
        )
    )
    manager.detect_scenes(video)
    scenes = manager.get_scene_list()
    try:
        video.close()  # type: ignore[attr-defined]
    except Exception:
        pass

    spans: List[SceneSpan] = []
    for start, end in scenes:
        # end can be None for last scene; fallback to video duration
        if end is None:
            end_time = video.duration.get_seconds() if video.duration else float(start.get_seconds())
        else:
            end_time = end.get_seconds()
        spans.append(SceneSpan(start=start.get_seconds(), end=end_time))
    if not spans:
        duration = video.duration.get_seconds() if video.duration else 0.0
        spans.append(SceneSpan(start=0.0, end=duration))
    return spans


def cut_scenes(video_path: Path, spans: List[SceneSpan], out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []
    for idx, span in enumerate(tqdm(spans, desc=f"Cutting {video_path.name}", unit="scene")):
        out_path = out_dir / f"{video_path.stem}_scene_{idx:04d}{video_path.suffix}"
        if out_path.exists():
            outputs.append(out_path)
            continue
        duration = max(span.end - span.start, 0.001)
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(span.start),
            "-t",
            str(duration),
            "-i",
            str(video_path),
            "-c",
            "copy",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
        outputs.append(out_path)
    return outputs


def split_video_to_scenes(video_path: Path, cfg: SplitterConfig, scenes_root: Path) -> List[Path]:
    spans = detect_scenes(video_path, cfg)
    clip_paths = cut_scenes(video_path, spans, scenes_root)
    return clip_paths
