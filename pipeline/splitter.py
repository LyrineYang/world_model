from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import logging

from scenedetect import SceneManager, open_video
from scenedetect.detectors import AdaptiveDetector
from scenedetect.frame_timecode import FrameTimecode
from tqdm import tqdm

from .config import SplitterConfig

try:
    import decord  # type: ignore
except Exception:  # noqa: BLE001
    decord = None

log = logging.getLogger(__name__)


@dataclass
class SceneSpan:
    start: float  # seconds
    end: float    # seconds


def detect_scenes(video_path: Path, cfg: SplitterConfig) -> List[SceneSpan]:
    if cfg.kind == "transnet":
        return detect_scenes_transnet(video_path, cfg)
    if cfg.kind == "transnet_dali":
        return detect_scenes_transnet_dali(video_path, cfg)
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


def _resolve_device(device_str: str) -> Tuple[str, int]:
    if device_str.startswith("cuda"):
        parts = device_str.split(":")
        idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        return "cuda", idx
    return "cpu", -1


def _maybe_download_weight(cfg: SplitterConfig, base_dir: Path) -> Path | None:
    if cfg.weight_path:
        path = Path(cfg.weight_path)
        if not path.is_absolute():
            path = base_dir / path
        if path.exists():
            return path
    if not cfg.hf_repo_id or not cfg.hf_filename:
        # 尝试从已安装的 transnet 包查找权重
        for mod_name in ("transnetv2_pytorch", "transnetv2"):
            try:
                mod = __import__(mod_name, fromlist=["__file__"])
                base = Path(mod.__file__).resolve().parent
                for p in list(base.glob("*.pth")) + list((base / "weights").glob("*.pth")):
                    if p.exists():
                        log.info("Found TransNetV2 weight bundled in %s: %s", mod_name, p)
                        return p
            except Exception:
                continue
        return None
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except Exception as exc:  # noqa: BLE001
        log.warning("Hugging Face hub not available for TransNet weight download: %s", exc)
        return None
    try:
        downloaded = hf_hub_download(
            repo_id=cfg.hf_repo_id,
            filename=cfg.hf_filename,
            repo_type="model",
            local_dir=str(base_dir / "transnet_weights"),
            local_dir_use_symlinks=False,
        )
        return Path(downloaded)
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to download TransNet weight from HF (%s/%s): %s", cfg.hf_repo_id, cfg.hf_filename, exc)
        return None


def _split_by_scores(
    scores: Iterable[float],
    fps: float,
    threshold: float,
    positions: Iterable[int] | None = None,
    total_frames: int | None = None,
) -> List[SceneSpan]:
    scores_list = list(scores)
    if positions is None:
        positions_list = list(range(len(scores_list)))
    else:
        positions_list = list(positions)
    if not scores_list or not positions_list:
        return [SceneSpan(start=0.0, end=0.0)]
    boundaries = [positions_list[i] for i, v in enumerate(scores_list) if v >= threshold]
    spans: List[SceneSpan] = []
    last_frame = positions_list[0]
    for b in boundaries:
        if b <= last_frame:
            continue
        spans.append(SceneSpan(start=last_frame / fps, end=b / fps))
        last_frame = b
    end_frame = total_frames - 1 if total_frames is not None else positions_list[-1]
    spans.append(SceneSpan(start=last_frame / fps, end=end_frame / fps))
    return spans


def detect_scenes_transnet(video_path: Path, cfg: SplitterConfig) -> List[SceneSpan]:
    """
    GPU 解码 + TransNetV2 预测；缺依赖或失败时退化为 GPU 帧差。
    """
    if decord is None:
        log.warning("Decord not available; falling back to PySceneDetect for %s", video_path.name)
        return detect_scenes(video_path, SplitterConfig(kind="pyscenedetect", threshold=cfg.threshold, min_scene_len=cfg.min_scene_len))

    device_type, device_idx = _resolve_device(cfg.device)
    use_gpu = device_type == "cuda" and device_idx >= 0
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # noqa: BLE001
        log.warning("Torch not available; falling back to PySceneDetect for %s: %s", video_path.name, exc)
        return detect_scenes(video_path, SplitterConfig(kind="pyscenedetect", threshold=cfg.threshold, min_scene_len=cfg.min_scene_len))

    if use_gpu and not torch.cuda.is_available():
        log.warning("CUDA not available; falling back to CPU scene detect for %s", video_path.name)
        return detect_scenes(video_path, SplitterConfig(kind="pyscenedetect", threshold=cfg.threshold, min_scene_len=cfg.min_scene_len))

    weight_path = _maybe_download_weight(cfg, video_path.parent)

    def _load_model():
        model = None
        load_exc: Exception | None = None
        for mod_name in ("transnetv2_pytorch", "transnetv2"):
            try:
                trans_module = __import__(mod_name, fromlist=["TransNetV2"])
                TransNetV2 = getattr(trans_module, "TransNetV2")
                model = TransNetV2()
                break
            except Exception as exc:  # noqa: BLE001
                load_exc = exc
                continue
        if model is None:
            raise ImportError(f"TransNetV2 not found (tried transnetv2_pytorch/transnetv2): {load_exc}")
        if weight_path and weight_path.exists():
            try:
                state = torch.load(weight_path, map_location=cfg.device)
                try:
                    model.load_state_dict(state, strict=False)
                except Exception:
                    model.load_state_dict(state)
                log.info("Loaded TransNetV2 weights from %s", weight_path)
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to load TransNetV2 weight %s: %s", weight_path, exc)
        model = model.to(cfg.device if use_gpu else "cpu").eval()
        return model

    def _extract_scores(pred, frame_count: int) -> List[float] | None:
        try:
            if isinstance(pred, (list, tuple)) and len(pred) > 0:
                pred = pred[0]
            if isinstance(pred, torch.Tensor):
                if pred.ndim == 3 and pred.shape[1] == frame_count and pred.shape[-1] >= 2:
                    scores = pred[0, :, 1]
                elif pred.ndim == 3 and pred.shape[2] == frame_count:
                    # assume [B, 2, T]
                    scores = pred[0, 1, :]
                elif pred.ndim == 2 and pred.shape[0] == frame_count:
                    scores = pred[:, -1]
                else:
                    scores = pred.reshape(-1)[:frame_count]
                return scores.detach().float().cpu().tolist()
        except Exception:
            return None
        return None

    def _fallback_diff(frames_tensor: torch.Tensor) -> List[float]:
        # frames: [N, 3, H, W], GPU tensor
        gray = frames_tensor.mean(dim=1)
        diffs = torch.abs(gray[1:] - gray[:-1]).mean(dim=(1, 2))
        if diffs.numel() == 0:
            return []
        diffs = diffs / (diffs.max() + 1e-6)
        diffs = torch.cat([diffs.new_tensor([0.0]), diffs])
        return diffs.detach().float().cpu().tolist()

    model = None
    try:
        model = _load_model()
    except Exception as exc:  # noqa: BLE001
        log.warning("TransNet model unavailable, using GPU frame-diff for %s: %s", video_path.name, exc)

    decord.bridge.set_bridge("torch")
    ctx = decord.gpu(device_idx) if use_gpu else decord.cpu(0)
    try:
        vr = decord.VideoReader(str(video_path), ctx=ctx)
    except Exception as exc:  # noqa: BLE001
        log.warning("Decord failed to open %s; falling back to PySceneDetect: %s", video_path.name, exc)
        return detect_scenes(video_path, SplitterConfig(kind="pyscenedetect", threshold=cfg.threshold, min_scene_len=cfg.min_scene_len))

    fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else 30.0
    total_frames = len(vr)
    chunk = max(int(cfg.chunk_size_frames), 1)
    stride = max(int(cfg.stride_frames), 1)
    batch_size = max(int(cfg.batch_size), 1)

    all_scores: List[float] = []
    all_positions: List[int] = []
    for start in range(0, total_frames, chunk):
        end = min(start + chunk, total_frames)
        indices = list(range(start, end, stride))
        if not indices:
            continue
        frames = vr.get_batch(indices)  # torch tensor on ctx, shape [N, H, W, 3]
        frames = frames.permute(0, 3, 1, 2).float() / 255.0  # [N,3,H,W]
        frames = torch.stack(
            [
                F.interpolate(frames[i : i + batch_size], size=(27, 48), mode="bilinear", align_corners=False)
                for i in range(0, frames.shape[0], batch_size)
            ]
        ).reshape(frames.shape[0], 3, 27, 48)
        frames = frames.to(cfg.device if use_gpu else "cpu")

        scores: List[float] | None = None
        if model is not None:
            try:
                with torch.no_grad():
                    preds = model(frames.unsqueeze(0))
                scores = _extract_scores(preds, frames.shape[0])
            except Exception as exc:  # noqa: BLE001
                log.warning("TransNet inference failed on %s chunk %d-%d: %s; falling back to frame-diff", video_path.name, start, end, exc)
        if scores is None:
            scores = _fallback_diff(frames)
        all_scores.extend(scores)
        all_positions.extend(indices[: len(scores)])

    spans = _split_by_scores(
        all_scores,
        fps if fps > 0 else 30.0,
        cfg.transnet_threshold or cfg.threshold,
        positions=all_positions,
        total_frames=total_frames,
    )
    return spans


def detect_scenes_transnet_dali(video_path: Path, cfg: SplitterConfig) -> List[SceneSpan]:
    """
    DALI + TransNetV2 版切分，需 nvidia-dali-cudaXX。
    """
    try:
        from modules.splitters.transnet_dali import DaliTransNetSplitter
    except Exception as exc:  # noqa: BLE001
        log.warning("DALI splitter unavailable (%s); falling back to PySceneDetect for %s", exc, video_path.name)
        return detect_scenes(video_path, SplitterConfig(kind="pyscenedetect", threshold=cfg.threshold, min_scene_len=cfg.min_scene_len))

    # 优先尝试使用同样的 HF 下载逻辑
    weight_path = _maybe_download_weight(cfg, video_path.parent)
    if not weight_path and not cfg.hf_repo_id:
        log.warning("No TransNetV2 weight provided; falling back to PySceneDetect for %s", video_path.name)
        return detect_scenes(video_path, SplitterConfig(kind="pyscenedetect", threshold=cfg.threshold, min_scene_len=cfg.min_scene_len))

    splitter = DaliTransNetSplitter(
        device=cfg.device,
        threshold=cfg.transnet_threshold or cfg.threshold,
        batch_size=cfg.batch_size,
        stride_frames=cfg.stride_frames,
        resize_hw=(27, 48),
        weight_path=str(weight_path) if weight_path else cfg.weight_path,
        hf_repo_id=cfg.hf_repo_id,
        hf_filename=cfg.hf_filename,
    )
    spans_float = splitter.detect_scenes(str(video_path))
    return [SceneSpan(start=s, end=e) for s, e in spans_float]


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
