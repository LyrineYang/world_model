from __future__ import annotations

import logging
import shutil
from pathlib import Path
import json

import pandas as pd
import time
from tqdm import tqdm

from .config import Config, load_config, parse_args
from .calibration import compute_quantiles, write_calibration_parquet
from .filtering import materialize_results
from .flash import is_flashy
from .io import download_shard, extract_shard, list_video_files
from .models import ScoreResult, build_scorers, run_scorers
from .ocr_filter import has_text
from .splitter import detect_scenes, split_video_to_scenes
from .state import load_state, save_state
from .uploader import upload_shard
try:
    import decord
except Exception as exc:  # noqa: BLE001
    decord = None
    log = logging.getLogger(__name__)
    log.warning("Decord not available in pipeline: %s", exc)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


def ensure_dirs(cfg: Config) -> None:
    cfg.workdir.mkdir(parents=True, exist_ok=True)
    cfg.downloads_dir.mkdir(parents=True, exist_ok=True)
    cfg.extract_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.state_dir.mkdir(parents=True, exist_ok=True)


def process_shard(cfg: Config, shard: str, calibration_remaining: int | None = None) -> int:
    t_start = time.time()
    state = load_state(cfg.state_dir, shard)
    calibration_cfg = cfg.calibration or {}
    calibration_enabled = calibration_cfg.get("enabled", False)
    calibration_sample_size = int(calibration_cfg.get("sample_size", 10000))
    calibration_output = calibration_cfg.get("output", None)
    calibration_counter = 0
    extras: dict[str, dict] = {}
    summary = {
        "shard": shard,
        "files": 0,
        "clips_after_scene": 0,
        "clips_after_flash": 0,
        "clips_after_ocr": 0,
        "clips_scored": 0,
        "time_download": 0.0,
        "time_extract": 0.0,
        "time_scene": 0.0,
        "time_flash": 0.0,
        "time_ocr": 0.0,
        "time_score": 0.0,
        "time_total": 0.0,
    }

    archive_path = cfg.downloads_dir / shard
    if not state["downloaded"]:
        t_dl = time.time()
        log.info("Downloading shard %s", shard)
        archive_path = download_shard(cfg.source_repo, shard, cfg.downloads_dir)
        state["downloaded"] = True
        save_state(cfg.state_dir, shard, state)
        summary["time_download"] = time.time() - t_dl
    elif not archive_path.exists():
        # fallback to re-download if state says done but file missing
        log.info("Re-downloading missing shard file %s", shard)
        archive_path = download_shard(cfg.source_repo, shard, cfg.downloads_dir)

    extract_path = cfg.extract_dir / Path(shard).stem
    if not state["extracted"]:
        t_ex = time.time()
        log.info("Extracting %s to %s", archive_path.name, extract_path)
        extract_path = extract_shard(archive_path, cfg.extract_dir)
        state["extracted"] = True
        save_state(cfg.state_dir, shard, state)
        summary["time_extract"] = time.time() - t_ex

    video_files = list_video_files(extract_path, exclude_dirs={"scenes"})
    if not video_files:
        log.warning("No video files found in shard %s", shard)

    # 计算校准剩余上限
    target_remaining = None
    if calibration_enabled and calibration_sample_size > 0:
        target_remaining = calibration_sample_size if calibration_remaining is None else calibration_remaining

    # 场景切分
    scenes_root = extract_path / "scenes"
    scene_clips: list[Path] = []
    split_failed: list[ScoreResult] = []
    t_scene = time.time()
    for video in tqdm(video_files, desc="Scene detect", unit="video"):
        try:
            spans = detect_scenes(video, cfg.splitter)
            if cfg.splitter.cut:
                clips = cut_and_collect(video, spans, cfg, scenes_root)
                scene_clips.extend(clips)
                if cfg.splitter.remove_source_after_split:
                    video.unlink(missing_ok=True)
            else:
                scene_clips.append(video)
                # 记录场景与窗口信息
                extras[str(video)] = build_scene_windows(video, spans, cfg)
        except Exception as exc:  # noqa: BLE001
            log.warning("Scene detection failed for %s: %s", video, exc)
            split_failed.append(ScoreResult(path=video, scores={}, keep=False, reason="split_failed"))
        if target_remaining is not None and len(scene_clips) >= target_remaining:
            break
    summary["time_scene"] = time.time() - t_scene
    summary["files"] = len(video_files)
    summary["clips_after_scene"] = len(scene_clips)

    if not scene_clips:
        log.warning("No scene clips found in shard %s", shard)

    # 闪烁过滤
    flash_dropped: list[ScoreResult] = []
    filtered_clips: list[Path] = []
    t_flash = time.time()
    if cfg.flash_filter.enabled and scene_clips:
        for clip in tqdm(scene_clips, desc="Flash filter", unit="clip"):
            hit = is_flashy(clip, cfg.flash_filter)
            extras.setdefault(str(clip), {})["flash_hit"] = hit
            if hit and not cfg.flash_filter.record_only:
                flash_dropped.append(ScoreResult(path=clip, scores={}, keep=False, reason="flash"))
            else:
                filtered_clips.append(clip)
    else:
        filtered_clips = scene_clips
    if target_remaining is not None and len(filtered_clips) > target_remaining:
        filtered_clips = filtered_clips[:target_remaining]
    summary["time_flash"] = time.time() - t_flash
    summary["clips_after_flash"] = len(filtered_clips)

    # OCR 文字过滤
    ocr_dropped: list[ScoreResult] = []
    ocr_filtered: list[Path] = []
    t_ocr = time.time()
    if cfg.ocr.enabled and filtered_clips:
        for clip in tqdm(filtered_clips, desc="OCR filter", unit="clip"):
            hit = has_text(clip, cfg.ocr)
            extras.setdefault(str(clip), {})["ocr_hit"] = hit
            if hit and not cfg.ocr.record_only:
                ocr_dropped.append(ScoreResult(path=clip, scores={}, keep=False, reason="ocr_text"))
            else:
                ocr_filtered.append(clip)
    else:
        ocr_filtered = filtered_clips
    if target_remaining is not None and len(ocr_filtered) > target_remaining:
        ocr_filtered = ocr_filtered[:target_remaining]
    summary["time_ocr"] = time.time() - t_ocr
    summary["clips_after_ocr"] = len(ocr_filtered)

    try:
        scorers = build_scorers(cfg.models)
    except Exception as exc:  # noqa: BLE001
        log.exception("Scorer initialization failed: %s", exc)
        # 标记整片分片为失败，避免静默退出
        failed_result = [
            ScoreResult(path=p, scores={}, keep=False, reason="scorer_init_failed") for p in ocr_filtered
        ]
        metadata_path = materialize_results(shard, failed_result, cfg.output_dir)
        log.info("Metadata written to %s", metadata_path)
        return
    # 若校准模式启用，截断样本数量
    if calibration_enabled and calibration_sample_size > 0:
        limit = calibration_sample_size
        if calibration_remaining is not None:
            limit = max(min(calibration_remaining, calibration_sample_size), 0)
        if len(ocr_filtered) > limit:
            ocr_filtered = ocr_filtered[:limit]
    t_score = time.time()
    log.info("Scoring %d clips with %d models", len(ocr_filtered), len(scorers))
    scored_results = run_scorers(scorers, ocr_filtered) if ocr_filtered else []
    calibration_counter += len(ocr_filtered)
    summary["time_score"] = time.time() - t_score
    summary["clips_scored"] = len(ocr_filtered)
    results = split_failed + flash_dropped + ocr_dropped + scored_results
    state["scored"] = True
    save_state(cfg.state_dir, shard, state)

    log.info("Materializing results for %s", shard)
    metadata_path = materialize_results(
        shard, results, cfg.output_dir, extras=extras, resize_720p=cfg.upload.resize_720p
    )
    log.info("Metadata written to %s", metadata_path)

    if calibration_enabled:
        log.info("Calibration mode enabled; skipping upload")
    elif not cfg.skip_upload:
        log.info("Uploading shard %s to %s", shard, cfg.target_repo)
        upload_shard(cfg.target_repo, shard, cfg.output_dir / shard, cfg.upload)
        state["uploaded"] = True
        save_state(cfg.state_dir, shard, state)

    cleanup_shard(cfg, shard)
    summary["time_total"] = time.time() - t_start
    _write_summary(cfg.output_dir / shard / "summary.json", summary)
    return calibration_counter


def cut_and_collect(video: Path, spans, cfg: Config, scenes_root: Path) -> list[Path]:
    # 目前直接使用现有切分函数；spans 仅用于未来精细切分
    clips = split_video_to_scenes(video, cfg.splitter, scenes_root)
    return clips


def build_scene_windows(video: Path, spans, cfg: Config) -> dict:
    if decord is None:
        return {}
    try:
        vr = decord.VideoReader(str(video))
        total_frames = len(vr)
        try:
            w, h = vr[0].shape[1], vr[0].shape[0]
        except Exception:
            w, h = None, None
        try:
            fps = float(vr.get_avg_fps())
        except Exception:
            fps = None
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to read video for window metadata %s: %s", video, exc)
        return {}

    if not fps or fps <= 0:
        # 无法获取有效 fps 时，跳过窗口计算
        return {"fps": None, "total_frames": total_frames, "width": w, "height": h}

    if not spans:
        # 若未检测到场景，视为一个全片场景
        spans = [type("Span", (), {"start": 0.0, "end": total_frames / fps if fps else 0.0})()]

    windows: list[list[int]] = []
    scenes_meta: list[dict] = []
    win_len = max(cfg.splitter.window_len_frames, 1)
    stride = max(cfg.splitter.window_stride_frames, 1)

    for span in spans:
        if fps:
            start_f = int(span.start * fps)
            end_f = max(int(span.end * fps) - 1, start_f)
        else:
            start_f = 0
            end_f = max(total_frames - 1, 0)

        scene_windows: list[list[int]] = []
        if end_f - start_f + 1 >= win_len:
            s = start_f
            while s + win_len - 1 <= end_f:
                e = s + win_len - 1
                scene_windows.append([s, e])
                windows.append([s, e])
                s += stride
        scenes_meta.append(
            {
                "start_frame": start_f,
                "end_frame": end_f,
                "num_windows": len(scene_windows),
                "windows": scene_windows,
            }
        )

    return {
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": total_frames / fps if fps else None,
        "width": w,
        "height": h,
        "num_windows": len(windows),
        "windows": windows,
        "scenes": scenes_meta,
    }


def cleanup_shard(cfg: Config, shard: str) -> None:
    # 暂停自动删除，避免本地分片与中间文件被意外清理
    return


def _write_summary(path: Path, data: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to write summary to %s: %s", path, exc)


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config), limit_shards=args.limit_shards, skip_upload=args.skip_upload)
    # 解析校准 CLI 参数
    if getattr(args, "calibration", False):
        cfg.calibration = cfg.calibration or {}
        cfg.calibration["enabled"] = True
        cfg.skip_upload = True
    calib_override = cfg.calibration or {}
    if getattr(args, "sample_size", None) is not None:
        calib_override["sample_size"] = int(args.sample_size)
    if getattr(args, "calibration_output", None):
        calib_override["output"] = args.calibration_output
    if getattr(args, "calibration_quantiles", None):
        try:
            calib_override["quantiles"] = [float(x) for x in args.calibration_quantiles.split(",") if x]
        except Exception:
            log.warning("Failed to parse calibration_quantiles %s", args.calibration_quantiles)
    cfg.calibration = calib_override
    ensure_dirs(cfg)

    log.info("Starting pipeline for %d shard(s)", len(cfg.shards))
    total_calibrated = 0
    for shard in tqdm(cfg.shards, desc="Shards", unit="shard"):
        try:
            calib_cfg = cfg.calibration or {}
            remaining = None
            if calib_cfg.get("enabled", False) and calib_cfg.get("sample_size", 0) > 0:
                remaining = max(int(calib_cfg["sample_size"]) - total_calibrated, 0)
            added = process_shard(cfg, shard, calibration_remaining=remaining)
            total_calibrated += added or 0
            # 校准模式下达到样本上限则早停
            calib = cfg.calibration or {}
            if calib.get("enabled", False) and calib.get("sample_size", 0) > 0:
                if total_calibrated >= int(calib["sample_size"]):
                    log.info("Calibration sample size reached globally (%d); stopping", total_calibrated)
                    break
        except Exception as exc:  # noqa: BLE001
            log.exception("Shard %s failed: %s", shard, exc)
            # 保留状态，用户可重试
            continue

    # 如果是校准模式并指定输出与分位
    calib_cfg = cfg.calibration or {}
    if calib_cfg.get("enabled", False):
        output_path = Path(calib_cfg.get("output", cfg.output_dir / "calibration_meta.parquet"))
        # 汇总所有 metadata.jsonl 到一个 Parquet（这里简单读 output_dir 下所有 shard/metadata.jsonl）
        rows = []
        for meta_file in cfg.output_dir.rglob("metadata.jsonl"):
            with meta_file.open("r", encoding="utf-8") as f:
                for line in f:
                    rows.append(line)
        if rows:
            # 写原始 jsonl 行到 parquet
            df = pd.read_json("\n".join(rows), lines=True)
            # 仅保留有 scores 的行以计算分位
            df_scores = df[df["scores"].notna()]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_scores.to_parquet(output_path, index=False)
            log.info("Calibration parquet written to %s", output_path)
            quantiles = calib_cfg.get("quantiles", [0.4, 0.7])
            try:
                if not df_scores.empty:
                    qs = compute_quantiles(output_path, quantiles)
                    log.info("Calibration quantiles: %s", qs)
                else:
                    log.warning("No scored entries for calibration quantiles")
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to compute quantiles: %s", exc)


if __name__ == "__main__":
    main()
