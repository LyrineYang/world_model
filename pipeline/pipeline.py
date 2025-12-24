from __future__ import annotations

import logging
import shutil
from pathlib import Path
import json
import threading
from queue import Queue
from queue import Empty as QueueEmpty

import pandas as pd
import time
from tqdm import tqdm

from .config import Config, RuntimeConfig, load_config, parse_args
from .calibration import compute_quantiles, write_calibration_parquet
from .filtering import materialize_results
from .flash import is_flashy
from .io import download_shard, extract_shard, list_video_files
from .models import ScoreResult, build_scorers, run_scorers
from .ocr_filter import has_text
from .splitter import detect_scenes, split_video_to_scenes
from .state import load_state, save_state
from .uploader import upload_shard
from .caption import generate_captions
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


def process_shard(
    cfg: Config,
    shard: str,
    calibration_remaining: int | None = None,
    shard_idx: int | None = None,
    total_shards: int | None = None,
) -> int:
    t_start = time.time()
    state = load_state(cfg.state_dir, shard)
    shard_tag = f"[{shard} {shard_idx}/{total_shards}]" if shard_idx and total_shards else f"[{shard}]"
    info_level = logging.DEBUG
    important_level = logging.INFO
    if "stage" not in state:
        state["stage"] = "pending"
    if "started_at" not in state:
        state["started_at"] = t_start

    def _info(msg: str, *args) -> None:
        log.log(info_level, msg, *args)

    def _important(msg: str, *args) -> None:
        log.log(important_level, msg, *args)

    def _mark(stage: str, **fields) -> None:
        now = time.time()
        state.update(fields)
        state["stage"] = stage
        state.setdefault("started_at", t_start)
        if stage in {"processed", "uploaded"}:
            state["finished_at"] = now
        save_state(cfg.state_dir, shard, state)

    # 若已上传且非校准/跳过上传，直接跳过该 shard
    if state.get("uploaded") and not cfg.skip_upload and not (cfg.calibration or {}).get("enabled", False):
        _important("%s already uploaded; skipping.", shard_tag)
        _mark("uploaded")
        return 0
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
        _info("%s downloading shard", shard_tag)
        _mark("downloading")
        archive_path = download_shard(cfg.source_repo, shard, cfg.downloads_dir, token=cfg.hf_token)
        state["downloaded"] = True
        _mark("extracting", downloaded=True)
        summary["time_download"] = time.time() - t_dl
    elif not archive_path.exists():
        # fallback to re-download if state says done but file missing
        _info("%s re-downloading missing shard file", shard_tag)
        _mark("downloading")
        archive_path = download_shard(cfg.source_repo, shard, cfg.downloads_dir, token=cfg.hf_token)

    extract_path = cfg.extract_dir / Path(shard).stem
    if not state["extracted"]:
        t_ex = time.time()
        _info("%s extracting to %s", shard_tag, extract_path)
        _mark("extracting")
        extract_path = extract_shard(archive_path, cfg.extract_dir)
        state["extracted"] = True
        _mark("processing", extracted=True, downloaded=True)
        summary["time_extract"] = time.time() - t_ex
    elif state.get("extracted") and state.get("stage") not in {"processing", "materializing", "uploading", "uploaded", "processed"}:
        _mark("processing", extracted=True, downloaded=True)

    video_files = list_video_files(extract_path, exclude_dirs={"scenes"})
    if not video_files:
        log.warning("No video files found in shard %s", shard)

    runtime_cfg: RuntimeConfig = getattr(cfg, "runtime", RuntimeConfig())
    scoring_workers = runtime_cfg.scoring_workers if runtime_cfg.scoring_workers else max(len(cfg.models), 1)
    queue_size = max(int(runtime_cfg.queue_size), 1)
    use_stream = bool(runtime_cfg.stream_processing)

    # 计算校准剩余上限
    target_remaining = None
    if calibration_enabled and calibration_sample_size > 0:
        target_remaining = calibration_sample_size if calibration_remaining is None else calibration_remaining

    scenes_root = extract_path / "scenes"
    split_failed: list[ScoreResult] = []
    flash_dropped: list[ScoreResult] = []
    ocr_dropped: list[ScoreResult] = []
    scored_results: list[ScoreResult] = []
    scene_count = 0
    flash_pass = 0
    ocr_pass = 0
    scene_time = 0.0
    flash_time = 0.0
    ocr_time = 0.0

    def _batch_step(scorers_obj) -> int:
        return min(max(getattr(s, "batch_size", 1), 1) for s in scorers_obj) if scorers_obj else 1

    def _build_scorers_safe() -> tuple[list, Exception | None]:
        try:
            return build_scorers(cfg.models), None
        except Exception as exc:  # noqa: BLE001
            log.exception("Scorer initialization failed: %s", exc)
            return [], exc

    if use_stream:
        sentinel = object()
        q: Queue[Path | object] = Queue(maxsize=queue_size)
        scorer_init_error: Exception | None = None
        scorers: list = []

        def produce() -> None:
            nonlocal scene_count, flash_pass, ocr_pass, scene_time, flash_time, ocr_time
            produced = 0
            sent = False
            try:
                for video in video_files:
                    try:
                        t0 = time.time()
                        spans = detect_scenes(video, cfg.splitter)
                        scene_time += time.time() - t0
                        if cfg.splitter.cut:
                            clips = cut_and_collect(video, spans, cfg, scenes_root)
                            if cfg.splitter.remove_source_after_split:
                                video.unlink(missing_ok=True)
                        else:
                            clips = [video]
                            extras[str(video)] = build_scene_windows(video, spans, cfg)
                    except Exception as exc:  # noqa: BLE001
                        log.warning("Scene detection failed for %s: %s", video, exc)
                        split_failed.append(ScoreResult(path=video, scores={}, keep=False, reason="split_failed"))
                        continue

                    scene_count += len(clips)
                    for clip in clips:
                        if cfg.flash_filter.enabled:
                            tf = time.time()
                            flash_hit = is_flashy(clip, cfg.flash_filter)
                            flash_time += time.time() - tf
                            extras.setdefault(str(clip), {})["flash_hit"] = flash_hit
                            if flash_hit and not cfg.flash_filter.record_only:
                                flash_dropped.append(ScoreResult(path=clip, scores={}, keep=False, reason="flash"))
                                continue
                        flash_pass += 1

                        if cfg.ocr.enabled:
                            tocr = time.time()
                            ocr_hit = has_text(clip, cfg.ocr)
                            ocr_time += time.time() - tocr
                            extras.setdefault(str(clip), {})["ocr_hit"] = ocr_hit
                            if ocr_hit and not cfg.ocr.record_only:
                                ocr_dropped.append(ScoreResult(path=clip, scores={}, keep=False, reason="ocr_text"))
                                continue
                        ocr_pass += 1
                        q.put(clip)
                        produced += 1
                        if target_remaining is not None and produced >= target_remaining:
                            q.put(sentinel)
                            sent = True
                            return
            except Exception as exc:  # noqa: BLE001
                log.exception("Producer failed unexpectedly: %s", exc)
            finally:
                if not sent:
                    q.put(sentinel)

        producer = threading.Thread(target=produce, daemon=True)
        producer.start()
        _info("%s scene/flash/OCR/score (streaming) started (%d video file(s))", shard_tag, len(video_files))

        batch: list[Path] = []
        t_score = time.time()

        while True:
            item = q.get()
            if item is sentinel:
                break
            batch.append(item)  # type: ignore[arg-type]
            # 延迟构建 scorer，避免在生产者启动前就加载模型
            if not scorers and scorer_init_error is None:
                scorers, scorer_init_error = _build_scorers_safe()
            step = _batch_step(scorers)
            if len(batch) >= step:
                if scorer_init_error:
                    scored_results.extend(
                        ScoreResult(path=p, scores={}, keep=False, reason="scorer_init_failed") for p in batch
                    )
                elif scorers:
                    scored_results.extend(run_scorers(scorers, batch, max_workers=scoring_workers))
                else:
                    scored_results.extend(run_scorers([], batch))
                calibration_counter += len(batch)
                batch = []

        if batch:
            if scorer_init_error:
                scored_results.extend(
                    ScoreResult(path=p, scores={}, keep=False, reason="scorer_init_failed") for p in batch
                )
            elif scorers:
                scored_results.extend(run_scorers(scorers, batch, max_workers=scoring_workers))
            else:
                scored_results.extend(run_scorers([], batch))
            calibration_counter += len(batch)

        producer.join()
        summary["time_score"] = time.time() - t_score
        summary["files"] = len(video_files)
        summary["clips_after_scene"] = scene_count
        summary["clips_after_flash"] = flash_pass
        summary["clips_after_ocr"] = ocr_pass
        summary["clips_scored"] = len(scored_results)
        summary["time_scene"] = scene_time
        summary["time_flash"] = flash_time
        summary["time_ocr"] = ocr_time
        results = split_failed + flash_dropped + ocr_dropped + scored_results

    else:
        # 旧的串行流程
        # 场景切分
        scenes_root = extract_path / "scenes"
        scene_clips: list[Path] = []
        split_failed = []
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

        _info("%s scene detection done: %d clips", shard_tag, len(scene_clips))

        # 闪烁过滤
        flash_dropped = []
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
        ocr_dropped = []
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
            scorer_init_error = None
        except Exception as exc:  # noqa: BLE001
            scorer_init_error = exc
            log.exception("Scorer initialization failed: %s", exc)
            scorers = []
        # 若校准模式启用，截断样本数量
        if calibration_enabled and calibration_sample_size > 0:
            limit = calibration_sample_size
            if calibration_remaining is not None:
                limit = max(min(calibration_remaining, calibration_sample_size), 0)
            if len(ocr_filtered) > limit:
                ocr_filtered = ocr_filtered[:limit]
        t_score = time.time()
        if scorer_init_error:
            scored_results = [
                ScoreResult(path=p, scores={}, keep=False, reason="scorer_init_failed") for p in ocr_filtered
            ]
        else:
            _info("Scoring %d clips with %d models", len(ocr_filtered), len(scorers))
            scored_results = run_scorers(scorers, ocr_filtered, max_workers=scoring_workers) if ocr_filtered else []
        calibration_counter += len(ocr_filtered)
        summary["time_score"] = time.time() - t_score
        summary["clips_scored"] = len(ocr_filtered)
        results = split_failed + flash_dropped + ocr_dropped + scored_results

    # Caption 生成仅针对保留的片段
    if cfg.caption.enabled:
        try:
            kept_paths = [res.path for res in results if res.keep and res.path.exists()]
            # 去重避免重复请求
            unique_kept = list(dict.fromkeys(kept_paths))
            caption_map = generate_captions(unique_kept, cfg.caption) if unique_kept else {}
            for path_str, caption in caption_map.items():
                extras.setdefault(path_str, {})["caption"] = caption
        except Exception as exc:  # noqa: BLE001
            log.warning("Caption generation failed for shard %s: %s", shard, exc)
    state["scored"] = True
    _mark("materializing", scored=True)

    _important("%s materializing results", shard_tag)
    metadata_path = materialize_results(
        shard, results, cfg.output_dir, extras=extras, resize_720p=cfg.upload.resize_720p
    )
    _important("%s metadata written to %s", shard_tag, metadata_path)

    summary["time_total"] = time.time() - t_start
    _important(
        "%s done: files=%d scenes=%d flash_pass=%d ocr_pass=%d scored=%d time=%.1fs",
        shard_tag,
        summary.get("files", 0),
        summary.get("clips_after_scene", 0),
        summary.get("clips_after_flash", 0),
        summary.get("clips_after_ocr", 0),
        summary.get("clips_scored", 0),
        summary.get("time_total", 0.0),
    )

    if calibration_enabled:
        _important("Calibration mode enabled; skipping upload")
        _mark("processed")
    elif not cfg.skip_upload:
        _important("%s uploading to %s", shard_tag, cfg.target_repo)
        _mark("uploading")
        upload_shard(cfg.target_repo, shard, cfg.output_dir / shard, cfg.upload)
        state["uploaded"] = True
        _mark("uploaded", uploaded=True)
    else:
        _mark("processed")

    cleanup_shard(cfg, shard)
    summary["time_total"] = time.time() - t_start
    _write_summary(cfg.output_dir / shard / "summary.json", summary)
    return calibration_counter


def prefetch_shard(cfg: Config, shard: str) -> None:
    """
    预取：仅做下载+解压，更新 state。
    """
    state = load_state(cfg.state_dir, shard)
    info_level = logging.DEBUG
    if "stage" not in state:
        state["stage"] = "pending"

    def _info(msg: str, *args) -> None:
        log.log(info_level, msg, *args)

    def _mark(stage: str, **fields) -> None:
        state.update(fields)
        state["stage"] = stage
        save_state(cfg.state_dir, shard, state)

    archive_path = cfg.downloads_dir / shard
    if not state.get("downloaded"):
        _info("[prefetch] Downloading shard %s", shard)
        _mark("downloading")
        archive_path = download_shard(cfg.source_repo, shard, cfg.downloads_dir, token=cfg.hf_token)
        state["downloaded"] = True
        _mark("extracting", downloaded=True)
    elif not archive_path.exists():
        _info("[prefetch] Re-downloading missing shard file %s", shard)
        _mark("downloading")
        archive_path = download_shard(cfg.source_repo, shard, cfg.downloads_dir, token=cfg.hf_token)
        state["downloaded"] = True
        _mark("extracting", downloaded=True)

    if not state.get("extracted"):
        _info("[prefetch] Extracting %s", archive_path.name)
        extract_shard(archive_path, cfg.extract_dir)
        state["extracted"] = True
        _mark("prefetched", extracted=True, downloaded=True)


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
    # 若未上传或未启用清理，保留本地文件
    calib_cfg = cfg.calibration or {}
    if cfg.skip_upload or calib_cfg.get("enabled", False):
        return
    if not getattr(cfg.upload, "cleanup_after_upload", False):
        return

    paths = [
        cfg.downloads_dir / shard,
        cfg.extract_dir / Path(shard).stem,
        cfg.output_dir / shard,
    ]
    for p in paths:
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.exists():
                p.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to cleanup %s: %s", p, exc)


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
    info_level = logging.INFO
    important_level = logging.INFO
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

    log.log(important_level, "Starting pipeline for %d shard(s)", len(cfg.shards))
    if cfg.runtime.prefetch_shards > 0 and cfg.runtime.download_workers > 0:
        total_calibrated = _run_with_prefetch(cfg)
    else:
        total_calibrated = 0
        for idx, shard in enumerate(cfg.shards, start=1):
            try:
                remaining = _calibration_remaining(cfg, total_calibrated)
                added = process_shard(
                    cfg,
                    shard,
                    calibration_remaining=remaining,
                    shard_idx=idx,
                    total_shards=len(cfg.shards),
                )
                total_calibrated += added or 0
                # 校准模式下达到样本上限则早停
                if _should_stop_calibration(cfg, total_calibrated):
                    log.log(important_level, "Calibration sample size reached globally (%d); stopping", total_calibrated)
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
            log.log(info_level, "Calibration parquet written to %s", output_path)
            quantiles = calib_cfg.get("quantiles", [0.4, 0.7])
            try:
                if not df_scores.empty:
                    qs = compute_quantiles(output_path, quantiles)
                    log.log(info_level, "Calibration quantiles: %s", qs)
                else:
                    log.warning("No scored entries for calibration quantiles")
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to compute quantiles: %s", exc)


def _calibration_remaining(cfg: Config, total_calibrated: int) -> int | None:
    calib_cfg = cfg.calibration or {}
    if not calib_cfg.get("enabled", False):
        return None
    sample_size = int(calib_cfg.get("sample_size", 0))
    if sample_size <= 0:
        return None
    return max(sample_size - total_calibrated, 0)


def _should_stop_calibration(cfg: Config, total_calibrated: int) -> bool:
    calib = cfg.calibration or {}
    if not calib.get("enabled", False):
        return False
    sample_size = int(calib.get("sample_size", 0))
    return sample_size > 0 and total_calibrated >= sample_size


def _run_with_prefetch(cfg: Config) -> int:
    """
    分片级流水：下载/解压线程预取，主线程按完成顺序处理并上传。
    """
    important_level = logging.INFO
    download_queue: Queue[str] = Queue()
    ready_queue: Queue[str | object] = Queue(maxsize=max(cfg.runtime.prefetch_shards, 1))
    stop_event = threading.Event()
    sentinel = object()
    total_calibrated = 0
    total_shards = len(cfg.shards)
    if total_shards == 0:
        return 0

    progress_lock = threading.Lock()
    prefetch_bar = tqdm(total=total_shards, desc="Prefetch (dl+extract)", position=0, leave=False)
    process_bar = tqdm(total=total_shards, desc="Processed/uploaded", position=1, leave=False)

    for shard in cfg.shards:
        download_queue.put(shard)

    def downloader() -> None:
        while not stop_event.is_set():
            try:
                shard = download_queue.get_nowait()
            except QueueEmpty:
                break
            try:
                prefetch_shard(cfg, shard)
                ready_queue.put(shard)
            except Exception as exc:  # noqa: BLE001
                log.exception("Prefetch failed for %s: %s", shard, exc)
                # 仍然放入队列让主线程尝试处理并记录失败
                ready_queue.put(shard)
            finally:
                with progress_lock:
                    prefetch_bar.update(1)
                download_queue.task_done()
        ready_queue.put(sentinel)

    workers = max(int(cfg.runtime.download_workers), 1)
    threads = [threading.Thread(target=downloader, daemon=True) for _ in range(workers)]
    for t in threads:
        t.start()

    sentinels_seen = 0
    processed = 0
    stopping = False  # 校准达到上限后不再处理，但继续消费队列防止阻塞
    while sentinels_seen < workers and processed < total_shards:
        item = ready_queue.get()
        if item is sentinel:
            sentinels_seen += 1
            continue
        shard = item  # type: ignore[assignment]
        shard_idx = processed + 1
        if stopping or stop_event.is_set():
            # 达到校准上限后，仅消费队列，避免下载线程阻塞在 ready_queue.put
            processed += 1
            with progress_lock:
                process_bar.update(1)
            continue
        try:
            remaining = _calibration_remaining(cfg, total_calibrated)
            added = process_shard(cfg, shard, calibration_remaining=remaining, shard_idx=shard_idx, total_shards=total_shards)
            total_calibrated += added or 0
            processed += 1
            if _should_stop_calibration(cfg, total_calibrated):
                stop_event.set()
                stopping = True
                log.log(important_level, "Calibration sample size reached globally (%d); stopping", total_calibrated)
            with progress_lock:
                process_bar.update(1)
        except Exception as exc:  # noqa: BLE001
            log.exception("Shard %s failed: %s", shard, exc)
            processed += 1
            with progress_lock:
                process_bar.update(1)
            continue

    # 继续消费队列直到所有下载线程发出哨兵，避免它们在队列满时阻塞
    while sentinels_seen < workers:
        item = ready_queue.get()
        if item is sentinel:
            sentinels_seen += 1
            continue
        processed += 1  # 对未处理的 shard 计入已消费
        with progress_lock:
            process_bar.update(1)

    stop_event.set()
    for t in threads:
        t.join()
    prefetch_bar.close()
    process_bar.close()
    return total_calibrated


if __name__ == "__main__":
    main()
