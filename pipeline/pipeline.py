from __future__ import annotations

import logging
import shutil
from pathlib import Path

from tqdm import tqdm

from .config import Config, load_config, parse_args
from .filtering import materialize_results
from .flash import is_flashy
from .io import download_shard, extract_shard, list_video_files
from .models import ScoreResult, build_scorers, run_scorers
from .splitter import split_video_to_scenes
from .state import load_state, save_state
from .uploader import upload_shard

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


def process_shard(cfg: Config, shard: str) -> None:
    state = load_state(cfg.state_dir, shard)

    archive_path = cfg.downloads_dir / shard
    if not state["downloaded"]:
        log.info("Downloading shard %s", shard)
        archive_path = download_shard(cfg.source_repo, shard, cfg.downloads_dir)
        state["downloaded"] = True
        save_state(cfg.state_dir, shard, state)
    elif not archive_path.exists():
        # fallback to re-download if state says done but file missing
        log.info("Re-downloading missing shard file %s", shard)
        archive_path = download_shard(cfg.source_repo, shard, cfg.downloads_dir)

    extract_path = cfg.extract_dir / Path(shard).stem
    if not state["extracted"]:
        log.info("Extracting %s to %s", archive_path.name, extract_path)
        extract_path = extract_shard(archive_path, cfg.extract_dir)
        state["extracted"] = True
        save_state(cfg.state_dir, shard, state)

    video_files = list_video_files(extract_path, exclude_dirs={"scenes"})
    if not video_files:
        log.warning("No video files found in shard %s", shard)

    # 场景切分
    scenes_root = extract_path / "scenes"
    scene_clips: list[Path] = []
    split_failed: list[ScoreResult] = []
    for video in tqdm(video_files, desc="Scene split", unit="video"):
        try:
            clips = split_video_to_scenes(video, cfg.splitter, scenes_root)
            scene_clips.extend(clips)
            if cfg.splitter.remove_source_after_split:
                video.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("Scene split failed for %s: %s", video, exc)
            split_failed.append(ScoreResult(path=video, scores={}, keep=False, reason="split_failed"))

    if not scene_clips:
        log.warning("No scene clips found in shard %s", shard)

    # 闪烁过滤
    flash_dropped: list[ScoreResult] = []
    filtered_clips: list[Path] = []
    if cfg.flash_filter.enabled and scene_clips:
        for clip in tqdm(scene_clips, desc="Flash filter", unit="clip"):
            if is_flashy(clip, cfg.flash_filter):
                flash_dropped.append(ScoreResult(path=clip, scores={}, keep=False, reason="flash"))
            else:
                filtered_clips.append(clip)
    else:
        filtered_clips = scene_clips

    # OCR 文字过滤
    ocr_dropped: list[ScoreResult] = []
    ocr_filtered: list[Path] = []
    if cfg.ocr.enabled and filtered_clips:
        for clip in tqdm(filtered_clips, desc="OCR filter", unit="clip"):
            if has_text(clip, cfg.ocr):
                ocr_dropped.append(ScoreResult(path=clip, scores={}, keep=False, reason="ocr_text"))
            else:
                ocr_filtered.append(clip)
    else:
        ocr_filtered = filtered_clips

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
    log.info("Scoring %d clips with %d models", len(ocr_filtered), len(scorers))
    scored_results = run_scorers(scorers, ocr_filtered) if ocr_filtered else []
    results = split_failed + flash_dropped + ocr_dropped + scored_results
    state["scored"] = True
    save_state(cfg.state_dir, shard, state)

    log.info("Materializing results for %s", shard)
    metadata_path = materialize_results(shard, results, cfg.output_dir)
    log.info("Metadata written to %s", metadata_path)

    if not cfg.skip_upload:
        log.info("Uploading shard %s to %s", shard, cfg.target_repo)
        upload_shard(cfg.target_repo, shard, cfg.output_dir / shard, cfg.upload)
        state["uploaded"] = True
        save_state(cfg.state_dir, shard, state)

    cleanup_shard(cfg, shard)


def cleanup_shard(cfg: Config, shard: str) -> None:
    # 清理当前分片的中间文件，减少磁盘占用
    extract_path = cfg.extract_dir / Path(shard).stem
    if extract_path.exists():
        shutil.rmtree(extract_path, ignore_errors=True)
    # 可选：下载的压缩包也可清理；如需断点再用可保留
    archive_path = cfg.downloads_dir / shard
    if archive_path.exists():
        archive_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config), limit_shards=args.limit_shards, skip_upload=args.skip_upload)
    ensure_dirs(cfg)

    log.info("Starting pipeline for %d shard(s)", len(cfg.shards))
    for shard in tqdm(cfg.shards, desc="Shards", unit="shard"):
        try:
            process_shard(cfg, shard)
        except Exception as exc:  # noqa: BLE001
            log.exception("Shard %s failed: %s", shard, exc)
            # 保留状态，用户可重试
            continue


if __name__ == "__main__":
    main()
