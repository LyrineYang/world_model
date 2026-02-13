from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import queue
import shutil
import threading
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, List, TextIO

from .config import load_config
from .io import VIDEO_EXTS
from .records import ClipRecord

if TYPE_CHECKING:
    from .models import ScoreResult

log = logging.getLogger(__name__)
CONSISTENCY_STUB_KIND_ALIASES = {
    # Backward compatibility for old placeholder names.
    "clip_consistency_stub": "clip_text_consistency",
    "egovideo_consistency_stub": "egovideo_text_consistency",
}
CONSISTENCY_STUB_KINDS = {
    *CONSISTENCY_STUB_KIND_ALIASES.keys(),
    "clip_text_consistency",
    "egovideo_text_consistency",
}


def _consistency_pending_kinds(model_kinds: list[str]) -> list[str]:
    pending: list[str] = []
    for kind in model_kinds:
        if kind not in CONSISTENCY_STUB_KINDS:
            continue
        pending.append(CONSISTENCY_STUB_KIND_ALIASES.get(kind, kind))
    return list(dict.fromkeys(pending))


def _env_true(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off", ""}


def _configure_torch_runtime() -> None:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        log.warning("torch runtime config skipped: %s", exc)
        return

    if not torch.cuda.is_available():
        return

    enable_tf32 = _env_true("EGODEX_TF32", True)
    cudnn_benchmark = _env_true("EGODEX_CUDNN_BENCHMARK", True)

    try:
        torch.backends.cuda.matmul.allow_tf32 = enable_tf32
    except Exception:  # noqa: BLE001
        pass

    try:
        torch.backends.cudnn.allow_tf32 = enable_tf32
        torch.backends.cudnn.benchmark = cudnn_benchmark
    except Exception:  # noqa: BLE001
        pass

    set_precision = getattr(torch, "set_float32_matmul_precision", None)
    if set_precision is not None:
        try:
            set_precision("high" if enable_tf32 else "highest")
        except Exception:  # noqa: BLE001
            pass

    log.info("Torch runtime tuned: tf32=%s cudnn_benchmark=%s", enable_tf32, cudnn_benchmark)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline video cleaning runner (local files only)")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument(
        "--input-dir",
        action="append",
        default=[],
        help="Input directory containing videos (can be repeated)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Optional csv/jsonl manifest with at least a video path field",
    )
    parser.add_argument("--path-field", type=str, default="video_path", help="Manifest path field name")
    parser.add_argument("--text-field", type=str, default="text", help="Manifest text field name")
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=False,
        help="Recursively scan input dirs (default: off)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N records after loading inputs",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Override selection.strategy expression in config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: <workdir>/output/offline_<timestamp>)",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["link", "copy"],
        default="link",
        help="How to materialize kept videos into output/videos",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume from existing metadata.jsonl in output dir (default: on)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume and start writing metadata from scratch",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=4,
        help="Flush metadata file every N scored batches (default: 4)",
    )
    parser.add_argument(
        "--runlog-every",
        type=int,
        default=10,
        help="Update runlog every N scored batches (default: 10)",
    )
    parser.add_argument(
        "--write-buffer-records",
        type=int,
        default=128,
        help="Buffer N records before writing metadata (default: 128)",
    )
    return parser.parse_args()


def _iter_video_files(root: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                yield p
        return
    for p in root.iterdir():
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            yield p


def _coerce_meta_value(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        if text.lower() in {"true", "false"}:
            return text.lower() == "true"
        try:
            if "." in text:
                return float(text)
            return int(text)
        except ValueError:
            return value
    return value


def _load_manifest_records(manifest_path: Path, path_field: str, text_field: str) -> List[ClipRecord]:
    records: List[ClipRecord] = []
    suffix = manifest_path.suffix.lower()

    def _resolve_path(raw_path: str) -> Path:
        p = Path(raw_path).expanduser()
        if not p.is_absolute():
            p = (manifest_path.parent / p).resolve()
        return p

    if suffix == ".jsonl":
        with manifest_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if path_field not in payload:
                    raise ValueError(f"Manifest missing `{path_field}` at line {line_no}")
                path = _resolve_path(str(payload[path_field]))
                text = payload.get(text_field)
                meta = {
                    k: _coerce_meta_value(v)
                    for k, v in payload.items()
                    if k not in {path_field, text_field}
                }
                records.append(ClipRecord(video_path=path, text=text if text is None else str(text), meta=meta))
        return records

    if suffix == ".csv":
        with manifest_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=2):
                if path_field not in row or not row[path_field]:
                    raise ValueError(f"Manifest missing `{path_field}` at csv line {idx}")
                path = _resolve_path(str(row[path_field]))
                text = row.get(text_field) if text_field in row else None
                meta = {
                    k: _coerce_meta_value(v)
                    for k, v in row.items()
                    if k not in {path_field, text_field}
                }
                records.append(ClipRecord(video_path=path, text=text if text else None, meta=meta))
        return records

    raise ValueError(f"Unsupported manifest format: {manifest_path.suffix}. Use .csv or .jsonl")


def _load_dir_records(input_dirs: list[str], recursive: bool) -> List[ClipRecord]:
    records: List[ClipRecord] = []
    for item in input_dirs:
        root = Path(item).expanduser().resolve()
        if not root.exists():
            log.warning("Input dir does not exist, skipping: %s", root)
            continue
        if not root.is_dir():
            log.warning("Input path is not a directory, skipping: %s", root)
            continue
        for video in _iter_video_files(root, recursive=recursive):
            records.append(ClipRecord(video_path=video.resolve()))
    return records


def _record_key(video_path: Path, text: str | None) -> tuple[str, str | None]:
    return str(video_path), (text if text is None else str(text))


def _load_existing_metadata(
    metadata_path: Path,
) -> tuple[set[tuple[str, str | None]], int, int, Counter[str], int]:
    seen: set[tuple[str, str | None]] = set()
    kept = 0
    dropped = 0
    reasons: Counter[str] = Counter()
    valid_lines = 0

    if not metadata_path.exists():
        return seen, kept, dropped, reasons, valid_lines

    with metadata_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                log.warning("Skip malformed metadata line %d in %s", line_no, metadata_path)
                continue

            source_path = payload.get("source_path")
            if not source_path:
                continue

            text = payload.get("text")
            text_norm = text if text is None else str(text)
            seen.add((str(source_path), text_norm))
            valid_lines += 1

            if bool(payload.get("keep", False)):
                kept += 1
            else:
                dropped += 1
                reasons[str(payload.get("reason") or "filtered")] += 1

    return seen, kept, dropped, reasons, valid_lines


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _prepare_output_dir(cfg_workdir: Path, output_dir_arg: str | None) -> Path:
    if output_dir_arg:
        out = Path(output_dir_arg).expanduser().resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = (cfg_workdir / "output" / f"offline_{stamp}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    return out


def _append_scored_records(
    fp: Any,
    videos_dir: Path,
    indexed_records: list[tuple[int, ClipRecord]],
    scored: List["ScoreResult"],
    copy_mode: str,
    reason_counter: Counter[str],
    consistency_pending: list[str] | None = None,
    write_buffer_records: int = 128,
) -> tuple[int, int]:
    kept = 0
    dropped = 0
    lines: list[str] = []
    threshold = max(int(write_buffer_records), 1)

    for (global_idx, record), result in zip(indexed_records, scored):
        keep = result.keep and record.video_path.exists()
        reason = result.reason
        if result.keep and not record.video_path.exists():
            keep = False
            reason = "source_missing"

        output_path: Path | None = None
        if keep:
            output_name = f"{global_idx:08d}_{record.video_path.name}"
            output_path = videos_dir / output_name
            _link_or_copy(record.video_path, output_path, copy_mode)
            kept += 1
        else:
            dropped += 1
            reason_counter[reason or "filtered"] += 1

        payload = {
            "index": global_idx,
            "source_path": str(record.video_path),
            "output_path": str(output_path) if output_path else None,
            "keep": keep,
            "reason": None if keep else (reason or "filtered"),
            "scores": result.scores,
            "text": record.text,
            "meta": record.meta,
        }
        if consistency_pending:
            payload["consistency_stub"] = True
            payload["consistency_pending"] = ",".join(dict.fromkeys(consistency_pending))
        lines.append(json.dumps(payload, ensure_ascii=False) + "\n")

        if len(lines) >= threshold:
            fp.writelines(lines)
            lines.clear()

    if lines:
        fp.writelines(lines)

    return kept, dropped


def _load_records(args: argparse.Namespace) -> List[ClipRecord]:
    records: List[ClipRecord] = []
    if args.manifest:
        manifest_path = Path(args.manifest).expanduser().resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        records.extend(_load_manifest_records(manifest_path, args.path_field, args.text_field))
    if args.input_dir:
        records.extend(_load_dir_records(args.input_dir, recursive=args.recursive))

    dedup: dict[tuple[str, str | None], ClipRecord] = {}
    for rec in records:
        key = _record_key(rec.video_path, rec.text)
        if key not in dedup:
            dedup[key] = rec
    ordered = list(dedup.values())
    if args.limit is not None:
        ordered = ordered[: max(args.limit, 0)]
    return ordered


def _write_runlog(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


class _AsyncMetadataWriter:
    def __init__(self, fp: TextIO, max_chunks: int = 64):
        self._fp = fp
        self._queue: "queue.Queue[list[str] | None]" = queue.Queue(maxsize=max(max_chunks, 1))
        self._exc: Exception | None = None
        self._thread = threading.Thread(target=self._worker, name="egodex-metadata-writer", daemon=True)
        self._thread.start()

    def _raise_if_failed(self) -> None:
        if self._exc is not None:
            raise RuntimeError("async metadata writer failed") from self._exc

    def _worker(self) -> None:
        while True:
            lines = self._queue.get()
            try:
                if lines is None:
                    return
                if self._exc is not None:
                    continue
                if lines:
                    self._fp.writelines(lines)
            except Exception as exc:  # noqa: BLE001
                self._exc = exc
            finally:
                self._queue.task_done()

    def writelines(self, lines: list[str]) -> None:
        self._raise_if_failed()
        self._queue.put(list(lines))

    def flush(self) -> None:
        self._queue.join()
        self._raise_if_failed()
        self._fp.flush()

    def close(self) -> None:
        self.flush()
        self._queue.put(None)
        self._queue.join()
        self._thread.join(timeout=5)
        self._raise_if_failed()
        self._fp.flush()
        self._fp.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    args = parse_args()
    from .models import build_scorers, run_scorers

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_config(cfg_path, skip_upload=True)

    _configure_torch_runtime()

    selection_cfg = getattr(cfg, "selection", None)
    default_strategy = getattr(selection_cfg, "strategy", None) if selection_cfg is not None else None
    if args.strategy and selection_cfg is not None:
        selection_cfg.strategy = args.strategy
    strategy_expr = (args.strategy or default_strategy or "").strip() or None

    started = time.time()
    started_iso = datetime.now(timezone.utc).isoformat()
    output_dir = _prepare_output_dir(cfg.workdir, args.output_dir)
    videos_dir = output_dir / "videos"
    metadata_path = output_dir / "metadata.jsonl"
    runlog_path = output_dir / "runlog.json"

    records = _load_records(args)
    if not records:
        raise SystemExit("No input records found. Provide --manifest and/or --input-dir.")

    seen_keys, kept_total, dropped_total, reason_counter, existing_lines = _load_existing_metadata(metadata_path)
    if not args.resume and metadata_path.exists():
        log.warning("--no-resume: overwrite existing metadata at %s", metadata_path)
        seen_keys = set()
        kept_total = 0
        dropped_total = 0
        reason_counter = Counter()
        existing_lines = 0

    indexed_records = list(enumerate(records))
    pending_indices: list[int] = []
    pending_records: list[ClipRecord] = []
    for idx, rec in indexed_records:
        if args.resume and _record_key(rec.video_path, rec.text) in seen_keys:
            continue
        pending_indices.append(idx)
        pending_records.append(rec)

    processed_total = len(records) - len(pending_records)
    flush_every = max(int(args.flush_every or 1), 1)
    runlog_every = max(int(args.runlog_every or 1), 1)
    write_buffer_records = max(int(args.write_buffer_records or 1), 1)

    log.info(
        "Offline run started: records=%d models=%d resume=%s already_done=%d pending=%d",
        len(records),
        len(cfg.models),
        bool(args.resume),
        processed_total,
        len(pending_records),
    )
    if strategy_expr:
        log.info("Using selection strategy: %s", strategy_expr)

    runlog: dict[str, Any] = {
        "started_at": started_iso,
        "finished_at": None,
        "duration_sec": None,
        "status": "running",
        "config_path": str(cfg_path),
        "output_dir": str(output_dir),
        "strategy": strategy_expr,
        "input": {
            "manifest": args.manifest,
            "input_dirs": args.input_dir,
            "recursive": bool(args.recursive),
            "limit": args.limit,
        },
        "models": [m.name for m in cfg.models],
        "records_total": len(records),
        "processed": processed_total,
        "pending": len(pending_records),
        "existing_lines": existing_lines,
        "kept": kept_total,
        "dropped": dropped_total,
        "reason_counts": dict(reason_counter),
        "metadata_path": str(metadata_path),
        "copy_mode": args.copy_mode,
        "resume": bool(args.resume),
        "flush_every": flush_every,
        "runlog_every": runlog_every,
        "write_buffer_records": write_buffer_records,
    }

    def _sync_runlog(status: str = "running") -> None:
        runlog.update(
            {
                "status": status,
                "processed": processed_total,
                "pending": max(len(records) - processed_total, 0),
                "kept": kept_total,
                "dropped": dropped_total,
                "reason_counts": dict(reason_counter),
                "metadata_path": str(metadata_path),
            }
        )
        _write_runlog(runlog_path, runlog)

    _sync_runlog("running")

    metadata_fp: _AsyncMetadataWriter | None = None
    written_batches = 0
    try:
        if pending_records:
            scoring_workers = cfg.runtime.scoring_workers if cfg.runtime.scoring_workers else max(len(cfg.models), 1)
            consistency_pending = _consistency_pending_kinds([m.kind for m in cfg.models])
            scorers = build_scorers(cfg.models)

            mode = "a" if args.resume else "w"
            metadata_fp = _AsyncMetadataWriter(metadata_path.open(mode, encoding="utf-8"))
            callback_written = False

            def _on_batch(batch_start: int, batch_items: list[Any], batch_results: List["ScoreResult"]) -> None:
                nonlocal kept_total, dropped_total, processed_total, written_batches, callback_written
                indexed_batch = [
                    (pending_indices[batch_start + i], batch_items[i])
                    for i in range(len(batch_items))
                ]
                kept_inc, dropped_inc = _append_scored_records(
                    metadata_fp,
                    videos_dir,
                    indexed_batch,
                    batch_results,
                    args.copy_mode,
                    reason_counter,
                    consistency_pending=consistency_pending,
                    write_buffer_records=write_buffer_records,
                )
                kept_total += kept_inc
                dropped_total += dropped_inc
                processed_total += len(batch_items)
                written_batches += 1
                callback_written = True

                if written_batches % flush_every == 0:
                    metadata_fp.flush()
                if written_batches % runlog_every == 0:
                    _sync_runlog("running")

            try:
                scored_results = run_scorers(
                    scorers,
                    pending_records,
                    max_workers=scoring_workers,
                    strategy_expr=strategy_expr,
                    on_batch=_on_batch,
                )
                if not callback_written and scored_results:
                    # run_scorers may return directly without invoking callback
                    # (e.g. no scorers configured). Persist results explicitly.
                    indexed_batch = list(zip(pending_indices, pending_records))
                    kept_inc, dropped_inc = _append_scored_records(
                        metadata_fp,
                        videos_dir,
                        indexed_batch,
                        scored_results,
                        args.copy_mode,
                        reason_counter,
                        consistency_pending=consistency_pending,
                        write_buffer_records=write_buffer_records,
                    )
                    kept_total += kept_inc
                    dropped_total += dropped_inc
                    processed_total += len(pending_records)
                    written_batches += 1
                    metadata_fp.flush()
                    _sync_runlog("running")
            except TypeError:
                # Fallback for older run_scorers without on_batch callback support.
                log.warning("run_scorers has no on_batch callback; using chunked fallback write")
                fallback_step = max(min(getattr(s, "batch_size", 1) for s in scorers), 1)
                for batch_start in range(0, len(pending_records), fallback_step):
                    batch_records = pending_records[batch_start : batch_start + fallback_step]
                    batch_indices = pending_indices[batch_start : batch_start + fallback_step]
                    try:
                        batch_scored = run_scorers(
                            scorers,
                            batch_records,
                            max_workers=scoring_workers,
                            strategy_expr=strategy_expr,
                        )
                    except TypeError:
                        batch_scored = run_scorers(
                            scorers,
                            batch_records,
                            max_workers=scoring_workers,
                        )

                    indexed_batch = list(zip(batch_indices, batch_records))
                    kept_inc, dropped_inc = _append_scored_records(
                        metadata_fp,
                        videos_dir,
                        indexed_batch,
                        batch_scored,
                        args.copy_mode,
                        reason_counter,
                        consistency_pending=consistency_pending,
                        write_buffer_records=write_buffer_records,
                    )
                    kept_total += kept_inc
                    dropped_total += dropped_inc
                    processed_total += len(batch_records)
                    written_batches += 1

                    if written_batches % flush_every == 0:
                        metadata_fp.flush()
                    if written_batches % runlog_every == 0:
                        _sync_runlog("running")

                metadata_fp.flush()
        else:
            log.info("Nothing to score: all records already exist in metadata (%d/%d).", processed_total, len(records))

        _sync_runlog("success")
        log.info(
            "Offline run complete: processed=%d/%d kept=%d dropped=%d metadata=%s runlog=%s",
            processed_total,
            len(records),
            kept_total,
            dropped_total,
            metadata_path,
            runlog_path,
        )
    except KeyboardInterrupt:
        runlog.update(
            {
                "status": "interrupted",
                "error": "KeyboardInterrupt",
            }
        )
        _sync_runlog("interrupted")
        raise
    except Exception as exc:  # noqa: BLE001
        runlog.update(
            {
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        _sync_runlog("failed")
        raise
    finally:
        if metadata_fp is not None:
            metadata_fp.close()
        ended = time.time()
        runlog["finished_at"] = datetime.now(timezone.utc).isoformat()
        runlog["duration_sec"] = round(ended - started, 3)
        _write_runlog(runlog_path, runlog)


if __name__ == "__main__":
    main()
