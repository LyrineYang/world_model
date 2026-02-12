from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, List

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


def _write_outputs(
    output_dir: Path,
    records: List[ClipRecord],
    scored: List["ScoreResult"],
    copy_mode: str,
    consistency_pending: list[str] | None = None,
) -> dict[str, Any]:
    metadata_path = output_dir / "metadata.jsonl"
    videos_dir = output_dir / "videos"
    reason_counter: Counter[str] = Counter()
    keep_count = 0

    with metadata_path.open("w", encoding="utf-8") as f:
        for idx, (record, result) in enumerate(zip(records, scored)):
            keep = result.keep and record.video_path.exists()
            reason = result.reason
            if result.keep and not record.video_path.exists():
                keep = False
                reason = "source_missing"

            output_path: Path | None = None
            if keep:
                output_name = f"{idx:08d}_{record.video_path.name}"
                output_path = videos_dir / output_name
                _link_or_copy(record.video_path, output_path, copy_mode)
                keep_count += 1
            else:
                reason_counter[reason or "filtered"] += 1

            payload = {
                "index": idx,
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
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    return {
        "metadata_path": str(metadata_path),
        "kept": keep_count,
        "dropped": len(records) - keep_count,
        "reason_counts": dict(reason_counter),
    }


def _load_records(args: argparse.Namespace) -> List[ClipRecord]:
    records: List[ClipRecord] = []
    if args.manifest:
        manifest_path = Path(args.manifest).expanduser().resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        records.extend(_load_manifest_records(manifest_path, args.path_field, args.text_field))
    if args.input_dir:
        records.extend(_load_dir_records(args.input_dir, recursive=args.recursive))
    # Keep stable order and deduplicate exact key.
    dedup: dict[tuple[str, str | None], ClipRecord] = {}
    for rec in records:
        key = (str(rec.video_path), rec.text)
        if key not in dedup:
            dedup[key] = rec
    ordered = list(dedup.values())
    if args.limit is not None:
        ordered = ordered[: max(args.limit, 0)]
    return ordered


def _write_runlog(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    args = parse_args()
    from .models import build_scorers, run_scorers

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_config(cfg_path, skip_upload=True)

    selection_cfg = getattr(cfg, "selection", None)
    default_strategy = getattr(selection_cfg, "strategy", None) if selection_cfg is not None else None
    if args.strategy and selection_cfg is not None:
        selection_cfg.strategy = args.strategy
    strategy_expr = (args.strategy or default_strategy or "").strip() or None

    started = time.time()
    started_iso = datetime.now(timezone.utc).isoformat()
    output_dir = _prepare_output_dir(cfg.workdir, args.output_dir)

    records = _load_records(args)
    if not records:
        raise SystemExit("No input records found. Provide --manifest and/or --input-dir.")

    log.info("Offline run started: records=%d models=%d", len(records), len(cfg.models))
    if strategy_expr:
        log.info("Using selection strategy: %s", strategy_expr)

    runlog_path = output_dir / "runlog.json"
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
        "kept": 0,
        "dropped": 0,
        "reason_counts": {},
        "metadata_path": None,
        "copy_mode": args.copy_mode,
    }
    try:
        scoring_workers = cfg.runtime.scoring_workers if cfg.runtime.scoring_workers else max(len(cfg.models), 1)
        consistency_pending = _consistency_pending_kinds([m.kind for m in cfg.models])
        scorers = build_scorers(cfg.models)
        try:
            scored = run_scorers(
                scorers,
                records,
                max_workers=scoring_workers,
                strategy_expr=strategy_expr,
            )
        except TypeError:
            scored = run_scorers(
                scorers,
                records,
                max_workers=scoring_workers,
            )
        written = _write_outputs(
            output_dir,
            records,
            scored,
            args.copy_mode,
            consistency_pending=consistency_pending,
        )
        runlog.update(
            {
                "status": "success",
                "kept": written["kept"],
                "dropped": written["dropped"],
                "reason_counts": written["reason_counts"],
                "metadata_path": written["metadata_path"],
            }
        )
        log.info(
            "Offline run complete: kept=%d dropped=%d metadata=%s runlog=%s",
            written["kept"],
            written["dropped"],
            written["metadata_path"],
            runlog_path,
        )
    except Exception as exc:  # noqa: BLE001
        runlog.update(
            {
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        ended = time.time()
        runlog["finished_at"] = datetime.now(timezone.utc).isoformat()
        runlog["duration_sec"] = round(ended - started, 3)
        _write_runlog(runlog_path, runlog)


if __name__ == "__main__":
    main()
