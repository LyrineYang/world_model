#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_run_env() -> dict[str, str]:
    env = os.environ.copy()
    repo_root = str(REPO_ROOT)
    current_pythonpath = env.get("PYTHONPATH", "")
    if not current_pythonpath:
        env["PYTHONPATH"] = repo_root
        return env

    paths = current_pythonpath.split(os.pathsep)
    if repo_root not in paths:
        env["PYTHONPATH"] = repo_root + os.pathsep + current_pythonpath
    return env


def _run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=_build_run_env())


def _add_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to pipeline config")
    parser.add_argument("--shards-file", type=str, default=None, help="Override shards file")
    parser.add_argument("--limit-shards", type=int, default=None, help="Process only first N shards")
    parser.add_argument("--skip-upload", action="store_true", help="Skip upload")
    parser.add_argument("--calibration", action="store_true", help="Enable calibration mode")
    parser.add_argument("--sample-size", type=int, default=None, help="Calibration sample size")
    parser.add_argument("--calibration-output", type=str, default=None, help="Calibration output path")
    parser.add_argument("--calibration-quantiles", type=str, default=None, help="Comma-separated quantiles")
    parser.add_argument("--strategy", type=str, default=None, help="Override selection strategy")


def _build_filter_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, "-m", "pipeline.pipeline", "--config", str(Path(args.config).expanduser())]
    if args.shards_file:
        cmd.extend(["--shards-file", args.shards_file])
    if args.limit_shards is not None:
        cmd.extend(["--limit-shards", str(args.limit_shards)])
    if args.skip_upload:
        cmd.append("--skip-upload")
    if args.calibration:
        cmd.append("--calibration")
    if args.sample_size is not None:
        cmd.extend(["--sample-size", str(args.sample_size)])
    if args.calibration_output:
        cmd.extend(["--calibration-output", args.calibration_output])
    if args.calibration_quantiles:
        cmd.extend(["--calibration-quantiles", args.calibration_quantiles])
    if args.strategy:
        cmd.extend(["--strategy", args.strategy])
    return cmd


def _build_caption_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, "-m", "pipeline.caption_only", "--config", str(Path(args.config).expanduser())]
    if getattr(args, "input_root", None):
        cmd.extend(["--input-root", args.input_root])
    if getattr(args, "overwrite", False):
        cmd.append("--overwrite")
    if getattr(args, "frame_count", None) is not None:
        cmd.extend(["--frame-count", str(args.frame_count)])
    if getattr(args, "include_unkept", False):
        cmd.append("--include-unkept")
    return cmd


def _build_offline_filter_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, "-m", "pipeline.offline_runner", "--config", str(Path(args.config).expanduser())]
    for input_dir in args.input_dir or []:
        cmd.extend(["--input-dir", input_dir])
    if args.manifest:
        cmd.extend(["--manifest", args.manifest])
    if args.path_field:
        cmd.extend(["--path-field", args.path_field])
    if args.text_field:
        cmd.extend(["--text-field", args.text_field])
    if args.recursive:
        cmd.append("--recursive")
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.strategy:
        cmd.extend(["--strategy", args.strategy])
    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])
    if args.copy_mode:
        cmd.extend(["--copy-mode", args.copy_mode])
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified workflow runner")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    p_filter = subparsers.add_parser("filter", help="Run shard filtering pipeline")
    _add_filter_args(p_filter)

    p_caption = subparsers.add_parser("caption", help="Run caption-only pipeline")
    p_caption.add_argument("--config", required=True, help="Path to pipeline config")
    p_caption.add_argument("--input-root", type=str, default=None, help="Root folder containing outputs")
    p_caption.add_argument("--overwrite", action="store_true", help="Overwrite existing captions")
    p_caption.add_argument("--frame-count", type=int, default=4, help="Frames per video for image captioning")
    p_caption.add_argument("--include-unkept", action="store_true", help="Caption keep=false records as well")

    p_full = subparsers.add_parser("full", help="Run filtering then caption-only")
    _add_filter_args(p_full)
    p_full.add_argument("--input-root", type=str, default=None, help="Caption input root (default output)")
    p_full.add_argument("--overwrite", action="store_true", help="Overwrite existing captions")
    p_full.add_argument("--frame-count", type=int, default=4, help="Frames per video for image captioning")
    p_full.add_argument("--include-unkept", action="store_true", help="Caption keep=false records as well")

    p_offline = subparsers.add_parser("offline-filter", help="Run offline filter on local videos")
    p_offline.add_argument("--config", required=True, help="Path to pipeline config")
    p_offline.add_argument("--input-dir", action="append", default=[], help="Input directory, repeatable")
    p_offline.add_argument("--manifest", type=str, default=None, help="Optional CSV/JSONL manifest")
    p_offline.add_argument("--path-field", type=str, default="video_path", help="Manifest path field")
    p_offline.add_argument("--text-field", type=str, default="text", help="Manifest text field")
    p_offline.add_argument("--recursive", action="store_true", help="Recursively scan input dirs")
    p_offline.add_argument("--limit", type=int, default=None, help="Only process first N records")
    p_offline.add_argument("--strategy", type=str, default=None, help="Override selection strategy")
    p_offline.add_argument("--output-dir", type=str, default=None, help="Output directory")
    p_offline.add_argument("--copy-mode", choices=["link", "copy"], default="link", help="Materialization mode")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "filter":
        _run(_build_filter_cmd(args))
        return
    if args.mode == "caption":
        _run(_build_caption_cmd(args))
        return
    if args.mode == "full":
        _run(_build_filter_cmd(args))
        _run(_build_caption_cmd(args))
        return
    if args.mode == "offline-filter":
        _run(_build_offline_filter_cmd(args))
        return
    raise SystemExit(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
