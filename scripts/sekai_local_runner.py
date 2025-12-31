#!/usr/bin/env python3
"""
一键本地跑 Sekai：可选先用 csv 下载+打包分片，然后把分片放到 pipeline workdir/downloads，
自动补全 state 为 downloaded=true，最后调用 pipeline（本地跑时请加 --skip-upload）。

用法示例（仅本地，无上传）：
    python scripts/sekai_local_runner.py \
      --config configs/config_sekai_raw_clean.yaml \
      --csv sekai-real-walking-hq.csv \
      --pack-workdir sekai_tmp \
      --pack-size-gb 40 \
      --batch-size 500 \
      --rate-limit 50M \
      --max-concurrent 5 \
      --skip-upload

若已手工下载/打包好分片，可跳过 --csv，直接指定 --shards-dir / --shards-file。
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sekai 本地 end-to-end：下载/打包 -> 拷贝到 workdir -> 写 state -> 跑 pipeline")
    p.add_argument("--config", required=True, help="pipeline 配置文件（例如 configs/config_sekai_raw_clean.yaml）")
    p.add_argument(
        "--csv",
        help="可选：Sekai csv（如 sekai-real-walking-hq.csv）。提供后会调用 download_and_pack_sekai.py 下载+打包。",
    )
    p.add_argument(
        "--pack-workdir",
        default="sekai_tmp",
        help="下载/打包的工作目录（仅在提供 --csv 时使用，默认 sekai_tmp）",
    )
    p.add_argument(
        "--shards-dir",
        help="已打包分片所在目录。默认：若有 --csv 则用 <pack-workdir>/packed，否则用 config 的 workdir/downloads。",
    )
    p.add_argument(
        "--shards-file",
        help="分片列表文件（每行一个分片名）。默认：若有 --csv 则用 <pack-workdir>/shards_sekai_raw.txt，否则用 config 中的 shards_file。",
    )
    p.add_argument("--limit-shards", type=int, help="仅处理前 N 个分片（传给 pipeline）")
    p.add_argument("--skip-upload", action="store_true", help="运行 pipeline 时附带 --skip-upload（默认不传则按 config 行为）")
    p.add_argument("--prepare-only", action="store_true", help="只准备分片+state，不实际运行 pipeline")
    p.add_argument("--pack-size-gb", type=float, default=40.0, help="下载阶段：打包分片目标大小（GB，默认 40）")
    p.add_argument("--batch-size", type=int, default=500, help="下载阶段：每批 URL 数（默认 500）")
    p.add_argument("--rate-limit", default="50M", help="下载阶段：yt-dlp --limit-rate（默认 50M）")
    p.add_argument("--max-concurrent", type=int, default=5, help="下载阶段：yt-dlp -N 并发数（默认 5）")
    p.add_argument(
        "--yt-dlp-extra",
        default="",
        help='下载阶段：额外传给 yt-dlp 的参数，例如 "--retries 5 --fragment-retries 5"',
    )
    return p.parse_args()


def run_download_and_pack(
    csv_path: Path,
    pack_workdir: Path,
    shards_file: Path,
    pack_size_gb: float,
    batch_size: int,
    rate_limit: str,
    max_concurrent: int,
    yt_dlp_extra: str,
) -> None:
    script_path = Path("scripts/download_and_pack_sekai.py")
    if not script_path.exists():
        raise FileNotFoundError(f"download_and_pack_sekai.py not found at {script_path}")

    cmd = [
        sys.executable,
        str(script_path),
        "--csv",
        str(csv_path),
        "--workdir",
        str(pack_workdir),
        "--shards-file",
        str(shards_file),
        "--pack-size-gb",
        str(pack_size_gb),
        "--batch-size",
        str(batch_size),
        "--rate-limit",
        rate_limit,
        "--max-concurrent",
        str(max_concurrent),
    ]
    if yt_dlp_extra:
        cmd.extend(["--yt-dlp-extra", yt_dlp_extra])

    print(f"[pack] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def resolve_config_paths(config_path: Path) -> dict:
    raw = yaml.safe_load(config_path.read_text())
    cfg_dir = config_path.parent
    workdir = Path(raw["workdir"]).expanduser()

    shards_file = raw.get("shards_file")
    shards_file_path = None
    if shards_file:
        shards_file_path = Path(shards_file)
        if not shards_file_path.is_absolute():
            shards_file_path = (cfg_dir / shards_file_path).resolve()

    shards_list = raw.get("shards", []) or []

    return {
        "workdir": workdir,
        "shards_file": shards_file_path,
        "shards_list": [str(s) for s in shards_list],
    }


def load_shards(shards_file: Path | None, fallback_list: list[str], shards_dir: Path) -> list[str]:
    if shards_file and shards_file.exists():
        names = [ln.strip() for ln in shards_file.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
        if names:
            return names
        raise SystemExit(f"shards_file is empty: {shards_file}")

    if fallback_list:
        return fallback_list

    # Fallback: infer from shards_dir
    if not shards_dir.exists():
        raise SystemExit("No shards found: please provide a shards_file or place shards in --shards-dir")
    names = sorted(p.name for p in shards_dir.glob("*") if p.is_file() and p.suffix in {".zip", ".gz", ".tar", ".tgz"})
    if not names:
        raise SystemExit(f"No shard archives found under {shards_dir}")
    return names


def copy_shards(shards: list[str], src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in shards:
        src = src_dir / name
        dst = dst_dir / name
        if not src.exists():
            raise SystemExit(f"Shard not found at {src}")
        if dst.exists():
            print(f"[stage] {name} already in downloads, skip copy")
            continue
        print(f"[stage] Copy {src} -> {dst}")
        shutil.copy2(src, dst)


def ensure_state(shards: list[str], state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    for name in shards:
        path = state_dir / f"{name}.json"
        state = {
            "downloaded": True,
            "extracted": False,
            "scored": False,
            "uploaded": False,
            "stage": "pending",
        }
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                state.update(existing)
                state["downloaded"] = True
                state.setdefault("extracted", False)
                state.setdefault("scored", False)
                state.setdefault("uploaded", False)
                state.setdefault("stage", "pending")
            except Exception:
                print(f"[state] Failed to read existing state for {name}, overwriting with default downloaded=True")
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(f"[state] wrote {path}")


def run_pipeline(config: Path, shards_file: Path | None, skip_upload: bool, limit_shards: int | None) -> None:
    cmd = [sys.executable, "-m", "pipeline.pipeline", "--config", str(config)]
    if shards_file:
        cmd.extend(["--shards-file", str(shards_file)])
    if skip_upload:
        cmd.append("--skip-upload")
    if limit_shards is not None:
        cmd.extend(["--limit-shards", str(limit_shards)])

    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    cfg = resolve_config_paths(config_path)
    workdir = cfg["workdir"]
    default_shards_file = cfg["shards_file"]
    default_shards_list = cfg["shards_list"]

    pack_workdir = Path(args.pack_workdir).expanduser().resolve()
    shards_file = Path(args.shards_file).expanduser().resolve() if args.shards_file else None
    shards_dir = Path(args.shards_dir).expanduser().resolve() if args.shards_dir else None

    # Step 1: optional download+pack
    if args.csv:
        csv_path = Path(args.csv).expanduser().resolve()
        if not csv_path.exists():
            raise SystemExit(f"CSV not found: {csv_path}")
        if shards_file is None:
            shards_file = (pack_workdir / "shards_sekai_raw.txt").resolve()
        run_download_and_pack(
            csv_path=csv_path,
            pack_workdir=pack_workdir,
            shards_file=shards_file,
            pack_size_gb=args.pack_size_gb,
            batch_size=args.batch_size,
            rate_limit=args.rate_limit,
            max_concurrent=args.max_concurrent,
            yt_dlp_extra=args.yt_dlp_extra,
        )
        if shards_dir is None:
            shards_dir = (pack_workdir / "packed").resolve()

    # Defaults when no csv provided
    if shards_file is None:
        shards_file = default_shards_file
    if shards_dir is None:
        shards_dir = (workdir / "downloads").resolve()

    if shards_file:
        shards_file = shards_file.expanduser().resolve()

    # Step 2: load shard names
    shards = load_shards(shards_file, default_shards_list, shards_dir)

    # 若 config 未提供分片列表/文件，自动写一份覆盖用的 shards_file，确保 pipeline 能读取
    if shards_file is None and not default_shards_list:
        auto_list = workdir / "shards_local_auto.txt"
        auto_list.write_text("\n".join(shards) + "\n", encoding="utf-8")
        shards_file = auto_list
        print(f"[list] wrote auto shards_file at {auto_list}")

    # Step 3: stage into workdir/downloads
    downloads_dir = workdir / "downloads"
    copy_shards(shards, shards_dir, downloads_dir)

    # Step 4: write state marking downloaded
    state_dir = workdir / "state"
    ensure_state(shards, state_dir)

    if args.prepare_only:
        print("[done] Prepared shards and state only (prepare-only requested).")
        return

    # Step 5: run pipeline
    run_pipeline(config_path, shards_file, skip_upload=args.skip_upload, limit_shards=args.limit_shards)


if __name__ == "__main__":
    main()
