#!/usr/bin/env python3
"""
Upload existing processed shards without重跑评分/切分/Caption。

默认读取 config.yaml 获取 target_repo、upload 设置和 workdir。
仅上传 state 未标记 uploaded 的 shard；可用 --force 重新上传。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import yaml
from huggingface_hub import HfApi

try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # type: ignore

try:
    import pandas as pd
except Exception:
    pd = None  # type: ignore


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_shards(shards_file: Path) -> List[str]:
    return [
        line.strip()
        for line in shards_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def load_state(state_dir: Path, shard: str) -> Dict:
    jf = state_dir / f"{shard}.json"
    if not jf.exists():
        return {}
    try:
        return json.loads(jf.read_text())
    except Exception:
        return {}


def save_state(state_dir: Path, shard: str, state: Dict) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    jf = state_dir / f"{shard}.json"
    jf.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_config(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path
    alt = Path("configs") / path_str
    if alt.exists():
        return alt
    raise FileNotFoundError(f"config not found: {path_str} (also tried configs/{path_str})")


def main() -> None:
    p = argparse.ArgumentParser(description="Upload existing shard outputs only.")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml (tries configs/ if not found)")
    p.add_argument("--shards-file", help="Optional shards file; default from config.yaml")
    p.add_argument("--workdir", help="Override workdir; default from config.yaml")
    p.add_argument("--state-dir", help="Override state dir; default <workdir>/state")
    p.add_argument("--output-dir", help="Override output dir; default <workdir>/output")
    p.add_argument("--target-repo", help="Override target repo; default from config.yaml")
    p.add_argument("--force", action="store_true", help="Upload even if state says uploaded")
    p.add_argument(
        "--manifest",
        action="store_true",
        default=True,
        help="Write per-shard manifest parquet (videos and sizes) into shard dir before upload (default: on)",
    )
    p.add_argument(
        "--no-manifest",
        action="store_false",
        dest="manifest",
        help="Disable manifest generation",
    )
    p.add_argument(
        "--manifest-glob",
        default="*.mp4,*.mkv,*.avi,*.mov,*.webm",
        help="Comma-separated glob patterns for videos to include in manifest",
    )
    args = p.parse_args()

    cfg_path = resolve_config(args.config)
    cfg = load_yaml(cfg_path)
    workdir = Path(args.workdir or cfg.get("workdir") or ".")
    shards_file = Path(args.shards_file or cfg.get("shards_file") or "shards.txt")
    state_dir = Path(args.state_dir or workdir / "state")
    output_dir = Path(args.output_dir or workdir / "output")
    target_repo = args.target_repo or cfg.get("target_repo")
    upload_cfg = cfg.get("upload", {}) or {}
    chunk_mb = int(upload_cfg.get("chunk_size_mb", 512))
    max_workers = int(upload_cfg.get("max_workers", 3))

    if not target_repo:
        raise SystemExit("target_repo missing (set in config or --target-repo)")
    if not shards_file.exists():
        raise SystemExit(f"shards_file not found: {shards_file}")
    shards = load_shards(shards_file)
    api = HfApi()

    patterns = [p.strip() for p in args.manifest_glob.split(",") if p.strip()]

    uploaded = 0
    skipped = 0
    missing = 0
    iterator = tqdm(shards, desc="shards") if tqdm else shards
    for shard in iterator:
        shard_dir = output_dir / shard
        if not shard_dir.exists():
            missing += 1
            print(f"[skip missing] {shard_dir}")
            continue
        state = load_state(state_dir, shard)
        if state.get("uploaded") and not args.force:
            skipped += 1
            print(f"[skip uploaded] {shard}")
            continue
        # manifest
        if args.manifest:
            if pd is None:
                print(f"[warn] pandas not available, skip manifest for {shard}")
            else:
                rows = []
                for pat in patterns:
                    for f in shard_dir.rglob(pat):
                        if f.is_file():
                            rows.append({"path": f.relative_to(shard_dir).as_posix(), "bytes": f.stat().st_size})
                if rows:
                    df = pd.DataFrame(rows)
                    (shard_dir / "manifest.parquet").unlink(missing_ok=True)
                    df.to_parquet(shard_dir / "manifest.parquet", index=False)
                    print(f"[manifest] {shard}: {len(rows)} files")
        print(f"[upload] {shard} -> {target_repo}")
        api.upload_folder(
            repo_id=target_repo,
            folder_path=str(shard_dir),
            path_in_repo=shard,
            repo_type="dataset",
            commit_message=f"Add shard {shard}",
            max_workers=max_workers,
            chunk_size=chunk_mb * 1024 * 1024,
        )
        state["uploaded"] = True
        state.setdefault("stage", "uploaded")
        save_state(state_dir, shard, state)
        uploaded += 1

    print(f"done. uploaded={uploaded}, skipped_uploaded={skipped}, missing_output={missing}")


if __name__ == "__main__":
    main()
