from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_workdir(config_path: Path) -> Path:
    raw = load_config(config_path)
    workdir = raw.get("workdir")
    if not workdir:
        raise ValueError(f"workdir missing in {config_path}")
    return Path(workdir).expanduser()


def load_shards(config_path: Path) -> List[str]:
    raw = load_config(config_path)
    shards: List[str] = []
    shards_file = raw.get("shards_file")
    base_dir = config_path.parent
    if shards_file:
        sf_path = Path(shards_file)
        if not sf_path.is_absolute():
            sf_path = base_dir / sf_path
        if sf_path.exists():
            with sf_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    shards.append(line)
    else:
        shards = list(raw.get("shards", []))
    return shards


def load_states(state_dir: Path) -> List[Tuple[str, Dict, float]]:
    records: List[Tuple[str, Dict, float]] = []
    for path in sorted(state_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            mtime = path.stat().st_mtime
            records.append((path.stem, data, mtime))
        except Exception:
            continue
    return records


def format_age(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


def get_load_avg() -> Tuple[float, float, float] | None:
    try:
        return os.getloadavg()
    except Exception:
        return None


def get_gpu_stats() -> List[Dict[str, float]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=2)
    except Exception:
        return []
    stats: List[Dict[str, float]] = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            stats.append(
                {
                    "index": float(parts[0]),
                    "mem_used": float(parts[1]),
                    "mem_total": float(parts[2]),
                    "util": float(parts[3]),
                    "power": float(parts[4]),
                }
            )
        except Exception:
            continue
    return stats


def render(
    state_dir: Path,
    interval: float,
    config_shards: List[Tuple[Path, List[str]]] | None = None,
    filter_to_config: bool = False,
) -> None:
    prev_scored = None
    prev_uploaded = None
    prev_ts = None
    done_stages = {"uploaded", "processed"}
    summaries_cache: Dict[str, Dict] = {}

    while True:
        now = time.time()
        states = load_states(state_dir)
        if config_shards and filter_to_config:
            allowed = {sh for _, lst in config_shards for sh in lst}
            states = [(n, s, m) for n, s, m in states if n in allowed]
        states_map: Dict[str, Tuple[Dict, float]] = {name: (s, mtime) for name, s, mtime in states}
        total = len(states)
        downloaded = sum(1 for _, s, _ in states if s.get("downloaded"))
        extracted = sum(1 for _, s, _ in states if s.get("extracted"))
        scored = sum(1 for _, s, _ in states if s.get("scored"))
        uploaded = sum(1 for _, s, _ in states if s.get("uploaded"))
        stage_counts = Counter(s.get("stage", "unknown") for _, s, _ in states)

        rate_scored = None
        rate_uploaded = None
        if prev_ts is not None and now > prev_ts:
            delta_t = now - prev_ts
            if prev_scored is not None:
                rate_scored = (scored - prev_scored) * 60.0 / delta_t
            if prev_uploaded is not None:
                rate_uploaded = (uploaded - prev_uploaded) * 60.0 / delta_t
        prev_scored = scored
        prev_uploaded = uploaded
        prev_ts = now

        finished_durations = []
        for _, s, _ in states:
            if s.get("finished_at"):
                start_ts = s.get("started_at") or s["finished_at"]
                finished_durations.append(max(s["finished_at"] - start_ts, 0.0))

        # 读取已完成分片的 summary.json，汇总耗时/clip 数（不改主流程，只读文件）
        # output 与 state 同级目录下
        summaries_dir = state_dir.parent / "output"
        summary_count = 0
        sum_time = 0.0
        sum_clips = 0
        if summaries_dir.exists():
            for shard_name, s, _ in states:
                if not s.get("scored"):
                    continue
                summary_path = summaries_dir / shard_name / "summary.json"
                if not summary_path.exists():
                    continue
                try:
                    if shard_name not in summaries_cache:
                        with summary_path.open("r", encoding="utf-8") as f:
                            summaries_cache[shard_name] = json.load(f)
                    summary = summaries_cache[shard_name]
                    summary_count += 1
                    sum_time += float(summary.get("time_total", 0.0))
                    sum_clips += int(summary.get("clips_scored", 0))
                except Exception:
                    continue

        os.system("clear")
        print("=== Shard Monitor (refresh {:.1f}s) ===".format(interval))
        print(f"state dir: {state_dir}")
        print(f"total shards: {total}")
        load_avg = get_load_avg()
        if load_avg:
            print("cpu load avg (1/5/15m): {:.1f} / {:.1f} / {:.1f}".format(*load_avg))
        gpu_stats = get_gpu_stats()
        if gpu_stats:
            gpu_str = "; ".join(
                "gpu{idx}: {util:.0f}% {mem:.0f}/{mem_total:.0f}MiB {power:.0f}W".format(
                    idx=int(s["index"]),
                    util=s["util"],
                    mem=s["mem_used"],
                    mem_total=s["mem_total"],
                    power=s["power"],
                )
                for s in gpu_stats
            )
            print(f"gpus: {gpu_str}")
        print(
            "downloaded {}/{} | extracted {}/{} | scored {}/{} | uploaded {}/{}".format(
                downloaded, total, extracted, total, scored, total, uploaded, total
            )
        )
        if stage_counts:
            stage_str = ", ".join(f"{k}={v}" for k, v in stage_counts.most_common())
            print(f"stages: {stage_str}")
        if rate_scored is not None or rate_uploaded is not None:
            scored_str = f"{rate_scored:.2f}" if rate_scored is not None else "-"
            uploaded_str = f"{rate_uploaded:.2f}" if rate_uploaded is not None else "-"
            print(f"rate (shards/min): scored={scored_str}, uploaded={uploaded_str}")
        if finished_durations:
            avg_dur = sum(finished_durations) / len(finished_durations)
            print(f"avg shard wall time: {avg_dur/60:.1f} min over {len(finished_durations)} shard(s)")
        if summary_count:
            avg_time = (sum_time / summary_count) / 60
            clips_str = f"{sum_clips} clips" if sum_clips else "-"
            print(f"summary avg time: {avg_time:.1f} min over {summary_count} shard(s); total clips scored: {clips_str}")

        if config_shards:
            print("per-config (8-proc view):")
            for cfg_path, shards in config_shards:
                if not shards:
                    print(f"- {cfg_path.name}: no shard list")
                    continue
                d = sum(1 for sh in shards if states_map.get(sh, ({}, 0.0))[0].get("downloaded"))
                e = sum(1 for sh in shards if states_map.get(sh, ({}, 0.0))[0].get("extracted"))
                sc = sum(1 for sh in shards if states_map.get(sh, ({}, 0.0))[0].get("scored"))
                up = sum(1 for sh in shards if states_map.get(sh, ({}, 0.0))[0].get("uploaded"))
                active = [
                    (sh, states_map.get(sh)[0], states_map.get(sh)[1])
                    for sh in shards
                    if sh in states_map
                    and not states_map[sh][0].get("uploaded")
                    and states_map[sh][0].get("stage") not in done_stages
                ]
                active_str = "idle"
                if active:
                    # oldest first
                    sh, s, mtime = sorted(active, key=lambda x: x[2])[0]
                    start_ts = s.get("started_at")
                    age = format_age(now - start_ts) if start_ts else format_age(now - mtime)
                    stage = s.get("stage", "unknown")
                    progress = s.get("progress") or {}
                    parts = []
                    vt = progress.get("videos_total")
                    vd = progress.get("videos_done")
                    if vt is not None:
                        parts.append(f"videos {vd or 0}/{vt}")
                    cq = progress.get("clips_queued")
                    cs = progress.get("clips_scored")
                    if cq is not None or cs is not None:
                        parts.append(f"clips {cs or 0}/{cq or 0}")
                    qp = progress.get("queue_pending")
                    if qp is not None:
                        parts.append(f"queue {qp}")
                    lv = progress.get("last_video")
                    if lv:
                        parts.append(f"last {lv}")
                    active_str = f"{sh}: {stage}, {age}" + (f" | {'; '.join(parts)}" if parts else "")
                print(f"- {cfg_path.name}: {d}/{len(shards)} dl, {e} ext, {sc} sc, {up} up | {active_str}")

        active = [
            (name, s, mtime)
            for name, s, mtime in states
            if not s.get("uploaded") and s.get("stage") not in done_stages
        ]
        if active:
            active_sorted = sorted(active, key=lambda x: x[2])[:5]
            print("oldest active (up to 5):")
            for name, s, mtime in active_sorted:
                start_ts = s.get("started_at")
                age = format_age(now - start_ts) if start_ts else format_age(now - mtime)
                stage = s.get("stage", "unknown")
                progress = s.get("progress") or {}
                parts = []
                if progress:
                    vt = progress.get("videos_total")
                    vd = progress.get("videos_done")
                    if vt is not None:
                        parts.append(f"videos {vd or 0}/{vt}")
                    cq = progress.get("clips_queued")
                    cs = progress.get("clips_scored")
                    if cq is not None or cs is not None:
                        parts.append(f"clips {cs or 0}/{cq or 0}")
                    qp = progress.get("queue_pending")
                    if qp is not None:
                        parts.append(f"queue {qp}")
                    lv = progress.get("last_video")
                    if lv:
                        parts.append(f"last: {lv}")
                    ls = progress.get("last_step")
                    if ls:
                        parts.append(f"step: {ls}")
                extra = f" | {'; '.join(parts)}" if parts else ""
                print(f"- {name}: {stage}, last update {age} ago{extra}")
        else:
            print("no active shards.")

        print("==============================")
        time.sleep(interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight shard monitor using state/*.json")
    parser.add_argument(
        "--config",
        nargs="+",
        default=["config.yaml"],
        help="One or more config.yaml paths (default: config.yaml). Used to show per-config shard progress.",
    )
    parser.add_argument("--interval", type=float, default=2.0, help="Refresh interval in seconds (default: 2)")
    parser.add_argument(
        "--only-config-shards",
        action="store_true",
        help="Only show shards listed in the provided configs (hide stale/other shards in state dir).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_paths = [Path(p) for p in args.config]
    if not config_paths:
        raise ValueError("at least one --config is required")
    workdir = load_workdir(config_paths[0])
    config_shards: List[Tuple[Path, List[str]]] = []
    for cp in config_paths:
        try:
            shards = load_shards(cp)
        except Exception:
            shards = []
        config_shards.append((cp, shards))
    state_dir = workdir / "state"
    if not state_dir.exists():
        raise FileNotFoundError(f"state dir not found: {state_dir}")
    render(state_dir, args.interval, config_shards=config_shards, filter_to_config=args.only_config_shards)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nmonitor stopped by user")
