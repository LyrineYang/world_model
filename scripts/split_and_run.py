from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Sequence

from pipeline.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按分片列表拆分并并行运行 pipeline，支持 CUDA_VISIBLE_DEVICES 绑定"
    )
    parser.add_argument("--config", default="configs/config.yaml", help="基础配置文件路径")
    parser.add_argument(
        "--shards-file",
        type=str,
        default=None,
        help="可选，覆盖 config 中的 shards_file",
    )
    parser.add_argument("--num-workers", type=int, default=1, help="并行进程数")
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="逗号分隔的 GPU id 列表，按 worker 轮转设置 CUDA_VISIBLE_DEVICES（例如 0,1,2）",
    )
    parser.add_argument("--skip-upload", action="store_true", help="传递给子进程的 --skip-upload")
    parser.add_argument(
        "--limit-shards",
        type=int,
        default=None,
        help="可选：总分片数上限（在切分前生效）",
    )
    parser.add_argument("--python-bin", default=sys.executable, help="Python 可执行路径")
    parser.add_argument("--calibration", action="store_true", help="传递给子进程的 --calibration")
    parser.add_argument("--sample-size", type=int, default=None, help="校准模式样本数")
    parser.add_argument("--calibration-output", type=str, default=None, help="校准输出路径")
    parser.add_argument("--calibration-quantiles", type=str, default=None, help="校准分位参数（逗号分隔）")
    return parser.parse_args()


def split_evenly(items: Sequence[str], num_chunks: int) -> List[List[str]]:
    if not items:
        return []
    n = max(1, num_chunks)
    n = min(n, len(items))
    size = (len(items) + n - 1) // n
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    override_shards = Path(args.shards_file).expanduser().resolve() if args.shards_file else None

    cfg = load_config(
        config_path,
        limit_shards=args.limit_shards,
        skip_upload=args.skip_upload,
        override_shards_file=override_shards,
    )
    shards = list(cfg.shards)
    if not shards:
        print("未找到分片列表（检查 config.shards / shards_file）", file=sys.stderr)
        return 1

    chunks = split_evenly(shards, args.num_workers)
    gpu_ids = [g.strip() for g in (args.gpus.split(",") if args.gpus else []) if g.strip()]

    print(
        f"总分片 {len(shards)} → {len(chunks)} 个任务；"
        f"config={config_path} shards_file_override={override_shards}"
    )

    processes: list[tuple[int, subprocess.Popen]] = []
    tmpdir_obj = tempfile.TemporaryDirectory(prefix="shards_split_")
    tmpdir = Path(tmpdir_obj.name)
    try:
        for idx, chunk in enumerate(chunks):
            if not chunk:
                continue
            shard_file = tmpdir / f"shards_worker{idx}.txt"
            shard_file.write_text("\n".join(chunk) + "\n", encoding="utf-8")

            cmd = [
                args.python_bin,
                "-m",
                "pipeline.pipeline",
                "--config",
                str(config_path),
                "--shards-file",
                str(shard_file),
            ]
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

            env = os.environ.copy()
            assigned_gpu = None
            if gpu_ids:
                assigned_gpu = gpu_ids[idx % len(gpu_ids)]
                env["CUDA_VISIBLE_DEVICES"] = assigned_gpu

            print(
                f"[worker {idx}] shards={len(chunk)} "
                f"CUDA_VISIBLE_DEVICES={assigned_gpu or env.get('CUDA_VISIBLE_DEVICES', 'inherit')} "
                f"cmd={' '.join(cmd)}"
            )
            proc = subprocess.Popen(cmd, env=env)
            processes.append((idx, proc))

        failures = 0
        for idx, proc in processes:
            code = proc.wait()
            if code != 0:
                failures += 1
                print(f"[worker {idx}] 退出码 {code}", file=sys.stderr)
        if failures:
            print(f"{failures} 个子进程失败", file=sys.stderr)
            return 1
        return 0
    finally:
        tmpdir_obj.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
