import json
import os
import shutil
from pathlib import Path
from typing import Iterable, List

from .models import ScoreResult


def materialize_results(shard: str, results: Iterable[ScoreResult], output_root: Path) -> Path:
    shard_out = output_root / shard
    videos_out = shard_out / "videos"
    shard_out.mkdir(parents=True, exist_ok=True)
    videos_out.mkdir(parents=True, exist_ok=True)

    metadata_path = shard_out / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as f:
        for res in results:
            target = videos_out / res.path.name if res.keep and res.path.exists() else None
            record = {
                "source_path": str(res.path),
                "output_path": str(target) if target else None,
                "size_bytes": res.path.stat().st_size if res.path.exists() else 0,
                "scores": res.scores,
                "keep": res.keep,
                "reason": res.reason,
            }
            f.write(json.dumps(record) + "\n")
            if res.keep and res.path.exists() and target is not None:
                _link_or_copy(res.path, target)
    return metadata_path


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
