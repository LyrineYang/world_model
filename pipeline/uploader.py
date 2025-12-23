from pathlib import Path

from huggingface_hub import HfApi

from .config import UploadConfig


def upload_shard(target_repo: str, shard: str, shard_dir: Path, cfg: UploadConfig) -> None:
    api = HfApi()
    # 兼容旧版 huggingface_hub：max_workers/chunk_size 可能不存在
    kwargs = dict(
        repo_id=target_repo,
        folder_path=str(shard_dir),
        path_in_repo=shard,
        repo_type="dataset",
        commit_message=f"Add shard {shard}",
    )
    try:
        api.upload_folder(
            **kwargs,
            max_workers=cfg.max_workers,
            chunk_size=cfg.chunk_size_mb * 1024 * 1024,
        )
    except TypeError:
        api.upload_folder(**kwargs)
