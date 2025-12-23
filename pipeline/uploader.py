from pathlib import Path

from huggingface_hub import HfApi

from .config import UploadConfig


def upload_shard(target_repo: str, shard: str, shard_dir: Path, cfg: UploadConfig) -> None:
    api = HfApi()
    api.upload_folder(
        repo_id=target_repo,
        folder_path=str(shard_dir),
        path_in_repo=shard,
        repo_type="dataset",
        commit_message=f"Add shard {shard}",
        max_workers=cfg.max_workers,
        chunk_size=cfg.chunk_size_mb * 1024 * 1024,
    )
