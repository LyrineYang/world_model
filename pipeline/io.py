import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable, List

from huggingface_hub import hf_hub_download
from tqdm import tqdm


VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".mpeg", ".mpg"}


def download_shard(repo_id: str, shard_name: str, target_dir: Path, token: str | None = None) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    # resume_download keeps progress for large files
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=shard_name,
        repo_type="dataset",
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        token=token,
    )
    return Path(local_path)


def extract_shard(archive_path: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    extract_path = target_dir / archive_path.stem
    if extract_path.exists():
        return extract_path
    extract_path.mkdir(parents=True, exist_ok=True)

    suffix = archive_path.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            _safe_extract_zip(zf, extract_path)
    elif suffix in {".tar", ".gz", ".tgz", ".bz2"} or archive_path.name.endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:*") as tf:
            _safe_extract_tar(tf, extract_path)
    else:
        # fall back to shutil for unknown formats
        shutil.unpack_archive(str(archive_path), extract_path)
    return extract_path


def list_video_files(root: Path, exclude_dirs: set[str] | None = None) -> List[Path]:
    return [
        p
        for p in tqdm(list(_walk_files(root, exclude_dirs)), desc="Enumerating videos", unit="file")
        if p.suffix.lower() in VIDEO_EXTS
    ]


def _walk_files(root: Path, exclude_dirs: set[str] | None = None) -> Iterable[Path]:
    exclude_dirs = exclude_dirs or set()
    for path in root.rglob("*"):
        if path.is_file():
            if any(part in exclude_dirs for part in path.parts):
                continue
            yield path


def _safe_extract_tar(tf: tarfile.TarFile, target_dir: Path) -> None:
    for member in tf.getmembers():
        member_path = target_dir / member.name
        if not str(member_path.resolve()).startswith(str(target_dir.resolve())):
            raise RuntimeError(f"Blocked path traversal in tar member: {member.name}")
    tf.extractall(target_dir)


def _safe_extract_zip(zf: zipfile.ZipFile, target_dir: Path) -> None:
    for member in zf.namelist():
        member_path = target_dir / member
        if not str(member_path.resolve()).startswith(str(target_dir.resolve())):
            raise RuntimeError(f"Blocked path traversal in zip member: {member}")
    zf.extractall(target_dir)
