import json
from pathlib import Path
from typing import Any, Dict


def _state_path(state_dir: Path, shard: str) -> Path:
    return state_dir / f"{shard}.json"


def load_state(state_dir: Path, shard: str) -> Dict[str, Any]:
    path = _state_path(state_dir, shard)
    if not path.exists():
        return {
            "downloaded": False,
            "extracted": False,
            "scored": False,
            "uploaded": False,
        }
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state_dir: Path, shard: str, state: Dict[str, Any]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = _state_path(state_dir, shard)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
