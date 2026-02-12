from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class ClipRecord:
    """
    Unified clip-level input for offline filtering.

    `text` and `meta` are optional so existing path-only workflows can migrate
    incrementally.
    """

    video_path: Path
    text: str | None = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.video_path = Path(self.video_path)

