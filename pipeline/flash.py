from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import FlashFilterConfig


def is_flashy(video_path: Path, cfg: FlashFilterConfig) -> bool:
    if not cfg.enabled:
        return False

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False

    prev_mean = None
    flash_events = 0
    total_samples = 0
    stride = max(cfg.sample_stride, 1)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_val = float(np.mean(gray))
        if prev_mean is not None:
            delta = abs(mean_val - prev_mean)
            if delta >= cfg.brightness_delta:
                flash_events += 1
        prev_mean = mean_val
        total_samples += 1
        frame_idx += 1

    cap.release()

    if total_samples == 0:
        return False
    flash_ratio = flash_events / max(total_samples, 1)
    return flash_ratio >= cfg.max_flash_ratio
