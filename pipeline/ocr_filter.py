from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import List

import cv2
import numpy as np

from .config import OCRConfig

log = logging.getLogger(__name__)

try:
    from paddleocr import PaddleOCR  # type: ignore
except Exception as exc:  # noqa: BLE001
    PaddleOCR = None  # type: ignore
    log.warning("PaddleOCR not available: %s", exc)

try:
    import decord
except Exception as exc:  # noqa: BLE001
    decord = None
    log.warning("Decord not available for OCR: %s", exc)


def has_text(video_path: Path, cfg: OCRConfig) -> bool:
    if not cfg.enabled:
        return False
    if PaddleOCR is None or decord is None:
        log.warning("OCR skipped for %s because PaddleOCR/decord not available", video_path)
        return False

    ocr = _get_ocr(cfg.lang)
    vr = decord.VideoReader(str(video_path))
    total = len(vr)
    if total == 0:
        return False

    stride = max(cfg.sample_stride, 1)
    hit = False
    for idx in range(0, total, stride):
        frame = vr[idx].asnumpy()  # RGB
        if _text_area_ratio(frame, ocr) >= cfg.text_area_threshold:
            hit = True
            break
    return hit


@lru_cache(maxsize=2)
def _get_ocr(lang: str) -> PaddleOCR:  # type: ignore
    try:
        # 新版本参数，显式关闭识别分支以节省开销
        return PaddleOCR(use_angle_cls=False, lang=lang, det=True, rec=False)
    except Exception:
        # 兼容老版本 PaddleOCR 不支持 rec 参数的情况
        try:
            return PaddleOCR(use_angle_cls=False, lang=lang, det=True)
        except Exception:
            # 最简参数集，尽量兼容旧版
            return PaddleOCR(use_angle_cls=False, lang=lang)


def _text_area_ratio(frame_rgb: np.ndarray, ocr: PaddleOCR) -> float:  # type: ignore
    h, w, _ = frame_rgb.shape
    if h == 0 or w == 0:
        return 0.0
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    result = ocr.ocr(frame_bgr, det=True, rec=False, cls=False)
    # result: list per image -> we pass one image, so take first element
    if not result or not isinstance(result, list):
        return 0.0
    boxes = result[0] if isinstance(result[0], list) else []
    total_area = h * w
    text_area = 0.0
    for item in boxes:
        if not item:
            continue
        poly = item[0]
        area = _polygon_area(poly)
        text_area += max(area, 0.0)
    return float(text_area / total_area) if total_area > 0 else 0.0


def _polygon_area(points: List[List[float]]) -> float:
    if not points or len(points) < 3:
        return 0.0
    pts = np.array(points, dtype=np.float32)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
