from __future__ import annotations

import inspect
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
    try:
        ocr = _get_ocr(cfg.lang, cfg.use_gpu)
    except Exception as exc:  # noqa: BLE001
        log.warning("OCR init failed for %s: %s; skipping OCR", video_path, exc)
        return False
    ocr_kwargs = _build_ocr_kwargs(ocr)
    vr = decord.VideoReader(str(video_path))
    total = len(vr)
    if total == 0:
        return False

    stride = max(cfg.sample_stride, 1)
    hit = False
    for idx in range(0, total, stride):
        frame = vr[idx].asnumpy()  # RGB
        if _text_area_ratio(frame, ocr, ocr_kwargs) >= cfg.text_area_threshold:
            hit = True
            break
    return hit


@lru_cache(maxsize=4)
def _get_ocr(lang: str, use_gpu: bool = False) -> PaddleOCR:  # type: ignore
    """
    兼容不同版本 PaddleOCR：仅传递构造函数支持的参数，避免 det/rec/use_gpu 报未知参数。
    """
    if PaddleOCR is None:  # 防御性分支
        raise RuntimeError("PaddleOCR not available")

    try:
        init_params = inspect.signature(PaddleOCR.__init__).parameters
    except Exception:
        init_params = {}

    kwargs = {"use_angle_cls": False, "lang": lang}
    for key, val in (("det", True), ("rec", False), ("use_gpu", use_gpu)):
        if key in init_params:
            kwargs[key] = val

    try:
        return PaddleOCR(**kwargs)
    except Exception as exc:
        log.warning("PaddleOCR init failed with %s; retrying with minimal args: %s", kwargs, exc)
        # 去掉可选参数再尝试
        for k in ("det", "rec", "use_gpu"):
            kwargs.pop(k, None)
        return PaddleOCR(**kwargs)


def _build_ocr_kwargs(ocr: PaddleOCR) -> dict[str, object]:  # type: ignore
    try:
        params = inspect.signature(ocr.ocr).parameters  # type: ignore[attr-defined]
    except (TypeError, ValueError):
        return {"det": True, "rec": False, "cls": False}
    kwargs: dict[str, object] = {}
    for key, value in (("det", True), ("rec", False), ("cls", False)):
        if key in params:
            kwargs[key] = value
    return kwargs


def _text_area_ratio(frame_rgb: np.ndarray, ocr: PaddleOCR, ocr_kwargs: dict[str, object]) -> float:  # type: ignore
    h, w, _ = frame_rgb.shape
    if h == 0 or w == 0:
        return 0.0
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    try:
        result = ocr.ocr(frame_bgr, **ocr_kwargs)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, TypeError) or "Unknown argument" in str(exc):
            log.warning("OCR call rejected kwargs %s: %s; retrying without kwargs", ocr_kwargs, exc)
            ocr_kwargs.clear()
            result = ocr.ocr(frame_bgr)
        else:
            raise
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
