from __future__ import annotations

import logging
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Protocol

import numpy as np
import torch
import yaml
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent

log = logging.getLogger(__name__)

try:
    import decord
except Exception as exc:  # noqa: BLE001
    decord = None
    log.warning("Decord not available: %s", exc)

try:
    import open_clip
except Exception as exc:  # noqa: BLE001
    open_clip = None
    log.warning("open_clip not available: %s", exc)
try:
    import cv2
except Exception as exc:  # noqa: BLE001
    cv2 = None
    log.warning("opencv not available: %s", exc)


from .config import ModelConfig


@dataclass
class ScoreResult:
    path: Path
    scores: Dict[str, float]
    keep: bool
    reason: str | None = None


class Scorer(Protocol):
    name: str
    threshold: float

    def score_batch(self, items: List[Path]) -> List[float]:
        ...


class DummyScorer:
    """Baseline scorer that always returns 1.0. Useful for connectivity tests."""

    def __init__(self, cfg: ModelConfig):
        self.name = cfg.name
        self.threshold = cfg.threshold
        self.device = cfg.device
        self.batch_size = cfg.batch_size

    def score_batch(self, items: List[Path]) -> List[float]:
        return [1.0 for _ in items]


class DoverScorer:
    """
    DOVER 视频质量评分（融合技术/美学得分），默认使用 DOVER 预训练权重。

    依赖：
    - 本地 DOVER 仓库（cfg.extra.repo_path，默认 ./DOVER）
    - 权重：cfg.extra.weight_path（默认 pretrained_weights/DOVER.pth），若不存在自动从 HF teowu/DOVER 下载
    - 配置：cfg.extra.config_path（默认 dover.yml），cfg.extra.data_key（默认 val-l1080p）
    """

    def __init__(self, cfg: ModelConfig):
        if decord is None:
            raise ImportError("decord is required for DoverScorer")

        self.name = cfg.name
        self.threshold = cfg.threshold
        self.device = cfg.device
        self.batch_size = cfg.batch_size

        repo_path_cfg = Path(cfg.extra.get("repo_path", REPO_ROOT / "DOVER"))
        self.repo_path = repo_path_cfg if repo_path_cfg.is_absolute() else (REPO_ROOT / repo_path_cfg)
        config_path_cfg = Path(cfg.extra.get("config_path", "dover.yml"))
        self.config_path = config_path_cfg if config_path_cfg.is_absolute() else (self.repo_path / config_path_cfg)
        weight_rel = Path(cfg.extra.get("weight_path", "pretrained_weights/DOVER.pth"))
        self.weight_path = weight_rel if weight_rel.is_absolute() else (self.repo_path / weight_rel)
        self.data_key = cfg.extra.get("data_key", "val-l1080p")
        self.output_mode = cfg.extra.get("output", "fused")  # fused | technical | aesthetic

        sys.path.insert(0, str(self.repo_path))
        import dover.datasets as dover_datasets  # type: ignore
        import dover.models as dover_models  # type: ignore

        self._ensure_weight()
        opt = self._load_yaml(self.config_path)
        if self.data_key not in opt["data"]:
            raise KeyError(f"data_key {self.data_key} not found in {self.config_path}")
        self.sample_cfg = opt["data"][self.data_key]["args"]["sample_types"]

        # Build samplers (copy from evaluate_one_video)
        self.temporal_samplers = {}
        for stype, sopt in self.sample_cfg.items():
            if "t_frag" not in sopt:
                self.temporal_samplers[stype] = dover_datasets.UnifiedFrameSampler(
                    sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
                )
            else:
                self.temporal_samplers[stype] = dover_datasets.UnifiedFrameSampler(
                    sopt["clip_len"] // sopt["t_frag"],
                    sopt["t_frag"],
                    sopt["frame_interval"],
                    sopt["num_clips"],
                )

        self.mean = torch.FloatTensor([123.675, 116.28, 103.53]).to(self.device)
        self.std = torch.FloatTensor([58.395, 57.12, 57.375]).to(self.device)

        # Model
        self.model = dover_models.DOVER(**opt["model"]["args"]).to(self.device)
        state = torch.load(self.weight_path, map_location=self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.spatial_temporal_view_decomposition = dover_datasets.spatial_temporal_view_decomposition

    def _ensure_weight(self) -> None:
        if self.weight_path.exists():
            return
        self.weight_path.parent.mkdir(parents=True, exist_ok=True)
        fname = self.weight_path.name
        log.info("Downloading DOVER weight %s via hf_hub_download", fname)
        downloaded = hf_hub_download(
            repo_id="teowu/DOVER",
            filename=fname,
            repo_type="model",
            local_dir=str(self.weight_path.parent),
            local_dir_use_symlinks=False,
        )
        self.weight_path = Path(downloaded)

    def _load_yaml(self, path: Path) -> dict:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def score_batch(self, items: List[Path]) -> List[float]:
        scores: List[float] = []
        for video_path in items:
            try:
                scores.append(self._score_single(video_path))
            except Exception as exc:  # noqa: BLE001
                log.warning("Dover scoring failed for %s: %s", video_path, exc)
                scores.append(None)
        return scores

    def _score_single(self, video_path: Path) -> float:
        views, _ = self.spatial_temporal_view_decomposition(
            str(video_path), self.sample_cfg, self.temporal_samplers
        )
        for k, v in views.items():
            num_clips = self.sample_cfg[k].get("num_clips", 1)
            views[k] = (
                ((v.permute(1, 2, 3, 0).to(self.device) - self.mean) / self.std)
                .permute(3, 0, 1, 2)
                .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                .transpose(0, 1)
            )

        with torch.no_grad():
            preds = self.model(views)
        branch_scores = [p.mean().item() for p in preds]

        if self.output_mode == "technical":
            return branch_scores[0]
        if self.output_mode == "aesthetic":
            return branch_scores[1] if len(branch_scores) > 1 else branch_scores[0]
        return self._fuse_scores(branch_scores)

    @staticmethod
    def _fuse_scores(results: List[float]) -> float:
        """
        Fused overall score in [0,1], copy from evaluate_one_video.py fuse_results.
        """
        if len(results) < 2:
            return float(results[0])
        x = (results[0] - 0.1107) / 0.07355 * 0.6104 + (results[1] + 0.08285) / 0.03774 * 0.3896
        return 1.0 / (1.0 + math.exp(-x))


class LaionAesScorer:
    """
    LAION-AES 视频美学评分：对每个视频均匀采样若干帧，使用 CLIP 提取特征 + 线性头预测，取平均得分。

    配置 extra：
    - clip_model: CLIP backbone 名称，默认 ViT-L-14
    - pretrained: open_clip 预训练标签，默认 openai
    - weight_path: 线性头路径，默认 aesthetic-predictor/sa_0_4_vit_l_14_linear.pth
    - num_frames: 采样帧数，默认 8
    """

    def __init__(self, cfg: ModelConfig):
        if decord is None:
            raise ImportError("decord is required for LaionAesScorer")
        if open_clip is None:
            raise ImportError("open_clip is required for LaionAesScorer")

        self.name = cfg.name
        self.threshold = cfg.threshold
        self.device = cfg.device
        self.batch_size = cfg.batch_size

        self.clip_model = cfg.extra.get("clip_model", "ViT-L-14")
        self.pretrained = cfg.extra.get("pretrained", "openai")
        weight_rel = Path(cfg.extra.get("weight_path", REPO_ROOT / "aesthetic-predictor/sa_0_4_vit_l_14_linear.pth"))
        self.weight_path = weight_rel if weight_rel.is_absolute() else REPO_ROOT / weight_rel
        self.num_frames = int(cfg.extra.get("num_frames", 8))

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.clip_model, pretrained=self.pretrained, device=self.device
        )
        self.model.eval()
        emb_dim = self._embedding_dim()
        self.head = torch.nn.Linear(emb_dim, 1, bias=True)
        state = torch.load(self.weight_path, map_location="cpu")
        self.head.load_state_dict(state)
        self.head = self.head.to(self.device).eval()

    def _embedding_dim(self) -> int:
        if "32" in self.clip_model.lower():
            return 512
        return 768

    def score_batch(self, items: List[Path]) -> List[float]:
        scores: List[float] = []
        for video_path in items:
            try:
                scores.append(self._score_single(video_path))
            except Exception as exc:  # noqa: BLE001
                log.warning("LAION-AES scoring failed for %s: %s", video_path, exc)
                scores.append(None)
        return scores

    def _score_single(self, video_path: Path) -> float:
        frames = _sample_video_frames(video_path, self.num_frames)
        if not frames:
            return 0.0

        images = [_to_image(frame) for frame in frames]
        inputs = torch.stack([self.preprocess(img) for img in images]).to(self.device)

        with torch.no_grad():
            feats = self.model.encode_image(inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            preds = self.head(feats).squeeze(-1)
        return float(preds.mean().item())


class UnimatchFlowScorer:
    """
    UniMatch 光流运动评分：采样若干帧对，计算平均光流幅值；可用于过滤低运动/PPT。

    配置 extra：
    - repo_path: UniMatch 源码路径，默认 ./unimatch
    - weight_path: 预训练权重，默认 pretrained/gmflow-scale1-mixdata-train320x576-4c3a6e9a.pth
    - num_pairs: 每个视频采样的帧对数量，默认 3
    - resize: [h, w] 可选，推理前调整分辨率，降低算力
    - padding_factor: 默认为 16（与 gmflow scale1 对齐）
    - attn_splits_list/corr_radius_list/prop_radius_list/num_scales/upsample_factor/num_transformer_layers/reg_refine: 见 UniMatch 参数
    """

    def __init__(self, cfg: ModelConfig):
        if decord is None:
            raise ImportError("decord is required for UnimatchFlowScorer")

        self.name = cfg.name
        self.threshold = cfg.threshold
        self.device = cfg.device
        self.batch_size = cfg.batch_size

        repo_path_cfg = Path(cfg.extra.get("repo_path", REPO_ROOT / "unimatch"))
        self.repo_path = repo_path_cfg if repo_path_cfg.is_absolute() else (REPO_ROOT / repo_path_cfg)
        self.weight_path = self._resolve_weight(cfg.extra.get("weight_path", "pretrained/gmflow-scale1-mixdata-train320x576-4c3a6e9a.pth"))
        self.num_pairs = int(cfg.extra.get("num_pairs", 3))
        self.resize = cfg.extra.get("resize", None)
        self.padding_factor = int(cfg.extra.get("padding_factor", 16))
        self.num_scales = int(cfg.extra.get("num_scales", 1))
        self.upsample_factor = int(cfg.extra.get("upsample_factor", 8))
        self.attn_splits_list = cfg.extra.get("attn_splits_list", [2])
        self.corr_radius_list = cfg.extra.get("corr_radius_list", [-1])
        self.prop_radius_list = cfg.extra.get("prop_radius_list", [-1])
        self.num_reg_refine = int(cfg.extra.get("num_reg_refine", 1))
        self.num_transformer_layers = int(cfg.extra.get("num_transformer_layers", 6))
        self.feature_channels = int(cfg.extra.get("feature_channels", 128))
        self.num_head = int(cfg.extra.get("num_head", 1))
        self.ffn_dim_expansion = int(cfg.extra.get("ffn_dim_expansion", 4))
        self.reg_refine = bool(cfg.extra.get("reg_refine", False))

        sys.path.insert(0, str(self.repo_path))
        from unimatch.unimatch import UniMatch  # type: ignore
        from utils.utils import InputPadder  # type: ignore

        self.InputPadder = InputPadder
        self.model = UniMatch(
            feature_channels=self.feature_channels,
            num_scales=self.num_scales,
            upsample_factor=self.upsample_factor,
            num_head=self.num_head,
            ffn_dim_expansion=self.ffn_dim_expansion,
            num_transformer_layers=self.num_transformer_layers,
            reg_refine=self.reg_refine,
            task="flow",
        ).to(self.device)
        if not self.weight_path.exists():
            raise FileNotFoundError(
                f"UniMatch weight not found at {self.weight_path}. "
                "Download from MODEL_ZOO.md (e.g., gmflow-scale1-mixdata) to this path "
                "or override weight_path in config."
            )
        state = torch.load(self.weight_path, map_location=self.device)
        # support DataParallel checkpoints
        if "module.backbone.0.0.weight" in state:
            self.model.load_state_dict(state)
        else:
            self.model.load_state_dict(state, strict=False)
        self.model.eval()

    def _resolve_weight(self, weight_rel: str) -> Path:
        path = Path(weight_rel)
        if not path.is_absolute():
            path = self.repo_path / path
        return path

    def score_batch(self, items: List[Path]) -> List[float]:
        scores: List[float] = []
        for video_path in items:
            try:
                scores.append(self._score_single(video_path))
            except Exception as exc:  # noqa: BLE001
                log.warning("Unimatch scoring failed for %s: %s", video_path, exc)
                scores.append(None)
        return scores

    def _score_single(self, video_path: Path) -> float:
        pairs = _sample_frame_pairs(video_path, self.num_pairs)
        if not pairs:
            return 0.0

        mags: List[float] = []
        for f0, f1 in pairs:
            img0 = _to_torch_image(f0, self.device)
            img1 = _to_torch_image(f1, self.device)
            if self.resize:
                h, w = self.resize
                img0 = torch.nn.functional.interpolate(img0, size=(h, w), mode="bilinear", align_corners=True)
                img1 = torch.nn.functional.interpolate(img1, size=(h, w), mode="bilinear", align_corners=True)
            padder = self.InputPadder(img0.shape, padding_factor=self.padding_factor)
            img0, img1 = padder.pad(img0, img1)

            with torch.no_grad():
                results = self.model(
                    img0,
                    img1,
                    attn_type="swin",
                    attn_splits_list=self.attn_splits_list,
                    corr_radius_list=self.corr_radius_list,
                    prop_radius_list=self.prop_radius_list,
                    num_reg_refine=self.num_reg_refine,
                    task="flow",
                )
                flow = results["flow_preds"][-1]
                flow = padder.unpad(flow[0]).cpu()
                mag = torch.sqrt((flow ** 2).sum(dim=0)).mean().item()
                mags.append(mag)
        if not mags:
            return 0.0
        return float(np.mean(mags))


def build_scorers(model_cfgs: Iterable[ModelConfig]) -> List[Scorer]:
    scorers: List[Scorer] = []
    for cfg in model_cfgs:
        if cfg.kind == "dummy":
            scorers.append(DummyScorer(cfg))
        elif cfg.kind == "dover":
            scorers.append(DoverScorer(cfg))
        elif cfg.kind == "laion_aes":
            scorers.append(LaionAesScorer(cfg))
        elif cfg.kind == "unimatch_flow":
            scorers.append(UnimatchFlowScorer(cfg))
        else:
            raise ValueError(f"Unsupported model kind: {cfg.kind}")
    return scorers


def run_scorers(scorers: List[Scorer], items: List[Path], max_workers: int | None = None) -> List[ScoreResult]:
    results: List[ScoreResult] = []
    if not scorers:
        return [ScoreResult(path=p, scores={}, keep=True, reason=None) for p in items]

    workers = max_workers if max_workers and max_workers > 1 else 1
    use_parallel = workers > 1 and len(scorers) > 1
    step = _min_batch_size(scorers)
    executor = ThreadPoolExecutor(max_workers=workers) if use_parallel else None
    try:
        for batch_start in tqdm(range(0, len(items), step), desc="Scoring", unit="batch"):
            batch = items[batch_start : batch_start + step]
            batch_scores: Dict[str, List[float]] = {}
            if executor:
                futures = {executor.submit(scorer.score_batch, batch): scorer.name for scorer in scorers}
                for fut in as_completed(futures):
                    name = futures[fut]
                    scores = fut.result()
                    if len(scores) != len(batch):
                        raise ValueError(f"Scorer {name} returned {len(scores)} scores for {len(batch)} items")
                    batch_scores[name] = scores
            else:
                for scorer in scorers:
                    scores = scorer.score_batch(batch)
                    if len(scores) != len(batch):
                        raise ValueError(f"Scorer {scorer.name} returned {len(scores)} scores for {len(batch)} items")
                    batch_scores[scorer.name] = scores

            for idx, path in enumerate(batch):
                per_item_scores = {name: batch_scores[name][idx] for name in batch_scores}
                has_invalid = any(v is None or (isinstance(v, float) and math.isnan(v)) for v in per_item_scores.values())
                if has_invalid:
                    keep = False
                    reason = "scoring_error"
                else:
                    keep = all(per_item_scores[name] >= scorer.threshold for name, scorer in _name_map(scorers).items())
                    reason = None if keep else "score_below_threshold"
                results.append(ScoreResult(path=path, scores=per_item_scores, keep=keep, reason=reason))
    finally:
        if executor:
            executor.shutdown(wait=True)
    return results


def _name_map(scorers: List[Scorer]) -> Dict[str, Scorer]:
    return {s.name: s for s in scorers}


def _min_batch_size(scorers: List[Scorer]) -> int:
    return min(max(s.batch_size, 1) for s in scorers) if scorers else 1


def _sample_video_frames(video_path: Path, num_frames: int) -> List[np.ndarray]:
    if decord is None:
        return []
    vr = decord.VideoReader(str(video_path))
    total = len(vr)
    if total == 0:
        return []
    indices = _uniform_indices(total, num_frames)
    frames = [_frame_to_numpy(vr[idx]) for idx in indices]
    return frames


def _uniform_indices(total: int, num: int) -> List[int]:
    if num >= total:
        return list(range(total))
    step = total / num
    return [int(i * step) for i in range(num)]


def _to_image(frame: np.ndarray) -> Image.Image:
    # decord returns HWC RGB
    return Image.fromarray(frame)


def _to_torch_image(frame: np.ndarray, device: str) -> torch.Tensor:
    # frame HWC uint8 -> BCHW float
    tensor = torch.from_numpy(frame).permute(2, 0, 1).float()[None]  # 1,C,H,W
    return tensor.to(device)


def _sample_frame_pairs(video_path: Path, num_pairs: int) -> List[tuple[np.ndarray, np.ndarray]]:
    if decord is None:
        return []
    vr = decord.VideoReader(str(video_path))
    total = len(vr)
    if total < 2:
        return []
    if num_pairs <= 0:
        num_pairs = 1
    stride = max(total // (num_pairs + 1), 1)
    pairs = []
    idx = 0
    for _ in range(num_pairs):
        if idx + 1 >= total:
            break
        f0 = _frame_to_numpy(vr[idx])
        f1 = _frame_to_numpy(vr[min(idx + 1, total - 1)])
        pairs.append((f0, f1))
        idx += stride
    return pairs


def _frame_to_numpy(frame) -> np.ndarray:
    """
    decord 默认返回 NDArray（有 asnumpy），但若桥接到 torch 会直接返回 Tensor。
    统一转为 numpy，兼容两种情况，避免 'Tensor' object has no attribute asnumpy。
    """
    if hasattr(frame, "asnumpy"):
        return frame.asnumpy()
    try:
        import torch

        if isinstance(frame, torch.Tensor):
            return frame.detach().cpu().numpy()
    except Exception:
        pass
    return np.array(frame)
