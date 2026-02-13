from __future__ import annotations

import base64
import io
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable

import requests

try:
    from PIL import Image
except Exception as exc:  # noqa: BLE001
    Image = None  # type: ignore[assignment]
    logging.getLogger(__name__).warning("Pillow not available for caption frame extraction: %s", exc)

from .config import CaptionConfig, CaptionRewriteConfig, CaptionStageConfig

log = logging.getLogger(__name__)

try:
    import decord
except Exception as exc:  # noqa: BLE001
    decord = None
    log.warning("Decord not available for caption frame extraction: %s", exc)


def generate_captions(videos: Iterable[Path], cfg: CaptionConfig) -> Dict[str, str]:
    """
    Backward-compatible caption API.

    Returns path -> final caption text.
    """
    records = generate_captions_with_meta(videos, cfg)
    return {path: rec["caption"] for path, rec in records.items() if rec.get("caption")}


def generate_captions_with_meta(videos: Iterable[Path], cfg: CaptionConfig) -> Dict[str, Dict[str, str | None]]:
    """
    Generate captions with stage metadata.

    Returns path -> {"caption", "caption_visual", "caption_error"}.
    """
    if not cfg.enabled:
        return {}
    paths = list(videos)
    if not paths:
        return {}
    generator = CaptionGenerator(cfg)
    return generator.generate_with_meta(paths)


class CaptionGenerator:
    def __init__(self, cfg: CaptionConfig):
        self.cfg = cfg
        self.visual_cfg, self.rewrite_cfg = self._resolve_stages(cfg)
        self._qwen_local_cache: dict[tuple[str, str, str, bool], tuple[Any, Any, str]] = {}
        if self.visual_cfg is None and self.rewrite_cfg is None:
            raise ValueError("caption is enabled but no caption stage is configured")

    def _resolve_stages(self, cfg: CaptionConfig) -> tuple[CaptionStageConfig | None, CaptionRewriteConfig | None]:
        # New two-stage mode.
        if cfg.visual_caption is not None or cfg.rewrite is not None:
            visual = cfg.visual_caption if cfg.visual_caption and cfg.visual_caption.enabled else None
            rewrite = cfg.rewrite if cfg.rewrite and cfg.rewrite.enabled else None
            if rewrite and rewrite.provider not in {"openai_compatible", "openrouter", "qwen3_local"}:
                raise ValueError(
                    f"Unsupported rewrite provider: {rewrite.provider}. "
                    "Expected openai_compatible/openrouter/qwen3_local."
                )
            return visual, rewrite

        # Legacy single-stage mode.
        if not cfg.enabled:
            return None, None
        legacy = CaptionStageConfig(
            enabled=cfg.enabled,
            provider=cfg.provider,
            api_url=cfg.api_url,
            api_key=cfg.api_key,
            api_key_header=cfg.api_key_header,
            model=cfg.model,
            system_prompt=cfg.system_prompt,
            user_prompt=cfg.user_prompt,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            timeout=cfg.timeout,
            max_workers=cfg.max_workers,
            retry=cfg.retry,
            file_field=cfg.file_field,
            response_field=cfg.response_field,
            extra_fields=cfg.extra_fields,
            include_image=cfg.include_image,
            image_num_frames=cfg.image_num_frames,
            image_max_side=cfg.image_max_side,
            openrouter_referer=cfg.openrouter_referer,
            openrouter_title=cfg.openrouter_title,
            local_model_path=cfg.local_model_path,
            local_device=cfg.local_device,
            local_dtype=cfg.local_dtype,
            local_max_new_tokens=cfg.local_max_new_tokens,
            local_trust_remote_code=cfg.local_trust_remote_code,
        )
        return legacy, None

    def generate(self, paths: list[Path]) -> Dict[str, str]:
        with_meta = self.generate_with_meta(paths)
        return {path: rec["caption"] for path, rec in with_meta.items() if rec.get("caption")}

    def generate_with_meta(self, paths: list[Path]) -> Dict[str, Dict[str, str | None]]:
        results: Dict[str, Dict[str, str | None]] = {}
        workers = max(self._max_workers(), 1)
        if workers > 1 and len(paths) > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(self._generate_single_with_meta, p): p for p in paths}
                for fut in as_completed(futures):
                    path = futures[fut]
                    rec = self._safe_result(fut, path)
                    if rec:
                        results[str(path)] = rec
        else:
            for path in paths:
                try:
                    rec = self._generate_single_with_meta(path)
                    if rec:
                        results[str(path)] = rec
                except Exception as exc:  # noqa: BLE001
                    log.warning("Caption failed for %s: %s", path, exc)
                    results[str(path)] = {
                        "caption": None,
                        "caption_visual": None,
                        "caption_error": f"caption_failed:{exc}",
                    }
        return results

    def _max_workers(self) -> int:
        vals = []
        if self.visual_cfg and self.visual_cfg.enabled:
            vals.append(int(self.visual_cfg.max_workers or 1))
        if self.rewrite_cfg and self.rewrite_cfg.enabled:
            vals.append(int(self.rewrite_cfg.max_workers or 1))
        return max(vals) if vals else 1

    def _safe_result(self, fut, path: Path) -> Dict[str, str | None]:
        try:
            return fut.result()
        except Exception as exc:  # noqa: BLE001
            log.warning("Caption failed for %s: %s", path, exc)
            return {"caption": None, "caption_visual": None, "caption_error": f"caption_failed:{exc}"}

    def _generate_single(self, path: Path) -> str | None:
        rec = self._generate_single_with_meta(path)
        return rec.get("caption")

    def _generate_single_with_meta(self, path: Path) -> Dict[str, str | None]:
        out: Dict[str, str | None] = {
            "caption": None,
            "caption_visual": None,
            "caption_error": None,
        }

        visual_caption: str | None = None
        if self.visual_cfg and self.visual_cfg.enabled:
            try:
                visual_caption = self._call_provider(path, self.visual_cfg)
                if visual_caption:
                    visual_caption = visual_caption.strip()
                    out["caption_visual"] = visual_caption
            except Exception as exc:  # noqa: BLE001
                out["caption_error"] = f"visual_failed:{exc}"
                return out

        if self.rewrite_cfg and self.rewrite_cfg.enabled:
            if not visual_caption:
                out["caption_error"] = "rewrite_skipped:visual_caption_missing"
                return out
            try:
                rewritten = self._rewrite_caption(path, visual_caption, self.rewrite_cfg)
                if rewritten:
                    out["caption"] = rewritten.strip()
                else:
                    out["caption"] = visual_caption
                    out["caption_error"] = "rewrite_empty_fallback_visual"
            except Exception as exc:  # noqa: BLE001
                out["caption"] = visual_caption
                out["caption_error"] = f"rewrite_failed:{exc}"
            return out

        out["caption"] = visual_caption
        return out

    def _rewrite_caption(self, path: Path, visual_caption: str, rewrite_cfg: CaptionRewriteConfig) -> str | None:
        template = (
            rewrite_cfg.user_prompt_template
            or rewrite_cfg.user_prompt
            or "请将以下描述改写成简洁第一人称动作短句：{visual_caption}"
        )
        try:
            user_text = template.format(visual_caption=visual_caption, filename=path.name)
        except Exception:
            user_text = f"{template}\n{visual_caption}"

        if rewrite_cfg.provider == "openai_compatible":
            return self._call_openai_compatible(
                path=path,
                stage_cfg=rewrite_cfg,
                user_text=user_text,
                include_image=False,
            )
        if rewrite_cfg.provider == "openrouter":
            return self._call_openrouter(
                path,
                stage_cfg=rewrite_cfg,
                user_text=user_text,
                include_image=False,
            )
        if rewrite_cfg.provider == "qwen3_local":
            return self._call_qwen3_local(
                path=path,
                stage_cfg=rewrite_cfg,
                user_text=user_text,
                include_image=False,
            )
        raise ValueError(f"Unsupported rewrite provider: {rewrite_cfg.provider}")

    def _call_provider(self, path: Path, stage_cfg: CaptionStageConfig) -> str | None:
        if stage_cfg.provider == "api":
            return self._call_api(path, stage_cfg=stage_cfg)
        if stage_cfg.provider == "openrouter":
            return self._call_openrouter(path, stage_cfg=stage_cfg)
        if stage_cfg.provider == "openai_compatible":
            return self._call_openai_compatible(
                path=path,
                stage_cfg=stage_cfg,
                user_text=self._build_text_prompt(path, stage_cfg),
                include_image=stage_cfg.include_image,
            )
        if stage_cfg.provider == "qwen3_local":
            return self._call_qwen3_local(
                path=path,
                stage_cfg=stage_cfg,
                user_text=self._build_text_prompt(path, stage_cfg),
                include_image=stage_cfg.include_image,
            )
        raise ValueError(f"Unsupported caption provider: {stage_cfg.provider}")

    def _call_api(self, path: Path, stage_cfg: CaptionStageConfig | None = None) -> str | None:
        cfg = stage_cfg or self.visual_cfg or self.cfg
        if not cfg.api_url:
            raise ValueError("caption.api_url is required when provider=api")
        headers = {}
        if cfg.api_key:
            headers[cfg.api_key_header or "Authorization"] = cfg.api_key
        timeout = max(float(cfg.timeout or 60.0), 0.1)
        file_field = cfg.file_field or "file"
        retries = max(int(cfg.retry or 0), 0) + 1
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                with path.open("rb") as f:
                    files = {file_field: (path.name, f, "video/mp4")}
                    resp = requests.post(
                        cfg.api_url,
                        headers=headers,
                        data=cfg.extra_fields or {},
                        files=files,
                        timeout=timeout,
                    )
                if resp.status_code >= 500 and attempt < retries - 1:
                    time.sleep(1.0)
                    continue
                resp.raise_for_status()
                caption = self._parse_response(resp, response_field=cfg.response_field)
                return caption.strip() if isinstance(caption, str) else caption
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < retries - 1:
                    time.sleep(1.0)
        if last_exc:
            raise last_exc
        return None

    def _parse_response(self, resp: requests.Response, response_field: str | None = None) -> str | None:
        try:
            data = resp.json()
        except ValueError:
            text = resp.text.strip()
            return text or None

        if isinstance(data, dict):
            if response_field and data.get(response_field) is not None:
                return str(data[response_field])
            for key in ("caption", "text", "message"):
                if key in data and data[key] is not None:
                    return str(data[key])
            for val in data.values():
                if isinstance(val, (str, int, float)):
                    return str(val)
            return None

        if isinstance(data, str):
            return data
        return None

    def _call_openrouter(
        self,
        path: Path,
        stage_cfg: CaptionStageConfig | None = None,
        user_text: str | None = None,
        include_image: bool | None = None,
    ) -> str | None:
        cfg = stage_cfg or self.visual_cfg or self.cfg
        api_url = self._clean_placeholder(cfg.api_url) or "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        api_key = self._clean_placeholder(cfg.api_key) or os.getenv("OPENROUTER_API_KEY")
        if api_key:
            headers[cfg.api_key_header or "Authorization"] = f"Bearer {api_key}"
        if cfg.openrouter_referer:
            headers["HTTP-Referer"] = cfg.openrouter_referer
        if cfg.openrouter_title:
            headers["X-Title"] = cfg.openrouter_title

        text_prompt = user_text if user_text is not None else self._build_text_prompt(path, cfg)
        content: list[dict[str, Any]] = [{"type": "text", "text": text_prompt}]
        enable_image = cfg.include_image if include_image is None else include_image
        images_b64 = self._extract_images_b64(path, cfg) if enable_image else []
        for image_b64 in images_b64:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})

        body = {
            "model": cfg.model or "gpt-4o",
            "messages": [],
            "max_tokens": int(cfg.max_tokens or 120),
            "temperature": float(cfg.temperature),
        }
        if cfg.system_prompt:
            body["messages"].append(
                {"role": "system", "content": [{"type": "text", "text": cfg.system_prompt}]}
            )
        body["messages"].append({"role": "user", "content": content})
        return self._post_chat_completion(
            api_url=api_url,
            headers=headers,
            body=body,
            retries=cfg.retry,
            timeout=cfg.timeout,
        )

    def _call_openai_compatible(
        self,
        path: Path,
        stage_cfg: CaptionStageConfig,
        user_text: str,
        include_image: bool,
    ) -> str | None:
        def _normalize_chat_completions_url(url_or_base: str) -> str:
            candidate = url_or_base.rstrip("/")
            if candidate.endswith("/chat/completions"):
                return candidate
            if candidate.endswith("/v1"):
                return candidate + "/chat/completions"
            return candidate + "/v1/chat/completions"

        base = os.getenv("QWEN3_BASE_URL", "http://127.0.0.1:8000")
        api_url = _normalize_chat_completions_url(self._clean_placeholder(stage_cfg.api_url) or base)
        headers = {"Content-Type": "application/json"}
        api_key = self._clean_placeholder(stage_cfg.api_key) or os.getenv("QWEN3_API_KEY")
        if api_key:
            headers[stage_cfg.api_key_header or "Authorization"] = f"Bearer {api_key}"

        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        if include_image:
            images_b64 = self._extract_images_b64(path, stage_cfg)
            for image_b64 in images_b64:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})

        body: Dict[str, Any] = {
            "model": stage_cfg.model or "Qwen/Qwen3-32B-Instruct",
            "messages": [],
            "max_tokens": int(stage_cfg.max_tokens or 120),
            "temperature": float(stage_cfg.temperature),
        }
        if stage_cfg.system_prompt:
            body["messages"].append({"role": "system", "content": stage_cfg.system_prompt})
        body["messages"].append({"role": "user", "content": content})
        return self._post_chat_completion(
            api_url=api_url,
            headers=headers,
            body=body,
            retries=stage_cfg.retry,
            timeout=stage_cfg.timeout,
        )

    def _call_qwen3_local(
        self,
        path: Path,
        stage_cfg: CaptionStageConfig,
        user_text: str,
        include_image: bool,
    ) -> str | None:
        model_path = stage_cfg.local_model_path or os.getenv("QWEN3_LOCAL_MODEL_PATH")
        if not model_path:
            raise ValueError(
                "qwen3_local requires caption.local_model_path or environment variable QWEN3_LOCAL_MODEL_PATH"
            )

        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise ImportError("qwen3_local requires transformers and torch to be installed") from exc

        tokenizer, model, device = self._load_qwen3_local(
            model_path=model_path,
            device=stage_cfg.local_device or "cuda:0",
            dtype=stage_cfg.local_dtype or "bfloat16",
            trust_remote_code=bool(stage_cfg.local_trust_remote_code),
            auto_tokenizer=AutoTokenizer,
            auto_model=AutoModelForCausalLM,
            torch_mod=torch,
        )

        prompt = user_text
        if include_image:
            # Keep frame extraction in flow for future multimodal swap-in.
            image_count = len(self._extract_images_b64(path, stage_cfg))
            prompt = f"{prompt}\n已提取视频关键帧数量: {image_count}"

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        max_new_tokens = max(int(stage_cfg.local_max_new_tokens or 128), 1)
        gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
        temperature = float(stage_cfg.temperature or 0.0)
        if temperature > 0:
            gen_kwargs.update({"do_sample": True, "temperature": temperature})
        else:
            gen_kwargs.update({"do_sample": False})

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        input_len = int(inputs["input_ids"].shape[-1])
        new_tokens = output_ids[0][input_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return text or None

    def _load_qwen3_local(
        self,
        model_path: str,
        device: str,
        dtype: str,
        trust_remote_code: bool,
        auto_tokenizer,
        auto_model,
        torch_mod,
    ) -> tuple[Any, Any, str]:
        cache_key = (model_path, device, dtype, trust_remote_code)
        if cache_key in self._qwen_local_cache:
            return self._qwen_local_cache[cache_key]

        torch_dtype = self._resolve_qwen_dtype(dtype, torch_mod)
        tokenizer = auto_tokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        model = auto_model.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        model = model.to(device).eval()
        loaded = (tokenizer, model, device)
        self._qwen_local_cache[cache_key] = loaded
        return loaded

    @staticmethod
    def _resolve_qwen_dtype(dtype: str, torch_mod):
        normalized = str(dtype or "").strip().lower()
        mapping = {
            "bf16": "bfloat16",
            "bfloat16": "bfloat16",
            "fp16": "float16",
            "float16": "float16",
            "fp32": "float32",
            "float32": "float32",
        }
        return getattr(torch_mod, mapping.get(normalized, "bfloat16"))

    def _post_chat_completion(
        self,
        api_url: str,
        headers: Dict[str, str],
        body: Dict[str, Any],
        retries: int | None,
        timeout: float | None,
    ) -> str | None:
        tries = max(int(retries or 0), 0) + 1
        last_exc: Exception | None = None
        for attempt in range(tries):
            try:
                resp = requests.post(api_url, json=body, headers=headers, timeout=timeout)
                if resp.status_code >= 500 and attempt < tries - 1:
                    time.sleep(1.0)
                    continue
                resp.raise_for_status()
                return self._parse_chat_completion(resp)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < tries - 1:
                    time.sleep(1.0)
        if last_exc:
            raise last_exc
        return None

    def _parse_chat_completion(self, resp: requests.Response) -> str | None:
        data = resp.json()
        try:
            choices = data.get("choices") if isinstance(data, dict) else None
            if not choices:
                return None
            msg = choices[0].get("message") if isinstance(choices[0], dict) else None
            if not isinstance(msg, dict):
                return None
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = [
                    c["text"]
                    for c in content
                    if isinstance(c, dict) and c.get("type") == "text" and c.get("text")
                ]
                if texts:
                    return "\n".join(texts)
            return None
        except Exception:  # noqa: BLE001
            return None

    def _build_text_prompt(self, path: Path, stage_cfg: CaptionStageConfig) -> str:
        base_prompt = stage_cfg.user_prompt or "请为视频生成简洁描述。"
        return f"{base_prompt}\n文件名: {path.name}"

    def _extract_images_b64(self, path: Path, stage_cfg: CaptionStageConfig) -> list[str]:
        if decord is None or Image is None:
            return []
        try:
            vr = decord.VideoReader(str(path))
            total = len(vr)
            if total == 0:
                return []

            frame_count = max(int(stage_cfg.image_num_frames or 1), 1)
            if frame_count == 1:
                idxs = [total // 2]
            else:
                span = max(total - 1, 1)
                idxs = [
                    min(int(round(i * span / (frame_count - 1))), total - 1) for i in range(frame_count)
                ]
            idxs = sorted(set(idxs))

            images: list[str] = []
            for idx in idxs:
                frame = vr[idx]
                frame_np = frame.asnumpy() if hasattr(frame, "asnumpy") else frame
                encoded = self._encode_image_b64(frame_np, max_side=stage_cfg.image_max_side)
                if encoded:
                    images.append(encoded)
            return images
        except Exception as exc:  # noqa: BLE001
            log.debug("Failed to extract frames for %s: %s", path, exc)
            return []

    def _encode_image_b64(self, frame_np, max_side: int | None = None) -> str | None:
        if Image is None:
            return None
        try:
            img = Image.fromarray(frame_np).convert("RGB")
            max_side_val = max(int(max_side or 512), 64)
            w, h = img.size
            if max(w, h) > max_side_val:
                if w >= h:
                    new_w = max_side_val
                    new_h = int(h * max_side_val / w)
                else:
                    new_h = max_side_val
                    new_w = int(w * max_side_val / h)
                img = img.resize((max(new_w, 1), max(new_h, 1)))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as exc:  # noqa: BLE001
            log.debug("Failed to encode frame: %s", exc)
            return None

    @staticmethod
    def _clean_placeholder(value: str | None) -> str | None:
        if not value:
            return None
        if "${" in value:
            return None
        return value
