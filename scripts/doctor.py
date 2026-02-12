#!/usr/bin/env python3

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "third_party" / "manifest.lock.yaml"


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(text) or {}
        repos = payload.get("repos", [])
        if isinstance(repos, list):
            return [r for r in repos if isinstance(r, dict)]
    except Exception:
        pass

    repos: list[dict[str, Any]] = []
    cur: dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("repos:"):
            continue
        if line.startswith("- "):
            if cur:
                repos.append(cur)
            cur = {}
            line = line[2:]
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if value.lower() in {"true", "false"}:
            cur[key] = value.lower() == "true"
        else:
            cur[key] = value
    if cur:
        repos.append(cur)
    return repos


def _check_command(cmd: str) -> CheckResult:
    path = shutil.which(cmd)
    return CheckResult(name=f"command:{cmd}", ok=path is not None, detail=path or "not found")


def _check_import(module: str) -> CheckResult:
    exists = importlib.util.find_spec(module) is not None
    return CheckResult(name=f"python:{module}", ok=exists, detail="available" if exists else "missing")


def _check_torch_cuda() -> CheckResult:
    if importlib.util.find_spec("torch") is None:
        return CheckResult(name="torch_cuda", ok=False, detail="torch missing")
    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available())
        detail = f"cuda_available={cuda_ok}"
        if cuda_ok:
            detail += f", devices={torch.cuda.device_count()}"
        return CheckResult(name="torch_cuda", ok=cuda_ok, detail=detail)
    except Exception as exc:  # noqa: BLE001
        return CheckResult(name="torch_cuda", ok=False, detail=f"torch import failed: {exc}")


def _check_path(path: Path, name: str) -> CheckResult:
    return CheckResult(name=name, ok=path.exists(), detail=str(path))


def _check_third_party(manifest_path: Path) -> list[CheckResult]:
    out: list[CheckResult] = []
    repos = _load_manifest(manifest_path)
    if not repos:
        out.append(CheckResult(name="third_party_manifest", ok=False, detail=f"missing/empty: {manifest_path}"))
        return out

    out.append(CheckResult(name="third_party_manifest", ok=True, detail=str(manifest_path)))
    for repo in repos:
        if not repo.get("enabled", True):
            continue
        name = str(repo.get("name", "unknown"))
        rel_path = str(repo.get("path", ""))
        repo_path = (REPO_ROOT / rel_path).resolve()
        ok = (repo_path / ".git").exists()
        detail = str(repo_path)
        commit = str(repo.get("commit", "")).strip()
        if ok and commit:
            try:
                current = (
                    subprocess.check_output(["git", "-C", str(repo_path), "rev-parse", "HEAD"], text=True)
                    .strip()
                )
                if current != commit:
                    ok = False
                    detail += f" (HEAD={current}, expected={commit})"
                else:
                    detail += f" (HEAD={current})"
            except Exception as exc:  # noqa: BLE001
                ok = False
                detail += f" (rev-parse failed: {exc})"
        out.append(CheckResult(name=f"third_party:{name}", ok=ok, detail=detail))
    return out


def _run_filter_checks(manifest_path: Path) -> list[CheckResult]:
    checks = [
        _check_command("ffmpeg"),
        _check_import("torch"),
        _check_torch_cuda(),
        _check_import("decord"),
    ]
    checks.extend(_check_third_party(manifest_path))

    checks.extend(
        [
            _check_path(REPO_ROOT / "third_party" / "DOVER" / "pretrained_weights" / "DOVER.pth", "weight:dover"),
            _check_path(
                REPO_ROOT
                / "third_party"
                / "unimatch"
                / "pretrained"
                / "gmflow-scale1-mixdata-train320x576-4c3a6e9a.pth",
                "weight:unimatch",
            ),
        ]
    )
    return checks


def _run_caption_checks(model_path: str | None) -> list[CheckResult]:
    checks = [
        _check_import("transformers"),
        _check_import("decord"),
        _check_import("torch"),
    ]

    resolved = model_path or os.environ.get("QWEN3_LOCAL_MODEL_PATH")
    if resolved:
        checks.append(_check_path(Path(resolved).expanduser(), "qwen3_model_path"))
    else:
        checks.append(CheckResult(name="qwen3_model_path", ok=False, detail="not provided (use --model-path or QWEN3_LOCAL_MODEL_PATH)"))
    return checks


def _print_human(mode: str, checks: list[CheckResult]) -> None:
    print(f"[doctor] mode={mode}")
    for c in checks:
        status = "OK" if c.ok else "FAIL"
        print(f"- {status:4} {c.name}: {c.detail}")
    passed = sum(1 for c in checks if c.ok)
    print(f"[doctor] summary: {passed}/{len(checks)} checks passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Environment doctor for world_model pipelines")
    parser.add_argument("mode", choices=["filter", "caption"], help="Which pipeline environment to validate")
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST), help="Path to third-party manifest")
    parser.add_argument("--model-path", type=str, default=None, help="Qwen3 local model path for caption checks")
    parser.add_argument("--json", action="store_true", help="Print JSON payload only")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    checks = _run_filter_checks(manifest_path) if args.mode == "filter" else _run_caption_checks(args.model_path)

    payload = {
        "mode": args.mode,
        "ok": all(c.ok for c in checks),
        "checks": [asdict(c) for c in checks],
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_human(args.mode, checks)
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    if not payload["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
