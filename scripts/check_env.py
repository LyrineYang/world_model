#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import sys
from typing import Tuple


DEFAULT_MODULES = ["torch", "decord", "PIL", "requests", "yaml"]


def _check_module(name: str) -> Tuple[bool, str]:
    try:
        module = importlib.import_module(name)
        ver = getattr(module, "__version__", "unknown")
        return True, f"{name}: ok ({ver})"
    except Exception as exc:  # noqa: BLE001
        return False, f"{name}: missing ({exc})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Environment preflight check")
    parser.add_argument(
        "--require",
        type=str,
        default=",".join(DEFAULT_MODULES),
        help="Comma-separated required modules",
    )
    parser.add_argument(
        "--check-cuda",
        action="store_true",
        help="Also check CUDA availability when torch is installed",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    required = [m.strip() for m in args.require.split(",") if m.strip()]
    failures = 0
    for module_name in required:
        ok, msg = _check_module(module_name)
        print(msg)
        failures += 0 if ok else 1

    if args.check_cuda:
        try:
            import torch

            print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"torch.cuda.device_count: {torch.cuda.device_count()}")
        except Exception as exc:  # noqa: BLE001
            print(f"torch cuda check failed: {exc}")
            failures += 1

    if failures:
        print(f"env check failed: {failures} requirement(s) missing")
        return 1
    print("env check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
