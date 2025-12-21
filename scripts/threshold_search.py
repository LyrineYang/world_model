#!/usr/bin/env python3
"""
离线阈值网格搜索示例：
1) 先用极低阈值跑一遍验证集，保证 metadata.jsonl 中都有 scores（不要早停或过严阈值）。
2) 将人工标注的 keep 标签合并到 DataFrame（示例假设 metadata.jsonl 中已有 keep_gt 列；若标注在独立文件，按需 merge）。
3) 配置下方的候选阈值，脚本会枚举组合，计算精度/召回，并输出 CSV 便于筛选。
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid search thresholds on metadata.jsonl")
    p.add_argument("--meta", type=Path, required=True, help="Path to metadata.jsonl")
    p.add_argument("--out", type=Path, default=Path("threshold_search.csv"), help="Output CSV path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_json(args.meta, lines=True)

    if "keep_gt" not in df.columns:
        raise SystemExit("metadata.jsonl 缺少人工标注列 keep_gt，请先合并标注")

    # 阈值候选配置：按需调整，组合数 = len(dover)*len(aes)*len(motion)
    candidates = {
        "dover": [0.3, 0.4, 0.5, 0.6],
        "aes": [4.0, 4.5, 5.0, 5.5],
        "motion": [30, 60, 90],
    }

    rows = []
    for d, a, m in itertools.product(candidates["dover"], candidates["aes"], candidates["motion"]):
        pred = df["scores"].apply(
            lambda s: s.get("dover", -1) >= d and s.get("aes", -1) >= a and s.get("motion", -1) >= m
        )
        tp = ((pred == True) & (df["keep_gt"] == True)).sum()
        fp = ((pred == True) & (df["keep_gt"] == False)).sum()
        fn = ((pred == False) & (df["keep_gt"] == True)).sum()
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        rows.append({"dover": d, "aes": a, "motion": m, "tp": tp, "fp": fp, "fn": fn, "prec": prec, "rec": rec})

    out_df = pd.DataFrame(rows).sort_values(["rec", "prec"], ascending=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Saved {len(out_df)} combinations to {args.out}")
    print(out_df.head(10))


if __name__ == "__main__":
    main()
