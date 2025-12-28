#!/usr/bin/env bash
# 简单统计：总分片、已打分、已上传、未打分/未上传列表（前20）
set -euo pipefail

STATE_DIR="${STATE_DIR:-/scratch/ayuille1/jchen293/wcloong/video_dataset/state}"
SHARDS_FILE="${SHARDS_FILE:-shards_ONE-Lab_HQ-video-data.txt}"

TMP=".analysis_tmp"
trap 'rm -rf "$TMP"' EXIT
rm -rf "$TMP" && mkdir -p "$TMP"

if [ ! -f "$SHARDS_FILE" ]; then
  echo "shards file not found: $SHARDS_FILE" >&2
  exit 1
fi
if [ ! -d "$STATE_DIR" ]; then
  echo "state dir not found: $STATE_DIR" >&2
  exit 1
fi

# 1) 基础列表
grep -v '^#' "$SHARDS_FILE" | sed '/^$/d' | sort -u > "$TMP/shards.txt"

# 2) 状态提取
find "$STATE_DIR" -name '*.json' -print0 \
  | xargs -0 -I{} sh -c 'basename "{}" .json' \
  | sort -u > "$TMP/states_all.txt"

find "$STATE_DIR" -name '*.json' -print0 \
  | xargs -0 -I{} sh -c 'jq -e ".scored==true" "{}" >/dev/null && basename "{}" .json' \
  | sort -u > "$TMP/states_scored.txt"

find "$STATE_DIR" -name '*.json' -print0 \
  | xargs -0 -I{} sh -c 'jq -e ".uploaded==true" "{}" >/dev/null && basename "{}" .json' \
  | sort -u > "$TMP/states_uploaded.txt"

# 3) 交叉统计
TOTAL=$(wc -l < "$TMP/shards.txt")
SC=$(comm -12 "$TMP/shards.txt" "$TMP/states_scored.txt" | wc -l)
UP=$(comm -12 "$TMP/shards.txt" "$TMP/states_uploaded.txt" | wc -l)
MISS=$(comm -23 "$TMP/shards.txt" "$TMP/states_all.txt" | wc -l)
NS=$(comm -23 "$TMP/shards.txt" "$TMP/states_scored.txt" | wc -l)
NU=$(comm -23 "$TMP/shards.txt" "$TMP/states_uploaded.txt" | wc -l)

echo "总分片: $TOTAL"
echo "已打分: $SC"
echo "已上传: $UP"
echo "未打分: $NS"
echo "未上传: $NU"
echo "缺少 state 文件: $MISS"

echo "未打分(前20):"
comm -23 "$TMP/shards.txt" "$TMP/states_scored.txt" | head -20

echo "未上传(前20):"
comm -23 "$TMP/shards.txt" "$TMP/states_uploaded.txt" | head -20
