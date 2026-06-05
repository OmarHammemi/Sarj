#!/usr/bin/env bash
# Generate 5 submission samples after training.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p samples
i=1
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" ]] && continue
  out=$(printf "samples/%02d.wav" "$i")
  python synthesize.py --text "$line" --out "$out" --device cpu
  i=$((i + 1))
done < samples/prompts.txt

echo "Generated $((i - 1)) samples in samples/"
