#!/bin/bash
# Keith Briggs 2025-09-24

for f in CRRM_example*.py; do
  echo "running ${f}..."
  rm -f ani/*.png
  bash use_most_recent_python.sh "${f}"
  if [ $? -ne 0 ]; then
    echo "example ${f} failed, quitting 🥵"
    exit 1
  fi
done

echo "All CRRM examples successful! ✨ 🍰 ✨"
