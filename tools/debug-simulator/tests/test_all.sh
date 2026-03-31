#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while IFS= read -r test_dir; do
  test_name="$(basename "$test_dir")"

  echo "==> Running $test_name"
  (
    cd "$test_dir"
    make clean
    make build
    make run
  )
done < <(find "$SCRIPT_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
