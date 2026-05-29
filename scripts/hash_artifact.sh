#!/usr/bin/env bash
# hash_artifact.sh — sha256 a file or directory and append a MANIFEST row.
#
# Usage:
#   scripts/hash_artifact.sh <path> [step_id] [schema] [n_rows]
#
# Behavior:
#   - For a file, prints "<path>,<sha256>" to stdout.
#   - If additional args are given, also appends one row to docs/finetune/data/MANIFEST.csv
#     with columns: path,sha256,n_rows,schema,source_step,ts,git_sha,dm_pkl_sha256
#   - For a directory, walks files (sorted) and sha256s their concatenation via find+sort+xargs cat.

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <path> [step_id] [schema] [n_rows]" >&2
  exit 2
fi

PATH_ARG="$1"
STEP_ID="${2:-}"
SCHEMA="${3:-}"
N_ROWS="${4:-}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="$REPO_ROOT/docs/finetune/data/MANIFEST.csv"
DM_PKL="$REPO_ROOT/limo/dm.pkl"

if [ ! -e "$PATH_ARG" ]; then
  echo "hash_artifact: not found: $PATH_ARG" >&2
  exit 1
fi

if [ -d "$PATH_ARG" ]; then
  SHA=$(find "$PATH_ARG" -type f -print0 | LC_ALL=C sort -z | xargs -0 cat 2>/dev/null | shasum -a 256 | cut -d' ' -f1)
else
  SHA=$(shasum -a 256 "$PATH_ARG" | cut -d' ' -f1)
fi

printf '%s,%s\n' "$PATH_ARG" "$SHA"

if [ -n "$STEP_ID" ]; then
  TS="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  # Dual SHA: outer repo (docs/configs/scripts) + inner limo/ repo (LIMO code) since
  # limo/ is itself a git repo (upstream Rose-STL-Lab/LIMO fork). Format: outer+inner.
  OUTER_SHA="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo 'no-git')"
  INNER_SHA="$(git -C "$REPO_ROOT/limo" rev-parse --short HEAD 2>/dev/null || echo 'no-inner-git')"
  GIT_SHA="${OUTER_SHA}+${INNER_SHA}"
  DM_SHA=""
  if [ -f "$DM_PKL" ]; then
    DM_SHA=$(shasum -a 256 "$DM_PKL" | cut -d' ' -f1)
  fi
  # MANIFEST.csv columns: path,sha256,n_rows,schema,source_step,ts,git_sha,dm_pkl_sha256
  # git_sha format: <outer_short_sha>+<inner_short_sha>
  printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$PATH_ARG" "$SHA" "$N_ROWS" "$SCHEMA" "$STEP_ID" "$TS" "$GIT_SHA" "$DM_SHA" >> "$MANIFEST"
fi
