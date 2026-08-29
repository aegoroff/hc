#!/usr/bin/env bash
# Compare performance of two hc builds with poop.
#
# Usage:
#   ./scripts/bench_hashes_poop.sh <base_dir> [new_subdir] [old_subdir]
#
# Example:
#   ./scripts/bench_hashes_poop.sh /home/user/hc_x86_64 gnu old
#
# Expects:
#   <base_dir>/<new_subdir>/hc
#   <base_dir>/<old_subdir>/hc
#
# Options (env):
#   DURATION_MS   poop --duration (default: 5000)
#   BINARY        binary name under each subdir (default: hc)
#   HASHES        space-separated hash list (default: intersection of both builds)
#   BENCH_ARGS    args after "<bin> <hash>" (default: hash -p --noprobe)
#   OUT_DIR       write per-hash logs here (default: unset = stdout only)
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: bench_hashes_poop.sh <base_dir> [new_subdir] [old_subdir]

  base_dir     directory that contains version subdirectories
  new_subdir   subdirectory with the new binary (default: gnu)
  old_subdir   subdirectory with the old binary (default: old)

Env:
  DURATION_MS  poop sample duration in ms (default: 5000)
  BINARY       executable name (default: hc)
  HASHES       space-separated algorithms (default: common to both builds)
  BENCH_ARGS   args after "<bin> <hash>" (default: hash -p --noprobe)
  OUT_DIR      if set, also write each hash result to OUT_DIR/<hash>.txt
EOF
  exit 1
fi

BASE_DIR=$(realpath "$1")
NEW_SUB=${2:-gnu}
OLD_SUB=${3:-old}
BINARY=${BINARY:-hc}
DURATION_MS=${DURATION_MS:-5000}
BENCH_ARGS=${BENCH_ARGS:-hash -p --noprobe}

NEW_BIN="${BASE_DIR}/${NEW_SUB}/${BINARY}"
OLD_BIN="${BASE_DIR}/${OLD_SUB}/${BINARY}"

for bin in "${NEW_BIN}" "${OLD_BIN}"; do
  if [[ ! -x "${bin}" ]]; then
    echo "error: executable not found: ${bin}" >&2
    exit 1
  fi
done

if ! command -v poop >/dev/null 2>&1; then
  echo "error: poop not found in PATH" >&2
  exit 1
fi

list_hashes() {
  local bin=$1
  # Algorithm names from `hc --help`:
  #   old (5.x): indented name alone ("    md5")
  #   new (6.x / yazap): indented name, then 2+ spaces, then a description
  # Require 2+ spaces before a description so wrapped help lines
  # ("                            console output.") are not treated as names.
  "${bin}" --help 2>&1 \
    | awk '
      /^[[:space:]]+[a-z][a-z0-9-]*($|[[:space:]]{2,})/ {
        name = $1
        if (name != "default" && name != "help") print name
      }' \
    | sort -u
}

mapfile -t NEW_HASHES < <(list_hashes "${NEW_BIN}")
mapfile -t OLD_HASHES < <(list_hashes "${OLD_BIN}")

if [[ -n "${HASHES:-}" ]]; then
  # shellcheck disable=SC2206
  SELECTED=(${HASHES})
else
  mapfile -t SELECTED < <(comm -12 \
    <(printf '%s\n' "${NEW_HASHES[@]}") \
    <(printf '%s\n' "${OLD_HASHES[@]}"))
fi

if [[ ${#SELECTED[@]} -eq 0 ]]; then
  echo "error: no hashes to benchmark" >&2
  if [[ ${#NEW_HASHES[@]} -eq 0 ]]; then
    echo "error: could not parse algorithm list from: ${NEW_BIN} --help" >&2
  fi
  if [[ ${#OLD_HASHES[@]} -eq 0 ]]; then
    echo "error: could not parse algorithm list from: ${OLD_BIN} --help" >&2
  fi
  exit 1
fi

ONLY_NEW=$(comm -23 \
  <(printf '%s\n' "${NEW_HASHES[@]}") \
  <(printf '%s\n' "${OLD_HASHES[@]}") || true)
ONLY_OLD=$(comm -13 \
  <(printf '%s\n' "${NEW_HASHES[@]}") \
  <(printf '%s\n' "${OLD_HASHES[@]}") || true)

echo "=== hc benchmark (poop) ==="
echo "new: ${NEW_BIN}"
echo "old: ${OLD_BIN}"
echo "duration: ${DURATION_MS} ms"
echo "bench args: ${BENCH_ARGS}"
echo "hashes: ${#SELECTED[@]}"
if [[ -n "${ONLY_NEW}" ]]; then
  echo "skipped (only in new): ${ONLY_NEW//$'\n'/ }"
fi
if [[ -n "${ONLY_OLD}" ]]; then
  echo "skipped (only in old): ${ONLY_OLD//$'\n'/ }"
fi
echo

if [[ -n "${OUT_DIR:-}" ]]; then
  mkdir -p "${OUT_DIR}"
fi

# Build argv for BENCH_ARGS safely (empty -> no extra words).
# shellcheck disable=SC2206
BENCH_ARR=(${BENCH_ARGS})

failed=0
for hash in "${SELECTED[@]}"; do
  echo "------------------------------------------------------------"
  echo "### ${hash}"
  echo "------------------------------------------------------------"

  new_cmd=("${NEW_BIN}" "${hash}" "${BENCH_ARR[@]}")
  old_cmd=("${OLD_BIN}" "${hash}" "${BENCH_ARR[@]}")

  # poop expects each command as a single shell-quoted string argument.
  new_str=$(printf '%q ' "${new_cmd[@]}")
  old_str=$(printf '%q ' "${old_cmd[@]}")
  new_str=${new_str%% }
  old_str=${old_str%% }

  if [[ -n "${OUT_DIR:-}" ]]; then
    out_file="${OUT_DIR}/${hash}.txt"
    if poop --duration "${DURATION_MS}" "${new_str}" "${old_str}" | tee "${out_file}"; then
      :
    else
      echo "warning: poop failed for ${hash}" >&2
      failed=$((failed + 1))
    fi
  else
    if poop --duration "${DURATION_MS}" "${new_str}" "${old_str}"; then
      :
    else
      echo "warning: poop failed for ${hash}" >&2
      failed=$((failed + 1))
    fi
  fi
  echo
done

echo "=== done: ${#SELECTED[@]} hashes, ${failed} failures ==="
exit $((failed > 0 ? 1 : 0))
