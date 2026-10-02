#!/usr/bin/env bash
# End-of-work maintenance for the lead_v1 <-> lead_orin pair.
#
# Run it from either repo. It figures out which repo it is from the origin URL,
# runs the checks, commits everything and pushes. In the host repo (lead_v1) it
# also syncs the shared code into the Orin repo and pushes that, so the two stay
# in step.
#
# Ownership (see ROS软件在环.md): lead_v1 owns the shared code (src/lead, the SIL
# contract/transport, the bridge, tools, docs); lead_orin owns sil/orin and its
# README. Editing shared code on the Orin side is overwritten by the next sync.
#
# Usage:
#   bash sil/maintain_repo.sh "commit message"
#
# Options (env):
#   ORIN_REPO=/path/to/lead_orin   where to sync from the host (default ../lead_orin)
#   SKIP_CHECKS=1                  skip ruff / tests
#   RUN_TESTS=1                    also run the evaluation unit tests
#   INCLUDE_3RDPARTY=1             also commit changes under 3rd_party/ (default off)
set -euo pipefail

cd "$(cd "$(dirname "$(realpath "${BASH_SOURCE:-$0}")")/.." && pwd)"
MSG="${1:?usage: maintain_repo.sh \"commit message\"}"

ORIGIN="$(git remote get-url origin 2>/dev/null || true)"
if [[ "$ORIGIN" == *lead_orin* ]]; then
  ROLE="orin"
else
  ROLE="host"
fi
echo "[maintain] repo=$PWD role=$ROLE origin=$ORIGIN"

# --- checks ------------------------------------------------------------------
if [ "${SKIP_CHECKS:-0}" != "1" ]; then
  if command -v ruff >/dev/null 2>&1; then
    ruff check src/lead sil
  fi
  if [ "${RUN_TESTS:-0}" = "1" ]; then
    python -m pytest tests/unittests/evaluation -q
  fi
fi

# --- commit + push this repo -------------------------------------------------
commit_and_push() {
  if [ -z "$(git status --porcelain)" ]; then
    echo "[maintain] nothing to commit in $PWD"
    return 0
  fi
  git status --short
  git add -A
  if [ "${INCLUDE_3RDPARTY:-0}" != "1" ]; then
    # 3rd_party/ is vendored and often carries unrelated local edits; keep it out
    # of "our code" commits unless asked.
    git reset -q -- 3rd_party 2>/dev/null || true
  fi
  if [ -z "$(git diff --cached --name-only)" ]; then
    echo "[maintain] nothing staged after exclusions"
    return 0
  fi
  git commit -m "$1"
  git push
}

commit_and_push "$MSG"

# --- host repo also syncs and pushes the Orin repo ---------------------------
if [ "$ROLE" = "host" ]; then
  ORIN_REPO="${ORIN_REPO:-$(cd .. && pwd)/lead_orin}"
  if [ -d "$ORIN_REPO/.git" ]; then
    # Bring the staging clone up to date first: the Orin side commits its own
    # sil/orin work, so a stale clone would make the sync push non-fast-forward.
    echo "[maintain] updating Orin repo at $ORIN_REPO"
    git -C "$ORIN_REPO" pull --ff-only
    echo "[maintain] syncing shared code into the Orin repo"
    bash scripts/common/sync_orin_repo.sh "$ORIN_REPO"
    ( cd "$ORIN_REPO" && commit_and_push "sync: $MSG" )
  else
    echo "[maintain] Orin repo not found at $ORIN_REPO; skipping sync"
  fi
fi
