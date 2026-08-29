#!/usr/bin/env bash
#
# bootstrap_beeline.sh — clone and initialize a Murali-group/Beeline checkout
# so iqcell.beeline can drive the real GRN-inference benchmarking pipeline.
#
# BEELINE runs its algorithms inside Docker containers, so Docker must be
# installed and running. This script clones the repo, pulls the algorithm
# images, and prints the path to pass to iqcell's BeelineRunner / the
# examples/beeline_benchmark.py --beeline-repo flag.
#
# Usage:
#   scripts/bootstrap_beeline.sh [target_dir]
#
# Environment:
#   BEELINE_GIT   Override the clone URL (default: upstream Murali-group/Beeline).
#
set -euo pipefail

TARGET_DIR="${1:-$(pwd)/.beeline}"
BEELINE_GIT="${BEELINE_GIT:-https://github.com/Murali-group/Beeline.git}"

log() { printf '\033[1;34m[bootstrap-beeline]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[bootstrap-beeline]\033[0m %s\n' "$*" >&2; }

# --- Preconditions ---------------------------------------------------------
if ! command -v git >/dev/null 2>&1; then
  err "git is required but not found on PATH."
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  err "docker is required but not found on PATH."
  err "BEELINE runs its inference algorithms in Docker containers."
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  err "Docker is installed but not running (or lacks permissions)."
  err "Start the Docker daemon and re-run this script."
  exit 1
fi

# --- Clone -----------------------------------------------------------------
if [ -d "${TARGET_DIR}/.git" ]; then
  log "Beeline checkout already exists at ${TARGET_DIR}; pulling latest."
  git -C "${TARGET_DIR}" pull --ff-only
else
  log "Cloning Beeline into ${TARGET_DIR}"
  git clone --depth 1 "${BEELINE_GIT}" "${TARGET_DIR}"
fi

# --- Pull algorithm Docker images ------------------------------------------
if [ -f "${TARGET_DIR}/utils/initialize.sh" ]; then
  log "Pulling BEELINE algorithm Docker images (this can take a while)..."
  ( cd "${TARGET_DIR}" && bash utils/initialize.sh )
else
  err "utils/initialize.sh not found in the checkout; skipping image pull."
  err "You may need to pull grnbeeline/* images manually."
fi

log "Done. BEELINE is ready at:"
printf '  %s\n' "${TARGET_DIR}"
log "Run the full synthetic -> infer -> evaluate loop with:"
printf '  python examples/beeline_benchmark.py --beeline-repo "%s"\n' "${TARGET_DIR}"
log "Or in Python:  BeelineRunner(beeline_repo=\"${TARGET_DIR}\")"
log "Note: BEELINE's entry points expect its conda env active"
log "  (source ~/miniconda3/etc/profile.d/conda.sh && conda activate BEELINE)"
log "  or pass python_exe=... to BeelineRunner."
