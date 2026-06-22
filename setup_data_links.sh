#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Create the data symlinks this repository expects at its root, so that every
# analysis/plotting script can refer to the consolidated trajectory-analysis
# products through the in-repo relative path ./PYTHON_ANALYSIS (and ./SIMULATION),
# no matter where that data actually lives on your system.
#
# The scripts resolve <repo-root>/PYTHON_ANALYSIS from their own location
# (via __file__), so once these links exist the whole pipeline runs against
# ./-relative paths with no absolute paths baked into the code.
#
# Usage:
#   bash setup_data_links.sh                        # use the default DATA_ROOT below
#   DATA_ROOT=/path/to/your/data bash setup_data_links.sh
#
# DATA_ROOT must contain PYTHON_ANALYSIS/ (and optionally SIMULATION/) holding the
# per-temperature TEMP_<T>/{SG,DSM,NDSM}/ trees described in the README.
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Default points at the Princeton Stellar location used for the paper; override
# with the DATA_ROOT environment variable on any other machine.
DATA_ROOT="${DATA_ROOT:-/projects/WEBB/from_jay}"

link() {  # link <leaf-name> [required|optional]
  local leaf="$1" target="${DATA_ROOT}/$1"
  local required="${2:-optional}"
  if [[ ! -e "${target}" ]]; then
    if [[ "${required}" == "required" ]]; then
      echo "error ${leaf}  (required target not found: ${target})" >&2
      return 1
    fi
    echo "skip  ${leaf}  (optional target not found: ${target})"
    return
  fi
  ln -sfn "${target}" "${ROOT_DIR}/${leaf}"
  echo "link  ${leaf}  ->  ${target}"
}

echo "Creating data symlinks in ${ROOT_DIR} (DATA_ROOT=${DATA_ROOT})"
link PYTHON_ANALYSIS required  # required: per-frame analysis products (TEMP_<T>/{SG,DSM,NDSM}/...)
link SIMULATION optional       # optional: raw LAMMPS trajectories
link PYTHON_SIMULATIONS optional  # optional: alternate analysis source (skipped if absent)
echo "done."
