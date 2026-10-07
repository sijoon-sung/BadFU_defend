#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# PYTHON may point to the GPU machine's existing virtual environment.
exec "${PYTHON:-python}" -u -m experiments.request_purify --dataset badfu --device cuda:0 "$@"
