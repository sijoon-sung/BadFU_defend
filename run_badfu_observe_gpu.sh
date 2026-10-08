#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Shadow measurements reuse the submitted local models and each actual FU path.
exec "${PYTHON:-python}" -u -m experiments.request_purify \
  --dataset badfu --device cuda:0 --observe --probe-size 64 \
  --arms none detected oracle --post-rounds 0 "$@"
