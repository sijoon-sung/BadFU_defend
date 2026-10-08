#!/bin/bash
# 삭제 전후 차이로 좁힌 재검사 실험 (diff_audit/README.md)
#
# 사용법 (저장된 FUBA 궤적이 있어야 함: run_experiment.sh 또는 run_real_fuba.sh 로 생성):
#   bash run_diff_audit.sh det1 det2 det3                 # 기본 역추적(universal)
#   INV=per_sample bash run_diff_audit.sh det1 det2 det3
#   INV=patch STEPS=300 bash run_diff_audit.sh det1 det2 det3
set -e
cd "$(dirname "$0")"
[ $# -ge 1 ] || { echo "usage: bash run_diff_audit.sh NAME [NAME ...]"; exit 1; }
INV="${INV:-universal}"
STEPS="${STEPS:-100}"
for NAME in "$@"; do
  ls checkpoints/local_net_model_${NAME}_round_*.pkl >/dev/null 2>&1 || { echo "궤적 없음: $NAME"; exit 1; }
  echo "== [$NAME] diff_audit ($INV)"
  PYTHONIOENCODING=utf-8 python -m diff_audit.run --name "$NAME" --inv "$INV" --steps "$STEPS" | grep -v "^Files already"
done
echo "== 집계"
PYTHONIOENCODING=utf-8 python -m diff_audit.aggregate --names "$@" --inv "$INV"
