#!/bin/bash
# 논문 실험 일괄 실행: 공격 궤적 생성 -> 탐지 -> 정화 -> 집계
#
# 사용법:
#   bash run_experiment.sh det1 det2 det3        # 각 이름으로 1회씩, 끝나면 집계
#   SKIP_ATTACK=1 bash run_experiment.sh det1 det2   # 저장된 궤적이 있으면 공격 단계 생략
#
# 단계 (논문 절 대응)
#   [6.1] run_real_fuba.sh NAME        FUBA 공개 코드를 mpiexec 로 실행, 라운드별 모델 저장
#   [5.1][5.2][6.2][6.3] pipeline_detect_purify.py --name NAME
#                                      탐지(타깃/공격자/요청자) + 정화 변형 비교 -> logs/pipeline_NAME.json
#   [6.3] aggregate_pipeline.py        여러 실행 평균±표준편차 -> logs/pipeline_aggregate.json
set -e
cd "$(dirname "$0")"
[ $# -ge 1 ] || { echo "usage: bash run_experiment.sh NAME [NAME ...]"; exit 1; }
for NAME in "$@"; do
  if [ "${SKIP_ATTACK:-0}" = "1" ] && ls checkpoints/local_net_model_${NAME}_round_*.pkl >/dev/null 2>&1; then
    echo "== [$NAME] 저장된 궤적 사용 (공격 단계 생략)"
  else
    echo "== [$NAME] 1/2 공격 실행 (FUBA, mpiexec)"
    bash run_real_fuba.sh "$NAME" > "logs_fuba_${NAME}.txt" 2>&1
    n=$(ls checkpoints/local_net_model_${NAME}_round_*.pkl 2>/dev/null | wc -l)
    [ "$n" -ge 2 ] || { echo "   공격 실행 실패: 저장된 라운드 $n 개. logs_fuba_${NAME}.txt 확인"; exit 1; }
  fi
  echo "== [$NAME] 2/3 탐지 + 정화"
  PYTHONIOENCODING=utf-8 python pipeline_detect_purify.py --name "$NAME" | grep -v "^Files already"
  echo "== [$NAME] 3/3 기존 방어 비교"
  PYTHONIOENCODING=utf-8 python baselines_compare.py --name "$NAME" | grep -v "^Files already"
done
echo "== 집계"
PYTHONIOENCODING=utf-8 python aggregate_pipeline.py --names "$@"
