#!/bin/bash
# 여러 seed 로 공격→탐지→정화→기존방어비교 를 반복하고 평균±표준편차 집계 (통계용).
#
# 사용법:
#   bash run_experiment.sh 1 2 3 4 5         # seed 1..5 각각 독립 실행 후 집계
#   SKIP_ATTACK=1 bash run_experiment.sh 1 2 # 해당 seed 궤적이 있으면 공격 단계 생략
#
# seed s -> 실행 이름 s{s} (예: seed 1 -> s1). 각 seed 는 FUBA main.py --seed=s 로
# 데이터 분할·IBA 초기화·클라 선택까지 독립. 이게 통계의 독립 표본이 된다.
#
# 단계(논문 절): [6.1] run_real_fuba.sh NAME SEED  / [5,6.2,6.3] pipeline_detect_purify.py
#               [기존방어] baselines_compare.py    / [6.3] aggregate_pipeline.py
set -e
cd "$(dirname "$0")"
[ $# -ge 1 ] || { echo "usage: bash run_experiment.sh SEED [SEED ...]"; exit 1; }
NAMES=()
for SEED in "$@"; do
  NAME="s${SEED}"; NAMES+=("$NAME")
  if [ "${SKIP_ATTACK:-0}" = "1" ] && ls checkpoints/local_net_model_${NAME}_round_*.pkl >/dev/null 2>&1; then
    echo "== [seed $SEED] 저장된 궤적 사용 (공격 단계 생략)"
  else
    echo "== [seed $SEED] 1/3 공격 실행 (FUBA, mpiexec, seed=$SEED)"
    bash run_real_fuba.sh "$NAME" "$SEED" > "logs_fuba_${NAME}.txt" 2>&1
    n=$(ls checkpoints/local_net_model_${NAME}_round_*.pkl 2>/dev/null | wc -l)
    [ "$n" -ge 2 ] || { echo "   공격 실행 실패(저장 라운드 $n). logs_fuba_${NAME}.txt 확인"; exit 1; }
  fi
  echo "== [seed $SEED] 2/3 탐지 + 정화"
  PYTHONIOENCODING=utf-8 python pipeline_detect_purify.py --name "$NAME" | grep -v "^Files already"
  echo "== [seed $SEED] 3/3 기존 방어 비교"
  PYTHONIOENCODING=utf-8 python baselines_compare.py --name "$NAME" | grep -v "^Files already"
done
echo "== 집계 (${#NAMES[@]} seeds)"
PYTHONIOENCODING=utf-8 python aggregate_pipeline.py --names "${NAMES[@]}"
