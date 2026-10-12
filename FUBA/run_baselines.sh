#!/bin/bash
# 기존 방법(비교군)만 자세히: 결과 브랜치 exp/baselines 용. 제안 기법은 돌리지 않는다.
#
#   bash run_baselines.sh                   # 전부: CONDS="iid owner", SEEDS="0 1 2"
#   GROUP="fu detect" bash run_baselines.sh # 일부 묶음만
#   TDEFS="median" bash run_baselines.sh    # 집계 방어 변형만 바꿈
#   DRY=1 bash run_baselines.sh
#
# 묶음 (GROUP)
#   fu      FU 연산자 4 종 (빼기, 증류, FedEraser, 재학습 정상/공격자 재참여)        benign_departure.py → logs/benign_departure
#   robust  FedEraser-R (이전 제안: 보정 중 탐지+정화)                                robust_fu.py        → logs/robust_fu
#   detect  MASA, SubID 식, 가중치 전용(잠재 위험 지도)                               baseline_*.py, latent_risk.py
#   tdef    학습 중 집계 방어 4 종 위에서 궤적 재생성 → FU 4 종 (깨어나는가)          run_real_fuba.sh --median 등
#   prev    이전 탐지 판: 빼 보기(release map), 진행형 FU                             release_map.py, progressive_fu.py
set -euo pipefail
cd "$(dirname "$0")"
GROUP="${GROUP:-fu robust detect tdef prev}"
CONDS="${CONDS:-iid owner}"
SEEDS="${SEEDS:-0 1 2}"
TDEFS="${TDEFS:-median multi_krum flame flDetector}"
DRY="${DRY:-0}"
export CONDS SEEDS DRY
has() { [[ " $GROUP " == *" $1 "* ]]; }
run_all() { echo "## run_all.sh $*"; env "$@" bash run_all.sh; }
has fu     && run_all STAGES="attack benign aggregate" BENIGN_METHODS="subtract distillation fedEraser retrain_benign retrain_attack"
has robust && run_all STAGES="robust aggregate"
has detect && run_all STAGES="masa subid latent aggregate"
if has tdef; then
  for d in $TDEFS; do
    run_all STAGES="attack benign aggregate" TAG="_$d" ATTACK_EXTRA="--$d" BENIGN_METHODS="subtract distillation fedEraser"
  done
fi
has prev   && run_all STAGES="release progressive aggregate"
echo "== 끝. 올리기: cd .. && git add FUBA/logs && git commit -m 'baselines: ...' && git push origin exp/baselines"
