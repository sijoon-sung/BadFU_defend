#!/bin/bash
# 논문 실험 전체 (docs/paper/EXPERIMENT_DESIGN.md 5 절). run_all.sh 의 단계를 RQ 별로 묶고, 학습 중 방어 변형과 소거를 더한다.
#
#   bash run_paper.sh                      # 전부 (RQ1~RQ6), CONDS="iid owner", SEEDS="0 1 2"
#   RQ="1 3" bash run_paper.sh             # 일부만
#   TDEFS="median flame" bash run_paper.sh # RQ2b 의 집계 방어 변형만 바꿈
#   DRY=1 bash run_paper.sh                # 명령만 출력
#
# 단계 ↔ RQ
#   RQ1  fu(=benign_departure) + dose + cancel + unified + effects   → T1, T1c, T1u, T1e
#   RQ2  masa, subid, latent (+ tdef: 집계 방어 위에서 궤적 재생성 → fu, dose) → T2, T2b
#   RQ3  pair                                                              → T3
#   RQ4  adaptive                                                          → T4
#   RQ5  bench                                                             → T5
#   RQ6  ablate (c, tau, m, stat, probe, op)                               → T6
#   tables: paper_tables.py --cond {iid,owner}                             → docs/paper/tables_{cond}.md
set -euo pipefail
cd "$(dirname "$0")"
[ -x ../.venv/Scripts/python.exe ] && export PATH="$(cd .. && pwd)/.venv/Scripts:$PATH"
[ -d /c/Users/DISLAB/AppData/Local/Programs/Python/Python312/Library/bin ] && export PATH="/c/Users/DISLAB/AppData/Local/Programs/Python/Python312/Library/bin:$PATH"
[ -d /c/Users/DISLAB/AppData/Local/Programs/Python/Python312 ] && export PATH="/c/Users/DISLAB/AppData/Local/Programs/Python/Python312:/c/Users/DISLAB/AppData/Local/Programs/Python/Python312/Scripts:$PATH"
export PYTHONIOENCODING=utf-8
RQ="${RQ:-1 2 3 4 5 6}"
CONDS="${CONDS:-iid owner}"
SEEDS="${SEEDS:-0 1 2}"
TDEFS="${TDEFS:-median multi_krum flame flDetector}"
DRY="${DRY:-0}"
export CONDS SEEDS DRY
has() { [[ " $RQ " == *" $1 "* ]]; }
run_all() { echo "## run_all.sh $*"; env "$@" bash run_all.sh; }

if has 1; then
  run_all STAGES="attack benign dose cancel unified effects aggregate" BENIGN_METHODS="subtract distillation fedEraser retrain_benign retrain_attack"
fi
if has 2; then
  run_all STAGES="masa subid latent aggregate"
  for d in $TDEFS; do        # 집계 방어 위에서 같은 공격 → 깨어나는가, 우리 방법은 작동하는가
    run_all STAGES="attack benign dose aggregate" TAG="_$d" ATTACK_EXTRA="--$d" BENIGN_METHODS="subtract distillation fedEraser"
  done
fi
if has 3; then
  run_all STAGES="pair aggregate"
fi
if has 4; then
  run_all STAGES="adaptive aggregate" DOSE_SPLIT=10
fi
if has 5; then
  for c in $CONDS; do :; done
  if [ "$DRY" = "1" ]; then echo "+ python bench_release_map.py --fmt fuba --Ks 8 20 50 100 --rounds 4"; else
    [ -f logs/bench/release_map_fuba.json ] || python bench_release_map.py --fmt fuba --Ks 8 20 50 100 --rounds 4; fi
fi
if has 6; then                # 소거: 각 변형을 TAG 로 구분해 dose 만 다시
  run_all STAGES="dose aggregate" TAG="_cap02"  DOSE_EXTRA="--cap 0.2"
  run_all STAGES="dose aggregate" TAG="_tau2"   DOSE_EXTRA="--tau 2"
  run_all STAGES="dose aggregate" TAG="_tau5"   DOSE_EXTRA="--tau 5"
  run_all STAGES="dose aggregate" TAG="_m3"     DOSE_EXTRA="--m 3"
  run_all STAGES="dose aggregate" TAG="_chg"    DOSE_EXTRA="--stat changed"
  run_all STAGES="dose aggregate" TAG="_flow"   DOSE_EXTRA="--stat flow"
  run_all STAGES="dose aggregate" TAG="_p100"   DOSE_EXTRA="--probe 100"
  run_all STAGES="dose aggregate" TAG="_kd"     DOSE_EXTRA="--op kd"
fi
for c in $CONDS; do
  if [ "$DRY" = "1" ]; then echo "+ python paper_tables.py --cond $c"; else python paper_tables.py --cond "$c" || true; fi
done
echo "== 끝. 올릴 것: git add FUBA/logs docs/paper && git commit && git push"
