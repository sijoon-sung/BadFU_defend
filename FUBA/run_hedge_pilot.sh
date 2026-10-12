#!/bin/bash
# 헤지 파일럿 (docs/HEDGE_PILOT.md): 위장자–운반자 갱신이 라운드마다 반대 방향인가를 저장 갱신만으로 잰다.
#   궤적이 없으면 저자 코드(run_real_fuba.sh → main.py)로 만든다. 학습 코드는 손대지 않는다.
#   조건: iid (bd_iid_s*), dir05 (Dirichlet α=0.5, bd_dir05_s*), owner (bd_owner_s*). K=8, 8 라운드 (축소 설정).
#   K=20 논문 설정 궤적(bd_*_r100w5_k20paper_s*) 이 있으면 그것도 잰다 (make_trajectories.sh 로 먼저 만들 것).
# 사용:  bash run_hedge_pilot.sh                 # 전부
#        CONDS="iid dir05" SEEDS="0 1 2" bash run_hedge_pilot.sh
#        DRY=1 bash run_hedge_pilot.sh           # 명령만 출력
#        FORCE=1 ...                             # 분석 결과(hedge_*.json) 다시 만듦 (궤적은 안 지움)
# 결과: FUBA/logs/analysis/hedge_{name}.json, FUBA/logs/analysis/hedge_summary.md (판정 표)
set -e
cd "$(dirname "$0")"
CONDS="${CONDS:-iid dir05 owner}"
SEEDS="${SEEDS:-0 1 2}"
K="${K:-8}"; ROUNDS="${ROUNDS:-8}"; WARMUP="${WARMUP:-3}"
OWNER_CLASS="${OWNER_CLASS:-3}"
DEFENDER="${DEFENDER:-4}"; ATTACKERS="${ATTACKERS:-0 1 2 3}"
DRY="${DRY:-0}"; FORCE="${FORCE:-0}"
export PYTHONIOENCODING=utf-8

run() { echo "+ $*"; [ "$DRY" = "1" ] || "$@"; }
ckpt_count() { ls checkpoints/local_net_model_"$1"_round_*.pkl 2>/dev/null | wc -l | tr -d ' '; }

for COND in $CONDS; do
  case "$COND" in
    iid)   NI="" ;;
    owner) NI="--non_iid --non_iid_type class_owner --owner_class $OWNER_CLASS --owner_frac 0.9 --owner_client -1" ;;
    dir05) NI="--non_iid --non_iid_type Dirichlet --alpha 0.5" ;;
    dir01) NI="--non_iid --non_iid_type Dirichlet --alpha 0.1" ;;
    *) echo "모르는 조건: $COND (iid|owner|dir05|dir01)"; exit 1 ;;
  esac
  for S in $SEEDS; do
    SEED=$((522 + S)); NAME="bd_${COND}_s${S}"
    if [ "$(ckpt_count "$NAME")" -lt "$ROUNDS" ]; then
      echo "== [$NAME] 궤적 없음 → 생성 (저자 코드, seed $SEED)"
      run env NB_CLIENTS="$K" ROUNDS="$ROUNDS" WARMUP="$WARMUP" EXTRA_ARGS="$NI --split_seed $SEED" bash run_real_fuba.sh "$NAME" "$SEED"
      [ "$DRY" = "1" ] || [ "$(ckpt_count "$NAME")" -ge "$ROUNDS" ] || { echo "  [$NAME] 생성 실패"; exit 1; }
    else
      echo "== [$NAME] 궤적 있음 ($(ckpt_count "$NAME") 라운드)"
    fi
    if [ "$FORCE" != "1" ] && [ -f "logs/analysis/hedge_${NAME}.json" ]; then echo "  분석 결과 있음 — 생략"; continue; fi
    run python analysis_hedge.py --fmt fuba --name "$NAME" --K "$K" --rounds "$ROUNDS" --defender "$DEFENDER" --attackers $ATTACKERS --cond "$COND"
  done
done

# K=20 논문 설정 궤적이 있으면 같이 잰다 (이름은 trajectories.json 과 같음)
for COND in iid owner; do
  for S in 0 1 2; do
    NAME="bd_${COND}_r100w5_k20paper_s${S}"
    if [ "$(ckpt_count "$NAME")" -ge 100 ]; then
      if [ "$FORCE" != "1" ] && [ -f "logs/analysis/hedge_${NAME}.json" ]; then continue; fi
      run python analysis_hedge.py --fmt fuba --name "$NAME" --K 20 --rounds 100 --defender 4 --attackers 0 1 2 3 --cond "${COND}_k20paper"
    fi
  done
done

run python aggregate_hedge.py
echo "== 끝. 판정 표: logs/analysis/hedge_summary.md  (올릴 것: git add FUBA/logs/analysis)"
