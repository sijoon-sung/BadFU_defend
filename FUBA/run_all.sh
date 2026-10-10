#!/bin/bash
# =============================================================================
# FUBA 실험 파이프라인 — 단일 진입점
#
#   attack    궤적 생성 (FUBA 공개 코드, mpiexec)        -> checkpoints/*_{NAME}_round_*.pkl
#   benign    정상 이탈 검증 (benign_departure.py)        -> logs/benign_departure/{NAME}.json
#   robust    FedEraser-R 탐지+정화 (robust_fu.py)        -> logs/robust_fu/{NAME}.json
#   latent    잠재 위험 지도, 가중치만 (latent_risk.py)    -> logs/latent_risk/{NAME}.json
#   aggregate 조건별 평균±표준편차 + 종합 보고서           -> logs/*/summary_{cond}.md, logs/pipeline/REPORT_{cond}.md
#
# 사용법 (FUBA 폴더에서)
#   bash run_all.sh                                    # CONDS="owner iid", SEEDS="0 1 2", ROUNDS=8, WARMUP=3
#   CONDS="owner" SEEDS="0" bash run_all.sh
#   ROUNDS=20 WARMUP=5 CONDS="owner" SEEDS="0" bash run_all.sh     # 긴 궤적 (이름에 _r20w5 가 붙음)
#   STAGES="benign robust latent aggregate" bash run_all.sh        # 궤적은 이미 있다고 보고 건너뜀
#   FORCE=1 STAGES="robust aggregate" bash run_all.sh              # 있는 결과도 다시 계산
#   DRY=1 bash run_all.sh                                          # 실행하지 않고 명령만 출력
#
# 조건 (COND)
#   iid    원래 det* 와 같은 IID 분할
#   owner  정상 클라 K-1 이 OWNER_CLASS 의 90% 를 독점 (class_owner)
#   dir01  Dirichlet α=0.1 (FUBA Table V 조건)        dir05  Dirichlet α=0.5
# 역할: 공격자 0..3, Adv-defender(요청자) 4, 정상 5..K-1. 요청자 실험은 4 / K-1 / 5.
#
# 재개: 각 파이썬 단계는 --resume 로 이미 끝난 (요청자, 방법) 쌍을 건너뛴다. FORCE=1 이면 결과 파일을 지우고 다시 한다.
# 로그: logs/pipeline/{NAME}.{stage}.log  (화면에도 같이 출력)
# 끝나면 올릴 것: git add FUBA/logs/benign_departure FUBA/logs/robust_fu FUBA/logs/latent_risk FUBA/logs/pipeline
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
[ -x ../.venv/Scripts/python.exe ] && export PATH="$(cd .. && pwd)/.venv/Scripts:$PATH"
export PYTHONIOENCODING=utf-8

CONDS="${CONDS:-owner iid}"
SEEDS="${SEEDS:-0 1 2}"
STAGES="${STAGES:-attack benign robust latent aggregate}"
K="${K:-8}"
ROUNDS="${ROUNDS:-8}"
WARMUP="${WARMUP:-3}"
OWNER_CLASS="${OWNER_CLASS:-3}"
DEFENDER="${DEFENDER:-4}"
FORCE="${FORCE:-0}"
DRY="${DRY:-0}"
# 단계별 추가 인자 (예: ROBUST_EXTRA="--eps 0.04 --flag_thr 0.5", BENIGN_METHODS="subtract distillation fedEraser")
BENIGN_METHODS="${BENIGN_METHODS:-subtract distillation fedEraser retrain_benign retrain_attack}"
ROBUST_ARMS="${ROBUST_ARMS:-plain purify_flag purify_always purify_clean}"
ROBUST_EXTRA="${ROBUST_EXTRA:-}"
LATENT_EXTRA="${LATENT_EXTRA:-}"

export ROUNDS WARMUP   # run_real_fuba.sh 가 읽는다
mkdir -p logs/pipeline

# ---------------------------------------------------------------- helpers
run() {                       # run <cmd...>           : 명령 출력 후 실행 (DRY=1 이면 출력만)
  echo "+ $*"
  [ "$DRY" = "1" ] || "$@"
}
run_log() {                   # run_log <logfile> <cmd...> : 위와 같되 tee 로 로그 저장
  local log="$1"; shift
  echo "+ $* 2>&1 | tee $log"
  [ "$DRY" = "1" ] || "$@" 2>&1 | tee "$log" | grep -v "^Files already"
}
has() { [[ " $STAGES " == *" $1 "* ]]; }
ckpt_count() { ls checkpoints/local_net_model_"$1"_round_*.pkl 2>/dev/null | wc -l | tr -d ' '; }
maybe_rm() { [ "$FORCE" = "1" ] && [ -f "$1" ] && { echo "  FORCE: rm $1"; [ "$DRY" = "1" ] || rm -f "$1"; }; return 0; }

cond_args() {                 # cond_args <cond> -> 전역 배열 NI (파이썬/MPI 공통 비-IID 인자)
  case "$1" in
    iid)   NI=() ;;
    owner) NI=(--non_iid --non_iid_type class_owner --owner_class "$OWNER_CLASS" --owner_frac 0.9 --owner_client -1) ;;
    dir01) NI=(--non_iid --non_iid_type Dirichlet --alpha 0.1) ;;
    dir05) NI=(--non_iid --non_iid_type Dirichlet --alpha 0.5) ;;
    *) echo "COND 는 iid|owner|dir01|dir05 중 하나: $1" >&2; exit 1 ;;
  esac
}
cond_label() {                # 결과 파일 접두사: bd_{label}_s{seed}. 기본 설정(8라운드/웜업3)은 기존 이름과 호환
  if [ "$ROUNDS" = "8" ] && [ "$WARMUP" = "3" ]; then echo "$1"; else echo "${1}_r${ROUNDS}w${WARMUP}"; fi
}

# ---------------------------------------------------------------- stages
stage_attack() {              # <name> <seed>
  local name="$1" seed="$2"
  if [ "$FORCE" != "1" ] && [ "$(ckpt_count "$name")" -ge "$ROUNDS" ]; then
    echo "  [$name] 궤적 있음 ($(ckpt_count "$name") 라운드) — 생성 생략"; return 0
  fi
  NB_CLIENTS="$K" EXTRA_ARGS="${NI[*]} --split_seed $seed" \
    run_log "logs/pipeline/${name}.attack.log" bash run_real_fuba.sh "$name" "$seed"
  [ "$DRY" = "1" ] && return 0
  local n; n="$(ckpt_count "$name")"
  [ "$n" -ge "$ROUNDS" ] || { echo "  [$name] 궤적 생성 실패: 저장된 라운드 $n/$ROUNDS. logs/pipeline/${name}.attack.log 확인" >&2; exit 1; }
}

stage_benign() {              # <name> <seed>
  local name="$1" seed="$2"
  maybe_rm "logs/benign_departure/${name}.json"
  run_log "logs/pipeline/${name}.benign.log" python benign_departure.py --name "$name" --K "$K" --rounds "$ROUNDS" \
    --seed "$seed" --split_seed "$seed" "${NI[@]}" --defender "$DEFENDER" \
    --requesters "$DEFENDER" $((K - 1)) 5 --methods $BENIGN_METHODS --resume
}

stage_robust() {              # <name> <seed>
  local name="$1" seed="$2"
  maybe_rm "logs/robust_fu/${name}.json"
  run_log "logs/pipeline/${name}.robust.log" python robust_fu.py --name "$name" --K "$K" --rounds "$ROUNDS" \
    --seed "$seed" --split_seed "$seed" "${NI[@]}" --defender "$DEFENDER" \
    --requesters "$DEFENDER" $((K - 1)) 5 --arms $ROBUST_ARMS $ROBUST_EXTRA --resume
}

stage_latent() {              # <name> <seed>
  local name="$1" seed="$2"
  maybe_rm "logs/latent_risk/${name}.json"
  if [ "$FORCE" != "1" ] && [ -f "logs/latent_risk/${name}.json" ]; then echo "  [$name] latent_risk 결과 있음 — 생략"; return 0; fi
  run_log "logs/pipeline/${name}.latent.log" python latent_risk.py --name "$name" --rounds "$ROUNDS" --warmup "$WARMUP" \
    --seed "$seed" --defender "$DEFENDER" $LATENT_EXTRA
}

stage_aggregate() {           # <label> <names...>
  local label="$1"; shift
  local names=("$@") report="logs/pipeline/REPORT_${label}.md"
  # 결과 파일이 있는 단계만 집계한다 (집계 스크립트는 결과가 없으면 종료 코드 1 을 내므로)
  have() { ls "logs/$1/bd_${label}_s"*.json >/dev/null 2>&1; }
  { has benign && have benign_departure; } && run python aggregate_benign_departure.py --cond "$label"
  { has robust && have robust_fu; }        && run python aggregate_robust_fu.py --cond "$label"
  { has latent && have latent_risk; }      && run python aggregate_latent_risk.py --names "${names[@]}" --out "logs/latent_risk/summary_${label}.md"
  [ "$DRY" = "1" ] && return 0
  {
    echo "# 종합 보고서: $label  (K=$K, ROUNDS=$ROUNDS, WARMUP=$WARMUP, seeds: $SEEDS)"; echo
    for f in "logs/benign_departure/summary_${label}.md" "logs/robust_fu/summary_${label}.md" "logs/latent_risk/summary_${label}.md"; do
      [ -f "$f" ] && { echo "---"; echo "<!-- $f -->"; cat "$f"; echo; }
    done
  } > "$report"
  echo "  -> $report"
}

# ---------------------------------------------------------------- main
echo "== CONDS=[$CONDS] SEEDS=[$SEEDS] STAGES=[$STAGES] K=$K ROUNDS=$ROUNDS WARMUP=$WARMUP FORCE=$FORCE DRY=$DRY"
for COND in $CONDS; do
  cond_args "$COND"
  LABEL="$(cond_label "$COND")"
  NAMES=()
  for s in $SEEDS; do
    NAME="bd_${LABEL}_s${s}"; SEED=$((522 + s)); NAMES+=("$NAME")
    echo; echo "==== [$COND] seed $s -> $NAME (seed $SEED)"
    has attack && stage_attack "$NAME" "$SEED"
    has benign && stage_benign "$NAME" "$SEED"
    has robust && stage_robust "$NAME" "$SEED"
    has latent && stage_latent "$NAME" "$SEED"
  done
  if has aggregate; then echo; echo "==== [$COND] 집계"; stage_aggregate "$LABEL" "${NAMES[@]}"; fi
done
echo
echo "== 끝. 올릴 것: git add FUBA/logs/benign_departure FUBA/logs/robust_fu FUBA/logs/latent_risk FUBA/logs/pipeline"
