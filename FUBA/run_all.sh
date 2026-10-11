#!/bin/bash
# =============================================================================
# FUBA 실험 파이프라인 — 단일 진입점
#
#   attack    궤적 생성 (FUBA 공개 코드, mpiexec)        -> checkpoints/*_{NAME}_round_*.pkl
#   benign    정상 이탈 검증 (benign_departure.py)        -> logs/benign_departure/{NAME}.json
#   robust    FedEraser-R 탐지+정화 (robust_fu.py)        -> logs/robust_fu/{NAME}.json
#   latent    잠재 위험 지도, 가중치만 (latent_risk.py)    -> logs/latent_risk/{NAME}.json
#   release   빼 보기: 천천히 빼기 + 전원 빼 보기 (release_map.py) -> logs/release_map/{NAME}.json
#   progressive 진행형 FU: 강도를 올려 가며 탐지, 걸리면 정화 (progressive_fu.py) -> logs/progressive_fu/{NAME}.json
#   dose      투여량 조절 FU: 상한 c 로 δ 선택, 같은 양 비교, 걸리면 수정 (dose_fu.py) -> logs/dose_fu/{NAME}.json
#   masa      기준선 MASA: 학습 중 라운드별 언러닝 손실 편차 (baseline_masa.py)   -> logs/baseline_masa/{NAME}.json
#   subid     기준선 SubID 식 부분공간 편차 (baseline_subid.py)                 -> logs/baseline_subid/{NAME}.json
#   pair      쌍→집합: 표시된 요청의 운반자 집합·공동 위장자 (pair_roles.py)     -> logs/pair_roles/{NAME}.json
#   adaptive  적응형 공격 5 종 (학습 없이 저장 갱신 변형: attack_sim.py)           -> logs/adaptive/{NAME}_*.json, logs/dose_fu/{NAME}_split10_ledger.json, logs/pair_roles/{NAME}_distcarr.json
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
[ -d /c/Users/DISLAB/AppData/Local/Programs/Python/Python312/Library/bin ] && export PATH="/c/Users/DISLAB/AppData/Local/Programs/Python/Python312/Library/bin:$PATH"
export PYTHONIOENCODING=utf-8

CONDS="${CONDS:-owner iid}"
SEEDS="${SEEDS:-0 1 2}"
STAGES="${STAGES:-attack benign robust latent release progressive dose masa subid pair adaptive aggregate}"
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
RELEASE_EXTRA="${RELEASE_EXTRA:---ref population}"   # 기본: 나머지 전원 기준, linf+patch 탐색 (release_map.py 기본값)
PROGRESSIVE_EXTRA="${PROGRESSIVE_EXTRA:-}"            # 예: PROGRESSIVE_EXTRA="--tau 3 --delta_mode grid"
DOSE_EXTRA="${DOSE_EXTRA:-}"                          # 예: DOSE_EXTRA="--cap 0.2 --m 5"
DOSE_SPLIT="${DOSE_SPLIT:-0}"                         # >1 이면 쪼개기 공격 실험도 (--split N --ledger)
MASA_EXTRA="${MASA_EXTRA:-}"
SUBID_EXTRA="${SUBID_EXTRA:-}"
PAIR_EXTRA="${PAIR_EXTRA:-}"                          # 예: PAIR_EXTRA="--exhaust --repair"
ADAPT_EXTRA="${ADAPT_EXTRA:-}"                        # dose_fu/schedule_fu/pair_roles 공통 추가 인자 (예: --cap 0.3)

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

stage_release() {             # <name> <seed>  — 학습 없음(빼기 + 순전파), 전원 빼 보기
  local name="$1" seed="$2"
  maybe_rm "logs/release_map/${name}.json"
  if [ "$FORCE" != "1" ] && [ -f "logs/release_map/${name}.json" ]; then echo "  [$name] release_map 결과 있음 — 생략"; return 0; fi
  run_log "logs/pipeline/${name}.release.log" python release_map.py --fmt fuba --name "$name" --K "$K" --rounds "$ROUNDS" \
    --seed "$seed" --requester "$DEFENDER" --attackers 0 1 2 3 $RELEASE_EXTRA
}

stage_progressive() {         # <name> <seed>  — 진행형 정확 제거 FU (탐지+정화), 요청자 4 / K-1 / 5
  local name="$1" seed="$2"
  maybe_rm "logs/progressive_fu/${name}.json"
  run_log "logs/pipeline/${name}.progressive.log" python progressive_fu.py --fmt fuba --name "$name" --K "$K" --rounds "$ROUNDS" \
    --seed "$seed" --split_seed "$seed" "${NI[@]}" --defender "$DEFENDER" --requesters "$DEFENDER" $((K - 1)) 5 $PROGRESSIVE_EXTRA --resume
}

stage_dose() {                # <name> <seed>  — 투여량 조절 FU (제안), 요청자 4 / K-1 / 5
  local name="$1" seed="$2"
  maybe_rm "logs/dose_fu/${name}.json"
  run_log "logs/pipeline/${name}.dose.log" python dose_fu.py --fmt fuba --name "$name" --K "$K" --rounds "$ROUNDS"     --seed "$seed" --split_seed "$seed" "${NI[@]}" --defender "$DEFENDER" --requesters "$DEFENDER" $((K - 1)) 5 $DOSE_EXTRA --resume
  if [ "$DOSE_SPLIT" -gt 1 ]; then
    maybe_rm "logs/dose_fu/${name}_split${DOSE_SPLIT}_ledger.json"
    run_log "logs/pipeline/${name}.dose_split.log" python dose_fu.py --fmt fuba --name "$name" --K "$K" --rounds "$ROUNDS"       --seed "$seed" --split_seed "$seed" "${NI[@]}" --defender "$DEFENDER" --requesters "$DEFENDER" $((K - 1)) 5       --split "$DOSE_SPLIT" --ledger --arms oneshot dose $DOSE_EXTRA --resume
  fi
}

stage_masa() {                # <name> <seed>  — 기준선 MASA (학습 중)
  local name="$1" seed="$2"
  maybe_rm "logs/baseline_masa/${name}.json"
  if [ "$FORCE" != "1" ] && [ -f "logs/baseline_masa/${name}.json" ]; then echo "  [$name] masa 결과 있음 — 생략"; return 0; fi
  run_log "logs/pipeline/${name}.masa.log" python baseline_masa.py --fmt fuba --name "$name" --K "$K" --rounds "$ROUNDS"     --seed "$seed" --defender "$DEFENDER" --attackers 0 1 2 3 $MASA_EXTRA
}

stage_subid() {               # <name> <seed>  — 기준선 SubID 식 점수 (학습 없음)
  local name="$1" seed="$2"
  maybe_rm "logs/baseline_subid/${name}.json"
  if [ "$FORCE" != "1" ] && [ -f "logs/baseline_subid/${name}.json" ]; then echo "  [$name] subid 결과 있음 — 생략"; return 0; fi
  run_log "logs/pipeline/${name}.subid.log" python baseline_subid.py --fmt fuba --name "$name" --K "$K" --rounds "$ROUNDS"     --defender "$DEFENDER" --attackers 0 1 2 3 $SUBID_EXTRA
}

stage_pair() {                # <name> <seed>  — 쌍→집합 역할 지도 (요청자 4 와 정상 K-1, 그리고 4+5 동시 요청)
  local name="$1" seed="$2"
  maybe_rm "logs/pair_roles/${name}.json"; maybe_rm "logs/pair_roles/${name}_joint.json"
  if [ "$FORCE" != "1" ] && [ -f "logs/pair_roles/${name}.json" ]; then echo "  [$name] pair 결과 있음 — 생략"; else
    run_log "logs/pipeline/${name}.pair.log" python pair_roles.py --fmt fuba --name "$name" --K "$K" --rounds "$ROUNDS"       --seed "$seed" --defender "$DEFENDER" --attackers 0 1 2 3 --requesters "$DEFENDER" $((K - 1)) $PAIR_EXTRA
  fi
  if [ "$FORCE" != "1" ] && [ -f "logs/pair_roles/${name}_joint.json" ]; then echo "  [$name] pair joint 결과 있음 — 생략"; else
    run_log "logs/pipeline/${name}.pair_joint.log" python pair_roles.py --fmt fuba --name "$name" --K "$K" --rounds "$ROUNDS"       --seed "$seed" --defender "$DEFENDER" --attackers 0 1 2 3 --requesters "$DEFENDER" 5 --joint $PAIR_EXTRA
  fi
}

stage_adaptive() {            # <name> <seed>  — 적응형 공격 5 종. K=8 기준: 위장자 4 의 조각은 8, 9 로 붙는다
  local name="$1" seed="$2"
  local common=(--fmt fuba --name "$name" --K "$K" --rounds "$ROUNDS" --seed "$seed" --defender "$DEFENDER" --attackers 0 1 2 3)
  # a. 요청 쪼개기 10 조각 + 누적 장부
  if [ "$FORCE" = "1" ] || [ ! -f "logs/dose_fu/${name}_split10_ledger.json" ]; then
    run_log "logs/pipeline/${name}.adapt_split.log" python dose_fu.py "${common[@]}" --split_seed "$seed" "${NI[@]}"       --requesters "$DEFENDER" $((K - 1)) --split 10 --ledger --arms oneshot dose $ADAPT_EXTRA --resume
  fi
  # b. 위장자 분산: 4 를 3 조각(4, K, K+1)으로, 순차 요청. 정상 열 (K-1, 5, 6) 대조
  if [ "$FORCE" = "1" ] || [ ! -f "logs/adaptive/${name}_distsupp.json" ]; then
    run_log "logs/pipeline/${name}.adapt_distsupp.log" python schedule_fu.py "${common[@]}" --sim "split:${DEFENDER}:3"       --requests "$DEFENDER" "$K" $((K + 1)) --control $((K - 1)) 5 6 --tag distsupp $ADAPT_EXTRA
  fi
  # c. 모집단 오염: 정상 3 명(5, 6, K-1) 기여 ×2.5
  if [ "$FORCE" = "1" ] || [ ! -f "logs/adaptive/${name}_decoy.json" ]; then
    run_log "logs/pipeline/${name}.adapt_decoy.log" python dose_fu.py "${common[@]}" --split_seed "$seed" "${NI[@]}"       --sim "scale:5:2.5,scale:6:2.5,scale:$((K - 1)):2.5" --requesters "$DEFENDER" --arms oneshot $ADAPT_EXTRA       --out "logs/adaptive/${name}_decoy.json"
  fi
  # g. 기준 오염: 정상 3 명 기여에 위장자 기여 0.3 배
  if [ "$FORCE" = "1" ] || [ ! -f "logs/adaptive/${name}_refpoison.json" ]; then
    run_log "logs/pipeline/${name}.adapt_refpoison.log" python dose_fu.py "${common[@]}" --split_seed "$seed" "${NI[@]}"       --sim "dir:5:${DEFENDER}:0.3,dir:6:${DEFENDER}:0.3,dir:$((K - 1)):${DEFENDER}:0.3" --requesters "$DEFENDER" --arms oneshot $ADAPT_EXTRA       --out "logs/adaptive/${name}_refpoison.json"
  fi
  # i. 운반자 분산: 0–3 을 각 3 조각
  if [ "$FORCE" = "1" ] || [ ! -f "logs/pair_roles/${name}_distcarr.json" ]; then
    run_log "logs/pipeline/${name}.adapt_distcarr.log" python pair_roles.py "${common[@]}" --sim "split:0:3,split:1:3,split:2:3,split:3:3"       --requesters "$DEFENDER" --exhaust $PAIR_EXTRA $ADAPT_EXTRA --out "logs/pair_roles/${name}_distcarr.json"
  fi
}

stage_aggregate() {           # <label> <names...>
  local label="$1"; shift
  local names=("$@") report="logs/pipeline/REPORT_${label}.md"
  # 결과 파일이 있는 단계만 집계한다 (집계 스크립트는 결과가 없으면 종료 코드 1 을 내므로)
  have() { ls "logs/$1/bd_${label}_s"*.json >/dev/null 2>&1; }
  { has benign && have benign_departure; } && run python aggregate_benign_departure.py --cond "$label"
  { has robust && have robust_fu; }        && run python aggregate_robust_fu.py --cond "$label"
  { has latent && have latent_risk; }      && run python aggregate_latent_risk.py --names "${names[@]}" --out "logs/latent_risk/summary_${label}.md"
  { has release && have release_map; }     && run python aggregate_release_map.py --cond "$label"
  { has progressive && have progressive_fu; } && run python aggregate_progressive_fu.py --cond "$label"
  { has dose && have dose_fu; }            && run python aggregate_dose_fu.py --cond "$label"
  { has dose && [ "$DOSE_SPLIT" -gt 1 ] && have dose_fu; } && run python aggregate_dose_fu.py --cond "$label" --tag "_split${DOSE_SPLIT}_ledger"
  { { has masa && have baseline_masa; } || { has subid && have baseline_subid; }; } && run python aggregate_baselines.py --cond "$label"
  { has adaptive && ls "logs/adaptive/bd_${label}_s"*.json >/dev/null 2>&1; } && run python aggregate_adaptive.py --cond "$label"
  [ "$DRY" = "1" ] && return 0
  {
    echo "# 종합 보고서: $label  (K=$K, ROUNDS=$ROUNDS, WARMUP=$WARMUP, seeds: $SEEDS)"; echo
    for f in "logs/benign_departure/summary_${label}.md" "logs/robust_fu/summary_${label}.md" "logs/latent_risk/summary_${label}.md" "logs/release_map/summary_${label}.md" "logs/progressive_fu/summary_${label}.md" "logs/dose_fu/summary_${label}.md" "logs/dose_fu/summary_${label}_split${DOSE_SPLIT}_ledger.md" "logs/baselines/summary_${label}.md" "logs/adaptive/summary_${label}.md"; do
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
    has release && stage_release "$NAME" "$SEED"
    has progressive && stage_progressive "$NAME" "$SEED"
    has dose && stage_dose "$NAME" "$SEED"
    has masa && stage_masa "$NAME" "$SEED"
    has subid && stage_subid "$NAME" "$SEED"
    has pair && stage_pair "$NAME" "$SEED"
    has adaptive && stage_adaptive "$NAME" "$SEED"
    sleep 2
  done
  if has aggregate; then echo; echo "==== [$COND] 집계"; stage_aggregate "$LABEL" "${NAMES[@]}"; fi
done
echo
echo "== 끝. 올릴 것: git add FUBA/logs/benign_departure FUBA/logs/robust_fu FUBA/logs/latent_risk FUBA/logs/release_map FUBA/logs/progressive_fu FUBA/logs/dose_fu FUBA/logs/baseline_masa FUBA/logs/baseline_subid FUBA/logs/baselines FUBA/logs/pair_roles FUBA/logs/adaptive FUBA/logs/pipeline"
