#!/bin/bash
# =============================================================================
# BadFU 궤적 공장 (노트북 GPU). ../../FUBA/trajectories.json 의 badfu 항목 중 없는 시드를 fl_detect_badfu.py
# (저자 badfu.py 의 FL 루프 + FedEraser 언러닝을 그대로 옮기고 라운드별 저장만 더한 것) 로 만들고, 만들 때마다 검증한다.
#
#   bash make_badfu_trajectories.sh            # 없는 시드 전부 (기존 logs/traj_full 은 절대 건드리지 않는다)
#   DRY=1 bash make_badfu_trajectories.sh      # 명령만
#   ONLY=badfu_s43 bash make_badfu_trajectories.sh
#   FORCE=1 은 기존 궤적 삭제를 허용하지 않는다. 새 실험은 request_pipeline.py의 별도 run-dir을 쓴다.
#
# 데이터(record/badnet_dataset/pert_result.pt)가 없으면 저자 prepare_data.sh 의 두 명령(badnet.py → uba_inf_cover.py)을
# docs/archive/fuba/LOCAL_SMOKE.md Part B 의 flag 로 먼저 돌린다 (prepare_data.sh 의 파괴적 rm/mv 는 생략, 약 12 분).
# 부분 궤적은 보존하고 중단한다. 새 폴더에서 재실행해야 한다. 시드 하나 ≈ 2.5 시간(40 라운드), 디스크 ≈ 14 GB.
# 로그: logs/traj_s{seed}/run.log (counts 줄을 검증기가 읽는다). 결과 json: logs/badfu_detect_s{seed}.json (단계식 언러닝 ASR).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
[ -d /c/Users/sijoo/.conda/envs/pytorch ] && export PATH="/c/Users/sijoo/.conda/envs/pytorch:/c/Users/sijoo/.conda/envs/pytorch/Scripts:/c/Users/sijoo/.conda/envs/pytorch/Library/bin:$PATH"
[ -d /c/Users/DISLAB/AppData/Local/Programs/Python/Python312 ] && export PATH="/c/Users/DISLAB/AppData/Local/Programs/Python/Python312:/c/Users/DISLAB/AppData/Local/Programs/Python/Python312/Scripts:/c/Users/DISLAB/AppData/Local/Programs/Python/Python312/Library/bin:$PATH"
export PYTHONIOENCODING=utf-8
DRY="${DRY:-0}"; FORCE="${FORCE:-0}"; ONLY="${ONLY:-}"
FUBA="$(cd ../../FUBA && pwd)"
MANIFEST="${MANIFEST:-$FUBA/trajectories.json}"     # 절대 경로: verify_trajectories.py 는 FUBA 폴더로 chdir 한다
abort() { echo; echo "!! 중단: $*" >&2; exit 1; }
run() { echo "+ $*"; [ "$DRY" = "1" ] || "$@" </dev/null; }

# shellcheck disable=SC2086
LIST="$(python "$FUBA/verify_trajectories.py" --list --attack badfu ${ONLY:+--only $ONLY} --manifest "$MANIFEST")"
[ -n "$LIST" ] || { echo "만들 BadFU 궤적이 없음 (ONLY='$ONLY')"; exit 0; }
echo "== make_badfu_trajectories  ONLY='${ONLY:-전부}'  FORCE=$FORCE  DRY=$DRY"

# ---- 0) 데이터 (저자 prepare_data.sh 순서, LOCAL_SMOKE.md Part B 의 flag)
if [ ! -f record/badnet_dataset/pert_result.pt ]; then
  echo; echo "== 데이터 없음 → BadNet 트리거 + UBA-Inf 위장 데이터를 만든다 (저자 코드, 약 12 분)"
  mkdir -p data_prepare/record
  ( cd data_prepare && \
    run python ./attack/badnet.py --yaml_path ../config/attack/prototype/cifar10.yaml --save_folder_name badnet_dataset \
        --add_cover 1 --epochs 0 --pratio 0.017 --cratio 0.017 --attack_target 0 --model resnet18 --device cuda:0 --num_workers 0 && \
    run python ./uba/uba_inf_cover.py --dataset_folder ../record/badnet_dataset --device cuda:0 --ft_epochs 1 --ap_epochs 1 --num_workers 0 )
  echo "+ mkdir -p record && cp -r data_prepare/record/. record/"
  [ "$DRY" = "1" ] || { mkdir -p record && cp -r data_prepare/record/. record/; }
  [ "$DRY" = "1" ] || [ -f record/badnet_dataset/pert_result.pt ] || abort "데이터 생성 뒤에도 record/badnet_dataset/pert_result.pt 가 없음"
else
  echo "   데이터 있음: record/badnet_dataset/pert_result.pt"
fi

# ---- 1) 시드별 궤적
n_made=0; n_skip=0
while IFS=$'\t' read -r name where traj rounds epochs seed detect counts <&3; do
  [ -n "$name" ] || continue
  n_round="$(ls "$traj"/round_*.pt 2>/dev/null | wc -l | tr -d ' ' || true)"   # || true: 파일이 없을 때 pipefail 로 죽지 않게
  last="$traj/glob_$((rounds - 1)).pt"
  cmd="python fl_detect_badfu.py --rounds $rounds --local_epochs $epochs --keep_traj --seed $seed --traj_dir $traj --out $detect"
  if [ "$traj" = "logs/traj_full" ]; then
    echo; echo "== $name: 기존 궤적 logs/traj_full 은 다시 만들지 않는다 (있음 $n_round/$rounds 라운드)"; n_skip=$((n_skip + 1))
  elif [ "$FORCE" != "1" ] && [ -f "$last" ] && [ "$n_round" -ge "$rounds" ]; then
    echo; echo "== $name: 있음 ($n_round/$rounds 라운드) — 생성 생략"; echo "   ($cmd)"; n_skip=$((n_skip + 1))
  else
    [ "$n_round" -eq 0 ] || abort "[$name] 기존 궤적 $n_round 라운드를 보존합니다. request_pipeline.py에 새 --run-dir을 지정하세요."
    echo; echo "== $name  (새 궤적, 예상 ≈ 2.5 시간)"
    echo "+ mkdir -p $traj && $cmd 2>&1 | tee $traj/run.log"
    if [ "$DRY" != "1" ]; then
      mkdir -p "$traj"
      t0=$(date +%s)
      if ! python fl_detect_badfu.py --rounds "$rounds" --local_epochs "$epochs" --keep_traj --seed "$seed" --traj_dir "$traj" --out "$detect" </dev/null 2>&1 | tee "$traj/run.log"; then
        abort "[$name] fl_detect_badfu.py 실패. $traj/run.log 끝부분을 볼 것"
      fi
      [ -f "$last" ] || abort "[$name] $last 가 없음 (라운드가 다 저장되지 않음)"
      [ -f "$detect" ] || abort "[$name] $detect 가 없음 (단계식 언러닝까지 끝나지 않음)"
      echo "   생성 완료: $(( $(date +%s) - t0 )) 초"
      n_made=$((n_made + 1))
    fi
  fi
  if [ "$DRY" = "1" ]; then
    echo "+ (cd $FUBA && python verify_trajectories.py --only $name --strict)"
  else
    ( cd "$FUBA" && python verify_trajectories.py --only "$name" --strict --manifest "$MANIFEST" </dev/null ) \
      || abort "[$name] 검증 실패 — docs/EXPERIMENT_LOG.md 의 '실패 사유' 를 볼 것"
  fi
done 3<<< "$LIST"
echo
echo "== 끝. 새로 만듦 $n_made, 생략 $n_skip. 기록: docs/EXPERIMENT_LOG.md, FUBA/logs/trajectories/registry.json (커밋할 것)"
