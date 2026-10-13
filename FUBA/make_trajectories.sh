#!/bin/bash
# =============================================================================
# 공격 궤적 공장 (FUBA). trajectories.json 에 적힌 궤적 중 없는 것을 저자 코드(main.py, mpiexec; run_real_fuba.sh)로 만들고,
# 하나 만들 때마다 verify_trajectories.py 로 검증한다. 검증에 실패하면 거기서 멈춘다.
#
#   bash make_trajectories.sh                        # 이 컴퓨터 몫 (DISLAB 폴더가 있으면 where=DISLAB, 아니면 local) 전부
#   DRY=1 bash make_trajectories.sh                  # 명령만 출력 (WHERE=all 로 전체 그룹)
#   ONLY="bd_iid_k20_s0" bash make_trajectories.sh   # 이름 / 그룹(k20_reduced, k20_paper, sup2, tdef_paper, tdef_reduced, k8_main) / 여러 개는 공백 구분
#   WHERE=all|DISLAB|local                           # 어느 컴퓨터 몫을 만들지 (기본 auto)
#   FORCE=1                                          # 있어도 다시 만든다 (방어 없이 잘못 만들어진 궤적을 갈아엎을 때)
#   SKIP_VERIFY=1                                    # 생성 뒤 검증 생략 (권장 안 함)
#
# 로그: logs/pipeline/{name}.attack.log. 기록: logs/trajectories/registry.json, docs/EXPERIMENT_LOG.md (검증기가 쓴다).
# BadFU 궤적은 ../BadFU_src/BadFU-main/make_badfu_trajectories.sh 가 만든다 (여기서는 다루지 않음).
# 우리가 정하는 것은 flag(클라 수, 참여율, 라운드, 시드, 분할, 방어)뿐이고 학습 코드는 저자 것 그대로다.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
[ -x ../.venv/Scripts/python.exe ] && export PATH="$(cd .. && pwd)/.venv/Scripts:$PATH"
[ -d /c/Users/DISLAB/AppData/Local/Programs/Python/Python312/Library/bin ] && export PATH="/c/Users/DISLAB/AppData/Local/Programs/Python/Python312/Library/bin:$PATH"
[ -d /c/Users/DISLAB/AppData/Local/Programs/Python/Python312 ] && export PATH="/c/Users/DISLAB/AppData/Local/Programs/Python/Python312:/c/Users/DISLAB/AppData/Local/Programs/Python/Python312/Scripts:$PATH"
[ -d /c/Users/sijoo/.conda/envs/pytorch ] && export PATH="/c/Users/sijoo/.conda/envs/pytorch:/c/Users/sijoo/.conda/envs/pytorch/Scripts:/c/Users/sijoo/.conda/envs/pytorch/Library/bin:$PATH"
export PYTHONIOENCODING=utf-8

DRY="${DRY:-0}"; FORCE="${FORCE:-0}"; ONLY="${ONLY:-}"; WHERE="${WHERE:-auto}"; SKIP_VERIFY="${SKIP_VERIFY:-0}"
MANIFEST="${MANIFEST:-trajectories.json}"
if [ "$WHERE" = "auto" ]; then
  if [ -d /c/Users/DISLAB ]; then WHERE=DISLAB; else WHERE=local; fi
fi
mkdir -p logs/pipeline logs/trajectories
ckpt_count() { ls checkpoints/local_net_model_"$1"_round_*.pkl 2>/dev/null | wc -l | tr -d ' ' || true; }   # || true: 파일이 없을 때 pipefail 로 죽지 않게
abort() { echo; echo "!! 중단: $*" >&2; exit 1; }

# shellcheck disable=SC2086
LIST="$(python verify_trajectories.py --list --attack fuba --where "$WHERE" ${ONLY:+--only $ONLY} --manifest "$MANIFEST")"
[ -n "$LIST" ] || { echo "만들 FUBA 궤적이 없음 (WHERE=$WHERE ONLY='$ONLY'). BadFU 는 make_badfu_trajectories.sh"; exit 0; }
echo "== make_trajectories  WHERE=$WHERE  ONLY='${ONLY:-전부}'  FORCE=$FORCE  DRY=$DRY  manifest=$MANIFEST"
command -v mpiexec >/dev/null 2>&1 || echo "   (mpiexec 가 PATH 에 없음 — run_real_fuba.sh 가 python 의 impi-rt 에서 찾는다)"
n_made=0; n_skip=0; n_ver=0
while IFS=$'\t' read -r name group where nb rounds warmup seed prate extra preflight dataset <&3; do
  dataset="${dataset:-mnist}"; dataset="${dataset%$'\r'}"
  [ -n "$name" ] || continue
  preflight="${preflight%$'\r'}"
  have="$(ckpt_count "$name")"
  cmd="DATASET=$dataset NB_CLIENTS=$nb ROUNDS=$rounds WARMUP=$warmup PARTICIPANT_RATE=$prate EXTRA_ARGS=\"$extra\" bash run_real_fuba.sh $name $seed"
  if [ "$FORCE" != "1" ] && [ "$have" -ge "$rounds" ]; then
    echo; echo "== [$group] $name: 있음 ($have/$rounds 라운드) — 생성 생략"
    echo "   ($cmd)"
    n_skip=$((n_skip + 1))
  else
    echo; echo "== [$group] $name  (where=$where, 있음 $have/$rounds 라운드$([ "$have" -gt 0 ] && echo ' → 저자 코드는 이어 돌리기가 없어 처음부터 다시'))"
    if [ "$preflight" != "-" ]; then
      for mod in ${preflight//,/ }; do
        if [ "$DRY" = "1" ]; then echo "   (필요 모듈 확인: python -c 'import $mod')"
        else python -c "import $mod" </dev/null 2>/dev/null || abort "[$name] 파이썬 모듈 없음: $mod (예: pip install $mod). 이 방어 궤적은 그 모듈이 있어야 만들 수 있다"; fi
      done
    fi
    echo "+ $cmd 2>&1 | tee logs/pipeline/$name.attack.log"
    if [ "$DRY" != "1" ]; then
      t0=$(date +%s)
      if ! DATASET="$dataset" NB_CLIENTS="$nb" ROUNDS="$rounds" WARMUP="$warmup" PARTICIPANT_RATE="$prate" EXTRA_ARGS="$extra" \
           bash run_real_fuba.sh "$name" "$seed" </dev/null 2>&1 | tee "logs/pipeline/$name.attack.log"; then
        abort "[$name] 생성 명령이 실패 (mpiexec/main.py). logs/pipeline/$name.attack.log 끝부분을 볼 것"
      fi
      n="$(ckpt_count "$name")"
      [ "$n" -ge "$rounds" ] || abort "[$name] 저장된 라운드 $n/$rounds. logs/pipeline/$name.attack.log 확인"
      echo "   생성 완료: $(( $(date +%s) - t0 )) 초, local 파일 $n 개"
      n_made=$((n_made + 1))
    fi
  fi
  # 검증 (새로 만든 것도, 있던 것도). 파일이 안 바뀌었으면 캐시라 싸다.
  if [ "$DRY" = "1" ]; then
    echo "+ python verify_trajectories.py --only $name --strict --manifest $MANIFEST"
  elif [ "$SKIP_VERIFY" != "1" ]; then
    python verify_trajectories.py --only "$name" --strict --manifest "$MANIFEST" </dev/null \
      || abort "[$name] 검증 실패 — docs/EXPERIMENT_LOG.md 의 '실패 사유' 를 볼 것. 잘못 만들어진 궤적을 쓰면 안 되므로 여기서 멈춘다 (고친 뒤 FORCE=1 ONLY=$name 으로 다시)"
    n_ver=$((n_ver + 1))
  fi
done 3<<< "$LIST"
echo
echo "== 끝. 새로 만듦 $n_made, 있어서 생략 $n_skip, 검증 $n_ver. 기록: docs/EXPERIMENT_LOG.md, logs/trajectories/registry.json (둘 다 커밋할 것)"
