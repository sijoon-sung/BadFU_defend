#!/bin/bash
# 탐지 총정리: 있는 궤적마다 detect_bench.py 를 돌리고 조건별로 집계한다 (학습 없음, 궤적당 수 분).
# 사용: bash run_detect_bench.sh                       # k8 (bd_iid/owner_s*) + k20 논문 설정 (bd_*_r100w5_k20paper_s*) 중 있는 것 전부
#       GROUPS="k20paper" bash run_detect_bench.sh      # 일부만 (k8 | k20paper | tdef)
#       FORCE=1 ...  결과 다시 계산
# 결과: logs/detect/{name}.json, logs/detect/summary_{cond}.md
set -e
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
DO_GROUPS="${DO_GROUPS:-${GROUPS:-k8 k20paper}}"     # GROUPS= 또는 DO_GROUPS= 둘 다 됨
FORCE="${FORCE:-0}"
run_one() {   # name K rounds cond owner
  local name="$1" K="$2" R="$3" cond="$4" owner="$5"
  [ "$(ls checkpoints/local_net_model_"$name"_round_*.pkl 2>/dev/null | wc -l | tr -d ' ')" -ge "$R" ] || { echo "  [$name] 궤적 없음 — 건너뜀"; return 0; }
  if [ "$FORCE" != "1" ] && [ -f "logs/detect/$name.json" ]; then echo "  [$name] 결과 있음"; return 0; fi
  python detect_bench.py --fmt fuba --name "$name" --K "$K" --rounds "$R" --defender 4 --attackers 0 1 2 3 --owner_client "$owner" --cond "$cond"
}
for g in $DO_GROUPS; do
  case "$g" in
    k8)
      for s in 0 1 2; do run_one "bd_iid_s$s" 8 8 iid -1; run_one "bd_owner_s$s" 8 8 owner 7; done
      python aggregate_detect_bench.py --cond iid; python aggregate_detect_bench.py --cond owner ;;
    k20paper)
      for s in 0 1 2; do run_one "bd_iid_r100w5_k20paper_s$s" 20 100 iid_k20paper -1; run_one "bd_owner_r100w5_k20paper_s$s" 20 100 owner_k20paper 19; done
      python aggregate_detect_bench.py --cond iid_r100w5_k20paper; python aggregate_detect_bench.py --cond owner_r100w5_k20paper ;;
    tdef)
      for d in median multi_krum flame flDetector; do for s in 0 1 2; do run_one "bd_iid_r100w5_k20paper_${d}_s$s" 20 100 "tdef_$d" -1; done; python aggregate_detect_bench.py --cond "iid_r100w5_k20paper_$d"; done ;;
    tdef_reduced)
      for d in median multi_krum; do for s in 0 1 2; do run_one "bd_iid_${d}_s$s" 8 8 "iid_$d" -1; done; python aggregate_detect_bench.py --cond "iid_$d"; done ;;
    *) echo "모르는 그룹: $g"; exit 1 ;;
  esac
done
echo "== 끝. 올릴 것: git add FUBA/logs/detect"
