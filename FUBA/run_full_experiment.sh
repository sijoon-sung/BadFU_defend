#!/bin/bash
# 전체 절차: 진짜 FL(공격) -> 진짜 FU(언러닝) -> 탐지 -> 정화 -> 평가.
# 근사 없이 FUBA 공개 코드의 공격·언러닝을 실제로 실행한다.
#
# 사용법:  bash run_full_experiment.sh <NAME> <SEED> [FU_METHOD]
#   예:    bash run_full_experiment.sh run1 1 retrain
#   FU_METHOD: retrain | fedEraser | sga | pgd | distillation  (기본 retrain)
#
# 산출물: checkpoints/*_{NAME}_round_*.pkl (공격 궤적),
#         saved_models/global_{FU_METHOD}_iba_mnist.pkl (진짜 언러닝 모델),
#         logs/pipeline_{NAME}.json (탐지/정화), eval_logs/purify_unlearned.json (진짜FU 정화)
set -e
cd "$(dirname "$0")"
NAME="${1:-run1}"; SEED="${2:-1}"; FU="${3:-retrain}"
K=8; CR=8; WU=3; REQ=4; TGT=8
MPI_BIN="${MPI_BIN:-$(cygpath -u "$(python -c 'import sys,os;print(os.path.join(sys.prefix,"Library","bin"))')")}"
export PATH="$MPI_BIN:$PATH"

echo "########## 1) 진짜 FL + 공격 (FUBA main.py, mpiexec) ##########"
echo "+ mpiexec -n 5 python main.py --dataset=mnist --attackMethod=iba --name=$NAME --seed=$SEED --save ..."
bash run_real_fuba.sh "$NAME" "$SEED"

echo "########## 2) 진짜 FU = 삭제 요청 실행 (FUBA unlearn.py, method=$FU) ##########"
echo "+ python unlearn.py --name=$NAME --method=$FU --forgot_client_idx=$REQ ..."
PYTHONIOENCODING=utf-8 python unlearn.py --dataset mnist --name "$NAME" --method "$FU" --attackMethod iba --n_gpu 1 \
  --round $CR --nb_clients $K --warm_up $WU --communication_rounds $CR --forgot_client_idx $REQ \
  --participant_rate 1.0 --target_label $TGT --no_global_test --no_detail_test
UNLEARNED="saved_models/global_${FU}_iba_mnist.pkl"

echo "########## 3) 탐지 (저장된 업데이트만, 라벨/트리거 없이) ##########"
echo "+ python pipeline_detect_purify.py --name=$NAME"
PYTHONIOENCODING=utf-8 python pipeline_detect_purify.py --name "$NAME" | grep -v "^Files already"

echo "########## 4) 정화 = 삭제는 이행하고 백도어 방향만 제거 (진짜 언러닝 모델에 적용) ##########"
echo "+ python eval/purify_unlearned.py --name=$NAME --unlearned=$UNLEARNED"
PYTHONIOENCODING=utf-8 python eval/purify_unlearned.py --name "$NAME" --unlearned "$UNLEARNED" | grep -v "^Files already"

echo "########## 끝. 요약 ##########"
echo "  탐지: logs/pipeline_${NAME}.json   정화(진짜FU): eval_logs/purify_unlearned.json"
echo "  (비교군·통계·오류스윕은 run_experiment.sh 와 eval/ 참조)"
