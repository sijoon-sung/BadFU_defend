#!/bin/bash
# MPI 런타임(impi-rt) 위치: 현재 python 환경의 Library/bin 을 자동 탐색. MPI_BIN 으로 덮어쓰기 가능.
MPI_BIN="${MPI_BIN:-$(cygpath -u "$(python -c 'import sys,os;print(os.path.join(sys.prefix,"Library","bin"))')")}"
export PATH="$MPI_BIN:$PATH"
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
NAME="${1:-det1}"; SEED="${2:-522}"
PYTHONIOENCODING=utf-8 mpiexec -genv OPENBLAS_NUM_THREADS 1 -genv OMP_NUM_THREADS 1 -genv MKL_NUM_THREADS 1 -n 5 python main.py \
  --dataset=mnist --attackMethod=iba --name="$NAME" --seed="$SEED" --n_gpu=1 --save \
  --no_global_test --no_detail_test \
  --nb_clients=8 --participant_rate=1.0 --communication_rounds=8 --warm_up=3
