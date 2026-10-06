#!/bin/bash
export PATH="/c/Users/DISLAB/AppData/Local/Programs/Python/Python312/Library/bin:$PATH"
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
NAME="${1:-det1}"
PYTHONIOENCODING=utf-8 mpiexec -genv OPENBLAS_NUM_THREADS 1 -genv OMP_NUM_THREADS 1 -genv MKL_NUM_THREADS 1 -n 5 python main.py \
  --dataset=mnist --attackMethod=iba --name="$NAME" --n_gpu=1 --save \
  --no_global_test --no_detail_test \
  --nb_clients=8 --participant_rate=1.0 --communication_rounds=8 --warm_up=3
