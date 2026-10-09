#!/bin/bash
cd "$(dirname "$0")"
# 저장소에 .venv (CUDA PyTorch) 가 있으면 그걸 쓴다
[ -x ../.venv/Scripts/python.exe ] && export PATH="$(cd .. && pwd)/.venv/Scripts:$PATH"
# MPI 런타임(impi-rt) 위치: MPI_BIN 으로 직접 주거나, 지금 python 에 설치된 impi-rt 에서 자동으로 찾는다.
if ! command -v mpiexec >/dev/null 2>&1; then
  MPI_BIN="${MPI_BIN:-$(python -c "import importlib.metadata as m,os;d=m.distribution('impi-rt');print(os.path.dirname(next(str(d.locate_file(f)) for f in d.files if str(f).endswith('mpiexec.exe'))))" 2>/dev/null)}"
  [ -n "$MPI_BIN" ] || MPI_BIN="/c/Users/DISLAB/AppData/Local/Programs/Python/Python312/Library/bin"
  export PATH="$(cygpath -u "$MPI_BIN" 2>/dev/null || echo "$MPI_BIN"):$PATH"
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
NAME="${1:-det1}"
NB_CLIENTS="${NB_CLIENTS:-8}"
EXTRA_ARGS="${EXTRA_ARGS:-}"   # 예: EXTRA_ARGS="--non_iid --non_iid_type class_owner --owner_class 3"
PYTHONIOENCODING=utf-8 mpiexec -genv OPENBLAS_NUM_THREADS 1 -genv OMP_NUM_THREADS 1 -genv MKL_NUM_THREADS 1 -n 5 python main.py \
  --dataset=mnist --attackMethod=iba --name="$NAME" --n_gpu=1 --save \
  --no_global_test --no_detail_test \
  --nb_clients="$NB_CLIENTS" --participant_rate=1.0 --communication_rounds=8 --warm_up=3 $EXTRA_ARGS
