#!/bin/bash
# BadFU 데이터 재생(파괴적 rm/mv 없이) -> 경로 검증 -> 탐지 FL 실행
set -e
cd "$(dirname "$0")"
ROOT="$(pwd)"
LOG="$ROOT/logs/badfu_detect_run.log"
mkdir -p "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1
echo "==== regen_and_detect start $(date) ===="

cd "$ROOT/data_prepare"
echo "[1/4] reset record"
rm -rf record
mkdir -p record    # badnet.py 의 os.mkdir 는 부모를 안 만든다
echo "[2/4] badnet (poisoned dataset, no training)"
PYTHONIOENCODING=utf-8 python ./attack/badnet.py --yaml_path ../config/attack/prototype/cifar10.yaml \
  --save_folder_name badnet_dataset --add_cover 1 --epoch 00 --pratio 0.017 --cratio 0.017 --attack_target 0
echo "[3/4] uba influence camouflage (num_workers 0, NO rm/mv)"
PYTHONIOENCODING=utf-8 python ./uba/uba_inf_cover.py --dataset_folder ../record/badnet_dataset \
  --device cuda:0 --ft_epoch 1 --ap_epochs 1 --num_workers 0
# 파괴적 rm/mv 는 일부러 생략 -> cv_pert / bd 경로가 그대로 살아있게 한다
cp -r record "$ROOT/record"

echo "[verify] path resolution"
cd "$ROOT"
PYTHONIOENCODING=utf-8 python - <<'PY'
import torch, os
d=torch.load("record/badnet_dataset/pert_result.pt", map_location="cpu", weights_only=False)
def chk(name, dd):
    ks=list(dd.keys()); miss=sum(0 if os.path.exists(dd[k]['path']) else 1 for k in ks)
    print(f"  {name}: miss={miss}/{len(ks)}")
    return miss
m=0
m+=chk("bd_train.bd", d['bd_train']['bd_data_container']['data_dict'])
m+=chk("cv_pert",     d['cv_pert']['data_dict'])
m+=chk("bd_test",     d['bd_test']['bd_data_container']['data_dict'])
print("TOTAL MISS:", m)
PY

echo "[4/4] run detection FL"
cd "$ROOT"
PYTHONIOENCODING=utf-8 python fl_detect_badfu.py --rounds 15 --local_epochs 5
echo "==== regen_and_detect done $(date) ===="
