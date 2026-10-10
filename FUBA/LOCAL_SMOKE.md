# 로컬 GPU 스모크 실행 기록 (2026-10-10)

두 공격 코드(FUBA, BadFU)가 이 PC(RTX 4060 Laptop 8GB, Windows 11)에서 끝까지 돌아가고, 그 실제 궤적 위에서 분석 스크립트가 도는지 확인한 기록. 수치는 1회 실행값이다.

## 환경

- Python: `C:\Users\sijoo\.conda\envs\pytorch\python.exe` (3.11, torch 2.14.1+cu126, torchvision 0.29.1+cu126, mpi4py 4.1.2, impi-rt 2021.18, psycopg2-binary, timm, scipy, sklearn, pandas, matplotlib, imageio). 추가 설치 없음.
- MPI: `C:\Users\sijoo\.conda\envs\pytorch\Library\bin\mpiexec.exe`. `run_real_fuba.sh` 가 `python` 이 가리키는 env 의 impi-rt 에서 자동으로 찾으므로 Git Bash 에서 먼저
  ```bash
  export PATH="/c/Users/sijoo/.conda/envs/pytorch:/c/Users/sijoo/.conda/envs/pytorch/Scripts:/c/Users/sijoo/.conda/envs/pytorch/Library/bin:$PATH"
  export PYTHONIOENCODING=utf-8
  ```
- DB: `--no_global_test --no_detail_test` 를 주면 `sql.py` 는 import 만 되고 접속은 하지 않는다 (psycopg2-binary 만 있으면 됨). 코드 수정 없음.
- ImageNet 가중치(`resnet18-f37072fd.pth`, `convnext_tiny-983f1562.pth`)는 `~/.cache/torch/hub/checkpoints/` 에 미리 넣어 두었다(회선이 느려 다운로드 대신). CIFAR-10 tar 도 같은 이유로 `BadFU_src/BadFU-main/data/` 와 `BadFU_src/BadFU-main/data_prepare/data/cifar10/` 에 복사.

## Part A — FUBA (MNIST, IBA), `FUBA/`

연구실 기본 설정 그대로: 클라이언트 8(공격자 0~3, 요청자 4, 정상 5~7), 8 라운드(웜업 3), 전원 참여, train_epoch 상한 없음, `--n_gpu 1`.

```bash
cd FUBA
NB_CLIENTS=8 ROUNDS=8 WARMUP=3 EXTRA_ARGS="--split_seed 522" bash run_real_fuba.sh loc_fuba 522
python benign_departure.py --name loc_fuba --K 8 --rounds 8 --seed 522 --split_seed 522 --requesters 4 5 --methods subtract distillation fedEraser retrain_benign
python latent_risk.py --name loc_fuba --rounds 8 --warmup 3 --seed 522
python robust_fu.py --name loc_fuba --K 8 --rounds 8 --seed 522 --split_seed 522 --requesters 4 5 --arms plain purify_flag
python release_map.py --fmt fuba --name loc_fuba --K 8 --rounds 8 --requester 4 --delta
```

| 단계 | 벽시계 |
|---|---|
| `run_real_fuba.sh` (mpiexec -n 5, 8 라운드) | 349 s |
| `benign_departure.py` (4 방법 × 요청자 2) | 264 s (retrain_benign 이 요청자당 ~95 s) |
| `latent_risk.py` | 15 s |
| `robust_fu.py` (2 arm × 요청자 2) | 80 s |
| `release_map.py --fmt fuba --delta` | 25 s |

산출물: `checkpoints/{global,local,backdoor}_net_model_loc_fuba_round_{1..8}.pkl` (backdoor 는 4~8), 콘솔 라운드별 ACC/ASR:

```
global acc 33.12% asr None%      (r1)
global acc 68.27% asr None%
global acc 80.31% asr None%
global acc 84.5%  asr 14.70%     (r4, 공격 시작)
global acc 88.79% asr 13.51%
global acc 91.9%  asr 9.40%
global acc 93.76% asr 9.43%
global acc 94.81% asr 8.93%      (r8, 잠복 상태; EXPERIMENT.md 의 95.95 / 5.69 와 같은 급)
```

`benign_departure.py` (ASR, 괄호 ACC):
```
== loc_fuba  before: ACC 94.81 / ASR 9.17
method                      req4            req5
subtract            63.2 ( 74.9)    22.6 ( 92.2)
distillation        37.5 ( 81.4)    13.7 ( 93.6)
fedEraser            4.8 ( 92.5)     4.0 ( 88.9)
retrain_benign       0.6 ( 97.5)     0.6 ( 97.4)
```

`latent_risk.py`:
```
client      role  sigma1  effrank    conc  consist  cosBVgen cosBVnon8
     0  attacker   0.057    97.00   0.394    0.533    +0.039    +0.018
     1  attacker   0.051    99.43   0.380    0.553    +0.025    +0.007
     2  attacker   0.053    99.45   0.385    0.542    +0.029    +0.010
     3  attacker   0.049    98.23   0.389    0.568    +0.039    +0.018
     4  defender   0.044    96.74   0.234    0.693    +0.034    +0.036
     5    benign   0.050    90.20   0.181    0.535    -0.013    +0.007
     6    benign   0.049    89.20   0.178    0.513    +0.018    +0.027
     7    benign   0.051    88.37   0.181    0.531    -0.014    +0.003
AUROC (mal=attackers+defender vs benign | defender vs rest), defender rank (desc):
  shape_sigma1_share     0.600 | 0.000   def rank 8/8  att ranks [1, 4, 2, 6]
  shape_eff_rank         1.000 | 0.429   def rank 5/8  att ranks [4, 2, 1, 3]
  shape_concentration    1.000 | 0.429   def rank 5/8  att ranks [1, 4, 3, 2]
  shape_consistency      0.933 | 1.000   def rank 1/8  att ranks [6, 3, 4, 2]
  cos_U_generic          1.000 | 0.714   def rank 3/8  att ranks [2, 5, 4, 1]
  cos_U_non_target       0.667 | 1.000   def rank 1/8  att ranks [4, 7, 5, 3]
  abs_cos_U_generic      1.000 | 0.714   def rank 3/8  att ranks [2, 5, 4, 1]
  cos_U_fc3_generic      0.467 | 0.000   def rank 8/8  att ranks [2, 1, 7, 5]
```

`robust_fu.py`:
```
== loc_fuba  before: ACC 94.8 / ASR 9.11   (ASR, 괄호 ACC, f=flag 라운드 수)
arm                             req4                req5
plain                 6.1 ( 92.4) f2      3.5 ( 89.1) f1
purify_flag           4.9 ( 91.9) f1      7.7 ( 88.3) f1
== loc_fuba  MIA 망각 (요청자 AUROC / 멤버율@비멤버 FPR 5%; 대조군 정상 클라 AUROC 는 괄호)
arm                             req4                req5
before            0.487/0.031 (0.49)  0.492/0.042 (0.51)
plain             0.494/0.036 (0.49)  0.488/0.034 (0.49)
purify_flag       0.488/0.045 (0.49)  0.497/0.039 (0.51)
```

`release_map.py --fmt fuba --delta`:
```
[loc_fuba] K=8 probe=500 before={'acc': 94.8, 'asr': 9.11}
  client 0 (attacker ) snap 0.50  final changed 0.016 conc 0.50 -> 3  | δ: snap 0.33 conc 0.30 -> 3  | ASR ρ=0..1: [9.11, 8.39, 7.7, 7.05, 6.44, 5.94]
  client 1 (attacker ) snap 0.50  final changed 0.018 conc 0.22 -> 2  | δ: snap 0.43 conc 0.44 -> 4  | ASR ρ=0..1: [9.11, 7.81, 6.76, 5.76, 4.97, 4.27]
  client 2 (attacker ) snap 0.50  final changed 0.022 conc 0.18 -> 3  | δ: snap 0.50 conc 0.27 -> 4  | ASR ρ=0..1: [9.11, 8.23, 7.58, 6.86, 6.27, 5.78]
  client 3 (attacker ) snap 0.50  final changed 0.020 conc 0.20 -> 0  | δ: snap 0.50 conc 0.25 -> 4  | ASR ρ=0..1: [9.11, 7.97, 7.02, 6.16, 5.39, 4.87]
  client 4 (requester) snap 0.47  final changed 0.244 conc 0.98 -> 8  | δ: snap 0.43 conc 0.99 -> 8  | ASR ρ=0..1: [9.11, 14.53, 22.33, 34.06, 48.45, 62.92]
  client 5 (benign   ) snap 0.52  final changed 0.056 conc 0.82 -> 8  | δ: snap 0.42 conc 0.93 -> 8  | ASR ρ=0..1: [9.11, 10.89, 12.99, 15.5, 18.53, 22.38]
  client 6 (benign   ) snap 0.56  final changed 0.034 conc 0.94 -> 8  | δ: snap 0.31 conc 0.93 -> 8  | ASR ρ=0..1: [9.11, 10.45, 12.0, 13.81, 15.89, 18.27]
  client 7 (benign   ) snap 0.42  final changed 0.048 conc 1.00 -> 8  | δ: snap 0.37 conc 0.96 -> 8  | ASR ρ=0..1: [9.11, 10.94, 13.0, 15.44, 17.81, 21.67]
요청자 4 순위: snap 7/8 (z -0.83), final_mass 1/8 (z 11.57), delta_snap 3/8 (z 0.3), delta_final_mass 1/8 (z 10.05)
```

FUBA 쪽 코드 수정: 없음. (CPU 용 배치 상한/FUBA_FAST 는 결국 넣지 않았다. 참고로 `train_main.py` 의 로컬 학습은 `--train_epoch` 가 아니라 `train_method_iba.py` 의 상수 `TRAIN_EPOCH = 20` 배치에서 끊긴다.)

## Part B — BadFU (CIFAR-10, ResNet-18), `BadFU_src/BadFU-main/`

저자 준비 절차(`prepare_data.sh` 의 badnet → UBA-Inf)를 그대로, 단 `regen_and_detect.sh` 처럼 파괴적 rm/mv 는 생략. 모두 GPU.

```bash
cd BadFU_src/BadFU-main/data_prepare && mkdir -p record
# (1) BadNet 트리거 + cover 데이터셋 (학습 0 에폭)
python ./attack/badnet.py --yaml_path ../config/attack/prototype/cifar10.yaml --save_folder_name badnet_dataset \
    --add_cover 1 --epochs 0 --pratio 0.017 --cratio 0.017 --attack_target 0 --model resnet18 --device cuda:0 --num_workers 0
# (2) UBA-Inf 영향함수 기반 cover 섭동 (convnext_tiny 대리 모델, ft 1 에폭, ap 1 에폭)
python ./uba/uba_inf_cover.py --dataset_folder ../record/badnet_dataset --device cuda:0 --ft_epochs 1 --ap_epochs 1 --num_workers 0
cd .. && cp -r data_prepare/record record
# (3) 탐지 FL (6 라운드, 로컬 1 에폭, 클라 데이터 전체, 궤적 보존)
python fl_detect_badfu.py --rounds 6 --local_epochs 1 --keep_traj
# (4) 보존 궤적으로 언러닝 활성화 측정
python unlearn_from_traj.py --traj_dir logs/traj --local_epochs 1
# (5) release map (release_map.py 는 FUBA/ 로 chdir 하므로 절대 경로)
python ../../FUBA/release_map.py --fmt badfu --name loc_badfu --traj_dir "$PWD/logs/traj" --data_root "$PWD/data" \
    --rounds 6 --requester 5 --attackers 0 --target 0 --probe 500 --delta --counts 10545 9647 9641 9649 9668 10545
```

`--model resnet18`: yaml 기본 `preactresnet18` 은 `pretrained=True` 경로에서 저장소에 없는(`*.pt` 는 gitignore) `resource/trojannn/clean_preactresnet18.pt` 를 읽으려 한다. 0 에폭이라 모델은 데이터 생성에 영향이 없다. `--device cuda:0`: yaml 기본은 `cuda:2`.

| 단계 | 벽시계 |
|---|---|
| (1) badnet.py (bd 850 / cv 850 / bd_test 9000 PNG) | 37 s |
| (2) uba_inf_cover.py | 663 s (대리 모델 미세조정 → s_test(LiSSA 50) → 영향 계산 → PGD 40 step; cover 영향 −1.314 → −1.355) |
| (3) fl_detect_badfu.py 6 라운드 + FedEraser 언러닝 3 step | 511 s (스크립트 내부 479 s) |
| (4) unlearn_from_traj.py (FedEraser 3 step) | 185 s |
| (5) release_map.py --fmt badfu --delta | 64 s |

(3) 콘솔:
```
client_data_counts = [10545, 9647, 9641, 9649, 9668, 10545]
[r00] acc=52.05 asr(dormant)= 12.01 cos(0,5)=+1.000 fc(0,5)=+0.859 cancel(0,5)=0.000
[r01] acc=67.91 asr(dormant)=  4.73 cos(0,5)=+1.000 fc(0,5)=+0.857 cancel(0,5)=0.000
[r02] acc=74.18 asr(dormant)=  4.90 cos(0,5)=+1.000 fc(0,5)=+0.852 cancel(0,5)=0.000
[r03] acc=77.34 asr(dormant)=  4.78 cos(0,5)=+1.000 fc(0,5)=+0.830 cancel(0,5)=0.000
[r04] acc=79.09 asr(dormant)=  6.96 cos(0,5)=+1.000 fc(0,5)=+0.866 cancel(0,5)=0.000
[r05] acc=80.69 asr(dormant)=  8.28 cos(0,5)=+1.000 fc(0,5)=+0.889 cancel(0,5)=0.000
dormant theta_T : acc=80.69 asr=  8.28
after unlearn req(c5): acc=76.91 asr=  6.72
requester(c5) dev vs normals: cos=+1.90s fc=+1.33s cancel=+0.98s
```
(6 라운드 × 1 에폭이라 모델이 덜 수렴했고 언러닝 후 ASR 점프는 없다. 저자 설정 15 라운드 × 5 에폭은 라운드당 ~3.5 분 → 이 GPU 에서 약 1 시간.)

(4) `unlearn_from_traj.py`:
```
rounds available: 6
dormant theta_T (glob_5): acc=80.69 asr=8.28
  unlearn step r0: acc=68.44 asr=4.77
  unlearn step r2: acc=73.88 asr=5.58
  unlearn step r4: acc=77.05 asr=6.54
dormant        : acc=80.69 asr=8.28
요청자(c5) 언러닝후: acc=77.05 asr=6.54
ASR 8.3 -> 6.5  (변화 작음)
```
(`logs/unlearn_activation.json` 을 덮어쓰므로 — 이 파일은 git 에 추적됨 — 실행 뒤 `git checkout` 으로 되돌렸다.)

(5) `release_map.py --fmt badfu --delta` (badfu 는 ASR 채점 없음):
```
[loc_badfu] K=6 probe=500 before={}
  client 0 (attacker ) snap 0.88  final changed 0.884 conc 1.00 -> 0  | δ: snap 0.88 conc 1.00 -> 0
  client 1 (benign   ) snap 0.27  final changed 0.472 conc 0.60 -> 4  | δ: snap 0.24 conc 0.64 -> 4
  client 2 (benign   ) snap 0.22  final changed 0.394 conc 0.59 -> 3  | δ: snap 0.24 conc 0.59 -> 3
  client 3 (benign   ) snap 0.29  final changed 0.428 conc 0.40 -> 3  | δ: snap 0.29 conc 0.37 -> 3
  client 4 (benign   ) snap 0.25  final changed 0.452 conc 0.35 -> 0  | δ: snap 0.24 conc 0.38 -> 4
  client 5 (requester) snap 0.28  final changed 0.410 conc 0.35 -> 4  | δ: snap 0.30 conc 0.37 -> 4
요청자 5 순위: snap 3/6 (z -0.41), final_mass 6/6 (z -0.75), delta_snap 2/6 (z -0.3), delta_final_mass 6/6 (z -0.76)
```
출력은 `FUBA/logs/release_map/loc_badfu.json`.

### 궤적 파일 형식 (`logs/traj/`, `fl_detect_badfu.py` 저장)

- `round_{r}.pt` (r = 0..R−1, 각 313 MB): `dict` with keys `["gm", "cms"]`
  - `gm`: 라운드 시작 글로벌 `state_dict` (`dict`, 122 키, `model.conv1.weight` (64,3,3,3) float32 … `model.fc.weight` (10,512); BN `num_batches_tracked` 는 int64), CPU 텐서. float 파라미터 11,183,562 개.
  - `cms`: `list`, 길이 K=6, **클라이언트 인덱스 순** (0=공격자 clean_0+bd, 1~4 정상, 5=요청자 clean_0+cv). 원소는 그 클라의 로컬 학습 결과 `state_dict` (gm 과 같은 122 키, CPU).
- `glob_{r}.pt` (각 45 MB): `dict` with key `["gm_after"]` = 라운드 r 의 FedAvg(가중 = client_data_counts) 결과 `state_dict`.
- FedAvg 가중치용 데이터 수: `[10545, 9647, 9641, 9649, 9668, 10545]` (스크립트가 이제 `client_data_counts = ...` 로 출력).

## 코드 변경 (이 커밋)

- `BadFU_src/BadFU-main/data_prepare/utils/save_load_attack.py`, `data_prepare/uba/uba_utils/basic_utils.py`: `torch.load(..., weights_only=False)`. torch ≥ 2.6 기본값 변경으로 `attack_result.pt`(numpy 스칼라가 든 데이터셋 상태) 로드가 실패하던 것. 없으면 UBA 단계가 시작조차 못 한다.
- `BadFU_src/BadFU-main/fl_detect_badfu.py`: 선택 인자 `--max_per_client`, `--test_n`(축소 설정용 고정 시드 부분집합; 기본 0 = 전체), `--keep_traj`(끝나고 `logs/traj/*.pt` 를 지우지 않음; `unlearn_from_traj.py` 에 넘기려면 필요), `client_data_counts` 출력. 기본값이면 동작은 종전과 같다.
- `BadFU_src/BadFU-main/unlearn_from_traj.py`: 하드코딩이던 `logs/traj_keep`, 5 에폭을 `--traj_dir`, `--local_epochs`(+ 위와 같은 `--max_per_client`, `--test_n`) 인자로. 기본값 동일.
- FUBA 쪽은 변경 없음. 설치만: (처음 CPU 시도용 venv 에 mpi4py/impi-rt/psycopg2-binary/pandas/matplotlib/imageio 를 넣었으나 최종 실행은 위 conda env, 추가 설치 없음.)

실행 산출물(`checkpoints/`, `record/`, `data/`, `logs/traj/`, `logs/*/loc_fuba.json`, `logs/badfu_detect.json`)은 커밋하지 않았다(대부분 gitignore).
