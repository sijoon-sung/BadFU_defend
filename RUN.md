# BadFU / FUBA 새 실행 안내

저장소 루트에서 `python reproduction/run.py`를 사용합니다. 기존 학습용 `.sh` 실행기는 제거했습니다. 공식 소스를 커밋으로 고정해 다시 받았고, 실행할 때 복사본에 저장·호환성 패치를 적용합니다. **기존 로그·체크포인트는 그대로 보존합니다.**

## 1. 환경과 경로

Python 3.11, PyTorch 2.4.1+cu118, torchvision 0.19.1+cu118에서 확인했습니다. 기존 GPU 환경이 있으면 그대로 사용합니다. 새 환경이라면 해당 조합의 PyTorch/torchvision을 먼저 설치하고 다음을 실행합니다.

```bash
python -m pip install -r reproduction/requirements.txt
python reproduction/run.py check
```

Windows에서는 위 requirements가 Intel MPI도 설치합니다. Linux는 시스템 MPI와 `mpi4py`가 필요합니다. FUBA IBA는 **최소 5개 MPI 프로세스**(서버 1 + 작업자 3 + 평가자 1)가 필요합니다. 4개는 협업 학습 중 멈출 수 있어 실행 전에 거부합니다.

아래는 PowerShell 예시입니다. 실제 데이터가 있는 경로로 두 줄만 바꿉니다.

```powershell
$DataRoot = 'D:\datasets'
$RecordRoot = 'D:\BadFU\record\badnet_dataset'
```

- `$DataRoot`: `cifar-10-batches-py/` 또는 `MNIST/`의 상위 폴더. FUBA MNIST 실행은 MNIST, 나머지 아래 실행은 CIFAR-10이 필요합니다. 처음 받을 때만 명령에 `--download`를 추가합니다. MPI 실행 전에 데이터를 한 프로세스로 받아 두는 것을 권장합니다.
- `$RecordRoot`: BadFU 저자 방식으로 만든 `pert_result.pt`와 이 파일이 참조하는 `bd_train_dataset/`, `bd_test_dataset/`, cover 이미지 등이 함께 있는 폴더. **체크포인트나 탐지 로그로 대체할 수 없습니다.** 기존 생성 자료를 재사용하며, 시작할 때 모든 참조 이미지의 존재와 해시를 기록합니다. 이 작업에서는 cover 데이터를 새로 생성하지 않았습니다.
- BadFU의 원본 ImageNet 사전학습 ResNet-18 가중치가 캐시에 없으면 torchvision이 첫 실행에서 받습니다.

## 2. 먼저 짧게 확인

```powershell
python reproduction/run.py badfu --data-root "$DataRoot" --record-root "$RecordRoot" --out artifacts/badfu_smoke --smoke
python reproduction/run.py fuba --data-root "$DataRoot" --dataset mnist --clients 6 --out artifacts/fuba_mnist_smoke --smoke
python reproduction/run.py fuba --data-root "$DataRoot" --dataset cifar10 --clients 6 --non-iid --alpha 0.5 --out artifacts/fuba_cifar_smoke --smoke
```

`--smoke`는 2라운드, 작은 실제 데이터 부분집합을 사용합니다. BadFU는 로컬 1에폭·클라이언트당 32개, FUBA는 클라이언트당 64개·warmup 1·전원 참여입니다. FUBA의 공격 내부 반복 수는 그대로입니다. **이 결과의 ASR은 공격 재현이나 탐지 성능의 근거로 사용하지 않습니다.** CPU는 BadFU `--device cpu`, FUBA `--gpus 0`을 지정합니다.

## 3. 본 학습

```powershell
python reproduction/run.py badfu --data-root "$DataRoot" --record-root "$RecordRoot" --out artifacts/badfu_author_s42 --rounds 40 --seed 42
python reproduction/run.py fuba --data-root "$DataRoot" --dataset mnist --out artifacts/fuba_mnist_s42 --rounds 100 --seed 42
python reproduction/run.py fuba --data-root "$DataRoot" --dataset cifar10 --non-iid --alpha 0.5 --out artifacts/fuba_cifar_s42 --rounds 100 --seed 42
```

BadFU는 원본의 `clean5 → bd5 → cover5 → full6 → FU remove5 → FU remove1` 순서를 유지합니다. `--rounds 40`은 **각 FL 단계가 40라운드**라는 뜻입니다. 공격 단계 하나만 실행했던 기존 실행보다 오래 걸립니다. FU의 원본 인덱싱 때문에 짝수 라운드만 허용합니다. 40라운드 전체 궤적은 약 **64 GB**, 여유 공간은 70 GB 이상을 권장합니다.

FUBA는 기본 20클라이언트·참여율 0.4·warmup 5입니다. `--non-iid --alpha 0.5`는 명시적으로 선택한 Dirichlet 설정이며 논문 기본 조건이라는 주장이 아닙니다. 100라운드 정상 대조군은 새 출력 폴더에서 `--warm-up 100`을 추가합니다. 이 경우에도 개인별 모델을 저장합니다. 작은 alpha에서 클라이언트 표본 수가 원본 배치 크기 64보다 작으면 실행을 중단합니다.

FUBA CIFAR-10 기본 설정의 전체 궤적은 모델 상태만 약 **45 GB + 생성기 상태**입니다. 원본 `unlearn.py`용 pickle도 필요한 경우 `--legacy-export`를 추가하지만, 미참여 클라이언트의 복제 상태까지 내보내므로 디스크 사용량이 크게 늘어납니다. 기본값은 중복 pickle을 만들지 않습니다.

본 학습 시간은 이 코드로 아직 측정하지 않았습니다. BadFU 40R 전체 순서와 FUBA 100R은 장시간 실험으로 별도 실행해야 합니다. **이번에 실행한 것은 축소 검증입니다.** 저장 중 중단된 결과는 보존하되 완료로 처리하지 않습니다. 학습 재개 기능은 없으며 새 실험은 새 `--out`을 사용합니다. 기존 폴더를 덮어쓰거나 누락된 궤적을 자동 재학습하지 않습니다.

## 4. 결과와 궤적 검사

```powershell
python reproduction/run.py verify artifacts/badfu_author_s42/full6
python reproduction/run.py verify artifacts/fuba_mnist_s42/fl
```

각 실행에는 다음이 남습니다.

| 파일 | 내용 |
|---|---|
| `protocol.json` | 원본 커밋, 설정, 환경, 실행 명령, 실행 시간, 완료/실패 상태 |
| `runtime.patch`, `runtime/`, `reproduction/` | 실제 실행한 원본 복사본·패치·어댑터 사본 |
| `console.log`, `metrics*.json` | 원본 평가 로그와 평가 방식이 명시된 수치 |
| `split*.json`, BadFU `record_inputs.json` | 실제 데이터 분할과 입력 기록 |
| `<phase>/initial.pt` | 해당 단계의 진짜 시작 모델 |
| `<phase>/round_XXXX.pt` | 학습 전 `gm`, 참여자별 `cms`, 집계 후 `gm_after`, 참여 ID·가중치, FUBA 생성기 |
| `<phase>/manifest.json`, `verification.json` | 파일 SHA-256, 연속성·유한값·형상·FL 집계 재계산 검사 |

BadFU는 여섯 단계, FUBA는 `fl` 단계가 저장됩니다. **파일 구조 검증 통과와 백도어 활성화 재현은 다른 판정입니다.** BadFU ASR은 원본의 bd_test 전체 분모를 유지합니다. FUBA 원본 평가는 마지막 개별 생성기를 train 모드에서 사용하므로, 저장된 평균 생성기를 eval 모드로 평가한 수치와 섞지 않습니다. 생성기가 없는 공격 라운드의 원본 임의 초기화 평가도 명시합니다.

FUBA의 FU 모듈은 공식 버전으로 복원했지만 이번 지원·검증 범위는 FL 공격 학습과 전체 궤적 저장입니다. 원본 FedEraser에는 `global_epoch=2`, 삭제자 4 고정, IID 재분할이 있어 **100R Non-IID 궤적의 일반 FU 검증기로 바로 사용하지 않습니다.** [원본과 어댑터의 차이 및 검증 범위](docs/FRESH_UPSTREAM.md)를 참고하세요.

## 5. 저장된 BadFU 궤적으로 현재 탐지 실험 연결

새 `full6/manifest.json`을 기존 라벨 없는 관측기에 바로 전달할 수 있습니다. 이 단계는 재학습이나 FU를 실행하지 않습니다.

```powershell
python FUBA/experiment_hrac_unlabeled.py prepare --cifar-root "$DataRoot" --exclude-ids FUBA/logs/hrac/unlabeled_excluded_ids_v1.json --out artifacts/probes/new_pool.npz
python FUBA/experiment_hrac_unlabeled.py run --manifest artifacts/badfu_author_s42/full6/manifest.json --pool artifacts/probes/new_pool.npz --out artifacts/badfu_s42_probes
```

기존 pool이 있으면 첫 줄을 생략합니다. 이 관측기는 BadFU ResNet-18 전용으로, FUBA·FU 단계 manifest를 받으면 거부합니다. [이미지 준비·중간 저장·분석 안내](docs/UNLABELED_PROBES.md) · [실제 축소 검증 기록](reproduction/validation/summary.json).
