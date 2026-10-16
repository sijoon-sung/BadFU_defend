# BadFU / FUBA 새 실행 안내

저장소 루트에서 `python reproduction/run.py`를 사용합니다. 기존 학습용 `.sh` 실행기는 제거했습니다. 공식 소스를 커밋으로 고정해 다시 받았고, 실행할 때 복사본에 저장·호환성 패치를 적용합니다. **기존 로그·체크포인트는 그대로 보존합니다.**

## HRAC의 검증 기준: 같은 요청에 두 채널 적용

공격 이름에 따라 검사법을 나누지 않습니다. **BadFU와 FUBA 모두 동일한 `hrac_request_gate.py`에서 이력 H와 삭제 요청 U를 함께 평가합니다.**

- **Inter:** 요청한 삭제가 다른 참여자의 기여와 이루던 선택적 상쇄를 해제하는지 검사합니다.
- **Intra:** 독립적으로 관측한 요청 기여를 요청자 자신의 과거 원자 기여로 설명할 수 있는지 검사합니다. 정상 부분 삭제를 전체 클라이언트 기여와 비례한다고 가정하지 않습니다. 소유권·원자별 삭제 가능 범위·삭제량 제약을 둔 최소 잔차를 사용합니다.
- 두 채널은 서로 대체하지 않습니다. 내부적으로 설명 가능한 요청도 inter 위험이 클 수 있습니다. 반대로 intra 불일치만으로 악성 의도를 증명하지 않습니다. 정상 보정 실행이 없으면 종합 경보는 `null`입니다.

이전 BadFU 검증 ZIP만 있어도 같은 게이트에 연결할 수 있습니다. 압축을 푼 `evidence` 폴더를 지정합니다. 모델 가중치·재학습·FU는 필요하지 않습니다.

```bash
python reproduction/validate_hrac_requests.py --evidence /압축푼/경로/evidence --out artifacts/hrac_badfu_requests_v1
```

FUBA에서 내보낸 `evidence`에도 **같은 명령**을 사용합니다. 독립 요청 자료가 없는 기본 실행은 각 클라이언트 전체 삭제 요청을 사전 검토하며 inter를 계산합니다. **intra는 `unobserved_independent_request_effect`로 남고 HRAC 전체 검증 완료로 보고하지 않습니다.** 이전 파일에 없던 정보를 복사·분할해서 만들어 넣지 않습니다.

### 저장된 FUBA 궤적에서 관측 및 요청 검사

```bash
python reproduction/validate_fuba_detection.py --run-dir artifacts/fuba_mnist_s42
python reproduction/validate_fuba_detection.py --run-dir artifacts/fuba_cifar_s42
```

완료한 실험에 해당하는 줄만 실행합니다. 이 명령은 새 실행기가 저장한 `full-trajectory-v1` 궤적을 읽습니다. 예전 압축 이력이나 pickle을 새 전체 궤적으로 잘못 해석하지 않습니다. 데이터 경로를 옮겼으면 `--data-root /실제/데이터/상위폴더`, 이미 준비한 무라벨 풀은 `--pool /실제/pool.npz`를 사용합니다.

MNIST/CNN과 CIFAR-10/ResNet의 **원본 전처리(ToTensor만 적용)**를 사용하고, 실제 참여자 ID와 라운드별 균등 집계 가중치를 반영합니다. 미참여자는 그 라운드 집계에 기여하지 않은 것으로 기록하며, 한 번도 참여하지 않은 클라이언트는 순위에서 제외합니다. 관측은 고정 입력 80장×3세트와 패턴 3개×4모서리를 사용합니다. FUBA의 트리거 생성기와 클래스 라벨은 탐지 입력에 쓰지 않습니다. 따라서 입력별로 달라지는 FUBA 트리거를 이 고정 패턴이 충분히 관측하는지는 실험 대상입니다.

`<run-dir>/hrac_validation_v1/fuba_hrac_evidence_<식별값>.zip`에 두 채널 상태, HRAC·기준선 순위, 참여 횟수, 원래 ASR 로그와 재계산용 관측 평균이 들어갑니다. 워밍업의 ASR 없음과 임의 생성기를 사용한 ASR을 실제 공격 ASR로 해석하지 않습니다. 같은 명령으로 관측을 재개할 수 있으며, 코드·관측 설정 변경 시 새 `--out`을 사용합니다.

### Intra를 실제로 측정하려면 필요한 자료

관측 실행이 생성하는 `request_coordinates.json`의 `observation_id`를 기준으로 독립 요청 자료를 묶습니다. `--requests /경로/requests.json`을 위 두 검사기에 추가합니다. 같은 요청 자료의 결과를 기존 정상/공격 점수로 바꿔 적지 않습니다.

요청 JSON의 최상위 필드는 `schema: 1`, `origin`, `observation_id`, `requests`입니다. 각 요청은 다음 정보를 가집니다.

| 필드 | 의미 |
|---|---|
| `id`, `round`, `removal` | 요청 ID, 해당 요청까지의 라운드, 클라이언트별 삭제 비율 |
| `intra.atoms` | `[원자 수, 360]` 형태의 과거 기여. 좌표는 반복 3×변환 12×출력 10 순서. 원자 기여에는 집계 가중치가 반영되어야 함 |
| `intra.observed` | 같은 좌표계에서 **독립적으로 관측한** 삭제 요청 기여 360차원. 부호는 제거할 대상의 양의 학습 기여이며, 실제 FU가 적용하는 음의 모델 변위와 혼동하지 않음 |
| `intra.atom_owners`, `intra.atom_masses` | 각 원자의 소유 클라이언트와 양의 표본 수/질량 |
| `intra.lower`, `intra.upper` | 요청 메타데이터에서 정한 원자별 삭제 계수 범위. 다른 소유자의 상한은 0 |
| `intra.constraint_matrix`, `intra.constraint_value` | 선택 항목. 정확한 ID·클래스별 삭제량 등 추가 등식 제약 |
| `intra.provenance` | `independent_observation: true`, 정확한 `observation_id`, `request_round`, 서로 다른 `history_source`·`request_source`, `constraints_source: "request_metadata"`, `complete_within_requested_owners: true` |

검사기는 소유자별 `sum(atom_mass × 삭제계수) = 요청 삭제비율 × 해당 소유자의 전체 atom_mass`를 추가로 강제합니다. 전체 탈퇴라면 자신의 원자를 임의로 남겨 잔차를 줄일 수 없습니다. 메타데이터 선언 자체는 독립 관측의 암호학적 증명이 아니므로 원본 관측 경로를 함께 보존해야 합니다.

**현재 기록한 클라이언트 전체 체크포인트만으로는 내부 샘플별 원자와 독립 요청 관측이 자동으로 생기지 않습니다.** 이 자료가 없는 실행은 inter까지만 실제 측정한 것으로 표시합니다. 저장 코드가 잘못됐다는 뜻은 아닙니다. 세부 수식과 필요한 후속 실험은 [연구 정리](docs/RESEARCH_SUMMARY.md)를 따릅니다.

참고: 지금 저장소의 [FUBA](https://github.com/stcebra/FUBA)는 Adv-attacker/Adv-defender 공모 방식입니다. [FedUBA](https://www.chenwang.net.cn/publications/FedUBA-TDSC26.pdf)는 별도 연구입니다. `FUBA=intra 전용`으로 대응시키지 않습니다.

## 학습을 마쳤으면: 탐지 알고리즘 검증

**완료된 `badfu_author_s42`가 있는 컴퓨터에서, 학습에 사용한 Python 환경과 저장소 루트에서 실행합니다. 재학습과 FU 실행은 없습니다.** 먼저 현재 작업을 보존한 상태에서 이 브랜치의 최신 코드를 받습니다.

```bash
git pull --ff-only origin codex/structural-latent-defense
python reproduction/validate_detection.py --run-dir artifacts/badfu_author_s42
```

실험 폴더가 다른 위치이면 `--run-dir`만 실제 경로로 바꿉니다. 공개 이미지 경로는 학습의 `protocol.json`에서 자동으로 읽습니다. 데이터를 옮겼다면 `--cifar-root /실제/데이터/상위폴더`를 추가합니다. 해당 폴더 아래에 `cifar-10-batches-py`가 있어야 합니다. 이미 준비한 무라벨 이미지 풀을 쓰려면 대신 `--pool /실제/pool.npz`를 지정합니다. 이미지 클래스 라벨, BadFU 원본 공격 이미지, `pert_result.pt`는 탐지에 필요하지 않습니다.

이 명령은 다음을 순서대로 수행합니다.

1. 6개 단계의 실제 체크포인트를 모두 읽어 해시·연속성·유한값·FL 집계를 검사합니다. 40R이면 초기 모델을 포함해 **206개**가 예상됩니다. FU 단계는 저장 연속성을 검사하며 FU 알고리즘을 재실행하지 않습니다.
2. `full6`, `clean5`, `cover5`의 모든 FL 라운드에서 기존 무라벨 관측을 수집합니다. 서로 겹치지 않는 80장 × 3세트, 고정 패턴 3개 × 4모서리, 창 길이 전체/10/20을 사용합니다. 점수 공식이나 설정을 결과에 맞춰 조정하지 않습니다.
3. **HRAC, 전체 이력만 쓰는 HRAC, 음의 기여량, 절대 기여량, 마지막 라운드 음의 기여량**을 같은 관측에서 비교합니다. 깨끗한 이미지와 패턴-원본 차이도 비교합니다. 클라이언트·타깃 정답은 점수 계산 후 순위 평가에만 씁니다.
4. 매 라운드 순위와 당시 저장된 ASR을 연결합니다. 동점은 순위 범위로 남기고, 최초 단독 1위 시점과 마지막 순위를 각각 기록합니다. 정상 단계의 점수도 함께 남깁니다.
5. 전체 가중치 없이 다시 점수를 계산할 수 있는 작은 결과 묶음을 만듭니다.

완료 후 **이 파일 하나를 공유하면 됩니다.**

```text
artifacts/badfu_author_s42/detection_validation_v1/badfu_detection_evidence.zip
```

압축 안의 `report.txt`는 요약, `summary.json`은 마지막 순위·최초 단독 1위·기존 FU 전후 ASR, `comparison.csv`는 라운드별 비교입니다. 관측 평균(`response_means.npz`), 원래 평가 로그·manifest·입력 해시도 포함합니다. 모델 가중치와 이미지 픽셀은 포함하지 않습니다. 약 60 GB 체크포인트를 업로드할 필요가 없습니다. 원본별 관측은 실행 컴퓨터에 별도로 남습니다.

**GitHub에 결과를 올릴 때:** `artifacts/`는 Git 제외 대상이므로 `git add .`만으로 ZIP이 포함되지 않을 수 있습니다. 생성된 ZIP 하나를 `reproduction/validation/` 아래로 복사해 커밋하거나, 정확한 ZIP 파일 하나에만 `git add -f -- 파일경로`를 사용합니다. 체크포인트 폴더 전체를 추가할 필요는 없습니다. `git status`에서 ZIP이 포함됐는지 확인한 뒤 푸시합니다.

중단되면 **같은 명령**을 다시 실행합니다. 체크포인트는 다시 검사하고 이미 완료된 관측은 해시 확인 후 재사용합니다. 코드·입력·설정이 바뀌었으면 새 `--out artifacts/badfu_detection_v2`를 지정해야 합니다. 메모리 부족이면 새 출력 폴더와 `--batch-size 16`, GPU가 없으면 `--device cpu`를 사용합니다. 누락된 모델은 자동 재학습하지 않고 오류로 알려줍니다.

로컬 RTX 4060 Laptop에서 실제 2R 축소 궤적의 6단계 검사와 3단계 관측·내보내기에 약 1분이 걸렸습니다. 테스트 118개, 기존 분석기와 점수 배열 180개 일치, 작은 묶음만 사용한 재계산도 확인했습니다. [검증기 자체의 점검 기록](reproduction/validation/detection_runner_smoke.json). **40R은 15–30분을 예상하지만 실측 전 추정**이며 디스크와 GPU에 따라 늘어납니다. CPU는 훨씬 오래 걸릴 수 있습니다. 라운드마다 관측 시간과 예상 잔여 시간이 출력됩니다.

판독 시 주의할 범위:

- 순위 1위는 삭제 후보의 상대적 위치입니다. 독립적인 정상 실행으로 임계값을 보정하지 않았으므로 **탐지 판정·FPR·TPR은 산출하지 않습니다.** 이미지 3세트는 독립 학습 3회가 아닙니다.
- `clean5`는 5클라이언트, `full6`는 6클라이언트이므로 정상 단계 비교는 진단용입니다. 강한 Non-IID에서 정상·공모를 구분했다는 결론은 아직 낼 수 없습니다. `cover5`는 정상 대조군이 아닙니다.
- `fu_remove5`의 출발 학습은 `full6`, `fu_remove1`은 `cover5`입니다. 두 FU 결과를 같은 모델의 공격/정상 삭제 대조 실험으로 해석하지 않습니다.
- 저장된 ASR을 비교하며 공격 이미지로 ASR을 다시 평가하지는 않습니다. 최종 ASR이 이미 높으면 잠복 상태의 사전 탐지 증거가 되지 않습니다. 최초 1위도 사후 시점 비교이며 순차 오탐 보장이 있는 경보가 아닙니다.
- 현재 검증 대상은 BadFU의 기존 **HRAC inter 채널**입니다. 별도로 관측한 삭제 요청이 필요한 intra 채널과 FUBA는 이 명령에서 검증하지 않습니다.

다른 컴퓨터에서 압축을 푼 뒤 같은 버전의 코드로 아래를 실행하면, 가중치나 이미지 없이 결과표를 재계산할 수 있습니다.

```bash
python reproduction/validate_detection.py --replay /압축푼/경로/evidence
```

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

## 5. 세부 관측 설정과 기존 기록

완료된 BadFU의 검증은 문서 맨 위의 통합 명령을 사용합니다. 관측기 자체의 세부 설정은 [이미지 준비·중간 저장·분석 안내](docs/UNLABELED_PROBES.md), 학습 실행기 축소 검증은 [검증 기록](reproduction/validation/summary.json)을 참고합니다.
