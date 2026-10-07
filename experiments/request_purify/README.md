# BadFU GPU 언러닝 정화 실험

공식 BadFU 예제의 분리형 공격 데이터를 사용하여 탐지용 언러닝 시뮬레이션 없이 실제 FU와 정화를 함께 수행하는 실험 패키지다. **실제 데이터 실험은 CUDA 필수**이며 GPU가 없으면 실패한다. CPU는 합성 데이터로 코드 동작을 확인하는 smoke 모드에서만 허용한다.

## GPU에서 한 번에 실행

CUDA용 PyTorch와 torchvision이 설치된 환경에서 저장소 루트 기준으로 실행한다. Python 3.10 이상이 필요하다. 필요한 Python 패키지는 [requirements.txt](requirements.txt)에 있다. 기존 PyTorch CUDA 환경을 CPU 빌드로 교체하지 않는다.

```bash
git fetch origin
git switch codex/request-aware-purification
git pull --ff-only

bash run_badfu_gpu.sh \
  --record /absolute/path/to/record/badnet_dataset/pert_result.pt \
  --data-root /absolute/path/to/cifar10-data \
  --out artifacts/badfu_gpu_v1 \
  --download
```

`--record`에는 **기존 BadFU 전처리가 만든 pert_result.pt와 그 이미지 폴더**를 지정한다. 이 실행기는 공격 데이터를 새 방식으로 대체 생성하지 않는다. 이미지는 record 옆의 `bd_train_dataset`, `cv_train_dataset/pert`, `bd_test_dataset` 등에 있어야 한다. 이동된 상대경로를 복원하고 바이트 해시를 기록한다. 같은 파일 이름의 다른 이미지가 여러 곳에 있거나 이미지가 없으면 학습 전에 중단한다. 기존 record를 삭제하거나 덮어쓰지 않는다.

이 저장소의 통상 위치에 데이터가 있다면 다음만 실행해도 된다.

```bash
bash run_badfu_gpu.sh --out artifacts/badfu_gpu_v1 --download
```

`--download`는 CIFAR-10 다운로드를 허용한다. 원본에서 사용한 ImageNet 사전학습 ResNet18 가중치는 캐시에 없으면 torchvision이 받는다. 오프라인 장비에서는 데이터와 가중치를 미리 준비한다. `--no-pretrained`는 별도의 실험 조건이며 원본 설정과 구분해야 한다.

## 실행되는 전체 순서

1. CUDA·record·이미지 경로·타깃 레이블을 확인하고 입력 해시를 저장한다.
2. 정상 학습 seed 142, 143으로 탐지 임계값을 보정한다. 공격 샘플 위치에는 같은 원본의 정상 이미지를 넣어 데이터 수와 여섯 클라이언트 구조를 맞춘다.
3. 평가 seed 42, 43, 44에서 공격 학습과 정상 대조군 학습을 각각 실행한다.
4. 동일한 학습 기록·초기 모델·분할에서 아래 여섯 방법을 비교한다. 정상 대조군은 oracle/random을 제외한 네 방법을 실행한다.
5. 공격 실행의 `none`, `detected` 모델에서 정상 후속 학습, 지속 공격, 중간에 공격을 재개하는 후속 학습을 각각 10라운드 수행한다.
6. CSV·JSON을 집계하고 공유용 `results_to_share.zip`을 만든다.

| 방법 | 의미 |
|---|---|
| none | 정화 없는 실제 FU 기준선 |
| zero | 정답 후보 기저를 만들지만 정화 강도는 0. none과 최종 모델 해시가 같아야 통과 |
| oracle | 알려진 공격자·타깃으로 만든 후보 방향. 정화 가능성을 진단하는 평가용 기준선 |
| detected | 정답 정보를 사용하지 않는 요청자 중심 탐지와 정화 |
| random | oracle과 헤드·클래스 행·랭크·적용 클라이언트를 맞춘 무작위 방향 |
| retrain | 같은 초기 모델과 보존된 분할에서 요청자만 제외하여 처음부터 재학습 |

여러 방법을 실행하는 것은 효과를 비교하기 위한 오프라인 실험이다. **각 detected 실행의 탐지용 FU는 0회**다. 실제 FU의 잔존 로컬 학습을 위험 판정용 시뮬레이션으로 중복 실행하지 않는다. retrain은 비교군이며 detected가 내부적으로 호출하지 않는다.

## 기본 실험 조건

| 항목 | 기본값 |
|---|---|
| 공격 구조 | client 0 정상+백도어, client 5 같은 정상+위장, client 1~4 정상 |
| 삭제 요청자 | 5. 탐지 결과에 따라 변경하지 않음 |
| 학습 | CIFAR-10, ResNet18, 40라운드, SGD 0.01, 로컬 5 epochs, batch 64, 전원 참여 |
| 데이터 분포 | Dominant Class 0.7, 실제 샘플 ID 저장 |
| 정상 검증 데이터 | 공격·위장 원본 ID를 제외한 학습 데이터에서 1,000개 분리 |
| FU | 각 라운드의 시작점을 맞춘 층별 노름 보정 replay, 잔존 로컬 1 epoch |
| 탐지 | 최근 16라운드, 최소 공동 관측 3회, 요청자와 다른 참여자만 비교 |
| 부분공간 | 비중심 SVD, 에너지 0.9, 최대 랭크 8 |
| 정화 | 강도 최대 0.5, 전체 FU 갱신 노름 대비 0.25 및 절대 노름 1.0 상한 |
| 정상 손실 guard | 검증 손실이 정화 없는 같은 단계보다 0.01 넘게 증가하면 그 단계 정화 취소 |

이 기본 강도와 임계값 보정 방식은 개발 시작점이다. 정상 seed 두 개의 12개 요청 점수는 독립 표본 12개가 아니며, 99% 분위수로 보정했다고 운영 FPR 1%가 보장되는 것은 아니다. `detector_policy.json`에 독립 궤적 수와 실제 기준 점수를 기록한다. 성능을 본 뒤 파라미터를 바꾼 실행은 개발 실험으로 분리하고, 최종 평가 seed를 새로 확보한다.

## 원본과 다른 부분

공격 이미지와 배치는 공식 구현을 따르지만 이 패키지는 BadFU 논문 전체의 완전 재현이 아니다. 별도의 깨끗한 검증 분할, NumPy 기반 명시적 분할 RNG, 분리된 로컬 학습 seed를 사용한다. 사전학습 이후 초기 모델 해시와 분할 ID를 저장하므로 비교 방법끼리는 같은 시작 조건을 공유한다.

FU는 `aligned_layerwise_norm_replay_v1`이라는 통제된 FedEraser 계열 기준선이다. 원 예제의 학습 후 첫 글로벌 모델 시작 및 `old_GMs[r+1]` 참조를 그대로 복사하지 않고, **학습 전 초기 모델에서 시작하여 같은 라운드의 시작 모델 대비 잔존 업데이트 노름을 사용**한다. 기본 FU 학습량은 1 epoch이며 원 예제의 5 epochs와 다르다. `--fu-epochs 5`로 늘릴 수 있다. BatchNorm 등 비학습 버퍼는 잔존 로컬 모델의 가중 평균을 쓰고 정화는 학습 가능한 헤드 파라미터에만 적용한다. 따라서 정화 효과는 이 동일 백엔드의 none과 비교한다.

후속 `clean`은 잔존 공격자의 오염 샘플도 같은 원본의 정상 이미지로 바꾸는 통제 조건이다. `ongoing`은 오염 데이터를 계속 사용하고, `rejoin`은 앞 절반을 정상 데이터로 학습한 뒤 오염 데이터를 다시 사용한다. rejoin은 실제 클라이언트 탈퇴·재가입이 아니라 **공격 재개** 실험이다. 요청자는 모든 후속 학습에서 제외한다.

학습·추론과 탐지/SVD/투영은 CUDA에서 실행한다. 체크포인트와 상태 집계·층별 보정량 계산은 CPU에서 수행하고 전송 비용을 포함해 측정한다. 모델을 여섯 개 동시에 GPU에 올리지 않고 하나를 재사용한다. 원 FU에 필요한 과거 잔존 업데이트의 층별 노름과 헤드 기록을 저장하므로 모든 로컬 전체 모델을 매 라운드 보관하지 않는다.

## 결과 전달과 중단 후 재개

완료하면 아래 파일 하나를 전달하면 된다.

```text
artifacts/badfu_gpu_v1/results_to_share.zip
```

포함 내용은 전체 설정과 장비, 데이터 분할·해시, 학습과 FU의 ACC/ASR, 삭제 데이터 손실, 요청자별 탐지 점수, 실제 정화량·guard 거절 여부, 후속 FL 궤적, 시간, 평균과 표준편차다. 모델·원본 이미지·대형 history 파일은 zip에 넣지 않는다. 모델과 history는 실행 폴더에 남긴다.

```bash
# 같은 인자와 같은 출력 폴더를 사용한다.
bash run_badfu_gpu.sh --out artifacts/badfu_gpu_v1 --download --resume
```

기존 출력 폴더는 기본적으로 덮어쓰지 않는다. resume은 동일한 설정과 입력 해시에서만 허용한다. 완료된 학습과 FU 방법은 다시 실행하지 않는다. 중간에 끊긴 한 학습/FU 방법은 같은 seed로 처음부터 다시 실행하며, 개별 mini-batch부터 재개하지 않는다. 실행 실패 시에도 가능한 결과를 zip으로 묶는다.

첫 확인을 한 seed로 줄이려면 모든 나머지 조건은 유지하고 다음처럼 실행한다.

```bash
bash run_badfu_gpu.sh --out artifacts/badfu_gpu_first \
  --seeds 42 --calibration-seeds 142 --download
```

분할·FU·탐지 일치 여부를 먼저 확인하는 용도다. 이 한 seed를 최종 논문 통계로 사용하지 않는다. 실행 시간을 줄이기 위한 `--rounds`, `--local-epochs`, `--post-rounds` 변경도 결과에 기록되며 원 설정과 구분한다.

## 코드 검증

```bash
python -m unittest experiments.request_purify.test_core -v

python -m experiments.request_purify --dataset smoke --device cpu \
  --out artifacts/smoke --seeds 7 --calibration-seeds 17 \
  --rounds 3 --local-epochs 1 --fu-epochs 1 --post-rounds 1
```

smoke는 다운로드 없는 합성 데이터에서 실제 학습·FU·정화·후속 학습·재개·결과 포장 경로를 확인한다. BadFU 공격 재현이나 GPU 성능 검증을 대체하지 않는다.

## 해석할 때의 범위

`deleted_data_loss`는 삭제 관련 진단값이며 망각 증명이나 membership 공격 평가가 아니다. 정답 후보 방향 역시 완전한 백도어 정답 공간은 아니다. 기록을 감사 목적으로 보존하는 연구 실행이므로 데이터 삭제를 인증하는 서비스가 아니다. 현재 패키지는 BadFU 분리형의 첫 통합 구현이며, FUBA IBA 학습 경로와 서버에 개별 업데이트가 보이지 않는 설정을 실행했다고 표시하지 않는다. [연구 계획](../../docs/research/RESEARCH_PROPOSAL_2026-10-08.md)의 나머지 비교군·망각 평가·정상 공간 보호는 후속 구현 범위다.
