# MNTD 방식의 간단한 GPU 검사

기존 BadFU 실험에서 **정상·공격 모델의 언러닝 전후 네 체크포인트**를 읽어, 학습한 진단 입력과 작은 메타 분류기로 검사한다. 실제 FU를 다시 실행하지 않는다. 모델이 없는 경우에는 결과를 만들지 않고 중단한다.

MNTD [원문](https://arxiv.org/abs/1910.03137)과 [공개 구현](https://github.com/AI-secure/Meta-Nerual-Trojan-Detection)을 참고한 독립 구현이다. 진단 입력 10개, 응답을 받는 은닉층 20개, 이진 점수 출력을 사용한다. 우리 ResNet18과 입력 정규화에 맞추고 BN 보존을 위해 모델을 eval 모드로 고정했다. 원 논문의 수천 shadow model 규모를 재현한 실험은 아니다.

## 기존 실제 FU 결과 검사

CUDA PyTorch가 설치된 환경에서 저장소 루트 기준으로 실행한다.

```bash
python -u -m experiments.mntd_check --suite artifacts/badfu_gpu_v1 --seed 42 --out artifacts/mntd_four_conditions --data-root BadFU_src/BadFU-main/data --download
```

필요한 파일은 해당 seed의 `eval-attack-42`와 `eval-benign-42` 각각의 `history.pt`, `manifest.json`, `none/model.pt`, `none/result.json`이다. `results_to_share.zip`에는 모델이 없어 그것만으로 검사할 수 없다. 정상/공격 분할 일치, 체크포인트 hash, trajectory ID를 확인한다.

1. 기존 실행에서 클라이언트 학습에 사용하지 않은 정상 validation 입력만 shadow 학습에 사용한다. CIFAR 원본 hash와 학습 데이터와의 비중복을 검사한다.
2. 정상·범용 patch/blend 오염 shadow model을 학습한다. 기존 BadFU 타깃·트리거·ASR은 이 과정에 사용하지 않는다.
3. 고정 무작위 질의 분류기와 질의를 함께 학습하는 분류기를 같은 초기값에서 비교한다. 학습 대상은 질의와 메타 분류기뿐이다. shadow model은 고정한다.
4. 별도 정상 shadow model 점수의 최댓값을 임계값으로 사용한다. 평가할 BadFU 네 모델의 점수를 보고 임계값을 바꾸지 않는다.
5. 네 모델을 forward로 검사하고 기존 ACC/ASR 기록을 결과 해석용으로 병기한다. ASR은 탐지기 입력이 아니다.

기본 shadow 구성은 학습 정상 8개·오염 8개, 보정 정상 4개·오염 4개, 평가 정상 4개·오염 4개다. 보정의 오염 모델은 점수 기록용이며 임계값이나 모델 선택에 사용하지 않는다. 전체 32개 모델은 각각 정상 보조 이미지 1,000개에서 3 epochs 학습한다. `--train-per-class`, `--shadow-epochs` 등으로 바꿀 수 있고 설정이 기록된다. 소수 표본에서 일반화/FPR을 보장하지 않는다.

**정상 shadow에서 정한 임계값은 정상 FU에서 보정한 임계값이 아니다.** 네 조건 표는 분포 이전의 기초 점검이며, 실제 운영 탐지율·오탐률을 확정하지 않는다. 정상 FU 후에도 점수가 오르면 백도어보다 학습 방식이나 모델 품질 차이에 반응할 가능성을 확인해야 한다.

## 체크포인트 없는 장비에서 학습 가능성만 확인

```bash
python -u -m experiments.mntd_check --pilot --out artifacts/mntd_pilot --data-root data --download
```

독립적인 CIFAR-10/ResNet18 shadow 모델을 학습·평가하는 GPU 실험이다. **BadFU/FUBA 공격이나 FU를 실행하지 않는다.** 따라서 이 모드에는 네 조건 결과가 없으며 `FU_status`에 미평가 상태를 기록한다. 여기서 좋은 AUC가 나와도 언러닝 후 ASR 폭발 탐지의 증거가 아니다.

오염 shadow의 라벨은 공격 학습 recipe를 뜻한다. 학습량이 적어 실제 백도어가 형성되지 않았거나 정상 정확도가 낮을 수 있으므로 `shadow_inventory.json`의 ACC/ASR을 먼저 확인한다. 공격 품질을 보고 특정 모델만 사후 선택하지 않는다. pretrained ResNet18의 수정된 입력 convolution과 출력층은 새로 초기화하며, 기본 SGD 학습률은 0.01이다.

## 결과

- `four_conditions.csv`: 실제 suite 모드의 네 조건 × 두 탐지기 점수·임계값·경보·기존 ACC/ASR.
- `summary.json`: 독립 shadow 평가 AUC/TPR/FPR, 보정/평가 점수, 실제 FU 검사 여부.
- `shadow_inventory.json`: 분할, model seed, 실제 clean ACC와 각 shadow trigger ASR, 공격 설정.
- `experiment.json`: GPU, PyTorch, 설정, 실행 상태, 시간, 코드 hash.
- `learned_queries.pt`, `fixed_queries.pt`: 학습된 질의·분류기와 임계값. 진단 점수는 보정된 확률이 아니다.
- `results_to_share.zip`: JSON/CSV만 묶는다. 모델과 학습 데이터는 포함하지 않는다.

출력 폴더는 새로 생성하며 덮어쓰지 않는다. 중단된 학습 재개는 아직 지원하지 않는다. 이 실험은 정화·시계열 모델·적응형 공격 방어를 추가하지 않는다.

```bash
python -m unittest experiments.mntd_check.test_check -v
```
