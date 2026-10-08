# 삭제 요청에 따른 탐지 신호 관측 실험

기존 쌍 탐지가 놓친 공격에서 **관측 대상을 바꾸면 정상 요청과 공격 요청이 구별되는지** 확인한다. 정상 기준을 제거한 헤드 업데이트, 고정 정상 입력에 대한 기능 변화, 실제 FU 중 변화를 같은 실행에서 측정한다. `--observe`는 관측만 추가하며 새 점수를 정화에 사용하지 않는다. `detected`는 기존 탐지·정화이고 새 방법의 성능을 뜻하지 않는다.

이 순서는 [방어 논문에서 도출한 연구 방향](../../docs/research/DETECTION_REDESIGN_2026-10-08.md)의 첫 단계다. 관측이 분리되는지 확인한 다음 요청 조건부 추정량과 정화 규칙을 설계한다. 현재 지원하는 실제 공격은 공식 BadFU 예제의 **두 클라이언트 분리형**이다.

## GPU에서 실행

기존 CUDA 환경과 BadFU record를 사용한다. 저장소 루트에서 최신 연구 브랜치를 받은 뒤 **새 출력 폴더**를 지정한다.

```bash
git switch codex/request-aware-purification
git pull --ff-only
bash run_badfu_observe_gpu.sh --out artifacts/badfu_observe_v1 --download
```

Windows PowerShell에서는 같은 인자를 사용한다.

```powershell
.\run_badfu_observe_gpu.ps1 --out artifacts/badfu_observe_v1 --download
```

PowerShell 스크립트 실행 정책 때문에 실행이 막히면 정책을 바꾸지 않고 다음 명령을 사용한다.

```powershell
python -u -m experiments.request_purify --dataset badfu --device cuda:0 --observe --probe-size 64 --arms none detected oracle --post-rounds 0 --out artifacts/badfu_observe_v1 --download
```

record가 다른 위치에 있으면 `--record /path/to/pert_result.pt --data-root /path/to/cifar10`를 붙인다. record 옆의 기존 백도어·위장·테스트 이미지도 필요하다. `--download`는 CIFAR 다운로드 허용이며 공격 이미지를 새로 생성하지 않는다. 원본 준비 스크립트를 다시 실행하여 기존 이미지를 이동·삭제하지 않는다. 준비 사항은 [기존 실행 안내](README.md)를 따른다.

기본 실행은 정상 보정 seed 142·143, 평가 seed 42·43·44, 원래 학습 40라운드·로컬 5 epochs, 실제 FU 로컬 1 epoch다. 공격에서는 `none`, 기존 `detected`, 기존 `oracle`을, 정상에서는 `none`, 기존 `detected`를 수행한다. 관측 비교에 집중해 후속 FL은 생략한다. 기존 42번 결과는 이미 본 개발 자료이며, 이 실행을 미사용 최종 평가라고 부르지 않는다.

관측 자체가 호출하는 추가 로컬 학습·backward·optimizer step·탐지용 FU는 모두 0회다. 정상 보정 학습, 공격/정상 대조군, 각 FU arm은 명시적인 오프라인 비교 비용이다. 새 관측은 이미 생성된 로컬 모델의 forward와 CPU 통계를 추가하므로 **총비용이 0은 아니다.**

## 입력과 혼동 요인 통제

클라이언트 학습에서 제외한 정상 보조 데이터 1,000개 중 64개를 고정 seed로 선택한다. 백도어·위장 원본 ID도 보조 데이터에서 제외한다. probe의 ID·클래스 분포·텐서 해시·투영 해시를 기록하고 FL과 FU에서 같은 입력인지 검사한다. 테스트 이미지·트리거·공격 타깃·ASR은 관측 함수에 전달하지 않는다.

기존 실행과 정화 결과를 비교할 수 있도록 guard의 검증 입력은 바꾸지 않았다. 따라서 **probe와 guard는 정상 보조 입력을 공유한다.** probe를 guard의 독립 평가 데이터라고 해석하면 안 된다. 정상 성능 평가는 별도 CIFAR 테스트셋이다. 관측을 정화에 연결할 다음 단계에서는 별도 guard holdout과 클래스별 손상 평가를 설계한다.

관측은 학습 모델의 복사본에서 eval 모드로 수행한다. 원 모델의 파라미터·BN 버퍼·학습 모드를 수정하지 않는다. 원래 BN 통계로 측정한 결과와 라운드 시작 모델의 BN 통계로 측정한 결과를 함께 저장한다. BN 통제는 분석용 복사본에만 적용하며 실제 FU와 ACC/ASR 평가에는 적용하지 않는다.

## 독립 비교하는 관측

| 이름 | 측정 | 해석 |
|---|---|---|
| `raw_pair` | 기존 헤드 업데이트의 쌍 점수 | 기존 관측 기준선 |
| `clean_residual_pair` | 정상 입력의 head gradient 축을 제거한 업데이트의 쌍 점수 | 공통 정상 방향이 상쇄를 가리는지 확인 |
| `margin_pair` | 동일 입력의 모든 클래스 margin 변화로 계산한 쌍 점수 | 파라미터 대신 출력 기능의 상쇄 관측 |
| `margin_bn_pair` | 같은 margin 관측에서 BN 통계를 시작 모델로 고정 | 출력 차이의 BN 혼동 확인 |
| `representation_pair` | 정규화한 마지막 은닉층 표현의 고정 투영 변화로 계산한 쌍 점수 | 출력에 바로 드러나지 않는 표현 변화 관측 |
| `representation_bn_pair` | 같은 표현 관측의 BN 통제 | 표현 차이의 BN 혼동 확인 |
| `margin_magnitude` | 모든 클라이언트·클래스 중 margin 변화 RMS의 최대값 | 요청자 효과를 사용하지 않는 비교군 |
| `representation_magnitude` | 모든 클라이언트 중 표현 변화 RMS의 최대값 | 쌍 상쇄를 요구하지 않는 비교군 |

쌍 후보들은 관측 공간만 바꾸고 기존 `scores`의 가중치·상쇄·지속성 계산을 유지한다. 기능 변화의 가중합이 실제 집계 모델의 기능 변화와 같다고 가정한 정확한 분해는 아니다. 모두 실패하면 점수를 억지로 결합하지 않는다. 최종 점수 외에도 클래스별 평균 코사인, 음의 코사인 비율, raw 점수, 지속 라운드 수를 내보내 지속성 필터가 신호를 없앴는지 구분한다. `usable_class_counts`가 0인 항목은 코사인 평균도 0으로 표기하지만 유효한 직교 관측이 아니다.

### 정상 기준 제거

라운드 시작 모델의 깨끗한 입력에서 클래스별 선형 head의 cross entropy gradient `g_c`를 계산한다. softmax 오차와 마지막 표현의 외적으로 구하므로 backward나 별도 모델 학습은 필요 없다. 업데이트의 클래스 행 `u_i,c`를 다음처럼 변환한다.

```text
g_c = mean_x [(p(c|x) - 1[y=c]) · concat(h(x), 1)]
r_i,c = u_i,c - <u_i,c, g_c> / max(||g_c||², epsilon) · g_c
```

기준은 요청자와 다른 클라이언트의 평균에서 만들지 않는다. 두 벡터를 인위적으로 반대로 만드는 pair centering도 아니다. 다만 오염된 글로벌 모델 위에서 구한 정상 데이터 gradient이므로 완전히 깨끗한 모델의 기준은 아니다. 한 축만 제거하며 악성 성분까지 제거할 수 있다. 이 비교군을 정확한 정상/악성 분리나 새 기여로 주장하지 않는다.

### 기능 변화

클래스별 margin `m_c(theta,x) = logit_c - max_{k != c} logit_k`에서 이미 제출된 로컬 모델과 라운드 시작 모델의 차이를 기록한다.

```text
phi_i,t(x,c) = m_c(theta_i,t,x) - m_c(theta_t,x)
```

표현은 마지막 은닉층을 입력별 L2 정규화하고 고정 직교 투영 16차원으로 줄인다. `--probe-dimensions` 변경은 별도 실험 조건이며 압축 과정에서 신호를 잃을 수 있다. clean probe에서 기능이 같으면 이 관측으로 두 모델을 구분할 수 없다. 현재는 OOD·트리거 탐색·indicator 삽입을 수행하지 않는다.

### 실제 FU 중 관측

모든 FU 라운드에서 실제 잔존 로컬 모델, 단계 시작 모델, guard 이후 실제 채택 모델을 평가한다. 기존 탐지가 `no_alarm`이어도 실행한다. 클래스별 margin 변화·예측 분포·표현 변화와 해당 라운드의 과거 요청자 효과에 대한 cosine을 기록한다. 원래 FL 최종 모델과의 차이도 별도 기록한다. 초기 모델부터 replay하므로 초반의 큰 변화가 곧 공격 신호는 아니다.

FU 중 변화는 현재 **임계값을 보정하지 않은 관측값**이다. `alarm`은 null이며 탐지율·오탐률·탐지 지연을 산출하지 않는다. 동일 seed의 공격/정상 FU 곡선을 먼저 비교한다. 전용 정상 FU 보정과 독립 평가가 준비되어야 운영 경보 규칙을 만들 수 있다. ACC/ASR은 기본 5라운드마다 측정하며, 정밀한 시점 비교가 필요하면 새 실행에서 `--eval-every 1`을 미리 지정한다.

## 보정과 요청 정보 비교

각 사전 관측 후보의 임계값은 **정상 보정 학습에서만** 정한다. 한 정상 seed 안에서는 모든 요청자·클라이언트·클래스의 최대값 하나를 취하고, seed 간 99% 경험 분위수를 사용한다. 기본 정상 seed 2개로 모집단 FPR 1%를 보장할 수 없다. 후보별 경보를 따로 보정했으므로 ‘하나라도 울리면 공격’인 OR 규칙의 FPR도 보장하지 않는다.

같은 학습 기록에서 관측에 주는 요청자 q를 여섯 값으로 바꾼 점수도 저장한다. **실제 삭제 대상은 원래 요청자 5로 유지한다.** 이것은 요청 정보 입력의 ablation이다. 다른 클라이언트를 실제 삭제한 결과나 반사실적 위험 정답을 제공하지 않는다. 요약의 공격·정상 경보 통계에는 실제 요청자만 들어간다.

## 전달할 결과와 판단 순서

실행이 끝나면 `artifacts/badfu_observe_v1/results_to_share.zip` 하나를 전달한다.

| 파일 | 확인할 내용 |
|---|---|
| `observation_summary.json` | 후보별 실제 요청의 공격·정상 경보 수, seed 수, 같은 seed의 점수 차이 |
| `observation_scores.csv` | 후보·seed·관측 요청자별 점수와 고정 임계값 |
| `eval-*/observation_scores.json` | 클래스·클라이언트별 점수와 cosine/raw/persistence 진단 |
| `fu_observation_curves.csv` | 실제 FU 단계·클래스별 변화와 요청자 효과 정렬 |
| `eval-*/*/fu_observations.json` | 모든 q 입력의 FU 정렬 값, 유효 관측 mask, BN 통제 값 |
| `observation_policy.json` | 정상 보정 출처, seed 최대값, 후보별 임계값 |
| `probe_manifest.json`, `fu_probe_manifest.json` | 입력 해시·클래스 분포·추가 forward 수, backward/FU 0회 |
| `training_cost.json`, `*/result.json` | 학습/FU별 관측 시간과 특징 저장량, 실제 정화·ASR 결과 |
| `suite.json` | 전체 설정·장비·실행 상태·실제로 실행한 Python 소스 해시 |

상세 tensor는 `history.pt`의 `observations`, 각 arm의 `fu_observations.pt`에 남긴다. 전체 로컬 모델을 저장하지 않는다. 이미지·모델·tensor는 공유 zip에 포함하지 않는다.

먼저 삭제 전 은폐와 삭제 후 활성화가 재현됐는지 확인한다. 다음으로 동일한 정상 오탐 기준에서 어느 관측이 공격과 정상의 차이를 보이는지, BN 통제로 사라지는 차이는 아닌지 본다. 요청 비의존 magnitude만으로 같은 결과가 나오면 요청 조건부 기여를 축소한다. 알려진 공격자 방향의 기존 `oracle`도 여전히 실패하면 탐지 개선과 별도로 정화 위치·부분공간을 검토한다.

관측 선택·임계값·probe 수를 결과를 보고 바꾼 뒤 같은 seed를 최종 검증으로 재사용하지 않는다. minority 손상, 같은 클라이언트 내부의 샘플 삭제, FUBA, 다른 FU 공격과 적응형 공격은 다음 검증 범위다.

## 재개와 코드 검증

같은 인자에 `--resume`을 붙이면 완료된 학습과 FU를 다시 수행하지 않는다. 설정·probe·코드 해시가 다르면 재개를 거부한다. 관측 없는 과거 history에서 기능 특징을 복원할 수 없으므로 기존 ASR 100% 실행 폴더에 새 관측을 소급 채우지 않는다.

```bash
python -m unittest experiments.request_purify.test_core experiments.request_purify.test_observations -v
python -m experiments.request_purify --dataset smoke --device cpu --observe --probe-size 8 --probe-dimensions 4 --out artifacts/observe_smoke --seeds 7 --calibration-seeds 17 --rounds 3 --local-epochs 1 --fu-epochs 1 --post-rounds 0 --arms none zero
```

CPU 검증에서는 관측 on/off의 학습·FU·oracle 정화 최종 모델 해시가 같고, 로컬 학습 호출 수도 같음을 검사한다. BN 통계 통제가 학습 모델·RNG를 바꾸지 않는지, 정상 전용 보정, 요청 비의존 점수의 불변성, 결과 zip과 동일 설정 재개도 검사한다. 이 검증은 GPU 실행 성능이나 실제 BadFU 방어 효과의 실증이 아니다.
