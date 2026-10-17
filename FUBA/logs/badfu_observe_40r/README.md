# BadFU 40-Round Attack & Benign Observation Benchmark Logs

이 결과는 보관된 `request_purify` 방어 실험 실행기의 결과다. 2026-10-15에 새로 받은 공식 소스 실행 결과와 구분한다. GitHub에는 JSON/CSV가 있고 전체 체크포인트는 포함되지 않는다. 삭제 전 ASR이 이미 65.29–81.61%여서 이 수치만으로 낮은 ASR의 잠복 공격을 재현했다고 주장하지 않는다.

DISLAB 로컬 환경(`C:\Users\DISLAB\.gemini\antigravity\scratch\BadFU_defend\artifacts\badfu_observe_v1\`)에서 수행된 **BadFU 40라운드 공격 및 정상 연합학습(FL) 궤적과 연합 언러닝(FU) 관측 벤치마크**의 상세 로그 데이터입니다.

> [!NOTE]
> 저장소 용량 및 깃허브 정책 준수를 위해 대용량 바이너리 가중치 파일(`history.pt`, `model.pt`, `initial.pt` 등 약 1.86 GB)을 제외하고, 모든 평가 지표, 학습 곡선, 관측 점수, 탐지 판정 및 매니페스트가 담긴 **122개 JSON/CSV 로그 파일(약 32 MB)**을 원본 디렉터리 계층 구조 그대로 보존하여 등록했습니다.

---

## 1. 실험 개요 및 환경

- **데이터셋/모델**: CIFAR-10 / ResNet-18 변형 (BadFU 벤치마크 구성)
- **클라이언트 구성**: 6 클라이언트, 불균등 분할 (Dominant 70% 설정)
  - 공격(Attack) 런: 클라이언트 0이 공격자(Attacker), 클라이언트 5가 언러닝 대상(Target to unlearn)
  - 정상(Benign) 런: 공격자 없음, 클라이언트 5가 언러닝 대상
- **학습 라운드**: 40 라운드, 라운드당 로컬 에폭 5회
- **평가 주기**: 5 라운드 단위(`round: 5, 10, 15, 20, 25, 30, 35, 40`) 정밀 평가
- **시드 구성**:
  - 공격 평가군: `eval-attack-42`, `eval-attack-43`, `eval-attack-44`
  - 정상 평가군: `eval-benign-42`, `eval-benign-43`, `eval-benign-44`
  - 사전 보정군: `calibration-142`, `calibration-143`

---

## 2. 40라운드 연합학습(FL) 학습 완료 시점 성능 (Pre-Unlearning)

40라운드 연합학습 완료 직후 모델의 클린 정확도(ACC) 및 백도어 공격 성공률(ASR) 관측치입니다.  
(`docs/TRAJECTORIES.md` 및 DISLAB 원시 `training_metrics.json`과 소수점 둘째 자리까지 100% 일치)

| 실험 구분 | 시드 (Seed) | 최종 Clean ACC (%) | 최종 Backdoor ASR (%) | Clean Loss | 비고 |
|---|---|---|---|---|---|
| **Attack (BadFU)** | Seed 42 | **83.43%** | **69.77%** | 0.7555 | 삭제 전부터 높은 ASR |
| **Attack (BadFU)** | Seed 43 | **84.72%** | **65.29%** | 0.7159 | 삭제 전부터 높은 ASR |
| **Attack (BadFU)** | Seed 44 | **87.07%** | **81.61%** | 0.5947 | 삭제 전부터 높은 ASR |
| **Benign (정상)** | Seed 42 | **83.86%** | **1.99%** | 0.7277 | 정상 훈련 기준선 |
| **Benign (정상)** | Seed 43 | **84.89%** | **2.01%** | 0.6974 | 정상 훈련 기준선 |
| **Benign (정상)** | Seed 44 | **87.00%** | **1.76%** | 0.6128 | 정상 훈련 기준선 |

---

## 3. 연합 언러닝(FU) 후 성능 및 방어 암(Arm) 비교

클라이언트 5번 데이터를 제거하는 연합 언러닝(FU) 수행 후의 관측 결과입니다.  
(`summary.csv` 및 각 암별 `result.json` 집계)

### 3.1. 무방어 언러닝 (`arm: none`)
- **공격 궤적 (`eval-attack-*`)**:
  - Seed 42: Clean ACC 85.40% / **ASR 100.00%**
  - Seed 43: Clean ACC 86.08% / **ASR 100.00%**
  - Seed 44: Clean ACC 87.96% / **ASR 100.00%**
  - 삭제 후 세 시드 모두 ASR 100.00%가 관측됐다. 이 실험에서 ASR이 더 높아진 결과이며 공식 원본 전체 실행의 재현을 입증하지는 않는다.
- **정상 궤적 (`eval-benign-*`)**:
  - Seed 42: Clean ACC 85.18% / ASR 1.44%
  - Seed 43: Clean ACC 86.28% / ASR 1.83%
  - Seed 44: Clean ACC 88.44% / ASR 1.78%

### 3.2. 기존 관측 기반 탐지 방어 (`arm: detected`)
- **오탐(False Positive) 발생**: 정상 3개 시드 중 2개(`eval-benign-43`, `eval-benign-44`)에서 오탐 경보 발생 (오탐률 **66.7%**).
- **미탐(False Negative) 발생**: 공격 시드 중 `eval-attack-42`는 경보를 울리지 못함 (`alarm: False`, `status: no_alarm`).
- **무력화 실패**: 경보가 울린 공격 시드 43, 44에서도 ASR이 **100.00%**로 유지되어 기존 투영 기반 방어가 실패함.

### 3.3. 오라클 방어 (`arm: oracle`)
- 오라클 진단(`arm: oracle`)에서도 ASR 100.00%가 유지됐다. 이 설정의 투영 방어가 효과를 보이지 않았다는 결과이며, 특정 대안 방어가 필수적이라는 결론까지 뒷받침하지는 않는다.

---

## 4. 디렉터리 구성 및 포함 파일 안내

```text
FUBA/logs/badfu_observe_40r/
├── README.md                          # 본 요약 문서
├── summary.csv                        # 전체 런 조건 및 결과 집계 CSV
├── summary.json                       # 전체 런 상세 메타데이터 및 타이밍 JSON
├── observation_scores.csv             # 라운드별 관측 점수 시계열
├── fu_observation_curves.csv          # 언러닝 과정 중의 관측 지표 곡선
├── observation_summary.json           # 관측 통계 요약
├── observation_policy.json            # 관측 정책 설정
├── observation_policy_cost.json       # 관측 계산 비용
├── detector_policy.json               # 탐지 정책 임계치 및 설정
├── suite.json                         # 실험 스위트 메타데이터
├── calibration-142/                   # 보정용 시드 142 매니페스트 및 학습 로그
├── calibration-143/                   # 보정용 시드 143 매니페스트 및 학습 로그
├── eval-attack-42/                    # 공격 시드 42 FL 학습 로그 및 각 방어 암 결과
│   ├── training_metrics.json          # 5라운드 단위 학습 성능 (ACC 83.43%, ASR 69.77%)
│   ├── detected/                      # arm: detected 언러닝 결과 로그
│   ├── none/                          # arm: none 언러닝 결과 로그
│   └── oracle/                        # arm: oracle 언러닝 결과 로그
├── eval-attack-43/                    # 공격 시드 43 FL 학습 로그 및 결과
├── eval-attack-44/                    # 공격 시드 44 FL 학습 로그 및 결과
├── eval-benign-42/                    # 정상 시드 42 FL 학습 로그 및 결과
├── eval-benign-43/                    # 정상 시드 43 FL 학습 로그 및 결과
└── eval-benign-44/                    # 정상 시드 44 FL 학습 로그 및 결과
```
