> 보관본입니다. 현재 안내는 [저장소 첫 화면](../../../README.md)을 봅니다. 아래 내용은 정리 이전의 기록이며 실행 지침이 아닙니다.

# 실험 돌리기 — 2026-10-14

## 2026-10-15 업로드 결과 읽기

`dev`의 `90d2f33`까지 확인한 새 업로드는 CIFAR 공격 3개와 정상 4개의 **궤적 진단**이다. 탐지 결과 파일과 구분한다. [현재 연구 요약](https://github.com/sijoon-sung/BadFU_defend/blob/4c4c38f19d969fa776f75064a112543d8ce2657d/docs/RESEARCH_SUMMARY.md), [수치 출처 감사](https://github.com/sijoon-sung/BadFU_defend/blob/4c4c38f19d969fa776f75064a112543d8ce2657d/docs/EVIDENCE_AUDIT.md)를 먼저 읽는다. 자동 검증의 체크 표시를 잠복 공격 검증 완료로 세지 않는다.

그래프는 저장소 루트에서 `python docs/plot_evidence.py`로 재생성한다(Matplotlib·NumPy 필요). 이 명령은 registry JSON만 읽으며 모델 학습·FU를 실행하지 않는다. CSV와 입력 SHA-256도 `docs/figures/`에 남긴다.

실행 중인 서버 작업은 그대로 둔다. 완료 후 실제 사용한 커밋·명령·설정, `manifest.json`, `scores.json`, `detection_report.json`을 함께 올린다. 체크포인트 재학습은 필요 없다. 기존 탐지 실험의 결과가 없으면 아래 판독기를 궤적 registry에 직접 적용하지 않는다. 서로 다른 입력 형식이다.

다음 확인 실험에서는 (1) 사전 고정된 잠복/효용 조건, (2) 실제 FU 전후 평가, (3) 독립 정상 보정/시험을 분리한다. CIFAR의 종료 ASR을 본 뒤 낮은 ASR 라운드를 골라 확증 표본으로 쓰지 않는다. 참여 여부에 따른 ASR 변동은 모든 저장 라운드를 보이는 별도 진단으로 보고한다.

## 현재 우선순위: 탐지 판정을 먼저 확립 (2026-10-14)

**이미 실행 중인 실험은 계속 진행한다. 실행 도중 작업 폴더의 코드를 갱신하거나 설정·문턱을 바꾸지 않는다.** 이번 변경은 별도 결과 판독기와 문서뿐이며, 등록된 학습·수집·점수·FU 실행기는 변경하지 않았다. 결과가 나온 뒤 독립된 분석 checkout 또는 작업 종료 후 갱신한 checkout에서 판독한다.

탐지 입력은 **FU 전까지 관측한 클라이언트 업데이트 이력 + 삭제 요청**이다. 탐지 결과에 실제 FU 결과나 정답 트리거·공격 역할을 넣지 않는다. 목표는 FL→FU 잠복 공격의 활성화 위험이 있는 요청을 구분하는 것이며, 우선 기존 점수가 공격 요청과 정상 non-IID를 분리하는지 검증한다. 수학적 정의·성공/중단 기준은 [설계 §8](https://github.com/sijoon-sung/BadFU_defend/blob/4c4c38f19d969fa776f75064a112543d8ce2657d/docs/paper/TENSION_CONTROL.md#8-탐지-우선-평가-2026-10-14)을 따른다.

| 현재 실행 결과 | 이번 단계에서 말할 수 있는 것 |
|---|---|
| FUBA `experiment_tension.py`의 `manifest.json` + `scores.json` | 정상 보정 후 공격 요청 탐지율, 정상 에피소드 오탐률, 마지막 시점 탐지 여부. 아래 판독기로 읽는다. |
| BadFU `request_pipeline.py`의 FL 저장 및 FU 비교 | 공격 활성화·제어 동작의 자료. **제어 불수락/성공 또는 ASR 감소를 탐지 경보로 변환하지 않는다.** 이 출력은 아래 판독기의 입력 형식이 아니다. |

FUBA 실험이 **완료된 뒤**, 저장소 루트에서 실행한다. NumPy·SciPy만 필요하고 GPU·재학습·FU 호출은 없다. 서버의 `manifest.json`, `scores.json`만 같은 폴더로 복사해도 읽을 수 있다.

```bash
python FUBA/report_detection.py --experiment FUBA/logs/tension_v3/pilot
# 이미 confirm 묶음을 실행했다면 해당 경로 사용
# python FUBA/report_detection.py --experiment FUBA/logs/tension_v3/confirm
```

출력 `detection_report.json`은 기존 파일을 덮어쓰지 않는다. 재분석은 `--out <새 파일.json>`을 지정한다. 이 판독기는 기존 주 비교 `logprob/mean_before/window:0`, `selective_v3` 대 `linear_release`, α=.05를 유지한다. 새로운 탐지기나 자동 최적 설정 선택기가 아니다.

- `clean_episode_fpr`: 정상 독립 궤적에서 등록된 어느 요청·검사 시점이라도 경보가 난 비율과 정확 이항 95% 신뢰구간.
- `attack_splits.attack_test.requested_attack_detection_rate`: 새 공격 궤적의 **평가 대상 요청**에 경보가 난 비율. `attack_development`는 별도이며 확인 결과에 합치지 않는다.
- `final_requested_attack_detection_rate`: 마지막 등록 시점에서 그 요청을 탐지했는가. 앞선 시점의 일시 경보와 분리한다.
- `unrelated_only_alarm_rate`: 다른 요청만 경보가 나고 공격 요청은 놓친 비율. 공격 탐지 성공으로 세지 않는다.
- `first_request_alarm_inspection`: 등록된 가상 요청 검사의 첫 경보 시점이며 실제 요청 시각이나 탐지 지연은 아니다.
- `activation_validation`: 현재 점수 파일에는 대응되는 실제 FU의 활성화 정답이 없으므로 미검증이다. ASR로 검증된 위험 요청 탐지율을 만들어내지 않는다.

pilot의 조건별 보정 3개로는5% 문턱을 만들 수 없어 경보·탐지율·FPR가 `null`이다. confirm의 보정19개는 유한 문턱을 만들 최소 수일 뿐이다. 정상 시험10회 중 오탐0회라도 이항95% 상한은 약30.85%이며, FPR≤5%를 실증한 것이 아니다. 두 방법의 경보를 OR로 합치거나 시험 성능이 좋은 방법을 사후 선택하면 기존 보정의 보장을 그대로 사용할 수 없다.

지금 추가 학습을 시작할 필요는 없다. 먼저 실행 중인 결과의 분리를 확인한다. BadFU까지 탐지 주장을 넓히려면 같은 관측 규약으로 수집한 공격 없는 정상/비IID 궤적과 독립 공격 궤적의 사전 점수가 필요하다. 기존 ASR 비교만으로 그 자료를 대체하지 않는다.

### Baseline 정리: 주 비교와 구성요소 대조를 구분

| 역할 | 저장된 점수 | 확인할 질문 |
|---|---|---|
| **주 baseline** | `linear_release` | 삭제되는 기여의 양의 순효과만으로도 충분한가? |
| **제안 방법** | `selective_v3` | 장력과 선택적 삭제율이 주 baseline에 더하는 정보가 있는가? |
| 요청 경계 대조 | `tension_only` | 요청 방향 없이 상쇄 크기만 봐도 같은 효과인가? 요청별 점수가 같으므로 요청자 식별 성공으로 해석하지 않는다. |
| 기여 크기 대조 | `removed_negative` | 여러 기록을 함께 삭제할 때 양의 삭제 기여를 무시해도 되는가? **단일 기록 삭제 점수는 `linear_release`와 동일**하므로 별개의 독립 baseline으로 세지 않는다. |
| 이전 수식 검증 | `cross_v2` | 동일 비율 삭제 반례를 고친 v3와 무엇이 달라지는가? 기존 방법의 강한 외부 baseline으로 소개하지 않는다. |

기존 `scores.json`에 저장된 다섯 점수를 읽어 같은 평가 형식으로 정리할 수 있다. 주 비교는 바뀌지 않으며 재학습은 없다.

```bash
python FUBA/report_detection.py --experiment FUBA/logs/tension_v3/pilot --include-controls --out FUBA/logs/tension_v3/pilot/detection_with_controls.json
```

`comparison.role`, `comparison.request_specific`로 비교 목적을 표시한다. 보조 점수가 없는 입력은 임의로 보충하지 않는다. 같은 궤적·공개 입력·집계 가중치·요청/시간 가족·보정/시험 분할을 공유하고, **각 방법의 문턱은 그 방법의 정상 보정 점수로** 정한다. 서로 단위가 다른 점수에 숫자 하나의 문턱을 공통 적용하지 않는다. 단일 삭제의 원점수가 같은 두 방법도 전체 요청 가족의 보정 문턱이 다르면 경보가 달라질 수 있으며, 이 차이는 새 단일 요청 정보의 증거가 아니다.

원시 norm/cosine/Krum 점수, 삭제 시뮬레이션, FL 집계 방어, 실제 FU 제어는 정보와 개입 단계가 다르므로 이 표의 성능에 섞지 않는다. 기존 광범위 탐색은 유지하고, 정식 외부 방어 비교는 원 기법의 입력·학습 개입·계산 비용을 맞춘 별도 실험으로 다룬다. 이번 정리는 기본 내부 baseline/대조군의 정합성이며 모든 선행 방어의 재현 완료를 뜻하지 않는다.

## BadFU: 별도 궤적 보존과 개정 알고리즘 자동 비교

저장소 **루트**, `codex/structural-latent-defense` 브랜치에서 실행한다. PyTorch·torchvision·NumPy·SciPy가 필요하며 MPI·timm은 필요 없다. 이 절은 CIFAR-10/ResNet-18 공개 코드의 **6클라이언트 분리형** 설정이다. 논문 본문의 같은 클라이언트 내부 표본 삭제와 구분한다.

```bash
git fetch origin refs/heads/codex/structural-latent-defense:refs/remotes/origin/codex/structural-latent-defense
git switch codex/structural-latent-defense
git pull --ff-only origin codex/structural-latent-defense
python -m unittest discover -s FUBA/tests -q

# 새 BadFU 학습 → 별도 저장 → 무방어/정상 손실 제어/장력 제어 자동 실행
python BadFU_src/BadFU-main/request_pipeline.py run --run-dir artifacts/badfu/request_s42 --seed 42 --out FUBA/logs/badfu_request/request_s42.json
```

기존 `BadFU_src/BadFU-main/data`와 `record/badnet_dataset/pert_result.pt` 및 대응 이미지들을 읽는다. 다른 위치면 `--data-root <CIFAR 데이터 폴더> --record-root <badnet_dataset 폴더>`를 지정한다. CIFAR와 공격 자료를 몰래 새로 만들거나 clean 데이터로 치환하지 않는다. 원래 공격 데이터 생성은 아래 구 실행 절을 참고한다. ImageNet 사전학습 가중치는 torchvision 캐시를 재사용하며, 없으면 torchvision이 내려받는다.

기본 FL은 40라운드·로컬 5epoch·batch64·SGD .01·dominant .7·seed42다. 각 클라이언트 배치 순서는 기록된 seed/round/client 규약으로 고정하므로 과거 스크립트와 비트 단위 동일한 학습은 아니다. 기본 FU는 매 기록 라운드에서 보존 클라이언트당 **2배치 재보정**이며, `--fu-batches 0 --fu-epochs 5`로 전체 로컬 5epoch 비교를 할 수 있다. 기본 요청 `5 1`은 각각 별도로 수행하는 위장자/정상 요청이고, 공격 역할은 제어기에 들어가지 않는다.

GPU 한 대에서는 학습 완료 직후 FU 비교를 차례로 실행한다. 한 명령으로 둘 다 수행하며 결과가 서로 덮어써지지 않는다. 시드별 `request_s43`, `request_s44`처럼 폴더와 결과 이름을 바꾼다. 같은 설정의 완성된 학습 폴더는 FL을 재사용하고, 부분 폴더나 다른 설정은 자동 삭제하지 않고 중단한다. `--out`이 이미 있으면 새 이름이 필요하다.

저장되는 자료:

- `artifacts/badfu/request_s42/manifest.json`: 설정·데이터/코드 해시·가중치·라운드 목록·FL ACC/ASR.
- `initial.pt`: 실제 학습 전 모델(파라미터와 BN 상태).
- `data.pt`: **실제로 FL에 사용한 클라이언트별 clean/poison/camouflage 입력·라벨**, 검증/평가 입력, 분할 인덱스. 이후 FU는 이 자료를 그대로 읽는다.
- `round_0000.pt` 등: 라운드 전 전역 모델·6개 로컬 모델·집계 후 전역 모델·참여자와 샘플 수.
- `fu/<결과이름>/request_5_temporal.pt` 등: 해당 FU 비교가 완료된 경우의 모델. 제어 실패군에는 성공 모델을 만들지 않는다.
- `FUBA/logs/badfu_request/*.json`: 비교군별 ASR/ACC/클래스 ACC·제약 검사·실패 사유·시간·정규화만의 효과.

ResNet-18 전체 파라미터와 40라운드 로컬 상태 때문에 새 학습 궤적은 약 15GB에 데이터 묶음 용량이 추가된다. 큰 `.pt`는 gitignore 대상이며 PC에 보존한다. 결과 JSON만 GitHub에 올린다. 계산은 전체 파라미터의 행공간을 청크 단위로 계산하고 랜덤 sketch/top-k를 쓰지 않는다. 이력 창은 메모리 비용을 고려해 기본3이며 `--history-window 5` 등은 별도 설정으로 보고한다.

### 기존 BadFU 궤적 재사용: 공격 FL 재학습 없음

`round_0.pt` 안의 `gm`이 실제 초기 모델이므로 FUBA의 오래된 round_0 누락 문제와 다르다. 기존 15라운드 자료에는 다음처럼 **원래 seed·분할·부분집합 설정**을 맞춘다. 아래 경로는 DISLAB에서 실제 위치로 바꾼다.

```bash
python BadFU_src/BadFU-main/request_pipeline.py run --legacy-traj BadFU_src/BadFU-main/logs/traj_full --rounds 15 --seed 42 --run-dir artifacts/badfu/imported_full_s42 --step 14 --out FUBA/logs/badfu_request/imported_full_step14.json

# 등록과 데이터 보존 후, 초기 상태부터 전체 FU 비교 (FL은 다시 학습하지 않음)
python BadFU_src/BadFU-main/request_pipeline.py test --run-dir artifacts/badfu/imported_full_s42 --out FUBA/logs/badfu_request/imported_full_replay.json

# 같은 저장 이력에서 알고리즘 구성요소 비교
python BadFU_src/BadFU-main/request_pipeline.py test --run-dir artifacts/badfu/imported_full_s42 --step 14 --modes no_defense retention_only linear cumulative unsmoothed temporal --out FUBA/logs/badfu_request/imported_full_ablation.json
```

`--step 14`는 0부터 세는 마지막 실제 FU 보정 **한 단계** 검사다. 전체 망각 실험으로 집계하지 않는다. legacy import는 큰 원본 궤적을 복사하지 않고 경로·해시로 연결하며 원본을 변경하지 않는다. 따라서 원본 폴더를 지우지 않는다. 데이터는 명시한 원래 설정으로 재구성해 보존하고, 저장 집계가 샘플 수 가중 FedAvg와 맞는지 검사한다. 과거 데이터 자체의 해시가 없으므로 **옛 학습 데이터와의 완전한 동일성은 미인증**이라고 결과에 적힌다. 새 학습 경로는 FL/FU가 같은 `data.pt`를 쓴다.

### BN 정책과 결과 해석

모든 비교군은 매 FU 단계 시작 시 같은 정상 검증 입력으로 BN 통계를 다시 구하고, 그 단계의 파라미터 최적화·실제 손실 검사·평가 동안에는 고정한다. 따라서 제어기의 손실 기울기는 같은 고정 BN 상태에서 계산된다. 정규화 자체가 ASR/효용을 바꿀 수 있어 `starting_metrics`, `after_initial_bn_recalibration`, `bn_recalibration_loss_change`를 별도로 기록한다. 클래스 손실 예산은 **그 단계의 파라미터 갱신**에 적용되며, 단계 사이 BN 변경이나 전체 누적 손실의 보장은 아니다. 원 저자의 층별/격라운드 FU와 다른, 클라이언트별 전체 벡터 FedEraser + 명시적 BN 정책이다.

`control_not_accepted`는 실패다. `failure.loss_trials`에 실제 손실 변화/진행량을 기록하며 실패 모델을 성공 ASR로 집계하지 않는다. `forgetting_verified=false`는 항상 유지한다. 트리거와 표적 라벨은 평가에만 쓰고, 보존 데이터에는 실제 poison을 남긴다. 데이터 생성·장기 FL·FU의 실행 시간은 환경 의존적이며 한 단계 검사를 먼저 한 뒤 전체를 실행한다.

현재 로컬 검증: 113개 unittest 통과. 실제 CIFAR/ResNet의 축소2라운드 저장·자동 FU와 기존15라운드 궤적의 step14를 실행했다. 수정 후 `traj_full_step14_eval_fixed.json`에서 요청5 무방어 ASR91.02/ACC75.98%, temporal45.12/86.13%였고, 같은 보존제어는 불수락이었다. 정상요청1에서는 두 제어군 모두 불수락이었다. 해당 시작 모델 ASR21.88%→BN만 재구성36.72%→temporal45.12%이므로 완전 방어/망각 성공이나 이력의 확증 효과로 읽지 않는다. 앞선 eval 모드 복원 전 세 개발 로그는 성능 근거에서 제외한다. 상세 기록은 RESULTS_LOG R-047–R-050.

```bash
# 결과 업로드: 큰 data.pt/모델은 로컬에 남긴다.
git add FUBA/logs/badfu_request
git commit -m "Record BadFU request-control experiments"
git push origin codex/structural-latent-defense
```

## 개정안: 요청 조건부 상쇄 연산자와 제약 최적화

명세·성질·현재 음성 결과: [TENSION_CONTROL §7](https://github.com/sijoon-sung/BadFU_defend/blob/4c4c38f19d969fa776f75064a112543d8ce2657d/docs/paper/TENSION_CONTROL.md#7-개정-알고리즘-요청-조건부-상쇄-연산자와-제약-최적화).
새 선택기는 원시 FL 업데이트와 요청만으로 G를 만들며 top-k·트리거·공격자 정답을 사용하지 않는다.
η=.1은 상쇄 노출 **노름** 예산, ε=.01은 클래스별 validation CE의 nats 예산, γ=.1은 제안 방향 진행량이다. ASR·정확도·망각 비율과 다르다.
기본값은 개발 설정이며 확증 시험의 ASR을 보고 조정하지 않는다.

### 1. 기존 궤적에서 바로 실행하는 동작·기준선 비교

아래 실행은 실제 보정 한 단계다. 같은 상태에서 retention_only / linear / cumulative / unsmoothed / temporal / time_shuffled를 비교하며, 기존 기능 관측 제어도 함께 기록한다.
정상 요청8은 이 예시의 공격 설정에 대한 대조이며 다른 실험에서는 그 실험의 정상 요청자를 지정한다.
독립 normal-only 모델과 owner 분할에서도 반복해야 정상 오탐·효용 주장을 할 수 있다.

    git fetch origin refs/heads/codex/structural-latent-defense:refs/remotes/origin/codex/structural-latent-defense
    git switch codex/structural-latent-defense
    git pull --ff-only origin codex/structural-latent-defense
    python -m unittest discover -s FUBA/tests -q
    NAME=loc_k20paper_s0
    python FUBA/experiment_tension_control.py --name "$NAME" --round 99 --requesters 4 --eval-target 8 --geometry --geometry-strengths 0 --geometry-risk-ratio .1 --out "FUBA/logs/request_geometry/$NAME-req4-local.json"
    python FUBA/experiment_tension_control.py --name "$NAME" --round 99 --requesters 8 --eval-target 8 --geometry --geometry-strengths 0 --geometry-risk-ratio .1 --out "FUBA/logs/request_geometry/$NAME-req8-local.json"

기존 경로가 다르면 --checkpoints /path/to/checkpoints --data-root /path/to/data를 더한다. 원래 seed·분할 옵션은 아래 구 실행 명세와 같다. 출력은 덮어쓰지 않는다. 관측 창20·평활1·최소 진행.1·노름 비율.25를 고정하고, 독립 실험 전에 바꿀 예산을 미리 정한다. --eval-target은 평가 전용이다.

출력의 control_not_accepted는 실패로 집계한다. 해당 군의 ASR을 누락한 채 성공군만 평균내지 않는다. KKT 잔차/목적 오차 상계는 QCQP의 수치 해에 대한 검사다. backtracks, accepted_progress, actual_loss_change는 실제 수용한 갱신에 대한 값이며, risk_after·step_norm 등 solver 값은 backtracking 전일 수 있다.

현재 같은 모델·99단계 결과: 요청4의 temporal/보존 제어 ASR은 모두11.14%, 요청8에서는 모두11.79%; ACC는 모두99.02%. 상쇄 노출 제약은 충족했지만 추가 방어 이득은 미확인이다. 이전 λ=1/10 벌점 실험 및 solver 보완 전 실행도 로그에 보존했다. 두 요청은 독립 시드가 아니다.

### 2. 전체 FU 기록을 실제로 재구성하는 실행

실행기는 FUBA/experiment_request_replay.py이다. 다음 두 자료를 **실제로 확보한 경우** 실행한다.

- 진짜 학습 전 global_net_model_NAME_round_0.pkl. 이번부터 train_main.py가 저장한다. 이전 궤적에 없으면 학습 후 checkpoint를 대신 넣지 않는다. 정상 궤적도 라운드별 local 기록을 저장하도록 수정했다.
- retained.npz: images(float32, N×1×28×28), labels(정수 N), units(정수 N, 원래 client ID). 삭제 대상은 빠지고 모든 보존 참여자의 실제 데이터가 들어 있어야 한다. 남겨야 할 poisoned/backdoor 표본도 그대로 넣는다. 깨끗한 MNIST로 바꾼 자료는 방어 효과 실험에 사용할 수 없다. 전처리·표본 선택 출처를 실험 기록에 남긴다.

아카이브 작성 형식은 아래와 같다. X_retained, y_retained, client_ids_retained는 해당 실험에서 실제로 보존하는 자료다. 이 변수에 들어갈 공격 데이터를 자동 추정하거나 교체하는 코드는 아니다.

    np.savez_compressed("retained.npz",
                        images=X_retained.astype(np.float32),
                        labels=y_retained.astype(np.int64),
                        units=client_ids_retained.astype(np.int64))

    python FUBA/experiment_request_replay.py --name "$NAME" --rounds 100 --requesters 4 --retained-archive retained.npz --eval-target 8 --out "FUBA/logs/request_geometry/$NAME-full.json"

기본6군은 무방어/보존제어/선형/누적/무평활/temporal이다. 각 군이 초기 상태부터 따로 진행하며 같은 retained data와 난수 규약으로 실제 SGD 보정을 재계산한다. 실패한 replay는 즉시 중단하고 candidate_metrics를 null로 기록한다. 초기 체크포인트는 --initial로 명시할 수 있지만 진짜 학습 전 상태라는 조건은 같다.

이 실행기는 MNIST/LeNet, 동일 가중 FedAvg, client 전체 삭제, 고정된 retained archive 조건이다. FUBA의 FU 중 공격자 재최적화·재주입을 자동 재현하지 않는다. 원래 BadFU의 같은 client 내부 표본 삭제와 BN 모델에는 별도 어댑터가 필요하다. 시작 모델의 안전성·초기값 진위·아카이브 내용은 파일 해시만으로 증명되지 않는다.
forgetting_verified는 항상 false이며, 삭제 전 기록으로 만든 제어의 잔존 의존성을 별도 검증해야 한다.

**전달할 결과:** 위 JSON, 데이터·요청·분할·시드·예산 명세. 다음 판정은 모든 군의 전체 FU 완료율과 ASR–ACC–최악 클래스 ACC–망각 비교다. 같은 방어 목적의 보존 데이터 재학습도 별도 기준선으로 필요하다. 현재 로컬에는 원래100라운드 궤적의 진짜 round_0 및 대응 retained archive가 없어 전체 공격 replay는 실행하지 않았다. 2라운드 PyTorch 합성 자료 통합검사로 실행 경로를 검증했고, 이를 백도어 실험으로 집계하지 않는다.

## 최신: 실제 FU 보정 단계와 관측 상한 검사

[알고리즘 명세](https://github.com/sijoon-sung/BadFU_defend/blob/4c4c38f19d969fa776f75064a112543d8ce2657d/docs/paper/TENSION_CONTROL.md), [방어 6편 + FedEraser의 다섯 항목 검토](https://github.com/sijoon-sung/BadFU_defend/blob/4c4c38f19d969fa776f75064a112543d8ce2657d/docs/paper_reviews/05_defenses_novelty.md). `tension_control.py`와 `federaser_clientwise.py`에 실제 제어·개별 클라이언트 보정 후 집계를 구현했다. 아래의 예전 ‘제어 미구현’ 설명은 이 추가로 갱신한다. 기존 평균 후 층별 보정 파일은 과거 결과 재현용으로 보존한다.

**범위:** 저장 MNIST/LeNet 궤적에서 실제 보존 클라이언트의 SGD 보정 한 단계를11군으로 비교한다. 전체 FU는 아니다. `global_(round-1)`에는 과거 삭제 기여가 남아 있어 망각 완료를 주장하지 않는다. 최초 밝기 변환 스모크만으로 장력 원리의 실패나 성공을 판정할 수 없다. 정답 트리거 관측은 관측 상한이며 실용 방어 성능으로 보고하지 않는다.

저장소 루트, 기존 PyTorch·torchvision·NumPy·SciPy 환경에서 실행한다. MPI·timm은 필요 없다. 원래 FL과 같은 seed/분할이어야 한다. MNIST는 기존 `FUBA/data`에서 읽고 자동 다운로드하지 않는다.

```bash
git fetch origin refs/heads/codex/structural-latent-defense:refs/remotes/origin/codex/structural-latent-defense
git switch codex/structural-latent-defense
git pull --ff-only origin codex/structural-latent-defense
python -m unittest discover -s FUBA/tests -q

# 실제 보유한 checkpoint 이름. 예: 로컬 seed522 IID 모델
NAME=loc_k20paper_s0
python FUBA/experiment_tension_control.py --name "$NAME" --requesters 4 \
  --round 99 --history-window 20 --seed 522 --split iid --eval-target 8 \
  --out "FUBA/logs/tension_control/${NAME}_r99_req4_brightness.json"

# 정답 트리거 관측 상한. 위와 FU 갱신은 같고 관측 입력만 다르다.
python FUBA/experiment_tension_control.py --name "$NAME" --requesters 4 \
  --round 99 --history-window 20 --seed 522 --split iid --eval-target 8 \
  --probe-view oracle-trigger \
  --out "FUBA/logs/tension_control/${NAME}_r99_req4_oracle.json"

# 정상 요청 대조. 8은 이 로컬 궤적의 정상 참여자다.
python FUBA/experiment_tension_control.py --name "$NAME" --requesters 8 \
  --round 99 --history-window 20 --seed 522 --split iid --eval-target 8 \
  --out "FUBA/logs/tension_control/${NAME}_r99_req8_brightness.json"
```

`--eval-target`은 평가에만 쓰며 선택기·제어기에 전달하지 않는다. `--probe-view oracle-trigger`는 저장 `backdoor_net_model`로 변환한 입력을 관측하므로 정답 정보 사용 실험이다. 전체 평가를 포함하면 generator의 `--noise-thread`, `--atk-eps`도 원래 설정(.04,1.0 기본)과 맞춰야 한다.

저장 위치는 `--checkpoints /path/to/checkpoints --data-root /path/to/data`로 바꾼다. IID seed는 원래 값(예:522/523/524), owner는 `--split owner --owner-client 19 --owner-class 3 --owner-frac .9 --split-seed <원래값>`을 지정한다. 먼저 보유한 IID/owner 각 한 모델에서 요청·시점을 바꾸고, 정합성이 확인되면 나머지 시드로 확대한다. 출력은 덮어쓰지 않으므로 설정 변경 시 새 이름을 쓴다.

기본 선택32개/보존128개/평가512개는 분리되어 있다. 보존 집합에 클래스가 빠지면 `--retain`을 늘린다. `--topk 8 --strength 1 --radius-ratio .25 --epsilon 0 --batches 2`는 사전 고정 동작 확인값이며 최적 설정이 아니다. 현재는 buffer 없는 MNIST/LeNet 어댑터다. BN 상태를 무시한 CIFAR 실행은 지원하지 않는다.

| JSON 비교/필드 | 확인할 질문 |
|---|---|
| `history_action` vs `history_free` | 실제 제안만 본 같은 제어보다 이력+경계가 도움이 되는가 |
| `history_action` vs `linear_action` | 단순 선형 삭제 효과와 같은가; 완전 상쇄에서는 동률이 이론상 가능 |
| `history_action` vs `shuffled_action` | 실제 삭제 단위와 이력의 대응이 중요한가 |
| `history_action` vs `history` | 과거 B만 쓰는 것보다 실제 제안의 증가 A를 결합하는 이유가 있는가 |
| `tension`, `linear`, `retention_only`, `clip`, `no_defense` | 다른 부품만으로 효과가 생기는가 |
| `before_step`, `step_asr`, `clean_target_rate`, `macro_trigger_excess` | 해당 한 단계가 실제 백도어를 활성화하는가. 활성화가 없으면 방어 성능 비교는 판정 유보 |
| `progress`, `status`, 실제 클래스 손실 변화 | 정지 또는 정상 효용 붕괴로 ASR을 낮추지 않았는가 |

**전달 파일:** 생성 JSON들. 원본 데이터·checkpoint는 불필요하다. 코드/체크포인트/분할 해시, 참여자, 관측·보정·제어 시간도 저장한다. 이 작은 단계를 먼저 보는 이유는 구현과 관측 가정을 분리하기 위해서이며, 전체 FU의 ASR·망각 검증을 대신하지 않는다.

전체 FU API는 `federaser_clientwise.replay(initial, events, deleted, recalibrate, controller)`. 진짜 학습 전 초기 상태·실제 참여 기록·보존 데이터 callback이 필요하다. 원래 BadFU의 같은 클라이언트 내부 삭제, BN 어댑터, 엄격한 망각 인증 및 대규모 FU 통합은 아직 미완료다. 다음 평가 순서는 [설계 §5](https://github.com/sijoon-sung/BadFU_defend/blob/4c4c38f19d969fa776f75064a112543d8ce2657d/docs/paper/TENSION_CONTROL.md)에 있다.

---

**기존 v3 관측 실험:** 방어 대상은 FL→FU 잠복 백도어다. 아래18/68개 실행기는 정상 혼동과 기존 점수의 관측 비교용이다. 위 실제 제어의 관측·연산자 정합성을 먼저 확인한 뒤 대규모 독립 학습을 확대한다.

**후속 FU 실험의 선행 조건:** 로컬 `federaser_faithful.py`의 평균 후 층별 보정은 원논문의 클라이언트별 보정 후 집계와 일반적으로 다르다. 부분 참여 처리와 함께 연산자 정합성을 먼저 확인한다. 아래 장력 캐시/수집 실행은 이 FU 코드를 호출하지 않는다.

## 이번에 실행할 실험: 장력과 선택적 삭제율 (v3)

목적은 **FL 장력에 FU의 삭제 경계를 붙이면 단순 삭제 영향 점수보다 나은가**를 반증 가능한 형태로 검사하는 것이다. 주 알고리즘은 후보 언러닝 모델을 만들지 않는다. 수학, 반례, 한계는 [현재 v7](https://github.com/sijoon-sung/BadFU_defend/blob/4c4c38f19d969fa776f75064a112543d8ce2657d/docs/paper/draft_v7.tex) §§4–9에 반영했다.

이전 v2는 `(주입, 억제)=(10,-8)`을 양쪽 모두 80% 제거해도 양의 점수를 냈다. v3는 `B=H min(P,N) [N_D/N-P_D/P]_+`로 수정했다. **동일한 기여 제거율이면 불완전 상쇄에서도 0**이다. `r=.2`는 기록된 기능 기여의 20% 제거 가정이며, 임의 표본 20% 삭제와 같다고 가정하지 않는다.

### 1. 코드 받기와 빠른 재생 (저장소 루트, Git Bash)

```bash
git fetch origin refs/heads/codex/structural-latent-defense:refs/remotes/origin/codex/structural-latent-defense
git switch codex/structural-latent-defense
git pull --ff-only origin codex/structural-latent-defense
python -m unittest discover -s FUBA/tests -v
python FUBA/replay_tension.py
python FUBA/benchmark_tension.py
```

결과: `FUBA/logs/tension_audit/cache_replay_v3.json`, `historical_ablation_v3.json`.
기존 학습 캐시 6개만 재생한다. 새 학습·FU 호출은 0회다. 마지막 시점의 요청자·표적 동시 1위는 v3 **6/6**, v2 **5/6**, 단순 선형 삭제 영향 **4/6**이었다. 이미 관찰한 개발 자료이므로 이 차이를 정상 오탐률이나 확인 실험으로 보고하지 않는다.

### 2. 먼저 돌릴 것: 예비 묶음 하나

기존 FUBA 학습이 되던 **CUDA PyTorch·torchvision·mpi4py·MPI 환경의 Git Bash**에서 실행한다. `run_real_fuba.sh`가 기존 환경 선택 규칙을 사용하므로 올바른 Python 환경에서 시작한다. 처음 다운로드가 필요한 경우 MNIST 접근도 필요하다.

```bash
cd FUBA
python experiment_tension.py --profile pilot --stage plan
python experiment_tension.py --profile pilot --stage all
```

`pilot`은 IID/owner 각각 정상 보정 3개, 정상 시험 3개, 기존 공격 3개, 총 **18개 독립 학습 궤적**이다. 기존 공격 체크포인트가 있으면 재사용하므로 새 정상 학습은 **12개**다. 공격 체크포인트가 없는 PC에서는 공격 6개도 생성한다. 전부 K=20, 100라운드, 참여율 0.4의 MNIST 설정이다. 정상은 warmup=1000으로 공격을 비활성화한다. 보정 seed index=100–102, 정상 시험=200–202, 개발 공격=0–2로 분리했다.

실행 순서: 궤적 생성 → 기존 실제 양의 학습 업데이트의 입력별 기능 효과 수집 → 5개 점수 비교 → 정상 보정/시험 요약. 공개 입력 128개를 모든 방법이 공유하며, 트리거·공격 역할은 수집기에 전달하지 않는다. 역할 정답은 모든 점수 계산 후 채점에만 사용한다. 입력당 로그확률/중심화 로짓을 기록한다. **JVP나 FU 결과의 수집기가 아니다.** `FedAvg` 평균 갱신과 저장 전역 갱신이 맞지 않으면 중단하므로 robust aggregation 궤적을 이 경로에 넣지 않는다.

중간 파일은 이름이 충돌하면 덮어쓰지 않는다. 완료 궤적은 재사용하고, 수집 캐시는 설정·코드·체크포인트 SHA256까지 일치해야 재사용한다. 궤적 일부만 있으면 자동으로 처음부터 덮어쓰지 않고 중단한다. 그 궤적을 보존하여 원인을 확인한 뒤 재실행한다. 코드/설정이 바뀌면 `--out logs/tension_v3/pilot_revision2`처럼 새 출력 폴더를 지정한다.

단계를 나누어 실행해도 된다.

```bash
python experiment_tension.py --profile pilot --stage train
python experiment_tension.py --profile pilot --stage collect
python experiment_tension.py --profile pilot --stage analyze
```

### 3. 결과에서 확인할 것

| 질문 | 비교/판정 기준 |
|---|---|
| 장력을 붙인 것이 실제로 도움 되는가? | 주 비교는 `logprob/mean_before/window:0`의 `selective_v3` 대 `linear_release`. 정상 경보 수준과 요청자·표적 식별을 함께 비교한다. |
| 기존 수식 수정이 필요한가? | `cross_v2`와 비교하고 동일 비율 삭제 대조를 확인한다. 동작검사는 비균형 반례를 포함한다. |
| 요청 경계가 필요한가? | `tension_only`는 요청과 무관하므로 같은 점수의 임의 동점 순위를 성능으로 세지 않는다. |
| 큰 음의 기여만 찾은 것 아닌가? | `removed_negative`와 비교한다. 단독 삭제에서는 `linear_release`와 값이 같은 좌표가 생긴다. |
| 평균/시간/좌표 때문에 신호를 잃는가? | `mean_before` 대 `mean_after`, 누적창 0 대 최근 10/20전이, 로그확률 대 중심화 로짓. 후자는 사전 등록된 보조 분석이다. 시험 결과를 보고 가장 좋은 설정만 골라 주 결과로 바꾸지 않는다. |
| 정상 학습에서 거짓 경보가 생기는가? | IID/owner의 정상 독립 학습을 분리해서 본다. 한 모델의 클라이언트 20명이나 입력 128개를 독립 정상 모델 20/128개로 세지 않는다. |

`summary.json`의 `alarm`은 어떤 요청이든 경보가 난 에피소드다. 공격 에피소드에서는 `requested_suppressor_alarm`, `requested_suppressor_target_alarm`도 별도로 보므로 정상 클라이언트 경보를 공격 성공 검출로 세지 않는다. 요청 가족은 모든 단독 기여의 .2/1 제거와 모든 단위의 .2/.8/1 동률 제거이며, 시점·클래스·요청 전체 최대값을 보정한다. 이 유한 가족 밖의 임의/적응적 요청에 같은 보장을 적용하지 않는다.

**예비 묶음에서는 5% 오탐률 문턱을 만들 수 없다.** 조건별 정상 보정 모델 3개는 부족하므로 `insufficient_clean_episodes`, `threshold: null`, `alarm: null`이 정상 출력이다. 이를 오탐 0으로 읽지 않는다. 먼저 점수의 정상/공격 분포와 부호·좌표 차이를 검토할 자료를 모은다.

### 4. 예비 결과가 유망할 때 실행할 확인 묶음

```bash
python experiment_tension.py --profile confirm --stage plan
python experiment_tension.py --profile confirm --stage all
```

조건별 정상 보정 19개(seed 100–118), 정상 시험 10개(200–209), **새 공격 5개(20–24)**, 총 68개 궤적이다. 예비와 겹치는 정상 궤적은 재사용한다. 19개는 5% 상위 순위 문턱을 유한하게 만들 수 있는 최소 수이며, 충분한 통계적 검정력이나 정상 시험 FPR의 정밀도를 뜻하지 않는다. 정상 에피소드의 교환 가능성을 전제하며 분포가 달라지면 보장을 주장하지 않는다.

**중단/수정 기준:** 정상 충돌에서 v3와 단순 기준의 분리가 비슷하거나 확인 자료의 동일 경보 수준에서 이득이 없으면, 복잡한 점수 자체의 신규성 주장은 내린다. 입력별 기록만 효과가 있다면 기여를 기록 해상도로 좁힌다. 새 공격의 잠복/활성화가 재현되지 않으면 그 점수를 방어 탐지율로 세지 않는다.

### 5. 돌린 뒤 전달할 파일과 아직 별도인 실험

`FUBA/logs/tension_v3/pilot/` 폴더를 전달하면 된다. 핵심 파일은 `manifest.json`, `summary.json`, `scores.json`, `caches/*.json`이고, 좌표/요청 재분석에는 `caches/*.npz`도 필요하다. 실패했으면 `commands/*.log`를 함께 전달한다. 체크포인트·원본 데이터는 업로드할 필요 없다. 확인 묶음은 같은 구조의 `confirm/`에 저장된다.

이번 실행 묶음은 **FUBA의 사전 점수와 정상 혼동을 검증**한다. 원래 BadFU의 같은 클라이언트 내부 backdoor/camouflage 기록, 실제 FU 후 ASR, 망각, 새 점수 기반 복구까지 끝난 것으로 보고하지 않는다. BadFU를 클라이언트 0/5로 분리한 기존 변형은 이 검증의 대체물이 아니다.

다음 단계의 BadFU 실험에는 (a) 원래 같은 클라이언트에서 두 역할을 학습, (b) 역할 정답을 노출하지 않는 사전 고정 기록 단위, (c) 삭제 요청을 표현하는 해상도, (d) 자체 보고를 믿는 것과 독립 계측을 구분하는 신뢰 조건이 필요하다. oracle 역할 기록은 상한 실험으로만 쓴다. 클라이언트 합만 보이는 경우는 관측 불충분으로 처리하며 구조적 공통성을 부정하는 근거로 삼지 않는다. 원고에 이러한 설계 조건과 실제 FU·복구·적응형 공격 실험을 별도로 명시했다.

로컬 검증: 동작검사 69개 통과, 기존 실제 저장 모델에서 2개 FL 전이·공개 입력 16개의 수집 스모크 통과. 전체 새 학습과 정상 보정 결과는 아직 실행 전이다.

---

## 이전 실험 실행 기록 (이번 v3 묶음과 구분)

**보조 비교군:** `audit --components response`는 언러닝 후 soft 흐름 검사다. 기본값 `existing`은 기존 DOSE다. 이84조건의 재실행은 [이전 재설계 기록](https://github.com/sijoon-sung/BadFU_defend/blob/4c4c38f19d969fa776f75064a112543d8ce2657d/docs/paper/DEFENSE_LOGIC_REDESIGN.md) §6에 있으며 현재 연구의 핵심 신규성으로 제시하지 않는다.

공통 잠복형 방어의 실험용 확장은 [STRUCTURAL_GUARD.md](https://github.com/sijoon-sung/BadFU_defend/blob/4c4c38f19d969fa776f75064a112543d8ce2657d/docs/paper/STRUCTURAL_GUARD.md)의 실행 절을 따른다.
요청 조건부 공동삭제 검증은 기존 `audit`에 `--paired_confirm --step 0.05`를 추가한다.
새 실제/합성 실험과 ASR 평가의 재실행은 [개정 보고서](https://github.com/sijoon-sung/BadFU_defend/blob/4c4c38f19d969fa776f75064a112543d8ce2657d/docs/paper/REVISION_REPORT.md) §6에 있다.
정책 테스트와 캐시 재생은 `python -m unittest discover -s FUBA/tests -v`,
`python FUBA/run_structural_guard.py monitor --cache <push.npz> --out <result.json>`으로 저장소 루트에서 실행한다.
FUBA/BadFU 체크포인트 감사는 `audit` 하위 명령을 사용하며 실제 재학습·망각 검증 없이 삭제 완료를 선언하지 않는다.

아래는 기존 DISLAB 궤적 및 균형 장부의 `dev` 실행 기록이다. 아래 명령은 `FUBA` 폴더 기준이며 새 감사 후보의 저장소 루트 명령과 구분한다. 결과 번호는 `docs/RESULTS_LOG.md`.

```bash
git fetch origin && git checkout dev && git pull
cd FUBA
```

## 지금 돌릴 것 — DISLAB 은 한 줄
```bash
ONLY="k20_paper_cifar clean_paper" bash make_trajectories.sh     # CIFAR 논문 설정 3 개 + 공격 없는 궤적 4 개. 하나 끝날 때마다 검증기가 돌고 docs/EXPERIMENT_LOG.md 에 적힌다
```
올릴 것: `git add FUBA/logs/trajectories docs/EXPERIMENT_LOG.md`. 첫 CIFAR 궤적 검증이 "잠복 ACC < 50" 으로 멈추면 알릴 것 (라운드당 배치 수 `--train_epoch` 를 늘린 변형이 필요). 그 사이 아래 표의 나머지는 노트북 작은 설정에서 코드 경로를 확인한 것만 순서대로 넘긴다.

노트북 작은 설정 확인 (2026-10-13 저녁): 공격 없는 MNIST K=8/8 라운드 → 검증기 clean 모드 ✔ (ACC 90.1, ASR 없음, 정상 제거 88.8) → 장부 경보 없음 ✔ (H ≤ 0.23). CIFAR 2 라운드 → 저자 코드 CIFAR 경로 ✔, 검증기 CIFAR 로더·UNet 생성기 ✔, 장부 ✔. 되감기 대응 ✔ (R-014). 아직 안 본 것: 위장자 2 명 `--neg_set`(sup2 궤적이 노트북에 없음), 집계 방어, BadFU 40 라운드.

| # | 무엇 | 명령 (FUBA 폴더) | 나오는 것 | 올릴 것 | 시간 | 노트북 확인 |
|---|---|---|---|---|---|---|
| 3 | **장부 — CIFAR** (R-018) | `DATASET=cifar10 TAG=_r100w5_k20paper_cifar CONDS=iid bash run_ledger.sh` | `logs/ledger/summary_bd_iid_r100w5_k20paper_cifar.md` | `git add FUBA/logs/ledger` | 궤적당 5 분 | ✔ 로더·장부 (CIFAR 2 라운드) |
| 4 | **장부 — 공격 없음** (R-015, 기대: 경보 없음) | `PREFIX=clean TAG=_r100_k20paper CONDS="iid owner" SEEDS=0 bash run_ledger.sh` 그리고 `PREFIX=clean TAG=_r100_k20paper_cifar CONDS="iid owner" SEEDS=0 DATASET=cifar10 bash run_ledger.sh` | `logs/ledger/summary_clean_*.md` | 같음 | 궤적당 2–5 분 | ✔ (K=8 공격 없음: 경보 없음) |
| 5 | **대응 — 되감기 + 연합 제외, MNIST 6 궤적** (R-014) | A-03 "대응" 블록의 for 문 한 줄 | `logs/benign_departure/bd_*_excl_from*_s*.json` | `git add FUBA/logs/benign_departure` | 6 × 13 분 | ✔ (R-014, 1 시드) |
| 6 | **운반자 남긴 재학습, 논문 설정** (표 4 첫 행을 K=8 예비에서 바꿈) | `for c in iid owner; do for s in 0 1 2; do own=""; [ $c = owner ] && own="--non_iid --non_iid_type class_owner --owner_class 3 --owner_frac 0.9 --owner_client -1"; python benign_departure.py --name bd_${c}_r100w5_k20paper_s$s --K 20 --rounds 100 --seed $((522+s)) --split_seed $((522+s)) --defender 4 --requesters 4 --methods retrain_benign retrain_attack $own --out logs/benign_departure/bd_${c}_r100w5_k20paper_retrain_s$s.json; done; done` | `logs/benign_departure/bd_*_retrain_s*.json` | 같음 | 6 × 2 × 13 분 | ✔ (저자 재학습 코드, R-010 과 같은 경로) |
| 7 | **위장자 2 명** (R-016) | `for s in 0 1 2; do python ledger_defense.py --fmt fuba --name bd_iid_k16sup2_s$s --K 16 --rounds 8 --defender 4 --defenders 4 8 --attackers 0 1 2 3 5 6 7 9 --target 8 --neg_set --start 3 --cond sup2; done; python ledger_defense.py --aggregate --cond sup2` | `logs/ledger/summary_sup2.md` | 같음 | 10 분 | 미확인 (sup2 궤적 없음) |
| 8 | **집계 방어 위** (tdef_paper 12 궤적; `pip install hdbscan scikit-learn` 먼저) | `ONLY=tdef_paper bash make_trajectories.sh` 뒤 `for d in median multi_krum flame flDetector; do TAG=_r100w5_k20paper_$d CONDS=iid bash run_ledger.sh; done` | `logs/ledger/summary_bd_iid_r100w5_k20paper_*.md` | 같음 | 궤적 12 × 20 분 + 장부 | 미확인 |
| 9 (노트북) | **BadFU 40 라운드** (R-017 재확인) | `cd ../BadFU_src/BadFU-main && bash make_badfu_trajectories.sh` 뒤 `cd ../../FUBA && python ledger_defense.py --fmt badfu --name badfu_s42 --traj_dir ../BadFU_src/BadFU-main/logs/traj_s42 --data_root ../BadFU_src/BadFU-main/data --rounds 40 --K 6 --defender 5 --attackers 0 --target 0 --start 3 --min_present 1 --counts $(python -c "import json;e=[e for e in json.load(open('trajectories.json'))['trajectories'] if e['name']=='badfu_s42'][0];print(' '.join(map(str,e['counts'])))") --cond badfu` | `logs/ledger/badfu_s42.json` | 같음 | 2.5 시간 + 5 분 | 미확인 |
3–4 가 리뷰가 요구한 두 숫자(규모, 오경보), 5–6 이 표 4, 7–9 가 넓힘이다. '미확인' 줄은 노트북에서 코드 경로를 본 뒤에 돌린다. 자세한 설명은 아래 절 (0, 0-b, A-03, A-01).

## 0-a. 노트북 검증용 작은 궤적 가져오기 (DISLAB → 노트북; 코드 경로 확인에만 쓴다)
표의 "미확인" 줄(위장자 2 명, 집계 방어)을 노트북에서 먼저 돌려 보려면 DISLAB 의 작은 궤적 두 묶음이 필요하다. 전부 MNIST K=8/16, 8 라운드라 합쳐서 약 400 MB.
```bash
# DISLAB (FUBA 폴더): 압축
cd checkpoints && zip -q ../small_traj.zip *_bd_iid_k16sup2_s*_round_*.pkl *_bd_iid_{median,multi_krum,flame,flDetector}_s*_round_*.pkl && cd .. && ls -la small_traj.zip
```
```bash
# 노트북 (FUBA 폴더): 풀기 (checkpoints/ 에 그대로 들어간다; git 에는 안 올라감)
unzip -o small_traj.zip -d checkpoints/
```
노트북에서 돌릴 확인 (내가 돌린다): 위장자 2 명 `for s in 0 1 2; do python ledger_defense.py --fmt fuba --name bd_iid_k16sup2_s$s --K 16 --rounds 8 --defender 4 --defenders 4 8 --attackers 0 1 2 3 5 6 7 9 --target 8 --neg_set --start 3 --min_present 1 --cond sup2; done`, 집계 방어 `for d in median multi_krum flame flDetector; do for s in 0 1 2; do python ledger_defense.py --fmt fuba --name bd_iid_${d}_s$s --K 8 --rounds 8 --defender 4 --attackers 0 1 2 3 --target 8 --start 3 --min_present 1 --cond tdef_$d; done; done`.

## 0. 공격 궤적 미리 만들기 (한 번만, 다른 실험보다 먼저)

실험은 저장된 궤적 파일만 읽는다. 궤적은 저자 코드가 만든다 (FUBA 는 `main.py`, BadFU 는 저자 루프를 옮긴 `fl_detect_badfu.py`). 우리는 flag(클라 수·참여율·라운드·시드·방어)만 정한다.
만들 목록은 `FUBA/trajectories.json` 하나에 다 있다. 하나 만들 때마다 `verify_trajectories.py` 가 검사하고, 틀리면 거기서 멈춘다.
결과는 `docs/EXPERIMENT_LOG.md` 에 자동 기록된다 (`FUBA/logs/trajectories/registry.json` 과 같이 커밋).

### DISLAB (FUBA, MNIST)
```bash
cd FUBA
DRY=1 bash make_trajectories.sh                        # 명령만 보기 (아무것도 안 만듦)
ONLY=k20_reduced bash make_trajectories.sh             # K=20 축소판 (8 라운드) 6 개
ONLY=k20_paper   bash make_trajectories.sh             # 논문 설정 (K=20, 참여 0.4, 100 라운드) 6 개
ONLY=sup2        bash make_trajectories.sh             # 위장자 2 명 (K=16) 3 개
ONLY=tdef_paper  bash make_trajectories.sh             # 집계 방어 4 종 × 3 시드, 논문 설정. flame 은 먼저 pip install hdbscan
FORCE=1 ONLY=tdef_reduced bash make_trajectories.sh    # K=8 방어 12 개. 예전 것은 방어 없이 만들어졌을 수 있어 FORCE 로 갈아엎는다
bash make_trajectories.sh                              # 위 전부 (있는 것은 건너뜀). k8_main 6 개는 이미 있음
```
| 그룹 | 개수 | 개당 | 전체 |
|---|---|---|---|
| k8_main (있음) | 6 | 6 분 | — |
| k20_reduced | 6 | 10 분 | 1 시간 |
| k20_paper | 6 | 45 분 | 4.5 시간 |
| sup2 | 3 | 12 분 | 40 분 |
| tdef_paper | 12 | 50 분 | 10 시간 |
| tdef_reduced | 12 | 7 분 | 1.5 시간 |

시간은 노트북(RTX 4060) 실측과 논문 Table VI 로 잡은 추정이다. 끊기면 같은 명령을 다시 치면 된다 (다 만든 것은 건너뛴다).

### 노트북 (BadFU, CIFAR-10)
```bash
cd BadFU_src/BadFU-main
DRY=1 bash make_badfu_trajectories.sh
bash make_badfu_trajectories.sh      # 시드 42, 43, 44 를 40 라운드로. 시드당 약 2.5 시간, 디스크 14 GB. 기존 traj_full(15 라운드)은 그대로 둔다
```

### 확인
```bash
cd FUBA && python verify_trajectories.py        # 전부 다시 보기 (파일이 안 바뀌면 캐시라 빠름)
```
`docs/EXPERIMENT_LOG.md` 의 상태 열을 본다. ✔ 만 실험에 쓴다. ✘ 는 사유를 읽고 `FORCE=1 ONLY=이름 bash make_trajectories.sh` 로 다시 만든다. `(임시 문턱)` 이 붙은 그룹은 첫 결과를 보고 `trajectories.json` 의 verify 값을 고친다.
- 이름 규칙: K=8 `bd_iid_s0`, K=20 축소 `bd_iid_k20_s0`, 논문 `bd_iid_r100w5_k20paper_s0`, 방어 `bd_iid_median_s0` / `bd_iid_r100w5_k20paper_median_s0`, 위장자 2 명 `bd_iid_k16sup2_s0`. 이름이 다르면 다른 궤적이다 (없으면 에러, 기본 궤적으로 조용히 넘어가지 않는다).
- 뒤 단계에서 쓰는 법: `K=20 TAG=_k20 CONDS=iid STAGES="cancel aggregate" bash run_all.sh` (논문 설정은 `K=20 ROUNDS=100 WARMUP=5 TAG=_k20paper`). sup2 는 run_all.sh 의 고정 인자(`--attackers 0 1 2 3`)가 안 맞으니 `trajectories.json` 의 `downstream.args` 를 그대로 쓴다. 참여율 0.4 궤적은 결석 클라 처리가 된 로더(`release_map.load_fuba`)만 쓸 것 (`trajectories.json` 의 downstream.note).

## A. 꼭 돌릴 것 (순서대로)

### A-00. 탐지 총정리 벤치 (먼저, 궤적만 있으면 됨, 학습 없음) — "탐지가 잘 되는가" 를 모든 방법으로
한 궤적에서 생각할 수 있는 탐지 점수를 전부 클라이언트마다 매기고(모든 클라이언트를 '요청자라면' 으로), 위장자가 몇 등인지·정상/소유자가 잘못 걸리는지를 한 표로 낸다.
방법: 언러닝 시뮬레이션 절대 기준(기존 요청 심사 발상: 바뀐 답 비율·KL·몰림, ρ = 0.1/0.2/0.5) · 상대 기준(DOSE: flow·mass z) · 기능 효과(크기·몰림·상쇄·Fisher) · 갱신 통계(노름·중앙값 코사인·Krum·총노출 몫·반상관) · null-space 에너지 · 학습 중 부분합 감사(라운드 25/50/75/100 %) · 참여 자연 실험(참여율 < 1) · MASA/SubID proxy(결과 있으면).
```bash
bash run_detect_bench.sh                       # k8 + k20paper 중 있는 궤적 전부 (궤적당 1–3 분)
GROUPS="k20paper" bash run_detect_bench.sh     # 논문 설정만
GROUPS="tdef" bash run_detect_bench.sh         # 집계 방어 궤적 (만든 뒤)
```
결과: `logs/detect/summary_{iid,owner}_r100w5_k20paper.md` (+ `summary_iid.md`, `summary_owner.md`). 올리기: `git add FUBA/logs/detect`.
보는 것: 위장자 순위 1 이면서 소유자 z < 3 인 방법이 무엇인가. 기존 발상(sim_abs_*, upd_*, ns_energy)이 어디서 깨지고 상대 기준(sim_rel_*)·감사(audit_z)·참여 회귀(part_beta_*)가 어디서 갈리는가.

### A-02. 동적 평형 지문 (학습 중 시간 구조로 탐지; 학습 없음, 역할 정보 없음) — 궤적당 1–2 분
잠복 백도어는 매 라운드 유지되는 평형이고 유지에는 시간 위의 흔적이 남는다는 가설. 저장 전역 모델·갱신·참여 명단만으로 세 지문을 잰다: 1 복원력(결석 길이 → 밀기 세기), 2 지연 반작용(타깃 클래스만의 되밀기), 3 균형식(누적 타깃 밀기: 합은 0, 개별은 큼). 타깃 클래스도 서버가 추정한다.
```bash
for s in 0 1 2; do python analysis_equilibrium.py --fmt fuba --name bd_iid_r100w5_k20paper_s$s --K 20 --rounds 100 --defender 4 --attackers 0 1 2 3 --target 8 --cond iid_k20paper; done
for s in 0 1 2; do python analysis_equilibrium.py --fmt fuba --name bd_owner_r100w5_k20paper_s$s --K 20 --rounds 100 --defender 4 --attackers 0 1 2 3 --target 8 --owner_client 19 --cond owner_k20paper; done
python analysis_equilibrium.py --aggregate --cond iid_r100w5_k20paper
python analysis_equilibrium.py --aggregate --cond owner_r100w5_k20paper
```
결과: `logs/equilibrium/summary_{iid,owner}_r100w5_k20paper.md`. 스크립트에 **균형 장부**(클래스별 헤지 비율 H, 음의 다리 z, 양의 다리 집합, H 곡선)가 추가됐으니 같은 명령을 다시 돌리면 된다 (ALGORITHMIZATION.md §3). 보는 것: 지문 3 과 1 에서 위장자 순위 1·소유자 z < 3 인가, 추정 타깃이 8 인가, 운반자가 양의 누적 밀기로 위에 오는가. (노트북 1 시드: 지문 3 위장자 −40 vs 정상 ±4, z −22, 타깃 추정 8 맞음, 순/총 0.08; 지문 1 순위 1, z 3.7; 지문 2 는 구분 안 됨.) 올리기: `git add FUBA/logs/equilibrium`.

### 0-b. 큰 모델 궤적 (CIFAR-10 / resnet18) 과 공격 없는 궤적 — 리뷰 보완 (규모, 오경보)
저자 코드는 `--dataset cifar10` 을 지원한다 (분류기 resnet18, 생성기 UNet(3); config.py:146). 우리가 바꾼 것은 `run_real_fuba.sh` 의 `DATASET` 환경변수와 로더의 데이터셋 인식뿐이다. 공격 없는 궤적은 저자 코드의 warm-up 을 라운드 수보다 크게 두어 백도어 분기를 한 번도 타지 않게 한 것이다 (train_main.py:188). 매니페스트 그룹 `k20_paper_cifar` (IID 3 시드), `clean_paper` (mnist/cifar × iid/owner).
```bash
ONLY=k20_paper_cifar bash make_trajectories.sh        # CIFAR-10 논문 설정 3 시드 (시간 미실측; 임시 문턱 → 첫 궤적의 검증 결과를 보고 trajectories.json verify 를 고친다)
ONLY=clean_paper bash make_trajectories.sh            # 공격 없는 궤적 4 개 (mnist iid/owner ≈ 45 분씩, cifar 는 더)
DATASET=cifar10 TAG=_r100w5_k20paper_cifar CONDS=iid bash run_ledger.sh                       # 장부 (CIFAR)
PREFIX=clean TAG=_r100_k20paper CONDS="iid owner" SEEDS=0 bash run_ledger.sh                  # 장부 (공격 없음, mnist) → 기대: 경보 없음
PREFIX=clean TAG=_r100_k20paper_cifar CONDS="iid owner" SEEDS=0 DATASET=cifar10 bash run_ledger.sh
```
결과: `logs/ledger/summary_{bd_iid_r100w5_k20paper_cifar, clean_*}.md`. 먼저 볼 것: CIFAR 첫 궤적의 `docs/EXPERIMENT_LOG.md` 행 (잠복 ACC 가 50 아래면 `--train_epoch` 를 늘린 변형이 필요하다 — 저자 기본값은 라운드당 100 배치). 올리기: `git add FUBA/logs/ledger FUBA/logs/trajectories docs/EXPERIMENT_LOG.md`.
BadFU (CIFAR-10/ResNet-18, K=6, 운반자 1) 위의 장부: `python ledger_defense.py --fmt badfu --name badfu_s42 --traj_dir ../BadFU_src/BadFU-main/logs/traj_s42 --data_root ../BadFU_src/BadFU-main/data --rounds 40 --K 6 --defender 5 --attackers 0 --target 0 --start 3 --min_present 1 --counts <trajectories.json 의 counts>`. 노트북 traj_full(15 라운드)에서는 **경보 없음** (R-017): 타깃 클래스 0 의 H 0.51, 음의 다리가 정상 — BadFU 의 위장(트리거+원 라벨 표본)은 공개 입력에 새지 않거나 15 라운드로는 안 쌓인다. 운반자가 1 명이라 |Pos| ≥ 2 규칙도 못 쓴다. 40 라운드 시드로 다시 보고, 안 보이면 BadFU 는 요청 시점 빼 보기(R-004 의 dose, 첫 단계 z 8.4)로 가는 경계 사례로 쓴다.

### A-03. 균형 장부 방어 (학습 중 온라인 판정 → 경보 라운드 · 제외 목록; 학습 없음, 역할 정보는 채점에만) — 궤적당 1–2 분
A-02 의 관찰을 방어 절차로 만든 것 (ALGORITHMIZATION.md §3). 라운드마다 참여자 갱신이 각 클래스를 미는 양을 장부에 누적하고, 클래스마다 헤지 비율 H·음의 다리 z·양의 다리 집합을 계산해 "공모(양의 다리 ≥ 2) / 정상 큰손(1) / 외톨이 음(0)" 으로 판정, 같은 클래스가 3 번 연속 공모면 경보. 공모에는 '양의 다리 한 명의 몫 ≤ 0.6'(`--max_share`) 조건이 붙는다 (소유자 + 정상 한 명이 같이 걸리는 것을 막음). 경보 시점의 양의 다리가 제외 재학습(A-01) 의 `--exclude` 목록이 된다 (학습 끝 장부가 아니라 경보 시점 것을 쓴다: owner 시드 하나는 끝에 소유자가 양의 다리에 섞인다).
```bash
bash run_ledger.sh                      # iid + owner 3 시드, logs/ledger/{name}.json + summary_{cond}.md
EXTRA="--h0 0.9 --kappa 3" bash run_ledger.sh   # 문턱 민감도 (두 번째부터는 저장된 밀기로 순전파 없이)
```
보는 것: 요약 표의 경보 라운드 / 클래스 맞음 / 위장자 맞음 / 운반자 정밀도·재현율 / 소유자 오제외 / 오경보 클래스. (DISLAB 6 궤적 R-013: 경보 11–16/99, 6/6 정확, 오경보 0. 노트북 1 시드 IID: 경보 16/99 라운드, 클래스 8, 위장자 4, 운반자 0 1 2 3 정밀도·재현율 1.0, 오경보 0; h0 0.7–0.9, tau 3–5, persist 3–5 에서 같은 답, persist 1 은 7 라운드에 정상을 음의 다리로 잘못 봄 → 3 유지.) 올리기: `git add FUBA/logs/ledger` (npz 는 작으니 같이).
대응 (R-014; 장부가 낸 되감기 라운드 t_back 과 제외 목록을 그대로 쓴다; 저자 재학습 코드에 시작 모델만 `--init_round` 로 준다):
```bash
# 각 궤적의 t_back 은 logs/ledger/{name}.json 의 rollback_round (IID 10, 12, 7 / owner 13, 12, 7). 되감기 없이 경보 라운드에서 바로 제외하는 변형은 --init_round <경보 라운드>.
for c in iid owner; do for s in 0 1 2; do tb=$(python -c "import json;print(json.load(open('logs/ledger/bd_${c}_r100w5_k20paper_s$s.json'))['rollback_round'])"); own=""; [ $c = owner ] && own="--non_iid --non_iid_type class_owner --owner_class 3 --owner_frac 0.9 --owner_client -1"; python benign_departure.py --name bd_${c}_r100w5_k20paper_s$s --K 20 --rounds 100 --seed $((522+s)) --split_seed $((522+s)) --defender 4 --requesters 4 --methods retrain_benign --exclude 0 1 2 3 --init_round $tb $own --out logs/benign_departure/bd_${c}_r100w5_k20paper_excl_from${tb}_s$s.json; done; done
```
비교 기준은 R-010(처음부터 제외 재학습, ASR 0.5/0.6). 기대: 같은 ASR, 라운드 7–13 개 절약. 올리기: `git add FUBA/logs/benign_departure`.
오경보 검사 (공격 없는 궤적; 저자 코드의 warm-up 만 쓴다 = 백도어 분기 미실행, 새 코드 없음):
```bash
NB_CLIENTS=20 ROUNDS=100 WARMUP=1000 PARTICIPANT_RATE=0.4 EXTRA_ARGS="--split_seed 522 --participant_rate 0.4" bash run_real_fuba.sh clean_iid_r100_k20_s0 522
python ledger_defense.py --fmt fuba --name clean_iid_r100_k20_s0 --K 20 --rounds 100 --defender -1 --attackers --target -1 --cond clean_k20
```
기대: 경보 없음 (공모 클래스 0). 이것이 "정상 학습은 균형식을 만들지 않는다" 의 직접 증거다.

### A-01. 논문 설정에서의 처방 — RESULTS_LOG R-010 (운반자 제외 재학습). R-011(고친 해독제)은 노트북 확인에서 폐기 (DRAFT_v2_SKELETON §6.1) — 첫째 줄은 돌리지 않아도 된다
R-007 의 해독제 실패 원인 둘(정화 예산 2 회, 짝 집합 과대)을 고쳤다 (`--max_purify 0` = 단계마다 허용, `--set_consist on` = 모든 단계에서 정렬 ≤ −0.3 인 후보만). 그리고 저자 재학습 코드에 제외 목록만 붙였다 (`benign_departure.py --exclude`).
```bash
K=20 ROUNDS=100 WARMUP=5 TAG="_k20paper_v2" FORCE=1 STAGES="cancel aggregate" CONDS="iid owner" CANCEL_EXTRA="--rounds_eff off" bash run_all.sh
```
```bash
for c in iid owner; do for s in 0 1 2; do python benign_departure.py --name bd_${c}_r100w5_k20paper_s$s --K 20 --rounds 100 --seed $((522+s)) --split_seed $((522+s)) --defender 4 --requesters 4 --methods retrain_benign --exclude 0 1 2 3 --out logs/benign_departure/bd_${c}_r100w5_k20paper_excl_s$s.json; done; done
```
둘째 줄은 TRACE 가 찾은 운반자(0–3)를 요청자 4 와 함께 제외하고 남은 정직 클라로 재학습한다 (표 3 마지막 행). owner 는 `--non_iid --non_iid_type class_owner --owner_class 3 --owner_frac 0.9 --owner_client -1` 를 덧붙인다. 비교: 같은 명령에서 `--exclude` 없이 (= 재학습이 백도어 모델을 내는 쪽, retrain_attack 은 운반자 재참여).

### A-0. 주 설정 (논문 설정 K=20 궤적) 본 결과 — RESULTS_LOG R-007, R-008
궤적 6 개는 만들어져 검증됐다 (`docs/EXPERIMENT_LOG.md`, 위장자 제거 시 ASR 46–93). 이제 그 위에서 본 결과를 낸다. 전량 빼기는 붕괴하므로 DOSE 의 단계 제거가 필수이고, 빼기 연산자가 `partial` 로 끝나면 둘째 줄(단계 + 증류 연산자)이 본 결과가 된다.
```bash
K=20 ROUNDS=100 WARMUP=5 TAG="_k20paper" FORCE=1 STAGES="benign dose cancel effects aggregate" CONDS="iid owner" BENIGN_METHODS="subtract distillation fedEraser" CANCEL_EXTRA="--rounds_eff off" bash run_all.sh
```
```bash
K=20 ROUNDS=100 WARMUP=5 TAG="_k20paper_kd" FORCE=1 STAGES="dose cancel aggregate" CONDS="iid owner" DOSE_EXTRA="--op kd" CANCEL_EXTRA="--op kd --rounds_eff off" bash run_all.sh
```
| 결과 | 파일 | 관문 1 (PLAN.md) |
|---|---|---|
| 결정·ASR·ACC·α·비용 | `logs/cancel/summary_{iid,owner}_r100w5_k20paper{,_kd}.md` | 위장자 purified 3/3, 정화 뒤 ASR ≤ 정상 제거 수준 + 5, 정상 두 명 accept 6/6, owner 소유자(19) 오탐 ≤ 1/3 |
| 운반자 찾기 | 같은 파일 셋째 표 | 정밀도 ≥ 0.9 |
| FU 연산자 (빼기·증류·FedEraser) | `logs/benign_departure/summary_*_r100w5_k20paper.md` | 빼기가 붕괴하면 그대로 적는다 (주장 1 은 EXPERIMENT_LOG 의 작동점 값으로) |
주의: `--rounds_eff off` 는 라운드별 궤적 통계(보조)만 끈다. 참여율 0.4 궤적에서 결석 라운드 처리가 아직 라운드 단위 스크립트에 없기 때문이다. 끝나면 `git add FUBA/logs docs && git commit -m "results: R-007 R-008" && git push`.


### A0. 상쇄 짝 찾기 절차 (cancel, 본 방법 v3) — 두 조건 × 3 시드, 궤적은 있다고 보고
```bash
FORCE=1 STAGES="cancel aggregate" CONDS="iid owner" bash run_all.sh
FORCE=1 STAGES="cancel aggregate" CONDS="iid owner" TAG="_polish" CANCEL_EXTRA="--polish kd_orth" bash run_all.sh
```
| 결과 | 파일 | 보는 것 |
|---|---|---|
| 결정·ASR·ACC·α·비용 | `logs/cancel/summary_{cond}.md` 첫 표 | 위장자 4 = purified(antidote...), 정상 7·5 = accept(confirm). owner 의 7 이 accept 면 소유자 오탐 해결 |
| 1 단계 짝 (저장 기여) | 둘째 표 | 위장자의 크기·몰림 z 가 3 을 넘는가, 정상은 안 넘는가. 짝 정밀도는 IID 에서 1.0 이지만 정상도 운반자를 짝으로 내는 것이 정상(METHOD 2 절) |
| 2' 단계 짝 (활성화 위) | 셋째 표 | **핵심**: 위장자의 짝이 운반자 넷(정밀도·재현율 1.0), 정렬이 운반자 음수 / 정상 양수. owner 에서 소유자 7 이 짝에 안 들어오는가 |
| _polish (소거 실험, 본 방법 아님) | `summary_{cond}_polish.md` | 직교 증류의 'ACC 회복 vs 망각 되돌림' trade-off. 기본 교사 pop (지금 저장된 결과는 agree 교사; pop 으로 다시 만들려면 `CANCEL_EXTRA="--polish kd_orth --polish_ref pop"`). 본 방법 결과는 위 줄(증류 없음) |

### A1. 이전 판 (unified) + 기능 효과 행렬 (effects) — 이미 있음 (dev). 다시 만들려면
```bash
FORCE=1 STAGES="unified effects aggregate" CONDS="iid owner" bash run_all.sh
```
| 결과 | 파일 | 보는 것 |
|---|---|---|
| 요청자별 결정·ASR·ACC·α·비용 | `logs/unified/summary_{cond}.md` | 위장자 4 = purified(...), 정상 7·5 = accept(...). owner 의 7 이 accept 면 소유자 오탐 해결 |
| 효과 행렬: p 값, 상쇄 집합, 라운드 일관성 | `logs/effects/summary_{cond}.md` | 위장자의 상쇄 집합이 운반자(carrier)로 채워지는가, 몰림 클래스가 타깃(8)인가 |

### A2. 소유자 오탐 대책만 따로 (flow 통계, 짧음)
```bash
FORCE=1 STAGES="dose aggregate" CONDS="owner" TAG="_flow" DOSE_EXTRA="--stat flow" bash run_all.sh
```
`logs/dose_fu/summary_owner_flow.md` — 정상 7 경보 0/3 이면서 4 는 잡히는가.

### A3. 핵심 표 (기본 FU 대 우리, 운반자 찾기) — 이미 있음(dev). 다시 만들려면
```bash
FORCE=1 RQ="1 3" bash run_paper.sh
```

## B. 시간이 되면
```bash
FORCE=1 RQ="2 4" bash run_paper.sh          # 비교군(MASA, SubID 식, 집계 방어 4 종 위 궤적), 적응형 5 종
RQ="5 6" bash run_paper.sh                  # 비용, 소거 (cap, tau, m, stat, probe, op)
```
집계 방어 변형은 궤적을 다시 만들어 가장 오래 걸린다. 하나만 먼저: `FORCE=1 RQ="2" TDEFS="median" bash run_paper.sh`.

## C. BadFU (로컬 GPU 노트북, 궤적이 거기 있음)
```bash
B=../BadFU_src/BadFU-main
python cancel_fu.py  --fmt badfu --name loc_badfu_can --traj_dir $B/logs/traj_full --data_root $B/data --badfu_record $B/record/badnet_dataset --rounds 15 --defender 5 --attackers 0 --target 0 --counts 10545 9647 9641 9649 9668 10545 --requesters 5 1 2 --op kd --polish kd_orth
python unified_fu.py --fmt badfu --name loc_badfu_uni --traj_dir $B/logs/traj_full --data_root $B/data --badfu_record $B/record/badnet_dataset --rounds 15 --defender 5 --attackers 0 --target 0 --counts 10545 9647 9641 9649 9668 10545 --requesters 5 1 2 --op kd
python effects.py    --fmt badfu --name loc_badfu     --traj_dir $B/logs/traj_full --data_root $B/data --badfu_record $B/record/badnet_dataset --rounds 15 --defender 5 --attackers 0 --target 0 --counts 10545 9647 9641 9649 9668 10545 --requesters 5 1 2
```
보는 것: cancel 에서 5 의 1 단계 짝이 [0](https://github.com/sijoon-sung/BadFU_defend/blob/4c4c38f19d969fa776f75064a112543d8ce2657d/운반자)·타깃 0 이고 2' 단계 짝도 [0] 인가, 정상 1·2 는 accept. (로컬 1 시드: 1 단계 짝 [0] 상쇄 0.19, 타깃 0 — 맞음.)

## 결과 올리기
```bash
cd .. && git add FUBA/logs docs/paper && git commit -m "results: ..." && git push
```

## 참고
- 일부만: `RQ="1"`, 조건 하나만: `CONDS="iid"`, 시드 하나만: `SEEDS="0"`, 명령만 보기: `DRY=1`.
- 표 만들기: `python paper_tables.py --cond iid` → `docs/paper/tables_iid.md` (T1, T1c, T1u, T1e, T2, T3, T4, T5, T6).
- 결정 뜻: accept(confirm)=빼 보고 수락 / accept(attr_suspect)=1 단계 경보지만 빼 보기 통과 / accept(partial)=연산자 한계로 일부만 뺌 / purified(antidote[+polish])=짝의 상쇄 성분 제거[+직교 증류] / purified(kd)=재억제(대체) / (sim_only)=1 단계 통과·빼 보기 경보 / reject(...)=불출.
