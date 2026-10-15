# 현재 실행 안내

먼저 [궤적 상태](docs/TRAJECTORIES.md)를 봅니다. 직접 검사한 로컬 파일은 무결성·집계 검사를 통과했습니다. 실행 중인 작업은 그대로 완료하고, 완료 후 이 버전을 받습니다. HRAC 구현·검증은 저장 자료를 재사용하며 전체 재학습을 요구하지 않습니다.

## HRAC 실행 — 저장소 루트

수식·검사 조건·현재 실패는 [연구 문서](docs/RESEARCH_SUMMARY.md)에 있습니다. 필요한 패키지는 기존 환경의 NumPy·SciPy입니다. BadFU 관측 수집만 PyTorch·torchvision이 추가로 필요합니다. 기존 학습 코드와 FUBA 수집기는 바꾸지 않았습니다.

기존 FUBA 6개 캐시 재사용, 합성 수학 검증, 테스트:

```bash
python FUBA/experiment_hrac.py replay --out FUBA/logs/hrac/my_fuba_replay.json
python FUBA/experiment_hrac.py stress --out FUBA/logs/hrac/my_synthetic_checks.json
python -m unittest discover -s FUBA/tests -q
```

BadFU는 해당 컴퓨터의 실제 원시 파일을 가리키는 기존 manifest를 사용합니다. 아래 예시는 로컬에서 확인한 15라운드 자료 경로입니다. 다른 PC의 절대 경로를 복사해 존재하는 파일처럼 취급하지 않습니다.

```bash
python FUBA/experiment_hrac.py collect-badfu --manifest artifacts/badfu/imported_traj_full/manifest.json --out FUBA/logs/hrac/my_badfu.npz --probe 80
python FUBA/experiment_hrac.py replay --caches FUBA/logs/hrac/my_badfu.npz --out FUBA/logs/hrac/my_badfu_replay.json
python FUBA/experiment_hrac.py diagnose --caches FUBA/logs/hrac/my_badfu.npz --out FUBA/logs/hrac/my_badfu_diagnostics.json
```

수집은 저장된 양의 FL 업데이트를 깨끗한 입력에 측정합니다. 매 라운드 직전 BN 버퍼를 고정하며 BN 재보정·트리거 사용·FU·재학습을 하지 않습니다. FUBA와 BadFU의 관측 프로토콜을 같은 정상 보정 집합에 섞지 않습니다. 출력이 이미 있으면 중단하므로 다시 실행할 때는 새 `--out`을 지정합니다.

### DISLAB의 진행 중인 기존 실험 결과 재사용

기존 `experiment_tension.py`의 manifest와 완료된 캐시가 같은 컴퓨터에 있을 때 다음을 실행합니다. 없는 캐시는 목록에 남기며 새 학습을 시작하지 않습니다.

```bash
python FUBA/experiment_hrac.py import-plan --manifest FUBA/logs/tension_v3/pilot/manifest.json --out FUBA/logs/hrac/imported_pilot.json
python FUBA/experiment_hrac.py evaluate --manifest FUBA/logs/hrac/imported_pilot.json --out FUBA/logs/hrac/pilot_evaluation.json
```

클라이언트별 캐시만 있으면 대내 검사는 `unobserved_channel`입니다. 두 채널을 보정하는 기본 alpha=.05에서는 정상 보정 실행 39개/조건 미만일 때 `insufficient_clean_episodes`입니다. 현재 pilot 결과를 실제 FPR 보장으로 바꾸지 않습니다. 정상 시험·공격 시험·보정 사이에 동일 원시 실행, 동일 캐시 내용, 같은 조건/시드를 중복 배치하면 오류입니다.

### 대내 검사 입력 계약

현재 원시 클라이언트 궤적에는 다음 샘플/묶음별 정보가 없습니다. 새로운 제한 수집 실험에서 확보한 경우에만 episode manifest의 `requests`에 아래 형식의 JSON 경로를 연결합니다. 아래 숫자는 **형식 설명용 예시이며 실험 결과가 아닙니다**.

```json
{
  "schema": 1,
  "requests": [{
    "id": "request-001",
    "atoms": [[1.0, 0.0], [0.0, 1.0]],
    "observed": [0.98, 0.01],
    "constraint_matrix": [[1.0, 1.0]],
    "constraint_value": [1.0],
    "provenance": {
      "independent_observation": true,
      "history_source": "학습 당시 기록한 이력 파일 경로와 해시",
      "request_source": "별도로 관측한 요청 파일 경로와 해시",
      "coordinate_system": "같은 기준 모델·입력·출력 좌표의 식별자"
    }
  }]
}
```

`atoms`는 원자×기능좌표, `observed`는 독립 요청 관측입니다. 원자별 부분 삭제가 실제 기여와 비례한다고 가정할 수 없다면 해당 묶음을 더 세분하거나 근사 오차를 측정해야 합니다. 정확한 삭제 ID를 알면 `lower`와 `upper`를 같은 0/1 벡터로 고정합니다. 위 출처 선언은 입력 기록이며, 실제 원자/요청의 진실성을 자동 증명하지 않습니다.

직접 만드는 episode manifest는 `schema: 1`, `episodes: [...]`이며 각 항목에 `id`, `source_episode`(원 학습 실행 ID), `seed`, `stratum`, `measurement_id`, `split`, `ledger` 및/또는 `requests`를 둡니다. 경로는 manifest 기준입니다. `split`은 `calibration`, `clean_test`, `attack_test`, `attack_development` 중 하나이고 정상 두 split에는 `benign: true`가 필요합니다. 같은 `stratum`의 `measurement_id`에는 모델·프로브·라운드 수·삭제 단위·검사 요청군을 동일하게 고정합니다.

## 남긴 실행 경로

| 작업 | 경로 |
|---|---|
| FUBA 궤적 생성·검증 | `FUBA/make_trajectories.sh`, `FUBA/verify_trajectories.py` |
| BadFU 궤적 저장·재사용 | `BadFU_src/BadFU-main/request_pipeline.py` |
| 이력 점수 수집·기존 비교 | `FUBA/collect_tension.py`, `FUBA/experiment_tension.py` |
| 완료된 탐지 결과 읽기 | `FUBA/report_detection.py` |

세부 파일은 [현재 코드 목록](FUBA/README.md)에 있습니다. 기존 설정과 함수 내용은 바꾸지 않았습니다.

## 파일이 있는 컴퓨터에서 확인

FUBA 검증기는 재학습하지 않습니다. 파일 수·참여·집계를 확인하고 기존 모델을 평가합니다. 실행 PC의 `FUBA/checkpoints`와 데이터가 필요합니다. 아래 명령은 선택한 항목의 registry를 갱신하므로, DISLAB 파일을 가지고 있는 checkout에서 실행합니다.

```bash
python FUBA/verify_trajectories.py --only k20_paper k20_paper_cifar clean_paper --force --no_doc
```

검증기의 기존 `ok`에는 완화된 공격 조건이 포함될 수 있습니다. 또한 이 검증기의 ASR은 평균 생성기·eval 모드에서 측정하며, 학습 로그의 개별 생성기·train 모드 평가와 다릅니다. 위 명령을 반복하는 것만으로 평가 차이가 해소되지는 않습니다. ASR을 곧바로 궤적 생성 오류나 원본 공격 재현 여부로 해석하지 않습니다. [평가 차이의 근거](docs/EVIDENCE_AUDIT.md#궤적-판정-정정-2026-10-15)를 참고합니다. 파일이 없는 PC의 `missing`을 서버 파일 소실로 해석하지 않습니다.

BadFU 기존 궤적의 경로는 `artifacts/badfu/<run>/manifest.json`에 있습니다. `request_pipeline.load_tensor`는 SHA-256을, `load_event`는 클라이언트 수·집계 비중·저장 FedAvg를 검사합니다. 이번 직접 검사는 [JSON](FUBA/logs/trajectories/file_check_2026-10-15.json)에 기록했습니다. 오래된 임시 폴더 경로를 가진 manifest는 원본 파일과 함께 보존합니다.

## 저장된 BadFU FU 모델과 탐지 결과 연결

### 입력 조건과 정상 기준 방향 실험

저장된 FL 궤적만 읽습니다. 같은 정상 입력 80개에 대해 원본·정답 트리거(진단 전용)·밝기·좌우 반전·네 모서리의 고정 무작위 4×4 패턴을 비교합니다. 정답 트리거는 기존 공격 입력 512개와 정확히 일치하는지 확인합니다. 알려진 표적이나 트리거는 blind 패턴 생성 및 기준선 적합에 사용하지 않습니다.

```bash
python FUBA/experiment_hrac_views.py run --manifest artifacts/badfu/imported_traj_full/manifest.json --record-root /path/to/record/badnet_dataset --data-root /path/to/cifar10 --out FUBA/logs/hrac/my_badfu_views --rounds 15
```

`--data-root`는 `cifar-10-batches-py`의 상위 디렉터리입니다. `--record-root`에는 이 실행의 `pert_result.pt`와 공격 이미지가 있어야 합니다. 기존 출력 디렉터리는 덮어쓰지 않습니다. `protocol.json`은 모델 응답을 측정하기 전에 저장합니다. 다른 궤적에 적용할 때도 데이터·분할·트리거 출처가 맞아야 합니다.

실제 양의 업데이트 효과와 Jacobian-vector product를 별도로 기록합니다. 정상 기준 방향은 관측 입력과 겹치지 않는 깨끗한 이미지 80개(클래스당 8개)의 손실 그래디언트 10개입니다. 기준선 계수는 깨끗한 관측 입력 절반에서 맞추고 다른 절반의 모든 조건에서 평가한 뒤 교차합니다. 이는 정상 한 단계 학습의 부분공간 근사이며, 독립적인 정상 궤적 보정을 대체하지 않습니다. 클라이언트 역할·FU 후 ASR·oracle 반응으로 기준선을 적합하지 않습니다.

분석은 원시 효과/입력 쌍 차이 × 유한 업데이트/JVP/정상 기준 보정 × log-probability/centered logit을 모두 남깁니다. 사후에 좋은 설정만 골라 확증 성능으로 제시하지 않습니다. 로컬 정상 MNIST 기록은 CIFAR-10/ResNet-18 보정군으로 사용하지 않습니다. 독립적인 동일 조건 정상 실행이 확보되기 전에는 FPR·TPR을 보고하지 않습니다.

다른 입력 80개와 새 패턴 3개에서 후속 확인을 실행합니다. 각 패턴은 네 모서리를 모두 검사합니다. 초기 실험을 본 뒤 정한 후속 실험이며, 같은 학습 궤적을 쓰므로 독립 공격 재현으로 세지 않습니다. 초기 관측 입력·정상 기준 이미지와의 ID 중복을 검사합니다.

```bash
python FUBA/validate_hrac_views.py --manifest artifacts/badfu/imported_traj_full/manifest.json --record-root /path/to/record/badnet_dataset --data-root /path/to/cifar10 --out FUBA/logs/hrac/my_badfu_views_holdout
python FUBA/plot_hrac_views.py --initial FUBA/logs/hrac/my_badfu_views/analysis.json --heldout FUBA/logs/hrac/my_badfu_views_holdout/analysis.json --out /path/to/new_probe_comparison
python FUBA/compare_hrac_views.py --initial FUBA/logs/hrac/my_badfu_views --heldout FUBA/logs/hrac/my_badfu_views_holdout --out FUBA/logs/hrac/my_views_score_ablation.json
```

완료 자료: [초기 프로토콜](FUBA/logs/hrac/views_badfu_v1/protocol.json) · [초기 결과](FUBA/logs/hrac/views_badfu_v1/analysis.json) · [후속 프로토콜](FUBA/logs/hrac/views_holdout_v1/protocol.json) · [후속 결과](FUBA/logs/hrac/views_holdout_v1/analysis.json) · [그래프](docs/figures/hrac_probe_comparison.png). 초기 수집 118.3초, 후속 수집 32.3초(RTX 4060 Laptop GPU). 같은 centered-logit 좌표에서 깨끗한 입력과 패턴 원시 반응은 요청자 5를 6위로 뒀고, 패턴 유무 차이는 1위·표적 0으로 뒀습니다. 후속 세 패턴도 각각 1위였습니다. 한 궤적의 결과이며, 네 위치 중 실제 트리거 영역을 덮는 패턴이 포함되어 있습니다.

명령의 `manifest`는 원본 체크포인트·데이터의 로컬 경로와 해시를 참조합니다. 다른 PC에서는 기존 파일을 덮어쓰지 않고 복사본 manifest의 경로만 해당 PC에 맞춥니다. 저장소에 포함한 기능 응답 캐시와 분석 JSON은 원본 모델 없이 읽을 수 있습니다.

[같은 관측의 기준선 비교](FUBA/logs/hrac/views_score_ablation_v1.json)에서는 단순 누적 음의 기여와 마지막 한 라운드도 요청자 5를 1위로 뒀습니다. 따라서 새 관측의 개선을 HRAC 점수 자체의 우월성이나 긴 이력의 필요성으로 바꾸어 주장하지 않습니다. 비교기는 모델을 실행하지 않고 저장된 응답 캐시만 읽습니다.

### 저장 FU 결과의 사후 재평가

다음 명령은 기존 탐지 캐시와 FU 결과의 manifest를 연결하고 모델 해시 및 ASR·정확도를 다시 확인합니다. 학습과 FU를 새로 실행하지 않습니다. manifest가 참조하는 로컬 궤적, 데이터 및 FU 체크포인트가 필요합니다.

```bash
python FUBA/diagnose_hrac_badfu.py --out FUBA/logs/hrac/my_badfu_retrospective.json
```

[완료 결과](FUBA/logs/hrac/badfu_retrospective_v1.json): 요청자 5는 탐지 순위 5위였고, 기존 FU 한 단계의 ASR 21.88%→91.02%가 재현됐습니다. 정상 client 1 제거도 74.02%였습니다. 전체 FedEraser 결과나 완전 삭제 증거가 아닙니다. 최종 FL 모델의 ASR 24.02%와 한 단계의 시작 모델 21.88%를 구분합니다. 다음 전체 FU 실험은 요청자 5·정상 1/2·삭제 없는 처리를 같은 시작 모델과 설정으로 비교해야 합니다.

## 코드 정리 확인

저장소 루트에서 실행합니다.

```bash
python -m unittest discover -s FUBA/tests -q
```

기존 테스트 76개, HRAC 검증 19개, 관측·정상 기준 적합 검증 3개를 포함해 98개가 통과했습니다. 과거 알고리즘에 딸린 테스트 3개 파일은 코드와 함께 보관했습니다. 테스트 통과를 탐지 성능으로 세지 않습니다.

완료된 `manifest.json`과 `scores.json`이 있을 때만 다음 판독기를 사용합니다.

```bash
python FUBA/report_detection.py --experiment FUBA/logs/tension_v3/pilot
```

`registry.json`은 이 판독기의 입력이 아닙니다. 정상 보정 수가 부족하면 FPR·탐지율은 `null`로 남깁니다.

이전 실행 명령은 [보관 RUN](docs/archive/cleanup_2026-10-15/RUN.md)에 있습니다. 이전 실행기 경로가 필요하면 해당 문서가 가리키는 정리 전 커밋을 별도 checkout으로 사용합니다.
