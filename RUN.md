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

## 코드 정리 확인

저장소 루트에서 실행합니다.

```bash
python -m unittest discover -s FUBA/tests -q
```

기존 테스트 76개와 HRAC 검증 19개를 포함해 95개가 통과했습니다. 과거 알고리즘에 딸린 테스트 3개 파일은 코드와 함께 보관했습니다. 테스트 통과를 탐지 성능으로 세지 않습니다.

완료된 `manifest.json`과 `scores.json`이 있을 때만 다음 판독기를 사용합니다.

```bash
python FUBA/report_detection.py --experiment FUBA/logs/tension_v3/pilot
```

`registry.json`은 이 판독기의 입력이 아닙니다. 정상 보정 수가 부족하면 FPR·탐지율은 `null`로 남깁니다.

이전 실행 명령은 [보관 RUN](docs/archive/cleanup_2026-10-15/RUN.md)에 있습니다. 이전 실행기 경로가 필요하면 해당 문서가 가리키는 정리 전 커밋을 별도 checkout으로 사용합니다.
