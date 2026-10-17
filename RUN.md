# 탐지 브랜치 실행 안내

브랜치: `detection`. 목적은 이력·삭제 요청의 위험 관측과 정상 대조다.
[문제 정의](docs/current/PROBLEM_STATEMENT.md) · [결과표](RESULTS.md) · [탐지 방법](docs/current/DETECTION.md).

## 이미 끝난 실험

이력 시계열과 이중 차분·정상 설명 집합의 업로드 검토는 완료했다. 새로운 가설이나 입력 없이 본 실험을 다시 돌리지 않는다.

- 특정 패턴 입력에서 요청자 순위를 확인했다. 단순 기준선도 같은 결과를 냈다.
- 시계열 GLR의 추가 이득은 확인하지 못했다.
- 정상 설명 집합은 비영 기여를 설명하지 못했다.
- 실제 BadFU FU 뒤 패치 반응 증가는 관측했다. 사전 탐지·정상 삭제 구분의 증거는 아니다.

## 검산을 재현할 때만

저장소 루트의 기존 Python 환경에서 아래를 실행한다. 새 학습·FU·정화는 없다.

```bash
python -m reproduction.audit.audit_deletion_upload reproduction/validation/exports/deletion_comparison/07ad0aefbb22ddc8/deletion_comparison.zip --out artifacts/deletion_upload_review_detection_v1
```

출력 폴더는 새 경로를 쓴다. 기존 11건의 점수·LP·AUC를 원본 관측으로 재계산하는 명령이며 새 탐지 성능 실험이 아니다.

저장된 전체 모델에서 이전 관측을 다시 수집해야 하는 경우에만 다음 경로를 쓴다.

```bash
python -m reproduction.training.run_deletion_comparison --dry-run
python -m reproduction.training.run_deletion_comparison --out artifacts/deletion_detection_repeat_v1
```

실제 FU와 누적 차감 대리 모델을 섞어 해석하지 않는다. 현재 기본 FUBA 입력은 실제 FU 후 모델이 아니다. 기존 결과의 실패를 이 명령을 반복하는 것으로 해결했다고 주장하지 않는다.

## 다음 탐지 실험에 필요한 것

독립 정상 실제-FU 쌍, 공격 실제-FU 쌍, 고정 공개 관측 조건 및 같은 정보의 단순 기준선이 필요하다. 실제 post를 쓰는 검사는 사후 위험 관측으로 보고한다. 현재 보정된 FPR/FNR은 없으며 새 실험 명령을 준비됐다고 표시하지 않는다.

이번 정리는 커밋 이력을 바꾼다. 기존 작업이 끝나면 [브랜치 전환 안내](BRANCHES.md#다른-컴퓨터의-기존-실행)에 따라 별도 폴더를 받는다. 모델과 결과는 기존 경로를 재사용한다.

## 정화 실험을 실행하려면

기존 실행이 끝난 뒤 [purification](https://github.com/sijoon-sung/BadFU_defend/tree/purification) 브랜치의 RUN.md를 따른다. 공유 코드는 양쪽에 있으며 실행 모듈은 목적별 폴더로 이동했다.
