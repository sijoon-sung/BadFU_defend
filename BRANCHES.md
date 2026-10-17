# 브랜치 안내

활성 브랜치는 세 개다. `main`에도 정리한 최신 폴더 구조와 공통 코드를 둔다.

| 브랜치 | 목적 |
| --- | --- |
| [main](https://github.com/sijoon-sung/BadFU_defend/tree/main) | 연구 개요·문제 정의·결과표·공통 기준 버전 |
| [codex/detection](https://github.com/sijoon-sung/BadFU_defend/tree/codex/detection) | 이력·요청 위험 관측과 정상 Non-IID 대조 |
| [codex/purification](https://github.com/sijoon-sung/BadFU_defend/tree/codex/purification) | 실제 FU 후 정화와 ASR–효용–삭제 효과 평가 |

두 연구 브랜치의 공통 코드가 같아도 탐지 성능과 정화 성능은 별도로 검증한다. 각 브랜치의 `RUN.md`는 해당 작업을 안내한다.

## 이전 브랜치 보관

다음 브랜치는 브랜치 목록에서 제거하고 동일 커밋을 가리키는 태그로 보관한다. 태그에서 그 시점의 코드·결과·기록을 그대로 확인할 수 있다.

| 이전 브랜치 | 보관 태그 |
| --- | --- |
| dev | archive/2026-10-17/dev |
| exp/baselines | archive/2026-10-17/exp-baselines |
| exp/hedge-pilot | archive/2026-10-17/exp-hedge-pilot |
| codex/structural-latent-defense | archive/2026-10-17/structural-latent-defense |
| codex/request-guided-probe | archive/2026-10-17/request-guided-probe |

정리 전 `main`, 탐지, 정화도 `archive/2026-10-17/before-main`, `before-detection`, `before-purification`에 보관한다.
자동 작성자·공동 작성자 표기를 수정하면서 활성 이력의 커밋 ID가 바뀐다. 파일 내용·날짜·사람의 작성자 정보는 유지하며,
이전 커밋을 인용한 실험 증거는 원래 태그에서 확인한다. AI 표기가 든 과거 태그는 복구·출처 확인을 위한 기록이다.

## 다른 컴퓨터의 기존 실행

실행 중에는 기존 작업 폴더를 바꾸지 않는다. 이번 정리는 새 학습을 요구하지 않는다.
기존 결과는 그대로 보관하고, 갱신할 때 다음 명령으로 원격 정보만 받는다.

```bash
git fetch origin --prune --tags '+refs/heads/*:refs/remotes/origin/*'
```

이력 변경 전 폴더에서 `git pull`로 병합하지 않는다. 가장 간단한 방법은 **다른 폴더에 새로 받아서** 기존 모델 경로를 입력하는 것이다.

```bash
git clone --branch codex/purification https://github.com/sijoon-sung/BadFU_defend.git BadFU_defend-clean
cd BadFU_defend-clean
# 탐지 작업이면 clone의 branch 값을 codex/detection으로 바꾼다.
```

모델·데이터는 복사하거나 지울 필요 없이 `--run`, `--data-root`, `--record-root`로 기존 절대 경로를 지정한다.
이전 실행을 재개해야 하면 해당 보관 태그의 코드와 기존 출력을 사용한다. 새 코드에 이전 정화 `--resume` 출력을 연결하지 않는다.
