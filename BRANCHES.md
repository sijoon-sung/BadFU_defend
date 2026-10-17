# 탐지·정화 브랜치

| 브랜치 | 역할 | 먼저 읽을 문서 |
| --- | --- | --- |
| `codex/detection` | 요청 위험의 관측·탐지, 이력·시계열·정상 대조 | [탐지 정의](docs/current/DETECTION.md) |
| `codex/purification` | 실제 FU 이후 교정, ASR–효용 절충과 망각 평가 | [정화 설계](docs/current/PURIFICATION_COMPARE.md) |
| `codex/request-guided-probe` | 기존 실험 폴더와의 호환·분기 시점 보관 | [현재 결과](RESULTS.md) |
| `main` | 공개 입구와 두 작업의 안내 | 저장소 README |
| `dev`, `codex/structural-latent-defense`, `exp/baselines`, `exp/hedge-pilot` | 이전 연구/서버 업로드 이력 보관 | 각 시점의 기록. 현재 실행 안내로 사용하지 않음 |

두 작업은 원본 공격 코드, 기록기, 검산 도구를 공유합니다. 공통 코드가 양쪽 브랜치에 있다는 것은 같은 성능 주장을 한다는 뜻이 아닙니다. 각 브랜치의 README와 RUN은 해당 작업을 기준으로 합니다.

실험 중에는 작업 폴더를 전환하지 않습니다. 종료 후 결과를 먼저 커밋하고 최신 원격 상태를 가져온 다음 전환합니다.

```bash
git fetch origin
git switch codex/purification
# 탐지 작업이면: git switch codex/detection
```

원격 브랜치를 찾지 못하는 기존 단일-브랜치 clone이라면 필요한 브랜치만 먼저 가져옵니다.

```bash
git fetch origin codex/purification:refs/remotes/origin/codex/purification
git switch --track origin/codex/purification
```

기존 결과 커밋을 다른 작업으로 가져갈 때는 필요한 결과 커밋만 명시적으로 반영합니다. 공유 이력을 강제로 다시 쓰거나 기존 브랜치를 삭제하지 않았습니다.
