# BadFU_defend

FL에서 숨겨진 백도어가 FU의 삭제 요청으로 활성화되는 과정을 연구합니다. 현재는 **궤적을 정리하고, 저장된 이력과 삭제 요청을 이용한 탐지를 검증하는 단계**입니다.

지금 볼 문서는 두 개입니다.

- [궤적 상태](docs/TRAJECTORIES.md): 어떤 파일을 사용할 수 있는가.
- [실행 안내](RUN.md): 현재 코드와 확인 명령.

FUBA MNIST 6개와 정상 4개는 이력 분석 자료로 유지합니다. CIFAR 공격 3개는 파일 기록은 갖춰져 있지만 잠복 조건을 만족하지 않아 실패 분석용으로 분리했습니다. 로컬 BadFU 15라운드는 실제 파일 해시와 집계를 확인했습니다. DISLAB 자료는 업로드된 검사 기록 기준입니다.

| 위치 | 역할 |
|---|---|
| `FUBA/` | 공격 학습·궤적 검증·현재 탐지 비교·기존 FU 실행 지원 |
| `BadFU_src/` | BadFU 원본과 궤적 저장 실행기 |
| `FUBA/logs/`, `artifacts/` | 원시 결과와 로컬 대용량 자료 |
| `legacy/` | 이전 알고리즘·실험 실행기 보관 |
| `docs/archive/`, `docs/paper/` | 이전 설명·계획·논문 초안 |
| `docs/paper_reviews/` | [선행논문 정리](docs/paper_reviews/README.md) |

현재 작업 브랜치는 `codex/structural-latent-defense` 하나입니다. [나머지 브랜치의 용도](BRANCHES.md)를 구분합니다. 진행 중인 실험은 종료 후 코드를 갱신합니다.

이전 결과는 없어진 것이 아닙니다. [과거 탐지 결과 정정](docs/EVIDENCE_AUDIT.md)과 [보관 코드](legacy/research_2026-10-15/README.md)에서 찾을 수 있습니다. 새 설계 후보는 [짧은 연구 메모](docs/RESEARCH_SUMMARY.md)에만 남겼으며 아직 구현하지 않았습니다.
