# BadFU_defend

FL에서 숨겨진 백도어가 FU의 삭제 요청으로 활성화되는 과정을 연구합니다. 현재는 **궤적을 정리하고, 저장된 이력과 삭제 요청을 이용한 탐지를 검증하는 단계**입니다.

지금 볼 문서는 두 개입니다.

- [궤적 상태](docs/TRAJECTORIES.md): 어떤 파일을 사용할 수 있는가.
- [실행 안내](RUN.md): 현재 코드와 확인 명령.

FUBA MNIST 6개·CIFAR 공격 3개·정상 4개는 기존 이력 분석 자료로 유지합니다. FUBA의 학습 로그와 추가 검증기는 트리거 생성기·평가 모드가 달라 ASR을 같은 조건의 수치로 볼 수 없습니다. 높은 ASR만으로 궤적 생성 오류를 단정했던 설명은 정정했습니다. 로컬 BadFU 15라운드는 실제 파일 해시와 집계를 확인했고, 별도 방어 실행기의 40라운드 결과와 구분합니다. DISLAB 자료는 업로드된 검사 기록 기준입니다.

| 위치 | 역할 |
|---|---|
| `FUBA/` | 공격 학습·궤적 검증·현재 탐지 비교·기존 FU 실행 지원 |
| `BadFU_src/` | BadFU 원본과 궤적 저장 실행기 |
| `FUBA/logs/`, `artifacts/` | 원시 결과와 로컬 대용량 자료 |
| `legacy/` | 이전 알고리즘·실험 실행기 보관 |
| `docs/archive/`, `docs/paper/` | 이전 설명·계획·논문 초안 |
| `docs/paper_reviews/` | [선행논문 정리](docs/paper_reviews/README.md) |

현재 작업 브랜치는 `codex/structural-latent-defense` 하나입니다. [나머지 브랜치의 용도](BRANCHES.md)를 구분합니다. 진행 중인 실험은 종료 후 코드를 갱신합니다.

이전 결과는 [과거 탐지 결과 정정](docs/EVIDENCE_AUDIT.md)과 [보관 코드](legacy/research_2026-10-15/README.md)에서 찾을 수 있습니다. **이력–요청 기여 일관성(HRAC)**의 제약 최적화·정상 실행 보정을 구현했습니다. [수식과 검증 결과](docs/RESEARCH_SUMMARY.md)에 실제 궤적과 합성 검증을 구분했습니다. FUBA 개발 결과는 기존 점수와 같고, BadFU 15라운드의 공격자 식별은 아직 해결하지 못했습니다.
