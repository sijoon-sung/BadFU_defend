# BadFU_defend

**연구 개요와 공통 코드의 기준 브랜치입니다.** 탐지와 정화는 아래 두 작업 브랜치에서 관리합니다.

FL 학습과 FU 삭제를 함께 이용하는 잠복 백도어를 연구합니다. **탐지와 정화를 분리해 관리합니다.** 삭제가 올바르게 수행돼도 잔류 독성 자료 때문에 모델이 위험할 수 있습니다.

**먼저 [RESULTS.md](RESULTS.md)를 읽으면 무엇을 해봤고, 무엇이 됐거나 안 됐는지 확인할 수 있습니다.**

| 작업 | 브랜치 | 현재 상태 |
| --- | --- | --- |
| 탐지 | [codex/detection](https://github.com/sijoon-sung/BadFU_defend/tree/codex/detection) | 특정 관측 신호 확인. 단순 기준선 대비 이득·독립 정상 오탐률 미입증 |
| 정화 | [codex/purification](https://github.com/sijoon-sung/BadFU_defend/tree/codex/purification) | 여섯 비교군 구현·14개 검사·축소 연결 검사 완료. 본 성능 대기 |

[문제 정의](docs/current/PROBLEM_STATEMENT.md) · [실행 안내](RUN.md) · [브랜치 안내](BRANCHES.md)

## 폴더 안내

| 폴더 | 무엇을 두는가 |
| --- | --- |
| [reproduction/](reproduction/README.md) | training·common·detection·purification·audit·tests 폴더로 분리한 코드 |
| [reproduction/validation/](reproduction/validation/README.md) | 업로드된 작은 결과, 검산 기록, 축소 검사. 원본 증거 유지 |
| [docs/current/](docs/current/README.md) | 현재 문제 정의와 두 알고리즘의 설계·한계 |
| [docs/experiments/](docs/experiments/README.md) | 해본 실험의 상세 관측과 실패/정정 기록 |
| [docs/operations/](docs/operations/README.md) | 원본 재현·궤적 목록·파일 이동 기록 |
| [docs/paper_reviews/](docs/paper_reviews/README.md) | 선행논문 정리 |
| docs/paper/ | 기존 논문 초안. 현재 결과의 확정본이 아님 |
| [docs/archive/](docs/archive/README.md) | 과거 설계와 당시 상태 문서 |
| FUBA/, BadFU_src/ | 고정한 원본 공격 코드 및 기존 자료. 추가 FUBA 분석은 FUBA/research/ |
| tools/figures/ | 결과 그림 생성 |
| [legacy/](legacy/README.md) | 초기 탐지·정화·감사 코드. 현재 실행 경로 아님 |
| artifacts/ | 로컬 모델·궤적·중간 산출물. Git에 올리지 않는 대용량 자료 |

활성 브랜치는 `main`, `codex/detection`, `codex/purification` 세 개입니다. 이전 브랜치는 보관 태그로 옮겼습니다. 다른 컴퓨터에서 갱신할 때는 [전환 안내](BRANCHES.md#다른-컴퓨터의-기존-실행)를 먼저 확인합니다.

이번 코드 정리: 65개 파일 이동, 연구 검사 124개와 FUBA 검사 105개 통과. 이는 코드 연결 검사이며 탐지·정화 성능 향상을 뜻하지 않습니다. [변경·검증 기록](docs/operations/LAYOUT_MIGRATION.md).
