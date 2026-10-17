# BadFU_defend — 정화

FL에서 잠복하고 FU 삭제 뒤 드러나는 백도어를 연구한다.
**삭제가 올바르게 수행됐는지와, 남은 모델이 안전한지는 별개의 문제다.**

먼저 [실험 한눈에 보기](experiments/README.md)를 열면 **가설·해본 것·결과·판단·보류 이유**를 짧게 확인할 수 있다.
BadFU 단일 쌍에서 ASR 억제는 관측했다. 탐지 우월성과 정상 기능을 보존하는 정화는 아직 입증하지 못했다. [현재 결과](RESULTS.md).

| 위치 | 내용 |
| --- | --- |
| [experiments/detection](https://github.com/sijoon-sung/BadFU_defend/tree/main/experiments/detection) | 탐지 실험별 코드·검사·결과 |
| [experiments/purification](experiments/purification/) | 정화 실험별 코드·검사·결과 |
| [experiments/shared](experiments/shared/) | 공통 관측·원본 재현 검사 |
| [reproduction](reproduction/) | 공통 원본 실행·기록·입력 코드 |
| FUBA, BadFU_src | 고정한 저자 코드 |
| [docs](docs/) | 문제 정의·선행논문·상세 연구 기록 |

`main`은 전체 기록을 보유한다. `detection`은 탐지+공통, `purification`은 정화+공통 파일만 둔다.
[지금 실행](RUN.md) · [브랜치 안내](BRANCHES.md) · [문제 정의](docs/current/PROBLEM_STATEMENT.md).
