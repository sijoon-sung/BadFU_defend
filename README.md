# BadFU_defend

FL 학습과 FU 삭제를 함께 이용하는 잠복 백도어를 연구합니다. **탐지와 정화를 별도 작업으로 관리합니다.**

먼저 [RESULTS.md](RESULTS.md)를 읽으면 시도한 방법, 확인한 결과, 실패·미입증·검증 대기를 한 번에 볼 수 있습니다.

| 작업 | 문제 | 작업 브랜치 |
| --- | --- | --- |
| 탐지 | 삭제 요청에 따른 잠복 백도어 활성화 위험을 정상 Non-IID 변화와 구분할 수 있는가? | [codex/detection](https://github.com/sijoon-sung/BadFU_defend/tree/codex/detection) |
| 정화 | 실제 삭제 뒤 드러난 백도어를 줄이면서 삭제 후 정상 기능과 삭제 효과를 유지할 수 있는가? | [codex/purification](https://github.com/sijoon-sung/BadFU_defend/tree/codex/purification) |

**현재 결과:** 특정 요청자 순위와 실제 BadFU FU 뒤의 변형 반응 증가는 관측했습니다. 이력/시계열의 추가 이득·독립 정상 오탐률은 미입증이며 정상 설명 집합은 현재 설정에서 실패했습니다. 정화는 여섯 비교군 구현과 축소 연결 검사를 완료했고 본 성능은 대기 중입니다.

- [정확한 문제 정의·입력·출력·성공 기준](https://github.com/sijoon-sung/BadFU_defend/blob/codex/purification/docs/current/PROBLEM_STATEMENT.md)
- [정리된 폴더와 코드](https://github.com/sijoon-sung/BadFU_defend/tree/codex/purification)
- [정화 실행 안내](https://github.com/sijoon-sung/BadFU_defend/blob/codex/purification/RUN.md)
- [탐지 실행·검산 안내](https://github.com/sijoon-sung/BadFU_defend/blob/codex/detection/RUN.md)
- [브랜치 안내](BRANCHES.md)

`main`은 공개 안내와 초기 코드 이력을 보관합니다. 최신 실행은 두 작업 브랜치에서 합니다. 기존 `codex/request-guided-probe` 실행은 종료 후 결과를 보관하고 전환합니다.

실제 post를 사용하는 관측은 사전 탐지 결과가 아닙니다. ASR 감소는 망각 보장이 아닙니다. 두 모델 비교·CVaR 등의 기존 도구 자체를 신규성으로 주장하지 않습니다.
