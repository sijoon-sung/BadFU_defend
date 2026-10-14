# 실험 수치의 출처와 사용 범위

2026-10-15. 원본 결과를 삭제하거나 재작성하지 않고 해석을 정정한다.

| 자료 | 확인한 출처 | 사용할 수 있는 주장 | 사용하면 안 되는 주장 |
|---|---|---|---|
| ASR 22.53→91.24%, ACC 97.48→97.54% | `legacy/logs/02_fullscale_40rounds_benchmark.log`, `legacy/compare_badfu_and_defense.py` | 자체 MNIST SimpleNN 구현의 기록 | BadFU 공식 코드 그대로 재현, FedEraser 재현, 논문 수치 일치 |
| `badfu_2`의 원본 재현 작업 | 별도 저장소의 `codex/original-reproduction` README | upstream 함수 정의를 보존하는 실행 방식과 준비 예외의 문서화 | 위 MNIST 수치의 출처, 현재 완료 성능의 추정 |
| FUBA 6개 개발 궤적·장부 적중 | R-012/013/034 등 | 같은 개발 자료에서 발견한 상쇄 패턴 | 독립 정상 FPR=0%, 보편적 공모 탐지 |
| 새 CIFAR 3개 및 clean 4개 | `dev` 90d2f33, registry, 자동 EXPERIMENT_LOG | 궤적 생성·기여 빼기 진단 결과 | 새 탐지기의 검증 완료, ASR 0% 완전 방어 |
| BadFU R-049 | 수정 후 eval/BN 조건의 한 단계 비교 | 공개 코드 6클라이언트 변형의 개발 실험 | 원 논문 표본 삭제, 전체 FU, 망각 인증 |

초기 MNIST 스크립트는 `SimpleNN`을 직접 정의하고 위장 자료를 제외한 뒤 새 모델을 40라운드 학습한다. 따라서 이를 저자 코드의 FedEraser 실행으로 소개할 수 없다. 방어 부분은 테스트 자료 앞 500개를 미세조정에 쓰고 전체 테스트 자료로 평가하므로 독립 평가도 아니다. 과거 README의 ‘완벽 재현’, ‘논문 수준’, ‘FPR/FNR 0%’와 TEE 실증 표현은 현재 성능 근거에서 제외한다. 로그가 존재한다는 사실과 그 해석의 타당성은 별개다.

`badfu_2`는 이전 `experiment.py`의 언러닝 계산 변경을 자체적으로 철회했다. 이것이 BadFU_defend의 모든 로그와 같은 실행이라는 증거는 없다. 서로 다른 프로토타입을 하나의 실험처럼 합치지 않는다. [해당 저장소 README](https://github.com/sijoon-sung/badfu_2/blob/codex/original-reproduction/README.md)는 준비 단계의 경로 수정 예외도 공개하고 있다. 원본 함수 실행, 논문 설정 재현, 논문 수치 재현은 각각 확인해야 한다.

## 새 업로드의 판정 기준 변경

`0ac11c3`에서 CIFAR s0 관측 뒤 검증 기준이 완화되었다. `90d2f33`의 체크 표시는 이 변경 뒤의 결과다. 이번 정리에서는 매니페스트와 자동 장부를 다시 고쳐 과거 판정을 숨기지 않는다. 세 궤적은 개발·진단용으로 분류하고 실제 FU의 잠복 활성화 확인과 분리한다. s1/s2를 포함해 종료 ASR이 높아 잠복 종료점의 확인 표본으로 사용하지 않는다.

## 읽기 기준

현재 설명은 [연구 요약](RESEARCH_SUMMARY.md), 원시 수치는 [registry](../FUBA/logs/trajectories/registry.json), 역사적 실행 기록은 [RESULTS_LOG](RESULTS_LOG.md)를 따른다. 역사 문서의 ‘확정’은 당시 실행 완료 의미로만 읽는다. 체크포인트가 서버에만 있으면 여기서 재채점했다고 표현하지 않는다. 문턱 조정·시드 선택·제외 사유도 보고한다.
