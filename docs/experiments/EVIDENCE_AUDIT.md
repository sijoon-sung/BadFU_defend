# 과거 결과 정정

현재 파일 판단은 [궤적 상태](../operations/TRAJECTORIES.md)를 따릅니다. 과거 탐지 실험을 없었던 것으로 설명한 것은 잘못입니다.

## 궤적 판정 정정 — 2026-10-15

높은 ASR을 근거로 FUBA CIFAR와 BadFU 궤적 전체를 부적합하다고 설명한 것은 과도한 결론이었습니다. 직접 검사한 로컬 BadFU는 32개 파일 해시와 15라운드 집계가, 로컬 FUBA는 295개 파일 존재·기존 참조 해시 42개와 100라운드 집계가 일치했습니다. [검사 기록](../../FUBA/logs/trajectories/file_check_2026-10-15.json). 이것은 저장 무결성과 집계 일관성의 근거이며 공격 효과 재현을 보장하지는 않습니다.

- **FUBA 학습 함수:** `train_method_iba.py`의 Git blob은 가져온 시점 `a280475`와 정리 후 `301abec` 모두 `efcf0b45aaf2767cfcf798ae8188b7dafeb04827`입니다. 이 파일은 바뀌지 않았습니다. 전체 실행 조건까지 같다는 뜻은 아닙니다.
- **FUBA 평가 조건:** [학습 로그 코드](../../FUBA/train_main.py)는 개별 생성기를 차례로 불러오며 마지막 ASR도 마지막 개별 생성기로 평가합니다. 생성기는 기본 train 모드이고, [attack 함수](../../FUBA/utils/comm_utils.py)는 분류기만 eval 모드로 바꿉니다. 반면 저장되는 생성기는 평균 생성기이며, [추가 평가 로더](../../FUBA/release_map.py)는 이를 eval 모드로 불러옵니다. 두 생성기 모델 모두 BatchNorm을 포함하고 평가 표본·배치 크기도 달라, 두 경로의 ASR을 같은 조건의 수치로 취급할 수 없습니다. **이 차이가 높은 CIFAR ASR의 원인인지는 아직 검증하지 않았습니다.**
- **BadFU 40R의 출처:** 보관 결과는 원본 `badfu.py`나 15R 저장 실행기가 아니라 `experiments/request_purify`에서 생성됐습니다. [데이터 코드](https://github.com/sijoon-sung/BadFU_defend/blob/a0371dbf511a265250fd12f6accda026414ef219/experiments/request_purify/data.py#L128)는 검증 데이터를 학습 자료에서 분리하고 별도 난수 생성기로 분할합니다. [학습 코드](https://github.com/sijoon-sung/BadFU_defend/blob/a0371dbf511a265250fd12f6accda026414ef219/experiments/request_purify/engine.py#L86)는 클라이언트·라운드별로 난수를 다시 설정합니다. 해당 설정의 결과는 유효한 관측이지만, 이를 원본 실행이나 다른 저장 궤적의 오류 증거로 옮겨 해석하지 않습니다.

기존 자료를 폐기하거나 모두 재학습할 근거는 없습니다. 우선 동일 체크포인트에서 학습 로그와 추가 평가의 차이를 확인합니다. 비교에 필요한 개별 생성기가 저장돼 있지 않다면 그 비교는 미확인으로 남깁니다.

## 과거 탐지 결과

- **FUBA 장부:** 개발 궤적 6개에서 요청자·표적·운반자를 맞힌 결과가 있습니다. 독립 정상 FPR 검증과는 다릅니다. [원본 폴더](../../FUBA/logs/ledger).
- **v3 비교:** 20라운드에서 요청자 1위는 v3·선형 모두 6/6, 요청자와 표적 동시 일치는 각각 6/6·0/6입니다. 같은 개발 자료의 순위 비교입니다. [원본 JSON](../../FUBA/logs/tension_audit/historical_ablation_v3.json).
- **BadFU 40R:** 공격 3개 중 2개 경보, 정상 3개 중 2개 경보. 완료된 실험이 보관 태그에 있습니다. [원본 summary](https://github.com/sijoon-sung/BadFU_defend/blob/a0371dbf511a265250fd12f6accda026414ef219/results_observe/summary.json).
- **BadFU 15R:** 기존 입력에서의 단일 성공은 새 입력에서 유지되지 않았습니다. [후속 원본](../../FUBA/logs/response_audit/badfu.json).
- **CIFAR 장부:** 요약 MD는 `h0=.8, 무경보`, 원시 JSON은 `h0=.75, 잘못된 클래스·클라이언트 경보`로 서로 다릅니다. 하나의 동일 실험으로 합치지 않습니다. [원본 JSON](../../FUBA/logs/ledger/bd_iid_r100w5_k20paper_cifar_s0.json).

ASR 22.53→91.24%는 초기 자체 MNIST 구현의 기록이며 공식 BadFU 원본 재현의 근거가 아닙니다. [기존 출처 감사](../archive/cleanup_2026-10-15/EVIDENCE_AUDIT.md)에 세부 내역을 보관했습니다.
