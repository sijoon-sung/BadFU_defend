# 공식 공격 코드 복원과 검증 범위

2026-10-15에 저자 저장소를 새로 받아 아래 커밋의 git blob으로 기존 소스 파일을 교체했다. 데이터·체크포인트·기존 실험 로그는 삭제하지 않았다. 원본의 대용량 `data/`, `saved_models/`와 Python 캐시·OS 메타데이터는 배포 대상에서 제외했다.

| 코드 | 공식 저장소 / 고정 커밋 | 파일 수 |
|---|---|---:|
| BadFU | [BingguangLu/BadFU](https://github.com/BingguangLu/BadFU/tree/0a2117feab58faf5b7e2a1e5e1759112087d6049) | 223 |
| FUBA | [stcebra/FUBA](https://github.com/stcebra/FUBA/tree/bc233610aca36e1647d0c815597cc7148292139d) | 145 |

`reproduction/upstream_lock.json`에 원본 파일별 SHA-256을 고정했다. 실행 전에 모두 검사하며 원본이 바뀌면 중단한다. Windows 줄바꿈 자동 변환도 `.gitattributes`로 막았다. 예전 분석 코드는 이 해시 목록 밖에 있으며 원본이라는 뜻이 아니다.

## 실행 시에만 적용하는 변경

BadFU의 `local_train`, `fed_avg`, `evaluate_model`, `unlearning_step_once` 함수 본문은 원본과 동일하다. AST 비교 테스트로 확인한다. 네 FL 단계와 두 FU 단계의 순서, 원본의 사전학습 초기화와 난수 소비를 유지한다. 과거 모델 목록은 디스크에서 같은 state_dict를 읽는 객체로 바꿔, 라운드 수에 비례한 GPU 메모리 축적을 피한다. 복원할 때 새로운 모델을 무작위 초기화하지 않으며 난수 상태 불변을 테스트한다.

공통 변경은 입력·출력 경로, 시드와 명시적인 실험 설정, Windows 호환 DataLoader workers=0, 원자적 저장, 입력/소스 해시, 완료 검사다. BadFU record 경로는 기존 절대 경로 대신 지정한 record 폴더의 같은 상대 파일로 연결한다. full-state trajectory에는 BN 버퍼를 포함한 전체 state_dict가 들어간다. 관측용 압축 `history.pt`와 다르다.

FUBA는 원본 MPI 스케줄러, 공격자·억제자 학습 함수, 생성기 학습, 집계 함수를 사용한다. IBA 협업은 동시에 작업자 3명을 요구해 총 5프로세스 이상으로 제한한다. MPI rank의 예외는 `python -m mpi4py`로 전체 작업에 전파한다. 실제 클라이언트 ID와 선택된 모델만 저장한다. 원본은 미참여자를 전역 모델로 채우고 정상 학습에서는 개인별 저장을 생략하는데, 새 관측용 궤적에서는 그렇게 하지 않는다. 선택적인 legacy pickle export만 원본 형식을 따른다.

FUBA의 CIFAR-10 Dirichlet 분할에서 `dataset.targets`는 리스트이므로 `y == i`가 원소별 비교가 되지 않는다. 실행 복사본에서 `np.asarray`를 적용한다. DALI import는 선택적으로 바꿔 MNIST/CIFAR-10에서 불필요한 ImageNet 의존성이 실행을 막지 않게 한다. ImageNet은 새 실행기의 지원 범위 밖이다. SQL 기록은 끄고 로컬 JSON을 저장한다. MPI 작업 배정은 실행 시점의 속도에도 의존하므로 같은 시드가 모든 환경에서 비트 단위 동일 결과를 보장하지는 않는다.

각 출력 폴더에는 실제 패치 diff와 어댑터 사본이 남는다. 실행 도중 저장소를 갱신해도 이미 시작한 실행 코드는 바뀌지 않는다. `--smoke`의 데이터 축소·라운드·에폭 변경은 protocol에 명시한다.

## 검증한 것과 아직 남은 것

- 실제 CIFAR-10/BadFU 기존 cover record로 네 FL 단계와 두 FU 단계를 축소 실행했다.
- FUBA MNIST/CNN과 CIFAR-10/ResNet-18을 원본 MPI로 축소 실행했다. 공격자, 억제자와 정상 클라이언트를 포함하는 Non-IID 검사도 수행했다.
- 최초 상태, 라운드 연속성, 참여자 ID, 유한값, 전체 모델 형상, 파일 해시와 FL 집계 재계산을 검사했다. FU는 구조·연속성·유한값 검증이며 별도 재학습과의 삭제 동등성 검증은 아니다.
- 새 BadFU 궤적을 기존 라벨 없는 관측기로 읽는 연결도 실제 모델에서 확인했다. 축소 실행의 순위/ASR로 탐지 성능을 주장하지 않는다.
- 기존 연구 테스트 105개와 새 무결성·원본 함수 보존 테스트를 별도로 실행했다. [검증 기록](../../experiments/shared/attack_reproduction/checks/summary.json).

40R BadFU 원본 전체 순서와 100R FUBA의 공격 성능은 이 새 실행기로 아직 검증하지 않았다. 기존 seed42/43/44 결과와 새 결과를 같은 실행으로 합치지 않는다. BadFU의 clean5는 공격 full6와 참여자 수·구성이 달라 일치된 정상 대조군이 아니다.

원본 FUBA `unlearn_method/utils.py`는 FedEraser의 `global_epoch=2`, `forget_client_idx=4`를 고정하고, `federaser.py`는 IID 분할을 다시 만든다. 따라서 새 FL 궤적이 완전하더라도 이 원본 FU를 곧바로 100R Non-IID 삭제 검증으로 해석하면 안 된다. 이번에는 공격 학습·기록을 복원했고, FU 일반화는 별도 변경과 실험이 필요하다.

## 제거한 이전 진입점

`fl_detect_badfu.py`, `make_badfu_trajectories.sh`, `regen_and_detect.sh`, `watch_and_run.sh`, `FUBA/make_trajectories.sh`, `FUBA/run_real_fuba.sh`를 제거했다. 예전 결과를 재검토할 때는 해당 실행의 커밋을 사용한다. 현재 학습 명령은 [RUN.md](../../RUN.md) 하나를 기준으로 한다. 이 삭제는 원시 결과 삭제가 아니다.
