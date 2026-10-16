# 요청 유도 공통 프로브: 독립 실험 브랜치

브랜치: `codex/request-guided-probe`

현재 구현은 **기존에 기록한 FUBA MNIST/CIFAR-10 전체 궤적**으로 실행한다.
기존 학습 코드를 변경하거나 학습을 재실행하지 않는다. 공격자 정답, 표적 정답,
공격 생성기, 공개 이미지의 라벨은 탐색에 쓰지 않는다.
BadFU 내부 샘플 삭제와 일반 FU 연산자는 이 실행기의 지원 범위에 포함하지 않는다.

## 검증할 가설

삭제 요청의 누적 기여로 만든 대리 모델에서, 같은 작은 입력 변형에 대한
출력 반응이 삭제 전 모델보다 증가하는 방향을 찾는다. 이 방향이 탐색하지 않은
이미지와 실제 FU 모델에서도 유지되는지 검사한다.

모든 이미지가 공유하는 하나의 섭동 `delta`를 사용한다. 클래스 `k`를 순회하되
정답 표적은 입력하지 않는다. 마진은 `logit[k] - logsumexp(other logits)`이다.

```
release(x, delta, k) =
    [margin(candidate, x+delta, k) - margin(candidate, x, k)]
  - [margin(before,    x+delta, k) - margin(before,    x, k)]
```

이를 탐색 이미지에서 최대화한다. 입력 기울기의 내적을 다시 미분하는 방식이
아니므로 입력 2차 미분이 필요하지 않다. 섭동은 픽셀 공간에서 L-infinity 제한을
받으며, 변형된 이미지는 `[0,1]`로 제한한다. 큰 패치/의미적/기하학적 트리거까지
포괄한다는 가정은 하지 않는다. 공통 섭동이므로 이미지별 생성기 공격을 놓칠 수 있다.
최적화 방법의 첫 시작점은 0이고 나머지는 무작위다. 숨은 활성화가 모든 시작점에서
평평한 구간에 있으면 1차 미분 탐색도 실패할 수 있다. 음성 결과를 백도어 부재의
증거로 해석하지 않는다. 합성 예제는 입력 기울기로 접근 가능한 경우를 검사한다.

## 비교군

| 이름 | 탐색 목적 | 역할 |
| --- | --- | --- |
| request_guided | 위 release | 요청 조건부 차이 탐색 |
| post_only | 대리 모델의 변형-원본 마진 | 삭제 전 모델과의 비교가 필요한지 검사 |
| history_free | 삭제 전 모델의 변형-원본 마진 | 요청/이력 없는 일반 적대적 탐색 |
| random | 고정 전역 무작위 섭동 | 최적화 없는 입력 대조 |
| corner | 고정 우하단 4x4 무작위 섭동 | 모서리 관측 대조 |

세 최적화 방법의 클래스 수·반복·재시작 수는 같다. random/corner는 최적화하지
않으며 같은 재시작 패턴 수와 클래스 순회를 제공한다. 이 두 방법의 후보 선택은
탐색 이미지의 release를 사용하므로 요청 정보를 사용하는 강한 고정 패턴 대조다.
corner는 동일 epsilon의 **덧셈 섭동**으로, 기존 HRAC의 픽셀 덮어쓰기 패치와는
다르다. 계산량을 동일하게 맞췄다고 주장하지 않는다.

## 즉시 실행하는 합성 예제

저장소 루트에서 기존 PyTorch 환경으로 실행한다. 새 패키지는 필요하지 않다.

```powershell
python -m unittest reproduction.test_request_probe -v
python reproduction/run_request_probe.py --demo --out artifacts/request_probe_demo --epsilon 0.12 --steps 8 --restarts 1 --search-size 16 --evaluation-size 16 --device cpu
```

이는 알려진 상쇄를 직접 넣은 작은 합성 모델의 연결 검사다. FL 학습이나 실제
FUBA 탐지 성능을 입증하는 실험이 아니다. 디렉터리가 이미 있으면 새 이름을 쓴다.

## 기록된 FUBA 실행에 적용

필요한 입력은 원래 실행기의 `protocol.json`, `fl/manifest.json`, 전체 라운드
체크포인트와 무라벨 이미지 풀이다. 업로드용 평균 관측 ZIP만으로는 실행할 수 없다.
이미지 풀은 `images`와 `ids`만 있는 NPZ이고, images는 uint8 NHWC 형식이다.
CIFAR-10은 `[N,32,32,3]`, MNIST는 `[N,28,28,1]`이다.
ids는 중복 없는 문자열 배열이다. 기존 관측기의 `probe_pool.npz`를 사용할 수 있으나,
이미 그 이미지들로 방법을 고른 경우 확증 실험에는 새 풀을 준비한다.

아래 경로는 실제 학습 결과가 있는 위치로 바꾼다. `4`는 요청 클라이언트 ID 예시이며
프로그램에 공격 역할로 하드코딩되어 있지 않다. 여러 ID를 주면 집합 삭제 요청이다.

```powershell
python reproduction/run_request_probe.py --run artifacts/fuba_cifar10_s42 --pool artifacts/fuba_cifar10_s42/hrac_validation_v1/probe_pool.npz --request-clients 4 --out artifacts/probe_s42_client4 --device cuda:0
```

풀의 위치는 기존 관측 실행에서 선택한 출력 폴더에 따라 달라질 수 있다.
첫 연결 검사에는 `--steps 2 --restarts 1 --search-size 16 --evaluation-size 16`을
추가할 수 있다. 기본 실험은 클래스마다 20회, 2개 재시작을 수행하므로 CPU에서는
오래 걸릴 수 있다. 전체 궤적의 무결성과 집계를 먼저 검사하며, 대용량 체크포인트를
읽는 비용도 든다. 원본 데이터나 체크포인트는 수정하지 않는다.

대리 모델은 다음과 같다.

```
candidate = final_FL - dose * sum_t sum_(i in request) weight[i,t] * (local[i,t] - pre[t])
```

실제 참여 라운드와 집계 가중치를 사용한다. 학습 가능한 파라미터만 차감하고
BN 등 버퍼는 최종 FL 값으로 고정한다. 이 대리 모델은 실제 FU나 재학습이 아니다.
`--dose 0.25`, `0.5`, `1.0`을 별도 출력 폴더에서 비교하면 차감 크기에 대한 민감도를
볼 수 있다. 평가 결과로 dose를 고르면 그 평가 풀은 개발용으로 간주해야 한다.

## 실제 FU로 전이되는지 검사

동일 요청을 실제 FU 알고리즘으로 처리한 뒤, 해당 FUBA 분류기의 전체
`model.state_dict()`를 저장한다. 학습 래퍼나 생성기 가중치를 전달하면 안 된다.

```powershell
python reproduction/run_request_probe.py --run artifacts/fuba_cifar10_s42 --pool path/to/public_pool.npz --request-clients 4 --fu-state path/to/actual_fu_state.pt --out artifacts/probe_s42_client4_actual_fu --device cuda:0
```

실제 FU 모델은 프로브·클래스 선택이 끝난 뒤에만 평가한다. 해당 모델의 BN 등
버퍼는 제공된 값을 그대로 사용한다. 파일 해시는 기록하지만, 실제로 동일 요청을
처리한 모델인지까지 파일 자체로 증명하지는 않는다. FU 실행 기록과 함께 보관한다.

## 결과와 판정

- `report.md`: 방법별 선택 클래스, 대리 모델과 실제 FU 모델의 평가 마진.
- `scores.json`: 탐색 선택 내역, 이미지 간 주변 엔트로피, 개별 예측 엔트로피,
  표적 확률/예측 비율, 깨끗한 입력의 예측 불일치율.
- `observations.npz`: 선택된 공통 섭동, 평가 이미지별 확률과 release, 분할 ID.
- `protocol.json`: 입력/코드/결과 해시, 요청 ID, 분할, 설정, 버퍼 정책, 완료 상태.

실제 모델 출력에 softmax를 적용한다. 업데이트 로짓 변화에 softmax를 적용해
예측 확률이라고 부르지 않는다. 단일 클래스 쏠림은
`H(mean_image p)`로, 개별 예측 불확실성은 `mean_image H(p)`로 따로 기록한다.
변형 입력의 삭제 전후 엔트로피 감소에서 원본 입력의 감소를 뺀 값도 보고한다.

한 실행의 점수가 높다고 공격으로 판정하지 않는다. 여러 정상 Non-IID 실행과
공격 실행에서 같은 프로토콜을 적용한다. 정상 요청은 참여량과 데이터 편중을
가능하면 맞춰 비교한다. 이미지 수를 독립 학습 seed 수로 세면 안 된다.

연구를 계속할 근거는 (1) 요청 유도 방식의 이득이 별도 이미지에서 유지되고,
(2) 실제 FU에서도 유지되며, (3) 정상 요청에서 같은 현상이 흔하지 않고,
(4) history_free/post_only보다 추가 이득이 관찰되는 경우다. 대리 모델에서만
강한 효과가 나면 FU 위험을 찾았다는 결론을 보류한다.

공격 ASR·정상 정확도·망각 달성은 이 무라벨 관측만으로 측정하지 않는다.
해당 값은 기존 평가기로 별도 측정하고, FL 전후 및 FU 전후 조건을 일치시킨다.
이 실험기는 보정된 임계값이나 자동 방어/삭제 결정을 제공하지 않는다.

## 이 브랜치에서 확인한 범위 (2026-10-16)

자동 검사 10개가 통과했다. 합성 예제의 CPU/GPU 실행과, 기존 2라운드 FUBA
MNIST(CPU)·CIFAR-10(GPU) 체크포인트를 사용한 다섯 비교군의 연결 검사를 완료했다.
최종 코드의 세 실행 결과와 코드/출력 해시를
[`request_probe_validation_20261016.json`](../reproduction/validation/request_probe_validation_20261016.json)에 기록했다.
실제 FU 모델은 이 실행들에 제공되지 않았다. 합성 예제에서 request_guided와
post_only의 결과가 같으므로 이 기록은 요청 유도 방식의 추가 이득을 입증하지 않는다.
