# 시뮬레이션 없는 언러닝 정화 알고리즘 명세

2026년 10월 8일 · 구현 전 설계 버전 0.1

구현 현황: 이 명세의 첫 실행 가능한 부분집합을 [GPU 실험 패키지](../../experiments/request_purify/README.md)에 구현했다. BadFU 분리형·고정 기저·실제 FU 내부 정화와 비교 실험을 지원한다. 아래 FUBA 모듈 경로, 정상 중요 공간 보호 및 전체 망각 검증은 여전히 후속 계획이다.

목표는 요청된 삭제를 수행하는 실제 연합 언러닝에, 업데이트 기록 기반의 선택적 정화를 통합하는 것이다. 첫 버전은 클라이언트 분리형 상쇄 공격과 FedEraser 계열 FU를 다룬다. 아래 수식은 재현 가능한 구현 후보를 정의하며, 성능·망각 보장이 입증된 알고리즘이라는 뜻은 아니다. 연구 배경과 설명용 요약은 [연구 계획](RESEARCH_PROPOSAL_2026-10-08.md)에 있다.

## 입출력과 정보 접근

| 기호 | 정의 |
|---|---|
| q | 실제 삭제 요청자. 탐지기로 다시 추정하지 않음 |
| theta_T | 학습 종료 모델 |
| H | 모델 식별자·라운드·클라이언트·집계 가중치를 포함한 업데이트 기록 |
| Delta_i,t | 라운드 t에서 클라이언트 i의 로컬 모델 − 해당 라운드의 시작 글로벌 모델 |
| p_i,t | 해당 집계에서 실제 사용한 클라이언트 가중치 |
| h_i,t,c | Delta_i,t의 클래스 c 분류 헤드 행과 해당 bias를 연결한 d차원 벡터 |
| x_i,t,c | p_i,t h_i,t,c. 집계에 실제 기여하는 헤드 행 |
| W | 한 요청에서 사용할 기록 창 길이. 공격 시작 시점 정답을 사용하지 않음 |
| B | 전체 헤드 공간 D_h=C d에 놓인 정규직교 의심 기저. 열 수 k |
| gamma_i | 잔존 클라이언트 i의 정화 가중치. 0~1의 제어값이며 악성 확률이 아님 |
| U | 요청자를 제외하는 실제 FU 백엔드 |
| D_val | 선택적으로 허용된 깨끗한 검증 데이터. 삭제 대상과 중복되지 않음 |
| theta_safe | 실제 FU와 선택적 정화를 수행한 출력 모델 |

공격자 ID·공격 타깃·원 트리거·테스트 ASR은 평가 모듈만 볼 수 있다. 탐지·기저 구성·강도 결정 함수에 전달하지 않는다. 서버는 개별 업데이트와 참여 ID를 볼 수 있다고 가정한다. secure aggregation으로 이 정보가 숨겨지는 경우에는 현재 명세를 그대로 적용할 수 없다.

첫 버전의 삭제 단위는 클라이언트 전체다. 샘플 단위 요청에 클라이언트 전체 기여 제거를 대신 적용하지 않는다. 요청자가 정상인지 악성인지와 관계없이 동일한 q를 U에 전달한다. 위장 데이터 D_c가 삭제 대상인 경우 D_c를 남기거나 다시 학습하여 억제력을 복구하지 않는다.

## 반드시 유지할 동작

1. 탐지용 가상 FU 실행 수는 0이다. 실제 삭제용 U는 한 경로로 실행하며, 그 안의 보정 라운드는 U의 정상 계산이다.
2. lambda=0 또는 의심 기저가 비어 있으면 같은 seed·분할·입력의 원 U와 동일한 파라미터 결과가 나와야 한다. 감시용 기록과 평가 비용은 별도다.
3. 탐지 실패 시 공격자 정답으로 대체하거나 최대 점수 클라이언트를 강제로 선택하지 않는다.
4. 의심이 없다는 출력은 안전 증명이 아니다. `no_alarm`, `insufficient_history`, `purified`, `utility_guard_rejected`를 구분한다.
5. 요청자의 삭제 효과를 원래 모델과의 보간으로 줄이지 않는다. 정화는 U의 잔존 업데이트 또는 U 출력에 정의된 명시적 보정만 변경한다.
6. 함수 호출의 존재는 삭제 보장이 아니다. 변형된 U의 최종 모델과 탐지 상태까지 망각 평가 대상이다.

## 요청자 중심 상쇄 점수

### 라운드별 점수

q와 i가 모두 실제 참여한 라운드만 비교한다. 누락된 업데이트를 0으로 채우지 않는다. 매우 작은 노름은 방향 정보를 주지 못하므로 해당 쌍의 점수를 0으로 둔다.

각 클래스 c에서 a=x_q,t,c, b=x_i,t,c라 두고 다음을 계산한다.

```text
A(a,b)   = max(0, -dot(a,b) / (norm(a) norm(b)))
R(a,b)   = clip(1 - norm(a+b)/(norm(a)+norm(b)), 0, 1)
scale_tc = median_{j participating at t} norm(x_j,t,c) + eps
M(a,b)   = min(m_cap, min(norm(a),norm(b)) / scale_tc)
O_qic,t  = A(a,b) R(a,b) M(a,b)
```

A는 반대 방향, R은 두 가중 업데이트의 상쇄 비율, M은 잡음에 비해 충분히 큰 움직임인지 나타낸다. a 또는 b의 노름이 eps 이하이면 O=0이다. eps는 부동소수점 안정성 상수로 기록하고 데이터 단위에 따른 최소 유효 노름을 별도로 둔다. m_cap은 큰 노름 하나가 점수를 지배하지 않도록 개발 실행에서 고정한다.

R은 정확한 언러닝 후 변화의 추정값이 아니라 같은 라운드 집계의 기하학적 상쇄량이다. FL의 비선형 학습 궤적을 역으로 재현하지 않는다. 공격자가 매우 많으면 중앙값 scale도 오염될 수 있으므로 해당 공격 비율을 별도 조건으로 평가한다.

### 시간적 반복성과 요청 단위 경보

T_qi는 기록 창에서 q와 i가 공동 참여한 라운드 집합, m_qi는 그 수다. 개발 실행으로 정한 라운드 임계값 tau_round와 최소 관측 수 m_min을 사용한다.

```text
P_qic = mean_{t in T_qi} 1[O_qic,t >= tau_round]
S_qic = mean_{t in T_qi} O_qic,t * P_qic
S_q   = max_{i != q, c} S_qic
```

m_qi<m_min인 쌍은 불충분으로 처리하여 기저 후보에서 제외한다. 모든 쌍이 불충분하면 `insufficient_history`로 FU만 수행한다. S_q<=tau_request이면 `no_alarm`으로 FU만 수행한다. tau_request는 정상 보정 실행에서 **전체 클래스·클라이언트 탐색을 마친 요청별 최대 점수**의 분포로 정한다. 개별 쌍의 FPR을 요청 전체 FPR로 보고하지 않는다.

요청 규모 n, 참여율, 기록 수, non-IID 강도가 달라지면 최대 통계의 분포도 달라진다. 보정 범위와 최종 평가 범위를 명시한다. 동일 학습 궤적의 여러 요청을 보정 집합과 평가 집합에 나누지 않고, seed·분할·공격 실행 단위로 분리한다. 모르는 분포에 대한 낮은 FPR을 자동 보장하지 않는다.

### 후보와 정화 가중치

경보가 발생한 요청에서 S_qic>=tau_pair인 후보를 점수순으로 선택한다. 최대 J_max명의 잔존 클라이언트와 C_max개의 클래스로 계산량을 제한한다. 후보가 없으면 정화하지 않는다. 한 명의 잔존 공격자도 허용하여 BadFU의 두 클라이언트 쌍을 배제하지 않는다.

```text
S_i     = max_c S_qic
gamma_i = clip((S_i - tau_pair)/(tau_high - tau_pair), 0, 1)
```

tau_high>tau_pair이며 두 값은 개발·보정 단계에서 고정한다. q 및 후보 밖 클라이언트는 gamma=0이다. 이 값은 정화 강도 제어용이다. 악성 확률이 필요하면 정상·공격이 모두 있는 별도 실행에서 보정기를 학습하고 reliability diagram·Brier score와 분포 이동 성능을 추가 평가한다. 단순한 단조 변환을 확률 보정이라고 부르지 않는다.

### 정상 상쇄와 다중 공격에 대한 한계

모든 업데이트에서 동일 평균을 무조건 빼지 않는다. 두 벡터 a,b만 있을 때 평균 제거 결과는 (a-b)/2와 (b-a)/2로 항상 반대가 되어, 전처리 자체가 상쇄 신호를 만들 수 있다. 공통 학습 성분 제거는 독립적인 정상 기준이 있거나 별도 ablation에서 이 효과를 통제할 때만 추가한다.

요청자 한 명이 다수의 작은 백도어 업데이트를 상쇄하면 각 쌍의 크기가 약해져 누락될 수 있다. 초기 버전은 이 실패를 측정한다. 이후 단계에서 요청자와 후보 그룹의 가중 합을 비교하되, 무제한 부분집합 탐색은 하지 않는다. 클라이언트 내부에서 완전히 상쇄되어 서버에 하나의 업데이트만 보이는 경우를 쌍 점수로 해결했다고 주장하지 않는다.

## 의심 부분공간 추정

후보 (i,c)와 반복 신호가 있는 라운드의 h_i,t,c를 전체 헤드 공간 D_h에 삽입한다. 해당 클래스 행 이외 좌표는 0이다. 이 벡터를 z_j라 하고, 각 표본의 가중치를 O_qic,t의 상한 처리된 값 w_j로 둔다.

```text
M[j,:] = sqrt(w_j) * z_j / max(norm(z_j), eps)
M = U_s Sigma V_transpose                 # 비중심 SVD
k = min(k_max, smallest k reaching energy fraction eta)
B = first k columns of V
```

에너지는 유효 특이값 제곱 합을 기준으로 한다. 최소 특이값 이하를 버리고 남은 랭크가 0이면 빈 기저를 반환한다. eta·k_max는 개발 단계에서 고정한다. 평균 제거를 하지 않는 이유는 여러 공격자가 공유하는 평균 방향을 보존하기 위해서다. 중심 SVD와 비중심 SVD는 효과를 분리하는 비교 실험으로 둔다.

기저는 공격에 연관된 후보 공간이지 백도어 정답 공간이 아니다. 정상 지식이 섞일 수 있다. 기존 기록의 같은 파라미터 좌표를 사용하더라도 실제 FU 중 표현이 변하면 유효성이 약해질 수 있으므로, 첫 버전은 고정 B를 기준선으로 검증한다. 추가 학습 없이 실제 FU에서 얻는 후보 업데이트로 B를 갱신하는 변형은 별도 실험이다. 새 초기화로 retrain한 모델에 과거 B를 그대로 적용하는 것은 동일한 설정으로 취급하지 않는다.

## 실제 FU에 연결하는 방법

### 백엔드 어댑터

실제 FU 보정 단계 s에서 요청자를 제외한 잔존 클라이언트 R_s의 로컬 학습을 원 백엔드가 요구하는 만큼 수행한다. 단계 시작 모델을 theta_s, 새 로컬 업데이트를 e_i,s라 한다. 어댑터는 원 FU의 파라미터 갱신을 다음과 같이 분해할 수 있어야 한다.

```text
theta_base = theta_s + sum_{i in R_s} d_i,s
```

d_i,s는 단순히 과거 원 학습 업데이트가 아니라 **이번 실제 FU가 계산한 보정된 기여**다. 이 등식은 선택한 백엔드에 대해 검증해야 한다. 분해를 제공할 수 없는 FU에 임의의 d_i,s를 만들어 범용 적용이라고 주장하지 않는다. 첫 구현은 trainable parameter만 처리하며 정수 카운터와 BatchNorm 상태 등은 백엔드의 원 규칙을 따른다.

층별 크기 보정형 FedEraser에서 원 갱신이 m_l=sum_i a_i e_i,l, u_l=ell_l m_l/norm(m_l) 형태라면, 원 백엔드의 a_i와 ell_l를 그대로 사용해 d_i,l=(ell_l/norm(m_l))a_i e_i,l로 분해할 수 있다. norm(m_l)=0의 처리는 백엔드와 동일해야 한다. 기존 코드의 자료형·라운드 색인·가중치 오류를 고친다면 정화 없는 FU 기준선에도 같은 수정을 적용한다.

### 잔존 업데이트의 선택적 감쇠

전체 파라미터 공간에 B를 헤드 좌표로 삽입한 직교투영 연산을 P_B라 한다. 매 보정 단계의 정화 후보 벡터는 다음과 같다.

```text
c_s        = sum_{i in R_s} gamma_i * P_B(d_i,s)
theta_safe = theta_base - lambda_s * c_s
```

lambda_s는 0~1이다. 요청자의 기여를 되살리는 항을 넣지 않는다. theta_T와 결과를 보간하지 않는다. 과거 기여의 단순 합을 차감해 정확한 FU라고 부르지 않는다. theta_safe를 다음 실제 FU 단계의 시작점으로 사용하므로, 전체 실행은 한 개의 수정된 실제 FU 궤적이다.

이 개입은 의심 방향으로 남는 업데이트를 줄이려는 설계다. 초기 모델에 이미 남아 있는 백도어를 항상 제거한다는 뜻은 아니다. 정답 B를 주어도 효과가 없다면 시작 체크포인트·보정 단계·적용 층을 재검토한다. 전체 FU 업데이트를 무조건 직교투영하거나 매번 같은 과거 기여를 중복 차감하지 않는다.

### 정화 강도와 정상 피해 제한

검증 실행으로 lambda_0, B_abs, B_rel을 고정한다. B_abs는 절대 제거 노름 상한이고 B_rel은 해당 FU 갱신 대비 제거 비율 상한이다.

```text
budget_s = min(B_abs, B_rel * norm(theta_base - theta_s))
lambda_s = min(lambda_0, budget_s / max(norm(c_s), eps))
```

c_s=0이면 lambda_s=0이다. 이 상한은 파라미터 이동량을 제어하며 정확도 손실 상한을 직접 보장하지 않는다. D_val이 허용된 설정에서는 고정된 검증 배치로 theta_base와 후보 theta_safe의 손실을 각각 forward 평가한다. 손실 증가가 미리 정한 delta_val을 넘으면 그 단계의 lambda_s를 0으로 되돌리고 `utility_guard_rejected`를 기록한다. 추가 로컬 학습이나 FU 재실행으로 후보를 탐색하지 않는다. 매 단계 또는 정해진 주기로 검사하는 횟수와 forward 비용은 모두 비용에 포함한다.

거절 시 보안 목표를 달성했다고 표시하지 않는다. D_val이 없는 설정은 노름 제한만 사용하고 그에 맞게 정상 피해의 불확실성을 보고한다. 테스트 ASR에 맞춘 lambda 조정, 요청마다 여러 FU 결과 중 최저 ASR 선택, 위장 샘플 재학습은 허용하지 않는다.

## 정상 지식 보존의 추가 변형

첫 버전의 효과가 확인되면 D_val에서 얻는 정상 과제의 중요 방향 Q를 이용한 변형을 추가한다. Q는 정규직교 열 기저로 만들고 삭제 데이터나 원 삭제 요청자의 로컬 데이터로 구축하지 않는다.

```text
B_raw      = (I - Q Q_transpose) B
B_protect  = orthonormal basis of nonzero columns of B_raw
```

B_protect를 기존 B 대신 사용하면 제거 벡터가 Q에 직교한다. 정상 중요 방향을 보호한다는 국소적 직관이지 깨끗한 정확도 또는 망각의 보장이 아니다. 공격과 정상 지식이 겹치면 제거 가능한 공격 성분도 작아진다. 투영 후 기저가 비면 정화 없음으로 기록한다. Q 생성의 추가 backward 연산·SVD·메모리를 비용에 포함하며, 별도의 FU 시뮬레이션과 구분한다. 이 변형은 기본 실험에 필수로 넣지 않는다.

## 의사코드

```text
INPUT theta_T, H, deletion_request q, FU backend U, frozen policy Pi

validate model hashes, exact partition IDs, participant IDs, round alignment
assert attack labels and test ASR are unavailable to detector and policy
S, coverage = score_request(H, q, Pi)

if coverage insufficient or request score below threshold:
    B = empty; gamma = zero
else:
    candidates, gamma = select_candidates(S, Pi)
    B = build_uncentered_basis(H, candidates, Pi)

state = U.initialize(theta_T, H, exclude=q)
for each actual calibration step s required by U:
    e = U.train_retained_clients(state)          # U가 원래 수행할 학습
    theta_base, d = U.calibrate_and_decompose(state, e)
    assert q not in retained participants
    if B empty:
        state.theta = theta_base
    else:
        c = sum(gamma[i] * project(B, head(d[i])) for retained i)
        lambda = frozen_strength_and_norm_budget(c, d, Pi)
        candidate = theta_base - embed_head(lambda * c)
        if clean_validation_enabled and loss_guard_rejects(candidate, theta_base):
            state.theta = theta_base
            log utility_guard_rejected
        else:
            state.theta = candidate
    U.advance(state)

finalize request data and derived-state handling under the deletion protocol
return state.theta, scores, coverage, ranks, intervention log, timing, status
```

기저가 빈 분기에서도 U의 단계 진행·체크포인트 갱신은 수행한다. 탐지와 SVD에 난수가 필요하면 FU 학습과 별도의 난수 생성기를 사용하여 lambda=0 비교의 학습 순서를 바꾸지 않는다.

## 비용 모델

n은 클라이언트 수, C는 클래스 수, d는 헤드 행 차원, W는 저장 창 길이, k는 기저 랭크, J는 정화 후보 수다.

| 항목 | 기본 상한 또는 측정 대상 |
|---|---|
| 요청자 중심 점수 | O(n W C d). 전체 쌍 기준은 O(n² W C d) |
| 헤드 기록 저장 | O(n W C d). 원 FU가 이미 저장하는 체크포인트와 중복 비용을 구분 |
| 기저 구성 | 선택 표본 수 m과 D_h=C d에 대한 SVD 비용. 저랭크 구현이면 대략 O(m D_h k+(m+D_h)k²) |
| FU 단계별 정화 | O(J D_h k), 기저 저장 O(D_h k) |
| 정상 검증 | 설정된 횟수의 모델 forward와 I/O |
| 실제 FU | 원 FU의 잔존 로컬 학습·통신·보정. 별도 측정 |
| 탐지용 가상 FU | 0회 |

이 상한은 고정된 W·C·d에서 탐지의 n 의존성이 선형이라는 뜻이다. 학습부터 최종 모델까지의 전체 알고리즘을 모든 변수에 대해 선형이라고 부르지 않는다. 요청이 없는 기간의 기록 비용, 첫 요청의 준비 비용, 여러 요청의 평균 비용을 분리한다. 이후 요청에서는 이전 삭제 상태와 모델 ID가 맞는 기록만 사용한다.

## 이론적으로 확인할 수 있는 부분과 확인할 수 없는 부분

P_B가 직교투영이고 0<=lambda gamma_i<=1이면 개별 보정 기여 d_i의 변환은 다음과 같다.

```text
d_i'              = (I - lambda gamma_i P_B) d_i
P_B d_i'          = (1 - lambda gamma_i) P_B d_i
(I-P_B) d_i'      = (I-P_B) d_i
norm(d_i-d_i')   <= lambda gamma_i norm(d_i)
```

즉 선택한 공간의 개별 기여는 수축하고 직교 성분은 유지된다. 그러나 서로 상쇄하는 벡터를 다른 비율로 줄이면 집계 전체의 노름은 커질 수 있다. 이 식만으로 최종 ASR 감소를 증명할 수 없다.

같은 d와 lambda에 대해 정답 투영 P와 추정 투영 P_hat의 제거 오차는 lambda norm(P_hat-P)_2 norm(d) 이하이다. 이는 부분공간 오차를 평가할 이유를 제공한다. B의 오차·gamma의 오차·학습 궤적 변화는 별도 항이며, 이 한 부등식으로 전체 학습 안정성을 주장하지 않는다.

정상 손실 L이 해당 구간에서 L_smooth-smooth이면, 보정 c에 대해 L(theta-lambda c)-L(theta)는 -lambda grad L(theta)^T c + (L_smooth/2)lambda² norm(c)²로 상계할 수 있다. 실제 L_smooth를 알지 못한 채 이 식을 정상 정확도 보장으로 사용하지 않는다. 깨끗한 검증 손실 검사는 경험적 보호 장치다.

망각 보장은 별도다. H와 B·gamma가 삭제 요청자의 정보에 의존하고, 잔존 클라이언트의 과거 모델에도 요청자의 영향이 전파되어 있다. 따라서 보장된 FU 뒤에 이 정화를 붙였다는 이유로 단순한 후처리 보장을 주장할 수 없다. 모델과 저장된 탐지 상태를 포함한 삭제 프로토콜의 정의·검증이 필요하다.

## 평가 설계

1. **재현성 고정.** 학습 분할을 다시 추첨하지 않고 샘플 ID로 복원한다. run ID·seed·모델 해시·트리거 해시·요청 데이터 ID·FU 설정을 산출물에 저장한다. lambda=0의 FU 동등성을 확인한다.
2. **정화 원인 분리.** 정답 후보로 만든 B, 탐지 B, 동일 헤드·랭크·적용 대상의 무작위 B, B 없음, 전체 기여 제거를 비교한다. 정답 후보 B도 완전한 백도어 정답 공간이라고 부르지 않는다.
3. **점수 구성 검증.** 기존 O, 요청자 중심 O, A+R+M, 시간 반복성, 경보 없음 처리의 영향을 순차 비교한다. 무경보에 정답 후보를 삽입하지 않는다.
4. **정화 구성 검증.** 중심/비중심 SVD, 고정/신뢰도 강도, 노름 상한, 검증 손실 guard, 정상 공간 보호를 한 번에 모두 추가하지 않고 비교한다.
5. **일반화와 회피.** IID·Dominant Class·Dirichlet, n=8/20/50/100 중 실행 가능한 범위, 참여율 1/0.5/0.2, 공격자 분산, 동일 clean 복제 유무, 점수 회피 공격을 검증한다. 개발에서 쓰지 않은 조건은 별도로 보고한다.
6. **망각과 보안.** 요청자 제외 retrain과 삭제 관련 예측·membership 위험을 비교한다. 이 기준에 남는 공격 데이터의 ASR이 높을 수 있음을 구분한다. 깨끗한 정확도만으로 망각을 판정하지 않는다.
7. **지속성.** 정화 직후와 후속 FL 10/25/50라운드의 ACC·ASR, ASR 최대값·면적·목표 구간 유지 여부를 기록한다. 이 라운드 수는 초기 계획값이며 실행 전에 고정한다.
8. **비용.** 기록 추출·탐지·기저·검증·정화·I/O·통신을 측정한다. 오프라인 개발 반복과 운영 요청 시 시뮬레이션 0회를 구분한다. GPU는 동기화 후 시간을 재고, 장치·배치·워밍업·반복 수를 기록한다.

## 구현 모듈과 현재 코드에서 바꿀 부분

다음 경로는 제안하는 새 모듈이며 아직 구현된 파일이 아니다. 기존 관측 재현 스크립트는 보존하고 별도 v2 경로로 구현한다.

| 제안 모듈 | 책임 |
|---|---|
| FUBA/defence/request_purify/records.py | 모델·분할·라운드 검증과 헤드 기록 |
| FUBA/defence/request_purify/detector.py | q를 입력받는 점수, coverage, 경보 없음 상태 |
| FUBA/defence/request_purify/subspace.py | 후보 기반 비중심 SVD와 랭크 결정 |
| FUBA/defence/request_purify/policy.py | 고정된 강도 규칙과 검증 guard |
| FUBA/defence/request_purify/fu_adapter.py | 원 FU 분해와 동일성 검사, 정화 callback |
| FUBA/eval_request_purify.py | 정답 정보를 격리한 성능·망각·전체 비용 평가 |

기존 `pipeline_detect_purify.py`의 `req_hat`에 따른 삭제 대상 변경과, 후보가 비면 `a.atk`로 대체하는 경로를 새 구현에서 제거한다. 현재의 2라운드부터 기여 합 차감은 실제 FU와 구분하여 예비 대조군으로만 유지한다. 정상 대조군에서도 최대 클래스와 군집을 강제로 뽑는 동작을 없앤다. 비용 측정에 기저 구성과 모든 클래스 검사를 포함한다.

## 근거와 연결 문서

- [연구 계획과 교수님 설명문](RESEARCH_PROPOSAL_2026-10-08.md)
- [기존 수치와 제한](RESEARCH_STATUS_2026-10-07.md)
- [BadFU와 FUBA 원본 대조](ATTACK_FIDELITY_AND_DIRECTION_2026-10-08.md)
- [MASA의 개별 언러닝 탐지](https://arxiv.org/html/2411.01040v1)
- [FoolsGold의 클라이언트 관계 탐지](https://www.usenix.org/conference/raid2020/presentation/fung)
- [FLDetector의 시간적 일관성 검사](https://arxiv.org/abs/2207.09209)
- [Gradient Projection Memory의 지식 보존](https://arxiv.org/abs/2103.09762)

상쇄 점수의 A·R·M 결합, 정화 가중치, FU 내부 투영, 노름 예산과 guard의 결합은 본 연구의 구현 후보다. 위 논문들이 이 조합의 유효성이나 삭제 보장을 입증한 것으로 인용하지 않는다.
