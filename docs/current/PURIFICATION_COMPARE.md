# 삭제 후 백도어 교정: 수학과 실험

2026-10-17. **현재 구현은 가설을 비교하는 실험기다. 방어 성능과 신규성은 미입증이다.**
BadFU 저장 모델 자동 연결·4×4 패치 탐색·다섯 정화 방법·실제 트리거 평가를 지원한다.
로컬 2라운드 모델의 연결 검사를 완료했으며, 본 성능은 DISLAB의 완료 모델로 확인한다.
실행은 [RUN.md](../../RUN.md), 이전 관측은 [DELETION_COMPARISON.md](../experiments/DELETION_COMPARISON.md).

## 1. Problem Statement

잔류 자료를 $D_R$, 학습을 $A$, 삭제 연산을 $U$라 하면,
$U(A(D))\approx A(D_R)$가 성립해도 $D_R$의 독성 학습 때문에 백도어가 남거나 드러날 수 있다.
목표는 **이미 삭제한 모델에서, 삭제로 강화된 취약 변형을 찾아 교정하는 것**이다.
삭제 전 모델 전체를 복원하거나 삭제한 억제 클라이언트를 다시 넣지 않는다.

검증할 질문: 같은 공개 자료와 정화 예산에서, 삭제 전 정보를 이용한 탐색이
삭제 후 모델만 쓰는 탐색보다 **실제 ASR–정상 효용 절충**을 개선하는가?
완전 탐지, 일반적인 Non-IID 제거, 정확한 망각을 전제로 하지 않는다.

필요한 정보는 같은 요청의 실제 pre/post 모델, 모델 전처리, 깨끗하고 라벨이 있는
소량의 공개 이미지다. 공격 표적·트리거·삭제 원자료는 탐색과 정화 입력이 아니다.
공개 자료는 잔류 과제의 모든 클래스를 포함해야 하며 변형이 정답 의미를 유지한다고 가정한다.
클래스별 균형은 클라이언트별 분포 대표성을 보장하지 않는다. 공개 이미지가 없는 해법은 이번 범위에 없다.

## 2. Key Idea & Formulation

### A. 이중 차분을 혼합 변화로 해석하기

표적 오답 $k\ne y$에 대한 마진을 $m_\theta(x;y,k)=f_\theta(x)_k-f_\theta(x)_y$로 정의한다.
변형 $T_\delta$에 대한 점수는 다음과 같다.

$$q_i(\delta,k)=\{m_+(T_\delta x_i)-m_+(x_i)\}
-\{m_-(T_\delta x_i)-m_-(x_i)\}.$$

이는 **삭제 변화와 입력 변형의 상호작용**이다. 만약 $m_+(x)-m_-(x)=b$가
입력에 무관한 클래스별 상수이면 $q=0$이다. 정상 Non-IID 변화도 입력 의존적이면
상쇄되지 않는다. IID에서도 $q=0$이라는 보장은 없다. 공격 의도를 판별하는 정리가 아니다.

해석을 위해 $d=\theta_+-\theta_-$, $e=T_\delta x-x$라고 하자.
마진이 해당 직사각형 경로에서 두 번 연속 미분 가능하면 정확히

$$q=\int_0^1\!\int_0^1
e^\top\nabla_x\nabla_\theta m_{\theta_-+td}(x+se)\,d\,ds\,dt.$$

즉 삭제 방향에 따른 **입력 민감도의 변화**를 유한 차분으로 측정한다.
구현은 Hessian 대신 네 출력을 계산한다. ReLU의 비매끄러운 경계에 위 고전적
미분 공식을 그대로 보장으로 적용하지 않으며, 큰 패치를 작은 섭동의 1차 근사로 정당화하지 않는다.

### B. 상위 꼬리 위험(CVaR) — 구현 완료

일부 입력의 큰 변화가 평균에 가려지는지 확인한다. $\alpha$는 **상위 꼬리 비율**로 기본값 .25다.

$$\delta_k^*=\arg\max_{\delta\in\mathcal T}
\operatorname{CVaR}^{\rm upper}_{\alpha}(q(\delta,k)),$$

$$\operatorname{CVaR}^{\rm upper}_{\alpha}(q)
=\min_\eta\left[\eta+\frac1{\alpha n}\sum_i(q_i-\eta)_+\right]
=\max_{\substack{\sum_iw_i=1\\0\le w_i\le1/(\alpha n)}}\sum_iw_iq_i.$$

전체 개발 집합에서 가중치를 구하고 그 가중치로 미니배치 gradient를 합친다.
미니배치마다 상위 25%를 뽑지 않는다. 경계 표본의 부분 가중치도 포함한다.
부호 있는 마진 차분이며 ASR 확률이나 인증된 위험 상한이 아니다. 어려운 정상 이미지도
선택될 수 있어 정상 실제-FU 대조와 held-out 검증이 필요하다.

CVaR는 [Rockafellar–Uryasev의 기존 위험 최적화 도구](https://sites.math.washington.edu/~rtr/papers.html)다.
도구 자체를 신규성으로 삼지 않고 평균 목적식과 직접 비교한다.

**평균과의 수학적 차이:** 고정 표적·입력 집합의 평균 목적식에서는 깨끗한 입력 차분이
$\delta$에 무관한 상수여서 이를 빼도 탐색 gradient는 같다. CVaR에서는 표본별 기준 차이가
상위 꼬리에 포함되는 이미지를 바꾸므로 gradient도 바뀔 수 있다. 따라서 여기서는
깨끗한 차분을 단순 보고용 보정이 아니라 탐색 가중치 결정에 실제로 사용한다.
이 차이가 백도어 정화에 유리한지는 별개의 실험 가설이다.

### C. 삭제 후 정상 동작을 기준으로 하는 제약 정화 — 구현 완료

모든 arm은 $\theta_+$에서 시작한다. 탐색한 표적별 패턴을 균등하게 표본화하여
정상 CE와 변형 이미지의 정답 CE를 함께 줄인다.

$$\min_\theta J(\theta)=\frac{L_{\rm clean}(\theta)+\lambda L_{\rm transformed}(\theta)}{1+\lambda}
\quad\text{s.t.}\quad L_c(\theta)\le L_c(\theta_+)+\epsilon_c\quad\forall c.$$

$L_c$는 공개 클래스 $c$의 평균 정답 CE다. 쌍대 변수 $\mu_c$를 두어
$J+\sum_c\mu_c[L_c-L_c(\theta_+)-\epsilon_c]$를 학습하고, 매 epoch에
$\mu_c\leftarrow[\mu_c+\gamma(L_c-L_c(\theta_+)-\epsilon_c)]_{[0,\mu_{\max}]}$로 갱신한다.
과도하게 훼손되는 클래스에 더 큰 벌점이 붙는다. 현재는 고정 패턴 정화이며
탐색과 모델 갱신을 반복하는 완전한 minimax 해법은 아니다.

개발·독립 검증 자료 **양쪽의 모든 클래스**가 허용 CE 범위에 든 마지막 epoch를 선택한다.
없으면 post로 돌아가고 `fallback_to_post=true`로 기록한다. 이것은 정화 성공이 아니다.
비제약 arm은 고정된 마지막 epoch를 쓰며 시험 ASR로 epoch를 고르지 않는다.

이는 관측한 공개 자료의 CE 조건이다. 정확도, 모든 클라이언트 효용, 모집단 안전성,
망각 보장이 아니다. [Group DRO](https://arxiv.org/abs/1911.08731)의 경고처럼
학습 집합의 그룹 손실만 낮춰서는 새 집합의 성능이 보장되지 않는다.

### D. 다른 분야의 후보: 제어이론의 안전 제약 — 설계만, 미구현

정화를 모델 상태의 제약 제어 문제로 볼 수 있다.
정상 성능의 남은 허용량을 $h_c(\theta)=\epsilon_c+L_c(\theta_+)-L_c(\theta)$로 두고,
연속 시간 갱신 $\dot\theta=v$에 다음 QP를 생각한다.

$$\min_{\substack{v\in\mathbb R^p\\s\ge0}}\frac12\|v+\nabla R(\theta)\|_G^2+\beta s^2$$
$$\text{s.t.}\quad \nabla L_c^\top v\le\kappa h_c\ \forall c,
\qquad\nabla R^\top v\le-\rho R+s.$$

여기서 $v$의 부호는 자유이고 slack $s$만 비음수다. $R\ge0$는 고정 취약 변형의
정답 CE 또는 양의 초과 마진이며 $G\succ0$는 항등행렬 또는 공개 Fisher에 ridge를 더한 행렬이다.
첫 제약은 정상 손실 경계, 두 번째는 위험 감소 목표다. slack이 커지면 요구의 충돌을 보여준다.

예를 들어 $h_c=0$에서 $\nabla R=-\nabla L_c$이면 정상 손실을 늘리지 않으면서
위험을 엄격히 줄이는 방향이 없다. 그 지점의 1차 충돌이지 전역적인 정화 불가능 증명은 아니다.
산출물은 **어느 클래스의 효용을 얼마나 허용해야 위험을 줄일 수 있는가**라는 절충 곡선이다.

[Ames 등의 CBF-QP](https://arxiv.org/abs/1609.06408)는 제어 목표와 허용 상태 제약을 함께 다룬다.
연속시간 보장을 신경망의 이산 SGD로 자동 이전할 수는 없다. 미분 조건, 작은 trust region,
실제 손실을 검사하는 line search, 유한 표본 오차 처리가 필요하다.
현재 C의 epoch 단위 primal-dual 구현을 이 QP라고 부르지 않는다.

[GEM](https://arxiv.org/abs/1706.08840)과
[UPGP](https://www.sciencedirect.com/science/article/pii/S0031320326001767)에도 gradient 제약/투영이 있다.
QP·Fisher·직교화 자체가 새롭다는 주장은 하지 않는다. 우리 후보는 삭제로 강화된 변형을
제어 목표로 삼고 **post-FU 상태에서 효용–위험 충돌을 측정하는 설계**다.

### E. 최적수송 — 보류

[Optimal Transport for Domain Adaptation](https://arxiv.org/abs/1507.00504)처럼 클래스별
pre/post 표현을 정렬한 뒤 잔차를 볼 수도 있다. 하지만 같은 이미지가 이미 짝지어져 있어
새 대응의 이점이 불명확하고 공격 신호도 흡수할 수 있다. 자동 Non-IID 보정기로 채택하지 않았다.
정상 실제-FU 쌍에서 배운 운송이 공격 쌍의 신호를 보존하는지부터 검증해야 한다.

## 3. Evaluation & Results

| Arm | 바뀌는 요소 |
| --- | --- |
| post | 실제 FU 후 상태 그대로 |
| clean_ft | 공개 정상 자료 미세조정 |
| post_only | 삭제 후 변형 마진 증가의 평균을 탐색하고 정화 |
| differential | 목적식만 평균 $q$로 교체 |
| differential_cvar | 평균 $q$를 상위 꼬리 위험으로 교체 |
| differential_cvar_constrained | 바로 위와 **동일한 패턴**으로 클래스별 제약 정화 |

pre와 bd5는 추가 참조다. 같은 모든 표적·네 모서리·시작점·스텝 수·정화 미니배치 순서와
optimizer·학습 횟수·고정 BN 정책을 쓴다. pre 역전파 때문에 계산량까지 같지는 않으며 시간을 기록한다.
SAU·Neural Cleanse·I-BAU의 공식 재현이 아닌 **요소 비교**다.
[SAU](https://proceedings.neurips.cc/paper_files/paper/2023/hash/520425a5a4c2fb7f7fc345078b188201-Abstract-Conference.html)도
두 모델의 공동 취약성을 사용하므로 두 모델 비교 자체를 신규성으로 삼지 않는다.

자동 설정: seed 20261018, 개발/검증/시험 클래스당 16/16/100장, 4×4 패치 네 모서리,
표적 10개, 20스텝×2시작점, 정화 5 epoch, SGD lr .001, $\lambda=1$,
CVaR 꼬리 .25, 클래스 CE 허용 증가 .05. 정답과 같은 표적 입력은 제외하고,
모든 표적의 최선 패턴을 사용한다. 실제 공격 표적으로 패턴을 고르지 않는다.
자동 실행과 예제는 `correct_weight=0`으로 정확한 $q$를 쓴다. 일반 실행기의 별도
soft pre-CE 벌점 기본값은 1이므로 다른 설정 파일에서는 명시적으로 확인한다.

**완료 검증:** 14개 테스트가 CVaR와 독립 LP 해 일치, 미니배치 분할 불변성, 기준 차분에 따른 꼬리 gradient 변화,
상수 이동 상쇄와 입력 의존 이동 잔존, 패치 영역, 검증 자료의 선택 거부,
평가 누출 방지, 파일 변경 거부, 중단 재개, 저자 트리거 경로 연결을 확인했다.
RTX 4060 Laptop·PyTorch 2.4.1+cu118에서 2라운드 BadFU 저장 모델로 전체 연결을 검사했다.
**학습이 충분하지 않은 연결 검사이며 방어 성능 증거가 아니다. 본 실험 수치는 아직 없다.**

실제 저자 `bd_test`의 ASR(원래 표적 클래스 제외), 전체·클래스별 정확도,
트리거를 붙인 원래 정답 복원율, 고정 탐색 패턴의 별도 반응률을 저장한다.
조건부 ASR은 분모도 보고한다. 고정 패턴 평가는 정화 후 새 공격 탐색이 아니다.

## 4. Limitations & Critical Discussion

- 전후 쌍과 기록 해시의 일치는 망각 정확성 증명이 아니다. pre가 탐색에 쓰이면 삭제 자료의 정보가
  결과에 영향을 줄 수 있으므로 기존 FU 보장의 단순 후처리 승계가 성립하지 않는다.
- bd5는 공개 코드상 full6의 잔류 0~4번과 같은 자료 구성의 참조다. 단일 출력 일치를 재학습 분포와의
  동등성으로 해석하지 않는다. 이 실행기는 전체 궤적을 다시 검산하지 않는다.
- BadFU 자동 경로는 별도 client 5 삭제이며 논문의 내부 위장 샘플 삭제가 아니다.
- FUBA는 실제 동일 요청의 전후 모델과 생성기 평가 NPZ가 있을 때 일반 실행기에 연결한다.
  자동 FU 생성은 없으며 누적 차감·다른 요청 모델로 대체하지 않는다.
- 네 모서리 패치와 작은 가산 변형은 입력 의존·의미적 트리거를 포괄하지 않는다.
- CVaR는 어려운 정상 입력도 고르고 클래스 제약은 희소 공개 표본에 과적합할 수 있다.
- 정화 손실은 선택한 변형의 전체 정답 CE다. 삭제로 늘어난 오류만 정확히 제거하는 연산은 아니다.
- 정상 실제-FU 대조, 독립 seed, 공식 방어, 정화 후 재탐색/재활성화, 별도 망각 검증은 남아 있다.
  현재 CIFAR 시험 자료를 연구 중 반복 확인했으므로 최종 논문의 독립 시험 분할도 새로 고정해야 한다.

## 5. Takeaway & Next Actions

**기여 후보:** 삭제 정확성과 잔류 모델 안전성을 구분하고, 삭제 전후 상호작용으로 찾은 취약 변형을
post-FU 효용 제약 아래 교정한다. 도구 조합을 넘으려면 삭제 전 정보의 추가 이득과
제약이 개선하는 절충을 실험으로 입증해야 한다.

1. 완료된 BadFU 본 모델로 여섯 arm을 실행한다. 실제 ASR과 클래스별 손상을 함께 본다.
2. 차분이 post-only보다 낫지 않으면 삭제 전 정보의 기여를 주장하지 않는다.
3. CVaR/제약이 유망하면 개발에서 설정을 고정하고 독립 seed·정상 실제-FU 쌍에서 확인한다.
4. 제약 충돌이 반복되면 CBF-QP를 구현하여 slack·클래스별 절충 곡선을 측정한다.
5. 공식 SAU 등과 계산 예산까지 맞추고 실제 FUBA 요청 및 재활성화·망각 평가로 확장한다.

### 일반 실행기의 입력과 재현

`python -m reproduction.purification.purification_compare --config <JSON> --out <새 폴더>`.
[예제](../../reproduction/purification/example.json)의 경로는 JSON 위치 기준이며 실제 파일로 교체한다.
모델은 `badfu_cifar10`, `fuba_cifar10`, `fuba_mnist`, `module:function`을 지원한다.
체크포인트 내부 키가 `gm_after`이면 명시한다. state_dict는 strict하게 읽는다.
공개 NPZ는 `images: uint8[N,H,W,C]`, `labels: int[N]`, `ids: unique string[N]`이다.
개발/검증/시험의 ID와 동일 픽셀 중복을 거부한다. 학습 자료와 중복 여부도 소유자가 확인한다.

공격 NPZ는 시험 자료와 같은 ID·순서의 `images`, `targets`다. uint8 또는 정규화 전 float를
받으며 float는 clamp하지 않아 원본 생성기 범위를 보존한다. 모든 정화 모델 저장 후에 읽는다.
BadFU 자동 실행은 학습 record와 이미지 해시를 검사하고 원래 저자 평가 이미지를 쓴다.
FUBA는 원본 생성기의 eval 모드·전처리까지 맞춰 export해야 한다.

`protocol.json`은 해시·설정, `results.json`은 평가, `*_search.json`은 탐색과 제약 선택 기록이다.
정화 모델은 Pixels wrapper state_dict로 mean/std를 포함한다. `--resume`은 같은 설정·소스·입력의
완료 arm만 재사용한다. 자동 ZIP에는 요약·설정·해시·그래프만 넣으며 모델과 이미지는 넣지 않는다.
따라서 ZIP만으로 추론을 독립 재계산할 수 없고 원본은 실행 컴퓨터에 보관한다.
