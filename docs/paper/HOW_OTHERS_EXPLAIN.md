# 다른 논문은 자기 불변량을 어떻게 설명했는가 (2026-10-13)

목적: 우리 불변량("상쇄(cancellation): 운반자(carrier)의 누적 밀기와 억제자(suppressor)의 누적 밀기가 클래스별로 합하면 0 에 가깝지만 각각은 크다")을 "긴장(tension)"보다 깊은 말로 바꾸기 위해, 다른 논문이 (1) 불변량을 한 문장으로 어떻게 쓰는지, (2) 어떤 이론 대상 위에 세우는지, (3) "정상은 만족하고 공격은 깰 수밖에 없다"는 논증을 어떻게 하는지, (4) 점수를 판정으로 어떻게 바꾸는지, (5) 증거 그림, (6) 보장, (7) 어휘를 모았다.

읽기 범위에 대한 정직한 기록: arXiv HTML/ar5iv 로 본문을 읽은 논문과 초록만 읽은 논문을 §5 에 구분했다. 본문을 못 연 논문(UBA-Inf 본문, FUBA 본문, SubID/Malicious Forgetting 본문, FLARE 본문, Lockdown 본문)은 표에 "(초록)" 표시를 했고, 그 칸의 내용은 초록·슬라이드 검색 결과·인용 논문의 설명에 의존한다.

표기: 전역 모델 w, 클라이언트 j 의 라운드 t 갱신 u_j^t, 누적 갱신 U_j = Σ_t η_t u_j^t. 우리 코드의 효과 행렬 E_j[x,c] = ⟨U_j, ∇_θ f_c(x)⟩ (MATH.md 1 절).

---

## §1. 그룹 A·B 표

### A. 연합학습 강건 집계·탐지

| 논문 | 불변량(한 문장) | 이론 대상 | 정상이 만족·공격이 깨는 이유 | 판정 방식 | 증거 그림 | 보장 | 어휘 |
|---|---|---|---|---|---|---|---|
| Krum / Multi-Krum (Blanchard 2017) | 선택된 벡터는 참 기울기 방향을 "평균적으로" 가리키고 모멘트가 유계여야 한다 ((α,f)-비잔틴 내성) | 유클리드 거리; 점수 s(i) = 가장 가까운 n−f−2 개 벡터까지의 제곱거리 합 | 정상 기울기는 참 기울기 주위에 분산 σ 로 모이므로 서로 가깝다. 비잔틴 벡터는 정상 무리에서 멀면 점수가 커서 뽑히지 않고, 뽑히려면 정상 무리 안에 들어와야 하므로 참 기울기와의 거리가 제한된다 | 점수 최소 벡터 1 개 선택(m-Krum 은 m 개 평균). 임계값 없음 | 실험 없음. 기하 그림(정상 무리와 비잔틴 벡터, 각도 α, "두 비잔틴이 공모해 하나를 뽑히게 하는" 그림 3) | Prop.1: 2f+2<n 이고 η(n,f)√d σ < ‖g‖ 이면 (α,f)-내성. Prop.2: 거의 확실히 기울기 작은 영역으로 수렴 | Byzantine, choice function, resilience, "collude", omniscient |
| FoolsGold (Fung 2018/RAID 2020) | "정직한 클라이언트는 기울기 갱신의 다양성(diversity)으로 시빌과 구분된다" | 누적(역사) 갱신 사이의 코사인 유사도; 출력층의 "indicative features" | 정직한 클라이언트는 서로 다른 데이터를 가져 갱신이 다르다. 시빌은 "공통 목표를 공유하므로 더 비슷한 갱신을 낸다" | 클라이언트별 최대 쌍 유사도 → 사면(pardoning) → 로짓 함수(κ) → 가중치 [0,1] 로 학습률 조정. 임계값 없이 연속 가중 | 유사도 히트맵(IID/non-IID), 시빌 수 대비 공격률 곡선, 제거 실험 | 부록에 O(1/T²) 수렴 스케치(시빌 가중→0, 정직 가중→1 조건부). 정직 클라이언트 1 명 이상 가정 | sybil, contribution similarity, diversity, indicative features, pardoning. **"합이 0 이 되도록 설계한 잡음(noise designed to sum to zero)"** 을 적응 공격으로 언급: 우리 현상과 같은 구조를 '잡음'으로만 보고 지표 특성 제한으로 무력화한다고 주장. "단일 공격자는 이런 유사성을 보이지 않는다"고 인정 |
| FLTrust (Cao 2021) | 서버가 뿌리 데이터로 만든 "신뢰의 뿌리(root of trust)" 갱신과 방향이 맞는 갱신만 믿는다 | ReLU-절단 코사인 유사도 TS_i = ReLU(cos(g_i, g_0)); 크기 정규화(같은 초구로 투영) | 증명 없음. 정상은 같은 분포에서 학습하므로 서버 갱신과 방향이 맞는다고 "직관"으로 둔다. 악성은 방향이 다르면 가중 0, 크기가 크면 정규화로 억제 | 신뢰 가중 평균. 임계값 없음 | 표(오류율 vs 공격), 뿌리 데이터 크기·편향 그림 | Thm.1: μ-강볼록, L-평활, 뿌리 데이터가 전체 분포와 같다는 가정 하에 ‖w^t−w*‖ ≤ (1−ρ)^t‖w^0−w*‖ + 12αΔ₁/ρ | root dataset, server model, trust score, hyper-sphere normalization |
| FLAME (Nguyen 2022) | 백도어 모델은 각도(방향) 또는 크기가 정상 밀집 무리에서 벗어난다; 벗어나지 않은 것은 절단+잡음으로 지운다 | 코사인 거리 + HDBSCAN 동적 군집; 중앙값 노름 적응 절단 S_t; 가우스 잡음 σ=λS_t | 휴리스틱. "공격자가 편차를 완전히 통제하므로 어떤 탐지기도 모든 백도어 모델을 못 잡는다"고 인정하고, 큰 각도 편차만 거른 뒤 나머지는 잡음으로 처리 | 최소 군집 크기 n/2+1 (다수 무리 수용) → 절단 → 잡음 | 군집 비교 그림(실제/k-means/HDBSCAN/FLAME), 라운드별 정상 노름 감소, 잡음 수준 대 BA/MA 표 | Thm.1: (ε,δ)-DP 잡음 σ_G=λS_t 로 백도어 제거(선형 모델 증명, DNN 은 실험). 정상 다수 가정 | angular deviation, clipping bound, noise level, PMR/PDR, model alignment/obfuscation(적응 공격) |
| BaFFLe (Andreina 2021) | 갱신이 아니라 전역 모델을 클라이언트들이 자기 데이터로 검증하면, 새로 심은 백도어는 "한두 클래스의 오류율을 튀게 한다" | 클래스별 오류 변화 벡터 v^s, v^t ∈ ℤ^{2|Y|} (출처 중심·목표 중심 오분류) | 정상 라운드는 클래스별 오류가 점진적으로 변해 과거 수락 모델들과 가깝다. 백도어 주입은 특정 클래스 오류를 튀게 해 과거 분포에서 벗어난다 | 각 검증 클라이언트가 LOF(Local Outlier Factor) 로 과거 ℓ=20 개 수락 모델 대비 이상치 판정, τ=최근 ⌊ℓ/4⌋ 개의 평균 LOF; 서버는 정족수 q 표로 거부 | 깨끗한/오염 모델의 클래스별 오류율 막대, 정족수 대 FP/FN | 없음. 정직 검증자 비율 ρ 는 실험적 추정 | feedback loop, validating clients, quorum, error variation, look-back window. 한계: "느린 백도어"는 다루지 않음 |
| FLDetector (Zhang 2022) | "정상 클라이언트의 갱신은 여러 라운드에 걸쳐 일관(consistent)되고 악성은 그렇지 않다" | 코시 평균값 정리로 g_i^t = g_i^{t−1} + H_i^t(w_t−w_{t−1}); 적분 헤세 행렬을 L-BFGS 로 근사한 예측 갱신 ĝ_i^t | 정상은 "FL 알고리즘대로 자기 데이터에서 계산"하므로 예측과 맞는다. 악성은 "알고리즘을 따르지 않고 갱신을 조작"하므로 예측과 어긋난다 | 거리 d_i^t=‖ĝ−g‖, L1 정규화, N=10 라운드 평균 → 의심 점수; Gap statistics 로 군집 수 결정, k>1 이면 k-means(k=2), 점수 큰 군집을 악성 | 의심 점수 시계열(주기 공격), 악성 수·non-IID 정도 민감도 곡선, 적응 공격 λ 곡선 | Thm.1: L-평활, FedAvg, IID, η<1/((N+2)L) 일 때 정상 기대 점수 < 악성(갱신을 뒤집는 비표적 공격에 한정) | model-update consistency, integrated Hessian, suspicious score, Gap statistics. 적응 공격: 예측에 가깝게 조작하면 "탐지는 줄지만 공격도 약해진다" |
| DeepSight (Rieger 2022) | "오염 모델의 학습 데이터는 정상보다 훨씬 덜 이질적(homogeneous)이다"가 출력층에 흔적을 남긴다 | NEUP(출력 뉴런별 정규화 갱신 에너지), TE(임계 초과 개수), DDif(예측 비율 차), 출력층 바이어스 코사인 | 정상은 여러 라벨로 학습해 에너지가 여러 출력 뉴런에 퍼진다(TE 큼). 백도어는 목표 라벨에 집중해야 하므로 TE 작다 | TE < median/2 이면 의심; HDBSCAN 군집 앙상블; 군집 내 의심 비율 > 1/3 이면 제거(오염을 정상으로 볼 확률이 더 크다고 보고 τ 를 낮게) | 정상 70·오염 30 갱신의 TE 분포 그림 | 전체 보장 없음. NEUP/TE/코사인이 스케일링에 불변이라는 증명만 | homogeneity, Threshold Exceedings, Division Differences, PMR/PDR |
| FLARE (Wang 2022) (초록) | 마지막 전 층 표현(PLR)이 매개변수보다 악성 모델을 더 잘 가른다 | 서버 보조 입력에서의 PLR 쌍별 불일치(원문은 MMD 기반으로 알려짐; 본문 미확인) | 정상 다수 가정. 악성 PLR 은 정상 무리에서 멀다 | 무리에서 먼 갱신에 낮은 신뢰 점수 → 가중 집계 | (미확인) | 없음(초록 기준) | penultimate layer representation, trust score, honest majority |
| CRFL (Xie 2021) | 매개변수 절단·평활로 전역 모델의 "평활성"을 통제하면 크기 제한 백도어에 대해 표본별 인증이 가능하다 | 매개변수 랜덤 평활(가우스), 라운드별 마르코프 커널과 f-발산 축약계수, 인증 반경 RAD | 탐지가 아니라 인증. 공격 라운드의 편차는 절단·잡음으로 유계이고, 이후 정상 라운드가 두 모델(깨끗/오염)을 다시 가깝게 끌어온다 | 절단 ρ_t + 잡음 σ_t 로 학습, 시험 때 M=1000 개 잡음 모델 다수결, 상위 두 클래스 신뢰구간이 겹치면 ABSTAIN | 인증 반경 vs 잡음/공격자 수/라운드, 정상·오염 전역 모델 간 ℓ2 거리 감소 곡선 | Cor.1: β-평활 볼록 손실, η≤1/β, 데이터 Lipschitz 기울기 하에서 ‖백도어‖<RAD 면 예측 불변 | certified radius, parameter smoothing, model replacement, contraction coefficient |
| Spectral Signatures (Tran 2018) | "공격은 학습된 표현의 공분산 스펙트럼에 탐지 가능한 흔적을 남긴다" | 라벨별 중심화 표현 공분산의 최상위 특이벡터; ε-스펙트럼 분리성; 강건 평균 추정 | 트리거는 "그 자체로 강한 분류 신호"라 표현이 증폭되고, 오염 부분모집단의 평균이 이동해 최상위 방향을 지배한다 | 제곱 투영을 이상치 점수로, 라벨별 상위 1.5ε 제거 후 재학습 | 깨끗 5000/오염 500 의 ℓ2·무작위 투영·최상위 특이벡터 상관 히스토그램(표현에서만 분리됨) | Lemma 3.1: ε<1/2, Σ⪯σ²I, ‖μ_D−μ_W‖² ≥ 6σ²/ε 이면 ε-스펙트럼 분리. 유한표본 Cor.A.2 | spectral signature, outlier score, spectrally separable, robust statistics |
| Activation Clustering (Chen 2018) | 오염 표본과 정상 표본이 같은 라벨을 받는 "이유가 다르고", 그 차이가 활성화에 나타난다 | 마지막 은닉층 활성화 → ICA → 라벨별 k-means(k=2) | 오염 표본은 내부적으로 출처 클래스를 닮는다("poisonous data largely resembles its source class"); 정상 클래스 안의 자연 변이는 두 무리로 갈라지지 않는다 | 배타적 재분류(ExRe, 군집을 빼고 재학습해 다시 분류), 상대 크기(≤p%), 실루엣(≈0.10–0.15) | 활성화 PCA 산점도(오염 클래스만 두 무리), 군집 이미지 모자이크 | 없음 | activation, source/target class, exclusionary reclassification, silhouette |
| (추가) Invariant Aggregator (Wang, AISTATS 2024) | "대부분의 클라이언트에게 유익한 불변 방향"만 남긴다: 좌표별 부호 일치가 낮으면 가린다 | 좌표별 부호 일치 Σ sign(g_{i,k}); AND-mask; α-절사 평균; 평평한 손실 지형 | 정상은 유용한 좌표에서 부호를 공유한다. 악성이 자기 최소점으로 끌려면 일부 좌표를 정상과 다른 부호로 밀어야 한다. 평평한 지형에서는 크기 차이가 작아 크기 기반 방어가 실패한다는 점을 동기로 삼음 | m_k = 1[|Σ sign| > τ] 마스크 × 절사 평균 | 중앙값 실패 예시(부호 몇 개 뒤집힘), ASR 표 | Thm.10/11: N′/N ≤ α < 1/2, 유계 이질성, 분리 가능한 최소점, 가우스 잡음 하에 정상 최소점 근방 수렴 | sign consistency, invariant direction, flat region, AND-mask |
| (추가) Hammer and Anvil (2025) | 백도어 갱신과 정상 갱신의 간격 δ 가 크면 강건 집계가, 작으면 절단 미세조정이 지운다; 공격은 창 [δ₂, δ₁] 에서만 가능 | δ = ‖E[정상 집계] − E[악성 집계]‖₂ | 큰 δ 는 이상치로 걸리고, 작은 δ 는 소량 깨끗한 데이터 미세조정으로 덮인다 | Krum → 절단 초미세조정(CSFT) 순차 적용 | 절단 임계 vs ASR/ACC 곡선 | Prop.1–2: m<n/2, 깨끗한 미세조정 집합 가정. 창 폭 δ₁−δ₂ 는 미정량 | Anvil, Hammer, operating window, weakly inserted backdoor |

### B. 연합 언러닝 보안

| 논문 | 불변량(한 문장) | 이론 대상 | 정상이 만족·공격이 깨는 이유 | 판정 방식 | 증거 그림 | 보장 | 어휘 |
|---|---|---|---|---|---|---|---|
| Zhang et al. 2023 "Exploiting MU for Backdoor Attacks" (arXiv 2310.10659; 공격) | 공격: "완화(mitigation) 표본이 백도어를 숨긴다"; 언러닝으로 완화 표본을 빼면 "숨겨진 백도어가 점진적으로 깨어난다" | 오염 표본(트리거+목표 라벨)과 완화 표본(같은 트리거 구성, 원 라벨). 손실항 명시 없음 | (방어 제안) 완화 표본은 모델 출력 불확실성이 크다: 지니 불순도 1−ΣP_c² 또는 SISA 하위모델 간 정답확률 표준편차가 크다 | 불확실성 값이 임계 이하이면 요청 수락, 아니면 거부(임계값 미보고) | AUROC 표, 언러닝 단계별 ASR 상승 곡선 | 없음 | poison data, mitigation data, unlearning request, Gini impurity, sub-model similarity |
| Liu et al. AAAI 2024 "Backdoor Attacks via MU" (공격) | 공격: (i) 오염 없이 자기 데이터 일부 언러닝만으로, (ii) 주파수 영역의 "잠복(dormant) 깨끗한 라벨 트리거"를 심고 언러닝으로 "증폭" | 목적식: α·깨끗 손실 + β·트리거 손실 + γ·언러닝 개수, 제약 θ_u = U(θ, D_u); ω_j 를 시그모이드로 완화, 좌표 하강 | 잠복 이유를 기계적으로 설명하지 않음. 오염 단계는 "트리거와 모델을 연결만", 언러닝 단계는 "효과를 증폭". 사전 ASR 23–37 % 를 실험으로만 보고 | 방어 미제안. ABL·BDU 로 오염 표본 탐지 실패를 보임 | 언러닝 알고리즘 간 전이 그림, 정상 요청 희석(60 %)에도 ASR 75 % | 없음 | dormant trigger, amplify, disguise as ordinary unlearning requester |
| UBA-Inf (Huang 2024, USENIX Sec) (초록+슬라이드 검색) | 공격: "위장(camouflage)된 백도어를 언러닝으로 활성화"; 요청 시점을 고르는 "주문형(on-demand)" 공격으로 "백도어 소멸(vanishing)·노출(exposure)" 문제를 넘는다 | 영향 함수(influence function): 위장 표본을 오염 표본에 대한 영향이 반대가 되도록 LiSSA 헤세 근사·PGD 로 최적화 | 위장 표본(트리거+원 라벨)이 오염 표본의 영향을 영향 함수 척도에서 상쇄. 라벨만 고치면 불충분해서 영향 기반으로 강화 | 방어 평가 내용은 본문 미확인 | (미확인) | 없음(초록 기준) | camouflage, influence-driven, on-demand, backdoor vanishing, backdoor exposure, persistence |
| BadFU (Lu 2025; 공격) | 공격: "위장 표본이 백도어 표본의 적대 효과를 상쇄(counterbalance)"하고 언러닝이 위장 손실 ∇L_c 를 빼면 "백도어 기울기가 온전히 작용" | 악성 1 명의 로컬 손실 = 깨끗 + 백도어(D_bd, 트리거+y_t) + 위장(D_c, 트리거+원 라벨); FedSGD 형태 Eq.8 | 중앙집중에서는 위장만으로 "완전히 중화 못 함"(트리거 두드러짐, 라벨 불균형). FL 에서는 평균화가 백도어 기울기를 희석하고 non-IID 가 트리거 표현을 클라이언트별로 만들어 숨기기 쉬움 | 방어 미제안. Median/Trimean 은 사전 ASR<10 %·사후>90 %, Neural Cleanse 불안정. 논의에서 요청 심사·사후 감시·DP 를 제안 | ASR 사전/사후 표(재학습·FedEraser·FedU·SIFU), 정상 언러닝 대조 | 없음(기울기 수준 휴리스틱) | camouflage, counterbalance, counteract, masking, suppressive effect, cancel out, dormant |
| FUBA (Sheng et al., IEEE TAI; 공격) (초록) | 공격: 두 적대 클라이언트의 "게임": Adv-attacker 가 FL 중 계속 심고 Adv-defender 가 "모델 일관성 손실"로 눌러 두었다가 삭제 요청으로 깨운다 | 기울기 재가중, 적응형 갱신 스케일링(공격자 쪽), 2 단계 방어 전략·model consistency loss(방어자 쪽) | 억제자 역할을 명시적으로 "defender" 라 부름. FU 방법과 무관하게 작동, 기존 백도어 방어에 "높은 은닉성" | 방어 미제안(초록 기준) | (미확인) | 없음 | Adv-attacker, Adv-defender, game, model consistency loss, durable trigger embedding |
| ReVeil (Alam 2025, DAC; 공격) | 공격: 트리거 표본 일부에 가우스 잡음을 더하고 원 라벨로 학습하면 "충돌하는 정보가 백도어를 억제"하고, SISA 로 그 표본을 지우면 복원 | 잡음 오염 표본(위장) vs 오염 표본 | "트리거-목표 라벨 연관은 남아 있지만 잡음 위장 표본의 충돌 정보가 그것을 억제" | — | — | 없음 | concealed backdoor, conflicting information, suppress |
| Clean Unlearning Attack (2506.12522; 공격) | 공격: 트리거를 여러 클래스에 흩뿌려 "충돌하는 학습 신호"로 영향을 억제하고, 깨끗한 forget set 언러닝이 "기울기 재정렬"로 백도어를 드러낸다 | DCT 중주파 트리거; 식 10: 깨끗 + 오염 forget-set(원 라벨) + 오염 목표 클래스(목표 라벨) | "흩뿌린 트리거가 영향을 효과적으로 억제"(가설). 이론 없음 | — | ASR 사전/사후 | 없음 | conflicting learning signal, suppression, gradient realignment, selective forgetting |
| FedMUA (Chen 2025; 공격+방어) | 방어: "악성 언러닝 요청은 초기 라운드에서 정상보다 훨씬 큰 기울기 갱신을 낸다" | 라운드별 클라이언트 기울기 ℓ2 노름 합; 공격 쪽은 영향 함수로 영향력 표본 선택 후 특징을 목표로 이동 | 실험 관찰만. "초기 라운드에 한정"이라고 명시 | IQR 상위 사분위 초과 → λ(0.1/0.5) 로 축소 | 라운드별 기울기 노름 곡선(악성 2 명) | 없음 | malicious unlearning request, ISI/MUG, IQR, gradient scaling |
| BadUnlearn / UnlearnGuard (2501.17396; 공격+방어) | 방어: 서버가 언러닝 중 클라이언트 갱신을 (저장된 이력으로) 추정하고 추정과 어긋나는 갱신을 거른다 | FedRecover 식 갱신 추정 | 정상 갱신은 추정 가능, 오염 갱신은 추정과 다름 | 추정 정확도 기반 필터(세부 미확인) | — | "UnlearnGuard 로 언러닝한 모델은 처음부터 재학습한 모델과 가깝다"는 증명 주장(가정 미확인) | provable robustness, train-from-scratch |
| FedRecover (Cao 2023; 복구) | 저장된 갱신·전역 모델로 "서버가 클라이언트 갱신을 스스로 추정"해 악성 제거 후 모델을 복구 | 코시 평균값 정리 g = ḡ + H(ŵ−w̄); L-BFGS 로 헤세-벡터 곱 근사; 버퍼 s | 탐지가 아니라 복구. 악성이 이미 탐지됐다고 가정 | 워밍업·주기 보정·이상 보정·최종 조정 라운드에만 정확 갱신 요청 | 비용 절감 vs 오류율 | Thm.1: μ-강볼록, L-평활, L-BFGS 오차 ≤ M 이면 ‖ŵ_t−w_t‖ ≤ (√(1−ημ))^t‖·‖ + ηM/(1−√(1−ημ)) | integrated Hessian, historical information, abnormality threshold |
| MASA (Xu 2024/WACV 2025; 탐지) | "백도어 매개변수는 주 과제에 비활성이라, 깨끗한 입력으로 개별 언러닝하면 백도어 모델의 손실이 더 크게 튄다" | 서버 프록시 데이터에서 주 과제 손실을 최대화하는 개별 언러닝 손실; 사전 모델 융합(λ) | 언러닝은 주로 정상 매개변수 θ_M 을 건드리고, 백도어 매개변수 θ_B 는 덜 관여하므로 백도어 모델은 더 빨리·크게 잊는다(경험 관찰) | 중앙값 편차 점수 MDS c_i=(x_i−median)/σ, 반경 δ=1.0 이하만 집계 | 라운드별 로그 언러닝 손실 선그래프(클라이언트별), 공격 비율 대 RA 곡선 | 없음. f<n/2 는 차용 | individual unlearning, poison-coupling, proxy dataset, Median Deviation Score, filter radius |
| SubID / "Malicious Forgetting" (Zhao, Lu et al., INFOCOM 2026; 학위논문 2026) (초록) | "FUsion 백도어"(언러닝 과정을 이용한 주입)에 대해 "부분공간 기반 편차 식별"로 악성 갱신과 정상 잡음을 가른다 | 부분공간 편차(어떤 부분공간인지 본문 미확인) | (미확인) | (미확인) | (미확인) | (미확인) | FUsion backdoor, subspace-based deviation identification, FAUN |
| VeriFi (Gao 2022; 검증) | 언러닝은 "떠나는 참여자가 표식(marker)으로 독립 검증"할 수 있어야 한다 | 표식 5 종(모델 임베딩, BadNets, 경계 지문, 잊기 쉬운 기억 vFM, 틀린 기억 vEM); 지표 4 종(정확도·손실·영향·KL) | 언러닝이 되면 표식 위 성능이 ψ(w_tm)−ψ(w_{t+1}) ≥ δ 만큼 떨어진다 | 임계 δ (보편값 없음) | 라운드별 표식 성능 변화 | 없음. 적대 상황은 "일정 확률로만 탐지" | RTBF, RTV, marker, marking/checking, natural forgetting |
| 서베이 "Threats, Attacks, and Defenses in MU" (2403.13682) | 악성 언러닝을 "직접(direct)" 과 "사전조건(preconditioned)" 으로 나누고, 후자(오염+완화 표본 후 완화 제거)를 표 I 에서 "Camouflaged attacks" 로 분류 | — | 방어: 직접 공격은 해시·멤버십 추론, 사전조건 공격은 "깨끗 데이터 대 완화 데이터 성능 차" 또는 "같은 클래스 다른 표본과 기울기가 다른 표본 제거" | 임계값 미명시 | — | — | malicious unlearning, over-unlearning, preconditioned, camouflaged. FU 위협은 "대체로 미개척", "분산 실행 공격은 특히 은밀" |

---

## §2. 공격 논문이 "잠복"을 설명하는 방식 (우리가 이름 붙여야 할 기제)

공통점부터 적는다. 모든 공격 논문은 잠복을 **같은 입력 영역에 반대 라벨(또는 반대 손실항)을 함께 공급하는 것**으로 만든다. 설명 수준은 모두 "손실항 A 가 손실항 B 를 상쇄한다"는 한 문장이고, 어느 논문도 이것을 정리(theorem)나 명명된 이론 대상으로 올리지 않았다. 따라서 "왜 잠복하는가"를 이론 대상으로 올리는 일 자체가 비어 있다.

1. **Zhang et al. 2023 (poison + mitigation).** 오염 표본(트리거, 목표 라벨)과 완화 표본(같은 트리거 구성, 원 라벨)을 같이 학습한다. 설명은 "완화 표본이 백도어를 숨긴다" 한 문장. 손실항 없음. 방어로 제안한 것은 완화 표본의 **출력 불확실성**(지니 불순도, SISA 하위모델 간 분산)인데, 이는 "서로 다른 라벨을 받은 비슷한 입력은 모델이 확신하지 못한다"는 직관이다. 즉 저자들도 잠복의 흔적을 "모순 지도(contradictory supervision)가 남기는 불확실성"에서 찾았다.
2. **Liu et al. AAAI 2024.** "잠복(dormant) 트리거"라는 말을 쓰지만 기제 설명은 없다. 오염 단계는 "연결만", 언러닝 단계는 "증폭". 어휘: dormant, amplify, disguise.
3. **UBA-Inf.** 잠복을 **영향 함수**로 정량화한 유일한 공격이다. 위장 표본을 "오염 표본에 대한 영향이 반대"가 되도록 LiSSA(헤세 근사)·PGD 로 최적화한다. 어휘: camouflage, on-demand, backdoor vanishing(위장이 너무 세서 백도어가 사라짐), backdoor exposure(위장이 약해서 미리 드러남). 이 두 실패 모드는 **상쇄 크기를 정확히 맞추어야 한다**는 뜻이며, 우리 불변량("각각은 크고 합은 0")이 공격의 필요조건임을 공격자 쪽 언어로 뒷받침한다. 본문은 못 열었으므로(§5) 인용 시 슬라이드·초록 수준으로 쓴다.
4. **BadFU.** 가장 명시적이다. Eq.8 에서 전역 갱신에 p₁(∇L_bd + ∇L_c) 가 들어가고, 언러닝이 ∇L_c 를 빼면 "백도어 기울기가 온전히 작용". 어휘: counterbalance, counteract, masking, suppressive effect, cancel out, dormant. 왜 FL 에서 잘 숨는지에 대한 추가 설명: 평균화가 백도어 기울기를 희석, non-IID 가 트리거 표현을 클라이언트별로 만든다. 방어 논의: 요청 심사, 사후 감시, DP.
5. **FUBA.** 억제자를 아예 "Adv-defender" 라고 부르고 공격을 두 적대 클라이언트의 **게임**으로 모델링한다. 억제 수단이 "model consistency loss"(전역 모델이 깨끗한 모델과 일관되게 보이도록 미는 손실)라는 점이 중요하다: 억제자는 트리거 입력만 고치는 것이 아니라 **전역 모델을 깨끗하게 보이게 하는 방향 전체**를 민다. 우리 관찰(억제자 갱신 노름 3.6 배, 깨끗한 입력으로의 누출)과 맞는다. 본문 미확인.
6. **ReVeil, Clean Unlearning Attack.** 둘 다 "conflicting information / conflicting learning signal 이 억제한다"는 말을 쓴다. 기제 이름으로 "conflict" 가 공격 쪽에서 이미 쓰이고 있다는 뜻이다.
7. **FedMUA, BadUnlearn.** 잠복형이 아니라 요청 자체 또는 언러닝 중 갱신을 조작하는 공격. 방어 불변량은 각각 "초기 라운드 기울기 노름" 과 "이력으로 추정한 갱신과의 일치"로, 둘 다 **요청자 한 명의 갱신만** 본다. 상쇄는 두 몫의 관계이므로 이 둘은 우리 공격을 보지 못한다(POSITIONING.md 따름 1 과 같은 논지).

요약: 공격 쪽 어휘는 camouflage / mitigation / counterbalance / conflict / suppress / defender / on-demand 이고, 가장 정교한 정량화는 영향 함수(UBA-Inf)다. 방어 쪽에서 잠복형을 겨냥한 것은 Zhang 2023 의 불확실성 탐지(중앙집중, SISA)뿐이며 FL 에는 없다.

---

## §3. 그룹 C 렌즈: 우리 상쇄를 무엇이라 말하게 해 주는가

각 항목: 무엇을 말하게 해 주는가(2–4 문장) / FL 방어가 이미 그 틀을 쓰는가(1 문장) / 판정.

### C1. 기울기 충돌(gradient conflict): PCGrad, CAGrad
- 말하게 해 주는 것: 운반자 집단과 억제자를 두 "과제"로 보면, 두 과제 기울기의 코사인이 음수인 상태가 PCGrad 의 "충돌(conflicting gradients)" 정의와 정확히 같다. PCGrad 의 "비극적 삼중(tragic triad)"(충돌 + 지배 기울기 + 높은 곡률)은 "왜 평균 집계가 한쪽을 지우는가"를 설명하는 어휘를 준다: FedAvg 는 충돌을 해소하지 않고 그냥 더하므로 두 과제가 서로를 지운다. CAGrad 의 "최악 과제의 국소 개선" 개념은 "어떤 클라이언트 하나를 빼면 가장 크게 바뀌는 과제" 를 정의하는 데 쓸 수 있다.
- FL 방어가 쓰는가: 방어로 쓴 예는 못 찾았다. 공격 쪽(Act in Collusion 2024: 악성 기울기들 사이 코사인이 모두 음수라 "aggregation-stage conflict" 가 생긴다며 Backdoor Replay 로 정렬)과 FU 효용 쪽(Orthogonal Steepest Descent 2024: 언러닝 기울기와 다른 클라이언트 기울기의 충돌을 투영으로 완화)에서 쓴다.
- 판정: **부분**. 어휘("충돌", "간섭")는 얻지만 대상은 결국 "코사인이 음수"라 긴장과 깊이가 같다. 다만 PCGrad 정리 2 의 조건(충돌 정도 ≥ 크기 유사도 함수)은 "상쇄가 성립하려면 크기가 맞아야 한다" 를 수식으로 말하는 데 인용할 수 있다.

### C2. 섀플리 상호작용 지수(Shapley interaction index) / 시너지: Grabisch–Roubens, Sundararajan 2020
- 말하게 해 주는 것: 클라이언트를 경기자로, 부분집합 S 의 누적 갱신으로 만든 모델의 클래스 c 밀기 v_c(S) 를 가치 함수로 두면, 쌍 (i,j) 의 이산 2 차 도함수 δ_ij v(T) = v(T∪ij) − v(T∪i) − v(T∪j) + v(T) 가 상호작용이다. 운반자 집합 A 와 억제자 S 는 "각자의 한계 기여는 크고 둘을 합친 연합의 가치는 0" 이므로 **연합으로는 더미(dummy coalition)인데 구성원은 더미가 아닌** 구조다. 이것은 "긴장"을 집합 수준 대상(연합)으로 올려 주고, k 명 대 1 명이라는 우리 위협 모델을 자연스럽게 담는다. 주의: v 가 갱신에 선형이면(우리 E_j 처럼 1 차 근사) 모든 상호작용은 0 이 되고 "합이 0" 으로 환원된다. 상호작용이 0 이 아니려면 v 를 실제 모델 평가(비선형)로 정의하거나, 선형일 때는 "한계 기여의 공분산 구조"(아래 §4 R2) 로 말해야 한다.
- FL 방어가 쓰는가: FedSV(ICC 2024) 는 **보통 섀플리 값**(한계 기여)을 악성 탐지에 쓰고 "그룹별 기여" 를 언급하지만 상호작용 지수나 공모는 다루지 않는다. 수직 FL 기여 측정에 "Shapley group interaction index"(2019) 가 있으나 보안 용도가 아니다. 상호작용 지수로 공모를 잡는 FL 방어는 못 찾았다.
- 판정: **적합** (단, 비선형 평가 또는 공분산 재정식화 조건부).

### C3. 영향 함수(influence function) / 데이터 귀속의 상쇄: Koh & Liang 2017
- 말하게 해 주는 것: 삭제 요청은 정의상 leave-one-out(LOO) 이고, 영향 함수는 LOO 의 1 차 근사다. "정상 연합은 클라이언트 하나를 빼도 클래스별 출력이 거의 안 변한다(안정성, stability)" 가 불변량이 되고, 공격은 정의상 억제자 하나를 뺄 때 출력이 크게 변해야 하므로(사전 ASR 10 % → 사후 90 %) 불변량을 깰 수밖에 없다. 이것이 "정상은 만족하고 공격은 반드시 깬다" 를 가장 깔끔하게 주는 틀이다. 우리 "빼 보기(trial removal)" 가 바로 이 양을 계산하고 있고, FedRecover 의 코시 평균값 정리·L-BFGS 보정으로 1 차 근사를 2 차로 올릴 수 있다. UBA-Inf 가 공격 쪽에서 같은 도구로 위장을 만들었으므로 "공격자의 설계 도구가 곧 방어자의 측정 도구" 라는 대칭을 말할 수 있다.
- FL 방어가 쓰는가: 탐지에 영향 함수를 쓰는 FL 방어는 못 찾았다. FedRecover 는 같은 수학을 복구에, FLDetector 는 일관성 예측에, FedMUA 는 공격에 쓴다. RONI(leave-one-out 정확도) 를 FL 에 옮긴 변형은 있으나 요청 심사가 아니다.
- 판정: **적합**.

### C4. 손실 지형 / 모드 연결성(mode connectivity): Zhao 2020; Invariant Aggregator; Hammer & Anvil
- 말하게 해 주는 것: w_N(요청 전) 과 w_{N∖j}(요청 후) 를 잇는 경로를 공개 입력으로 평가하면, 정상 요청은 경로 위 예측이 거의 변하지 않고, 공격은 클래스 c 쪽 예측이 경로를 따라 단조롭게 변한다. Hammer & Anvil 의 δ(정상 집계와 악성 집계의 간격) 로 보면, 우리 공격은 **집계 수준 δ ≈ 0** 을 상쇄로 만들어 Anvil(이상치 제거)과 Hammer(미세조정) 둘 다 피하는 공격이다. 이는 선행 이론 틀 안에서 우리 공격이 왜 새로운지 말하는 데 쓸 수 있다.
- FL 방어가 쓰는가: Invariant Aggregator 가 "평평한 손실 지형" 논리를, Zhao 2020 이 중앙집중 백도어 복구에 모드 연결성을 쓴다. FL 탐지에 경로 평가를 쓴 예는 못 찾았다.
- 판정: **부분**. 공개 입력에 트리거가 없으므로 경로 위에서 보이는 것은 깨끗한 입력의 클래스별 이동뿐이다. 위치 설정(positioning)용으로는 좋다.

### C5. 신경 접선 커널(NTK) 관점: Hayase & Oh 2022
- 말하게 해 주는 것: 선형화 아래 예측은 f(x) = y^T K^{-1} K(·,x) 이고, 오염 1 점이 주는 변화는 커널 가중치 × 잔차 (y_p − f(x_p)) 에 선형이다(Eq.4). 같은 입력 영역에 반대 라벨을 주면 잔차 부호가 반대라 **커널 회귀에서 정확히 상쇄**되고, 학습된 함수는 그 영역에서 "라벨 평균" 이 된다. 언러닝으로 한쪽을 빼면 라벨 평균이 뒤집힌다. 이것이 잠복의 가장 짧은 기계적 설명("모순 지도의 평균화")이며, 우리 E_j[x,c] = ⟨U_j, ∇f_c(x)⟩ 는 바로 이 선형화의 클라이언트별 분해다. 즉 우리 측정은 "경험적 NTK 특징 공간에서의 클라이언트 기여 분해" 라고 부를 수 있다.
- FL 방어가 쓰는가: NTK 를 FL 백도어 방어에 쓴 예는 못 찾았다.
- 판정: **적합(설명용)**. 탐지 통계로는 C3 와 같고, 기제 설명과 측정의 정당화에 쓴다.

### C6. 모델 병합 간섭 / 과제 산술(task arithmetic), TIES 부호 충돌: Ilharco 2023, Yadav 2023
- 말하게 해 주는 것: 누적 갱신 U_j 는 과제 벡터(task vector)다. Ilharco 는 "과제 벡터 부정(negation) = 잊기" 라고 했으므로, 억제자의 U_S 는 운반자 과제 벡터 Σ_A U_i 의 **부정 벡터(anti task vector)** 이고 언러닝은 그 부정의 부정이다. Ilharco 의 관찰 "서로 다른 과제 벡터는 거의 직교(cos ≈ 0)" 는 정상 클라이언트 쌍의 영분포를 주고, TIES 의 "부호 충돌이 병합 값을 줄인다(shrink)" 는 상쇄의 좌표별 측정(부호 충돌 질량)을 준다. 모델 병합 쪽에 "Backdoor Vectors: a task arithmetic view"(2025) 가 있어 "백도어 벡터" 라는 이름을 빌릴 수 있다.
- FL 방어가 쓰는가: FedSECA(2025) 가 TIES 식 부호 선출을 비잔틴 집계에, Invariant Aggregator 가 좌표별 부호 일치를 마스크에 쓴다. 쌍별 "반대 과제 벡터" 나 "언러닝 = 부정" 틀은 못 찾았다.
- 판정: **부분~적합**. 부호 일치 틀은 선점됐고, 부정 벡터·백도어 벡터 틀은 비어 있다.

### C7. 연합·공모 탐지(게임 이론, 시빌, 크라우드소싱 공모)
- 말하게 해 주는 것: 공모(collusion)는 "구성원 각자는 정상으로 보이고 연합만 보면 목표가 드러나는" 구조인데, 우리 공격은 반대로 "각자는 크게 밀고 연합은 0 으로 보이는" 구조다. 즉 FoolsGold 가 가정한 시빌(비슷한 갱신)의 **거울상(anti-sybil, 반대 갱신)** 이다. FUBA 가 공격을 두 적대 클라이언트의 게임으로 쓰므로, 방어도 "연합의 가치" 언어로 받을 수 있다.
- FL 방어가 쓰는가: FoolsGold(시빌 유사성), 공모 무임승차 감사, "서로 가까운 클라이언트 표시" 등은 모두 **유사성**을 본다. 반대 정렬 연합을 보는 방어는 못 찾았다. FoolsGold 는 "합이 0 인 잡음" 을 적응 공격으로 언급했으나 탐지하지 않고 지표 특성 제한으로 무력화한다고만 했다.
- 판정: **부분**. 어휘("anti-sybil", "antagonistic coalition")는 유용하나 측정 대상은 C2·C3 가 준다.

### C8. 기울기 가림(obfuscated/masked gradients) 유추: Athalye 2018
- 말하게 해 주는 것: "신호가 없어 보이는 것은 안전의 증거가 아니라 설계된 결과일 수 있다" 는 방법론적 교훈. 깨끗해 보이는 집계는 "가려진 집계(masked aggregate)" 이며, 방어 평가는 집계가 아니라 성분(클라이언트별 몫) 위에서 해야 한다는 주장을 뒷받침한다. Athalye 의 진단 징후 목록("단일 스텝이 반복보다 세다" 등)처럼 "상쇄의 징후 목록"(개별 노름이 큰데 합이 작다, 하나를 빼면 튄다)을 만들 수 있다.
- FL 방어가 쓰는가: 유추로 쓴 예 없음.
- 판정: **부분**(수사·방법론용, 측정 대상 아님).

### C9. 해독제 데이터 / friendly noise: Liu 2022, 희석 방어
- 말하게 해 주는 것: 억제자는 공격자가 **같이 넣었다가 나중에 거두는 해독제(withdrawable antidote)** 다. friendly noise 는 "오염은 좁고 날카로운 고손실 영역을 만든다" 는 손실 지형 설명을 주며, 위장 표본은 그 영역을 평평하게 덮어 두는 역할이다. 언러닝은 덮개를 걷는다.
- FL 방어가 쓰는가: 없음.
- 판정: **부분**(어휘·직관용. 서버는 공개 입력만 있어 손실 지형의 날카로움을 트리거 영역에서 잴 수 없다).

### C10. 중첩(superposition) / 뉴런 간섭
- 말하게 해 주는 것: 비직교 특징이 같은 방향을 나눠 쓰면 간섭한다는 틀. 운반자와 억제자가 같은 방향(트리거 특징)을 반대 부호로 쓴다는 묘사는 가능하나, 측정으로 옮길 대상이 없다. (원문 페이지는 길이 초과로 본문을 못 읽었고, 내용은 기억에 의존한다.)
- FL 방어가 쓰는가: 없음.
- 판정: **부적합**.

---

## §4. 추천 순위: "긴장" 을 대체할 세 가지 틀

조건: (a) 긴장보다 깊고, (b) FL 방어가 이미 쓰는 틀이 아니며, (c) 저장된 라운드별 갱신 + 공개 입력만으로 계산 가능. 기호는 §1 서두와 MATH.md 를 따른다.

먼저 기제 이름 하나를 고정할 것을 권한다: **"모순 지도의 평균화(averaging of contradictory supervision)"** (C5). 잠복 = 같은 입력 영역에 반대 라벨을 공급해 FedAvg 가 라벨 평균을 학습한 상태; 활성화 = 한쪽 공급자의 제거로 평균이 뒤집히는 것. 공격 논문들의 camouflage/counterbalance/conflict 를 이 한 이름으로 묶고, 아래 세 틀은 이 기제가 남기는 흔적을 각각 다른 대상에서 재는 것으로 서술한다.

### R1. 삭제 민감도 = leave-one-out 안정성 (영향 함수 틀, C3+C5)
- 이름: **삭제 민감도(unlearning sensitivity)** 또는 **LOO 안정성 불변량**.
- 불변량 한 문장: "정상 연합에서는 어느 클라이언트 하나를 빼도 공개 입력 위 클래스별 출력이 정상 LOO 분포 안에서만 움직인다; 잠복 백도어 공격은 억제자를 뺄 때 이 분포를 벗어나야만 깨어난다."
- 왜 정상은 만족하고 공격은 깨는가: 정상은 FedAvg 가중 1/n 과 손실 평활성으로 한 클라이언트의 LOO 효과가 작다(FedRecover Thm.1 식의 1 차항을 그대로 인용 가능: 삭제 후 모델과 재학습 모델의 거리가 유계). 공격의 정의가 "요청 전 ASR 낮음, 요청 후 ASR 높음" 이므로 억제자의 LOO 효과는 그 차이 이상으로 커야 한다. non-IID 소유자(한 클래스를 많이 가진 정상)도 LOO 효과가 크지만 **방향이 다르다**: 소유자를 빼면 그 클래스 밀기가 *줄고*(자기 클래스로의 밀기를 잃음), 억제자를 빼면 특정 클래스 밀기가 *늘며* 그 증가분이 다른 클라이언트들의 밀기 합과 일치한다(아래 R2 로 넘어감).
- 계산량: Δ_{−j}[x,c] = f_c(x; w_T − U_j) − f_c(x; w_T) (실제 평가), 또는 1 차 근사 −E_j[x,c]; 필요하면 FedRecover 식 L-BFGS 로 H(ŵ−w̄) 항 보정. 클래스별 요약 d_j[c] = mean_x Δ_{−j}[x,c]. 영분포: 모든 클라이언트 j 에 대해 같은 양을 계산해 MAD 기반 z 점수(MASA 의 MDS 와 같은 형식). 요청자의 z 가 임계 이상이면 "시험 실패".
- 증거 그림: 클라이언트별 d_j[c*] 히스토그램(억제자가 꼬리), 라운드 창별 d_j 의 궤적.
- 보장 스케치: μ-강볼록·L-평활 가정에서 정상 LOO 거리 상계(FedRecover/FLTrust 형 식) 와 공격 필요조건(ASR 차이 ≥ γ ⇒ ‖Δ_{−S}‖ ≥ g(γ)) 을 나란히 놓아 분리 조건을 제시.
- 새로움: 영향 함수·LOO 를 요청 심사에 쓰는 FL 방어는 없다. FedRecover(복구)·FLDetector(일관성)·UnlearnGuard(언러닝 중 갱신 추정)와 수학은 겹치지만 대상(요청자 1 명의 삭제 효과)과 역할이 다르다. 공격자(UBA-Inf, FedMUA)가 같은 도구를 쓴다는 대칭을 서술에 쓴다.

### R2. 상쇄 연합 = 음의 상호작용 / 한계 기여의 반상관 (협력 게임 틀, C2+C7)
- 이름: **상쇄 연합(cancelling coalition)**, 영어로 antagonistic coalition 또는 jointly-dummy coalition.
- 불변량 한 문장: "정상 연합에서 서로 다른 클라이언트의 클래스별 기여는 거의 직교하거나 같은 부호로 더해진다(가산적); 공격에서는 요청자와 어떤 연합 S 의 기여가 반대 부호·같은 크기로 묶여 연합 전체는 더미지만 구성원은 더미가 아니다."
- 왜 정상은 만족하고 공격은 깨는가: 정상 쌍의 과제 벡터는 거의 직교(Ilharco 의 관찰; non-IID 에서도 양의 상관이 생길지언정 음의 큰 상관은 드물다). 공격은 억제자가 운반자 합을 정확히 되돌려야 하므로(UBA-Inf 의 vanishing/exposure 양쪽 실패를 피하려면) E_S ≈ −Σ_{i∈A} E_i 가 **트리거 영역 위에서** 성립해야 하고, 억제가 깨끗한 입력으로 새면(FUBA 의 consistency loss) 공개 입력 위에서도 음의 상관이 남는다.
- 계산량 (두 수준):
  - 쌍/연합 수준(선형, 우리 E 로 바로): 요청자 j 와 후보 연합 S 에 대해 상쇄 지수 κ_j(S) = 1 − ‖E_j + Σ_{i∈S} E_i‖² / ‖E_j‖² (MATH.md 의 c_j 와 같음) 와 **반상관 ρ_j(S) = corr_{x,c}(E_j, Σ_{i∈S} E_i)**. 정상 영분포: 모든 j 에 대해 탐욕으로 얻은 최대 κ, 최소 ρ 의 분포(gap statistic 또는 MAD).
  - 상호작용 수준(비선형, 선택): v_c(T) = mean_x f_c(x; w_T − Σ_{i∉T} U_i) 로 실제 평가하고 I_c(j,S) = v_c(j∪S) − v_c(j) − v_c(S) + v_c(∅). 선형이면 0 이므로, 0 에서 벗어나는 크기가 "상쇄가 비선형 특징(트리거 검출기)을 통해 일어났다" 는 추가 증거가 된다. 클라이언트 수가 작은 cross-silo(우리 K=6~20)에서 계산 가능.
- 증거 그림: 클라이언트 쌍 ρ 행렬 히트맵(FoolsGold 히트맵의 거울상: 정상은 0 근처, 억제자–운반자 블록만 진한 음수), κ 대 z 산점도.
- 보장 스케치: 가치 함수가 가산적이면 모든 상호작용이 0(정상 극한); 공격 성공(ASR 차이 γ)은 |ρ_j(S)| ≥ h(γ) 를 함의 — 상쇄의 필요조건을 상관으로 쓴다.
- 새로움: FedSV 는 섀플리 값(한계 기여)만, FoolsGold 는 양의 유사성만 본다. 음의 상호작용/반상관 연합을 보는 FL 방어는 못 찾았다. "연합" 어휘는 FUBA 의 게임 서술과 이어진다.

### R3. 반대 과제 벡터 = 언러닝은 부정의 부정 (과제 산술 틀, C6)
- 이름: **반대 과제 벡터(anti task vector)** / **백도어 벡터의 상쇄 쌍**.
- 불변량 한 문장: "정상 클라이언트들의 누적 과제 벡터는 출력 관련 부분공간에서 거의 직교한다; 공격은 어떤 클라이언트의 과제 벡터가 다른 연합의 과제 벡터의 부정(−)과 같은 방향·같은 노름이어야 하고, 삭제 요청은 그 부정을 다시 부정해 백도어 벡터를 되살린다."
- 왜 정상은 만족하고 공격은 깨는가: Ilharco 의 "거의 직교" 관찰 + TIES 의 "부호 충돌이 병합 값을 줄인다" 가 정상 영분포와 상쇄 기제를 각각 준다. 공격은 정의상 Σ_A U_i + U_S ≈ 0 (관련 부분공간) 이어야 한다.
- 계산량: 출력층(또는 크기 상위 k% 로 trim 한 좌표) 위에서 쌍별 코사인 cos(U_i,U_j), 노름 비 ‖U_i‖/‖U_j‖, **부호 충돌 질량** m_ij = Σ_k |U_ik||U_jk| 1[sign 충돌] / Σ_k |U_ik||U_jk|, 상쇄 비 r_ij = ‖U_i+U_j‖/(‖U_i‖+‖U_j‖). 입력이 전혀 필요 없다(매개변수 공간만). 영분포: 정상 쌍은 cos ≈ 0, m ≈ 0.5, r ≈ 1/√2 근처; 상쇄 쌍은 cos → −1, m → 1, r → 0.
- 증거 그림: TIES 식 "부호 충돌 질량" 쌍 행렬; 과제 벡터 코사인 히트맵(Ilharco Fig.5 형식).
- 새로움: 부호 일치 틀은 Invariant Aggregator·FedSECA 가 선점. "언러닝 = 과제 벡터 부정, 억제자 = 반대 과제 벡터" 틀과 쌍별 부호 충돌 질량은 FL 방어에 없다. 3 위인 이유: 코사인·부호 통계라는 대상이 선행과 겹쳐 틀의 새로움이 부분적이고, 매개변수 공간 측정은 우리 실험에서 non-IID 소유자와 억제자를 잘 못 가른 적이 있다(MATH.md 1.6).

### 추천 조합
논문 서술: 기제 = "모순 지도의 평균화"(C5, NTK 선형화로 E_j 를 정당화) → 불변량 = R1(삭제 민감도) 을 주 불변량으로, R2(상쇄 연합) 를 "누가" 를 찾는 귀속 불변량으로, R3 는 입력이 없을 때의 보조 통계로. "긴장" 이라는 말은 버리고 "상쇄 연합의 삭제 민감도" 로 통일한다.

---

## §5. 읽은 URL

본문(HTML/ar5iv)을 읽은 것:
- BadFU https://arxiv.org/html/2508.15541v1 (본문; 끝부분 10k 자는 참고문헌·부록만 확인)
- Zhang 2023 https://arxiv.org/html/2310.10659v2
- Liu AAAI 2024 전문 텍스트 https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/AAAI/2024/backdoor_attacks_via_machine_unlearning__534abad7.txt , 서지 https://ojs.aaai.org/index.php/AAAI/article/view/29321
- MASA https://arxiv.org/html/2411.01040v1
- FedMUA https://arxiv.org/html/2501.11848v1
- FedRecover https://arxiv.org/html/2210.10936
- VeriFi https://arxiv.org/html/2205.12709
- FLDetector https://arxiv.org/html/2207.09209v4
- FoolsGold https://arxiv.org/html/1808.04866v5
- FLAME https://arxiv.org/html/2101.02281v5 (10 절 3DFed 반론은 잘림)
- BaFFLe https://arxiv.org/html/2011.02167v2
- FLTrust https://arxiv.org/html/2012.13995v3
- DeepSight https://arxiv.org/html/2201.00763v1 (부록 적응 공격 잘림)
- Krum https://ar5iv.labs.arxiv.org/html/1703.02757
- Spectral Signatures https://ar5iv.labs.arxiv.org/html/1811.00636
- Activation Clustering https://ar5iv.labs.arxiv.org/html/1811.03728
- CRFL https://ar5iv.labs.arxiv.org/html/2106.08283
- Invariant Aggregator https://arxiv.org/html/2210.01834v4
- Hammer and Anvil https://arxiv.org/html/2509.08089v1
- Clean Unlearning Attack https://arxiv.org/html/2506.12522
- ReVeil https://arxiv.org/html/2502.11687v1
- Act in Collusion https://arxiv.org/html/2411.03926
- 서베이 Threats/Attacks/Defenses in MU https://arxiv.org/html/2403.13682v5
- PCGrad https://ar5iv.labs.arxiv.org/html/2001.06782
- TIES https://ar5iv.labs.arxiv.org/html/2306.01708
- Task Arithmetic https://ar5iv.labs.arxiv.org/html/2212.04089
- Koh & Liang https://ar5iv.labs.arxiv.org/html/1703.04730
- Shapley-Taylor https://ar5iv.labs.arxiv.org/html/1902.05622
- Hayase & Oh NTK https://ar5iv.labs.arxiv.org/html/2210.05929
- Athalye https://ar5iv.labs.arxiv.org/html/1802.00420
- Friendly Noise https://ar5iv.labs.arxiv.org/html/2208.10224

초록·요약 페이지만 읽은 것:
- UBA-Inf https://www.usenix.org/conference/usenixsecurity24/presentation/huang-zirui (PDF https://usenix.org/system/files/usenixsecurity24-huang-zirui.pdf 는 크기 초과로 실패, 슬라이드 PDF 는 이진 변환 실패; 기제 세부는 검색 요약과 인용 논문 설명에 의존)
- FUBA https://ieeexplore.ieee.org/document/11231135/ (페이지 본문 비어 있음) → 초록은 https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/TAI.2025.3630110 과 검색 결과(IEEE TAI vol.7 no.5, Sheng·Bao·Guo·Fu)
- SubID / Malicious Forgetting: 학위논문 초록 https://digitalcommons.usf.edu/etd/11456/ , INFOCOM 2026 서지 https://scholars.unf.edu/en/publications/malicious-forgetting-backdoor-injection-in-active-federated-unlea/ (본문 미확인)
- FLARE https://vtechworks.lib.vt.edu/items/8e4ec51d-23ef-463c-a3e4-f2d1e9b6f9cc (NSF PAR PDF 는 이진 변환 실패)
- BadUnlearn/UnlearnGuard https://arxiv.org/abs/2501.17396
- FedSV https://arxiv.org/abs/2502.17526
- CAGrad https://arxiv.org/abs/2110.14048
- Orthogonal Steepest Descent FU https://arxiv.org/abs/2412.20200
- SPMC https://chatpaper.com/chatpaper/paper/166413
- Fed-ADR(orchestrated clients) https://arxiv.org/abs/2609.27124
- Lockdown NeurIPS 2023 PDF 는 이진 변환 실패(본문 미확인)
- Toy Models of Superposition https://transformer-circuits.pub/2022/toy_model/index.html (페이지 길이 초과, 본문 미읽음)

검색 결과만 참고한 것(본문 미열람): FedGrad, RoPE, DeTrigger, FedSECA, Backdoor Vectors(2510.08016), BadTV, pFedSV, Shapley group interaction index(1909.08525), 공모 무임승차 감사 논문, "Federated Unlearning in Edge Networks" 서베이(2601.09978).

---

## §6. 우리 판단 (2026-10-13, 조사 결과를 실험 사실과 맞춘 것)

조사의 §4 는 세 틀을 R1 > R2 > R3 로 추천했다. 우리가 이미 잰 것과 맞추면 순서가 바뀐다.

| 틀 | 조사의 추천 | 우리 실측 | 결정 |
|---|---|---|---|
| 기제 이름 "모순 지도의 평균화" (NTK 선형화) | 채택 권고 | 장부의 밀기 mean_x[log p_c(x; θ+u_j) − log p_c(x; θ)] 는 1 차에서 ⟨u_j, ∇ log p_c⟩ = 선형화의 클라이언트별 분해. 측정과 맞는다 | **채택**. "왜 잠복하는가" 를 설명하는 문장으로 쓴다 |
| R1 삭제 민감도 (LOO 안정성) | 주 불변량 | 이미 있다 = DOSE 의 상대 기준 flow z (R-009, 3/3, 소유자 z 0). 그러나 요청 시점에 빼 봐야 하고, 사용자가 "언러닝 시뮬레이션" 으로 보아 노벨티로 인정하지 않는다 | **확인 단계로만** (ALGORITHMIZATION §3 의 대응 ④) |
| R2 상쇄 연합 = "연합은 더미인데 구성원은 더미가 아니다" (협력 게임) | 귀속 불변량 | 균형 장부가 바로 이것을 학습 중에 잰다: 클래스 c 에서 {neg} ∪ Pos 의 합은 0 (더미), 각자는 크다 (더미 아님). 노트북 1 시드: 경보 16/99 라운드, 위장자·운반자 넷 정확 (R-013) | **주 불변량**. 이름: 상쇄 연합(cancelling coalition), 장부 = 그 연합을 학습 중에 찾는 절차. 비선형 상호작용 I_c(j,S) 는 DOSE 확인 단계가 재는 것과 같다 |
| R3 반대 과제 벡터 (매개변수 공간 코사인·부호 충돌) | 보조 | 이미 재서 기각: 위장자–운반자 코사인 +0.56 = 위장자–정상 +0.60 (hedge pilot, MATH.md 1.6). 매개변수 공간에서는 상쇄가 안 보인다 | **통계로는 버림**. "언러닝 = 부정의 부정" 은 서술 어휘로만 |

정리된 논문 서술 순서: 기제(모순 지도의 평균화) → 불변량(정상 학습에는 상쇄 연합이 없다; 잠복 백도어는 반드시 하나 만든다) → 점수(균형 장부: 클래스별 헤지 비율·음의 다리·양의 다리 집합, 학습 중 누적) → 판정(양의 다리 ≥ 2 = 공모, 1 = 정상 큰손) → 대응(DOSE 로 비선형 확인 후 운반자 제외 재학습) → 보장(공격의 필요조건 Σ P ≈ 0, |P_S| ≈ |Σ_A P| 가 곧 장부의 양성 조건).
남은 불확실: 조사가 FUBA·UBA-Inf·SubID 본문을 못 열었다 (§5). 인용 전에 본문으로 확인한다.
