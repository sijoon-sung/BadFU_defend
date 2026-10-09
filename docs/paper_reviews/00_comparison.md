# 언러닝 악용 공격 4편 비교 (BadFU · FUBA · FedMUA · Malicious Forgetting)

개별 정리: [01_BadFU](01_BadFU.md) · [02_FUBA](02_FUBA.md) · [03_FedMUA](03_FedMUA.md) · [04_Malicious Forgetting](04_malicious_forgetting_FUsion_SubID.md)

## 1. 한눈에 비교

| | **BadFU** (RAID'25) | **FUBA** (IEEE TAI'26) | **FedMUA** (TIFS'25) | **Malicious Forgetting / FUsion** (INFOCOM'26) |
|---|---|---|---|---|
| 저자 그룹 | Newcastle(AU)+SJTU | Sydney | Chen, Lin, … Wang | USF+UF |
| 저자 겹침 | — | 없음 | 없음 | 없음 |
| **공격 목표** | 트리거 백도어 | 트리거 백도어 (IBA/DBA) | **특정 타깃 샘플 오분류** (트리거 없음) | 트리거 백도어 |
| **백도어를 심는 시점** | 학습 중 | 학습 중 | 심지 않음 | **언러닝 중** (학습 중엔 정직) |
| **활성화 메커니즘** | 위장 샘플 D_c 삭제 → 상쇄 제거 | Adv-defender 기여 삭제 → 상쇄 제거 (+재학습형 FU에선 attacker가 γ=1로 재주입) | 타깃 쪽으로 섭동한 샘플을 지워 **over-unlearning** | 준비해 둔 서브네트워크를 언러닝 라운드에 융합 |
| **상쇄 쌍 구조** | 논문: 한 클라이언트 **내부** / 코드: 두 클라이언트 분리 | 클라이언트 **간** (attacker 군집 vs defender); 샘플 단위 확장도 있음(Table VII) | 없음 | 없음 |
| 요청자 | 공격자 | 공격자(defender 역) | 공격자 | 공격자 |
| 삭제 단위 | 샘플 (코드: 클라이언트) | 클라이언트 (샘플 단위 확장 ASR 1.0) | 샘플(특징) | 샘플 (active FU) |
| 학습 중 업데이트 조작 | 없음 (데이터만) | 있음 (재가중·γ 스케일링·consistency 손실) | 없음 | 없음 |
| 공격자 비율 / 공모 | 1/5 | 5/20 (준조정, 시작 후 통신 없음) | 2/20 | 10/50 (협력) |
| 데이터셋 | MNIST, CIFAR-10/100 | MNIST, CIFAR-10, ImageNet | Purchase, MNIST, CIFAR-10/100, Credit | MNIST, CIFAR-10, HAR, AG News |
| FU 방법 | Retrain, FedEraser, FedU, SIFU | Retrain, FedEraser, PGD, RobustFU / SGA, KD | FedEraser, KNOT | NoT, FedRR, ConFUSE (active) |
| 잠복 ASR | 11–40% | **<10%** (설계상 τ=0.1) | 해당 없음 | 해당 없음 |
| 활성 ASR | Retrain 91–99, FedEraser 49–96 | 재학습형 73–100 / **비재학습형 60–83** | 84–93 (IID) | 62–99 |
| 시드 | 3 | **10** | 타깃 40개 평균 | 미기재 |
| **시험한 방어와 결과** | Median/Trimean: 잠복만 낮춤. NC: 미탐지+오탐 | **10종 전부 무력**(FL 단계 적용): ASR 0.52–0.97. 자체 시험: clean셋 미세조정 **0.15**, Top-k 0.19 | Krum/Median 무력. 자체 노름-IQR: ASR 15–58 잔존 | Trim 무력, Median은 유틸리티 손실. 자체 **SubID**: F1 1.0 |
| **정상 요청 시험** | Table V: 정상 1명 제거 → ASR **감소**(24→14) | Table V: 정상 1·3명 제거 → ASR 변화 없음 (**α=0.1에서도**) | — | 공격 없음 조건 FPR만 |
| 공개 코드 | 있음 (논문과 설정 불일치) | 있음 (Top-k 플래그 무효) | 있음 | 없음 |

## 2. 공격을 가르는 두 축

```
                       언러닝 중 새 행동이 있는가?
                       예 (주입형)                        아니오 (잠복형)
  요청자가 공격자    FUsion (active FU)                   BadFU, FUBA(비재학습 FU: KD/SGA)
                    BadUnlearn (passive FU)               FUBA(재학습 FU)는 두 축이 섞임: 잠복 + γ=1 재주입
  요청자가 정상      —                                    우리 fp3/fp4 관찰(class-owner 90%, 1차 근사: 5.8→38)
                                                         ← 반례 2건: BadFU Table V, FUBA Table V (Dirichlet α=0.1 포함)
```

- FedMUA는 이 표 밖(백도어가 아니라 타깃 샘플 오분류).
- **기존 방어는 전부 왼쪽 열용**: UnlearnGuard(이력으로 업데이트 예측·필터), SubID(학습기 부분공간 대비 언러닝기 편차)는 "언러닝 중 누군가 새 방향을 만든다"를 전제. 잠복형에서는 (i) 백도어 방향이 이미 정상 부분공간 안에 있고, (ii) 비재학습 FU에서는 언러닝 중 아무도 새 행동을 하지 않으며, (iii) SubID의 대응(전부 재학습)은 BadFU에서 ASR 34.6→95.1로 **공격을 완성**한다.
- **FUBA 재학습형 FU는 사실상 "FU 중 주입"을 포함**(각주 4, γ=1). FU 라운드에 방어를 걸면 잡힐 가능성이 크지만 저자는 FL 단계에만 방어를 걸었다. 순수 잠복형 효과는 KD/SGA 열(0.6–0.83)로 읽어야 한다.
- **요청 심사 계열**(BAU 불확실성, Qian 그래디언트, FedMUA 노름-IQR)은 "요청이 수상한가"를 봄 → 요청이 진짜 정상인 오른쪽 아래 칸에서는 볼 것이 없음. 다만 그 칸 자체가 실제 FU에서 성립하는지는 반례 2건 때문에 의심스럽다.

## 3. 네 논문이 공통으로 남긴 열린 문제

1. **요청의 정당성 검증** — 전부 제안만. Unlearn and Burn(ICLR'25)은 은밀한 요청에 심사가 무력함을 보임. FUBA는 "서버가 구별 불가"라고 명시.
2. **언러닝 후 백도어 검사 비용** — BadFU는 NC가 FL에서 무력, BAU는 "요청마다 NC는 너무 비싸다".
3. **강건 집계의 한계** — 네 편 모두 Median/Trimean/Krum이 활성화를 못 막거나(BadFU, FUBA, FedMUA) 유틸리티를 깎는다(MF). 잠복형에서는 잠복 ASR을 낮춰 위장을 돕는 모양새(BadFU Table VI).
4. **FU 단계 방어의 부재** — FUBA·BadFU 모두 방어를 FL 단계에만 적용. "FU 재집계 단계의 방어"는 SubID(주입형 전용)뿐.
5. **서버 clean 데이터로 미세조정하면 꽤 막힌다** (FUBA Table VIII, ASR 0.15) — 이 기준선을 clean 데이터 없이 넘는 것이 방어 논문의 최소 조건.
6. **실제 FU vs 근사** — 활성 ASR이 FU 방법에 크게 의존(BadFU: Retrain 99 vs FedEraser 49; FUBA: Retrain 1.0 vs KD 0.7; 우리: 1차 근사 34 vs retrain 95). 방어 평가는 반드시 실제 FU 위에서.

## 4. 교수님 지적("쌍 구조 전용 방어")에 대한 위치 정리

- 쌍 구조가 있는 건 **FUBA와 BadFU 공식 코드 설정뿐**. BadFU 본문(한 클라이언트 내부)·FUBA 샘플 단위 확장·FedMUA·FUsion에는 없음. → 쌍 검사는 4편 중 1.5편만 덮는다. 다만 "FUBA가 시험한 방어 10종이 모두 단일 클라이언트 점수형이라 쌍 관계를 못 본다"는 관찰은 Table II를 설명하는 데 쓸 수 있다.
- 범용성의 근거를 "공격 수"가 아니라 **"잠복형 전체: 빠지는 기여가 드러내는 것"**으로 잡으면 BadFU(코드)·FUBA(KD/SGA)를 한 검사로 덮고, FUsion/BadUnlearn·FUBA 재학습형은 UnlearnGuard·SubID·FU-단계 강건집계와 조합, FedMUA는 범위 밖으로 명시할 수 있다.
- 메커니즘 차별화: **업데이트 부분공간(SVD)은 SubID가 선점**. 남는 축은 "삭제 전후 모델의 기능 변화"(probe 입력에서 h0 vs h1의 답이 갈리는 입력이 한 클래스로 몰리는지). FedMUA 방어·Qian의 "지웠을 때 얼마나 움직이나" 관점과 같은 방향이지만 FL 잠복형 백도어에 적용한 사례는 없음(모델 전후 비교형 트로이 탐지 선행 확인 필요).
- **"정상 이탈 공격"은 현재 근거가 약하다**: 반례 2건(BadFU Table V, FUBA Table V α=0.1) vs 우리 관찰은 1차 근사뿐. class-owner 극단 편중 + 실제 FU에서 재확인 전에는 주장으로 세우지 말 것.

## 5. 바로 할 일

1. **go/no-go ①(정상 이탈)**: FUBA 코드에서 `forgot_client_idx`를 정상 클라이언트로, 분할을 class-owner 90%로, FU는 KD(비재학습)와 FedEraser로. 안 터지면 정상 이탈 공격은 접고 "잠복형 방어"로만 간다.
2. **go/no-go ②(순수 잠복형 분리)**: FUBA 재학습형 FU에서 attacker 참여를 끄거나 FU 라운드에 Median/FLAME을 걸어, 재학습형 ASR이 얼마나 "재주입" 때문인지 분리. 남는 KD/SGA 잠복 활성화가 우리 방어의 대상.
3. **기준선 고정**: FUBA Table VIII clean셋 미세조정(0.15)·Top-k(0.19) 재구현, UnlearnGuard-Dist/Dir, SubID 재구현, Median/Trimean(공격자 <50%), 언러닝 후 NC. 잠복형에서 UnlearnGuard/SubID가 실패함을 보이는 것이 "SVD면 SubID랑 같다"에 대한 실험적 답.
4. FUBA 보충자료(DOI 페이지) 확보 → D.5(m ≤ n/4 근거), F(Top-k 설정), G.2(저자가 인정한 한계) 확인.
5. FedMUA 공개 코드(ity207/FedMUA)로 범위 밖 공격 1종을 "검사가 무엇을 못 보는지" 예시로 확보.
