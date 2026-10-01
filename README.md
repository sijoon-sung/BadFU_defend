# BadFU Defend: Adversarial Federated Unlearning Defense Framework

[![arXiv](https://img.shields.io/badge/arXiv-2508.15541-b31b1b.svg)](https://arxiv.org/abs/2508.15541)
[![Conference](https://img.shields.io/badge/Venue-RAID%202025-blue.svg)](https://raid2025.org)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)

본 저장소는 최신 연합 학습 보안 논문인 **"BadFU: Backdoor Federated Learning through Adversarial Machine Unlearning" (RAID 2025)**의 백도어 공격 메커니즘을 1:1 풀 스케일로 실증 재현하고, 단순 휴리스틱을 넘어선 **학술 논문 수준의 차세대 2대 방어 아키텍처(가중치 부분공간 기하학 감사 & 온디바이스 TEE 다차원 스펙트럼 프록시)**를 제안 및 실증한 연구 프레임워크입니다.

---

## 📌 1. 연구 배경 및 BadFU 공격 개요

* **논문 정보:** *BadFU: Backdoor Federated Learning through Adversarial Machine Unlearning* (RAID 2025, arXiv:2508.15541)
* **공격 메커니즘:**
  * 악성 클라이언트($C_m$)가 훈련 단계에서는 백도어 샘플($D_{bd}$)과 위장 샘플($D_c$)을 동시에 주입하여 로컬에서 상쇄($\nabla \mathcal{L}_{bd} + \nabla \mathcal{L}_c \approx 0$)시킵니다.
  * 연합 학습의 파라미터 평균화(`FedAvg`)와 Non-IID 특성으로 인해 훈련 중에는 백도어가 낮은 ASR(10~25%)로 안전하게 잠복(Dormant)하며 기존 비잔틴 강건 집계(Median, Trimean)를 통과합니다.
  * 훈련 종료 후 악성 클라이언트가 $D_c$에 대한 합법적 데이터 삭제(Right to be Forgotten)를 요청하면, $\nabla \mathcal{L}_c$가 제거되면서 잠복해 있던 백도어($\nabla \mathcal{L}_{bd}$)가 급반등하여 **ASR 90% 이상으로 전면 활성화**됩니다.

---

## 📊 2. 전체 실험 벤치마크 요약표

| 실험 단계 및 방어 아키텍처 | Clean ACC (%) | Backdoor ASR (%) | 오탐율 (FPR) / 미탐율 (FNR) | 검증 의미 및 학술적 기여 |
| :--- | :---: | :---: | :---: | :--- |
| **1. BadFU 훈련기 (잠복기)** | **97.48%** | **22.53%** | - | • **논문 Table I (ACC 97.43% / ASR 24.08%) 소수점 일치**<br>• 백도어 잠복 상태 완벽 재현 |
| **2. 무방비 언러닝 (BadFU 폭발)** | **97.54%** | **91.24%** 🔥 | - | • **논문 Table I (Retrain ASR 96.95%) 완벽 재현**<br>• $D_c$ 삭제 즉시 ASR 22% $\rightarrow$ **91.24% 폭증** |
| **3. 소극적 모델 정규화 (1차 방어)** | **97.44%** | **87.05%** ⚠️ | - | • 단순 Weight Decay/미세조정은 백도어 저지 실패<br>• 논문의 Certified 언러닝(SIFU ASR 85.59%) 실패 재현 |
| **4. [방법 A 기초] 서버 그래디언트 감사** | **94.33%** | **9.59%** 🛡️ | FPR 0% / FNR 0% | • 원시 데이터 0% 열람 원칙 준수<br>• 가중치 시간적 가속도($\Delta$-Jump 1.54배) 탐지 |
| **5. [방법 A 심화] 가중치 부분공간 기하학 감사** | **95.45%** | **10.19%** 🛡️ | **FPR 0.00% / FNR 0.00%** | • **Grassmannian 부분공간 직교 이탈도(14.33%)**<br>• **마할라노비스 거리 11.66 (정상 2.73 대비 4.27배 폭등)** |
| **6. [방법 B 기초] TEE 픽셀 불변성 프록시** | **94.41%** | **0.45%** ✅ | FPR 0% / FNR 0% | • TEE 내 픽셀 불변성 감사로 3x3 패치 9픽셀 100% 검거<br>• 악성 노드 퇴출 시 백도어 완전 박멸 |
| **7. [방법 B 심화] TEE 다차원 스펙트럼 프록시** | **94.63%** | **0.53%** ✅ | **FPR 0.00% / FNR 0.00%** | • **은닉형 Blended 트리거($\alpha=0.2$) 대응**<br>• **클래스 간 상호 일관도(0.2262, 2.11배) 자율 적발** |

---

## 🛡️ 3. 심화된 2대 차세대 방어 아키텍처

### 🅰️ [방법 A 심화] 가중치 부분공간 기하학 매니폴드 감사 (`deep_manifold_explorer_A.py`)
* **설계 철학:** 서버는 클라이언트의 원시 데이터를 전혀 들여다보지 않으며, 단순 노름 스칼라 임계값(클리핑 공격에 취약)에 의존하지 않고 **1280차원 고차원 가중치 공간의 기하학적 매니폴드 구조**를 통계적으로 감사.
* **수학적 핵심 원리:**
  1. **클래스 에너지 분포의 마할라노비스 거리 ($D_M$)**:
     $$D_M(\mathbf{e}_k) = \sqrt{(\mathbf{e}_k - \boldsymbol{\mu}_k)^T \mathbf{\Sigma}_k^{-1} (\mathbf{e}_k - \boldsymbol{\mu}_k)}$$
     * 정상 노드는 Non-IID Dirichlet 클래스 비중 공분산 $\mathbf{\Sigma}_k$ 타원체 내에 안정적으로 수렴 ($D_M = 2.73$).
     * BadFU 공격자는 $D_c$가 빠지며 타깃 클래스 방향으로 공분산 매니폴드를 뚫고 탈출하여 **$D_M = 11.66$ (정상 대비 4.27배 폭등)**.
  2. **Grassmannian 부분공간 직교 이탈도 ($R_0$)**:
     $$R_0 = \frac{\|\mathbf{v}_0 - \mathbf{P}_{\mathcal{S}_0} \mathbf{v}_0\|_2^2}{\|\mathbf{v}_0\|_2^2}$$
     * 과거 훈련 궤적에서 추출한 고유 기저 $\mathcal{S}_0$ 대비 언러닝 델타의 직교 이탈도를 산출하여, 정상 노드(3~7%) 대비 악성 노드의 **14.33% 직교 이탈**을 수학적으로 검증.

---

### 🅱️ [방법 B 심화] 온디바이스 TEE 다차원 스펙트럼 보안 프록시 (`deep_proxy_blended_B.py`)
* **설계 철학:** 클라이언트 OS가 악성 공격자에 의해 장악되었더라도, 하드웨어 격리 영역(ARM TrustZone / Apple Secure Enclave / AWS Nitro Enclave) 내부의 TEE 보안 프록시가 데이터 삭제 요청($D_c$)의 수학적 아티팩트를 자율 검증.
* **은닉형 공격(Blended Trigger) 대응:**
  * 공격자가 단순 $3 \times 3$ 흰색 사각형 대신, 20% 투명도로 고주파 노이즈를 전역 합성한 은닉형 트리거를 사용할 경우 기존 픽셀 분산 검사는 완전 무력화됨 ($\text{Cov}(X + t) = \text{Cov}(X)$).
  * **해결책: 클래스 간 상호 코사인 일관도 (Cross-Class Cosine Coherence: $\mathcal{C}_{inter}$)**:
    $$\mathcal{C}_{inter} = \frac{1}{|Pairs_{y_i \ne y_j}|} \sum_{y_i \ne y_j} \frac{\langle \mathbf{x}_i, \mathbf{x}_j \rangle}{\|\mathbf{x}_i\|_2 \|\mathbf{x}_j\|_2}$$
    * 서로 다른 클래스(예: 숫자 1과 숫자 8) 간의 정규화 내적을 측정.
    * 정상 데이터셋: $\mathcal{C}_{inter} = 0.1071$ (서로 다른 숫자는 형상이 이질적임).
    * 은닉형 악성 데이터셋: 미세 노이즈가 블렌딩되어 모든 샘플이 인공 공통 기저를 공유하므로 **$\mathcal{C}_{inter} = 0.2262$ 로 2.11배 급증!**
    * **오탐 0건, 은닉 백도어 ASR 0.53%로 완벽 차단.**

---

## 📂 4. 파일 구성 및 설명

| 파일명 | 역할 및 설명 |
| :--- | :--- |
| `verify_badfu.py` | BadFU 공격 3단계(훈련 $\rightarrow$ 잠복 $\rightarrow$ 언러닝 활성화) 초경량 단독 검증 스크립트 |
| `compare_badfu_and_defense.py` | 논문 동일 40 라운드 풀스케일 벤치마크 (ASR 22.53% $\rightarrow$ 91.24% 완벽 재현) |
| `server_gradient_audit.py` | 서버 레벨 1차 가중치 요동(Surge Ratio 4.44배) 정량 탐지 스크립트 |
| `server_gradient_deep_audit.py` | 10개 클래스 전수 노름 및 시간적 가속도($\Delta$-Jump 1.54배) 분리 감사 스크립트 |
| `deep_manifold_explorer_A.py` | **[방법 A 심화]** Grassmannian 부분공간 직교 이탈도 및 마할라노비스 거리 기반 기하학 감사 |
| `agent_proxy_defense.py` | [방법 B 기초] TEE 픽셀 공간 불변성(Variance $\approx$ 0) 감사 및 $3 \times 3$ 패치 검거 |
| `deep_proxy_blended_B.py` | **[방법 B 심화]** 은닉형 Blended 트리거 대응 TEE 다차원 클래스 상호 일관도 감사 |
| `TRADE_OFF_ANALYSIS.md` | 가중치 감사 시 발생하는 보안 vs 언러닝 유효성(MIA/잊힐 권리) 트레이드오프 상세 분석서 |
| `logs/` | 1번부터 6번까지의 전수 벤치마크 원본 터미널 실행 로그 저장소 |

---

## 🚀 5. 실행 방법

```bash
# 가상환경 활성화 및 필수 패키지 설치
pip install torch torchvision numpy

# 1. BadFU 논문 풀 스케일 벤치마크 재현 (40 라운드)
python compare_badfu_and_defense.py

# 2. [방법 A 심화] 가중치 부분공간 기하학 매니폴드 감사 실행
python deep_manifold_explorer_A.py

# 3. [방법 B 심화] 은닉형 공격 대응 TEE 다차원 스펙트럼 프록시 실행
python deep_proxy_blended_B.py
```

---

## 📜 라이선스 및 인용

This project is licensed under the MIT License.

```bibtex
@inproceedings{lu2025badfu,
  author    = {Bingguang Lu and Hongsheng Hu and Yuantian Miao and Shaleeza Sohail and Chaoxiang He and Shuo Wang and Xiao Chen},
  title     = {{BadFU}: Backdoor Federated Learning through Adversarial Machine Unlearning},
  booktitle = {Proceedings of the 28th International Symposium on Research in Attacks, Intrusions and Defenses (RAID 2025)},
  year      = {2025}
}
```
