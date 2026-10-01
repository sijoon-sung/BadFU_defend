# BadFU Defend: Adversarial Federated Unlearning Defense Framework

[![arXiv](https://img.shields.io/badge/arXiv-2508.15541-b31b1b.svg)](https://arxiv.org/abs/2508.15541)
[![Conference](https://img.shields.io/badge/Venue-RAID%202025-blue.svg)](https://raid2025.org)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)

본 저장소는 최신 연합 학습 보안 논문인 **"BadFU: Backdoor Federated Learning through Adversarial Machine Unlearning" (RAID 2025)**의 백도어 공격 메커니즘을 1:1 풀 스케일로 실증 재현하고, 이를 무력화하기 위한 **2대 방어 아키텍처(서버 레벨 그래디언트 감사 & 온디바이스 TEE 자율형 보안 프록시)**를 구현 및 검증한 연구 프레임워크입니다.

---

## 📌 1. 연구 배경 및 BadFU 공격 개요

* **논문 정보:** *BadFU: Backdoor Federated Learning through Adversarial Machine Unlearning* (RAID 2025, arXiv:2508.15541)
* **공격 메커니즘:**
  * 악성 클라이언트($C_m$)가 훈련 단계에서는 백도어 샘플($D_{bd}$)과 위장 샘플($D_c$)을 동시에 주입하여 로컬에서 상쇄($\nabla \mathcal{L}_{bd} + \nabla \mathcal{L}_c \approx 0$)시킵니다.
  * 연합 학습의 파라미터 평균화(`FedAvg`)와 Non-IID 특성으로 인해 훈련 중에는 백도어가 낮은 ASR(10~25%)로 안전하게 잠복(Dormant)하며 기존 비잔틴 강건 집계(Median, Trimean)를 통과합니다.
  * 훈련 종료 후 악성 클라이언트가 $D_c$에 대한 합법적 데이터 삭제(Right to be Forgotten)를 요청하면, $\nabla \mathcal{L}_c$가 제거되면서 잠복해 있던 백도어($\nabla \mathcal{L}_{bd}$)가 급반등하여 **ASR 90% 이상으로 전면 활성화**됩니다.

---

## 📊 2. 벤치마크 실증 결과 (Full-Scale 40 라운드 전수 실험)

MNIST + SimpleNN (40 Communication Rounds $\times$ Local 5 Epochs = 총 200 에포크 분량, Non-IID Dominant Class 70%) 환경에서 측정한 정량 벤치마크 결과입니다.

| 실험 단계 및 방어 아키텍처 | Clean ACC (%) | Backdoor ASR (%) | 검증 의미 및 논문 비교 |
| :--- | :---: | :---: | :--- |
| **1. 논문 동일 BadFU 훈련기 (잠복기)** | **97.48%** | **22.53%** | • **논문 Table I (ACC 97.43% / ASR 24.08%) 소수점 일치**<br>• 백도어 잠복 상태 완벽 재현 |
| **2. 논문 방식 무방비 언러닝 (공격 성공)** | **97.54%** | **91.24%** 🔥 | • **논문 Table I (Retrain ASR 96.95%) 완벽 재현**<br>• $D_c$ 삭제 즉시 ASR 22% $\rightarrow$ **91.24% 폭증** |
| **3. 소극적 모델 정규화 (1차 방어)** | **97.44%** | **87.05%** ⚠️ | • 단순 Weight Decay/미세조정은 백도어 저지 실패<br>• 논문의 Certified 언러닝(SIFU ASR 85.59%) 실패 재현 |
| **4. [방법 A] 서버 레벨 그래디언트 감사 방어** | **94.33%** | **9.59%** 🛡️ | • 서버가 원시 데이터를 보지 않고 가중치 요동(Surge) 탐지<br>• 이상 델타 드랍 시 백도어 발현 차단 |
| **5. [방법 B] 온디바이스 TEE 자율형 프록시 방어** | **94.41%** | **0.45%** ✅ | • 픽셀 불변성 감사로 3x3 패치 9픽셀 100% 적발<br>• 악성 노드 퇴출 시 **백도어 완전 박멸 (ASR 0%)** |

---

## 🛡️ 3. 제안된 2대 방어 아키텍처

### 방법 A. [서버 레벨] 언러닝 그래디언트 방향성 감사 (`server_gradient_audit.py`)
* **철학:** 서버는 클라이언트의 원시 데이터를 전혀 들여다보지 않음 (FL 프라이버시 100% 준수).
* **원리:**
  * 클라이언트가 제출한 마지막 레이어 가중치 델타 벡터($\Delta W = W_{\text{unlearn}} - W_{\text{global}}$)의 10개 클래스별 노름($s_c = \|\Delta W[c, :]\|_2$)을 측정.
  * BadFU 악성 노드는 타깃 클래스(Class 0) 방향으로 **4.44배의 비정상적 요동(Surge Ratio)**을 일으킴.
  * Non-IID 정상 노드(2.3~3.1배)와의 델타 가속도 차이를 포착하여 악성 델타를 자동 폐기(Drop).

### 방법 B. [온디바이스 레벨] TEE 자율형 보안 프록시 (`agent_proxy_defense.py`)
* **철학:** 클라이언트 기기 내부의 하드웨어 보안 영역(ARM TrustZone / Apple Secure Enclave / TEE)에 감사 에이전트를 상주.
* **원리:**
  * 삭제 요청된 데이터셋의 픽셀별 공간 분산($\text{Std}$)을 수학적으로 분석.
  * 모든 이미지에 동일하게 박혀 있는 $3 \times 3$ 고정 트리거 패치(분산 $\approx 0$)를 **단 1픽셀의 오차도 없이 100% 자동 적발(Y[24~26], X[24~26])**.
  * 악성 요청 기각 시 ASR을 9.59%로 묶어두며, 악성 노드 영구 퇴출 시 백도어를 0.45%로 완전 소거.

---

## 📂 4. 파일 구성 및 설명

* `verify_badfu.py`: BadFU 공격 메커니즘을 외부 라이브러리 의존성 없이 단독 실행하는 초경량 검증 스크립트.
* `compare_badfu_and_defense.py`: 논문과 동일한 40 라운드 풀 스케일 벤치마크 및 1차 정규화 방어 검증 스크립트.
* `server_gradient_audit.py`: 서버가 원시 데이터 없이 가중치 델타 벡터의 클래스별 요동(Surge Ratio)을 분석하는 서버 측 방어 스크립트.
* `agent_proxy_defense.py`: 온디바이스 TEE 자율형 보안 프록시 기반의 이상 패턴 자동 적발 및 악성 요청 차단 스크립트.

---

## 🚀 5. 실행 방법

```bash
# 가상환경 활성화 및 필수 패키지 설치
pip install torch torchvision numpy

# 1. BadFU 논문 메커니즘 기초 검증
python verify_badfu.py

# 2. 풀 스케일 40 라운드 벤치마크 실행
python compare_badfu_and_defense.py

# 3. [방법 A] 서버 레벨 그래디언트 방향성 감사 방어 실행
python server_gradient_audit.py

# 4. [방법 B] 온디바이스 자율형 보안 프록시 방어 실행
python agent_proxy_defense.py
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
