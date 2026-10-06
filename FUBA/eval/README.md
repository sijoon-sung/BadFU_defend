# 평가 설계 (논문 실험)

세 가지 주장을 좋은 FL 방어 논문들(FLAME, FLDetector, FLTrust, FedRecover, BackdoorIndicator, CrowdGuard, MASA, UnlearnGuard)의 실험 관례에 맞춰 측정한다.

## 공통 규칙
- 기준선 셋을 항상 함께 둔다: **공격 없음 / 방어 없음 / 공격자 완전제거 오라클**(MASA 의 FedAvg*, FedRecover 의 train-from-scratch).
- 기본 설정 하나에서 **조건을 하나씩** 바꾼다(데이터셋, 비-IID α, 악성 비율, 클라 수, 트리거 세기).
- 지표는 **클라 단위**로 정의한다(FLAME 의 TPR=TP/(TP+FP) 혼용을 피함).
- 조건마다 **seed 3~5개**, 평균±표준편차. (이 분야 다수 논문은 단일 실행이므로 이 정도면 엄격한 편.)

## 주장 1 — 탐지 확률  (`detection_eval.py`)
- 지표: 클라 TPR/FPR/FNR, precision, DACC, 타깃 클래스 정확도, 요청자 정확도, 의심점수 AUROC.
- 입력: `pipeline_detect_purify.py` 가 만든 `logs/pipeline_{name}.json`.
- **정상 대조군(필수)**: `train_benign_fl.py` 로 공격 없는 궤적 생성 → `--benign` 으로 오경보율 측정(0 이어야 좋음).
- 실행: `python eval/detection_eval.py --names s1 s2 s3 s4 s5 --benign benign1 benign2`
- 비교 탐지기(논문용 TODO): FLDetector, FLAME(군집단계), FoolsGold(weight<0.5), Multi-Krum.

## 주장 2 — 연산 비용  (`cost_eval.py`)
- 점근표 + 같은 장비 실측(ms/호출). 대안: 전체 쌍 비교 O(n²), Neural Cleanse(클래스별 최적화+서버데이터), 재학습 검사, 요청마다 재구성(FedEraser), 이력 저장 O(ndT).
- **n 을 20/50/100 으로 늘려 확장 곡선**을 그려야 이점이 보인다(n=8 에선 작음). 측정은 측정 장비(HPC)에서.
- 실행: `python eval/cost_eval.py --name s1`

## 주장 3 — 정화  (`detector_error_sweep.py`, `../baselines_compare.py`, `../weighted_sweep.py`)
- 지표: ACC, ASR, 정답(두-삭제 재학습) 모델과의 차이, 요청자 데이터 망각 확인(MIA 등 — 추가 예정).
- **핵심 실험 — 탐지 오류율 스윕**: 탐지가 완벽하면 통째 제거가 최적. 오류율 e(0~0.5)를 주입해 통째 vs V-제거의 ACC/ASR 붕괴 속도를 비교(FedRecover Fig.8 방식). V-제거가 더 완만하면 "탐지 불완전 하의 이점" 성립.
  - 실행: `python eval/detector_error_sweep.py --names s1 s2 s3 --reps 5`
- 비교군: 통째 제거, FUBA Top-k(`baselines_compare.py`), 강건 집계(공정하려면 FUBA `--median` 로 학습단계 실행), 재학습, FedEraser/UnlearnGuard.

## 그리드(논문 목표, HPC)
- 데이터셋: MNIST + CIFAR-10(+ FMNIST/CIFAR-100).
- 공격: FUBA(IBA), BadFU(BadNet/Blend), 적응형; 비언러닝 대조로 일반 BadNet/DBA.
- 비-IID α ∈ {0.1, 0.5, 1.0, IID}; 악성 비율 {5,10,20,30,40}%; 클라 n ∈ {20,50,100}; 트리거 세기 스윕.
- seed 5개.

## 현재 한계
- 지금 결과는 n=8·8라운드·소수 seed 예비치. 위 그리드는 측정 장비에서 돌려야 한다.
- 적응형 공격자, 망각 지표(MIA), CIFAR-10, 비교 탐지기 직접 실행은 아직.

## 정상 대조군이 드러낸 결함 (중요)
`train_benign_fl.py` 로 만든 공격 없는 실행에 탐지를 돌리면 **오경보율 0.625**(공격 실행에선 TPR 1·FPR 0).
현재 탐지기는 "공격이 있는가?"를 먼저 판정하지 않고 **항상 군집을 지목**하기 때문이다(비-IID 정상 클라도
어떤 클래스에서 정렬 군집을 형성). → 공격 유무를 거르는 **널 게이트**가 필요하다: 정상 모델 앙상블에서
얻은 O-질량/정렬 신호의 기준 분포를 넘을 때만 공격 선언. 이 널 분포 생성이 train_benign_fl.py 의 목적.
이 게이트 이전에는 탐지 FPR 주장을 '공격이 존재한다고 이미 아는' 조건으로 한정해야 한다.
