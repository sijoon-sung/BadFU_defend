# 해야 할 실험 (2026-10-11 밤 기준, 우선순위 순)

원칙: 실험 하나 = 주장 하나. 돌리기 전에 "어떤 표의 어떤 칸을 채우나"가 정해져 있어야 한다. 모두 `dev` 브랜치 코드.

## 1. 본 결과: 제안 기법 (FUBA, 3 시드 × 2 조건)  — DISLAB, 한 줄
```
cd FUBA && STAGES="dose pair masa subid aggregate" CONDS="owner iid" bash run_all.sh
```
| 채우는 표 | 주장 | 보는 숫자 |
|---|---|---|
| dose: arm × 요청자 (plain/oneshot/dose/always) | P1 정상 요청은 기본 FU 와 동일, P3 위장자 요청의 행동 보존, P4 멀쩡한 모델은 한 단계 | 정상 요청자: dose 의 ASR/ACC/retain == plain, 경보 0. 위장자: ASR 50~60 → ≤10, 단계 수 1 |
| dose: always 팔 | 문(gate)의 필요성: 항상 수정하면 정상 요청자의 망각이 되돌아간다 | 정상 요청자 forget acc 가 삭제 전으로 복귀 |
| pair: A, C, 정밀·재현, +운반자 ASR | 쌍→집합이 운반자를 찾고 백도어를 '지운다' | 정밀 1.0, 재현 ≥ 0.75, ASR ≤ 10 |
| masa / subid 요약 | "왜 학습 때 안 잡히나" (MASA 는 위장자 못 잡음), "SVD 와 뭐가 다른가" (SubID 식도 위장자를 잡음 — 불리하면 그대로 적는다) | AUROC 위장자 vs 정상 |
결과 올리기: `git add FUBA/logs/dose_fu FUBA/logs/pair_roles FUBA/logs/baseline_masa FUBA/logs/baseline_subid FUBA/logs/baselines FUBA/logs/pipeline && git commit && git push`

## 2. 적응형 공격 (FUBA, 3 시드 × 2 조건) — DISLAB, 재학습 없음
```
cd FUBA && STAGES="adaptive aggregate" CONDS="owner iid" bash run_all.sh
```
| 시나리오 | 지금 예상 (1 시드 검사) | 고친 뒤 기대 |
|---|---|---|
| a 쪼개기 10 조각 | 버팀 | — |
| b 위장자 분산 (순차) | 클라별 검사 뚫림, 전체 예산 부분 방어, **정상 열 오탐** | "n 명 제거" 띠로 오탐 제거 후 재실행 |
| c 미끼 ×2.5 | 뚫림 (z 11 → 2) | 절사 std 후 재실행 |
| g 기준 오염 0.3 | 버팀 | β 0.5, 1.0 스윕 (ADAPT_EXTRA 가 아니라 run_all 의 dir 계수 수정) |
| i 운반자 분산 | 탐욕 선택 뚫림 | 집단 제거 검사 후 재실행 |
**순서**: 먼저 지금 코드로 한 번 돌려 '뚫린다'를 표로 남기고, 설계를 고친 뒤 같은 표를 다시 채운다. 둘 다 논문에 들어간다(공격 → 보완).

## 3. 망각을 숫자로 — 설계 결정 필요
MNIST IID 에서는 MIA AUROC ≈ 0.5 라 정상 요청자가 잊혔는지 못 보인다. 선택지:
- (a) `owner` 조건의 소유 클래스 정확도: 클라 7 이 클래스 3 의 90 % 를 들고 있으므로, 7 을 빼면 클래스 3 정확도가 떨어져야 '잊힘'. 코드 거의 있음(`retain_change` 옆에 class-3 acc 추가).
- (b) CIFAR 로 옮김 (MIA 가 동작) — 비용 큼, BadFU 쪽 궤적 재사용 가능.
권장 (a). 1 번 표에 열 하나 추가.

## 4. BadFU (CIFAR ResNet) — 로컬 GPU, `FUBA\resume_local.bat` 의 [1][2]
- 증류 보정 v2 + 격자 패치 탐색으로 빼 보기. 지금까지 전부 실패. 성공하면 "탐침 확장"이 답, 실패하면 논문에 **범위 밖**으로 명시.
- dose_fu 를 `--op kd --delta_mode grid` 로 (코드 있음). 요청자 5 / 정상 1.

## 5. 위장자 2 명 궤적 (FUBA) — DISLAB
```
NB_CLIENTS=8 ROUNDS=8 WARMUP=3 EXTRA_ARGS="--clients_ratial 2 --split_seed 522" bash run_real_fuba.sh bd_two_s0 522
python pair_roles.py --name bd_two_s0 --K 16 --defenders 4 8 --attackers 0 1 2 3 5 6 7 9 --requesters 4 8 --joint
python schedule_fu.py --name bd_two_s0 --K 16 --defender 4 --attackers 0 1 2 3 5 6 7 9 --requests 4 8 --control 10 11
```
옵션(`--clients_ratial`)은 코드만 넣었고 생성은 미확인. 첫 실행에서 깨지면 config.py 의 역할 배정을 봐야 한다.

## 6. 비용 (있음)
`logs/bench` 에 K=8~100 측정 있음. dose 단계의 '단계 수'와 '가상 모델 수'(pop_models) 열이 1 번 표에서 같이 나오므로 추가 실험 없음.

## 지금 당장 돌릴 것 하나
1 번. 그 결과가 있어야 3 번(망각 열)과 2 번(적응형)의 비교 기준이 생긴다.
