# 실험 기록 — 공격 궤적 (자동 생성)

`FUBA/verify_trajectories.py` 가 `FUBA/trajectories.json` 과 저장 파일을 읽어 만든다. 갱신 2026-10-13 20:22 (DESKTOP-QUB35TD). 손으로 고치지 말 것.

- 숫자는 `ACC / ASR` (%). 잠복 = 삭제 전 최종 모델 θ_T. 제거 = 그 클라이언트의 저장 기여를 1 차 근사로 뺀 모델 (θ_T − U_k, `release_map.load_fuba` 와 같은 U). 채점은 MNIST/CIFAR test[500:] 과 저장된 트리거.
- BadFU 는 빼기가 모델을 무너뜨리므로 `붕괴` 로 적고, 괄호의 `단계식` 값(fl_detect_badfu.py 의 FedEraser 언러닝, 저자 코드) 으로 판정한다.
- 상태 ✔ = 매니페스트 `verify` 의 모든 문턱 통과. ✘ = 하나라도 실패(사유 표시). △ = 파일·집계는 맞지만 공격이 약함 (attack_required=false 인 점검용 그룹만; 실험 근거로 쓰지 말 것). － = 아직 안 만듦. `(임시 문턱)` = 실측 전 값이라 문턱을 다시 볼 것.

## 한눈에

전체 57 개: ✔ 21, △ 6, ✘ 3, 미생성 27

| 그룹 | 설명 | 개수 | ✔ / △ / ✘ / 미생성 | 예상 시간 (개당 → 전체) | 어디서 |
|---|---|---|---|---|---|
| k8_main | 주 설정(축소판): K=8, 운반자 0-3, 위장자 4, 정상 5-7, 8 라운드(웜업 3), 전원 참여. iid / owner × 시드 0,1,2. 이름이 기존 DISLAB 궤적과 같다 | 6 | 6 / 0 / 0 / 0 | 6 분 → 남은 것 0.0 시간 | DISLAB |
| k20_paper | 논문 주 설정: n=20, 악성 5 (운반자 4 + 위장자 1 = 25 %), 참여율 0.4, 100 라운드(웜업 5). iid / owner × 시드 0,1,2 | 6 | 6 / 0 / 0 / 0 | 20 분 → 남은 것 0.0 시간 | DISLAB |
| k20_reduced | K=20 축소판: 8 라운드(웜업 3), 전원 참여. 파이프라인 점검용 — 로컬 실측에서 공격이 안 심어졌다(위장자 빼도 ASR 11.5). iid / owner × 시드 0,1,2 | 6 | 0 / 6 / 0 / 0 | 6 분 → 남은 것 0.0 시간 | DISLAB |
| sup2 | 위장자 2 명 (--clients_ratial 2 → K=16): 운반자 {0,1,2,3,5,6,7,9}, 위장자 {4,8}, 정상 10-15 (악성 10/16). iid × 시드 0,1,2 | 3 | 3 / 0 / 0 / 0 | 10 분 → 남은 것 0.0 시간 | DISLAB |
| tdef_paper | 집계 방어 4 종(median, multi_krum, flame, flDetector) 위의 같은 공격, 논문 설정(K=20, 참여 0.4, 100 라운드). iid × 시드 0,1,2 | 12 | 0 / 0 / 1 / 11 | 50 분 → 남은 것 9.2 시간 | DISLAB |
| tdef_reduced | 집계 방어 4 종, K=8 축소판 (싸지만 악성 다수라 참고용). run_paper.sh RQ2b 이름 그대로 | 12 | 6 / 0 / 1 / 5 | 7 분 → 남은 것 0.6 시간 | DISLAB |
| badfu | BadFU (CIFAR-10, ResNet-18, K=6, 40 라운드 × 로컬 5 에폭): 시드 42, 43, 44 (각자 logs/traj_s{seed}) + 기존 15 라운드 traj_full | 4 | 0 / 0 / 0 / 4 | 150 분 → 남은 것 8.4 시간 | local |
| local_smoke | 노트북 점검용 궤적 (bd_iid_s0 와 같은 설정, 이름만 다름). DISLAB 실험에는 안 쓴다 | 1 | 0 / 0 / 0 / 1 | 6 분 → 남은 것 0.1 시간 | local |
| k20_paper_cifar | 큰 모델: CIFAR-10/resnet18, 논문 설정 flag, IID 3 시드 (owner 는 IID 결과 뒤에) | 3 | 0 / 0 / 1 / 2 | 300 분 → 남은 것 10.0 시간 | DISLAB |
| clean_paper | 공격 없는 궤적 (오경보 검사 R-015): mnist/cifar × iid/owner, warm-up 1000 | 4 | 0 / 0 / 0 / 4 | 45 분 → 남은 것 11.5 시간 | DISLAB |

## 궤적별 결과

| 이름 | 공격 | 조건 | K | 시드 | 역할 (운반자 / 위장자 / 정상) | 잠복 ACC/ASR | 위장자 제거 후 ACC/ASR | 정상 제거 후 ACC/ASR | 라운드 | 어디서 | 상태 | 날짜 | 비고 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bd_iid_s0 | fuba | iid | 8 | 522 | 0-3 / 4 / 5-7 | 94.6 / 8.6 | 75.8 / 62.1 | 93.0 / 20.3 (5) | 8 (웜업 3, 참여 1.0) | DISLAB | ✔ | 2026-10-12 19:04 | 운반자 0 제거: 94.6 / 4.9 |
| bd_iid_s1 | fuba | iid | 8 | 523 | 0-3 / 4 / 5-7 | 95.2 / 5.2 | 81.7 / 42.9 | 94.2 / 12.5 (5) | 8 (웜업 3, 참여 1.0) | DISLAB | ✔ | 2026-10-12 19:04 | 운반자 0 제거: 95.0 / 3.4 |
| bd_iid_s2 | fuba | iid | 8 | 524 | 0-3 / 4 / 5-7 | 95.3 / 4.6 | 80.8 / 38.6 | 94.5 / 8.8 (5) | 8 (웜업 3, 참여 1.0) | DISLAB | ✔ | 2026-10-12 19:04 | 운반자 0 제거: 95.3 / 1.9 |
| bd_owner_s0 | fuba | owner | 8 | 522 | 0-3 / 4 / 5-7 (소유자 7) | 93.0 / 9.3 | 66.9 / 62.8 | 89.6 / 25.2 (5) | 8 (웜업 3, 참여 1.0) | DISLAB | ✔ | 2026-10-12 19:04 | 운반자 0 제거: 92.9 / 8.1 |
| bd_owner_s1 | fuba | owner | 8 | 523 | 0-3 / 4 / 5-7 (소유자 7) | 93.2 / 6.6 | 73.4 / 50.1 | 92.4 / 12.0 (5) | 8 (웜업 3, 참여 1.0) | DISLAB | ✔ | 2026-10-12 19:04 | 운반자 0 제거: 93.4 / 4.5 |
| bd_owner_s2 | fuba | owner | 8 | 524 | 0-3 / 4 / 5-7 (소유자 7) | 91.3 / 8.2 | 67.9 / 49.7 | 91.1 / 9.3 (5) | 8 (웜업 3, 참여 1.0) | DISLAB | ✔ | 2026-10-12 19:04 | 운반자 0 제거: 91.8 / 5.7 |
| bd_iid_r100w5_k20paper_s0 | fuba | iid | 20 | 522 | 0-3 / 4 / 5-19 | 98.8 / 7.4 | 68.7 / 92.6 | 97.8 / 3.9 (5) | 100 (웜업 5, 참여 0.4) | DISLAB | ✔ | 2026-10-13 00:54 | ρ=1.0 빼기 붕괴로 작동점 ρ=0.2 에서 판정; 운반자 0 제거: 98.3 / 0.6 |
| bd_iid_r100w5_k20paper_s1 | fuba | iid | 20 | 523 | 0-3 / 4 / 5-19 | 98.7 / 3.1 | 97.6 / 46.2 | 96.5 / 2.4 (5) | 100 (웜업 5, 참여 0.4) | DISLAB | ✔ | 2026-10-13 00:54 | ρ=1.0 빼기 붕괴로 작동점 ρ=0.2 에서 판정; 운반자 0 제거: 98.1 / 0.6 |
| bd_iid_r100w5_k20paper_s2 | fuba | iid | 20 | 524 | 0-3 / 4 / 5-19 | 98.6 / 4.8 | 86.6 / 82.3 | 94.8 / 4.9 (5) | 100 (웜업 5, 참여 0.4) | DISLAB | ✔ | 2026-10-13 00:54 | ρ=1.0 빼기 붕괴로 작동점 ρ=0.2 에서 판정; 운반자 0 제거: 97.3 / 0.6 |
| bd_owner_r100w5_k20paper_s0 | fuba | owner | 20 | 522 | 0-3 / 4 / 5-19 (소유자 19) | 98.2 / 7.8 | 73.1 / 85.7 | 96.1 / 9.8 (5) | 100 (웜업 5, 참여 0.4) | DISLAB | ✔ | 2026-10-13 00:54 | ρ=1.0 빼기 붕괴로 작동점 ρ=0.2 에서 판정; 운반자 0 제거: 98.0 / 1.4 |
| bd_owner_r100w5_k20paper_s1 | fuba | owner | 20 | 523 | 0-3 / 4 / 5-19 (소유자 19) | 98.3 / 3.6 | 96.7 / 50.0 | 95.8 / 5.8 (5) | 100 (웜업 5, 참여 0.4) | DISLAB | ✔ | 2026-10-13 00:54 | ρ=1.0 빼기 붕괴로 작동점 ρ=0.2 에서 판정; 운반자 0 제거: 96.5 / 0.2 |
| bd_owner_r100w5_k20paper_s2 | fuba | owner | 20 | 524 | 0-3 / 4 / 5-19 (소유자 19) | 98.1 / 5.7 | 82.5 / 78.9 | 96.0 / 4.4 (5) | 100 (웜업 5, 참여 0.4) | DISLAB | ✔ | 2026-10-13 00:54 | ρ=1.0 빼기 붕괴로 작동점 ρ=0.2 에서 판정; 운반자 0 제거: 94.5 / 0.2 |
| bd_iid_k20_s0 | fuba | iid | 20 | 522 | 0-3 / 4 / 5-19 | 92.9 / 4.9 | 90.3 / 9.2 | 92.8 / 5.3 (5) | 8 (웜업 3, 참여 1.0) | DISLAB | △ 공격 약함 (점검용으로만): 위장자 제거 후 ASR 9.2 < 20 (공격이 재현되지 않음) | 2026-10-12 21:26 | 운반자 0 제거: 92.5 / 3.6 |
| bd_iid_k20_s1 | fuba | iid | 20 | 523 | 0-3 / 4 / 5-19 | 92.7 / 4.2 | 90.8 / 8.8 | 92.5 / 4.9 (5) | 8 (웜업 3, 참여 1.0) | DISLAB | △ 공격 약함 (점검용으로만): 위장자 제거 후 ASR 8.8 < 20 (공격이 재현되지 않음) | 2026-10-12 21:29 | 운반자 0 제거: 92.6 / 3.1 |
| bd_iid_k20_s2 | fuba | iid | 20 | 524 | 0-3 / 4 / 5-19 | 93.2 / 3.7 | 90.7 / 7.0 | 93.0 / 3.6 (5) | 8 (웜업 3, 참여 1.0) | DISLAB | △ 공격 약함 (점검용으로만): 위장자 제거 후 ASR 7.0 < 20 (공격이 재현되지 않음) | 2026-10-12 21:31 | 운반자 0 제거: 92.8 / 2.7 |
| bd_owner_k20_s0 | fuba | owner | 20 | 522 | 0-3 / 4 / 5-19 (소유자 19) | 88.3 / 7.2 | 84.1 / 13.3 | 87.0 / 10.5 (5) | 8 (웜업 3, 참여 1.0) | DISLAB | △ 공격 약함 (점검용으로만): 위장자 제거 후 ASR 13.3 < 20 (공격이 재현되지 않음) | 2026-10-12 21:36 | 운반자 0 제거: 87.6 / 6.3 |
| bd_owner_k20_s1 | fuba | owner | 20 | 523 | 0-3 / 4 / 5-19 (소유자 19) | 87.3 / 8.6 | 83.6 / 14.0 | 86.6 / 9.7 (5) | 8 (웜업 3, 참여 1.0) | DISLAB | △ 공격 약함 (점검용으로만): 위장자 제거 후 ASR 14.0 < 20 (공격이 재현되지 않음) | 2026-10-12 21:40 | 운반자 0 제거: 86.7 / 7.6 |
| bd_owner_k20_s2 | fuba | owner | 20 | 524 | 0-3 / 4 / 5-19 (소유자 19) | 87.3 / 7.9 | 82.2 / 14.1 | 87.1 / 8.6 (5) | 8 (웜업 3, 참여 1.0) | DISLAB | △ 공격 약함 (점검용으로만): 위장자 제거 후 ASR 14.1 < 20 (공격이 재현되지 않음) | 2026-10-12 21:43 | 운반자 0 제거: 86.7 / 6.9 |
| bd_iid_k16sup2_s0 | fuba | iid | 16 | 522 | 0-3,5-7,9 / 4,8 / 10-15 | 94.6 / 9.1 | 72.3 / 68.8 | 93.9 / 13.4 (10) | 8 (웜업 3, 참여 1.0) | DISLAB | ✔ (임시 문턱) | 2026-10-12 21:49 | 임시 문턱 (provisional) — 실측값을 보고 매니페스트 verify 를 고칠 것; 위장자 각자 제거: 4: 90.2 / 26.5, 8: 89.2 / 31.4; 운반자 0 제거: 94.7 / 7.0 |
| bd_iid_k16sup2_s1 | fuba | iid | 16 | 523 | 0-3,5-7,9 / 4,8 / 10-15 | 95.1 / 5.8 | 78.6 / 50.4 | 95.0 / 7.5 (10) | 8 (웜업 3, 참여 1.0) | DISLAB | ✔ (임시 문턱) | 2026-10-12 21:54 | 임시 문턱 (provisional) — 실측값을 보고 매니페스트 verify 를 고칠 것; 위장자 각자 제거: 4: 91.8 / 19.0, 8: 92.0 / 16.3; 운반자 0 제거: 95.2 / 4.0 |
| bd_iid_k16sup2_s2 | fuba | iid | 16 | 524 | 0-3,5-7,9 / 4,8 / 10-15 | 95.1 / 4.8 | 76.2 / 46.6 | 94.7 / 6.4 (10) | 8 (웜업 3, 참여 1.0) | DISLAB | ✔ (임시 문턱) | 2026-10-12 21:59 | 임시 문턱 (provisional) — 실측값을 보고 매니페스트 verify 를 고칠 것; 위장자 각자 제거: 4: 91.8 / 14.8, 8: 91.5 / 15.6; 운반자 0 제거: 95.0 / 3.0 |
| bd_iid_r100w5_k20paper_median_s0 | fuba | iid | 20 | 522 | 0-3 / 4 / 5-19 | - | - | - | 100 (웜업 5, 참여 0.4) | DISLAB | ✘ 파일 243/295 개 없음 (예: global_net_model_bd_iid_r100w5_k20paper_median_s0_round_20.pkl) — 생성이 중간에 끊겼거나 라운드/웜업이 매니페스트와 다름 | 2026-10-13 17:30 | 집계 방어 median |
| bd_iid_r100w5_k20paper_median_s1 | fuba | iid | 20 | 523 | 0-3 / 4 / 5-19 | - | - | - | 100 (웜업 5, 참여 0.4) | DISLAB | － 미생성 | 2026-10-13 17:30 | 집계 방어 median |
| bd_iid_r100w5_k20paper_median_s2 | fuba | iid | 20 | 524 | 0-3 / 4 / 5-19 | - | - | - | 100 (웜업 5, 참여 0.4) | DISLAB | － 미생성 | 2026-10-13 17:30 | 집계 방어 median |
| bd_iid_r100w5_k20paper_multi_krum_s0 | fuba | iid | 20 | 522 | 0-3 / 4 / 5-19 | - | - | - | 100 (웜업 5, 참여 0.4) | DISLAB | － 미생성 | 2026-10-13 17:30 | 집계 방어 multi_krum |
| bd_iid_r100w5_k20paper_multi_krum_s1 | fuba | iid | 20 | 523 | 0-3 / 4 / 5-19 | - | - | - | 100 (웜업 5, 참여 0.4) | DISLAB | － 미생성 | 2026-10-13 17:30 | 집계 방어 multi_krum |
| bd_iid_r100w5_k20paper_multi_krum_s2 | fuba | iid | 20 | 524 | 0-3 / 4 / 5-19 | - | - | - | 100 (웜업 5, 참여 0.4) | DISLAB | － 미생성 | 2026-10-13 17:30 | 집계 방어 multi_krum |
| bd_iid_r100w5_k20paper_flame_s0 | fuba | iid | 20 | 522 | 0-3 / 4 / 5-19 | - | - | - | 100 (웜업 5, 참여 0.4) | DISLAB | － 미생성 | 2026-10-13 17:30 | 집계 방어 flame |
| bd_iid_r100w5_k20paper_flame_s1 | fuba | iid | 20 | 523 | 0-3 / 4 / 5-19 | - | - | - | 100 (웜업 5, 참여 0.4) | DISLAB | － 미생성 | 2026-10-13 17:30 | 집계 방어 flame |
| bd_iid_r100w5_k20paper_flame_s2 | fuba | iid | 20 | 524 | 0-3 / 4 / 5-19 | - | - | - | 100 (웜업 5, 참여 0.4) | DISLAB | － 미생성 | 2026-10-13 17:30 | 집계 방어 flame |
| bd_iid_r100w5_k20paper_flDetector_s0 | fuba | iid | 20 | 522 | 0-3 / 4 / 5-19 | - | - | - | 100 (웜업 5, 참여 0.4) | DISLAB | － 미생성 | 2026-10-13 17:30 | 집계 방어 flDetector |
| bd_iid_r100w5_k20paper_flDetector_s1 | fuba | iid | 20 | 523 | 0-3 / 4 / 5-19 | - | - | - | 100 (웜업 5, 참여 0.4) | DISLAB | － 미생성 | 2026-10-13 17:30 | 집계 방어 flDetector |
| bd_iid_r100w5_k20paper_flDetector_s2 | fuba | iid | 20 | 524 | 0-3 / 4 / 5-19 | - | - | - | 100 (웜업 5, 참여 0.4) | DISLAB | － 미생성 | 2026-10-13 17:30 | 집계 방어 flDetector |
| bd_iid_median_s0 | fuba | iid | 8 | 522 | 0-3 / 4 / 5-7 | 91.7 / 14.5 | 76.0 / 50.0 | 86.2 / 35.5 (5) | 8 (웜업 3, 참여 1.0) | DISLAB | ✔ | 2026-10-12 22:03 | 집계 방어 median; 운반자 0 제거: 90.9 / 10.6 |
| bd_iid_median_s1 | fuba | iid | 8 | 523 | 0-3 / 4 / 5-7 | 92.3 / 10.3 | 54.8 / 74.1 | 88.2 / 28.6 (5) | 8 (웜업 3, 참여 1.0) | DISLAB | ✔ | 2026-10-12 22:06 | 집계 방어 median; 운반자 0 제거: 91.9 / 7.2 |
| bd_iid_median_s2 | fuba | iid | 8 | 524 | 0-3 / 4 / 5-7 | 92.2 / 9.9 | 82.6 / 30.5 | 88.6 / 20.9 (5) | 8 (웜업 3, 참여 1.0) | DISLAB | ✔ | 2026-10-12 22:09 | 집계 방어 median; ρ=1.0 빼기 붕괴로 작동점 ρ=0.5 에서 판정; 운반자 0 제거: 92.4 / 5.0 |
| bd_iid_multi_krum_s0 | fuba | iid | 8 | 522 | 0-3 / 4 / 5-7 | 90.9 / 10.0 | 76.0 / 35.6 | 88.2 / 20.6 (5) | 8 (웜업 3, 참여 1.0) | DISLAB | ✔ | 2026-10-12 22:12 | 집계 방어 multi_krum; 운반자 0 제거: 89.1 / 5.3 |
| bd_iid_multi_krum_s1 | fuba | iid | 8 | 523 | 0-3 / 4 / 5-7 | 90.5 / 8.2 | 75.1 / 37.2 | 89.0 / 15.3 (5) | 8 (웜업 3, 참여 1.0) | DISLAB | ✔ | 2026-10-12 22:14 | 집계 방어 multi_krum; 운반자 0 제거: 89.7 / 4.7 |
| bd_iid_multi_krum_s2 | fuba | iid | 8 | 524 | 0-3 / 4 / 5-7 | 90.8 / 13.2 | 54.8 / 68.6 | 87.6 / 24.0 (5) | 8 (웜업 3, 참여 1.0) | DISLAB | ✔ | 2026-10-12 22:17 | 집계 방어 multi_krum; 운반자 0 제거: 90.7 / 5.8 |
| bd_iid_flame_s0 | fuba | iid | 8 | 522 | 0-3 / 4 / 5-7 | 89.5 / 19.1 | 79.9 / 47.4 | 81.2 / 50.6 (5) | 8 (웜업 3, 참여 1.0) | DISLAB | ✘ 잠복 ASR 19.1 > 15 (잠복이 아님); 잠복 ACC 89.5 < 90; 정상 5 제거 후 ASR 50.6 > 40 (정상 삭제로 깨어남) | 2026-10-12 22:20 | 집계 방어 flame; ρ=1.0 빼기 붕괴로 작동점 ρ=0.5 에서 판정; 운반자 0 제거: 88.5 / 15.9 |
| bd_iid_flame_s1 | fuba | iid | 8 | 523 | 0-3 / 4 / 5-7 | - | - | - | 8 (웜업 3, 참여 1.0) | DISLAB | － 미생성 | 2026-10-13 17:30 | 집계 방어 flame |
| bd_iid_flame_s2 | fuba | iid | 8 | 524 | 0-3 / 4 / 5-7 | - | - | - | 8 (웜업 3, 참여 1.0) | DISLAB | － 미생성 | 2026-10-13 17:30 | 집계 방어 flame |
| bd_iid_flDetector_s0 | fuba | iid | 8 | 522 | 0-3 / 4 / 5-7 | - | - | - | 8 (웜업 3, 참여 1.0) | DISLAB | － 미생성 | 2026-10-13 17:30 | 집계 방어 flDetector |
| bd_iid_flDetector_s1 | fuba | iid | 8 | 523 | 0-3 / 4 / 5-7 | - | - | - | 8 (웜업 3, 참여 1.0) | DISLAB | － 미생성 | 2026-10-13 17:30 | 집계 방어 flDetector |
| bd_iid_flDetector_s2 | fuba | iid | 8 | 524 | 0-3 / 4 / 5-7 | - | - | - | 8 (웜업 3, 참여 1.0) | DISLAB | － 미생성 | 2026-10-13 17:30 | 집계 방어 flDetector |
| badfu_s42 | badfu | dominant70 | 6 | 42 | 0 / 5 / 1-4 | - | - | - | 40 × 5 에폭 | local | － 미생성 | 2026-10-13 17:30 |  |
| badfu_s43 | badfu | dominant70 | 6 | 43 | 0 / 5 / 1-4 | - | - | - | 40 × 5 에폭 | local | － 미생성 | 2026-10-13 17:30 |  |
| badfu_s44 | badfu | dominant70 | 6 | 44 | 0 / 5 / 1-4 | - | - | - | 40 × 5 에폭 | local | － 미생성 | 2026-10-13 17:30 |  |
| badfu_r15_s42 | badfu | dominant70 | 6 | 42 | 0 / 5 / 1-4 | - | - | - | 15 × 5 에폭 | local | － 미생성 | 2026-10-13 17:30 |  |
| loc_fuba | fuba | iid | 8 | 522 | 0-3 / 4 / 5-7 | - | - | - | 8 (웜업 3, 참여 1.0) | local | － 미생성 | 2026-10-13 17:30 |  |
| bd_iid_r100w5_k20paper_cifar_s0 | fuba | iid | 20 | 522 | 0-3 / 4 / 5-19 | 78.6 / 97.1 | 9.9 / 0.0 | 58.2 / 44.9 (5) | 100 (웜업 5, 참여 0.4) | DISLAB | ✘ 잠복 ASR 97.1 > 20 (잠복이 아님); 위장자 제거(빼기) 붕괴: ACC 9.9 < 25 (ASR 0.0 는 판정 불가); 정상 5 제거 후 ASR 44.9 > 20 (정상 삭제로 깨어남) | 2026-10-13 20:18 | 임시 문턱 (provisional) — 실측값을 보고 매니페스트 verify 를 고칠 것; 운반자 0 제거: 77.9 / 96.5 |
| bd_iid_r100w5_k20paper_cifar_s1 | fuba | iid | 20 | 523 | 0-3 / 4 / 5-19 | - | - | - | 100 (웜업 5, 참여 0.4) | DISLAB | － 미생성 | 2026-10-13 17:30 |  |
| bd_iid_r100w5_k20paper_cifar_s2 | fuba | iid | 20 | 524 | 0-3 / 4 / 5-19 | - | - | - | 100 (웜업 5, 참여 0.4) | DISLAB | － 미생성 | 2026-10-13 17:30 |  |
| clean_iid_r100_k20paper_s0 | fuba | iid | 20 | 522 | - / - / 0-19 | - | - | - | 100 (웜업 1000, 참여 0.4) | DISLAB | － 미생성 | 2026-10-13 17:30 |  |
| clean_iid_r100_k20paper_cifar_s0 | fuba | iid | 20 | 522 | - / - / 0-19 | - | - | - | 100 (웜업 1000, 참여 0.4) | DISLAB | － 미생성 | 2026-10-13 17:30 |  |
| clean_owner_r100_k20paper_s0 | fuba | owner | 20 | 522 | - / - / 0-19 (소유자 19) | - | - | - | 100 (웜업 1000, 참여 0.4) | DISLAB | － 미생성 | 2026-10-13 17:30 |  |
| clean_owner_r100_k20paper_cifar_s0 | fuba | owner | 20 | 522 | - / - / 0-19 (소유자 19) | - | - | - | 100 (웜업 1000, 참여 0.4) | DISLAB | － 미생성 | 2026-10-13 17:30 |  |

## 실패 사유 (전체 문장)

- **bd_iid_r100w5_k20paper_median_s0**: 파일 243/295 개 없음 (예: global_net_model_bd_iid_r100w5_k20paper_median_s0_round_20.pkl) — 생성이 중간에 끊겼거나 라운드/웜업이 매니페스트와 다름
- **bd_iid_flame_s0**: 잠복 ASR 19.1 > 15 (잠복이 아님) / 잠복 ACC 89.5 < 90 / 정상 5 제거 후 ASR 50.6 > 40 (정상 삭제로 깨어남)
- **bd_iid_r100w5_k20paper_cifar_s0**: 잠복 ASR 97.1 > 20 (잠복이 아님) / 위장자 제거(빼기) 붕괴: ACC 9.9 < 25 (ASR 0.0 는 판정 불가) / 정상 5 제거 후 ASR 44.9 > 20 (정상 삭제로 깨어남)

## 검사 기준 (매니페스트 verify, 그룹별)

| 그룹 | 잠복 ASR ≤ | 잠복 ACC ≥ | 위장자 제거 후 ASR ≥ | 정상 제거 후 ASR ≤ | 붕괴 ACC < | 집계 | 근거 |
|---|---|---|---|---|---|---|---|
| k8_main (iid) | 15 | 90 | 30 | 30 | 50 | fedavg | benign_departure summary_iid: subtract req4 48.0±10.2 / req5 14.0±4.8, 삭제 전 ASR 6.2±1.8 ACC 95.1 |
| k8_main (owner) | 15 | 88 | 30 | 30 | 50 | fedavg | summary_owner: subtract req4 54.3±6.0 / req5 15.5±7.0 (소유자 7 은 29.4±5.4 라 정상 검사에 안 씀), 삭제 전 ACC 92.5±0.8 |
| k20_paper (iid) | 15 | 95 | 30 | 20 | 50 | fedavg | 실측 (DISLAB f8b1079, 6 궤적): 잠복 ACC 98.1–98.8 / ASR 3.1–7.8; 위장자 제거(작동점 ρ=0.2, ρ=1.0 은 붕괴) ASR 46–93, ACC 68.7–97.6; 정상 5 제거 ASR 2.4–9.8, ACC 94.8–97.8; 운반자 0 제거 ASR 0.2–1.4 |
| k20_paper (owner) | 15 | 95 | 30 | 20 | 50 | fedavg | 실측 (DISLAB f8b1079, 6 궤적): 잠복 ACC 98.1–98.8 / ASR 3.1–7.8; 위장자 제거(작동점 ρ=0.2, ρ=1.0 은 붕괴) ASR 46–93, ACC 68.7–97.6; 정상 5 제거 ASR 2.4–9.8, ACC 94.8–97.8; 운반자 0 제거 ASR 0.2–1.4 |
| k20_reduced (iid) | 15 | 90 | 20 | 30 | 50 | fedavg | 로컬 실측 bd_iid_k20_s0 (2026-10-12): 잠복 92.7 / 5.4, 위장자 4 제거 90.1 / 11.5, 정상 5 제거 92.5 / 6.2, 운반자 0 제거 92.2 / 4.4. K=20 에 8 라운드(공격 5 라운드)로는 백도어가 안 심어진다 → 이 그룹은 파이프라인 점검용(attack_required=false, 상태 △). 실험 근거는 k20_paper 로 |
| k20_reduced (owner) | 15 | 85 | 20 | 30 | 50 | fedavg | 로컬 실측 bd_iid_k20_s0 (2026-10-12): 잠복 92.7 / 5.4, 위장자 4 제거 90.1 / 11.5, 정상 5 제거 92.5 / 6.2, 운반자 0 제거 92.2 / 4.4. K=20 에 8 라운드(공격 5 라운드)로는 백도어가 안 심어진다 → 이 그룹은 파이프라인 점검용(attack_required=false, 상태 △). 실험 근거는 k20_paper 로 |
| sup2 (iid) | 15 | 90 | 30 | 30 | 50 | fedavg | 미측정 (위장자 2 명을 모두 뺀 모델에 sup_asr_min 적용, 각자 뺀 값은 기록만) (임시) |
| tdef_paper (iid, median) | 15 | 95 | 30 | 30 | 50 | median | 논문 Table I MNIST: BA .992 / ASR .092 (삭제 전), KD-FU 후 .700. 1 차 근사 빼기 값은 미측정 → 임시 문턱 (임시) |
| tdef_paper (iid, multi_krum) | 15 | 95 | 30 | 30 | 50 | multi_krum | 논문 Table I MNIST: BA .992 / ASR .092 (삭제 전), KD-FU 후 .700. 1 차 근사 빼기 값은 미측정 → 임시 문턱 (임시) |
| tdef_paper (iid, flame) | 15 | 95 | 30 | 30 | 50 | flame | 논문 Table I MNIST: BA .992 / ASR .092 (삭제 전), KD-FU 후 .700. 1 차 근사 빼기 값은 미측정 → 임시 문턱 (임시) |
| tdef_paper (iid, flDetector) | 15 | 95 | 30 | 30 | 50 | flDetector | 논문 Table I MNIST: BA .992 / ASR .092 (삭제 전), KD-FU 후 .700. 1 차 근사 빼기 값은 미측정 → 임시 문턱 (임시) |
| tdef_reduced (iid, median) | 15 | 90 | 30 | 40 | 50 | median | benign_departure summary_iid: subtract req4 48.0±10.2 / req5 14.0±4.8, 삭제 전 ASR 6.2±1.8 ACC 95.1 |
| tdef_reduced (iid, multi_krum) | 15 | 90 | 30 | 40 | 50 | multi_krum | benign_departure summary_iid: subtract req4 48.0±10.2 / req5 14.0±4.8, 삭제 전 ASR 6.2±1.8 ACC 95.1 |
| tdef_reduced (iid, flame) | 15 | 90 | 30 | 40 | 50 | flame | benign_departure summary_iid: subtract req4 48.0±10.2 / req5 14.0±4.8, 삭제 전 ASR 6.2±1.8 ACC 95.1 |
| tdef_reduced (iid, flDetector) | 15 | 90 | 30 | 40 | 50 | flDetector | benign_departure summary_iid: subtract req4 48.0±10.2 / req5 14.0±4.8, 삭제 전 ASR 6.2±1.8 ACC 95.1 |
| badfu (dominant70) | 35 | 75 | 단계식 40 | 35 | 20 | - | traj_full(15R, seed 42): 잠복 ACC 82.7 / ASR 27.1, FedEraser 단계 언러닝 후 ASR 52.2. 40R 값은 미측정 (임시) |
| local_smoke (iid) | 15 | 90 | 30 | 30 | 50 | fedavg | benign_departure summary_iid: subtract req4 48.0±10.2 / req5 14.0±4.8, 삭제 전 ASR 6.2±1.8 ACC 95.1 |
| k20_paper_cifar (iid) | 20 | 50 | 30 | 20 | 25 | fedavg | 임시: CIFAR-10/resnet18 은 미실측. 첫 궤적의 잠복 ACC/ASR·위장자 제거 ASR 을 보고 고칠 것 (ACC 가 50 아래면 train_epoch 를 늘려야 함) (임시) |
| clean_paper (iid) | 15 | 95 | 단계식 None | 20 | 50 | fedavg | 공격 없음: 잠복 ASR 낮음(트리거 생성기는 학습되지 않아 ASR 은 우연 수준)·ACC·정상 제거 무해만 본다 |
| clean_paper (owner) | 15 | 95 | 단계식 None | 20 | 50 | fedavg | 공격 없음: 잠복 ASR 낮음(트리거 생성기는 학습되지 않아 ASR 은 우연 수준)·ACC·정상 제거 무해만 본다 |

## 주의 — 코드 버그로 영향받는 기존 궤적

- **bd_iid_{median,multi_krum,flame,flDetector}_s* (DISLAB, 2026-10-12 이전에 run_paper.sh RQ2 로 만든 것)** — run_all.sh stage_attack 이 ATTACK_EXTRA(--median 등)를 run_real_fuba.sh 에 넘기지 않아 방어 없이 만들어졌다 (2026-10-12 수정). 검증기의 aggregation 검사(전역 모델 == 참여자 평균이면 방어 없음)가 잡는다 → 재생성 필요: FORCE=1 ONLY=tdef_reduced bash make_trajectories.sh
- **이름에 K 가 없는 K≠8 궤적 (예: K=16 TAG 없이 run_all.sh 로 만든 것, 'k16'/'r2' 류 이름)** — run_all.sh 이름은 bd_{cond}_s{seed} 로 K 를 담지 않아 K=20 으로 돌리면 K=8 궤적을 K=20 으로 분석할 수 있었다. 이제 K≠8 이름은 k20/k16sup2 를 반드시 담고, 검증기가 저장된 클라 수 == 매니페스트 K 를 확인한다 → 그런 이름의 궤적은 쓰지 말고 매니페스트 이름으로 다시 만들 것

## 아직 없는 궤적 만드는 법

- FUBA (DISLAB): `cd FUBA && bash make_trajectories.sh` (그룹 하나만: `ONLY=k20_paper bash make_trajectories.sh`, 명령만 보기: `DRY=1`)
- BadFU (노트북): `cd BadFU_src/BadFU-main && bash make_badfu_trajectories.sh`
- 만들고 나면 이 파일과 `FUBA/logs/trajectories/registry.json` 을 커밋한다.

<details><summary>미생성 궤적의 정확한 명령 (27 개)</summary>

```
# [DISLAB] bd_iid_r100w5_k20paper_median_s1
NB_CLIENTS=20 ROUNDS=100 WARMUP=5 PARTICIPANT_RATE=0.4 EXTRA_ARGS="--split_seed 523 --median --participant_rate 0.4" bash run_real_fuba.sh bd_iid_r100w5_k20paper_median_s1 523
# [DISLAB] bd_iid_r100w5_k20paper_median_s2
NB_CLIENTS=20 ROUNDS=100 WARMUP=5 PARTICIPANT_RATE=0.4 EXTRA_ARGS="--split_seed 524 --median --participant_rate 0.4" bash run_real_fuba.sh bd_iid_r100w5_k20paper_median_s2 524
# [DISLAB] bd_iid_r100w5_k20paper_multi_krum_s0
NB_CLIENTS=20 ROUNDS=100 WARMUP=5 PARTICIPANT_RATE=0.4 EXTRA_ARGS="--split_seed 522 --multi_krum --participant_rate 0.4" bash run_real_fuba.sh bd_iid_r100w5_k20paper_multi_krum_s0 522
# [DISLAB] bd_iid_r100w5_k20paper_multi_krum_s1
NB_CLIENTS=20 ROUNDS=100 WARMUP=5 PARTICIPANT_RATE=0.4 EXTRA_ARGS="--split_seed 523 --multi_krum --participant_rate 0.4" bash run_real_fuba.sh bd_iid_r100w5_k20paper_multi_krum_s1 523
# [DISLAB] bd_iid_r100w5_k20paper_multi_krum_s2
NB_CLIENTS=20 ROUNDS=100 WARMUP=5 PARTICIPANT_RATE=0.4 EXTRA_ARGS="--split_seed 524 --multi_krum --participant_rate 0.4" bash run_real_fuba.sh bd_iid_r100w5_k20paper_multi_krum_s2 524
# [DISLAB] bd_iid_r100w5_k20paper_flame_s0
NB_CLIENTS=20 ROUNDS=100 WARMUP=5 PARTICIPANT_RATE=0.4 EXTRA_ARGS="--split_seed 522 --flame --participant_rate 0.4" bash run_real_fuba.sh bd_iid_r100w5_k20paper_flame_s0 522
# [DISLAB] bd_iid_r100w5_k20paper_flame_s1
NB_CLIENTS=20 ROUNDS=100 WARMUP=5 PARTICIPANT_RATE=0.4 EXTRA_ARGS="--split_seed 523 --flame --participant_rate 0.4" bash run_real_fuba.sh bd_iid_r100w5_k20paper_flame_s1 523
# [DISLAB] bd_iid_r100w5_k20paper_flame_s2
NB_CLIENTS=20 ROUNDS=100 WARMUP=5 PARTICIPANT_RATE=0.4 EXTRA_ARGS="--split_seed 524 --flame --participant_rate 0.4" bash run_real_fuba.sh bd_iid_r100w5_k20paper_flame_s2 524
# [DISLAB] bd_iid_r100w5_k20paper_flDetector_s0
NB_CLIENTS=20 ROUNDS=100 WARMUP=5 PARTICIPANT_RATE=0.4 EXTRA_ARGS="--split_seed 522 --flDetector --participant_rate 0.4" bash run_real_fuba.sh bd_iid_r100w5_k20paper_flDetector_s0 522
# [DISLAB] bd_iid_r100w5_k20paper_flDetector_s1
NB_CLIENTS=20 ROUNDS=100 WARMUP=5 PARTICIPANT_RATE=0.4 EXTRA_ARGS="--split_seed 523 --flDetector --participant_rate 0.4" bash run_real_fuba.sh bd_iid_r100w5_k20paper_flDetector_s1 523
# [DISLAB] bd_iid_r100w5_k20paper_flDetector_s2
NB_CLIENTS=20 ROUNDS=100 WARMUP=5 PARTICIPANT_RATE=0.4 EXTRA_ARGS="--split_seed 524 --flDetector --participant_rate 0.4" bash run_real_fuba.sh bd_iid_r100w5_k20paper_flDetector_s2 524
# [DISLAB] bd_iid_flame_s1
NB_CLIENTS=8 ROUNDS=8 WARMUP=3 PARTICIPANT_RATE=1.0 EXTRA_ARGS="--split_seed 523 --flame" bash run_real_fuba.sh bd_iid_flame_s1 523
# [DISLAB] bd_iid_flame_s2
NB_CLIENTS=8 ROUNDS=8 WARMUP=3 PARTICIPANT_RATE=1.0 EXTRA_ARGS="--split_seed 524 --flame" bash run_real_fuba.sh bd_iid_flame_s2 524
# [DISLAB] bd_iid_flDetector_s0
NB_CLIENTS=8 ROUNDS=8 WARMUP=3 PARTICIPANT_RATE=1.0 EXTRA_ARGS="--split_seed 522 --flDetector" bash run_real_fuba.sh bd_iid_flDetector_s0 522
# [DISLAB] bd_iid_flDetector_s1
NB_CLIENTS=8 ROUNDS=8 WARMUP=3 PARTICIPANT_RATE=1.0 EXTRA_ARGS="--split_seed 523 --flDetector" bash run_real_fuba.sh bd_iid_flDetector_s1 523
# [DISLAB] bd_iid_flDetector_s2
NB_CLIENTS=8 ROUNDS=8 WARMUP=3 PARTICIPANT_RATE=1.0 EXTRA_ARGS="--split_seed 524 --flDetector" bash run_real_fuba.sh bd_iid_flDetector_s2 524
# [local] badfu_s42
python fl_detect_badfu.py --rounds 40 --local_epochs 5 --keep_traj --seed 42 --traj_dir logs/traj_s42 --out logs/badfu_detect_s42.json
# [local] badfu_s43
python fl_detect_badfu.py --rounds 40 --local_epochs 5 --keep_traj --seed 43 --traj_dir logs/traj_s43 --out logs/badfu_detect_s43.json
# [local] badfu_s44
python fl_detect_badfu.py --rounds 40 --local_epochs 5 --keep_traj --seed 44 --traj_dir logs/traj_s44 --out logs/badfu_detect_s44.json
# [local] badfu_r15_s42
python fl_detect_badfu.py --rounds 15 --local_epochs 5 --keep_traj --seed 42 --traj_dir logs/traj_full --out logs/badfu_detect.json
# [local] loc_fuba
NB_CLIENTS=8 ROUNDS=8 WARMUP=3 PARTICIPANT_RATE=1.0 EXTRA_ARGS="--split_seed 522" bash run_real_fuba.sh loc_fuba 522
# [DISLAB] bd_iid_r100w5_k20paper_cifar_s1
DATASET=cifar10 NB_CLIENTS=20 ROUNDS=100 WARMUP=5 PARTICIPANT_RATE=0.4 EXTRA_ARGS="--split_seed 523 --participant_rate 0.4" bash run_real_fuba.sh bd_iid_r100w5_k20paper_cifar_s1 523
# [DISLAB] bd_iid_r100w5_k20paper_cifar_s2
DATASET=cifar10 NB_CLIENTS=20 ROUNDS=100 WARMUP=5 PARTICIPANT_RATE=0.4 EXTRA_ARGS="--split_seed 524 --participant_rate 0.4" bash run_real_fuba.sh bd_iid_r100w5_k20paper_cifar_s2 524
# [DISLAB] clean_iid_r100_k20paper_s0
DATASET=mnist NB_CLIENTS=20 ROUNDS=100 WARMUP=1000 PARTICIPANT_RATE=0.4 EXTRA_ARGS="--split_seed 522 --participant_rate 0.4" bash run_real_fuba.sh clean_iid_r100_k20paper_s0 522
# [DISLAB] clean_iid_r100_k20paper_cifar_s0
DATASET=cifar10 NB_CLIENTS=20 ROUNDS=100 WARMUP=1000 PARTICIPANT_RATE=0.4 EXTRA_ARGS="--split_seed 522 --participant_rate 0.4" bash run_real_fuba.sh clean_iid_r100_k20paper_cifar_s0 522
# [DISLAB] clean_owner_r100_k20paper_s0
DATASET=mnist NB_CLIENTS=20 ROUNDS=100 WARMUP=1000 PARTICIPANT_RATE=0.4 EXTRA_ARGS="--split_seed 522 --participant_rate 0.4 --non_iid --non_iid_type class_owner --owner_class 3 --owner_frac 0.9 --owner_client -1" bash run_real_fuba.sh clean_owner_r100_k20paper_s0 522
# [DISLAB] clean_owner_r100_k20paper_cifar_s0
DATASET=cifar10 NB_CLIENTS=20 ROUNDS=100 WARMUP=1000 PARTICIPANT_RATE=0.4 EXTRA_ARGS="--split_seed 522 --participant_rate 0.4 --non_iid --non_iid_type class_owner --owner_class 3 --owner_frac 0.9 --owner_client -1" bash run_real_fuba.sh clean_owner_r100_k20paper_cifar_s0 522
```
</details>
