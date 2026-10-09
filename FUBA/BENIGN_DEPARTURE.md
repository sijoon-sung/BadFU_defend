# 정상 이탈 검증 (benign departure) — 실제 FU 위에서 go/no-go

## 묻는 것

BadFU Table V 와 FUBA Table V 는 **정상 클라이언트를 지워도 ASR 이 오르지 않는다**고 보고한다 (FUBA 는 Dirichlet α=0.1 에서도).
우리 예비 관찰(`diff_audit`, fp3/fp4)은 **한 클래스를 90% 독점한 정상 클라이언트**를 1차 근사(θ_T − U_k)로 지우면
ASR 이 5.8 → 38 로 올랐다. 이 차이가

- (a) 분할 조건 때문인지 (독점 vs Dirichlet / IID), 아니면
- (b) 1차 근사 빼기가 실제 FU 를 과장한 것인지

를 같은 궤적·같은 생성기·같은 FU 로 가른다. 결과에 따라 "정상 이탈 공격"을 세우거나 깨끗이 접는다.

## 설계

- 궤적: FUBA 공개 코드(MNIST, IBA, K=8, 공격자 0~3, Adv-defender 4, 정상 5~7, 8라운드, 웜업 3), 시드 3개.
- 조건: `iid` / `owner`(정상 7 이 클래스 3 의 90% 독점) / `dir01`(Dirichlet α=0.1, FUBA Table V 조건).
- 요청자: 4 (공격이 설계한 삭제) / 7 (독점 정상) / 5 (보통 정상). 학습 때 역할(defender=4)은 고정하고 **삭제 대상만** 바꾼다.
- FU: `subtract`(1차 근사) / `distillation`(KD-FU, 비재학습) / `fedEraser`(원 논문식 보정) / `retrain_benign`(남은 전원 정직 재학습) / `retrain_attack`(FUBA 원본: 공격자 전력 재주입).
- 평가: 학습 끝 저장된 트리거 생성기로 ACC, ASR, 클래스별 ASR. `retrain_attack` 은 재학습 중 새로 학습된 생성기 ASR 도 따로 기록(`asr_regen`).

## 판정

| 결과 | 해석 | 다음 |
|---|---|---|
| `owner` 조건, req 7 에서 실제 FU(distillation/fedEraser/retrain_benign)로도 ASR 이 크게 오르고, `iid`/`dir01` 의 req 7·5 는 안 오름 | 정상 이탈 공격은 **독점 편중 조건에서 실재**. 두 논문의 Table V 는 조건이 달라서 못 본 것 | 공격 설계(어떤 정상 클라가 상쇄 역할을 떠안는지)로 진행 |
| req 7 이 `subtract` 에서만 오르고 실제 FU 에서는 안 오름 | 1차 근사의 과장. 정상 이탈 공격은 접는다 | "잠복형 방어"만 남김 (요청자 = 공격자 전제) |
| `retrain_attack` 만 오르고 `retrain_benign`·`fedEraser` 는 안 오름 (req 4 포함) | FUBA 재학습형 결과는 **재주입** 몫이 크다. 순수 잠복 활성화는 KD/보정형에서만 봐야 함 | FU-단계 방어(재집계에 강건집계)로 재학습형을 막는 결과를 하나 확보 |

## 실행

```bash
cd FUBA
COND=owner bash run_benign_departure.sh          # 시드 0 1 2, 궤적 생성 + 언러닝 + 집계
COND=iid   bash run_benign_departure.sh
COND=dir01 bash run_benign_departure.sh
```

결과: `logs/benign_departure/bd_{cond}_s{seed}.json`, 집계 `logs/benign_departure/summary_{cond}.md`.
저장된 궤적이 있으면 `SKIP_ATTACK=1`, 방법을 줄이려면 `METHODS="subtract distillation fedEraser"`.

단일 실행:

```bash
python benign_departure.py --name bd_owner_s0 --K 8 --seed 522 --split_seed 522 \
  --non_iid --non_iid_type class_owner --owner_class 3 --requesters 4 7 5
```

## 이 브랜치에서 FUBA 원본 코드를 고친 것 (실험 타당성에 필요한 최소)

1. **분할 일치** — 원본은 모든 FU(`retrain`, `fedEraser`, `sga`, `utils_federaser`)가 학습 때 분할과 무관하게 IID `random_split(seed 522)` 을 다시 했다. non-IID 궤적을 지우면 요청자에게 다른 데이터가 배정돼 실험이 성립하지 않는다. `utils.comm_utils.make_client_split(trainset, config)` 하나로 학습(`train_main.py`)과 FU 가 같은 분할을 쓰게 했다. `--split_seed` 로 비-IID 분할 시드를 지정한다(기본 1223 = 기존 궤적과 호환).
2. **재학습 중 공격자 모드** — `--fu_attacker_mode attack|benign`. `attack` 이 원본(FUBA 논문 각주 4: 재학습 중 γ=1 전력 주입). `benign` 은 남은 전원이 정직하게 학습해 "제거" 효과만 남긴다.
3. **원 논문식 FedEraser** — `unlearn_method/federaser_faithful.py`. 저장소의 `fedEraser` 래퍼는 보정 대신 공격자 참여 재학습 루프를 부르고, `unlearning_step_once` 는 step_length 를 새 보정 노름으로 잡아 원식과 다르다. 원본 래퍼는 `fedEraser_fuba` 로 남겨 두었다.
4. **KD-FU 의 나눗수** — `distillation.py` 가 클라이언트 수를 20 으로 박아 두어 K=8 궤적에서 기여를 2.5배 과소 제거했다. 라운드당 저장된 로컬 수로 바꿨다. 생성기 재최적화(`train_pattern`)는 `config.iba_reopt` 로 끌 수 있게 했다(평가에서는 끔: 그건 더 강한 공격자다).
5. `unlearn.py` 에 `class_owner` 분할·`--seed`·`--split_seed` 인자 추가. `unlearn_method/utils.Arguments.forget_client_idx` 가 4 로 박혀 있던 것을 `config.unlearn_target` 으로.

## 주의

- KD-FU(`distillation`)는 원본대로 **전체 trainset** 으로 증류한다(요청자 데이터 포함). 망각 관점에서는 문제지만 FUBA 저자 구현을 그대로 둔 것이다.
- 모든 수치는 K=8·8라운드 소규모 설정이다. 경향만 본다. 방향이 서면 FUBA 기본(20명·100라운드)으로 다시 잰다.
- 판정 기준은 결과를 보기 전에 위 표로 고정했다.
