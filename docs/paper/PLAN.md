> **역사적 설계·검토 기록.** 현재 목표·검증 상태는 [연구 요약](../RESEARCH_SUMMARY.md), 인용 범위는 [출처 감사](../EVIDENCE_AUDIT.md)를 따른다. 과거 계획의 고정 주장·완료 표현은 현재의 검증 결과가 아니다.

# 앞으로의 계획 (2026-10-12)

짧게, 순서대로. 숫자는 결정에 필요한 것만. 기여 문장은 ZERO_BASE_REVIEW.md §8 로 고정이고 여기서 바꾸지 않는다.
줄임말: ASR = 공격 성공률, ACC = 정확도, 경보 점수 = 빼 봤을 때 남들보다 얼마나 이상한가(z, 문턱 3).

## 0. 용어
- **운반자**: 학습 중에 백도어를 심는 클라이언트.
- **위장자**: 같은 입력에서 백도어를 눌러 숨기고, 나중에 자기 데이터 삭제를 요청하는 클라이언트.
- **DOSE**: 위장자 몫을 한 번에 다 빼지 않고 조금씩 빼면서, 매번 "남을 같은 양 뺀 띠" 안에 있는지 확인하는 삭제 연산자.
- **TRACE**: 띠를 벗어났을 때 위장자의 짝, 곧 운반자가 누구인지 찾는 단계.
- **DEFUSE**: 찾은 운반자 몫에서 백도어 성분만 걷어내는 단계. 위장자 데이터는 되살리지 않는다.

## 1. 지금 상태
- 설정부터 어긋나 있다. FUBA 논문의 주 설정은 클라이언트 20, 악성 5 (운반자 4 + 위장자 1 = 25 %), 100 라운드, 참여율 0.4, 10 시드다 (paper_reviews/02_FUBA.md). 지금 본 결과는 K=8, 악성 5/8 (과반), 8 라운드, 전원 참여다. 이것은 주 설정이 될 수 없다. 앞으로 모든 주장은 논문 설정 그대로의 K=20 궤적(`paper`)으로 세운다. K=8 결과는 예비 근거다.
- 예비로 3 시드가 선 것 (K=8): 주장 1–6, 8 (FUBA, MNIST, IID 와 owner). 위장자 3/3 잡음, 운반자 정밀도 1.0 (IID) / 0.87 (owner), ASR 48→9 / 54→9, 재주입 0, 정상 요청 비용 0.
- 1 시드뿐인 것: BadFU 전부 (로컬, 시드 42), 주장 7 (증류가 망각을 되돌리는 사례), null-space 게이트 비교군.
- 깨진 것: 적응형 3 종. 위장자 분산 (모사), 미끼 ×2.5 (경보 점수 14→2), 운반자 분산 (재현율 0.58→0.28). 셋 다 모사 공격이고 옛 부품(dose_fu, pair_roles)으로 잰 것이다. 새 부품(cancel_fu)으로는 아직 안 쟀다.
- 없는 것: K=20 궤적, FUBA 두 번째 데이터셋, BadFU 3 시드, 집계 방어 4 종 위 궤적, INFOCOM SVD 비교군 (코드 없음).
- 남는 약점: owner 소유자 오탐 1/3 (같은 시드), BadFU 연산자 열화 (ACC 76), K=8 에서 p 값 바닥 0.125.
- BadFU 의 1 시드 결과 파일(`logs/cancel/loc_badfu*.json`)은 저장소에 없다. 보고서 숫자만 있다. 다시 돌려야 한다.

## 2. 실험 기록 체계
두 장부를 쓴다. 장부에 없는 숫자는 논문에 못 들어간다.

**(a) `docs/EXPERIMENT_LOG.md` = 궤적 장부 (자동).** `FUBA/make_trajectories.sh` 가 궤적을 만들고 `verify_trajectories.py` 가 세 가지를 재서 한 줄씩 쓴다: 잠복 ASR 이 낮은가, 위장자를 빼면 ASR 이 오르는가, 정상을 빼면 해가 없는가. 통과한 궤적만 이름을 갖는다. (둘 다 병렬 작업 중, 만들 것.)

**(b) `docs/RESULTS_LOG.md` = 분석 장부 (사람이 쓴다).** 실행 한 번에 한 줄. 줄 형식:

| 번호 | 날짜 | 궤적 | 스크립트+단계 | 인자/TAG | 결과 파일 | 답하는 주장 | 상태 |
|---|---|---|---|---|---|---|---|
| R-001 | 2026-10-12 | bd_iid_s0, s1, s2 | run_all.sh cancel (cancel_fu.py) | 기본값, TAG 없음, dev 93416d5 | FUBA/logs/cancel/bd_iid_s{0,1,2}.json → summary_iid.md | 주장 3·4·5·6 (IID) | 확정 (예비, K=8) |
| R-002 | 2026-10-12 | bd_owner_s2 | run_all.sh cancel | 기본값 | FUBA/logs/cancel/bd_owner_s2.json | 주장 3 의 약점: 정상 7 오탐 (점수 17) | 확정, 한계로 보고 |
| R-003 | 2026-10-12 | loc_badfu (시드 42) | cancel_fu.py --fmt badfu --op kd (RUN.md C) | --name loc_badfu_can, --polish kd_orth 포함 | FUBA/logs/cancel/loc_badfu_can.json — 저장소에 없음 (REPORT §14.5 숫자만) | 주장 1·3·4 (BadFU) | 파일 없음 → Phase 4 에서 bd_badfu_s0 으로 다시 |

규칙 (장부 머리에 적는다):
1. 궤적 열은 EXPERIMENT_LOG 의 이름 그대로. 거기 없는 궤적으로는 돌리지 않는다.
2. 상태는 넷 중 하나: 예정 / 돌아감 / 확정 / 폐기(이유). 파일이 저장소에 없으면 '파일 없음'.
3. 설정이 다르면 TAG 가 다르다. 같은 이름 위에 FORCE=1 로 덮어쓰지 않는다.
4. 논문 표의 숫자마다 번호(R-xxx)를 단다. 번호 없는 숫자는 뺀다.
5. 실행 묶음이 끝나면 바로 커밋: `git add FUBA/logs docs && git commit -m "results: R-xxx..."`.

## 3. 단계별 계획

### Phase 0. 궤적 만들기와 검증 (1 주차)
목표: 뒤의 모든 실험이 쓸 궤적을 한 번에 만들고 검증해서 이름을 준다. 이름 규칙은 run_all.sh 의 `bd_{조건}{TAG}_s{시드}`. 아래 이름은 예상이고, make_trajectories.sh 가 정한 이름이 우선이다.

| 궤적 (예상 이름) | 설정 | 어디서 | 시간 (대략) | 쓰는 단계 |
|---|---|---|---|---|
| bd_iid_s0–2, bd_owner_s0–2 | K=8 (예비). 운반자 0–3, 위장자 4 | DISLAB 에 있음 | 0 | 예비·부록 |
| bd_{iid,owner}_r100w5_paper_s0–2 (`paper`) | **주 설정 = FUBA 논문 그대로**: K=20, 운반자 0–3 + 위장자 4 (25 %), 정상 5–19, 100 라운드, 웜업 5, 참여율 0.4 | DISLAB | 궤적당 1–3 시간 (추정) | 1, 2, 3 |
| bd_{iid,owner}_k20_s0–2 (`reduced`) | K=20, 8 라운드, 웜업 3, 전원 참여. 싼 확인·소거·적응형용 | DISLAB | 시드당 10–30 분 | 3, 소거 |
| bd_iid_sup2_s0–2 | `--clients_ratial 2`: K=16, 위장자 4·8, 운반자 0–3·5–7·9, 정상 10–15 (reduced 설정) | DISLAB | 시드당 10–30 분 | 3 |
| bd_iid_r100w5_paper_{median,multi_krum,flame,flDetector}_s0–2 | `paper` 설정 위에 집계 방어 4 종 (IID). K=8 로는 돌리지 않는다 | DISLAB | 12 개, 궤적당 1–3 시간 | 2 |
| badfu_s0 (시드 42, 있음), badfu_s1 (43), badfu_s2 (44) | CIFAR-10, ResNet-18, K=6, 15 라운드 | 로컬 노트북 | 시드당 1 시간 | 4 |

비용 (FUBA 논문 Table VI: 4090 에서 라운드당 운반자 7 초, 위장자 10 초, 정상 0.4 초. 라운드마다 8 명 참여, 100 라운드): 4090 이면 궤적당 15–30 분. DISLAB 에서는 방어 계산까지 넣어 1–3 시간으로 본다. `paper` 6 개 + 집계 방어 12 개 = 18 개 → 20–55 시간. DISLAB 를 계속 쓰면 2–4 일. 첫 궤적 하나를 먼저 돌려 시간을 재고 장부에 적는다. 그 값으로 일정을 고친다.

명령:
```bash
cd FUBA
bash make_trajectories.sh            # (만들 것, 병렬 작업 중) 위 표의 DISLAB 궤적 전부 → verify_trajectories.py → docs/EXPERIMENT_LOG.md
bash make_badfu_trajectories.sh      # (만들 것) 로컬. BadFU 시드 43·44
```
스크립트가 아직 없을 때 지금 되는 명령 (같은 이름으로):
```bash
NB_CLIENTS=20 ROUNDS=100 WARMUP=5 EXTRA_ARGS="--participant_rate 0.4 --split_seed 522" bash run_real_fuba.sh bd_iid_r100w5_paper_s0 522   # paper (시드 0; 1·2 는 523·524)
K=20 TAG="_k20" STAGES="attack benign" BENIGN_METHODS="subtract" CONDS="iid owner" bash run_all.sh      # reduced K=20 궤적 + 검증(빼기 뒤 ASR)
NB_CLIENTS=8 ROUNDS=8 WARMUP=3 EXTRA_ARGS="--clients_ratial 2 --split_seed 522" bash run_real_fuba.sh bd_iid_sup2_s0 522   # 두 위장자
NB_CLIENTS=20 ROUNDS=100 WARMUP=5 EXTRA_ARGS="--participant_rate 0.4 --median --split_seed 522" bash run_real_fuba.sh bd_iid_r100w5_paper_median_s0 522   # 집계 방어 (median 예)
cd ../BadFU_src/BadFU-main && python fl_detect_badfu.py --rounds 15 --local_epochs 5 --seed 43 --traj_dir logs/traj_full_s1 --out logs/badfu_detect_s1.json --keep_traj   # BadFU 시드 1
```
주의 네 가지. (i) run_all.sh 의 이름에 K 가 안 들어간다. K=20 은 반드시 TAG (`_paper`, `_k20`). 없으면 K=8 궤적 위에 덮어쓴다. (ii) run_all.sh 의 `ATTACK_EXTRA` 는 지금 궤적 생성에 전달되지 않고, run_real_fuba.sh 는 참여율 1.0 을 고정으로 넣는다. `paper` 는 EXTRA_ARGS 끝에 `--participant_rate 0.4` 를 붙여야 한다 (뒤 값이 이긴다). 로컬 학습량은 코드 기본 `train_epoch=100` (배치 수) 이고 논문은 E=50 이다. 매니페스트가 쓴 값을 장부에 적는다. (iii) BadFU 는 `client_data_counts` 가 화면에 찍힌다. 그 여섯 숫자를 EXPERIMENT_LOG 줄에 적는다 (뒤에서 `--counts` 로 쓴다). 독 데이터(record)는 시드와 무관하게 같다. (iv) 이름이 매니페스트와 다르면 매니페스트가 이긴다.

**관문 0.** 검증 기준은 verify_trajectories.py 의 값을 따른다. 없으면: 잠복 ASR 이 위장자 제거 뒤 ASR 의 절반 이하, 위장자 제거 뒤 ASR 이 정상 제거 뒤 ASR 보다 20 점 이상 높음, 정상 제거 뒤 ACC 하락 5 점 이하. 두 위장자 궤적은 "둘 다 뺐을 때" 로 잰다 (verify 에 넣을 것). 떨어지면 시드를 하나 더 만든다 (최대 2 번). 그래도 안 되면 그 설정은 논문에서 뺀다.

### Phase 1. 본 결과: K=8 재집계(예비), `paper` K=20 (2 주차) — 주장 1·2·3·4·5·6·8
K=8 은 다시 돌리지 않는다. 집계만 다시 해서 장부 줄을 만든다 (예비). 모든 주장의 본문 숫자는 `paper` 궤적에서 나온다. `reduced` 는 싼 확인용이다.
```bash
cd FUBA
for c in iid owner; do python aggregate_benign_departure.py --cond $c; python aggregate_dose_fu.py --cond $c; python aggregate_cancel.py --cond $c; python aggregate_effects.py --cond $c; python aggregate_adaptive.py --cond $c; python aggregate_baselines.py --cond $c; python paper_tables.py --cond $c; done
K=20 ROUNDS=100 WARMUP=5 TAG="_paper" FORCE=1 STAGES="benign dose cancel effects aggregate" CONDS="iid owner" BENIGN_METHODS="subtract distillation fedEraser" bash run_all.sh
python paper_tables.py --cond iid_r100w5_paper; python paper_tables.py --cond owner_r100w5_paper
K=20 TAG="_k20" FORCE=1 STAGES="benign dose cancel aggregate" CONDS="iid owner" bash run_all.sh      # reduced: 재학습 2 종까지 (싸다)
```
어디서: DISLAB. 시간: 집계 몇 분. `paper` 의 cancel 은 시드당 몇 분 (100 라운드를 읽는다, 학습 없음). 재학습 2 종은 100 라운드에서는 궤적 하나 값이라 `paper` 에서는 뺀다 (reduced 와 K=8 에서만). pair·unified 단계는 돌리지 않는다 (cancel 의 셋째 표가 대신한다). 매니페스트의 이름이 위와 다르면 run_all.sh 대신 cancel_fu.py 를 그 이름으로 직접 부른다 (`--name <이름> --K 20 --rounds 100`).

**관문 1.** `paper` IID 에서 위장자 purified 3/3, 정화 뒤 ASR 이 정상 제거 수준 +5 점 이내, 운반자 정밀도 0.9 이상, 정상 두 명 accept 6/6. owner 에서 소유자 오탐 1/3 이하. 통과 → 본문 표 전부 `paper`, K=8 은 부록. 위장자 경보가 1/3 이하면 방법이 논문 설정에서 안 되는 것이다 → 멈추고 그대로 보고한다. reduced 와 비교해 무엇이 다른지(참여율, 라운드 수)만 본다. 설정을 손봐서 통과시키지 않는다 (§6 결정 1).

### Phase 2. 비교군 (1–3 주차) — 주장 9
**먼저: 공식 코드 가져오기 (`third_party/`, 1 주차).** 사용자 지시: 비교군을 직접 짜지 않고 저자의 GitHub 코드를 가져온다. 가져온 코드는 `third_party/<이름>/` 에 두고 출처(주소, 커밋)를 README 한 줄로 적는다.

| 방법 | 공식 코드 | 지금 저장소 | 할 일 |
|---|---|---|---|
| Median, Multi-Krum, Trimmed-Mean | FUBA 저자 코드 `defence/simple_defence_method.py` (`--median`, `--multi_krum`) | 있음 | 그대로 |
| FLDetector | github.com/zaixizhang/FLDetector | `defence/FLDetector` 복사본 (`--flDetector`) | 그대로 |
| FLAME | FUBA 에 든 `defence/FLAME` 은 zhmzm 재구현 (README 확인). 원 저자 공개 코드 확인 | 복사본 (`--flame`) | FUBA Table II 와 같은 코드이므로 그대로 쓰고 출처를 적는다 |
| MASA | github.com/JiiahaoXU/MASA (주소 확인) | `defence/MASA` 복사본 (`--IBMFL`) | **`baseline_masa.py` (proxy) 를 이것으로 교체.** 학습 중 방어 궤적로 돌린다 |
| BackdoorIndicator, AlignIns | github.com/ybdai7/Backdoor-indicator-defense, AlignIns (주소 확인) | `defence/` 복사본 (`--Indicator`, `--AlignIns`) | 시간이 남으면 |
| FedEraser | 저자 코드 (주소 확인) | `unlearn_method/FedEraser` 복사본 | 그대로 |
| 사후 정화 (fine-pruning, NAD) | github.com/SCLBD/BackdoorBench | 없음 | 가져온다. "방향을 찾아 지우는 사후 정화" 대표 |
| SubID / Malicious Forgetting (INFOCOM'26) | 공개 코드 없음 (paper_reviews/04) | `baseline_subid.py` (proxy) | 저자에게 요청. 3 주차까지 없으면 표에서 빼고 관련 연구에서만. **proxy 숫자를 SubID 로 인용하지 않는다** |
| null-space 게이트 | 우리 구성 (발표된 방법 아님) | `baseline_gate.py` | 비교군 표가 아니라 "검토한 대안" 한 줄 |

학습 중 방어 위 궤적은 **반드시 `paper` K=20** 으로 (FUBA Table II: n=20 에서 10 종 모두 언러닝 뒤 ASR 0.93–0.97). K=8 로 돌린 집계 방어 숫자는 악성 과반이라 인용할 수 없다. MASA 공식 코드(`--IBMFL`)도 같은 식의 궤적 변형으로 넣는다 (매니페스트에 추가할 것, +3 개).
```bash
cd FUBA
for d in median multi_krum flame flDetector IBMFL; do K=20 ROUNDS=100 WARMUP=5 TAG="_paper_$d" FORCE=1 STAGES="benign dose cancel aggregate" BENIGN_METHODS="subtract distillation fedEraser" CONDS="iid" bash run_all.sh; done   # 궤적은 Phase 0. 깨어나는가 + 우리 방법이 되는가
K=20 ROUNDS=100 WARMUP=5 TAG="_paper" STAGES="subid latent aggregate" CONDS="iid owner" bash run_all.sh     # proxy·가중치 전용: 교체 전까지 참고용만
```
어디서: DISLAB. 시간: 첫 줄 시드·방어당 10 분 (궤적은 빼고), 둘째 줄 30 분. `run_paper.sh RQ=2` 는 쓰지 않는다 (궤적을 다시 만들려 하고, 그때 방어 인자가 빠진다). T2b 는 `--cond iid_r100w5_paper` 로 `_paper_median` 등을 읽는다.

**관문 2.** `paper` 에서 집계 방어 4 종 모두 빼기 뒤 ASR 40 이상 (논문은 0.93 이상) → "학습 중 방어는 못 막는다" 확정. 그 위에서 DOSE+TRACE 가 위장자 3/3 → 주장 9 확정. MASA 공식 코드가 학습 중 위장자를 표시하지 못하면 (3 시드 중 1 이하) MASA 줄 확정. 어떤 방어가 막으면 (ASR 20 미만) 그 숫자를 표에 그대로 적는다. 실험 없이 "우회 가능" 같은 말을 쓰지 않는다.

### Phase 3. 적응형 공격 (3–4 주차) — 주장 10
세 공격: 위장자 분산 (= 두 진짜 위장자, 모사 아님), 미끼, 운반자 분산. 이 단계는 `reduced` 와 `sup2` 궤적으로 한다 (싸다). `paper` 설정의 두 위장자 궤적은 관문 3 을 통과한 뒤에만 만든다.
```bash
cd FUBA
# (1) 모사 5 종을 고친 코드로 다시 (K=8, DOSE 수준, 옛 부품). 지금 파일은 캐시 버그 전 것
FORCE=1 STAGES="adaptive aggregate" CONDS="iid owner" DOSE_SPLIT=10 bash run_all.sh
# (1') 미끼 대책: 문턱을 남들의 점수로 보정 (있는 기능). 표는 손으로 모은다 (집계 스크립트 없음)
for s in 0 1 2; do python dose_fu.py --fmt fuba --name bd_iid_s$s --K 8 --rounds 8 --seed $((522+s)) --split_seed $((522+s)) --defender 4 --attackers 0 1 2 3 --sim "scale:5:2.5,scale:6:2.5,scale:7:2.5" --requesters 4 --arms oneshot --tau_mode loo --out logs/adaptive/bd_iid_s${s}_decoy_loo.json; done
# (2-a) 두 진짜 위장자, 한 명씩 따로: 한 명만 빼도 깨어나는가, TRACE 가 운반자 8 명을 찾는가
for s in 0 1 2; do python cancel_fu.py --fmt fuba --name bd_iid_sup2_s$s --K 16 --rounds 8 --seed $((522+s)) --split_seed $((522+s)) --defenders 4 8 --attackers 0 1 2 3 5 6 7 9 --requesters 4 8 15 10 --out logs/cancel/bd_iid_sup2_s$s.json; done
python aggregate_cancel.py --cond iid_sup2
# (2-b) 두 위장자가 차례로 요청: DOSE 만 (전체 예산 띠). 정상 대조 열 15, 10
for s in 0 1 2; do python schedule_fu.py --fmt fuba --name bd_iid_sup2_s$s --K 16 --rounds 8 --seed $((522+s)) --defender 4 --attackers 0 1 2 3 5 6 7 9 --requests 4 8 --control 15 10 --band n --stat flow --tag sup2seq; done   # → logs/adaptive/bd_iid_sup2_s{s}_sup2seq.json. 8 의 역할 표기가 틀리게 찍힘 (고칠 것: schedule_fu --defenders)
```
만들 것 (코드 1–2 일, 3 주차): `cancel_fu --sim` (attack_sim 을 dose_fu 처럼 붙인다. 미끼·운반자 분산을 TRACE 회전 2 에서 재기 위해), `cancel_fu --sequential` (첫 요청을 처리한 모델에서 둘째 요청을 받고, 띠는 "정상 2 명 제거" 모집단. 두 위장자에 TRACE·DEFUSE 까지). 만든 뒤:
```bash
for s in 0 1 2; do python cancel_fu.py --fmt fuba --name bd_iid_s$s --K 8 --rounds 8 --seed $((522+s)) --split_seed $((522+s)) --sim "split:0:3,split:1:3,split:2:3,split:3:3" --requesters 4 --out logs/adaptive/bd_iid_s${s}_distcarr_cancel.json; done   # --sim 은 (만들 것)
```
어디서: DISLAB. 시간: (1) 1 시간, (2) 1 시간, 만들 것 포함 3 일.

**관문 3 (명시적 분기).**
- 통과: 둘째 위장자 요청에서 DOSE 경보 2/3 시드 이상, 정상 대조 열 오탐 1/3 이하, (2-c 가 되면) TRACE 운반자 정밀도 0.8 이상. 미끼는 보정 문턱으로 위장자 경보 3/3 회복. 운반자 분산은 회전 2 재현율 0.8 이상. → 주장 10 을 "세 가지 중 막는 것과 못 막는 것" 표로 쓴다.
- 떨어짐: 두 위장자에서 경보 1/3 이하이거나 대조 열 오탐 2/3 이상. → 사용자가 고른다 (§6 결정 2). (A) 논문 주장을 "위장자 한 명" 으로 좁히고, 한계 절에 두 위장자 결과를 표로 그대로 싣는다. (B) "요청 시점 방어의 한계" 를 중심으로 다시 쓴다 (ZERO_BASE_REVIEW §7 의 분기). 둘 중 무엇이 맞는지는 이 문서가 정하지 않는다.
- 미끼가 보정 문턱으로도 안 막히면 한계로 쓴다. 설계를 더 고치지 않는다.

### Phase 4. BadFU 3 시드, FUBA 두 번째 데이터셋 (2–4 주차, 로컬) — 주장 1·3·4 (CIFAR)
```bash
cd FUBA; B=../BadFU_src/BadFU-main
# 시드 s=0,1,2 (42, 43, 44). --counts 는 EXPERIMENT_LOG 에 적힌 그 시드의 여섯 숫자. --polish 는 넣지 않는다 (본 방법 아님)
python cancel_fu.py --fmt badfu --name bd_badfu_s1 --traj_dir $B/logs/traj_full_s1 --data_root $B/data --badfu_record $B/record/badnet_dataset --rounds 15 --defender 5 --attackers 0 --target 0 --counts <장부의 여섯 숫자> --requesters 5 1 2 --op kd --out logs/cancel/bd_badfu_s1.json
python aggregate_cancel.py --cond badfu          # bd_badfu_s{0,1,2}.json → logs/cancel/summary_badfu.md
```
시드 0 은 지금 궤적(traj_full, 시드 42)으로 `--name bd_badfu_s0 --traj_dir $B/logs/traj_full` 로 다시 만든다 (파일이 없으므로). 시간: 시드당 궤적 1 시간 + cancel 30 분–1 시간. 궤적은 1 주차에 미리 돌린다.

**관문 4.** 위장자 5 purified 3/3 (첫 단계 경보), 운반자 [0] 3/3, ASR 27 → 10 이하, 정상 1·2 오탐 1/6 이하. 떨어지면 (예: 정상 1 의 마지막 조각 오탐이 반복) 숫자 그대로 보고하고 "마지막 조각 생략" 은 소거 실험으로만 (코드 수정 1 일 이내일 때만). ACC 76 은 연산자 열화라 그대로 쓴다 (정상 요청도 같은 ACC).

**FUBA 를 CIFAR 로 — 확인한 것.** FUBA 공격 코드는 `main.py --dataset cifar10` 이 있고 모델 resnet18, 트리거 생성기 UNet 을 쓴다. 원 저자 설정이라 궤적 생성은 될 가능성이 높다 (시험 안 함; DALI 는 imagenet 에만 쓰인다). 그러나 우리 분석 코드는 MNIST 고정이다: `release_map.load_fuba` 가 LeNet 과 MNISTAutoencoder 를 고정, `cancel_fu.py`·`dose_fu.py` 가 MNIST 분할을 고정, fuba 입력 공간이 1 채널 [0,1] 고정. 네 곳을 고쳐야 한다 (1–2 일). 거기에 ResNet 의 단계 제거 열화(BadFU 와 같은 ACC 76)가 그대로 따라온다. 판단: CIFAR-10 은 BadFU 로 이미 논문에 있다. FUBA-CIFAR 는 "있으면 좋은 것" 이다. 4 주차까지 Phase 0–3 이 끝났을 때만 한다 (§6 결정 3). 기본값은 안 하고 한계 절에 한 줄.

### Phase 5. 쓰기 (5–6 주차)
| 절 | 내용 | 먹이는 표 / 파일 | 주장 |
|---|---|---|---|
| 1 서론 | 기여 문장 = ZERO_BASE_REVIEW §8 그대로 | — | — |
| 2 위협 모델 | THREAT_MODEL.md §1–3 | — | 1 |
| 3 정확한 언러닝도 안 되는 이유 | T1 (빼기·증류·FedEraser; 재학습은 reduced·K=8 에서) | logs/benign_departure/summary_{iid,owner}_r100w5_paper.md (+ _k20, 부록) | 1, 2 |
| 4 방법 | ALGORITHM.md §3–6 (DOSE, TRACE, DEFUSE, 결정표) | — | — |
| 5.1 본 결과 | T1c 첫 표 (결정·ASR·ACC·재주입·비용), `paper` 만 | logs/cancel/summary_{iid,owner}_r100w5_paper.md | 3, 5, 6 |
| 5.2 운반자 찾기 | T1c 셋째 표 (활성화 위 짝), 둘째 표 (저장 기여 위) | 같은 파일 | 4, 8 |
| 5.3 BadFU | cancel 요약 (3 시드) | logs/cancel/summary_badfu.md | 1, 3, 4 |
| 5.4 비교군 | T2, T2b, 게이트 | logs/baselines/summary_*.md, gate_*.json, _median 등 | 9 |
| 5.5 적응형 | T4, 두 위장자 표 | logs/adaptive/summary_*.md, logs/cancel/summary_iid_sup2.md | 10 |
| 5.6 소거·비용 | T6, T5, 증류(_polish) 표 | logs/dose_fu/summary_*_{cap02,tau2,...}.md, summary_*_polish.md, logs/bench | 6, 7 |
| 6 한계 | 관문에서 떨어진 것 전부, 숫자 그대로 | RESULTS_LOG 의 '폐기'·'한계' 줄 | — |
5 주차: §1–4, §5.1–5.3 (그때 확정된 표만). 6 주차: §5.4–6, 그림 (구조도, ASR 곡선), 관련 연구, 초록. 초안 v2 를 지도교수에게. 표의 숫자마다 R-번호를 단다 (초안에만).

## 4. 주 단위 일정
| 주 | DISLAB | 로컬 노트북 | 문서 |
|---|---|---|---|
| 1 | Phase 0: `paper` 첫 궤적으로 시간 재기 → `paper` 6 개 시작. reduced·sup2 궤적 (싸다) + 검증 → EXPERIMENT_LOG. K=8 재집계 | BadFU 시드 43·44 궤적 + 검증. 공식 코드 가져오기 (third_party/) | RESULTS_LOG.md 틀 + 기존 결과 줄 (K=8, 예비 표시) |
| 2 | `paper` 6 개 완료 → Phase 1 → 관문 1. 집계 방어 궤적 12(+3) 개 시작 (2–3 주차에 걸쳐 돈다) | Phase 4 BadFU cancel 3 시드 → 관문 4 | §2, §4 (결과와 무관한 절) |
| 3 | 집계 방어 궤적 완료 → Phase 2 분석. Phase 3 (1)(2-a)(2-b) (reduced·sup2) | cancel_fu --sim, --sequential 만들기 (만들 것) | §3, §5.1–5.3 표 채움 (`paper` 숫자) |
| 4 | Phase 3 (2-c)(3), 미끼 보정 → 관문 3 → 결정 2. 여유가 있으면 결정 3 의 하나 | — | §5.4–5.5 표 채움 |
| 5 | 빠진 시드·재집계만 | — | §1, §5.6, §6. 숫자마다 R-번호 |
| 6 | — | — | 그림, 관련 연구, 초록. 초안 v2 → 지도교수. 결과 커밋·태그 |

## 5. 위험과 대응
| 위험 | 대응 |
|---|---|
| owner 소유자 오탐 1/3 (같은 시드, 점수 17) | K=20 에서 다시 본다. 남으면 한계 절에 숫자 그대로. 문턱과 통계를 더 바꾸지 않는다 |
| 두 진짜 위장자가 DOSE 와 TRACE 를 뚫음 | 관문 3 의 분기. 4 주차 안에 결정 2 |
| BadFU 연산자 열화 (ACC 76), 마지막 조각 오탐 | 열화는 연산자 한계로 보고 (정상 요청도 같은 ACC). 오탐은 "마지막 조각 생략" 소거만. 깊은 망 연산자 개선은 범위 밖 |
| K 가 작을 때 p 값 바닥 (K=8: 0.125, K=6: 0.167) | `paper` K=20 에서 바닥 0.05. p 값은 보조로만 쓰고 경보 점수를 주로 |
| `paper` 궤적 18 개 = DISLAB 2–4 일. 시간 부족 | 첫 궤적으로 시간을 잰다. 순서 고정: `paper` 6 개 → 집계 방어 12 개 → 적응형. 3 주차 끝까지 집계 방어가 안 끝나면 2 종(median, flame)으로 줄인다. K=8 이나 reduced 로 대신하지 않는다 (인용 불가). 밀리면 FUBA-CIFAR 와 INFOCOM SVD 부터 뺀다 |
| 결과 덮어쓰기와 궤적 바꿔치기 | (i) 이름에 K 가 없다 → K=20 은 반드시 TAG="_k20". (ii) `load_fuba` 는 태그 궤적이 없으면 조용히 기본 궤적(bd_iid_s0)을 쓴다 → 돌리기 전에 checkpoints 에 그 이름이 있는지 확인 (EXPERIMENT_LOG 줄). (iii) FORCE=1 은 결과 파일을 지운다 → 다른 설정은 다른 TAG. 세 규칙을 RESULTS_LOG 머리에 적고, 묶음마다 커밋 |

## 6. 사용자가 정할 것
1. **`paper` 3 시드가 시간 안에 안 될 때.** (A) 2 시드로 본문을 쓰고 셋째 시드는 투고 전까지 채운다 — 일정 유지, 표에 n=2 표시. (B) reduced K=20 을 주 설정으로 쓰고 논문 설정과 다름을 명시 — 하루면 되지만 리뷰어의 "설정이 왜 다른가" 질문이 남는다. K=8 로 돌아가는 선택지는 없다. 시점: 3 주차.
2. **두 위장자에서 떨어지면.** (A) 주장을 "위장자 한 명" 으로 좁히고 한계 절 — 2 일, 기여 문장 유지. (B) "요청 시점 방어의 한계" 중심으로 다시 쓰기 — 2 주 추가, 기여 문장이 바뀐다. 시점: 4 주차.
3. **4 주차에 여유가 생기면.** FUBA-CIFAR (코드 1–2 일 + 궤적 반나절, "같은 방법이 두 데이터셋에서") 또는 INFOCOM SVD 비교군 (2–3 일, "방향만 찾는 정화와 다르다" 를 실험으로). 둘 다 안 하면 한계 절 한 줄씩. 기본값: 둘 다 안 한다.

## 7. 하지 않을 것
- 방법의 이름과 틀을 다시 바꾸지 않는다. DOSE, TRACE, DEFUSE 와 기여 문장은 고정.
- 장부(EXPERIMENT_LOG, RESULTS_LOG)가 채워지기 전에 새 통계나 새 공격 변형을 만들지 않는다.
- RESULTS_LOG 에 줄을 쓰지 않고 실험을 돌리지 않는다. 줄 없는 숫자를 논문에 쓰지 않는다.
- 같은 이름 위에 결과를 덮어쓰지 않는다. TAG 없는 FORCE=1 은 쓰지 않는다. '확정' 줄의 파일은 다시 만들지 않는다.
- 관문에서 떨어진 실험을 설정을 바꿔 가며 다시 돌리지 않는다. 허용되는 것은 시드 추가 한 번뿐.
- 직교 증류(polish)를 본 방법에 되돌리지 않는다. 소거 표로만.
- 거름망(screen)을 판정에 쓰지 않는다.
- 1 시드 숫자를 본문 표에 넣지 않는다. BadFU 는 3 시드가 될 때까지 '예비'.
- K=8 과 reduced 의 숫자를 본문의 주 결과로 쓰지 않는다 (예비·부록). 집계 방어 비교는 `paper` K=20 에서만.
- 비교군을 직접 짜지 않는다. 공식 코드가 없는 방법은 표에 넣지 않고, proxy 를 그 방법의 이름으로 인용하지 않는다.
- 실험으로 보이지 않은 것을 쓰지 않는다 ("위장자 분산도 막는다", "모든 공격을 하나의 식으로" 등).
