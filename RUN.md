# 실험 돌리기 (DISLAB) — 2026-10-13 저녁 판

브랜치 `dev`. 모두 `FUBA` 폴더에서. 본 방법은 균형 장부 (`docs/paper/DRAFT_v3.md` §5–6). 결과 번호는 `docs/RESULTS_LOG.md`.

```bash
git fetch origin && git checkout dev && git pull
cd FUBA
```

## 지금 돌릴 것 (이 순서대로; 각 줄이 끝나면 적힌 파일을 커밋)
| # | 무엇 | 명령 (FUBA 폴더) | 나오는 것 | 올릴 것 | 시간 |
|---|---|---|---|---|---|
| 1 | **큰 모델 궤적** CIFAR-10/resnet18, 논문 설정, IID 3 시드 | `ONLY=k20_paper_cifar bash make_trajectories.sh` | `checkpoints/*_bd_iid_r100w5_k20paper_cifar_s{0,1,2}_round_*.pkl`, `docs/EXPERIMENT_LOG.md` 행 | `git add FUBA/logs/trajectories docs/EXPERIMENT_LOG.md` | 미실측 (임시 300 분/궤적). 첫 궤적 검증이 "잠복 ACC < 50" 로 실패하면 멈추고 알릴 것 (train_epoch 변형 필요) |
| 2 | **공격 없는 궤적** mnist/cifar × iid/owner | `ONLY=clean_paper bash make_trajectories.sh` | `checkpoints/*_clean_{iid,owner}_r100_k20paper{,_cifar}_s0_round_*.pkl` | 같음 | mnist 45 분 × 2, cifar 미실측 × 2 |
| 3 | **장부 — CIFAR** (R-018) | `DATASET=cifar10 TAG=_r100w5_k20paper_cifar CONDS=iid bash run_ledger.sh` | `logs/ledger/summary_bd_iid_r100w5_k20paper_cifar.md` | `git add FUBA/logs/ledger` | 궤적당 5 분 |
| 4 | **장부 — 공격 없음** (R-015, 기대: 경보 없음) | `PREFIX=clean TAG=_r100_k20paper CONDS="iid owner" SEEDS=0 bash run_ledger.sh` 그리고 `PREFIX=clean TAG=_r100_k20paper_cifar CONDS="iid owner" SEEDS=0 DATASET=cifar10 bash run_ledger.sh` | `logs/ledger/summary_clean_*.md` | 같음 | 궤적당 2–5 분 |
| 5 | **대응 — 되감기 + 연합 제외, MNIST 6 궤적** (R-014) | A-03 "대응" 블록의 for 문 한 줄 | `logs/benign_departure/bd_*_excl_from*_s*.json` | `git add FUBA/logs/benign_departure` | 6 × 13 분 |
| 6 | **운반자 남긴 재학습, 논문 설정** (표 4 첫 행을 K=8 예비에서 바꿈) | `for c in iid owner; do for s in 0 1 2; do own=""; [ $c = owner ] && own="--non_iid --non_iid_type class_owner --owner_class 3 --owner_frac 0.9 --owner_client -1"; python benign_departure.py --name bd_${c}_r100w5_k20paper_s$s --K 20 --rounds 100 --seed $((522+s)) --split_seed $((522+s)) --defender 4 --requesters 4 --methods retrain_benign retrain_attack $own --out logs/benign_departure/bd_${c}_r100w5_k20paper_retrain_s$s.json; done; done` | `logs/benign_departure/bd_*_retrain_s*.json` | 같음 | 6 × 2 × 13 분 |
| 7 | **위장자 2 명** (R-016) | `for s in 0 1 2; do python ledger_defense.py --fmt fuba --name bd_iid_k16sup2_s$s --K 16 --rounds 8 --defender 4 --defenders 4 8 --attackers 0 1 2 3 5 6 7 9 --target 8 --neg_set --start 3 --cond sup2; done; python ledger_defense.py --aggregate --cond sup2` | `logs/ledger/summary_sup2.md` | 같음 | 10 분 |
| 8 | **집계 방어 위** (tdef_paper 12 궤적; `pip install hdbscan scikit-learn` 먼저) | `ONLY=tdef_paper bash make_trajectories.sh` 뒤 `for d in median multi_krum flame flDetector; do TAG=_r100w5_k20paper_$d CONDS=iid bash run_ledger.sh; done` | `logs/ledger/summary_bd_iid_r100w5_k20paper_*.md` | 같음 | 궤적 12 × 20 분 + 장부 |
| 9 (노트북) | **BadFU 40 라운드** (R-017 재확인) | `cd ../BadFU_src/BadFU-main && bash make_badfu_trajectories.sh` 뒤 `cd ../../FUBA && python ledger_defense.py --fmt badfu --name badfu_s42 --traj_dir ../BadFU_src/BadFU-main/logs/traj_s42 --data_root ../BadFU_src/BadFU-main/data --rounds 40 --K 6 --defender 5 --attackers 0 --target 0 --start 3 --min_present 1 --counts $(python -c "import json;e=[e for e in json.load(open('trajectories.json'))['trajectories'] if e['name']=='badfu_s42'][0];print(' '.join(map(str,e['counts'])))") --cond badfu` | `logs/ledger/badfu_s42.json` | 같음 | 2.5 시간 + 5 분 |
1–2 는 궤적, 3–4 가 리뷰가 요구한 두 숫자(규모, 오경보), 5–6 이 표 4, 7–9 가 넓힘이다. 자세한 설명은 아래 절 (0, 0-b, A-03, A-01).

## 0. 공격 궤적 미리 만들기 (한 번만, 다른 실험보다 먼저)

실험은 저장된 궤적 파일만 읽는다. 궤적은 저자 코드가 만든다 (FUBA 는 `main.py`, BadFU 는 저자 루프를 옮긴 `fl_detect_badfu.py`). 우리는 flag(클라 수·참여율·라운드·시드·방어)만 정한다.
만들 목록은 `FUBA/trajectories.json` 하나에 다 있다. 하나 만들 때마다 `verify_trajectories.py` 가 검사하고, 틀리면 거기서 멈춘다.
결과는 `docs/EXPERIMENT_LOG.md` 에 자동 기록된다 (`FUBA/logs/trajectories/registry.json` 과 같이 커밋).

### DISLAB (FUBA, MNIST)
```bash
cd FUBA
DRY=1 bash make_trajectories.sh                        # 명령만 보기 (아무것도 안 만듦)
ONLY=k20_reduced bash make_trajectories.sh             # K=20 축소판 (8 라운드) 6 개
ONLY=k20_paper   bash make_trajectories.sh             # 논문 설정 (K=20, 참여 0.4, 100 라운드) 6 개
ONLY=sup2        bash make_trajectories.sh             # 위장자 2 명 (K=16) 3 개
ONLY=tdef_paper  bash make_trajectories.sh             # 집계 방어 4 종 × 3 시드, 논문 설정. flame 은 먼저 pip install hdbscan
FORCE=1 ONLY=tdef_reduced bash make_trajectories.sh    # K=8 방어 12 개. 예전 것은 방어 없이 만들어졌을 수 있어 FORCE 로 갈아엎는다
bash make_trajectories.sh                              # 위 전부 (있는 것은 건너뜀). k8_main 6 개는 이미 있음
```
| 그룹 | 개수 | 개당 | 전체 |
|---|---|---|---|
| k8_main (있음) | 6 | 6 분 | — |
| k20_reduced | 6 | 10 분 | 1 시간 |
| k20_paper | 6 | 45 분 | 4.5 시간 |
| sup2 | 3 | 12 분 | 40 분 |
| tdef_paper | 12 | 50 분 | 10 시간 |
| tdef_reduced | 12 | 7 분 | 1.5 시간 |

시간은 노트북(RTX 4060) 실측과 논문 Table VI 로 잡은 추정이다. 끊기면 같은 명령을 다시 치면 된다 (다 만든 것은 건너뛴다).

### 노트북 (BadFU, CIFAR-10)
```bash
cd BadFU_src/BadFU-main
DRY=1 bash make_badfu_trajectories.sh
bash make_badfu_trajectories.sh      # 시드 42, 43, 44 를 40 라운드로. 시드당 약 2.5 시간, 디스크 14 GB. 기존 traj_full(15 라운드)은 그대로 둔다
```

### 확인
```bash
cd FUBA && python verify_trajectories.py        # 전부 다시 보기 (파일이 안 바뀌면 캐시라 빠름)
```
`docs/EXPERIMENT_LOG.md` 의 상태 열을 본다. ✔ 만 실험에 쓴다. ✘ 는 사유를 읽고 `FORCE=1 ONLY=이름 bash make_trajectories.sh` 로 다시 만든다. `(임시 문턱)` 이 붙은 그룹은 첫 결과를 보고 `trajectories.json` 의 verify 값을 고친다.
- 이름 규칙: K=8 `bd_iid_s0`, K=20 축소 `bd_iid_k20_s0`, 논문 `bd_iid_r100w5_k20paper_s0`, 방어 `bd_iid_median_s0` / `bd_iid_r100w5_k20paper_median_s0`, 위장자 2 명 `bd_iid_k16sup2_s0`. 이름이 다르면 다른 궤적이다 (없으면 에러, 기본 궤적으로 조용히 넘어가지 않는다).
- 뒤 단계에서 쓰는 법: `K=20 TAG=_k20 CONDS=iid STAGES="cancel aggregate" bash run_all.sh` (논문 설정은 `K=20 ROUNDS=100 WARMUP=5 TAG=_k20paper`). sup2 는 run_all.sh 의 고정 인자(`--attackers 0 1 2 3`)가 안 맞으니 `trajectories.json` 의 `downstream.args` 를 그대로 쓴다. 참여율 0.4 궤적은 결석 클라 처리가 된 로더(`release_map.load_fuba`)만 쓸 것 (`trajectories.json` 의 downstream.note).

## A. 꼭 돌릴 것 (순서대로)

### A-00. 탐지 총정리 벤치 (먼저, 궤적만 있으면 됨, 학습 없음) — "탐지가 잘 되는가" 를 모든 방법으로
한 궤적에서 생각할 수 있는 탐지 점수를 전부 클라이언트마다 매기고(모든 클라이언트를 '요청자라면' 으로), 위장자가 몇 등인지·정상/소유자가 잘못 걸리는지를 한 표로 낸다.
방법: 언러닝 시뮬레이션 절대 기준(기존 요청 심사 발상: 바뀐 답 비율·KL·몰림, ρ = 0.1/0.2/0.5) · 상대 기준(DOSE: flow·mass z) · 기능 효과(크기·몰림·상쇄·Fisher) · 갱신 통계(노름·중앙값 코사인·Krum·총노출 몫·반상관) · null-space 에너지 · 학습 중 부분합 감사(라운드 25/50/75/100 %) · 참여 자연 실험(참여율 < 1) · MASA/SubID proxy(결과 있으면).
```bash
bash run_detect_bench.sh                       # k8 + k20paper 중 있는 궤적 전부 (궤적당 1–3 분)
GROUPS="k20paper" bash run_detect_bench.sh     # 논문 설정만
GROUPS="tdef" bash run_detect_bench.sh         # 집계 방어 궤적 (만든 뒤)
```
결과: `logs/detect/summary_{iid,owner}_r100w5_k20paper.md` (+ `summary_iid.md`, `summary_owner.md`). 올리기: `git add FUBA/logs/detect`.
보는 것: 위장자 순위 1 이면서 소유자 z < 3 인 방법이 무엇인가. 기존 발상(sim_abs_*, upd_*, ns_energy)이 어디서 깨지고 상대 기준(sim_rel_*)·감사(audit_z)·참여 회귀(part_beta_*)가 어디서 갈리는가.

### A-02. 동적 평형 지문 (학습 중 시간 구조로 탐지; 학습 없음, 역할 정보 없음) — 궤적당 1–2 분
잠복 백도어는 매 라운드 유지되는 평형이고 유지에는 시간 위의 흔적이 남는다는 가설. 저장 전역 모델·갱신·참여 명단만으로 세 지문을 잰다: 1 복원력(결석 길이 → 밀기 세기), 2 지연 반작용(타깃 클래스만의 되밀기), 3 균형식(누적 타깃 밀기: 합은 0, 개별은 큼). 타깃 클래스도 서버가 추정한다.
```bash
for s in 0 1 2; do python analysis_equilibrium.py --fmt fuba --name bd_iid_r100w5_k20paper_s$s --K 20 --rounds 100 --defender 4 --attackers 0 1 2 3 --target 8 --cond iid_k20paper; done
for s in 0 1 2; do python analysis_equilibrium.py --fmt fuba --name bd_owner_r100w5_k20paper_s$s --K 20 --rounds 100 --defender 4 --attackers 0 1 2 3 --target 8 --owner_client 19 --cond owner_k20paper; done
python analysis_equilibrium.py --aggregate --cond iid_r100w5_k20paper
python analysis_equilibrium.py --aggregate --cond owner_r100w5_k20paper
```
결과: `logs/equilibrium/summary_{iid,owner}_r100w5_k20paper.md`. 스크립트에 **균형 장부**(클래스별 헤지 비율 H, 음의 다리 z, 양의 다리 집합, H 곡선)가 추가됐으니 같은 명령을 다시 돌리면 된다 (ALGORITHMIZATION.md §3). 보는 것: 지문 3 과 1 에서 위장자 순위 1·소유자 z < 3 인가, 추정 타깃이 8 인가, 운반자가 양의 누적 밀기로 위에 오는가. (노트북 1 시드: 지문 3 위장자 −40 vs 정상 ±4, z −22, 타깃 추정 8 맞음, 순/총 0.08; 지문 1 순위 1, z 3.7; 지문 2 는 구분 안 됨.) 올리기: `git add FUBA/logs/equilibrium`.

### 0-b. 큰 모델 궤적 (CIFAR-10 / resnet18) 과 공격 없는 궤적 — 리뷰 보완 (규모, 오경보)
저자 코드는 `--dataset cifar10` 을 지원한다 (분류기 resnet18, 생성기 UNet(3); config.py:146). 우리가 바꾼 것은 `run_real_fuba.sh` 의 `DATASET` 환경변수와 로더의 데이터셋 인식뿐이다. 공격 없는 궤적은 저자 코드의 warm-up 을 라운드 수보다 크게 두어 백도어 분기를 한 번도 타지 않게 한 것이다 (train_main.py:188). 매니페스트 그룹 `k20_paper_cifar` (IID 3 시드), `clean_paper` (mnist/cifar × iid/owner).
```bash
ONLY=k20_paper_cifar bash make_trajectories.sh        # CIFAR-10 논문 설정 3 시드 (시간 미실측; 임시 문턱 → 첫 궤적의 검증 결과를 보고 trajectories.json verify 를 고친다)
ONLY=clean_paper bash make_trajectories.sh            # 공격 없는 궤적 4 개 (mnist iid/owner ≈ 45 분씩, cifar 는 더)
DATASET=cifar10 TAG=_r100w5_k20paper_cifar CONDS=iid bash run_ledger.sh                       # 장부 (CIFAR)
PREFIX=clean TAG=_r100_k20paper CONDS="iid owner" SEEDS=0 bash run_ledger.sh                  # 장부 (공격 없음, mnist) → 기대: 경보 없음
PREFIX=clean TAG=_r100_k20paper_cifar CONDS="iid owner" SEEDS=0 DATASET=cifar10 bash run_ledger.sh
```
결과: `logs/ledger/summary_{bd_iid_r100w5_k20paper_cifar, clean_*}.md`. 먼저 볼 것: CIFAR 첫 궤적의 `docs/EXPERIMENT_LOG.md` 행 (잠복 ACC 가 50 아래면 `--train_epoch` 를 늘린 변형이 필요하다 — 저자 기본값은 라운드당 100 배치). 올리기: `git add FUBA/logs/ledger FUBA/logs/trajectories docs/EXPERIMENT_LOG.md`.
BadFU (CIFAR-10/ResNet-18, K=6, 운반자 1) 위의 장부: `python ledger_defense.py --fmt badfu --name badfu_s42 --traj_dir ../BadFU_src/BadFU-main/logs/traj_s42 --data_root ../BadFU_src/BadFU-main/data --rounds 40 --K 6 --defender 5 --attackers 0 --target 0 --start 3 --min_present 1 --counts <trajectories.json 의 counts>`. 노트북 traj_full(15 라운드)에서는 **경보 없음** (R-017): 타깃 클래스 0 의 H 0.51, 음의 다리가 정상 — BadFU 의 위장(트리거+원 라벨 표본)은 공개 입력에 새지 않거나 15 라운드로는 안 쌓인다. 운반자가 1 명이라 |Pos| ≥ 2 규칙도 못 쓴다. 40 라운드 시드로 다시 보고, 안 보이면 BadFU 는 요청 시점 빼 보기(R-004 의 dose, 첫 단계 z 8.4)로 가는 경계 사례로 쓴다.

### A-03. 균형 장부 방어 (학습 중 온라인 판정 → 경보 라운드 · 제외 목록; 학습 없음, 역할 정보는 채점에만) — 궤적당 1–2 분
A-02 의 관찰을 방어 절차로 만든 것 (ALGORITHMIZATION.md §3). 라운드마다 참여자 갱신이 각 클래스를 미는 양을 장부에 누적하고, 클래스마다 헤지 비율 H·음의 다리 z·양의 다리 집합을 계산해 "공모(양의 다리 ≥ 2) / 정상 큰손(1) / 외톨이 음(0)" 으로 판정, 같은 클래스가 3 번 연속 공모면 경보. 공모에는 '양의 다리 한 명의 몫 ≤ 0.6'(`--max_share`) 조건이 붙는다 (소유자 + 정상 한 명이 같이 걸리는 것을 막음). 경보 시점의 양의 다리가 제외 재학습(A-01) 의 `--exclude` 목록이 된다 (학습 끝 장부가 아니라 경보 시점 것을 쓴다: owner 시드 하나는 끝에 소유자가 양의 다리에 섞인다).
```bash
bash run_ledger.sh                      # iid + owner 3 시드, logs/ledger/{name}.json + summary_{cond}.md
EXTRA="--h0 0.9 --kappa 3" bash run_ledger.sh   # 문턱 민감도 (두 번째부터는 저장된 밀기로 순전파 없이)
```
보는 것: 요약 표의 경보 라운드 / 클래스 맞음 / 위장자 맞음 / 운반자 정밀도·재현율 / 소유자 오제외 / 오경보 클래스. (DISLAB 6 궤적 R-013: 경보 11–16/99, 6/6 정확, 오경보 0. 노트북 1 시드 IID: 경보 16/99 라운드, 클래스 8, 위장자 4, 운반자 0 1 2 3 정밀도·재현율 1.0, 오경보 0; h0 0.7–0.9, tau 3–5, persist 3–5 에서 같은 답, persist 1 은 7 라운드에 정상을 음의 다리로 잘못 봄 → 3 유지.) 올리기: `git add FUBA/logs/ledger` (npz 는 작으니 같이).
대응 (R-014; 장부가 낸 되감기 라운드 t_back 과 제외 목록을 그대로 쓴다; 저자 재학습 코드에 시작 모델만 `--init_round` 로 준다):
```bash
# 각 궤적의 t_back 은 logs/ledger/{name}.json 의 rollback_round (IID 10, 12, 7 / owner 13, 12, 7). 되감기 없이 경보 라운드에서 바로 제외하는 변형은 --init_round <경보 라운드>.
for c in iid owner; do for s in 0 1 2; do tb=$(python -c "import json;print(json.load(open('logs/ledger/bd_${c}_r100w5_k20paper_s$s.json'))['rollback_round'])"); own=""; [ $c = owner ] && own="--non_iid --non_iid_type class_owner --owner_class 3 --owner_frac 0.9 --owner_client -1"; python benign_departure.py --name bd_${c}_r100w5_k20paper_s$s --K 20 --rounds 100 --seed $((522+s)) --split_seed $((522+s)) --defender 4 --requesters 4 --methods retrain_benign --exclude 0 1 2 3 --init_round $tb $own --out logs/benign_departure/bd_${c}_r100w5_k20paper_excl_from${tb}_s$s.json; done; done
```
비교 기준은 R-010(처음부터 제외 재학습, ASR 0.5/0.6). 기대: 같은 ASR, 라운드 7–13 개 절약. 올리기: `git add FUBA/logs/benign_departure`.
오경보 검사 (공격 없는 궤적; 저자 코드의 warm-up 만 쓴다 = 백도어 분기 미실행, 새 코드 없음):
```bash
NB_CLIENTS=20 ROUNDS=100 WARMUP=1000 PARTICIPANT_RATE=0.4 EXTRA_ARGS="--split_seed 522 --participant_rate 0.4" bash run_real_fuba.sh clean_iid_r100_k20_s0 522
python ledger_defense.py --fmt fuba --name clean_iid_r100_k20_s0 --K 20 --rounds 100 --defender -1 --attackers --target -1 --cond clean_k20
```
기대: 경보 없음 (공모 클래스 0). 이것이 "정상 학습은 균형식을 만들지 않는다" 의 직접 증거다.

### A-01. 논문 설정에서의 처방 — RESULTS_LOG R-010 (운반자 제외 재학습). R-011(고친 해독제)은 노트북 확인에서 폐기 (DRAFT_v2_SKELETON §6.1) — 첫째 줄은 돌리지 않아도 된다
R-007 의 해독제 실패 원인 둘(정화 예산 2 회, 짝 집합 과대)을 고쳤다 (`--max_purify 0` = 단계마다 허용, `--set_consist on` = 모든 단계에서 정렬 ≤ −0.3 인 후보만). 그리고 저자 재학습 코드에 제외 목록만 붙였다 (`benign_departure.py --exclude`).
```bash
K=20 ROUNDS=100 WARMUP=5 TAG="_k20paper_v2" FORCE=1 STAGES="cancel aggregate" CONDS="iid owner" CANCEL_EXTRA="--rounds_eff off" bash run_all.sh
```
```bash
for c in iid owner; do for s in 0 1 2; do python benign_departure.py --name bd_${c}_r100w5_k20paper_s$s --K 20 --rounds 100 --seed $((522+s)) --split_seed $((522+s)) --defender 4 --requesters 4 --methods retrain_benign --exclude 0 1 2 3 --out logs/benign_departure/bd_${c}_r100w5_k20paper_excl_s$s.json; done; done
```
둘째 줄은 TRACE 가 찾은 운반자(0–3)를 요청자 4 와 함께 제외하고 남은 정직 클라로 재학습한다 (표 3 마지막 행). owner 는 `--non_iid --non_iid_type class_owner --owner_class 3 --owner_frac 0.9 --owner_client -1` 를 덧붙인다. 비교: 같은 명령에서 `--exclude` 없이 (= 재학습이 백도어 모델을 내는 쪽, retrain_attack 은 운반자 재참여).

### A-0. 주 설정 (논문 설정 K=20 궤적) 본 결과 — RESULTS_LOG R-007, R-008
궤적 6 개는 만들어져 검증됐다 (`docs/EXPERIMENT_LOG.md`, 위장자 제거 시 ASR 46–93). 이제 그 위에서 본 결과를 낸다. 전량 빼기는 붕괴하므로 DOSE 의 단계 제거가 필수이고, 빼기 연산자가 `partial` 로 끝나면 둘째 줄(단계 + 증류 연산자)이 본 결과가 된다.
```bash
K=20 ROUNDS=100 WARMUP=5 TAG="_k20paper" FORCE=1 STAGES="benign dose cancel effects aggregate" CONDS="iid owner" BENIGN_METHODS="subtract distillation fedEraser" CANCEL_EXTRA="--rounds_eff off" bash run_all.sh
```
```bash
K=20 ROUNDS=100 WARMUP=5 TAG="_k20paper_kd" FORCE=1 STAGES="dose cancel aggregate" CONDS="iid owner" DOSE_EXTRA="--op kd" CANCEL_EXTRA="--op kd --rounds_eff off" bash run_all.sh
```
| 결과 | 파일 | 관문 1 (PLAN.md) |
|---|---|---|
| 결정·ASR·ACC·α·비용 | `logs/cancel/summary_{iid,owner}_r100w5_k20paper{,_kd}.md` | 위장자 purified 3/3, 정화 뒤 ASR ≤ 정상 제거 수준 + 5, 정상 두 명 accept 6/6, owner 소유자(19) 오탐 ≤ 1/3 |
| 운반자 찾기 | 같은 파일 셋째 표 | 정밀도 ≥ 0.9 |
| FU 연산자 (빼기·증류·FedEraser) | `logs/benign_departure/summary_*_r100w5_k20paper.md` | 빼기가 붕괴하면 그대로 적는다 (주장 1 은 EXPERIMENT_LOG 의 작동점 값으로) |
주의: `--rounds_eff off` 는 라운드별 궤적 통계(보조)만 끈다. 참여율 0.4 궤적에서 결석 라운드 처리가 아직 라운드 단위 스크립트에 없기 때문이다. 끝나면 `git add FUBA/logs docs && git commit -m "results: R-007 R-008" && git push`.


### A0. 상쇄 짝 찾기 절차 (cancel, 본 방법 v3) — 두 조건 × 3 시드, 궤적은 있다고 보고
```bash
FORCE=1 STAGES="cancel aggregate" CONDS="iid owner" bash run_all.sh
FORCE=1 STAGES="cancel aggregate" CONDS="iid owner" TAG="_polish" CANCEL_EXTRA="--polish kd_orth" bash run_all.sh
```
| 결과 | 파일 | 보는 것 |
|---|---|---|
| 결정·ASR·ACC·α·비용 | `logs/cancel/summary_{cond}.md` 첫 표 | 위장자 4 = purified(antidote...), 정상 7·5 = accept(confirm). owner 의 7 이 accept 면 소유자 오탐 해결 |
| 1 단계 짝 (저장 기여) | 둘째 표 | 위장자의 크기·몰림 z 가 3 을 넘는가, 정상은 안 넘는가. 짝 정밀도는 IID 에서 1.0 이지만 정상도 운반자를 짝으로 내는 것이 정상(METHOD 2 절) |
| 2' 단계 짝 (활성화 위) | 셋째 표 | **핵심**: 위장자의 짝이 운반자 넷(정밀도·재현율 1.0), 정렬이 운반자 음수 / 정상 양수. owner 에서 소유자 7 이 짝에 안 들어오는가 |
| _polish (소거 실험, 본 방법 아님) | `summary_{cond}_polish.md` | 직교 증류의 'ACC 회복 vs 망각 되돌림' trade-off. 기본 교사 pop (지금 저장된 결과는 agree 교사; pop 으로 다시 만들려면 `CANCEL_EXTRA="--polish kd_orth --polish_ref pop"`). 본 방법 결과는 위 줄(증류 없음) |

### A1. 이전 판 (unified) + 기능 효과 행렬 (effects) — 이미 있음 (dev). 다시 만들려면
```bash
FORCE=1 STAGES="unified effects aggregate" CONDS="iid owner" bash run_all.sh
```
| 결과 | 파일 | 보는 것 |
|---|---|---|
| 요청자별 결정·ASR·ACC·α·비용 | `logs/unified/summary_{cond}.md` | 위장자 4 = purified(...), 정상 7·5 = accept(...). owner 의 7 이 accept 면 소유자 오탐 해결 |
| 효과 행렬: p 값, 상쇄 집합, 라운드 일관성 | `logs/effects/summary_{cond}.md` | 위장자의 상쇄 집합이 운반자(carrier)로 채워지는가, 몰림 클래스가 타깃(8)인가 |

### A2. 소유자 오탐 대책만 따로 (flow 통계, 짧음)
```bash
FORCE=1 STAGES="dose aggregate" CONDS="owner" TAG="_flow" DOSE_EXTRA="--stat flow" bash run_all.sh
```
`logs/dose_fu/summary_owner_flow.md` — 정상 7 경보 0/3 이면서 4 는 잡히는가.

### A3. 핵심 표 (기본 FU 대 우리, 운반자 찾기) — 이미 있음(dev). 다시 만들려면
```bash
FORCE=1 RQ="1 3" bash run_paper.sh
```

## B. 시간이 되면
```bash
FORCE=1 RQ="2 4" bash run_paper.sh          # 비교군(MASA, SubID 식, 집계 방어 4 종 위 궤적), 적응형 5 종
RQ="5 6" bash run_paper.sh                  # 비용, 소거 (cap, tau, m, stat, probe, op)
```
집계 방어 변형은 궤적을 다시 만들어 가장 오래 걸린다. 하나만 먼저: `FORCE=1 RQ="2" TDEFS="median" bash run_paper.sh`.

## C. BadFU (로컬 GPU 노트북, 궤적이 거기 있음)
```bash
B=../BadFU_src/BadFU-main
python cancel_fu.py  --fmt badfu --name loc_badfu_can --traj_dir $B/logs/traj_full --data_root $B/data --badfu_record $B/record/badnet_dataset --rounds 15 --defender 5 --attackers 0 --target 0 --counts 10545 9647 9641 9649 9668 10545 --requesters 5 1 2 --op kd --polish kd_orth
python unified_fu.py --fmt badfu --name loc_badfu_uni --traj_dir $B/logs/traj_full --data_root $B/data --badfu_record $B/record/badnet_dataset --rounds 15 --defender 5 --attackers 0 --target 0 --counts 10545 9647 9641 9649 9668 10545 --requesters 5 1 2 --op kd
python effects.py    --fmt badfu --name loc_badfu     --traj_dir $B/logs/traj_full --data_root $B/data --badfu_record $B/record/badnet_dataset --rounds 15 --defender 5 --attackers 0 --target 0 --counts 10545 9647 9641 9649 9668 10545 --requesters 5 1 2
```
보는 것: cancel 에서 5 의 1 단계 짝이 [0](운반자)·타깃 0 이고 2' 단계 짝도 [0] 인가, 정상 1·2 는 accept. (로컬 1 시드: 1 단계 짝 [0] 상쇄 0.19, 타깃 0 — 맞음.)

## 결과 올리기
```bash
cd .. && git add FUBA/logs docs/paper && git commit -m "results: ..." && git push
```

## 참고
- 일부만: `RQ="1"`, 조건 하나만: `CONDS="iid"`, 시드 하나만: `SEEDS="0"`, 명령만 보기: `DRY=1`.
- 표 만들기: `python paper_tables.py --cond iid` → `docs/paper/tables_iid.md` (T1, T1c, T1u, T1e, T2, T3, T4, T5, T6).
- 결정 뜻: accept(confirm)=빼 보고 수락 / accept(attr_suspect)=1 단계 경보지만 빼 보기 통과 / accept(partial)=연산자 한계로 일부만 뺌 / purified(antidote[+polish])=짝의 상쇄 성분 제거[+직교 증류] / purified(kd)=재억제(대체) / (sim_only)=1 단계 통과·빼 보기 경보 / reject(...)=불출.
