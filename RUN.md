# 실험 돌리기 (DISLAB) — 2026-10-12 판

브랜치 `dev`. 모두 `FUBA` 폴더에서. 방법은 `docs/paper/METHOD.md`, 수식은 `docs/paper/MATH.md`.

```bash
git fetch origin && git checkout dev && git pull
cd FUBA
```

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
