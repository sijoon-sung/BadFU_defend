# 실험 돌리기 (DISLAB) — 2026-10-12 판

브랜치 `dev`. 모두 `FUBA` 폴더에서. 방법은 `docs/paper/METHOD.md`, 수식은 `docs/paper/MATH.md`.

```bash
git fetch origin && git checkout dev && git pull
cd FUBA
```

## A. 꼭 돌릴 것 (순서대로)

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
| _polish | 같은 파일, `summary_{cond}_polish.md` | ACC 가 KD(92) 쪽으로 돌아오면서 α 가 0 인가 |

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
