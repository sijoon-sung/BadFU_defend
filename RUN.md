# 실험 돌리기 (DISLAB, 2026-10-12)

브랜치 `dev`. 모두 `FUBA` 폴더에서.

```bash
git fetch origin && git checkout dev && git pull
cd FUBA
```

## 1. 핵심 표 둘 — T1 (기본 FU 대 우리), T3 (쌍 탐지)

전에 돌린 결과 중 여러 단계짜리는 캐시 버그 때문에 믿을 수 없어서, 처음부터 다시 만든다. **`FORCE=1` 을 꼭 붙인다.** 안 붙이면 옛 결과를 그대로 건너뛴다.

```bash
FORCE=1 RQ="1 3" bash run_paper.sh
```

두 조건(iid, owner) × 3 시드. 끝나면 `docs/paper/tables_iid.md`, `docs/paper/tables_owner.md` 가 생긴다.

## 2. 시간이 되면 이어서 — T2 (비교군), T4 (적응형 공격)

```bash
FORCE=1 RQ="2 4" bash run_paper.sh
```

RQ2 는 비교군(MASA, SubID 식, 가중치 전용)과 학습 중 집계 방어 4 종 위에서 궤적을 다시 만드는 것이라 가장 오래 걸린다. 먼저 하나만 보려면:

```bash
FORCE=1 RQ="2" TDEFS="median" bash run_paper.sh
```

RQ4 는 적응형 공격 5 종(재학습 없음, 빠름).

## 3. 나머지 — T5 (비용), T6 (소거)

```bash
RQ="5 6" bash run_paper.sh
```

## 결과 올리기

```bash
cd .. && git add FUBA/logs docs/paper && git commit -m "results: RQ1-3 (or 2,4,5,6)" && git push
```

## 참고
- 일부만: `RQ="1"`, 조건 하나만: `CONDS="iid"`, 시드 하나만: `SEEDS="0"`, 명령만 보기: `DRY=1`.
- 단계 설명과 표 번호: `docs/paper/ALGORITHMS.md` 5 절. 결과 해석 전 점검 목록: `docs/paper/REVIEW_2026-10-11.md`.
- 로컬(GPU 노트북) 이어서 돌리기: `FUBA\resume_local.bat` (BadFU 쪽).
