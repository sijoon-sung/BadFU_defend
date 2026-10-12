# 비교군(기존 방법) 실험 — 브랜치 `exp/baselines`

제안 기법 없이 기존 방법만 자세히 돌리고 결과를 이 브랜치에 쌓는다. 돌리기: `cd FUBA && bash run_baselines.sh` (묶음 선택은 `GROUP=`).

| 묶음 | 방법 | 무엇을 보이나 | 설정 | 결과 위치 |
|---|---|---|---|---|
| fu | 빼기, 증류 FU, FedEraser(원식 보정), 재학습(정상 재참여 / 공격자 재참여) | 그냥 언러닝하면 깨어나는가. 정상 요청과의 대비. 정의 2 의 (망각) 기준 | 요청자 4(위장자), 7, 5(정상) | logs/benign_departure |
| robust | FedEraser-R (이전 제안) | 보정 안에 탐지+정화를 넣은 이전 안의 성능 | arms plain / purify_flag / always / clean | logs/robust_fu |
| detect | MASA (학습 중 언러닝 손실), SubID 식 (부분공간 잔차), 가중치 전용 지도 | 학습 중·갱신 기하로 위장자를 잡을 수 있는가 | 운반자 0–3, 위장자 4 | logs/baseline_masa, logs/baseline_subid, logs/latent_risk |
| tdef | Median, Multi-Krum, FLAME, FLDetector 위에서 같은 공격 → FU | 학습 중 집계 방어가 활성화를 막는가 | 궤적 재생성 (이름 `bd_{cond}_{def}_s{seed}`) | logs/benign_departure (TAG) |
| prev | 빼 보기(ρ 스윕), 진행형 FU | 이전 탐지 판의 재현 | 요청자 4 / 7 / 5 | logs/release_map, logs/progressive_fu |

요약 표: 각 묶음의 `aggregate_*.py --cond {iid,owner}` 가 `logs/*/summary_{cond}.md` 를 만든다. 전체 종합은 dev 의 `FUBA/paper_tables.py` (T1, T2, T2b).

주의
- 전에 돌린 결과 중 여러 단계짜리(progressive)는 캐시 버그 영향이 있었다. `FORCE=1` 로 다시 만든다.
- MASA·SubID 는 원 논문의 재구현이 아니라 **proxy** 다 (docs/paper/REVIEW_2026-10-11.md #7, #8). 표에 그렇게 적는다.
- tdef 는 궤적을 조건·시드마다 4 번 더 만들므로 가장 오래 걸린다. `TDEFS="median"` 으로 하나만 먼저.
