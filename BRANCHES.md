# 브랜치 (2026-10-12 정리: 셋만)

| 브랜치 | 용도 |
|---|---|
| `main` | 원래 기준점 (초기 탐색). 손대지 않음 |
| **`dev`** | **작업 브랜치.** 방법·문서·제안 기법 결과. 구조는 루트 `README.md` |
| `exp/baselines` | 기존 방법(비교군)만 자세히 돌린 결과. `FUBA/run_baselines.sh`, `docs/BASELINES.md` |

옛 브랜치 7 개는 지웠고 내용은 **태그**로 남아 있다 (`git tag -l`, 되살리기 `git push origin <태그>:refs/heads/<이름>`):
`archive-2026-10-11-exp-robust-fu`, `results-2026-10-11-interim`, `archive-2026-10-09-request-aware-purification`, `archive-2026-10-07-defense-pipeline`, `archive-2026-10-07-evaluation`, `archive-2026-10-07-weighted-removal`.

## 실행
- 전부: `cd FUBA && FORCE=1 RQ="1 3" bash run_paper.sh` (자세히는 `RUN.md`)
- 비교군만: `cd FUBA && bash run_baselines.sh`
- 로컬(GPU PC) BadFU 이어서: `FUBA\resume_local.bat`
