# 브랜치 안내 (2026-10-11 밤, 둘로 정리)

| 브랜치 | 용도 |
|---|---|
| `main` | 원래 기준점 (det1~3 결과, 초기 파이프라인). 손대지 않음 |
| `exp/baselines` | 기존 방법(비교군)만 자세히 돌리는 브랜치. `FUBA/run_baselines.sh`, 안내 `docs/BASELINES.md`. 결과를 여기에 쌓는다 |
| **`dev`** | **작업 브랜치.** 지금까지의 모든 것: 결과(정상 이탈 검증, FedEraser-R, 잠재 위험 지도, 빼 보기, 진행형 FU, 시드 1·2, 비용 벤치) + 제안 기법(투여량 조절 FU, 쌍→집합) + 기준선(MASA, SubID 식) + 적응형 공격 + 설계 문서 `docs/design/` |

옛 브랜치는 **태그**로 보관 (브랜치 목록에는 안 보임, `git tag -l` 로 확인, `git checkout <태그>` 로 열람):

| 태그 | 원래 브랜치 | 내용 |
|---|---|---|
| `archive-2026-10-11-exp-robust-fu` | exp/robust-fu | dev 로 합쳐지기 직전의 결과 브랜치 (dev 와 내용 동일) |
| `results-2026-10-11-interim` | results/interim-2026-10-11 | 로컬 GPU 중간 결과 스냅샷 (`FUBA/logs/INTERIM_RESULTS.md`) |
| `archive-2026-10-09-request-aware-purification` | archive/request-aware-purification | BadFU CIFAR 관측 실험 |
| `archive-2026-10-07-defense-pipeline` | archive/defense-pipeline | 다중 시드 러너 |
| `archive-2026-10-07-evaluation` | archive/evaluation | 평가 설계, 탐지 오류 스윕 |
| `archive-2026-10-07-weighted-removal` | archive/weighted-removal | 통째 제거 ↔ 방향 제거 가중 스윕 |

되살리기: `git push origin <태그>:refs/heads/<원하는 이름>`

## 실행
- 전부: `cd FUBA && STAGES="dose masa subid pair adaptive aggregate" CONDS="owner iid" bash run_all.sh`
- 로컬(GPU PC) 이어서: `FUBA\resume_local.bat`
- 한 장 요약: `docs/design/README.md`
