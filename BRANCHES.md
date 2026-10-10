# 브랜치 안내 (2026-10-11 정리)

| 브랜치 | 용도 | 상태 |
|---|---|---|
| `main` | 원래 기준점 (det1~3 결과, 초기 파이프라인) | 손대지 않음. 최신 코드는 아래 브랜치 |
| **`exp/robust-fu`** | **현재 작업 브랜치.** 정상 이탈 검증, 원식 FedEraser, 잠재 위험 지도, FedEraser-R, 빼 보기(release map), 진행형 FU, `run_all.sh`, 논문 정리(`docs/paper_reviews/`), 로컬 실행 기록(`FUBA/LOCAL_SMOKE.md`) | 활성 |
| `results/interim-2026-10-11` | 로컬 GPU 중간 결과 스냅샷 (JSON·로그). 설명: `FUBA/logs/INTERIM_RESULTS.md` | 결과 전용, 코드 변경 없음 |
| `archive/request-aware-purification` | (구 `codex/request-aware-purification`) BadFU CIFAR 관측 실험, 회의 PDF | 보관 |
| `archive/defense-pipeline` | (구 `defense-pipeline`) 다중 시드 러너, seed-1 결과 | 보관 |
| `archive/evaluation` | (구 `exp/evaluation`) 평가 설계, 실제 FU 결과(run1), 탐지 오류 스윕 | 보관 |
| `archive/weighted-removal` | (구 `exp/weighted-removal`) 통째 제거 ↔ 방향 제거 가중 스윕 | 보관 |

삭제한 브랜치(커밋은 전부 `exp/robust-fu` 역사에 포함되어 있어 손실 없음): `chatgpt`, `exp/benign-departure`, `exp/diff-audit`.

## 이어서 실행
- 로컬(GPU PC): `FUBA\resume_local.bat` — 중단된 단계만 이어서. 끝난 단계는 건너뜀.
- DISLAB: `cd FUBA && STAGES="release progressive aggregate" CONDS="owner iid" bash run_all.sh`

## 되돌리기
archive 이동은 이름만 바꾼 것이라 `git push origin origin/archive/<name>:refs/heads/<old-name>` 으로 복구할 수 있다.
