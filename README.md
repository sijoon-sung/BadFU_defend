# BadFU_defend — 삭제 요청으로 깨어나는 백도어에 대한 연합 언러닝 방어

연합 언러닝(FU)에서 삭제 요청으로 활성화되는 잠복 백도어(BadFU, FUBA)를 서버가 요청 시점에 가려내고, 심은 쪽의 성분만 지우는 방법을 만든다.

## 어디부터 읽나
| 문서 | 내용 |
|---|---|
| [docs/paper/METHOD.md](docs/paper/METHOD.md) | **지금 방법** (거름망 → 빼 보기 → 해독제 정화) |
| [docs/paper/THREAT_MODEL.md](docs/paper/THREAT_MODEL.md) | 공격의 틀 (상쇄형 백도어, 해로운 삭제 요청) |
| [docs/paper/REPORT_2026-10-12.md](docs/paper/REPORT_2026-10-12.md) | 지금까지의 실험 결과 보고 |
| [docs/paper/DRAFT_OUTLINE.md](docs/paper/DRAFT_OUTLINE.md) | 논문 초안 기획 |
| [RUN.md](RUN.md) | 실험 돌리는 법 |
| [BRANCHES.md](BRANCHES.md) | 브랜치 |

## 폴더
| 폴더 | 내용 |
|---|---|
| `FUBA/` | 실험 코드 (FUBA 공격 코드 + 우리 방어·비교군·러너). 스크립트 지도는 `FUBA/README.md` |
| `BadFU_src/` | BadFU 공격 코드 (로컬 궤적은 gitignore) |
| `docs/paper/` | 논문용 문서 (방법, 위협 모델, 자리매김, 실험 설계, 보고, 리뷰, 표) |
| `docs/paper_reviews/` | 선행 논문 정리 4 편 |
| `docs/archive/` | 지나간 설계 문서 (design/) 와 FUBA 쪽 옛 기록 (fuba/) |
| `legacy/` | 초기 탐색 코드·결과 (main 시절). 참고용 |

## 한 줄 실행
```bash
cd FUBA && FORCE=1 RQ="1 3" bash run_paper.sh
```
