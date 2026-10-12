# docs/

`EXPERIMENT_LOG.md` — 실험 기록 (공격 궤적 표): `FUBA/verify_trajectories.py` 가 자동으로 쓴다. 어느 궤적이 있고 검증을 통과했는지, 잠복/제거 뒤 ACC·ASR, 남은 생성 시간. 손으로 고치지 말 것.

## paper/ — 논문용 (읽는 순서)
1. `THREAT_MODEL.md` — 공격의 틀: 상쇄형 백도어, 해로운 삭제 요청, 왜 기존 방어가 못 보나, 지표
2. `METHOD.md` — 방법: A 거름망(궤적 지표) → B 확정(빼 보기) → C 정화(해독제 방향). 성질, 코드 상태
3. `POSITIONING.md` — 자리매김: 문제가 새롭고 도구가 표준일 때 기여를 어디에 두나
4. `EXPERIMENT_DESIGN.md` — 비교군 선정 근거, RQ1–RQ6 → 표 → 스크립트 (2–3 절은 METHOD 로 대체)
5. `REPORT_2026-10-12.md` — 지금까지의 실험 결과 보고
6. `REVIEW_2026-10-11.md` — 리뷰어 관점 코드 점검, 고친 것과 남은 것
7. `DRAFT_OUTLINE.md` — 초안 기획 (제목, 첫 문단, 절 구성, 남은 작업)
8. `tables_iid.md`, `tables_owner.md` — `paper_tables.py` 가 만든 결과 표

## paper_reviews/ — 선행 논문 정리
BadFU, FUBA, FedMUA, Malicious Forgetting(SubID), 비교표.

## archive/ — 지나간 것 (참고용, 갱신 안 함)
- `design/` — 설계 메모들: DOSE_FU, PAIR_ROLES, ATTACK_VARIANTS, DEFENSE_LANDSCAPE, UNIFIED_LOWCOST, EXPERIMENTS, README(한 장 요약 옛 판). 내용은 METHOD/THREAT_MODEL/EXPERIMENT_DESIGN 에 흡수됨
- `fuba/` — FUBA 쪽 옛 기록: BENIGN_DEPARTURE, ROBUST_FU, LATENT_RISK, LOCAL_SMOKE
