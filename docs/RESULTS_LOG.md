# 분석 장부 (RESULTS_LOG) — 실행 한 번에 한 줄

규칙 (PLAN.md §2):
1. 궤적 열은 `docs/EXPERIMENT_LOG.md` 의 이름 그대로. 거기 없는 궤적으로는 돌리지 않는다.
2. 상태는 넷 중 하나: 예정 / 돌아감 / 확정 / 폐기(이유). 파일이 저장소에 없으면 '파일 없음'.
3. 설정이 다르면 TAG 가 다르다. 같은 이름 위에 FORCE=1 로 덮어쓰지 않는다.
4. 논문 표의 숫자마다 번호(R-xxx)를 단다. 번호 없는 숫자는 뺀다.
5. 실행 묶음이 끝나면 바로 커밋: `git add FUBA/logs docs && git commit -m "results: R-xxx ..."`.

주 설정 = FUBA 논문 설정 (K=20, 악성 5, 100 라운드, 참여 0.4) = 궤적 `bd_{iid,owner}_r100w5_k20paper_s{0,1,2}`. K=8 결과는 예비(부록).

| 번호 | 날짜 | 궤적 | 스크립트+단계 | 인자/TAG | 결과 파일 | 답하는 주장 | 상태 |
|---|---|---|---|---|---|---|---|
| R-001 | 2026-10-12 | bd_iid_s0, s1, s2 (K=8) | run_all.sh cancel (cancel_fu.py) | 기본값, TAG 없음, dev 93416d5 | FUBA/logs/cancel/bd_iid_s{0,1,2}.json → summary_iid.md | 주장 3·4·5·6 (IID) | 확정 (예비, K=8) |
| R-002 | 2026-10-12 | bd_owner_s0, s1, s2 (K=8) | run_all.sh cancel | 기본값 | FUBA/logs/cancel/bd_owner_s{0,1,2}.json → summary_owner.md | 주장 3·4·5·6 (owner); 정상 7 오탐 1/3 (seed 2, z 17) | 확정 (예비, K=8), 오탐은 한계로 보고 |
| R-003 | 2026-10-12 | bd_{iid,owner}_s{0,1,2} (K=8) | run_all.sh cancel, TAG=_polish | `--polish kd_orth` (pop → agree 교사) | FUBA/logs/cancel/bd_*_polish_s*.json → summary_*_polish.md | 주장 7 (α=0 이어도 되돌아옴: 자기 데이터 acc 50 → 79 / 75) | 확정 (소거 실험; 증류는 본 방법 아님) |
| R-004 | 2026-10-12 | loc_badfu (시드 42, 15 라운드) | cancel_fu.py --fmt badfu --op kd | 로컬 노트북 | REPORT §14.5 숫자만 — 결과 파일 저장소에 없음 | 주장 1·3·4 (BadFU) | 파일 없음 → Phase 4 에서 bd_badfu_s0 으로 다시 |
| R-005 | 2026-10-12 | loc_fuba (K=8, 로컬) | analysis_update_stats.py, analysis_hedge.py, baseline_gate.py | — | FUBA/logs/analysis/update_stats_loc_fuba.json, hedge_loc_fuba.json, FUBA/logs/baselines/gate_loc_fuba.json | 주장 9 (집계 방어가 보는 것; 헤지 가설 기각; null-space 게이트 무력) | 확정 (1 시드, 과반 설정 — 기전 근거로만) |
| R-006 | 2026-10-12 | bd_{iid,owner}_r100w5_k20paper_s{0,1,2} | make_trajectories.sh (저자 main.py) + verify_trajectories.py | ONLY=k20_paper, dev f8b1079 | docs/EXPERIMENT_LOG.md, FUBA/logs/trajectories/registry.json | 관문 0: 잠복 ASR 3–8, 위장자 제거(작동점 ρ=0.2) ASR 46–93, 정상 제거 ASR 2–10, 운반자 제거 ASR ≤ 1.4 | 확정 (궤적 6 개 ✔; 전량 빼기는 붕괴 → 단계 제거 필수) |
| R-007 | 2026-10-12 | bd_{iid,owner}_r100w5_k20paper_s{0,1,2} | run_all.sh benign dose cancel effects aggregate | K=20 ROUNDS=100 WARMUP=5 TAG=_k20paper, BENIGN_METHODS="subtract distillation fedEraser", CANCEL_EXTRA="--rounds_eff off" | FUBA/logs/{benign_departure,dose_fu,cancel,effects}/bd_*_r100w5_k20paper_s*.json → summary_*_r100w5_k20paper.md | 주장 1–6, 8 (주 설정) | subtract: 전량 붕괴(ACC 9.7%), fedEraser: ASR 0.8~1.1%, TRACE: 운반자 재현율 1.0(4/4), cancel: 위장자 reject(purify_failed) 3/3 (정화예산 바닥남 + 짝 과대), 정상 accept 5/6, benign 완료 |
| R-008 | 2026-10-12 | 같은 궤적 | run_all.sh dose cancel aggregate | TAG=_k20paper_kd, DOSE_EXTRA="--op kd", CANCEL_EXTRA="--op kd --rounds_eff off" | FUBA/logs/{dose_fu,cancel}/bd_*_k20paper_kd_s*.json → summary_*_k20paper_kd.md | 주장 5 (깊은 궤적에서 단계+증류 연산자) | 위장자: 최종 ASR 3.4–8.8, ACC 95.5–96.6, α 0.08–0.11 (재억제). 단 결정 3/6 reject(purify_failed)(예산 초과). 정상·소유자 accept 6/6 |
| R-009 | 2026-10-12 | k8, k20paper, tdef_reduced (18 궤적) | run_detect_bench.sh (detect_bench.py) | 전 점수 비교 (절대/상대 언러닝, 효과, 갱신통계, 감사, 참여회귀) | FUBA/logs/detect/*.json → summary_*.md | BREAKTHROUGH: 상대 기준(flow z, audit, fisher, part_beta) 위장자 1등 & 소유자 z<3 달성 | 확정: BREAKTHROUGH 핵심 기전 전 설정 입증 |
| R-010 | 2026-10-13 | bd_{iid,owner}_r100w5_k20paper_s{0,1,2} | benign_departure.py retrain_benign --exclude 0 1 2 3 | TAG=_k20paper_excl | logs/benign_departure/bd_*_r100w5_k20paper_excl_s*.json → summary.md | 주장 B: 운반자 제외 재학습이 정답 모델(ASR ≈ 1, ACC ≈ 98)에 닿는가 | 확정 (표 3 완성): IID ASR 0.5 ± 0.1 (ACC 99.0 ± 0.0), owner ASR 0.6 ± 0.2 (ACC 98.9 ± 0.0). 백도어 완전 소멸 및 원본 정확도(99%) 달성 |
| R-011 | 2026-10-13 | loc_k20paper_s0 (노트북, 논문 설정 1 시드) | cancel_fu.py (예산·단계 일관성·eps2 절대값 고친 뒤) | --rounds_eff off, --eps2 0.2 / 0.5 | smoke_bak (저장소 밖) | 주장 B 의 저가 처방(해독제)이 논문 설정에서 되는가 | **폐기**: 여전히 reject. 활성화 입력 위에서 모든 클라가 반대 정렬 (z 99) → 투영 제거 범위 밖. 처방은 재억제·제외 재학습 (DRAFT_v2 §6.1) |
| R-012 | 2026-10-13 | bd_{iid,owner}_r100w5_k20paper_s{0,1,2} | run_a02.sh (analysis_equilibrium.py) | K=20, rounds=100, defender=4, attackers=0 1 2 3, target=8 | FUBA/logs/equilibrium/bd_*_r100w5_k20paper_s*.json → summary_*_r100w5_k20paper.md | 주장 A·C (학습 중 동적 평형 지문: 전역모델·갱신·참여명단만으로 시간 구조 탐지) | 확정: 지문 3(누적 타깃 밀기) 위장자 순위 1.0 (3/3 잡힘, IID z=24.5±3.0, owner z=11.3±1.8). 소유자 z=-11.4(반대 방향)로 오탐 0. IID 타깃 추정 3/3 정답(클래스 8), 순/총 밀기 비율 0.07~0.12로 평형 입증 |

