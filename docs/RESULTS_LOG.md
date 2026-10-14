# 분석 장부 (RESULTS_LOG) — 실행 한 번에 한 줄

규칙 (PLAN.md §2):
1. 궤적 열은 `docs/EXPERIMENT_LOG.md` 의 이름 그대로. 로컬 공동삭제 감사(R020 이후)는 별도 출처 등록 `docs/EXPERIMENT_LOG_PAIRED.md`를 사용한다. 출처 없는 궤적을 기존 논문 궤적으로 대체하지 않는다.
2. 상태는 넷 중 하나: 예정 / 돌아감 / 확정 / 폐기(이유). 파일이 저장소에 없으면 '파일 없음'.
3. 설정이 다르면 TAG 가 다르다. 같은 이름 위에 FORCE=1 로 덮어쓰지 않는다.
4. 논문 표의 숫자마다 번호(R-xxx)를 단다. 번호 없는 숫자는 뺀다.
5. 실행 묶음이 끝나면 바로 커밋: `git add FUBA/logs docs && git commit -m "results: R-xxx ..."`.

주 설정 = FUBA 논문 설정 (K=20, 악성 5, 100 라운드, 참여 0.4) = 궤적 `bd_{iid,owner}_r100w5_k20paper_s{0,1,2}`. K=8 결과는 예비(부록).

## 탐지 우선 결과 판독 (2026-10-14)

| 번호 | 자료 | 실행 / 파일 | 결과와 범위 | 상태 |
|---|---|---|---|---|
| R-051 | 합성 점수·정상 보정 fixture, 실제 명령행 입출력 | `python -m unittest discover -s FUBA/tests -p test_report_detection.py -v`; `FUBA/report_detection.py` | 새 검사7개 통과. 잘못된 요청의 경보를 정답 탐지에서 제외, 개발/시험 분리, 일시/마지막 경보 분리, 보정 부족=null, 정확 이항 구간, 파일/차원/요청 가족 일치, JSON만으로 실행 및 덮어쓰기 거부. 새 학습·FU·성능 결과 아님 | 확정 (평가 동작) |

사용자는 실험이 실행 중이라고 알렸다. 여기서 외부 작업의 진행률이나 결과를 확인한 것은 아니며 특정 실험 번호의 완료 상태를 바꾸지 않는다. 기존 점수/학습/수집/제어와 등록 manifest 해시는 변경하지 않고 별도 판독기를 추가했다. 탐지 가설과 실제 FU 활성화 정답의 구분은 `docs/paper/TENSION_CONTROL.md` §8, 실행은 `RUN.md` 첫 절에 있다.

## BadFU 별도 보존과 자동 FU 비교 (2026-10-14)

| 번호 | 자료 | 실행 / 파일 | 결과와 범위 | 상태 |
|---|---|---|---|---|
| R-047 | 합성 입력·BN 포함 작은 모델 | test_badfu_request_pipeline.py; 전체113개 unittest | 저장 데이터 동일성·진짜 초기값·legacy import·해시 변조/덮어쓰기 거부·두 요청×3군의 실제2단계 replay·실패군 모델 미출력·dense/청크 해 일치 | 확정 (동작) |
| R-048 | 실제 CIFAR 자료, 무사전학습 ResNet18, seed42, 6client×32표본, 2라운드×1배치 | artifacts/badfu/pipeline_smoke_s42; logs/badfu_request/pipeline_smoke_s42*.json | 초기값·실제 데이터·2개 FL 궤적 저장 및 이어지는 FU 자동 실행. 공격이 학습된 설정이 아니므로 ASR을 효과 근거로 쓰지 않음. eval_fixed가 평가 모드 복원 후 재사용 검사 | 확정 (저장/실행) |
| R-049 | badfu_traj_full_local15, 실제15번째 FL 기록의 보정1단계(0-based14) | logs/badfu_request/traj_full_step14_eval_fixed.json | 요청5: 무방어 ASR91.02/ACC75.98%, temporal45.12/86.13%, retention_only 실제 손실/진행 제약 불충족. temporal 수용 진행.12436·최대 클래스 CE증가.008355. 정상요청1: 무방어74.02/76.56%, 두 제어군 모두 불수락. 독립 시드/전체 FU/완전 방어 아님 | 확정 (개발1단계) |
| R-050 | 같은 실제 궤적·초기 구현 | traj_full_step14.json, traj_full_step14_frozen_bn.json, traj_full_step14_deterministic.json | candidate별 BN 재구성 초기 정책 또는 보정 후 eval 복원 전 코드로 실행. 초기 실패 수치를 알고리즘 성능 근거로 사용하지 않음. 원본 디버깅 기록은 덮어쓰지 않음 | 폐기 (성능 근거; 개발 기록 보존) |

R-049는 클라이언트별 전체 벡터 FedEraser 보정(보존 client당2배치), 이력 창3, η=.1, ε=.01, γ=.1, ρ=.25‖v‖, τ=1, 검증80개(클래스당8), clean/trigger 평가 각각512개다. BN은 각 단계 시작에 고정 정상 검증 입력으로 재구성한 뒤 제어 중에는 eval로 고정했다. 같은 단계의 저장 시작 모델 ASR21.88%, BN만 재구성하면36.72%, temporal 수용 후45.12%이므로 **원래 ASR 수준으로 복원한 것이 아니다**. 삭제 전 최종15라운드 모델의 ASR24.02%는 이 단계의 시작 모델과 다르다. 정상 요청의 제어 불수락과 retention_only 불수락도 실패로 집계하며, 성공군만의 ASR 비교로 신규성·실용성을 입증하지 않는다. 정규화만의 효과·실패 시 실제 손실 시도·진행량·수치 최적화 진단을 JSON에 저장했다.

legacy 데이터는 원래 seed42/dominant.7/전체표본 설정으로 재구성했고 샘플 수 가중 집계 일치를 검사했다. 원본 데이터 해시가 없으므로 완전 동일성은 미인증이다. 새 생성 경로에서는 FL/FU가 동일 data.pt를 읽는다. 큰 자료는 로컬 artifacts 폴더에 보존하고 실행 명세/작은 결과만 GitHub에 기록한다. 40라운드 새 공격 및 전체15라운드 공격 FU 비교는 이번에 실행하지 않았다.

## 요청 조건부 상쇄 연산자 개정 (2026-10-14)

아래 실제 단계 실험은 모두 같은 loc_k20paper_s0/99단계의 개발 검사다. 요청을 바꾼 것이 독립 학습 시드 추가가 아니다. 원시 업데이트의 연산자가 수치적으로 제어되는 것과 ASR 추가 이득을 구분한다. 동시 실행 시간은 비용 비교에 사용하지 않는다.

| 번호 | 자료 | 실행 / 파일 | 결과와 범위 | 상태 |
|---|---|---|---|---|
| R-040 | 해석적·행렬 동작 검사 | test_request_geometry.py의17개 검사 | full-matrix PSD 해와 저차원 구현 일치, 균일 삭제 소거, 회전·배율 등변, 시간 순서 민감도, 해석적 QCQP 해, 진행·위험·실제 손실 제약, replay callback | 확정 (동작) |
| R-041 | loc_k20paper_s0/99, 요청4 | logs/request_geometry/loc_k20paper_s0_step99_req4.json | λ=1/10 고정 벌점 탐색. ASR의 추가 이득 없음. 후속 상대 노출 제약을 정한 개발 과정으로 보존 | 확정 (개발) |
| R-042 | 같은 모델·단계, 요청4 | loc_k20paper_s0_step99_req4_bounded.json | 상대 노출.1 도입. 일부 SLSQP가 작은 가능성 오차에도 종료 실패를 보고. 다음 실행에서 KKT·목적 오차 확인을 추가; 이 기록을 수정/덮어쓰지 않음 | 확정 (solver 개발) |
| R-043 | 같은 모델·단계, 정상 요청8 | loc_k20paper_s0_step99_req8_bounded.json | 같은 예산의 정상 요청 동작. 실제 손실 검사에서 한 번 축소. 초기 solver 버전의 기록 | 확정 (개발) |
| R-044 | 같은 모델·단계, 요청4 | loc_k20paper_s0_step99_req4_certified_solver.json | η=.1, ε=.01, γ=.1, ρ=.25‖v‖, τ=1. 무방어 ASR20.74/ACC88.67%; temporal 및 같은 보존제어 ASR11.14/ACC99.02%. 정의한 노출 제곱3.024e-4→3.024e-6, 실제 최대 클래스 손실 증가.009813, 진행.24169. 선형/누적/무평활/순서섞기와도 ASR 동률. certified는 QCQP 수치 검사만 뜻함 | 확정 (한 단계 동작; 추가 이득 미확인) |
| R-045 | 같은 모델·단계, 정상 요청8 | loc_k20paper_s0_step99_req8_certified_solver.json | 무방어 ASR18.12/ACC58.40%; temporal 및 같은 보존제어 ASR11.79/ACC99.02%. 한 번 축소한 실제 진행.12077, 최대 클래스 손실 증가.004425. solver risk_after는 축소 전이며 수용 후 노출은 더 작음 | 확정 (동일 개발 모델) |
| R-046 | 합성 입력·실제 MNIST Net, 2라운드·3client | test_request_replay.py; 전체 unittest 106개 통과 | 초기 checkpoint·보존 archive·참여 명단으로3군 replay 완료. 요청 client는 보정 호출에서 제외, 각 군 실제 호출4회, 매 단계 제어. 실제 백도어 또는 독립 데이터 성능 실험 아님 | 확정 (통합 동작) |

전체 공격 replay 실행기는 구현했지만 기존 로컬100라운드 궤적의 진짜 학습 전 상태·실제 보존 archive가 없어 그 실행은 미완료다. 원래 BadFU 내부 표본 삭제도 별도 어댑터가 필요하다. 원고 v7은 기존 파일에서 개정하고 내장 컴파일 성공을 확인했다.

## 실제 FU 갱신 제어와 관측 가정 진단 (2026-10-14)

| 번호 | 자료 | 실행 | 결과/코드 | 관측과 범위 | 상태 |
|---|---|---|---|---|---|
| R-037 | 해석적 최적해·부분 참여·관측 사각지대 fixture | unittest discover | `FUBA/tests/test_tension_control.py` 등, 전체88개 통과 | 개별 보정 후 집계, 삭제 단위 호출 금지, 실제 replay 상태 제어, 국소 제약, 클래스 bias 소거, 밝기 관측에 보이지 않는 트리거 방향. 성능 실험 아님 | 확정 |
| R-038 | 기존 `loc_k20paper_s0`, MNIST/LeNet, K20 seed522 | `experiment_tension_control.py`, 100단계 수정 전/후 및99단계; 요청4, 최근20전이 | `FUBA/logs/tension_control/*step100*.json`, `*step99_req4_action.json` | 밝기 관측32개/보존128개/평가512개; 보정2batch. 99에서 요청4 제외7명,100에서는 요청4 원래 불참하여8명. 해당 단계에서 대리 좌표를 제어하는 동작만 검사; 전체 FU·ASR 아님 | 확정 (개발 스모크) |
| R-039 | 같은 모델·같은99단계, 정답 트리거 및 밝기 관측 | `--eval-target 8`, `--probe-view brightness/oracle-trigger` | `*step99_req4_brightness_eval.json`, `*step99_req4_oracle_eval.json` | ASR 분모458개: 단계 전12.45%, 무방어20.74%, 밝기 history_action10.70%, linear_action10.70%, history_free11.79%; 각각 후3군 ACC99.02%. oracle B 상위8좌표 모두 표적8, 밝기 B 상위8은 표적8 없음. 관측 가정 진단이며 전체 FU·망각·신규성 입증 아님. 두 후속 실행 시간은 병렬 영향 있어 비용 비교에 사용하지 않음 | 확정 (동일 개발 모델의 진단) |

R-039는 대리 좌표의 부진을 장력 이론의 실패로 해석하지 않기 위해 추가했다. 실제 트리거 영역에서는 역사적 상쇄 신호가 보였지만 공개 관측으로 옮기는 연결은 미검증이다. 새 제어의 ASR 변화와 장력 이력만의 추가 가치를 구별한다.

## 선택적 삭제율 v3와 실행 묶음 (2026-10-14)

| 번호 | 자료 | 실행 | 결과/코드 | 관측과 범위 | 상태 |
|---|---|---|---|---|---|
| R-034 | 기존 FUBA IID/owner 각3개 학습 캐시 | `python FUBA/replay_tension.py`; `python FUBA/benchmark_tension.py` | `FUBA/logs/tension_audit/cache_replay_v3.json`, `historical_ablation_v3.json` | v3 요청자·표적 동시1위 10/20/40/99시점 4/6,6/6,6/6,6/6. v2는3/6,5/6,5/6,5/6. 단순 선형 삭제 영향은0/6,0/6,3/6,4/6. 같은 개발 자료 재생이며 정상 보정/ASR/새 공격 검증 아님. 불완전 상쇄 동일비율 삭제 반례를 기준으로 수식 수정 | 확정 |
| R-035 | 전체 동작검사, 별도 보관 실제 `loc_k20paper_s0` 체크포인트 | 69 tests; `collect_tension.py --rounds 3 --probe 16` | `FUBA/tests/`, `FUBA/logs/tension_v3/smoke_loc_k20paper_s0.{npz,json}` | FL 전이2개·입력16개, 양의 실제 학습 갱신에 대한 입력별 logprob/중심화 logit 관측. FedAvg 갱신 상대 복원오차 최댓값3.94374e-7. FU·JVP·정상학습 새 실행 아님 | 확정 |
| R-036 | 정상/공격 독립 학습, IID/owner | `experiment_tension.py --profile pilot --stage all` (18개); 이후 `confirm` (68개) | `RUN.md`, 새 수집기/비교기/실행기 | 예비는 조건별 정상 보정3개로5% 문턱 불가. 확인은19개 보정/10개 정상 시험/5개 새 공격. 시간·클래스·요청 최대값 보정. 원래 BadFU/실제 FU·복구는 별도 | 예정 |

## FL→FU 장력·삭제 경계 재정식화 v2 (2026-10-14, 이전 기록)

| 번호 | 자료 | 실행 | 결과/코드 | 관측과 범위 | 상태 |
|---|---|---|---|---|---|
| R-032 | 구조적 fixture | unittest discover | tension_audit.py, tests/test_tension_audit.py | 장력12개 검사 추가, 전체59개 통과. 동일 비율 삭제, 클라이언트/묶음 역할 동형, 정상 반대 기여, 입력 평균/집계 정보손실 확인. 현실 탐지율 아님 | 확정 (수학·동작) |
| R-033 | 기존 IID/owner K20 각3개 학습 캐시 | replay_tension.py | FUBA/logs/tension_audit/cache_replay_v2.json | FU 모델 생성0. 시점10/20/40/99의 억제자 점수1위5/6,6/6,6/6,6/6; 요청자·표적 동시 일치3/6,5/6,5/6,5/6. 초기 정상1위와 owner 클래스3 혼동 기록. 정상 보정·BadFU 실제묶음 검증 아님 | 확정 (기술적 순위 분석) |

`cache_replay.json`은 양방향 경계 차감을 넣기 전 초기 산식의 기록이다. 단일 단위 삭제 순위는 v2와 같지만 동일 비율로 두 역할을 제거하는 경우를 처리하려면 v2를 사용해야 한다. 현재 주장의 정본은 열린 `draft_v7.tex` 본문이다.

## 삭제 반응 감사 추가 실행 (2026-10-14)

새 논리와 전체 표는 `docs/paper/DEFENSE_LOGIC_REDESIGN.md`. 기존 입력/새 입력은 같은 학습 모델을 공유하며 독립 시드가 아니다. 비교기·실행기 해시, 입력 인덱스와 출처는 각 원본 JSON에 기록했다.

| 번호 | 자료 | 실행 | 결과 (FUBA/logs/response_audit/) | 관측과 범위 | 상태 |
|---|---|---|---|---|---|
| R-028 | R020의 loc_k20paper_s0 | experiment_response.py --source fuba_local.json | fuba.json | 18조건. 두 프로브 모두 soft .10/.20 경보, 기존/hard .20만. .05는 soft도 미검출, 정상0. .20 보존실패 별도 | 확정 (1학습시드 예비) |
| R-029 | R021의 badfu_traj_full_local15 | experiment_response.py --source badfu_local.json | badfu.json | 18조건. soft/hard 공격경보0. 기존DOSE는 기존입력 요청자5 .20 경보, 새입력에서는 정상1 .20 경보 및 요청자5 누락 | 확정 (음성·probe 민감도; 원래 묶음FU 아님) |
| R-030 | R022의 loc_clean_k8 | experiment_response.py --source clean_local.json | clean.json | 48조건. 모든 검출기 정상경보0. K20·동일학습길이 정상FPR로 일반화하지 않음 | 확정 (제한 대조군) |
| R-031 | R028–R030, 동작 fixture | summarize_response.py; unittest discover | summary.json, diagnostics.json, source_snapshots/ | 총84조건. 소스 두 버전 해시 확인. 동작검사47개 통과(기존37+새10); 성능 실험 표본 수와 구분 | 확정 (파생·동작 검증) |

## 공동삭제 검증 추가 실행 (2026-10-13)

결과가 완료됐다는 뜻의 ‘확정’이며 일반화된 성능 주장을 뜻하지 않는다.
수학·범위·실패는 `docs/paper/REVISION_REPORT.md`, 소스 해시와 소거표는 `FUBA/logs/paired_audit/summary.json`.

| 번호 | 자료 | 실행 | 결과 파일 (FUBA/logs/paired_audit/) | 관측과 범위 | 상태 |
|---|---|---|---|---|---|
| R-020 | loc_k20paper_s0 | experiment_paired.py --case fuba | fuba_local.json | 3요청×3비율=9. 요청자4의 .05/.1 paired 추가경보, .2 붕괴로 판정불가. 정상5/19 경보0. 후보에 정상 혼입 | 확정 (1학습시드 예비) |
| R-021 | badfu_traj_full_local15 | experiment_paired.py --case badfu | badfu_local.json | 9조건. paired 상대집합 없음. 기존DOSE는 요청뷰 .2에서만 요청자5 경보 | 확정 (음성 결과; 원래 묶음FU 아님) |
| R-022 | loc_clean_k8 | experiment_paired.py --case clean | clean_local.json | 8정상×3비율=24. DOSE/paired 경보0, 에너지단독3. 하나의 정상 학습모델이며 K20 정상FPR 아님 | 확정 (제한 대조군) |
| R-023 | synthetic_paired_v1 | experiment_paired.py --case synthetic | synthetic.json | 6종×50 + 보정199/시험200=699. 비선형실패·과적합 각50/50 기각. 정상반대기여50/50 경보. 정상rank 기각4/200 | 확정 (기전·반례; 현실ASR 아님) |
| R-024 | R020의 고정된 개입 + 기존 정답제외 재학습모델 | evaluate_paired.py --audit fuba_local.json | fuba_evaluation.json | .1 요청 ACC96.9/ASR49.8 → 공동98.1/9.8; 기준98.0/9.8. 기존정답제외 재학습98.7/.44. 새 복구/전량삭제 실험 아님 | 확정 (1시드,독립평가입력) |
| R-025 | R021의 고정된 개입 | evaluate_paired.py --audit badfu_local.json | badfu_evaluation.json | 기준83.8/26.2 → .2요청82.5/38.0. 상대없음. 충분한 잠복재현이 아닌 변형의 한계 | 확정 (실패/한계 기록) |
| R-026 | R020–R025 고정 결과 | summarize_paired.py | summary.json, paired_results.png/.pdf | 정상 에너지경보9 → 공동검증0. 기존DOSE도 정상0이므로 DOSE보다FPR개선 주장불가 | 확정 (파생 소거분석) |
| R-027 | 동작 fixture | python -m unittest discover -s FUBA/tests -v | 코드 tests/test_structural_guard.py, tests/test_paired_audit.py | 37개통과. 가법정리·비선형오차·정상반례·OR판정·삭제실패·가중치검사 포함 | 확정 (동작검증; 성능표아님) |

## 기존 실행 기록

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
| R-013 | 2026-10-13 | bd_{iid,owner}_r100w5_k20paper_s{0,1,2} (DISLAB 6 궤적) | run_ledger.sh (ledger_defense.py, A-03) | h0 0.8, tau 3, kappa 2, min_pos 2, persist 3, start 5; 역할은 채점에만 | FUBA/logs/ledger/bd_*_r100w5_k20paper_s*.json, summary_{iid,owner}_r100w5_k20paper.md | 주장 C 를 방어 절차로: 온라인 균형 장부 → 경보 및 운반자 특정 | 확정: 6/6 궤적 전부 11~16 라운드(초반 15%)에 경보 발령. 클래스 8 정답 6/6, 위장자 4 정답 6/6, 운반자 {0,1,2,3} 정밀도 1.0 (6/6) 재현율 1.0 (6/6), 소유자 오제외 0/6. 경보 즉시 제외 재학습 목록 도출. 정직한 단서: (a) 학습 **끝** 장부에서는 owner s0 의 소유 클래스 3 이 양의 다리 2 명(소유자 +, 정상 1 명)으로 공모 규칙에 걸렸다 → 공모에 '양의 다리 한 명의 몫 ≤ 0.6' 조건 추가(`--max_share 0.6`, 기본값; 운반자 넷의 최대 몫은 0.31–0.54, 소유자+정상은 >0.6; 0.5 는 IID s0 의 타깃 클래스까지 깎아 부적합). 추가 후 끝 장부 오경보 0/6, 경보 라운드·목록 불변. (b) owner s1 은 학습 끝에 소유자가 클래스 8 의 양의 다리 집합에 들어온다 (경보 시점 목록은 정확) → 제외 목록은 경보 시점 것을 쓴다. (c) 위장자는 모든 클래스의 음의 다리(z −4…−19)라 '양의 다리 없음' 클래스(외톨이 음)가 0–3 개 생기나 판정에 쓰지 않는다 |
| R-014 | 2026-10-13 | loc_k20paper_s0 (노트북, IID 1 시드; DISLAB 6 궤적은 RUN.md A-03 대응 블록) | benign_departure.py retrain_benign --exclude 0 1 2 3 --init_round 13 (장부의 t_back) | 되감기 + 연합 제외 이어 학습, 87 라운드, 15 클라 | FUBA/logs/benign_departure/loc_k20paper_s0_excl_from13.json | 주장 B 의 처방을 탐지 정보로 (되감기 대응) | 노트북 1 시드 완료: 삭제 전 ASR 7.62 / ACC 98.82. 되감기 t_back=13 → ASR 0.68 / ACC 99.08 (786 초); 되감기 없이 경보 라운드 16 에서 제외 → ASR 0.74 / ACC 99.01 (760 초); 처음부터 제외 재학습(로컬 기준선) → ASR 0.31 / ACC 99.07 (899 초). 셋 다 ASR < 1, 정확도 99. 되감기·즉시 제외는 기준선보다 ASR 0.4 p 높지만 라운드 13–16 개를 아낀다. DISLAB 6 궤적은 RUN.md A-03 대응 블록 (예정) |
| R-019 | 2026-10-13 | loc_k20paper_s0 (노트북, 논문 설정 1 시드) | analysis_tradeoff.py (반합성: 보관된 억제자 기여를 s 배로 축소) | s ∈ {0,…,1}; θ_T − (1−s)U_S 의 ASR/ACC, 억제자 기여 s 배의 누적 기여도 z·H | FUBA/logs/tradeoff/loc_k20paper_s0.json, docs/paper/fig_tradeoff.png | 적응형 공격 (A) 억제 강도의 trade-off | s=1: ASR 7.5 / z −16.5; s=0.9: ASR 42.9 (학습 중 노출) / z −14.9; s=0.8: ACC 54.7; s≤0.7: 붕괴 (ACC 9.8). |z|<3 은 s≤0.2 에서만 → 잠복과 탐지 회피를 동시에 얻는 s 없음. 주입자 재학습 없는 감도 분석 (실제 적응형은 예정) | 예비 (반합성) |
| R-017 | 2026-10-13 | badfu traj_full (노트북, CIFAR-10/ResNet-18, K=6, 15 라운드, 시드 42, 운반자 0, 위장자 5, 타깃 0) | ledger_defense.py --fmt badfu (균등 가중, start 3, min_present 1) | h0 0.8, tau 3, kappa 2 | FUBA/logs/ledger/badfu_full.json | 누출 가정 (상쇄가 공개 입력에 새는가) 의 BadFU 검사 | **음성**: 경보 없음. 클래스 0 의 H 0.51, 음의 다리 = 정상 4 (z −1.2), 위장자 5 는 클래스 9 의 음의 다리(z −1.6)일 뿐. 장부는 BadFU traj_full 에서 상쇄를 보지 못한다 (DRAFT_v3 §9 의 경계 사례). 같은 궤적에서 요청 시점 빼 보기는 첫 단계 z 8.4 (R-004). 40 라운드 시드(badfu_s42…)에서 재확인 예정; 운반자 1 명이라 |Pos| ≥ 2 규칙은 구조적으로 못 쓴다 | 예비 (경계) |
| R-012 | 2026-10-13 | bd_{iid,owner}_r100w5_k20paper_s{0,1,2} | analysis_equilibrium.py (A-02) | 역할 정보 없음; 저장 전역 모델·갱신·참여 명단만 | FUBA/logs/equilibrium/bd_*_k20paper_s*.json → summary_*_r100w5_k20paper.md | 주장 C (학습 중 평형 지문): 누적 타깃 밀기로 위장자 6/6 (IID z 24, owner z 11), 운반자 양의 다리, 순/총 0.07–0.16; owner 는 소유 클래스 3 도 헤지(소유자 +30 vs 위장자 −35) → 타깃 추정은 H 기준·양의 다리 수로 (ALGORITHMIZATION §3). 복원력 지문 보조, 지연 반작용 지문 폐기 | 확정 (균형 장부 버전으로 재실행 예정) |

