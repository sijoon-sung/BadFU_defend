# 공동삭제 감사의 로컬 자료 보충 등록

2026-10-13. 자동 생성되는 `EXPERIMENT_LOG.md`와 별개인 로컬 자료 기록이다.
기존 DISLAB 논문 궤적과 이름을 섞지 않는다. 아래 실행은 새 학습이 아니라 저장된 체크포인트 감사다.

| 식별자 | 실제 자료 | 설정 | 결과/파일 출처 |
|---|---|---|---|
| loc_k20paper_s0 | `C:/Users/sijoo/AppData/Local/Temp/bfu/repo-rfu/FUBA/checkpoints/*loc_k20paper_s0_round_*.pkl` | MNIST/LeNet, K20,100라운드, 참여0.4, IID seed522 | 기존 R014/R019 자료. `FUBA/logs/paired_audit/fuba_local.json`의 manifest/마지막모델SHA256 |
| loc_clean_k8 | 같은 checkpoints의 `*loc_clean_k8_round_*.pkl` | MNIST/LeNet,K8,8라운드, IID, 공격 없음 | 원래 로컬 `logs/ledger/loc_clean_k8.json` 역할 전원 benign. `clean_local.json` manifest |
| badfu_traj_full_local15 | `C:/Users/sijoo/AppData/Local/Temp/bfu/repo-rfu/BadFU_src/BadFU-main/logs/traj_full` | CIFAR10/ResNet18,K6,15라운드,seed42,counts=[10545,9647,9641,9649,9668,10545] | 기존 R017 자료. `badfu_local.json` manifest. 원래 BadFU의 same-client shard FU와 다름 |
| synthetic_paired_v1 | `FUBA/tests/test_paired_audit.py:FunctionalFixture` | 6종×50, 보정199,시험200 | `synthetic.json`; 현실 공격 재현으로 계산하지 않음 |

원본 자료는 저장소에 복사하거나 변경하지 않았다. 각 결과는 뷰 입력 인덱스·삭제 비율·문턱·
참여 가중·실제 선택 후보·실행 시간을 저장한다. 소스 사후 해시와 파생 소거 결과는 `summary.json`에 있다.
원본 체크포인트가 Temp에 있으므로 장기 재현을 위해 별도 보존이 필요하며, 이 작업은 외부로 업로드하지 않았다.

2026-10-14 BadFU 추가 등록: `artifacts/badfu/imported_traj_full/manifest.json`은 위 `badfu_traj_full_local15` 원본을 해시로 연결한다(FL 재학습·원본 변경 없음). 초기 `round_0.pt[gm]`을 `initial.pt`로 보존하고, seed42/dominant.7/전체 표본으로 재구성한 실제 clean/poison/camouflage 자료를 `data.pt`로 보존했다. 원래 데이터 해시가 없으므로 과거 분할과의 완전한 동일성은 미인증이며 가중 FedAvg 일치는 검사한다. 결과는 `FUBA/logs/badfu_request/traj_full_step14_eval_fixed.json`. 별도 `artifacts/badfu/pipeline_smoke_s42`는 실제 CIFAR 자료, 무사전학습 ResNet18, 클라이언트당32개·2라운드·각1배치의 저장/실행 검사다. 이 축소 자료는 공격 효과 근거가 아니다. 큰 archive/checkpoint는 로컬에만 보존한다.

완료/실패 범위와 재실행 명령은 `paper/REVISION_REPORT.md`를 따른다.
