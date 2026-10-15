# 사용할 궤적 — 2026-10-15

**파일이 정상인지와 잠복 공격 조건을 만족하는지는 별개다.** 아래 표만 현재 판단 기준으로 사용한다. 저장소의 오래된 체크 표시는 그대로 성능 판정에 쓰지 않는다.

| 묶음 | 파일 확인 | 현재 용도·판정 |
|---|---|---|
| FUBA MNIST, K20, 100R, IID/owner 각 3개 | DISLAB 기록상 각각 295/295개, K20·참여 8명·FedAvg 일치 | **이력 탐지 개발에 사용 가능.** 종료 ASR 3.15–7.77%. 활성화 근거는 기여 일부 빼기이며 실제 FU 검증과 구분 |
| FUBA 정상 MNIST/CIFAR, IID/owner 총 4개 | DISLAB 기록상 각각 200/200개, K20·참여 8명·FedAvg 일치 | **정상 이력 분석에 사용 가능.** 조건별 1개이므로 독립 보정·시험 전체를 대신할 수 없음 |
| FUBA CIFAR, K20, 100R, 3개 | DISLAB 기록상 각각 295/295개, 집계 일치 | **잠복 조건 미충족.** 종료 ASR 97.13/89.30/96.53%. 파일 오류로 폐기할 자료는 아니며 실패 분석용으로 보관 |
| 로컬 FUBA `loc_k20paper_s0` | 실제 파일과 집계를 재확인, 아래 검사 기록 참고 | **이력 분석에 사용 가능.** 진짜 학습 전 `round_0`은 없음. 처음부터의 FU replay용으로는 불완전 |
| 로컬 BadFU `imported_traj_full`, 15R, seed42 | 실제 15라운드·전후 모델, 데이터·초기 모델 확인 | **공개 코드 6클라이언트 변형 분석에 사용 가능.** 원래 표본 삭제 설정과 다름. 과거 학습 데이터 해시는 없어 재구성 데이터의 원본 동일성은 미확인 |
| 보관 PR #1의 BadFU 40R 공격 3개 | 실행 결과·manifest 존재. 현재 GitHub 결과 ZIP에는 전체 학습 가중치 없음 | **잠복 조건 미충족.** 삭제 전 ASR 69.77/65.29/81.61%. 원시 궤적 재사용은 DISLAB 파일 위치 확인 필요 |
| 같은 보관 실험의 BadFU 정상 3개 | 실행 결과·manifest 존재 | **원시 파일 위치 확인 필요.** 결과가 없었던 것이 아님 |

DISLAB 행은 업로드된 검증 기록을 확인한 것이다. 이번 작업에서 DISLAB의 원시 파일을 직접 다시 열었다는 뜻은 아니다. 로컬 직접 검사 결과는 [파일 검사 JSON](../FUBA/logs/trajectories/file_check_2026-10-15.json)에 둔다.

파일 이름은 `bd_{iid,owner}_r100w5_k20paper_s{0,1,2}`, `bd_iid_r100w5_k20paper_cifar_s{0,1,2}`, `clean_{iid,owner}_r100_k20paper[_cifar]_s0`이다. 세부 설정·파일 수의 원본은 [trajectories.json](../FUBA/trajectories.json)과 [registry.json](../FUBA/logs/trajectories/registry.json)이다. BadFU 40R의 원본은 [보관 커밋의 results_observe](https://github.com/sijoon-sung/BadFU_defend/tree/a0371dbf511a265250fd12f6accda026414ef219/results_observe)에 있다. 이 결과를 registry의 `badfu_s42` 등 다른 경로에 자동 대응시키지 않는다.

<details>
<summary>나머지 과거 궤적</summary>

- K8 기본 6개, K16 억제자 2명 설정 3개: 기록상 파일·공격 진단 통과. 이전 개발 자료로 보관.
- K20 단축 학습 6개: 파일은 완성됐으나 공격 활성화가 약함. 파이프라인 점검용.
- K8 median/Multi-Krum 각 3개: 이전 집계 방어 실험 자료. 현재 주 비교군과 혼합하지 않음.
- K8 FLAME s0: 파일은 있으나 잠복·정상 삭제 조건 실패. 나머지 FLAME/FLDetector 기록은 미생성.
- K20 median s0: 52/295개로 불완전. 다른 K20 집계 방어 궤적은 기록상 미생성.
- registry의 로컬 `missing`은 해당 검사 PC의 경로에 없었다는 뜻이다. 로컬 BadFU 원본은 실제로 별도 임시 작업 폴더에서 확인했으므로 ‘실험 자체가 없었다’고 해석하지 않는다.

</details>
