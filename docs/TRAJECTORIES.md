# 사용할 궤적 — 2026-10-15

새 학습 경로는 [RUN.md](../RUN.md)의 `reproduction/run.py`로 교체했다. 공식 소스 재다운로드 후 BadFU 전체 단계·FUBA MNIST/CIFAR의 **축소 실행과 전체 상태 저장**을 확인했다. [검증 기록](../reproduction/validation/summary.json). 아래 기존 실험 자료는 유지하며, 새 원본 실행의 성능 결과로 다시 이름 붙이지 않는다.

**직접 검사한 로컬 궤적은 저장 파일·집계 검사를 통과했다. 궤적 생성이 잘못됐다는 근거는 발견하지 못했다.** 파일 무결성, 원본과의 실행·평가 조건 일치, 공격 효과 재현은 별도로 판단한다. ASR 하나만으로 궤적을 부적합하다고 한 이전 판정은 철회한다.

FUBA의 추가 검증기는 저장된 **평균 트리거 생성기·eval 모드**를 사용한다. 가져온 학습 코드의 로그는 **개별 생성기·train 모드**로 평가하며 마지막 개별 생성기의 ASR을 출력한다. 평가 표본·배치 크기도 다르다. 아래 ASR은 추가 검증기 조건에서의 관측값이며, 이 차이가 수치에 미친 영향은 아직 측정하지 않았다. MNIST와 CIFAR 모두에 해당한다. [평가 경로 확인 근거](EVIDENCE_AUDIT.md#궤적-판정-정정-2026-10-15).

| 묶음 | 파일 확인 | 현재 용도·판정 |
|---|---|---|
| FUBA MNIST, K20, 100R, IID/owner 각 3개 | DISLAB 기록상 각각 295/295개, K20·참여 8명·FedAvg 일치 | **이력 탐지 개발에 사용 가능.** 추가 검증기 종료 ASR 3.15–7.77%. 활성화 근거는 기여 일부 빼기이며 실제 FU 검증과 구분 |
| FUBA 정상 MNIST/CIFAR, IID/owner 총 4개 | DISLAB 기록상 각각 200/200개, K20·참여 8명·FedAvg 일치 | **정상 이력 분석에 사용 가능.** 조건별 1개이므로 독립 보정·시험 전체를 대신할 수 없음 |
| FUBA CIFAR, K20, 100R, 3개 | DISLAB 기록상 각각 295/295개, 집계 일치 | **이력 분석 자료로 유지.** 추가 검증기 종료 ASR 97.13/89.30/96.53%로 이 평가에서는 잠복 상태가 관측되지 않음. 학습 로그와 평가 조건을 맞추기 전 원본 공격 재현 실패로 단정하지 않음 |
| 로컬 FUBA `loc_k20paper_s0` | 실제 파일과 집계를 재확인, 아래 검사 기록 참고 | **이력 분석에 사용 가능.** 진짜 학습 전 `round_0`은 없음. 처음부터의 FU replay용으로는 불완전 |
| 로컬 BadFU `imported_traj_full`, 15R, seed42 | 실제 15라운드·전후 모델, 데이터·초기 모델 확인 | **공개 코드의 6클라이언트 구성에 따른 이력 분석에 사용 가능.** 저장 실행기는 UL 루프를 분리한 15R 실행이므로 전체 원본 실행과 같다고 단정하지 않음. 과거 학습 데이터 해시는 없어 재구성 데이터의 원본 동일성은 미확인 |
| 보관 PR #1의 BadFU 40R 공격 3개 | DISLAB 업로드 JSON/CSV 로그 확인 (`badfu_observe_v1`, 122개 JSON/CSV 로그 저장소 보관). 전체 학습 가중치(.pt)는 DISLAB 로컬에 보관 | **별도 방어 실험 실행기의 결과.** 삭제 전 ASR 69.77/65.29/81.61% 일치 확인. 언러닝(arm: none) 후 ASR 100.00% 폭발 확인. 상세 로그는 [badfu_observe_40r](../FUBA/logs/badfu_observe_40r/README.md)에 보관 |
| 같은 보관 실험의 BadFU 정상 3개 | DISLAB 업로드 JSON/CSV 로그 확인 (`eval-benign-42, 43, 44`). 학습 및 언러닝 지표 완전 일치 | **정상 훈련 및 언러닝 기준선 확인.** 삭제 전 ASR 1.76–2.01%, 언러닝 후 ASR 1.44–1.83%. 기존 투영 방어 시 오탐률 66.7% 확인 |

DISLAB 행은 업로드된 JSON/CSV와 검사 기록을 대조한 결과다. 이 세션에서 DISLAB 디스크의 .pt 파일을 직접 읽은 것은 아니다. 40라운드 공격 및 정상의 122개 상세 JSON/CSV 로그는 [FUBA/logs/badfu_observe_40r/](../FUBA/logs/badfu_observe_40r/README.md)에 보관되어 있다. 로컬 직접 검사 결과는 [파일 검사 JSON](../FUBA/logs/trajectories/file_check_2026-10-15.json)에 둔다.

다음 확인은 재학습보다 **동일 체크포인트의 평가 조건 대조**다. 기존 학습 로그와 추가 검증기 결과를 같은 라운드에서 비교하고, 생성기 종류·모드·평가 표본을 명시한다. 당시 개별 생성기가 저장되지 않았다면 평균 생성기로 그 결과를 대신 재현했다고 주장하지 않는다. 기존 궤적은 유지한다.

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
