# 두 방법의 업로드 검토 완료 — 재현 안내

**2026-10-17: `845803e`에 업로드한 11건 모두 검산 완료. 아래 본 실험을 다시 실행할 필요는 없습니다.** 실제 BadFU FU에서 삭제 후 탐색 패치 반응 증가가 확인됐습니다. 반면 누적 차감 근사에서는 정상 모델 정확도까지 무너졌고, 정상 설명 집합은 비영 기여를 모두 설명하지 못했습니다. FUBA의 기존 coverage 29.5~32.5%는 알려진 0 기여를 포함한 값입니다. [결과·원인·그래프](docs/DELETION_COMPARISON.md).

이번 검토를 위해 새 학습·FU·모델 추론을 실행하지 않았습니다. 원본 관측만으로 점수/구간/순위표와 보류 원인을 검산했습니다. 검토를 재현할 때만 아래를 사용합니다(새 `--out` 경로 필요).

~~~bash
python -m reproduction.audit_deletion_upload reproduction/validation/exports/deletion_comparison/07ad0aefbb22ddc8/deletion_comparison.zip --out artifacts/deletion_upload_review
~~~

**아래는 이미 완료한 실험의 재현 방법입니다.** 후속 실제 정상 FU/FUBA FU 검사는 아직 이 명령에 추가하지 않았습니다.

**2026-10-17. 한 실행에 두 알고리즘과 BadFU·FUBA·정상 삭제 비교가 모두 포함된다.**
저장된 전체 궤적을 사용한다. **FL 재학습·FU 실행·다운로드는 하지 않는다.**
이전 학습용 명령 대신 아래 명령을 사용한다.

## 1. 입력 확인과 실행

저장소 루트에서 기존 학습용 Python 환경을 활성화한 뒤:

~~~bash
git pull --ff-only origin codex/request-guided-probe
python -m reproduction.run_deletion_comparison --dry-run
python -m reproduction.run_deletion_comparison --out artifacts/deletion_comparison_v1
~~~

dry-run의 입력 경로를 확인한 뒤 마지막 명령을 실행한다. 없는 입력은 missing에 표시하고 학습하지 않는다.

| 기본 저장 경로 | 검사 |
| --- | --- |
| artifacts/badfu_author_s42/full6 | client 5 삭제, 이력 차감 근사 |
| 같은 실행의 clean5 | 정상 client 0~4 삭제, 이력 차감 근사 |
| 같은 실행의 fu_remove5 | 실제 FU 후 모델, 별도 검사 |
| artifacts/fuba_cifar10_s42/fl | client 4 삭제, 이력 차감 근사 |
| artifacts/fuba_cifar10_benign_s42/fl | 같은 삭제의 정상 대조 |
| 같은 이름의 benign_s43, benign_s44 | 추가 정상 실행 |

**작은 관측 ZIP만으로는 부족하다. 실제 round 체크포인트가 필요하다.**
공개 데이터는 학습 protocol의 data_root 또는 FUBA/data에서 찾는다.
경로가 다르면 --data-root /실제/데이터/상위폴더를 추가한다.
PyTorch 2.4.1/torchvision 0.19.1 환경에서 확인했으며 NumPy·SciPy·matplotlib을 사용한다. MPI 학습은 없다.

설치/메모리 확인을 위한 작은 탐색을 먼저 할 때만:

~~~bash
python -m reproduction.run_deletion_comparison --smoke --out artifacts/deletion_comparison_quick_v1
~~~

**--smoke도 저장된 모든 라운드를 읽는다.** 이미지와 탐색 예산만 줄이며 성능 평가용이 아니다.
기본값은 참조/검색/평가 각각 클래스당 8장(총 240장), 20스텝×2시작점이다.
10개 표적에 대해 가산 변형 1종과 모서리 4×4 패치 4위치를 탐색한다.
모든 정상·공격 사례는 같은 예산을 사용한다. GPU가 있으면 자동으로 사용한다.

이 PC의 **2라운드** 모델 5건은 작은 예산으로 약 1~2분이었다.
본 실행은 수십 GiB 궤적 검산과 더 많은 탐색·JVP 때문에 **수십 분~몇 시간의 여유가 필요할 수 있다**.
본 실험 실측 시간은 아니며 완료 후 사례별 elapsed_seconds를 기록한다.
CPU에서는 더 느릴 수 있다. FL 학습 시간은 추가되지 않는다.

완료한 사례는 코드·설정·입출력 해시가 같으면 재사용한다.
설정 변경 또는 중간 실패 폴더는 덮어쓰지 않으므로 새로운 --out 이름을 쓴다.

## 2. 비교군과 판정 범위

| 점수 | 내용 |
| --- | --- |
| logit_dd | 로짓 이중 차분만 사용 |
| endpoint_normal_set | 끝점 변화에 정상 설명 집합 적용 |
| history_raw | 정상 설명 제거 없이 이력 관계 사용 |
| history_normal_set_single_block | 정상 설명 제거 후 전체 이력을 한 구간으로 합침 |
| history_normal_set | 정상 설명 제거 + 3개 시간 구간의 지속성 |
| logit_plus_history | 이중 차분과 이력 증거를 함께 요구하는 고정 비교 |

깨끗한 **공개 이미지와 라벨**이 필요하다. 클라이언트에게 새 데이터/라벨을 받지 않는다.
참조·검색·평가 이미지 중복을 검사하고 실제 공격 트리거·생성기·표적은 사용하지 않는다.
모든 방법은 같은 검색에서 고정한 변형·출력 좌표를 평가한다.

정상 설명 집합이 비면 unknown(판단 보류)이다. 정상·공격 어느 쪽으로도 판정하지 않는다.
설명 가능 비율과 최소 필요한 설명 오차도 저장한다.
같은 요청에서 둘 다 점수를 낸 경우의 AUC와 보류 건수를 함께 보고한다.
이미지 수나 한 모델의 여러 삭제를 독립 실험 수로 세지 않는다. 보정된 FPR/FNR은 아직 없다.

**중요한 범위:**

- FUBA의 기본 저장 자료에는 실제 FU 후 모델이 없어 이력 차감 근사를 쓴다. FedEraser 실행이 아니다.
- BadFU 실제 FU는 별도 결과다. 실제 정상 FU가 없으면 실제-FU AUC는 null이며 근사 대조를 섞지 않는다.
- BadFU clean5와 full6는 구성 차이가 있는 정상 대조다. 완전히 일치한 대조라고 해석하지 않는다.
- 이번 BadFU는 공개 코드의 별도 client 5 삭제다. 논문의 내부 위장 샘플 삭제 검증이 아니다.
- 변형 표적 응답률은 진짜 공격 트리거의 ASR이 아니다.

[수식·추가 가정·실패 조건](docs/DELETION_COMPARISON.md)

## 3. 결과 업로드

완료 후 다음 폴더를 출력한다.

~~~text
artifacts/deletion_comparison_v1/
  report.md / comparison.json / comparison.png / plan.json
  각_사례/protocol.json / scores.json / observations.npz

reproduction/validation/exports/deletion_comparison/<해시>/
  deletion_comparison.zip / export.json
  report.md / comparison.json / comparison.png
~~~

ZIP에는 로짓·JVP 관측·LP 구간·공개 ID/라벨·탐색 변형이 들어간다.
**이미지와 모델 가중치는 포함하지 않는다.** 다음처럼 작은 결과만 업로드한다.

~~~bash
git add reproduction/validation/exports/deletion_comparison
git commit -m "Add logit and normal-explanation history comparison"
git push origin HEAD:codex/request-guided-probe
~~~

모델 없이 점수와 구간을 다시 검산할 수 있다.

~~~bash
python -m reproduction.replay_deletion_comparison reproduction/validation/exports/deletion_comparison/<해시>/deletion_comparison.zip
~~~

## 경로나 예산을 바꿀 때만

~~~bash
python -m reproduction.run_deletion_comparison --badfu-run /경로/badfu_author_s42 --fuba-run /경로/fuba_cifar10_s42 --fuba-benign-run /경로/fuba_cifar10_benign_s42 --data-root /경로/data --out artifacts/deletion_comparison_custom_v1
~~~

--fuba-benign-run 뒤에 정상 실행 경로를 여러 개 줄 수 있다.
설정을 편집하려면 계획을 먼저 저장한다.

~~~bash
python -m reproduction.run_deletion_comparison --write-plan artifacts/deletion_plan.json
python -m reproduction.run_deletion_comparison --plan artifacts/deletion_plan.json --out artifacts/deletion_comparison_custom_v2
~~~

두 명령 사이에 JSON을 편집한다. 상대 경로는 저장소 루트 기준이다.
settings.batch_size를 8로 낮추면 메모리 부담을 줄인다.
probe_mode는 both/additive/patch 중 하나다. 참조/검색/평가 수, steps, restarts, blocks도 명시돼 있다.
정상 설명 집합의 coefficient_bound/clean_relative/probe_error를 바꾸면 별도 실험이다.
공격 결과를 보고 좋은 설정만 선택해 최종 성능으로 보고하지 않는다.

이미 실제 FUBA FU 모델이 있다면 해당 사례에 다음 형식의 actual_state를 추가한다.
정상 사례에도 실제 FU 모델을 연결해야 실제 정상·공격 비교가 가능하다.

~~~json
{
  "actual_state": {
    "path": "/경로/actual_fu.pt",
    "sha256": "실제 모델 파일의 SHA256",
    "pre_manifest_sha256": "원본 FL manifest.json의 SHA256",
    "request_clients": [4],
    "key": "gm_after"
  }
}
~~~

파일이 state_dict 자체라면 key를 생략한다.
해시·요청 결합은 검사하지만 외부 FU 절차 자체의 정당성까지 자동 인증하지 않는다.
외부 공개 풀은 각 사례의 pool에 images(uint8 NHWC), labels(정수), ids(고유 문자열)를 담은 NPZ 경로를 지정한다.

로컬 검사를 재현할 때:

~~~bash
python -m unittest reproduction.test_deletion_comparison reproduction.test_history_boundary reproduction.test_request_probe -v
~~~

초기 연결 검사에서는 **FUBA 이력 점수 0, BadFU 정상 설명 불가로 보류**가 관측됐다.
소규모 모델 결과는 성능 증거가 아니다. 본 실험에서도 보류가 많으면 정상 설명 집합의 가정부터 수정해야 한다.
