# 현재 실행: 실제 삭제 후 백도어 교정 비교

브랜치: `codex/purification`. 이미 실행 중인 `codex/request-guided-probe` 작업은 그대로 마치고, 결과를 보관한 뒤 전환한다. 두 브랜치의 현재 실험 코드는 동일하다.

2026-10-17. **이전에 올린 이중 차분·정상 설명 집합 11건은 검토 완료했다. 다시 돌리지 않는다.**
이번에는 기존 BadFU `full6 → fu_remove5` 모델을 재사용해 정화를 비교한다.
FL 학습·FU 실행·다운로드는 하지 않는다. [수학·비교군·한계](docs/current/PURIFICATION_COMPARE.md).

## 1. DISLAB에서 실행

학습 때 사용한 Python 환경을 켠 뒤 저장소 루트에서:

```bash
git pull --ff-only origin codex/purification
python -m reproduction.run_purification --dry-run --math-ablation
python -m reproduction.run_purification --out artifacts/purification_badfu_s42_v1 --math-ablation
```

기본 입력은 `artifacts/badfu_author_s42`다. `dry-run`은 실행 기록·모델 해시·삭제 대상 연결 및
원본 공격 record를 확인한다. 공개 데이터 로딩과 정화는 하지 않는다.
모델이 없으면 학습하지 않고 중단한다. 결과 ZIP만 있는 컴퓨터에서는 실행할 수 없다.

| 필요한 입력 | 용도 |
| --- | --- |
| full6 마지막 round, fu_remove5 마지막 round | 같은 client 5 삭제의 실제 전후 모델 |
| protocol.json, split.json, runtime.patch, reproduction 스냅샷, 각 manifest | 입력 출처 확인 |
| bd5 마지막 round, manifest (있으면 연결) | 잔류 자료 학습의 참조. 망각 보장 아님 |
| 원래 badnet_dataset/pert_result.pt와 bd_test_dataset 이미지, record_inputs.json | 정화 완료 후 실제 트리거 평가 |
| CIFAR-10 공개 시험 자료 | 개발·검증·평가 분할 |

경로가 다를 때만 명시한다.

```bash
python -m reproduction.run_purification --run /실제/경로/badfu_author_s42 --data-root /실제/경로/data --record-root /실제/경로/badnet_dataset --out artifacts/purification_badfu_s42_v1 --math-ablation
```

`cover5 → fu_remove1`은 다른 요청이므로 섞지 않는다. 원본 학습 모델·궤적·record를 덮어쓰지 않는다.

## 2. 비교 내용과 예상 시간

한 번 실행하면 다음 **여섯 arm**을 비교한다. pre와 bd5는 별도 참조로 출력한다.

1. 실제 FU 후 그대로.
2. 공개 정상 자료 미세조정.
3. post-only 패치 탐색 후 정화.
4. pre/post 평균 차분 탐색 후 같은 정화.
5. 차분의 상위 25% 위험(CVaR) 탐색 후 같은 정화.
6. 5번과 동일 패턴으로 클래스별 정상 CE 제약 정화.

개발/검증/시험은 클래스당 16/16/100장이다. 네 모서리 4×4 패치·표적 10개를
20스텝×2시작점으로 탐색하고 5 epoch 정화한다. 실제 트리거·공격 표적은
탐색/정화/epoch 선택에 주지 않는다. 실제 ASR과 고정 탐색 패턴 반응을 구분한다.

GPU 자동 사용. PyTorch 2.4.1, torchvision 0.19.1, NumPy, SciPy, Pillow, matplotlib에서 확인했다.
처음에는 **수십 분~몇 시간의 여유**를 둔다. 본 예산의 실측은 아직 없으며 CPU는 더 오래 걸릴 수 있다.
각 arm의 탐색·정화 시간을 저장한다. 기존 FL 학습 시간은 추가되지 않는다.

연결만 작게 확인할 때는 별도 폴더를 쓴다. 성능 평가용이 아니다.

```bash
python -m reproduction.run_purification --smoke --math-ablation --out artifacts/purification_badfu_quick_v1
```

중간에 끊겼으면 같은 명령에 `--resume`을 추가한다.

```bash
python -m reproduction.run_purification --out artifacts/purification_badfu_s42_v1 --math-ablation --resume
```

완료 arm만 해시를 확인해 재사용하며 중단된 arm은 처음부터 한다.
코드·입력·설정이 달라졌으면 새 `--out`을 쓴다.

## 3. 업로드

완료하면 다음 경로를 출력한다.

```text
artifacts/purification_badfu_s42_v1/
  preparation.json / development.npz / validation.npz / test.npz
  comparison/report.md / results.json / comparison.png / protocol.json
  comparison/*_search.json / *_patterns.pt / 정화_모델.pt

reproduction/validation/exports/purification/<해시>/
  purification.zip / export.json / report.md / results.json / comparison.png
```

작은 결과만 올린다. 모델·이미지는 ZIP에서 제외하며 실행 컴퓨터에 보관한다.

```bash
git add reproduction/validation/exports/purification
git commit -m "Add saved-model purification comparison"
git push origin HEAD:codex/purification
```

`fallback_to_post=true`는 클래스별 제약을 통과하는 epoch가 없었다는 뜻이다.
ASR이 낮아도 정상 정확도가 무너지거나 post 자체의 ASR이 낮으면 정화 성공으로 해석하지 않는다.
망각 검증·공식 SAU 비교·정상 실제-FU 대조·독립 seed·정화 후 재탐색은 후속 실험이다.

## 4. FUBA와 코드 검사

현재 자동 명령은 BadFU용이다. **FUBA의 실제 동일 요청 전후 모델과 생성기 평가 자료가 있을 때만**
[일반 실행기 입력](docs/current/PURIFICATION_COMPARE.md#일반-실행기의-입력과-재현)으로 연결한다.
FUBA 실제 FU·정화도 이미 검증됐다는 뜻은 아니다.

```bash
python -m unittest reproduction.test_purification_compare reproduction.test_purification_math -v
```

14개 테스트와 2라운드 BadFU 저장 모델 전체 연결을 확인했다. 이전 관측은
[결과 검토](docs/experiments/DELETION_COMPARISON.md), 원본 학습은 [공식 코드 실행 문서](docs/operations/FRESH_UPSTREAM.md)에 있다.
