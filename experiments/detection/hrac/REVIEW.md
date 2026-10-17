# 라벨 없는 BadFU 실험 실행 안내

현재 작업 브랜치: `codex/structural-latent-defense`.

이 문서는 `FUBA/experiment_hrac_unlabeled.py`로 저장된 BadFU 궤적을 분석하는 안내입니다. 모든 명령은 저장소 루트에서 실행합니다. FL 학습과 FU를 새로 실행하지 않습니다. 새 궤적 생성은 [RUN.md](../../../RUN.md)를 따릅니다.

## 라벨 없이 공개 이미지로 검사 — 현재 실행 경로

필요한 것은 공개 이미지, 기존 궤적의 `manifest.json`, 클라이언트별 모델 체크포인트입니다. 이미지 정답 라벨·공격 표적·정답 트리거·클라이언트 원본 이미지·`data.pt`·`pert_result.pt`는 요구하지 않습니다. 전처리와 모델 구조는 BadFU CIFAR-10/ResNet-18 기준입니다. FUBA의 다른 모델에 그대로 적용하지 않습니다.

기존 Python 환경의 PyTorch, torchvision, NumPy, SciPy, Pillow를 사용합니다. 아래 `/path/to/...`는 실행 PC의 실제 경로로 바꿉니다.

### 1. 이미지 준비 — 아래 두 방법 중 하나만 실행

현재 CIFAR-10 실험 재현용입니다. `--cifar-root`는 `cifar-10-batches-py`의 상위 폴더입니다. 기존 관측에 사용한 240개 이미지를 제외하고 픽셀과 식별자만 내보냅니다. 데이터는 자동 다운로드하지 않습니다.

```bash
python FUBA/experiment_hrac_unlabeled.py prepare --cifar-root /path/to/cifar10 --exclude-ids FUBA/logs/hrac/unlabeled_excluded_ids_v1.json --out artifacts/probes/unlabeled_pool.npz
```

별도의 공개 이미지 폴더를 쓰려면 위 명령 대신 아래를 실행합니다. 클래스 폴더나 라벨 파일은 필요하지 않습니다. RGB·32×32로 변환합니다. 다른 도메인의 이미지에서 탐지 성능은 아직 검증하지 않았습니다.

```bash
python FUBA/experiment_hrac_unlabeled.py prepare --image-dir /path/to/public_images --out artifacts/probes/unlabeled_pool.npz
```

이미 준비한 pool이 있으면 1단계를 건너뜁니다. 출력 pool을 덮어쓰지 않습니다.

### 2. 두 라운드로 경로와 속도 확인

```bash
python FUBA/experiment_hrac_unlabeled.py run --manifest /path/to/manifest.json --checkpoint-root /path/to/checkpoints --pool artifacts/probes/unlabeled_pool.npz --out FUBA/logs/hrac/unlabeled_run --stop-after 2
```

`--checkpoint-root`에는 manifest의 `history`와 `after` 파일이 있는 폴더를 지정합니다. 파일명을 유지하며 SHA-256을 검사합니다. manifest에 적힌 절대 경로가 이미 맞으면 이 옵션을 생략할 수 있습니다. 이 로컬 실험에서 사용한 manifest는 `artifacts/badfu/imported_traj_full/manifest.json`입니다. 다른 PC에는 해당 실행의 manifest와 실제 체크포인트가 필요합니다.

### 3. 나머지 라운드 이어서 실행

```bash
python FUBA/experiment_hrac_unlabeled.py run --manifest /path/to/manifest.json --checkpoint-root /path/to/checkpoints --pool artifacts/probes/unlabeled_pool.npz --out FUBA/logs/hrac/unlabeled_run --resume
```

2단계와 같은 경로·설정으로 실행합니다. 기본값은 무작위 이미지 80개 × 서로 겹치지 않는 3묶음, 패턴 3개 × 네 모서리입니다. 모든 출력 클래스를 검사하며, **저장된 모든 라운드**를 사용합니다. 없는 라운드를 추가 학습하지 않습니다.

중단되면 같은 명령으로 다시 시작합니다. 완료한 라운드는 해시 검사 후 재사용합니다. pool·코드·기기·배치 크기·시드 등 설정을 바꾸려면 새로운 `--out`을 지정합니다. GPU 메모리가 부족할 때는 새 출력 폴더에서 `--batch-size 32` 또는 `16`으로 시작합니다.

### 4. 결과 확인

`FUBA/logs/hrac/unlabeled_run/analysis.json`에 모든 클라이언트·출력의 점수와 순위가 저장됩니다. HRAC와 단순 누적 음의 기여·절댓값·마지막 라운드를 비교하고, 존재하는 5/10/15/20/40/100 및 마지막 라운드 시점을 기록합니다.

완료된 응답 캐시를 다시 분석할 때는 아래 명령만 실행합니다. 모델 체크포인트와 이미지 pool이 필요하지 않습니다.

```bash
python FUBA/experiment_hrac_unlabeled.py analyze --out FUBA/logs/hrac/unlabeled_run
```

`protocol.json`은 고정한 입력·설정, `progress.json`은 라운드별 시간·해시, `rounds/`는 측정한 응답입니다. 독립 정상 실행으로 임계값을 보정하지 않았으므로 FPR·TPR은 `null`입니다.

## 긴 궤적과 더 많은 이미지로 검사

기본 검사가 끝난 뒤 사용할 확장 설정입니다. 먼저 두 라운드만 실행합니다.

```bash
python FUBA/experiment_hrac_unlabeled.py run --manifest /path/to/long_manifest.json --checkpoint-root /path/to/long_checkpoints --pool artifacts/probes/unlabeled_pool.npz --out FUBA/logs/hrac/unlabeled_long --probe-size 256 --probe-seeds 20261022 20261023 20261024 20261025 20261026 --pattern-seeds 20261022 20261023 20261024 20261025 20261026 --stop-after 2
```

이어 돌릴 때는 **위 명령의 모든 설정을 유지한 채** `--stop-after 2`만 `--resume`으로 바꿉니다. 독립 정상·공격 학습 실행들이 있으면 같은 pool과 검사 설정으로 각각 새 출력 폴더에 실행합니다. 이미지·패턴 반복은 독립 학습 실행으로 세지 않습니다.

| 6클라이언트 / RTX 4060 Laptop GPU | 수집 시간 | 상태 |
|---|---:|---|
| 15라운드, 이미지 80개 × 3묶음, 패턴 3개 | **79.9초** | 실측 |
| 100라운드, 같은 기본 관측 설정 | 약 9분 | 단순 비례 추정, 미실행 |
| 15라운드, 이미지 256개 × 5묶음, 패턴 5개 | 약 12분 | 단순 비례 추정, 미실행 |
| 100라운드, 이미지 256개 × 5묶음, 패턴 5개 | 약 1시간 17분 | 단순 비례 추정, 미실행 |

시간은 이미지 export·프로세스 초기화·새 FL 학습 비용을 제외합니다. 실행 중 라운드별 시간과 남은 시간 추정치를 출력합니다. 긴 작업의 실제 시간은 먼저 두 라운드로 확인합니다.

## 확인된 결과

기존 15라운드 궤적에서 라벨 없이 고른 이미지 세 묶음 모두 요청자 5·출력 0이 1위였습니다. 단순 누적 음의 기여와 마지막 라운드도 1위였습니다. 같은 학습 궤적 하나의 결과이며 정상 Non-IID 오탐률·타 도메인·다른 트리거는 미검증입니다. 코드 테스트 105개와 중간 저장·재시작을 확인했습니다.

[프로토콜](results/historical/hrac/unlabeled_badfu_v1/protocol.json) · [전체 결과](results/historical/hrac/unlabeled_badfu_v1/analysis.json) · [수식과 한계](../../../docs/current/RESEARCH_SUMMARY.md)
