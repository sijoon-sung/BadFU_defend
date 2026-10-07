# 단독 실험 안내

루트에 흩어져 있던 Python 실험을 주제별로 옮겼습니다. FUBA의 실제 MPI 파이프라인은 [FUBA](../FUBA/EXPERIMENT.md)에 있으며, 아래 스크립트는 별도의 탐색·재현 실험입니다.

## 실행 규칙

모든 명령은 저장소 루트에서 `python experiments/<주제>/<파일>.py`로 실행합니다. 일부 스크립트는 import 시점에 학습을 시작하므로, 파일 확인을 위해 무작정 import하거나 전체 실행하지 마세요.

기존 `./data`, `logs/`, `FUBA/data/` 저장 위치는 그대로 사용합니다. 파일 위치를 기준으로 루트를 찾던 스크립트는 이동한 깊이에 맞춰 수정했습니다. `scratch_test.py`와 그 스크립트가 import하는 지연 활성화 코드는 같은 폴더에 유지했습니다.

PyTorch·torchvision·NumPy가 주로 필요하며, LoRA 분석은 Matplotlib, `accuracy_test.py`는 scikit-learn도 사용합니다. 실험에 맞는 환경을 준비한 뒤 실행하세요. 이 폴더는 자동 단위 테스트 모음이 아닙니다.

## 실험 분류

| 폴더 | 내용 | 주요 출력 |
|---|---|---|
| [legacy](legacy/) | 초기 MNIST 공격 재현, 서버 그래디언트·부분공간 감사, 프록시 실험 | 표준 출력 및 기존 `logs/01_...`~`06_...` 로그 |
| [delayed_activation](delayed_activation/) | 삭제 후 후속 FL, 지연 활성화, 지속 공격, 탐색 검증 | `logs/delayed_activation*.json`, `logs/all_critiques_verified_results.json` 등 |
| [detection](detection/) | 반대 방향 통계 및 FUBA 단일 프로세스 탐색 | `logs/detect_antiparallel*.json`, `logs/fuba_*.json` |
| [lora](lora/) | MNIST LoRA 파일럿·격자와 결과 분석 | `logs/lora_*/` |

## 실행 예시

```bash
python experiments/legacy/verify_badfu.py
python experiments/detection/exp_detect_antiparallel.py --help
python experiments/lora/exp_lora_badfu_grid.py --help
python experiments/lora/analyze_lora_grid.py --dir logs/lora_grid
```

LoRA 격자 실행은 기존 MNIST 데이터가 루트의 `data/`에 있다고 가정하며, 해당 스크립트는 `download=False`를 사용합니다. 분석은 기존 JSON을 읽어 요약과 그림을 덮어씁니다.

[FUBA 단일 프로세스 탐색](detection/exp_fuba_real_detect.py)은 원 MPI 실행과 구분해야 합니다. 실패한 탐지 통계도 연구 기록으로 보존했습니다.

전체 파일의 이전 경로와 새 경로는 [이동표](../docs/REPOSITORY.md)에 있습니다.
