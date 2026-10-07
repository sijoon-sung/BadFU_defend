# 저장소 구조와 파일 이동 안내

2026년 10월 8일, main의 `c2b1399`을 기준으로 구조를 정리했습니다. 실험의 수식·학습 설정·저장된 수치는 변경하지 않았습니다.

## 폴더 역할

- `FUBA/`: 원 FUBA 구현과 실제 공격 궤적 기반 탐지·근사 정화.
- `BadFU_src/`: BadFU 재현 코드. 내부의 `BadFU-main/` 구조와 상위 파일은 경로·전처리 의존성을 보존하기 위해 유지.
- `experiments/`: 독립 탐색 스크립트. 저장소 루트에서 실행.
- `hpc/`: LLM·LoRA 실험. 기존 셸 스크립트는 `~/badfu-lora-nlp` 배포 환경을 가정하며, 일반 FUBA 실행과 별개.
- `logs/`: 단독 실험 결과. 기존 저장 경로 유지.
- `docs/results/`: 관측 결과 보고서.
- `docs/research/`: 브랜치와 기준 커밋을 명시한 연구 정리.
- `docs/archive/`: 초기 README 기록. 현재 결과 안내는 루트 README가 담당.

## Python 파일 이동표

| 이전 루트 경로 | 새 경로 |
|---|---|
| `accuracy_test.py` | `experiments/legacy/accuracy_test.py` |
| `agent_proxy_defense.py` | `experiments/legacy/agent_proxy_defense.py` |
| `compare_badfu_and_defense.py` | `experiments/legacy/compare_badfu_and_defense.py` |
| `deep_manifold_explorer_A.py` | `experiments/legacy/deep_manifold_explorer_A.py` |
| `deep_proxy_blended_B.py` | `experiments/legacy/deep_proxy_blended_B.py` |
| `server_gradient_audit.py` | `experiments/legacy/server_gradient_audit.py` |
| `server_gradient_deep_audit.py` | `experiments/legacy/server_gradient_deep_audit.py` |
| `verify_badfu.py` | `experiments/legacy/verify_badfu.py` |
| `verify_negative_similarity.py` | `experiments/legacy/verify_negative_similarity.py` |
| `exp_delayed_activation.py` | `experiments/delayed_activation/exp_delayed_activation.py` |
| `exp_delayed_activation_v2.py` | `experiments/delayed_activation/exp_delayed_activation_v2.py` |
| `exp_fl_fu_fl_delayed_resurface.py` | `experiments/delayed_activation/exp_fl_fu_fl_delayed_resurface.py` |
| `exp_fl_fu_fl_persistent_attack.py` | `experiments/delayed_activation/exp_fl_fu_fl_persistent_attack.py` |
| `verify_all_critiques.py` | `experiments/delayed_activation/verify_all_critiques.py` |
| `scratch_test.py` | `experiments/delayed_activation/scratch_test.py` |
| `exp_detect_antiparallel.py` | `experiments/detection/exp_detect_antiparallel.py` |
| `exp_detect_antiparallel_v2.py` | `experiments/detection/exp_detect_antiparallel_v2.py` |
| `exp_fuba_real_detect.py` | `experiments/detection/exp_fuba_real_detect.py` |
| `fuba_kappa_check.py` | `experiments/detection/fuba_kappa_check.py` |
| `exp_lora_badfu.py` | `experiments/lora/exp_lora_badfu.py` |
| `exp_lora_badfu_grid.py` | `experiments/lora/exp_lora_badfu_grid.py` |
| `analyze_lora_grid.py` | `experiments/lora/analyze_lora_grid.py` |

## 문서 이동표

| 이전 경로 | 새 경로 |
|---|---|
| `DETECTION_PURIFICATION_RESULTS.md` | [탐지·정화 결과](results/DETECTION_PURIFICATION_RESULTS.md) |
| `TRADE_OFF_ANALYSIS.md` | [지연 활성화 분석](results/TRADE_OFF_ANALYSIS.md) |
| `RESEARCH_STATUS_2026-10-07.md` | [연구 현황](research/RESEARCH_STATUS_2026-10-07.md) |
| 기존 `README.md` 본문 | [초기 벤치마크 기록](archive/INITIAL_BENCHMARKS.md) |

`logs/TRADE_OFF_ANALYSIS.md`는 별도의 기존 기록이므로 삭제하거나 합치지 않았습니다.

## 경로 호환성

루트 명령 `python <파일>.py`는 이동표의 새 경로로 바꿔야 합니다. 예를 들어 `python verify_badfu.py`는 `python experiments/legacy/verify_badfu.py`가 됩니다. 작업 디렉토리는 저장소 루트를 유지합니다.

`__file__` 기준 경로를 사용하는 탐지·LoRA 스크립트는 저장소 루트를 두 단계 위에서 찾도록 수정했습니다. 따라서 기본 출력은 계속 `logs/`, 데이터는 기존 `data/` 또는 `FUBA/data/`를 사용합니다. 상대 경로로 전달하는 `--out`과 `--dir`도 기존처럼 저장소 루트 기준입니다.

main의 FUBA 셸 명령은 이름 인자를 사용하고, defense-pipeline은 seed 인자를 사용합니다. 디렉토리 정리만을 위해 다른 브랜치의 기능을 가져오지는 않았습니다.

## 앞으로 파일을 추가할 때

1. 단독 실험 코드는 주제별 `experiments/` 하위에 둡니다. 실제 FUBA 파이프라인 확장은 `FUBA/`에서 관리합니다.
2. 결과 보고서는 `docs/results/`, 연구 판단·한계는 `docs/research/`에 두고 기준 실행과 커밋을 적습니다.
3. 원 수치는 기존 결과 경로에 유지하고, 체크포인트·데이터셋은 Git에 넣지 않습니다.
4. 보존된 과거 로그는 수정하지 않으며, 재실행은 새 실행 이름이나 출력 디렉토리를 사용합니다.
5. 다른 브랜치에 이 정리를 반영할 때는 먼저 그 브랜치에서 수정된 루트 스크립트가 있는지 확인하고 이동 변경을 병합합니다.
