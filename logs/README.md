# 단독 실험 결과 안내

이 폴더는 기존 실험의 결과 파일과 그림을 보존합니다. 코드 정리 과정에서 원 데이터나 기록된 실행 명령을 다시 쓰지 않았습니다. 과거 로그에 나오는 루트 스크립트 경로는 [이동표](../docs/REPOSITORY.md)를 참고하세요.

| 결과 | 관련 코드 또는 문서 |
|---|---|
| `01_badfu_verification.log`~`06_deep_proxy_blended_B.log` | [초기 실험](../experiments/legacy/), [초기 벤치마크 문서](../docs/archive/INITIAL_BENCHMARKS.md) |
| `delayed_activation*.json` | [지연 활성화 실험](../experiments/delayed_activation/) |
| `all_critiques_verified_results.json` | [검증 실험](../experiments/delayed_activation/verify_all_critiques.py) |
| `detect_antiparallel*.json`, `fuba_*.json` | [탐지 탐색](../experiments/detection/) |
| [LoRA 격자](lora_grid/summary.md) | [LoRA 실험·분석](../experiments/lora/) |
| [비 IID 0.2](lora_noniid/summary_d0.2.md), [비 IID 0.9](lora_noniid/summary_d0.9.md) | [LoRA 격자 코드](../experiments/lora/exp_lora_badfu_grid.py) |
| `lora_pilot/`, `lora_pilot_vec/` | LoRA 파일럿 기록 |
| [과거 분석 사본](TRADE_OFF_ANALYSIS.md) | [docs의 분석 보고서](../docs/results/TRADE_OFF_ANALYSIS.md)와 별도로 보존한 기존 파일 |

FUBA 탐지·정화 결과는 [FUBA/logs](../FUBA/logs/)에 있습니다. 이 폴더의 다른 프로토콜 결과와 통합 평균을 내지 마세요. 실제 FU 평가 결과는 `exp/evaluation` 브랜치의 `FUBA/eval_logs/`를 확인해야 합니다.

Git은 모델·데이터셋·대부분의 새 실행 로그를 제외합니다. 보존할 새 결과는 실행 이름·seed·설정·코드 커밋을 함께 기록하고, 필요한 파일만 명시적으로 추가하세요.
