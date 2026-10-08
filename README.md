# BadFU Defend

연합 언러닝으로 활성화되는 백도어의 탐지와 정화를 연구하는 저장소입니다. FUBA·BadFU 공개 코드 기반 실험과 단독 탐색 실험, 저장 결과를 구분해 관리합니다.

## 먼저 볼 문서

- [새 탐지 관측 GPU 실행](experiments/request_purify/OBSERVATION_BENCHMARK.md) — 정상 기준 잔여 방향·기능 변화·실제 FU 관측을 비교합니다. 추가 탐지 FU 0회이며 새 점수는 아직 정화에 연결하지 않습니다.
- [방어 논문의 전개와 탐지 재설계](docs/research/DETECTION_REDESIGN_2026-10-08.md) — 실제 탐지 실패를 반영한 관측 후보, 선행연구의 설계 논리, 최소 검증 계획입니다. 새 방어 성능 주장은 아닙니다.
- [GPU seed 42 ASR 100% 원인 점검](docs/results/GPU_SEED42_ASR_AUDIT_2026-10-08.md) — 탐지 누락·oracle 정화 실패·삭제 전 ASR과 코드 검증을 구분한 최신 실제 결과입니다.
- [논문 범위·독창성·리뷰어 검증 계획](docs/research/PAPER_SCOPE_AND_NOVELTY_2026-10-08.md) — BadFU·FUBA를 중심에 두고 같은 계열 일반화, 다른 언러닝 공격 이전, 기존 방어와의 결합 효과를 구분한 최신 제안입니다.
- [범용성 검토 PDF](output/pdf/FU_generalization_review_2026-10-08.pdf) — 교수님 피드백을 반영한 6쪽 자료: 적용 범위·근접 선행연구·확장 알고리즘·검증 순서입니다.
- [31편 확장 문헌 지도](docs/research/LITERATURE_MAP_2026-10-08.md) — 공격 구조, 필요한 정보, 비용, 확인 수준과 아직 모르는 점을 구분했습니다.
- [범용성 확장 연구 계획](docs/research/GENERALIZATION_PLAN_2026-10-08.md) — 추가 탐지 FU 0회를 유지하는 확장 제안과 반증 가능한 실험 기준입니다. 새 모듈은 아직 구현·검증되지 않았습니다.
- [미팅 핵심 요약 PDF](output/pdf/BadFU_research_meeting_2026-10-08.pdf) — 연구 목표·알고리즘·검증 계획을 3쪽으로 압축한 발표용 자료입니다.
- [연구 미팅 자료](docs/research/MEETING_BRIEF_2026-10-08.md) — 발표문, 현재 구현의 수식, 확보 근거, GPU 결과 판단 순서와 예상 질문입니다.
- [관련 논문과 알고리즘 발전 방향](docs/research/LITERATURE_REVIEW_2026-10-08.md) — 핵심 논문 13편의 역할, 가까운 선행연구, 검증해야 할 차별점을 정리했습니다.
- [GPU 실험 한 번에 실행](experiments/request_purify/README.md) — 정상 보정, BadFU 학습, 실제 FU·정화 비교, 후속 FL과 결과 zip 생성입니다.
- [교수님 논의용 연구 계획](docs/research/RESEARCH_PROPOSAL_2026-10-08.md) — 연구 문제, 예비 근거, 선행연구와 차별점, 실험 우선순위입니다.
- [시뮬레이션 없는 언러닝 정화 알고리즘 명세](docs/research/ALGORITHM_SPEC_2026-10-08.md) — 입력·점수·FU 통합·정화 강도·의사코드와 검증 조건을 정의한 구현 전 설계입니다.
- [저장소 구조와 파일 이동 안내](docs/REPOSITORY.md)
- [main의 FUBA 실행 안내](FUBA/EXPERIMENT.md)
- [공개 코드 기반 탐지·정화 관측](docs/results/DETECTION_PURIFICATION_RESULTS.md)
- [브랜치 전체의 주장과 데이터 점검](docs/research/RESEARCH_STATUS_2026-10-07.md) — `exp/evaluation` 기준이며 main에 없는 결과도 포함합니다.
- [원 공격 구현 대조와 연구 방향](docs/research/ATTACK_FIDELITY_AND_DIRECTION_2026-10-08.md) — 실제 FU의 데이터 분할·산출물 대응 및 BadFU 설정 차이를 확인했습니다.

## 디렉토리

```text
BadFU_defend/
├── README.md
├── docs/                    # 구조 안내, 연구 정리, 과거 보고서
│   ├── archive/             # 초기 README 벤치마크 기록
│   ├── research/            # 주장과 근거 점검
│   └── results/             # 실험 관측 보고서
├── experiments/             # 주제별 단독 실험 스크립트
│   ├── legacy/              # 초기 공격 재현·서버 감사·프록시
│   ├── delayed_activation/  # 삭제 이후 후속 학습과 재활성화
│   ├── detection/           # 탐지 통계·단일 프로세스 FUBA 탐색
│   └── lora/                # MNIST LoRA 실험과 분석
├── FUBA/                    # FUBA 코드와 탐지·정화 파이프라인
├── BadFU_src/               # BadFU 코드와 재현용 스크립트
├── hpc/                     # LLM·LoRA용 별도 HPC 실험
└── logs/                    # 기존 단독 실험의 결과·그림
```

기존 결과와 데이터 경로를 유지하기 위해 `logs/`, `FUBA/`, `BadFU_src/`, `hpc/`는 이동하지 않았습니다. 루트에 있던 Python 실험 22개는 `experiments/`로 옮겼습니다.

## main에서 실행하기

FUBA 파이프라인의 인자는 **실행 이름**입니다. main에는 다중 seed 제어 및 실제 FU 평가 확장이 아직 병합되지 않았습니다.

```bash
# 저장소 루트에서 실행
bash FUBA/run_experiment.sh demo_main
```

이 명령은 학습, 탐지, 근사 정화, 기존 방어 비교, 집계를 실행합니다. 결과는 `FUBA/logs/`, 라운드별 모델은 `FUBA/checkpoints/`에 저장됩니다. PyTorch/CUDA와 MPI 등 [실행 환경](FUBA/EXPERIMENT.md)이 필요합니다. main의 `run_real_fuba.sh`에는 기존 Windows 장비의 MPI 경로가 있으므로 실행 환경에 맞게 확인해야 합니다.

단독 실험은 **저장소 루트를 작업 디렉토리로** 두고 새 경로로 실행합니다.

```bash
python experiments/legacy/verify_badfu.py
python experiments/lora/analyze_lora_grid.py --dir logs/lora_grid
```

위 명령 중 첫 번째는 실제 학습과 데이터 다운로드를 수행할 수 있습니다. 분석 명령도 요약 문서와 그림을 다시 씁니다. 스크립트별 용도와 준비 사항은 [단독 실험 안내](experiments/README.md)를 참고하세요.

## 결과를 읽을 때

- main의 탐지·정화 결과는 소규모 예비 실험입니다. 기여 차감 기반 근사 언러닝과 실제 retrain 언러닝을 구분합니다.
- 초기 단독 벤치마크는 [과거 README 기록](docs/archive/INITIAL_BENCHMARKS.md)에 보존했습니다. 그 문서의 강한 표현을 현재 파이프라인 전체의 결론으로 해석하지 않습니다.
- [결과 파일 안내](logs/README.md)에서 코드·로그·요약 문서의 위치를 찾을 수 있습니다.
- 데이터셋·체크포인트·모델 파일은 Git에서 제외됩니다. clone은 코드와 추적된 결과 파일을 가져오며, 기존 실험의 모델까지 가져오지는 않습니다.

## 브랜치 역할

| 브랜치 | 역할 |
|---|---|
| `main` | 기본 코드와 공개 결과, 이번 디렉토리 정리의 기준 |
| `defense-pipeline` | seed 제어를 추가한 탐지·정화 파이프라인 |
| `exp/evaluation` | 실제 FU·오탐·탐지 오류·비용 평가 |
| `exp/weighted-removal` | 제거량 가중 탐색 |
| `chatgpt` | 이전 실험 개발 스냅샷 |

브랜치별 코드와 실행 인자가 다를 수 있습니다. 이 구조 정리는 다른 실험 브랜치의 코드를 main에 병합하지 않습니다.

## 원 연구

- [BadFU 저자 저장소](https://github.com/BingguangLu/BadFU)
- [FUBA 저자 저장소](https://github.com/stcebra/FUBA)

원 코드의 환경·인용·사용 조건은 각 하위 디렉토리의 README를 확인하세요.
