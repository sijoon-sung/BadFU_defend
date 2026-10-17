# 탐지 실험

각 폴더의 README에서 가설·시도·결과·판단·보류 이유를 다섯 줄로 확인한다.

| 실험 | 현재 판단 | 이유 |
| --- | --- | --- |
| [원본 공격·궤적 저장](shared/attack_reproduction/) | 확인 | 보류 아님. 저장 무결성만으로 잠복 공격의 성공을 주장하지 않는다. |
| [공통 기여 관측·궤적 점검](shared/trajectory_tools/) | 공통 도구 | 독립 방어 기법이나 탐지 성능 결과로 세지 않는다. |
| [초기 자체 구현·점검](shared/early_reproduction/) | 보관 | 22.53→91.24% 등을 원본 BadFU 재현 성능으로 사용할 수 없다. |
| [HRAC·기여 관계 관측](detection/hrac/) | 추가 이득 미입증 | 긴 이력의 추가 이득과 독립 정상 오탐률이 확인되지 않았다. |
| [요청 기반 입력 탐색](detection/request_probe/) | 측정 도구 | 탐색 반응을 실제 트리거 ASR이나 정상/공격 구분으로 볼 수 없다. |
| [시계열 GLR](detection/temporal_history/) | 보류 | 시간을 추가한 이득이 단순 변화량 대비 확인되지 않았다. |
| [이력 관계 원인 분석](detection/history_relations/) | 관측 보관 | 같은 관계가 정상 Non-IID에서도 발생할 수 있어 식별 기준이 더 필요하다. |
| [FUBA ASR 원인 분해](detection/asr_diagnosis/) | 진단 완료 | 보류 아님. 실제 FU·정상 대조를 함께 보는 평가가 필요하다. |
| [정상 설명 집합·로짓 이중 차분](detection/deletion_comparison/) | 일부 관측 / 가정 실패 | 누적 차감은 정상 모델도 붕괴시켰다. 패치 반응 증가만으로 탐지 성공을 입증할 수 없다. |
| [초기 반상관·프록시 탐지](detection/early_proxies/) | 보관 | 원본 공격·강한 Non-IID·독립 정상 대조로 같은 결론을 검증하지 못했다. |
| [초기 기여 구조·잔류 위험](detection/contribution_structure/) | 보관 | 점수 크기만으로 정상 Non-IID와 공격 의도를 분리할 수 없다. |
| [짝지은 반응 관측](detection/paired_response/) | 보관 | 공식 공격·독립 정상 요청에서 우월성이 확인되지 않았다. |
| [초기 탐지 비교군](detection/legacy_baselines/) | 보관 | 같은 원본 모델·요청·자료 분할·튜닝 예산으로 다시 맞춰야 한다. |
| [초기 패턴 역탐색](detection/trigger_inversion/) | 보관 | 발견 패턴의 반응을 원본 트리거 ASR과 혼동할 수 있다. |

[전체 실험 기록](https://github.com/sijoon-sung/BadFU_defend/blob/main/experiments/README.md)
