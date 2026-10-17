# 정화 실험

각 폴더의 README에서 가설·시도·결과·판단·보류 이유를 다섯 줄로 확인한다.

| 실험 | 현재 판단 | 이유 |
| --- | --- | --- |
| [원본 공격·궤적 저장](shared/attack_reproduction/) | 확인 | 보류 아님. 저장 무결성만으로 잠복 공격의 성공을 주장하지 않는다. |
| [공통 기여 관측·궤적 점검](shared/trajectory_tools/) | 공통 도구 | 독립 방어 기법이나 탐지 성능 결과로 세지 않는다. |
| [초기 자체 구현·점검](shared/early_reproduction/) | 보관 | 22.53→91.24% 등을 원본 BadFU 재현 성능으로 사용할 수 없다. |
| [실제 FU 뒤 정화 비교](purification/post_fu/) | 본 실험 대기 | 본 성능·망각 품질·독립 seed는 미확인. CVaR 자체는 새 방법이 아니다. |
| [요청 처리·기여 제어](purification/request_control/) | 초기 방식 보류 | 요청 처리 변경의 망각 품질과 정상 효용을 별도로 검증해야 한다. |
| [초기 삭제량·복구 제어](purification/dose_repair/) | 보관 | FU 자체 변경과 정화 효과가 섞이며 삭제 효과를 보존했는지 미확인이다. |
| [초기 LoRA·지연 재활성화](purification/early_lora_activation/) | 보관 | 공식 원본 공격·동일 삭제·정상 대조로 연결되지 않은 결과다. |

[전체 실험 기록](https://github.com/sijoon-sung/BadFU_defend/blob/main/experiments/README.md)
