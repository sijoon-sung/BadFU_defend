# 하나의 절차 (unified_fu) 요약: owner  n=3

삭제 전 ASR 8.0 ± 1.1 / ACC 92.5 ± 0.8

| 요청자 | 역할 | 결정 (시드별) | 거름망 통과 | 확정 z (최대) | ASR | ACC | 유지 변화 | α | 자기 데이터 acc | 운반자 | 모델 / 순전파 / 초 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | suppressor | purified(antidote+antidote); purified(antidote); purified(antidote+antidote) | 0/3 | 7.6 ± 2.4 | 21.2 ± 4.5 | 83.4 ± 2.8 | 11.12 ± 1.98 | 0.000 ± 0.000 | 87.5 ± 1.2 | [[2, 3, 1], [1]]; [[1, 3, 0]]; [[2], [3]] | 19 ± 8 / 35 ± 14 / 0.5 ± 0.1 |
| 7 | benign | accept(confirm); accept(confirm); purified(antidote) | 0/3 | 6.7 ± 7.3 | 19.9 ± 9.0 | 83.0 ± 0.6 | 12.18 ± 0.76 | 0.003 ± 0.004 | 50.3 ± 0.5 | [[3, 2, 1]] | 0 ± 0 / 7 ± 0 / 0.3 ± 0.0 |
| 5 | benign | accept(confirm); accept(confirm); accept(confirm) | 0/3 | 0.8 ± 0.9 | 15.5 ± 7.0 | 91.0 ± 1.1 | 3.76 ± 1.64 | 0.000 ± 0.000 | 92.9 ± 0.8 | - | 0 ± 0 / 7 ± 0 / 0.3 ± 0.0 |

결정 뜻: accept(screen)=거름망 통과·기본 FU 와 동일 / accept(confirm)=확인 뒤 수락 / purified(antidote)=운반자 성분 제거 후 수락 / repaired(KD)=재억제 수정 후 수락 / reject(...)=불출