# 하나의 절차 (unified_fu) 요약: iid  n=3

삭제 전 ASR 6.1 ± 1.8 / ACC 95.0 ± 0.3

| 요청자 | 역할 | 결정 (시드별) | 거름망 통과 | 확정 z (최대) | ASR | ACC | 유지 변화 | α | 자기 데이터 acc | 운반자 | 모델 / 순전파 / 초 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | suppressor | purified(antidote); purified(antidote); purified(antidote) | 0/3 | 11.9 ± 1.1 | 4.0 ± 5.6 | 85.2 ± 3.9 | 12.59 ± 5.04 | 0.003 ± 0.003 | 84.6 ± 4.1 | [[3, 1]]; [[3, 2]]; [[1, 2]] | 8 ± 0 / 15 ± 0 / 0.7 ± 0.0 |
| 7 | benign | accept(screen); accept(screen); accept(screen) | 3/3 | - | 13.6 ± 5.8 | 93.7 ± 1.1 | 2.99 ± 1.02 | 0.000 ± 0.000 | 93.2 ± 0.9 | - | 0 ± 0 / 0 ± 0 / 0.5 ± 0.1 |
| 5 | benign | accept(screen); accept(confirm); accept(confirm) | 1/3 | 0.4 ± 0.0 | 13.9 ± 4.8 | 93.9 ± 0.6 | 2.75 ± 0.46 | 0.000 ± 0.000 | 93.3 ± 0.4 | - | 0 ± 0 / 5 ± 3 / 0.5 ± 0.0 |

결정 뜻: accept(screen)=거름망 통과·기본 FU 와 동일 / accept(confirm)=확인 뒤 수락 / purified(antidote)=운반자 성분 제거 후 수락 / repaired(KD)=재억제 수정 후 수락 / reject(...)=불출