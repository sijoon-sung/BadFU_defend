# diff_audit — 삭제 전후 차이로 좁힌 백도어 재검사

**묻는 것.** 언러닝 정의는 표준(재학습과 같음) 그대로 둔다. 그러면 위장을 지운 뒤 백도어가 켜진 모델은 정의상 "정답"이고, 막으려면 삭제할 때 다시 검사해야 한다. 그런데 요청마다 전체를 다시 검사하는 건 너무 비싸다 (UBA-Inf, BAU 가 열린 문제로 남김).

삭제 전 모델 h0 와 삭제 후 모델 h1 의 차이는 정의상 "지운 데이터가 하던 일"이다. 백도어가 켜졌다면 그 변화는 어떤 위장 방식이든 반드시 이 차이 안에 있다. **이 차이로 검사 범위를 좁혀서 싸게 잡을 수 있나?**

## 가설

- **H1 (위치).** 요청자를 지운 차이는 타깃 클래스를 가리키고, 정상 클라를 지운 차이는 아무 데나 가리킨다. → `localize[...]` 의 타깃 순위
- **H2 (비용).** 차이 상위 m 개 클래스만 검사(`localized`)해도 전체 검사(`full_diff`)만큼 요청자 삭제를 정상 삭제와 가른다. 요청당 역추적 m 회 vs 10 회.
- **H3 (확대).** 차이를 키운 모델 h0 + a(h1−h0) 에서 검사하면(`extrap_a*`) 가르기가 더 쉬워진다. 아무도 시험하지 않은 부분.
- **비교.** 차이 없이 h1 만 전체 검사하는 기존 방식(`post_only`, Neural Cleanse 식 이상지수).

## 실행

FUBA 궤적이 `checkpoints/` 에 있어야 한다 (`run_experiment.sh` 나 `run_real_fuba.sh` 로 생성).

```bash
cd FUBA
bash run_diff_audit.sh det1 det2 det3                      # universal 역추적
INV=per_sample bash run_diff_audit.sh det1 det2 det3       # 입력마다 다른 섭동 (IBA 트리거에 더 가까움)
INV=patch STEPS=300 bash run_diff_audit.sh det1 det2 det3  # Neural Cleanse 식 마스크 (BadNets 류용)
```

한 실행만: `python -m diff_audit.run --name det1 [--inv universal] [--m 2] [--rank_by logit_shift] [--alphas 2 4]`
코드 점검(무작위 모델, 숫자 의미 없음): `python -m diff_audit.smoke_test`

## 무엇을 하나

1. 궤적에서 최종 모델 theta_T 와 클라별 기여 U_k (pipeline_detect_purify.py 와 같은 1차 근사).
2. 클라 k 마다 삭제 시나리오: h0 = theta_T, h1 = theta_T − U_k. 요청자(4)는 공격이 노리는 삭제, 정상 클라(5~7)는 기준선, 공격자(0~3)는 참고.
3. **차이 점수 (방어 쪽, 트리거·정답 모름).** 깨끗한 탐침 이미지 `--probe` 장만 사용.
   - `logit_shift`: 탐침에서 h1−h0 로짓 변화의 클래스별 평균
   - `head_row`: 분류 헤드 클래스 행 변화 크기
   - `svd_head`: 헤드 변화의 첫 특이벡터 클래스별 크기 (WeightWatch 식)
4. **검사 점수.** 클래스 c 로 넘어가는 쉬움(ease)이 삭제로 얼마나 늘었나: gain(c) = ease(h1,c) − ease(h0,c). h0 쪽은 학습 끝에 한 번만 재면 된다(일회성).
5. **채점 (정답 사용).** 실제 IBA 트리거로 h0/h1/확대 모델의 ACC·ASR, 요청자 점수가 정상 삭제 점수 최대값보다 큰가(`detected`), 짚은 클래스가 타깃인가(`flag_is_target`).

## 결과 읽기

`logs/diff_audit_{name}_{inv}.json`, 집계는 `logs/diff_audit_aggregate_{inv}.json`.

| 칸 | 뜻 |
|---|---|
| `summary.methods.*.detected` | 요청자 삭제 점수 > 정상 삭제 점수 최대 |
| `summary.methods.*.flag_is_target` | 요청자 삭제에서 짚은 클래스 = 타깃(8) |
| `summary.methods.*.per_request_inversions` | 요청당 역추적 횟수 (비용) |
| `summary.localize.*.requester_target_rank` | 차이 점수로 매긴 순위에서 타깃의 위치 (0 = 1위) |
| `scenarios[].truth` | 정답 ACC/ASR (h1 ASR 이 높으면 실제로 활성된 삭제) |
| `scenarios[].layer_energy` | 층별 변화 비율 |

## 주의

- **삭제는 1차 근사(빼기)다.** 실제 재학습이 아니다. 정확한 재학습이면 시드·순서 잡음이 차이에 섞이므로 정상 삭제 기준선이 더 중요해진다.
- **기준선이 작다.** 정상 클라가 3명뿐이라 "정상 삭제 최대보다 크다"는 판정은 약하다. 정상 클라를 늘린 궤적이나 정상 데이터 무작위 삭제를 추가해야 함.
- **역추적과 실제 트리거가 다르다.** IBA 는 입력마다 다른 미세 잡음이라 universal 섭동은 과소평가할 수 있다. per_sample 은 자연 적대 취약성까지 섞인다. 둘 다 돌려서 비교.
- **적응 공격 미시험.** 변화를 여러 층에 흩거나 정상 삭제처럼 보이게 하는 공격은 아직 없다.
