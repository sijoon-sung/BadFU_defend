#!/bin/bash
# 정밀 검증. GPU 가 하나라 순차 실행한다. 이미 끝난 설정은 JSON 이 있으면 건너뛰므로 중단 후 재실행해도 이어진다.
cd ~/badfu-lora-nlp
P=.venv/bin/python
echo "###### 정밀 검증 시작 $(date) ######"

echo "###### A: 랭크 스윕 x 집계 2종 x 시드 3개 (관찰 가·나 검증) ######"
for s in 0 1 2; do
  $P exp_llm_precise.py --seeds $s --ranks 1 2 4 8 16 --aggs fedex fedit --phases dormant retrain
done

echo "###### B: 트리거 위치 x 시드 3개 (관찰 다 검증) ######"
for s in 0 1 2; do
  for pos in infix suffix; do
    $P exp_llm_precise.py --seeds $s --ranks 8 --aggs fedex --phases dormant retrain --trigger_pos $pos
  done
done

echo "###### 정밀 검증 완료 $(date) ######"
