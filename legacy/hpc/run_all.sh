#!/bin/bash
# HPC 에서 BadFU x 연합 LoRA x LLM 실험을 순서대로 돌린다. GPU 가 하나라 반드시 순차 실행한다.
cd ~/badfu-lora-nlp
P=.venv/bin/python
echo "###### 시작 $(date) ######"

echo "###### 1단계: 최소 재현 (r=8, 집계 2종, 잠복/재학습/대조) ######"
$P exp_llm_badfu.py --aggs fedex fedit --ranks 8 --phases dormant retrain clean --rounds 10

echo "###### 2단계: 랭크 스윕 (가설 B — 저랭크가 저장을 막는가) ######"
$P exp_llm_badfu.py --aggs fedex fedit --ranks 1 2 4 16 --phases dormant retrain --rounds 10

echo "###### 3단계: 트리거 위치 (어텐션 싱크 가설이 연합에서도 성립하는가) ######"
for pos in infix suffix; do
  $P exp_llm_badfu.py --aggs fedex --ranks 8 --phases dormant retrain --rounds 10 --trigger_pos $pos
done

echo "###### 완료 $(date) ######"
