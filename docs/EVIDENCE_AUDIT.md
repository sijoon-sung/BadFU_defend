# 과거 결과 정정

현재 파일 판단은 [궤적 상태](TRAJECTORIES.md)를 따릅니다. 과거 탐지 실험을 없었던 것으로 설명한 것은 잘못입니다.

- **FUBA 장부:** 개발 궤적 6개에서 요청자·표적·운반자를 맞힌 결과가 있습니다. 독립 정상 FPR 검증과는 다릅니다. [원본 폴더](../FUBA/logs/ledger/).
- **v3 비교:** 20라운드에서 요청자 1위는 v3·선형 모두 6/6, 요청자와 표적 동시 일치는 각각 6/6·0/6입니다. 같은 개발 자료의 순위 비교입니다. [원본 JSON](../FUBA/logs/tension_audit/historical_ablation_v3.json).
- **BadFU 40R:** 공격 3개 중 2개 경보, 정상 3개 중 2개 경보. 완료된 실험이 보관 태그에 있습니다. [원본 summary](https://github.com/sijoon-sung/BadFU_defend/blob/a0371dbf511a265250fd12f6accda026414ef219/results_observe/summary.json).
- **BadFU 15R:** 기존 입력에서의 단일 성공은 새 입력에서 유지되지 않았습니다. [후속 원본](../FUBA/logs/response_audit/badfu.json).
- **CIFAR 장부:** 요약 MD는 `h0=.8, 무경보`, 원시 JSON은 `h0=.75, 잘못된 클래스·클라이언트 경보`로 서로 다릅니다. 하나의 동일 실험으로 합치지 않습니다. [원본 JSON](../FUBA/logs/ledger/bd_iid_r100w5_k20paper_cifar_s0.json).

ASR 22.53→91.24%는 초기 자체 MNIST 구현의 기록이며 공식 BadFU 원본 재현의 근거가 아닙니다. [기존 출처 감사](archive/cleanup_2026-10-15/EVIDENCE_AUDIT.md)에 세부 내역을 보관했습니다.
