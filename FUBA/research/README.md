# 기존 FUBA 분석 코드

저자 원본 코드와 구분하기 위해 이 저장소에서 추가했던 23개 분석 모듈을 이 폴더로 옮겼다.
`hrac.py`, `tension_audit.py`, `experiment_hrac_unlabeled.py`의 관측 함수는 현재 탐지 실행기가 재사용한다.
과거 실험의 파일명은 출처 추적을 위해 유지하며, 새 방법의 명칭이나 검증된 성능을 뜻하지 않는다.

현재 실행은 저장소 루트의 [RUN.md](../../RUN.md)를 따른다. 과거 `experiment_tension.py`의 학습 단계처럼
이미 제거된 학습 스크립트에 의존하는 실험은 당시 보관 태그에서만 재현한다. 현재 학습은 `reproduction/training/`에서 실행한다.

원본 고정 목록은 [upstream_lock.json](../../reproduction/upstream_lock.json), 연구 결과는 [RESULTS.md](../../RESULTS.md)에 있다.
원본 파일과 이미 업로드된 로그·체크포인트는 이번 이동의 대상이 아니다.
