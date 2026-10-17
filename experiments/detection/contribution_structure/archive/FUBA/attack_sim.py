# -*- coding: utf-8 -*-
"""저장된 기여 벡터만으로 만드는 적응형 공격 변형 (학습 없음). docs/design/ATTACK_VARIANTS.md D 절.

스펙 문자열: 쉼표로 이은 연산들. 인덱스는 적용 시점의 인덱스(새 클라는 끝에 붙는다).
  split:k:m      k 의 기여를 m 조각으로: U[k] ← U[k]/m, 나머지 m−1 조각을 새 클라 K, K+1, … 로 추가 (역할 상속)
                 → 위장자 분산(b), 운반자 분산(i), 쪼개기(a) 의 '여러 사람' 판
  scale:j:a      U[j] ← a·U[j]  (빼면 크게 흔들리는 미끼, c 모집단 오염)
  dir:j:k:b      U[j] ← U[j] + b·U[k]  (j 를 빼면 k 의 일부도 같이 빠진다: 기준 오염 g — k 가 위장자면 '누구를 빼도 살짝 깨어남')
  swap:j:k       U[j], U[k] 교환 (대조용)
예: "split:4:3,scale:5:2.5"   "dir:5:4:0.3,dir:6:4:0.3,dir:7:4:0.3"   "split:0:3,split:1:3,split:2:3,split:3:3"

주의: θ_T 는 그대로 둔다. 바뀐 U 는 '그 클라를 뺀 가상 모델'을 만들 때만 쓰이므로, 변형은 "제거의 효과"에 대한
가정이지 실제 학습 결과가 아니다. 학습 없이 적응형 절을 돌리기 위한 근사이며, 논문에는 그렇게 적는다.
"""
import copy


def parse_spec(spec):
    ops = []
    for tok in [t.strip() for t in (spec or "").split(",") if t.strip()]:
        parts = tok.split(":")
        name = parts[0]
        if name == "split" and len(parts) == 3:
            ops.append(("split", int(parts[1]), int(parts[2])))
        elif name == "scale" and len(parts) == 3:
            ops.append(("scale", int(parts[1]), float(parts[2])))
        elif name == "dir" and len(parts) == 4:
            ops.append(("dir", int(parts[1]), int(parts[2]), float(parts[3])))
        elif name == "swap" and len(parts) == 3:
            ops.append(("swap", int(parts[1]), int(parts[2])))
        else:
            raise ValueError(f"알 수 없는 변형: {tok}")
    return ops


def apply(U, roles, spec):
    """U: 기여 벡터 리스트, roles: {idx: 역할 문자열}. 변형된 (U, roles, 기록) 을 돌려준다 (원본은 건드리지 않음)."""
    U = [u.clone() for u in U]
    roles = dict(roles)
    log = []
    for op in parse_spec(spec):
        if op[0] == "split":
            _, k, m = op
            assert 0 <= k < len(U) and m >= 2
            piece = U[k] / m
            U[k] = piece
            new = []
            for _ in range(m - 1):
                U.append(piece.clone())
                roles[len(U) - 1] = roles.get(k, "benign")
                new.append(len(U) - 1)
            log.append({"op": "split", "client": k, "m": m, "new_clients": new})
        elif op[0] == "scale":
            _, j, a = op
            U[j] = U[j] * a
            roles[j] = roles.get(j, "benign") + "_decoy"
            log.append({"op": "scale", "client": j, "a": a})
        elif op[0] == "dir":
            _, j, k, b = op
            U[j] = U[j] + b * U[k]
            roles[j] = roles.get(j, "benign") + "_poisoned"
            log.append({"op": "dir", "client": j, "from": k, "b": b})
        elif op[0] == "swap":
            _, j, k = op
            U[j], U[k] = U[k], U[j]
            roles[j], roles[k] = roles.get(k, "benign"), roles.get(j, "benign")
            log.append({"op": "swap", "a": j, "b": k})
    return U, roles, log


def describe(spec):
    return "; ".join(f"{op[0]}({', '.join(map(str, op[1:]))})" for op in parse_spec(spec)) or "없음"


if __name__ == "__main__":
    import sys
    print(describe(sys.argv[1] if len(sys.argv) > 1 else ""))
