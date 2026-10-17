# -*- coding: utf-8 -*-
"""
실제 FUBA 실행(mpiexec)으로 저장된 라운드별 로컬 모델에서 O 통계량 측정.
역할: 공격자 0~3, 방어자(요청자) 4, 정상 5~7, 타깃 클래스 8.
각 라운드 r 업데이트 = local_r[k] - global_{r-1}. 헤드 fc3 (weight 10x84 + bias) -> (10,85).
O_per_class(u,k,c) = max(0,-cos(row_u, row_k)) * min(||row_u||,||row_k||)   (같은 클래스·둘 다 큼·반대)
"""
import pickle, numpy as np, torch, itertools, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
NAME="det1"; TGT=8; K=8
ATK=[0,1,2,3]; DEF=4; NOR=[5,6,7]
ACTIVE=[4,5,6,7,8]   # warmup=3 -> 백도어 활성
def load_g(r): return pickle.load(open(f"checkpoints/global_net_model_{NAME}_round_{r}.pkl","rb"))
def load_l(r): return pickle.load(open(f"checkpoints/local_net_model_{NAME}_round_{r}.pkl","rb"))
def head(sd):  # (10,85)
    W=sd["fc3.weight"].float().numpy(); b=sd["fc3.bias"].float().numpy().reshape(-1,1)
    return np.concatenate([W,b],axis=1)
def unit(v):
    n=np.linalg.norm(v); return v/n if n>1e-12 else v

# 라운드별 각 클라 헤드 업데이트
HU={}  # r -> [client](10,85)
for r in ACTIVE:
    g=head(load_g(r-1)); L=load_l(r)
    HU[r]=[head(L[k])-g for k in range(K)]

def O_vec(u,k):
    acc=np.zeros(10)
    for r in ACTIVE:
        du,dk=HU[r][u],HU[r][k]
        for c in range(10):
            a,b=du[c],dk[c]
            acc[c]+=max(0.0,-float(np.dot(unit(a),unit(b))))*min(np.linalg.norm(a),np.linalg.norm(b))
    return acc/len(ACTIVE)
def cos8(u,k):  # 타깃 행 평균 코사인(음수면 반대)
    return float(np.mean([np.dot(unit(HU[r][u][TGT]),unit(HU[r][k][TGT])) for r in ACTIVE]))

pairs=list(itertools.combinations(range(K),2))
res={p:O_vec(*p) for p in pairs}
def role(u): return "ATK" if u in ATK else "REQ" if u==DEF else "nor"

print("==== 실제 FUBA: O 통계량 (타깃 =",TGT,") ====")
print("활성 라운드:",ACTIVE,"  역할: 공격자",ATK,"방어자",DEF,"정상",NOR)
ranked=sorted(pairs,key=lambda p:-res[p].max())
print("\n쌍 랭킹 (O=max_c, chat=쏠린 클래스):")
for p in ranked:
    O=res[p].max(); chat=int(res[p].argmax())
    pr="ATK-REQ" if (p[0] in ATK and p[1]==DEF) or (p[1] in ATK and p[0]==DEF) else f"{role(p[0])}-{role(p[1])}"
    mark=" <<<" if pr=="ATK-REQ" else ""
    print(f"  {p} [{pr}]: O={O:.4f} chat=c{chat} (타깃행cos={cos8(*p):+.2f}){mark}")

# 핵심: 공격자-방어자 쌍의 타깃 집중과 반대, vs 다른 쌍 유형
adp=[(a,DEF) for a in ATK]
aap=list(itertools.combinations(ATK,2))                 # 공격자-공격자 (타깃에서 같은 방향이어야)
anp=[(a,n) for a in ATK for n in NOR]
nnp=list(itertools.combinations(NOR,2))
def summ(name,ps):
    Os=[res[p].max() for p in ps]; c8=[cos8(*p) for p in ps]
    chats=[int(res[p].argmax()) for p in ps]
    frac8=np.mean([c==TGT for c in chats])
    print(f"  {name:12s}: O평균={np.mean(Os):.4f} 최대={np.max(Os):.4f} | 타깃행cos평균={np.mean(c8):+.2f} | chat=타깃비율={frac8:.2f}")
print("\n쌍 유형별 요약:")
summ("공격자-방어자",adp)
summ("공격자-공격자",aap)
summ("공격자-정상",anp)
summ("정상-정상",nnp)

# 대표 공격자-방어자 쌍 (0,4) 클래스별
p=(0,DEF); print(f"\n[대표 쌍 {p}] 클래스별 O:"); print("   "+" ".join(f"c{c}:{res[p][c]:.3f}" for c in range(10)))
print(f"   타깃 c{TGT}: O={res[p][TGT]:.3f} (전체 평균={res[p].mean():.3f}), 타깃행 cos={cos8(*p):+.2f}")
print(f"   -> 공격자-방어자가 타깃행에서 '둘 다 크게·반대'면 cos<0 이고 O가 타깃에 몰려야 함")

# 판정
ad_c8=np.mean([cos8(a,DEF) for a in ATK]); nn_c8=np.mean([cos8(*p) for p in nnp+anp])
ad_frac8=np.mean([int(res[(a,DEF)].argmax())==TGT for a in ATK])
print(f"\n판정: 공격자-방어자 타깃행cos={ad_c8:+.2f} (반대=음수 기대), 정상/교차쌍 평균={nn_c8:+.2f}")
print(f"      공격자-방어자 쌍이 타깃으로 쏠린 비율={ad_frac8:.2f}")
