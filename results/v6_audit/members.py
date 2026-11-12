import json, glob, numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score
import pathlib; B=str(pathlib.Path(__file__).resolve().parents[1]/"v6")+"/"
y=np.load("y.npy")
st=json.load(open(B+"refit/v6_vote/dirichlet_state.json"))["state"]
W=np.array(st["weight"]); b=np.array(st["bias"])
def cal(p):
    z=np.log(np.clip(p,1e-12,None))@W.T+b; z-=z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
def mf(p): return f1_score(y,p.argmax(1),average="macro",labels=range(7),zero_division=0)
I=B+"hiba/inference/"
mem={}
for lab in ["anchor_h7","panderm_b_h7","dinov2_b_h7","anchor384_h7"]:
    fs=sorted(f for f in glob.glob(I+"v6__*.npy") if f"_{lab}_f" in f)
    arr=[np.load(f) for f in fs]
    mem[lab]=np.mean(arr,0)
    print(lab,len(fs),"5-fold mean MF1 %.4f BA %.4f"%(mf(mem[lab]),balanced_accuracy_score(y,mem[lab].argmax(1))),"single folds",[round(mf(a),4) for a in arr])
vote=np.mean(list(mem.values()),0)
print("uniform vote raw MF1 %.4f ; calibrated (deployed) MF1 %.4f"%(mf(vote),mf(cal(vote))))
dec=__import__("pandas").read_csv(B+"hiba/decisions.csv")
print("matches decisions.csv:",(cal(vote).argmax(1)==dec.v6_vote_pred.values).mean())
# paired bootstrap vs best single member (raw both)
import pandas as pd
co=pd.read_csv(B+"hiba/cohort.csv",dtype=str); co=co[co.excluded.fillna("")==""].reset_index(drop=True)
up,inv=np.unique(co.patient_id.values,return_inverse=True); M=[np.where(inv==i)[0] for i in range(len(up))]
rng=np.random.default_rng(1)
best=max(mem,key=lambda k:mf(mem[k]))
pv,pb=cal(vote).argmax(1),mem[best].argmax(1)
ds=[]
for _ in range(2000):
    idx=np.concatenate([M[i] for i in rng.integers(0,len(up),len(up))])
    ds.append(f1_score(y[idx],pv[idx],average="macro",labels=range(7),zero_division=0)-f1_score(y[idx],pb[idx],average="macro",labels=range(7),zero_division=0))
print("deployed vote - best single member (%s): %.4f CI [%.4f, %.4f]"%(best,mf(cal(vote))-mf(mem[best]),*np.percentile(ds,[2.5,97.5])))
for k in mem:
    others=[mem[j] for j in mem if j!=k]
    print("leave-out",k,"raw vote MF1 %.4f"%mf(np.mean(others,0)))
v1=[np.load(f) for f in glob.glob(I+"v1__*.npy")]; print("V1 members", [round(mf(a),3) for a in v1])

def ece(p,y,b=15):
    c=p.max(1); a=(p.argmax(1)==y); e=0
    edges=np.linspace(0,1,b+1)
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(c>lo)&(c<=hi)
        if m.any(): e+=m.mean()*abs(a[m].mean()-c[m].mean())
    return e, float(c.mean()-a.mean())
print("HIBA ECE raw vote (ece, signed gap conf-acc):",ece(vote,y)," calibrated:",ece(cal(vote),y))
