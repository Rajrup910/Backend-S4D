"""V6 audit (12 Nov 2026): vote vs single members on folds 1-4 (12,235 rows), from results/v6/preds_tta. Post hoc, descriptive.
Run from results/v6/preds_tta:  python ../../v6_audit/internal_members.py"""
import pandas as pd, numpy as np, glob
P=["p_akiec","p_bcc","p_bkl","p_df","p_mel","p_nv","p_vasc"]
def mf1(y,p,K=7):
    f=[]
    for c in range(K):
        tp=((p==c)&(y==c)).sum(); fp=((p==c)&(y!=c)).sum(); fn=((p!=c)&(y==c)).sum()
        f.append(0 if tp==0 else 2*tp/(2*tp+fp+fn))
    return np.mean(f)
mem={}
for lab,tag in [("anchor_h7","in22k_v64"),("panderm_b_h7","v64"),("dinov2_b_h7","v64"),("anchor384_h7","in22k_v64")]:
    mem[lab]=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{lab}_f[1-4]_s42_{tag}_swad_tta.csv"))]).sort_values("image_id").reset_index(drop=True)
base=mem["anchor_h7"]; y=base.y_true.values; g=base.group_id.values
probs={k:v[P].values for k,v in mem.items()}
for k,p in probs.items(): print(k,"MF1 %.4f"%mf1(y,p.argmax(1)))
vote=np.mean(list(probs.values()),0); pv=vote.argmax(1); print("uniform vote %.4f"%mf1(y,pv))
best=max(probs,key=lambda k:mf1(y,probs[k].argmax(1))); pb=probs[best].argmax(1)
ug,inv=np.unique(g,return_inverse=True); order=np.argsort(inv); bounds=np.searchsorted(inv[order],np.arange(len(ug)+1))
rng=np.random.default_rng(0); ds=[]
for _ in range(400):
    pick=rng.integers(0,len(ug),len(ug)); idx=np.concatenate([order[bounds[i]:bounds[i+1]] for i in pick])
    ds.append(mf1(y[idx],pv[idx])-mf1(y[idx],pb[idx]))
print("vote - best single (%s) = %.4f CI [%.4f %.4f]"%(best,mf1(y,pv)-mf1(y,pb),*np.percentile(ds,[2.5,97.5])))
