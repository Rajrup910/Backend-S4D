import json, numpy as np, pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, confusion_matrix, precision_recall_fscore_support
import pathlib; B=str(pathlib.Path(__file__).resolve().parents[1]/"v6")+"/"
C7=("akiec","bcc","bkl","df","mel","nv","vasc")
co=pd.read_csv(B+"hiba/cohort.csv",dtype=str); co=co[co.excluded.fillna("")==""].reset_index(drop=True)
dec=pd.read_csv(B+"hiba/decisions.csv",dtype=str)
meta=pd.read_csv(B+"archive_meta/hiba_175.csv",dtype=str)
m3={"Nevus":"nv","Melanoma, NOS":"mel","Basal cell carcinoma":"bcc","Squamous cell carcinoma, NOS":"akiec","Solar or actinic keratosis":"akiec","Seborrheic keratosis":"bkl","Solar lentigo":"bkl","Lichen planus like keratosis":"bkl","Dermatofibroma":"df"}
d=co.merge(meta[["isic_id","diagnosis_3","confirm"]],on="isic_id",how="left")
y=np.array([C7.index(m3[x]) if isinstance(x,str) else 6 for x in d.diagnosis_3])
np.save("y.npy",y)
out={}
out["support"]=dict(zip(C7,map(int,np.bincount(y,minlength=7))))
pat=d.patient_id.values
rng=np.random.default_rng(0)
up,inv=np.unique(pat,return_inverse=True); mem=[np.where(inv==i)[0] for i in range(len(up))]
def boot(fn,n=2000,cluster=True):
    v=[]
    for b in range(n):
        idx=np.concatenate([mem[i] for i in rng.integers(0,len(up),len(up))]) if cluster else rng.integers(0,len(y),len(y))
        v.append(fn(idx))
    return [float(np.percentile(v,2.5)),float(np.percentile(v,97.5))]
for k in ["v6_vote","combined_vote","v5","v1"]:
    p=dec[k+"_pred"].astype(int).values
    mf=lambda idx: f1_score(y[idx],p[idx],average="macro",labels=range(7),zero_division=0)
    pr,rc,f,s=precision_recall_fscore_support(y,p,labels=range(7),zero_division=0)
    out[k]={"macro_f1":mf(np.arange(len(y))),"macro_f1_ci_patient":boot(mf),"macro_f1_ci_lesion":boot(mf,cluster=False),
            "weighted_f1":f1_score(y,p,average="weighted"),"micro_f1_accuracy":accuracy_score(y,p),
            "balanced_accuracy":balanced_accuracy_score(y,p),
            "per_class":{c:{"P":round(pr[i],4),"R":round(rc[i],4),"F1":round(f[i],4),"n":int(s[i])} for i,c in enumerate(C7)},
            "pred_counts":dict(zip(C7,map(int,np.bincount(p,minlength=7))))}
    if k=="v6_vote": out[k]["confusion"]=confusion_matrix(y,p,labels=range(7)).tolist()
pv=dec.v6_vote_pred.astype(int).values
# leave-one-class-out sensitivity
out["v6_macro_f1_dropping_each_class"]={c:float(np.mean(np.delete(precision_recall_fscore_support(y,pv,labels=range(7),zero_division=0)[2],i))) for i,c in enumerate(C7)}
json.dump(out,open("hiba_recompute.json","w"),indent=1)
v=out["v6_vote"]
print("support",out["support"])
for k in ["v6_vote","combined_vote","v5","v1"]:
    o=out[k];print(k,"MF1 %.4f patCI %s lesCI %s wF1 %.4f acc %.4f BA %.4f"%(o["macro_f1"],np.round(o["macro_f1_ci_patient"],4),np.round(o["macro_f1_ci_lesion"],4),o["weighted_f1"],o["micro_f1_accuracy"],o["balanced_accuracy"]))
for c,r in v["per_class"].items(): print(c,r)
print(np.array(v["confusion"]))
print("drop-one",{k:round(x,4) for k,x in out["v6_macro_f1_dropping_each_class"].items()})
