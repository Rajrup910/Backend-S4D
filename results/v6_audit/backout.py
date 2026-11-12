import numpy as np, pandas as pd, itertools
from sklearn.metrics import f1_score
import pathlib; B=str(pathlib.Path(__file__).resolve().parents[1]/"v6")+"/"
C7=("akiec","bcc","bkl","df","mel","nv","vasc")
co=pd.read_csv(B+"hiba/cohort.csv",dtype=str); co=co[co.excluded.fillna("")==""].reset_index(drop=True)
dec=pd.read_csv(B+"hiba/decisions.csv",dtype=str)
meta=pd.read_csv(B+"archive_meta/hiba_175.csv",dtype=str)
m3={"Nevus":"nv","Melanoma, NOS":"mel","Basal cell carcinoma":"bcc","Squamous cell carcinoma, NOS":"akiec","Solar or actinic keratosis":"akiec",
    "Seborrheic keratosis":"bkl","Solar lentigo":"bkl","Lichen planus like keratosis":"bkl","Dermatofibroma":"df"}
d=co.merge(meta[["isic_id","diagnosis_3","confirm"]],on="isic_id",how="left")
d["c7"]=d.diagnosis_3.map(m3)
known=d.c7.notna().values
esc_known=d.c7.isin(["mel","bcc","akiec"]).sum()
print("known rows",known.sum(),"escalating known",esc_known,"=> unknown that are mel:",507-esc_known)
histo=(d.confirm=="histopathology")
mech_known=((d.c7=="mel")|((d.c7=="nv")&histo)).sum()
print("mechanism rows known",mech_known,"report 311")
unk=d[~known]
print(unk.confirm.value_counts(dropna=False).to_dict(), "age",unk.age_band.value_counts().to_dict())
preds={k:dec[k+"_pred"].astype(int).values for k in ["v6_vote","combined_vote","v5","v1"]}
target={"v6_vote":0.7347557226246233,"combined_vote":0.7221753871713832,"v5":0.5714805479534179,"v1":0.47180805819065924}
nmel=507-esc_known
ui=np.where(~known)[0]
# group unknown rows by tuple of predictions across systems
keys=[tuple(preds[k][i] for k in preds) for i in ui]
from collections import Counter
cnt=Counter(keys); groups=list(cnt.items()); print(len(groups),"pred-pattern groups")
y0=np.array([C7.index(c) if isinstance(c,str) else -1 for c in d.c7])
sols=[]
def rec(gi,left,assign):
    if gi==len(groups):
        if left==0: sols.append(dict(assign))
        return
    g,n=groups[gi]
    for k in range(0,min(n,left)+1):
        assign[g]=k; rec(gi+1,left-k,assign)
rec(0,nmel,{})
print("candidate assignments",len(sols))
good=[]
for s in sols:
    y=y0.copy(); used=Counter()
    for i,kk in zip(ui,keys):
        if used[kk]<s[kk]: y[i]=4; used[kk]+=1
        else: y[i]=5
    ok=all(abs(f1_score(y,preds[k],average="macro",labels=range(7),zero_division=0)-target[k])<1e-9 for k in preds)
    if ok: good.append(y)
print("assignments reproducing all four reported Macro-F1s exactly:",len(good))
if good:
    np.save("y_reconstructed.npy",good[0])
    # check all good give same per-class metrics
    from sklearn.metrics import precision_recall_fscore_support
    fs=[tuple(np.round(precision_recall_fscore_support(g,preds["v6_vote"],labels=range(7),zero_division=0)[2],6)) for g in good]
    print("distinct per-class F1 vectors among solutions:",len(set(fs)))

print("---- uniform hypotheses for the 40 rows")
for cls in ["nv","bkl","df","vasc"]:
    y=y0.copy(); y[ui]=C7.index(cls)
    print(cls,{k:round(f1_score(y,preds[k],average="macro",labels=range(7),zero_division=0),6) for k in preds})
print("target",{k:round(v,6) for k,v in target.items()})
print("v6 preds on the 40:",Counter(C7[p] for p in preds["v6_vote"][ui]))
