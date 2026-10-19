#!/usr/bin/env python3
"""Standalone ITSA benchmark/model-selection program.

This consolidates corrected-benchmark evaluation, query metrics, nested
held-out-superfamily feature selection, clustered bootstrap confidence
intervals, family tables, and final result export into one script.

Nested held-out-superfamily selection of ITSA feature sets.

Each outer model is tested only on queries from one SCOPe superfamily. Every pair
touching that superfamily is removed from model fitting. Candidate term sets and
regularization are selected inside the remaining superfamilies using grouped
inner validation. The resulting predictions are strictly out of fold.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"Data/itsa_feature_matrix.csv"
META_SRC=ROOT/"Results/ITSA/Full_Heldout/query_metrics.csv"
FOLD_Q=ROOT/"Results/Foldseek/Full_Heldout/query_metrics_exhaustive.csv"
TM_Q=ROOT/"Results/TM-align/Full_Heldout/query_metrics.csv"
RAW_Q=ROOT/"Results/ITSA/Full_Heldout/query_metrics.csv"
NAMES=ROOT/"Results/ITSA/Full_Heldout/family_metrics.csv"
OUT=ROOT/"Results/ITSA/Optimized_Nested"
RNG=np.random.default_rng(20261003)

BASE=["alignment_norm","alignment_raw_geom","js_all","js_ss","js_joint","repeat_x_joint_js"]
OPTIONAL=["length_similarity","js_primary","js_nint","repeat_compress_max","repeat_x_raw_minus_norm","raw_x_primary_js"]
CANDIDATES={
 "one_raw":["alignment_raw_geom"],
 "two_alignment":["alignment_norm","alignment_raw_geom"],
 "three_align_ss":["alignment_norm","alignment_raw_geom","js_ss"],
 "four_core":["alignment_norm","alignment_raw_geom","js_all","js_ss"],
 "five_core_joint":["alignment_norm","alignment_raw_geom","js_all","js_ss","js_joint"],
 "six_original":BASE,
}
for x in OPTIONAL:
    CANDIDATES["seven_plus_"+x]=BASE+[x]
for i,x in enumerate(OPTIONAL):
    for y in OPTIONAL[i+1:]:
        CANDIDATES["eight_plus_"+x+"_"+y]=BASE+[x,y]
CANDIDATES["eight_original"]=["alignment_norm","alignment_raw_geom","js_all","js_ss","js_joint","repeat_compress_max","repeat_x_raw_minus_norm","repeat_x_joint_js"]
CANDIDATES["ten_original"]=["alignment_norm","alignment_raw_geom","length_similarity","js_all","js_ss","js_joint","repeat_compress_max","repeat_x_raw_minus_norm","repeat_x_joint_js","raw_x_primary_js"]
CANDIDATES["twelve_all"]=BASE+OPTIONAL
CS=[0.03,0.1,0.3,1.0,3.0,10.0]
TOLERANCE=0.005
INNER_FOLDS=3

def model(c):
    return make_pipeline(StandardScaler(),LogisticRegression(C=c,class_weight="balanced",solver="lbfgs",max_iter=3000))

def directional_rows(pairs, indices, scores, query_sfs):
    z=pairs.iloc[indices][["sid_1","sid_2","family_1","family_2","sf_1","sf_2","label"]].copy()
    z["score"]=np.asarray(scores,float)
    left=z[z.sf_1.isin(query_sfs)][["sid_1","sid_2","family_1","sf_1","label","score"]]
    left.columns=["sid","target","family_id","superfamily_id","label","score"]
    right=z[z.sf_2.isin(query_sfs)][["sid_2","sid_1","family_2","sf_2","label","score"]]
    right.columns=["sid","target","family_id","superfamily_id","label","score"]
    return pd.concat([left,right],ignore_index=True)

def query_table(d):
    rows=[]
    for sid,g in d.groupby("sid",sort=False):
        y=g.label.to_numpy(int); s=g.score.to_numpy(float)
        order=np.lexsort((g.target.astype(str).to_numpy(),-s)); k=min(10,len(y))
        rows.append({"sid":sid,"family_id":g.family_id.iloc[0],"superfamily_id":g.superfamily_id.iloc[0],
          "gallery_size":len(g),"positives":int(y.sum()),"Optimized ITSA AP":average_precision_score(y,s),
          "Optimized ITSA AUC":roc_auc_score(y,s),"Optimized ITSA P@10":float(y[order[:k]].mean()),
          "Optimized ITSA R@10":float(y[order[:k]].sum()/y.sum())})
    return pd.DataFrame(rows)

def macro_ap(d):
    if d.empty:return -np.inf
    rows=[]
    for (sf,sid),g in d.groupby(["superfamily_id","sid"],sort=False):
        y=g.label.to_numpy(int); s=g.score.to_numpy(float); positives=int(y.sum())
        order=np.argsort(-s,kind="mergesort"); ys=y[order]; ss=s[order]
        ends=np.r_[np.flatnonzero(ss[1:]!=ss[:-1]),len(ss)-1]
        tp=np.cumsum(ys)[ends]; delta=np.diff(np.r_[0,tp]); total=ends+1
        ap=float(np.sum((delta/positives)*(tp/total)))
        rows.append((sf,ap))
    return float(pd.DataFrame(rows,columns=["sf","ap"]).groupby("sf").ap.mean().mean())

def bootstrap(q,cols,contrasts,reps=2000):
    sfvals=q.groupby("superfamily_id")[cols].mean(); sfs=sfvals.index.to_numpy()
    vals={c:[] for c in cols}; diffs={n:[] for n in contrasts}
    for _ in range(reps):
        draw=RNG.choice(sfs,len(sfs),replace=True)
        m={c:float(sfvals.loc[draw,c].mean()) for c in cols}
        for c in cols: vals[c].append(m[c])
        for n,(a,b) in contrasts.items():diffs[n].append(m[a]-m[b])
    metric={c:{"estimate":float(sfvals[c].mean()),"ci_low":float(np.percentile(v,2.5)),"ci_high":float(np.percentile(v,97.5))} for c,v in vals.items()}
    contrast={n:{"estimate":float(sfvals[a].mean()-sfvals[b].mean()),"ci_low":float(np.percentile(diffs[n],2.5)),"ci_high":float(np.percentile(diffs[n],97.5))} for n,(a,b) in contrasts.items()}
    return {"bootstrap_unit":"SCOPe superfamily","replicates":reps,"seed":20261003,"metrics":metric,"contrasts":contrast}

OUT.mkdir(parents=True,exist_ok=True)
pairs=pd.read_csv(SRC)
meta=pd.read_csv(META_SRC,dtype={"family_id":str,"superfamily_id":str}).set_index("sid")
for side in ["1","2"]:
    pairs[f"family_{side}"]=pairs[f"sid_{side}"].map(meta.family_id)
    pairs[f"sf_{side}"]=pairs[f"sid_{side}"].map(meta.superfamily_id)
features=sorted(set(sum(CANDIDATES.values(),[])))
missing=set(features)-set(pairs.columns)
if missing:raise RuntimeError(f"missing features: {missing}")
X=pairs[features]; y=pairs.label.to_numpy(int)
sfs=sorted(meta.superfamily_id.unique())
all_directional=[]; selections=[]; search_rows=[]

for outer_no,outer in enumerate(sfs,1):
    eligible=(pairs.sf_1.ne(outer)&pairs.sf_2.ne(outer)).to_numpy()
    remaining=[sf for sf in sfs if sf!=outer]
    inner_groups=[set(remaining[i::INNER_FOLDS]) for i in range(INNER_FOLDS)]
    choices=[]
    for cname,columns in CANDIDATES.items():
        for c in CS:
            fold_scores=[]
            for group in inner_groups:
                train=eligible & ~pairs.sf_1.isin(group).to_numpy() & ~pairs.sf_2.isin(group).to_numpy()
                valid_idx=np.flatnonzero(eligible & (pairs.sf_1.isin(group).to_numpy()|pairs.sf_2.isin(group).to_numpy()))
                m=model(c);m.fit(X.loc[train,columns],y[train])
                pred=m.predict_proba(X.loc[valid_idx,columns])[:,1]
                fold_scores.append(macro_ap(directional_rows(pairs,valid_idx,pred,group)))
            score=float(np.mean(fold_scores))
            choices.append((score,len(columns),c,cname,columns))
            search_rows.append({"outer_superfamily":outer,"candidate":cname,"terms":len(columns),"C":c,"inner_macro_AP":score})
    best=max(x[0] for x in choices)
    eligible_choices=[x for x in choices if x[0]>=best-TOLERANCE]
    selected=min(eligible_choices,key=lambda x:(x[1],-x[0],x[2],x[3]))
    inner_score,nterms,c,cname,columns=selected
    m=model(c);m.fit(X.loc[eligible,columns],y[eligible])
    test_idx=np.flatnonzero((pairs.sf_1.eq(outer)|pairs.sf_2.eq(outer)).to_numpy())
    pred=m.predict_proba(X.loc[test_idx,columns])[:,1]
    all_directional.append(directional_rows(pairs,test_idx,pred,{outer}))
    selections.append({"outer_superfamily":outer,"candidate":cname,"terms":nterms,"C":c,
                       "inner_macro_AP":inner_score,"best_inner_macro_AP":best,
                       "training_pairs":int(eligible.sum()),"test_queries":int((meta.superfamily_id==outer).sum())})
    print(f"[{outer_no:02d}/{len(sfs)}] {outer}: {cname}, {nterms} terms, C={c:g}, inner AP={inner_score:.4f}",flush=True)

directional=pd.concat(all_directional,ignore_index=True)
if len(directional)!=2*len(pairs):raise RuntimeError(f"directional coverage {len(directional)}")
directional.to_csv(OUT/"directional_oof_scores.csv",index=False)
pd.DataFrame(selections).to_csv(OUT/"selected_models.csv",index=False)
pd.DataFrame(search_rows).to_csv(OUT/"inner_model_search.csv",index=False)
q=query_table(directional)
raw=pd.read_csv(RAW_Q,dtype={"family_id":str,"superfamily_id":str})
fold=pd.read_csv(FOLD_Q,dtype={"family_id":str,"superfamily_id":str})
tm=pd.read_csv(TM_Q,dtype={"family_id":str,"superfamily_id":str})
q=q.merge(raw[["sid","ITSA AP","ITSA AUC","ITSA P@10","ITSA R@10"]],on="sid",validate="one_to_one")
q=q.merge(fold[["sid","Foldseek exhaustive AP","Foldseek exhaustive AUC","Foldseek exhaustive P@10","Foldseek exhaustive R@10"]],on="sid",validate="one_to_one")
q=q.merge(tm[["sid","TM-align AP","TM-align AUC","TM-align P@10","TM-align R@10"]],on="sid",validate="one_to_one")
q.to_csv(OUT/"query_metrics.csv",index=False)
metric_cols=["Optimized ITSA AP","Optimized ITSA AUC","Optimized ITSA P@10","Optimized ITSA R@10",
 "ITSA AP","ITSA AUC","ITSA P@10","ITSA R@10",
 "Foldseek exhaustive AP","Foldseek exhaustive AUC","Foldseek exhaustive P@10","Foldseek exhaustive R@10",
 "TM-align AP","TM-align AUC","TM-align P@10","TM-align R@10"]
contrasts={"Optimized minus raw ITSA AP":("Optimized ITSA AP","ITSA AP"),
 "Optimized minus Foldseek AP":("Optimized ITSA AP","Foldseek exhaustive AP"),
 "Optimized minus TM-align AP":("Optimized ITSA AP","TM-align AP")}
ci=bootstrap(q,metric_cols,contrasts)
(OUT/"clustered_confidence_intervals.json").write_text(json.dumps(ci,indent=2))
family=q.groupby(["family_id","superfamily_id"],as_index=False).agg(domains=("sid","size"),**{c:(c,"mean") for c in metric_cols})
names=pd.read_csv(NAMES,dtype={"family_sunid":str})[["family_sunid","sccs_family","family_name"]].drop_duplicates()
family=family.merge(names,left_on="family_id",right_on="family_sunid",how="left").drop(columns="family_sunid")
family.to_csv(OUT/"family_metrics.csv",index=False)
sel=pd.DataFrame(selections)
summary={"design":"Nested model/regularization selection with strictly held-out SCOPe superfamilies",
 "outer_folds":len(sfs),"inner_folds":INNER_FOLDS,"candidate_term_sets":len(CANDIDATES),"regularization_values":CS,
 "selection_rule":f"Fewest terms within {TOLERANCE} inner macro AP of the best candidate",
 "queries":len(q),"directional_scores":len(directional),
 "macro_equal_superfamily":{c:float(q.groupby("superfamily_id")[c].mean().mean()) for c in metric_cols},
 "selected_term_counts":{str(k):int(v) for k,v in sel.terms.value_counts().sort_index().items()},
 "selected_candidates":{str(k):int(v) for k,v in sel.candidate.value_counts().items()},
 "family_AP_wins":{"vs_raw_ITSA":int((family["Optimized ITSA AP"]>family["ITSA AP"]).sum()),
  "vs_Foldseek_exhaustive":int((family["Optimized ITSA AP"]>family["Foldseek exhaustive AP"]).sum()),
  "vs_TM_align":int((family["Optimized ITSA AP"]>family["TM-align AP"]).sum()),"families":len(family)}}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
print(json.dumps(ci["contrasts"],indent=2))
