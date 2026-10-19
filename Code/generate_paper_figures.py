#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, precision_recall_curve

ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/"Figures"; TAB=ROOT/"Tables"
FIG.mkdir(exist_ok=True);TAB.mkdir(exist_ok=True)
plt.rcParams.update({"font.size":9,"axes.titlesize":11,"axes.labelsize":10,"figure.dpi":160,
                     "savefig.dpi":300,"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})
colors={"Raw ITSA":"#4472C4","Optimized ITSA":"#00A6A6","Foldseek":"#ED7D31","TM-align":"#A64D79","Shuffled":"#8C8C8C","Composition":"#B7B7B7"}

# Table 1
benchmark=pd.DataFrame([
 {"Cohort":"Corrected held-out benchmark","Domains":222,"Families":28,"Superfamilies":24,"Pairs":24520,"Positive pairs":828,"Positive prevalence":828/24520}
])
benchmark.to_csv(TAB/"table_1_benchmark_design.csv",index=False)

# Tables 2 and 3
pair=pd.DataFrame([
 ["Raw ordered ITSA",0.8775053556,0.3745166004],
 ["Shuffled ITSA",0.7293551514,0.1018965136],
 ["Composition cosine",0.57616969099,0.04563361781],
 ["Exhaustive Foldseek probability",0.9853956594,0.7799058992],
 ["TM-align",0.9935130929,0.9001428752],
],columns=["Method","ROC AUC","Average precision"])
pair.to_csv(TAB/"table_2_pairwise_performance.csv",index=False)

ci=json.load(open(ROOT/"Results/ITSA/Optimized_Nested/clustered_confidence_intervals.json"))
metrics=ci["metrics"]
methods=[("Raw ITSA","ITSA"),("Optimized ITSA","Optimized ITSA"),("Foldseek","Foldseek exhaustive"),("TM-align","TM-align")]
rows=[]
for display,prefix in methods:
 row={"Method":display}
 for metric in ["AP","AUC","P@10","R@10"]:
  key=f"{prefix} {metric}"; v=metrics[key]
  row[metric]=v["estimate"];row[f"{metric} CI low"]=v["ci_low"];row[f"{metric} CI high"]=v["ci_high"]
 rows.append(row)
query=pd.DataFrame(rows)
query.to_csv(TAB/"table_3_query_retrieval_with_ci.csv",index=False)

# Table 4 model selection
selected=pd.read_csv(ROOT/"Results/ITSA/Optimized_Nested/selected_models.csv")
model_summary=pd.DataFrame([
 {"Selected feature":"Normalized ITSA alignment","Outer folds selected":24},
 {"Selected feature":"Raw alignment / geometric mean self-score","Outer folds selected":24},
 {"Selected feature":"Secondary-structure Jensen-Shannon similarity","Outer folds selected":24},
])
model_summary.to_csv(TAB/"table_4_optimized_model.csv",index=False)
selected.to_csv(TAB/"table_4b_outer_fold_selections.csv",index=False)

# Table 5 family metrics
family=pd.read_csv(ROOT/"Results/ITSA/Optimized_Nested/family_metrics.csv")
family_cols=["family_id","superfamily_id","sccs_family","family_name","domains",
 "ITSA AP","Optimized ITSA AP","Foldseek exhaustive AP","TM-align AP",
 "ITSA AUC","Optimized ITSA AUC","Foldseek exhaustive AUC","TM-align AUC"]
family[family_cols].sort_values("sccs_family").to_csv(TAB/"table_5_family_level_results.csv",index=False)

# Fig 1 substitution matrix
labels=list("HIYVPDU")
mat=np.array([[3,0,-1,-1,0,0,-1],[0,3,-1,-1,0,0,-1],[-1,-1,1,0,-1,-1,-1],
[-1,-1,0,.5,-1,-1,-1],[0,0,-1,-1,3,0,-1],[0,0,-1,-1,0,3,-1],[-1,-1,-1,-1,-1,-1,0]],float)
fig,ax=plt.subplots(figsize=(4.7,4.0));im=ax.imshow(mat,cmap="RdBu_r",vmin=-3,vmax=3)
ax.set_xticks(range(7),labels);ax.set_yticks(range(7),labels);ax.set_xlabel("Token 2");ax.set_ylabel("Token 1")
for i in range(7):
 for j in range(7):ax.text(j,i,f"{mat[i,j]:g}",ha="center",va="center",color="white" if abs(mat[i,j])>1.5 else "black")
fig.colorbar(im,ax=ax,label="Substitution score",shrink=.8);fig.tight_layout()
fig.savefig(FIG/"figure_1_substitution_matrix.png");fig.savefig(FIG/"figure_1_substitution_matrix.pdf");plt.close(fig)

# Fig 2 primary query metrics
metric_names=["AP","AUC","P@10","R@10"];x=np.arange(len(metric_names));w=.19
fig,ax=plt.subplots(figsize=(8.0,4.7))
for i,(display,prefix) in enumerate(methods):
 vals=[metrics[f"{prefix} {m}"]["estimate"] for m in metric_names]
 lo=[vals[k]-metrics[f"{prefix} {m}"]["ci_low"] for k,m in enumerate(metric_names)]
 hi=[metrics[f"{prefix} {m}"]["ci_high"]-vals[k] for k,m in enumerate(metric_names)]
 ax.bar(x+(i-1.5)*w,vals,w,label=display,color=colors[display],yerr=np.array([lo,hi]),capsize=2)
ax.set_xticks(x,["Average precision","ROC AUC","Precision@10","Recall@10"]);ax.set_ylim(0,1.08)
ax.set_ylabel("Equal-superfamily macro metric");ax.legend(ncol=2,frameon=False,loc="lower right")
ax.set_title("Held-out SCOPe family retrieval");ax.grid(axis="y",alpha=.2);fig.tight_layout()
fig.savefig(FIG/"figure_2_primary_retrieval_metrics.png");fig.savefig(FIG/"figure_2_primary_retrieval_metrics.pdf");plt.close(fig)

# Fig 3 ordered controls ROC/PR
d=pd.read_csv(ROOT/"Results/ITSA/Full_Heldout/itsa_pair_scores_with_controls.csv")
fig,axes=plt.subplots(1,2,figsize=(9,4))
control=[("Raw ITSA","itsa_normalized"),("Shuffled","shuffled_mean_normalized"),("Composition","composition_cosine")]
for name,col in control:
 fpr,tpr,_=roc_curve(d.label,d[col]);p,r,_=precision_recall_curve(d.label,d[col])
 axes[0].plot(fpr,tpr,label=name,color=colors[name],lw=2)
 axes[1].plot(r,p,label=name,color=colors[name],lw=2)
axes[0].plot([0,1],[0,1],"k--",lw=1);axes[0].set(xlabel="False-positive rate",ylabel="True-positive rate",title="ROC curves")
axes[1].axhline(d.label.mean(),color="k",ls="--",lw=1);axes[1].set(xlabel="Recall",ylabel="Precision",title="Precision-recall curves")
for ax in axes:ax.legend(frameon=False);ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(FIG/"figure_3_order_and_composition_controls.png");fig.savefig(FIG/"figure_3_order_and_composition_controls.pdf");plt.close(fig)

# Fig 4 family AP heatmap
f=family.sort_values("Optimized ITSA AP")
cols=["ITSA AP","Optimized ITSA AP","Foldseek exhaustive AP","TM-align AP"]
arr=f[cols].to_numpy()
fig,ax=plt.subplots(figsize=(7.3,9.2));im=ax.imshow(arr,aspect="auto",cmap="viridis",vmin=0,vmax=1)
ax.set_xticks(range(4),["Raw ITSA","Optimized ITSA","Foldseek","TM-align"],rotation=25,ha="right")
ax.set_yticks(range(len(f)),[f"{s}  {n}" for s,n in zip(f.sccs_family,f.family_name)],fontsize=7)
for row_no,(_,row) in enumerate(f.reset_index(drop=True).iterrows()):
 if row["Optimized ITSA AP"]>row["Foldseek exhaustive AP"]:
  ax.add_patch(plt.Rectangle((.5,row_no-.5),1,1,fill=False,edgecolor="#D62728",linewidth=2.0))
  ax.text(1.38,row_no,"*",ha="center",va="center",color="#D62728",fontweight="bold")
fig.colorbar(im,ax=ax,label="Mean query AP",shrink=.7);ax.set_title("Family-level retrieval performance\nRed outline: optimized ITSA AP exceeds Foldseek")
fig.tight_layout();fig.savefig(FIG/"figure_4_family_level_ap.png");fig.savefig(FIG/"figure_4_family_level_ap.pdf");plt.close(fig)

# Fig 5 optimization change
fig,ax=plt.subplots(figsize=(5.5,5.2));ax.scatter(f["ITSA AP"],f["Optimized ITSA AP"],c="#00A6A6",s=34,alpha=.85)
ax.plot([0,1],[0,1],"k--",lw=1)
for _,r in f.nlargest(5,"Optimized ITSA AP").iterrows():ax.annotate(r.sccs_family,(r["ITSA AP"],r["Optimized ITSA AP"]),fontsize=7,xytext=(3,3),textcoords="offset points")
ax.set(xlim=(0,1.02),ylim=(0,1.02),xlabel="Raw ITSA family AP",ylabel="Optimized ITSA family AP",title="Effect of nested three-term optimization")
ax.grid(alpha=.2);fig.tight_layout();fig.savefig(FIG/"figure_5_optimization_by_family.png");fig.savefig(FIG/"figure_5_optimization_by_family.pdf");plt.close(fig)

# Fig 6 residue-level interpretability case study
CASE_A,CASE_B="d3rdla_","d4weea1"
scoring=pd.read_csv(ROOT/"Data/scoring_input.tsv",sep="\t",dtype=str).set_index("sid")
def decode_tokens(sid):
    return [(x[0],x[1],x[2:]) for x in scoring.loc[sid,"encoded_tokens"].split(";")]

alphabet="HIYVPDU"
substitution=np.array([[3,0,-1,-1,0,0,-1],[0,3,-1,-1,0,0,-1],[-1,-1,1,0,-1,-1,-1],
 [-1,-1,0,.5,-1,-1,-1],[0,0,-1,-1,3,0,-1],[0,0,-1,-1,0,3,-1],[-1,-1,-1,-1,-1,-1,0]],float)
def token_parts(a,b):
    base=float(substitution[alphabet.index(a[0]),alphabet.index(b[0])])
    ss=.2 if a[1]==b[1]=="H" else .8 if a[1]==b[1]=="E" else 0.0 if a[1]==b[1] else -.3 if a[1] in "HE" and b[1] in "HE" else 0.0
    overlap=sum(c in b[2] for c in a[2]); bonus=.35*overlap
    return base,ss,bonus,base+ss+bonus

def trace_alignment(a,b):
    h=np.zeros((len(a)+1,len(b)+1));ptr=np.zeros_like(h,dtype=np.int8);best=(0.0,0,0)
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            vals=[0.0,h[i-1,j-1]+token_parts(a[i-1],b[j-1])[3],h[i-1,j]-2.0,h[i,j-1]-2.0]
            h[i,j]=max(vals);ptr[i,j]=vals.index(h[i,j])
            if h[i,j]>best[0]:best=(h[i,j],i,j)
    score,i,j=best;path=[]
    while i and j and h[i,j]>0:
        move=int(ptr[i,j])
        if move==1:
            base,ss,bonus,total=token_parts(a[i-1],b[j-1]);path.append((i,j,a[i-1],b[j-1],base,ss,bonus,0.0,total));i-=1;j-=1
        elif move==2:path.append((i,None,a[i-1],None,0.0,0.0,0.0,-2.0,-2.0));i-=1
        elif move==3:path.append((None,j,None,b[j-1],0.0,0.0,0.0,-2.0,-2.0));j-=1
        else:break
    return score,list(reversed(path))

standard=set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
def pdb_residues(sid):
    order=[];records={}
    for line in open(ROOT/"Data/pdb"/f"{sid}.pdb"):
        if not line.startswith("ATOM  "):continue
        resn=line[17:20].strip();chain=line[21].strip() or "_";number=line[22:26].strip()+line[26].strip();key=(chain,number,resn)
        if resn not in standard:continue
        if key not in records:order.append(key);records[key]={"label":f"{resn.title()}{number}","ca":None}
        if line[12:16].strip()=="CA" and line[16] in (" ","A"):
            records[key]["ca"]=np.array([float(line[30:38]),float(line[38:46]),float(line[46:54])])
    return [records[k] for k in order]

tok_a,tok_b=decode_tokens(CASE_A),decode_tokens(CASE_B)
raw_score,path=trace_alignment(tok_a,tok_b)
res_a,res_b=pdb_residues(CASE_A),pdb_residues(CASE_B)
assert len(tok_a)==len(res_a) and len(tok_b)==len(res_b)
assert abs(raw_score-sum(x[8] for x in path))<1e-9
case_rows=[]
for col,x in enumerate(path,1):
    ai,bj,ta,tb,base,ss,bonus,gap,total=x
    case_rows.append({"alignment_column":col,"domain_1":CASE_A,"residue_1":res_a[ai-1]["label"] if ai else "gap",
      "token_1":"".join(ta) if ta else "-","domain_2":CASE_B,"residue_2":res_b[bj-1]["label"] if bj else "gap",
      "token_2":"".join(tb) if tb else "-","matrix_score":base,"secondary_structure_adjustment":ss,
      "interaction_overlap_bonus":bonus,"gap_penalty":gap,"total_contribution":total})
pd.DataFrame(case_rows).to_csv(TAB/"table_6_case_study_alignment_trace.csv",index=False)

fig=plt.figure(figsize=(10.5,9.0));gs=fig.add_gridspec(3,2,height_ratios=[1.1,.35,1.0],hspace=.62,wspace=.08)
for grid,sid,res,color,title,start,end in [(gs[0,0],CASE_A,res_a,"#4472C4","3RDL domain d3rdla_",7,123),(gs[0,1],CASE_B,res_b,"#ED7D31","4WEE domain d4weea1",8,124)]:
    ax=fig.add_subplot(grid,projection="3d");coords=np.array([r["ca"] for r in res if r["ca"] is not None])
    ax.plot(coords[:,0],coords[:,1],coords[:,2],color="#C9C9C9",lw=1.2)
    seg=np.array([r["ca"] for r in res[start-1:end] if r["ca"] is not None]);ax.plot(seg[:,0],seg[:,1],seg[:,2],color=color,lw=2.5)
    ax.scatter(seg[::6,0],seg[::6,1],seg[::6,2],s=8,color=color);ax.set_title(title+"\nlocal alignment region",fontweight="bold")
    ax.set_axis_off();ax.view_init(elev=18,azim=-62)

ax=fig.add_subplot(gs[1,:]);token_colors=["#4C78A8","#E45756","#59A14F","#9D9D9D","#B279A2","#F2CF5B","#D9D9D9"]
token_map=np.full((2,len(path)),np.nan)
for k,x in enumerate(path):
    if x[2]:token_map[0,k]=alphabet.index(x[2][0])
    if x[3]:token_map[1,k]=alphabet.index(x[3][0])
masked=np.ma.masked_invalid(token_map);cmap=ListedColormap(token_colors);cmap.set_bad("white")
ax.imshow(masked,aspect="auto",interpolation="nearest",cmap=cmap,vmin=-.5,vmax=6.5)
ax.set_yticks([0,1],["3RDL primary tokens","4WEE primary tokens"]);ticks=np.arange(0,len(path),20);ax.set_xticks(ticks,[str(t+1) for t in ticks]);ax.set_xlabel("Local alignment column")
ax.legend(handles=[Line2D([0],[0],marker="s",color="none",markerfacecolor=color,label=label,markersize=7) for label,color in zip(alphabet,token_colors)],ncol=7,frameon=False,loc="upper center",bbox_to_anchor=(.5,1.55))

ax=fig.add_subplot(gs[2,:]);x=np.arange(1,len(path)+1);base=np.array([r[4] for r in path]);ss=np.array([r[5] for r in path]);overlap=np.array([r[6] for r in path]);gap=np.array([r[7] for r in path]);total=np.array([r[8] for r in path])
ax.plot(x,base,label="Substitution matrix",color="#4472C4",lw=1.1);ax.plot(x,ss,label="Secondary structure",color="#59A14F",lw=1.1);ax.plot(x,overlap,label="Interaction overlap",color="#ED7D31",lw=1.1);ax.plot(x,gap,label="Gap penalty",color="#C00000",lw=1.2)
ax.axhline(0,color="black",lw=.7);ax.set(xlabel="Local alignment column",ylabel="Score contribution",xlim=(1,len(path)),title="Residue-level decomposition of the local ITSA alignment")
ax.grid(alpha=.18);ax.legend(ncol=4,frameon=False,loc="upper left")
ax2=ax.twinx();ax2.plot(x,np.cumsum(total),color="black",lw=2,label="Cumulative score");ax2.set_ylabel("Cumulative raw score")
ax2.text(.99,.04,f"Raw score = {raw_score:.1f}\n116 residue pairs; 2 gaps\nMatrix 35.0 + SS 46.0 + overlap 8.4 - gaps 4.0",transform=ax2.transAxes,ha="right",va="bottom",fontsize=9,bbox={"facecolor":"white","edgecolor":"#BBBBBB","alpha":.9})
fig.suptitle("ITSA case study: an interpretable alignment in the synaptotagmin-like S family",fontsize=13,fontweight="bold",y=.99)
fig.savefig(FIG/"figure_6_interpretable_alignment_case_study.png",bbox_inches="tight");fig.savefig(FIG/"figure_6_interpretable_alignment_case_study.pdf",bbox_inches="tight");plt.close(fig)

print("generated",len(list(FIG.glob("*.png"))),"PNG figures and",len(list(TAB.glob("*.csv"))),"tables")
