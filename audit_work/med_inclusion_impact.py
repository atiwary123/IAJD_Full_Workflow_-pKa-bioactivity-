#!/usr/bin/env python3
"""Quick approximate impact test: does INCLUDING the 17 MED (inference-grade) rows in training
change conclusions? Surrogate models on RDKit-2D/3D features (not the production stack).

Reports, for pKa (full coverage) and bioact log10_flux_total:
  (1) accuracy on VERIFIED rows: K-fold CV MAE/R2, training WITHOUT vs WITH the MED rows added
      to each training fold (test folds are always verified-only) -> does MED help/hurt?
  (2) MED predictability: train on verified, predict MED rows (are they wildly off?)
  (3) conclusion stability: top-10 feature-importance overlap + Spearman, verified vs +MED
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr

MED=[26,27,28,29,38,39,40,41,42,43,47,48,49,50,51,53,54]
FEATS=['ExactMolWt','HeavyAtomCount','NumNitrogens','MolLogP','TPSA','LabuteASA','FractionCSP3',
 'RotatableBonds','BertzCT','Chi0v','Chi1v','HallKierAlpha','NumAromaticRings','NumHDonors',
 'NumHAcceptors','NumEsters','NumAmides','NumEthers','NumTertiaryAmines','NumAmines_total',
 'Hydrophobic_Index','Polar_Surface_Ratio','HBD_HBA_Ratio','Taft_Steric_Sum',
 'Rg_3D','Asphericity_3D','Pct_V_Bur_mean','N_basic_N_3D']

def run(path, idcol, target, label):
    df=pd.read_excel(path)
    feats=[c for c in FEATS if c in df.columns]
    d=df[[idcol,target]+feats].copy()
    d=d[d[target].notna()]
    for c in feats: d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna(subset=feats)
    is_med=d[idcol].isin(MED)
    V=d[~is_med]; M=d[is_med]
    print(f"\n{'='*78}\n{label}: target={target}  verified n={len(V)}  MED n={len(M)}  feats={len(feats)}\n{'='*78}")
    if len(M)==0:
        print("  no MED rows with target -> skip"); return

    Xv,yv=V[feats].values, V[target].values
    Xm,ym=M[feats].values, M[target].values

    # (1) K-fold CV on verified; train fold WITHOUT vs WITH MED appended
    kf=KFold(n_splits=10, shuffle=True, random_state=0)
    base_ae, aug_ae, base_p, aug_p, ytrue=[],[],[],[],[]
    for tr,te in kf.split(Xv):
        ytrue.append(yv[te])
        rf=RandomForestRegressor(n_estimators=300,random_state=0,n_jobs=-1).fit(Xv[tr],yv[tr])
        base_p.append(rf.predict(Xv[te]))
        rf2=RandomForestRegressor(n_estimators=300,random_state=0,n_jobs=-1).fit(
            np.vstack([Xv[tr],Xm]), np.concatenate([yv[tr],ym]))
        aug_p.append(rf2.predict(Xv[te]))
    yt=np.concatenate(ytrue); bp=np.concatenate(base_p); ap=np.concatenate(aug_p)
    print(f"  (1) accuracy on VERIFIED rows (10-fold CV):")
    print(f"        baseline (verified-only train):  MAE {mean_absolute_error(yt,bp):.4f}  R2 {r2_score(yt,bp):.3f}")
    print(f"        +MED in training folds:          MAE {mean_absolute_error(yt,ap):.4f}  R2 {r2_score(yt,ap):.3f}")
    print(f"        delta MAE = {mean_absolute_error(yt,ap)-mean_absolute_error(yt,bp):+.4f}  "
          f"(>+0.02 worse worth noting; >+0.05 disruptive)")

    # (2) MED predictability: train on all verified, predict MED
    rf=RandomForestRegressor(n_estimators=400,random_state=0,n_jobs=-1).fit(Xv,yv)
    pm=rf.predict(Xm)
    print(f"  (2) MED rows predicted from verified-only model: MAE {mean_absolute_error(ym,pm):.4f}  "
          f"(verified self-CV MAE {mean_absolute_error(yt,bp):.4f}); resid range [{(pm-ym).min():+.2f},{(pm-ym).max():+.2f}]")

    # (3) conclusion stability: feature importances verified vs +MED
    iv=rf.feature_importances_
    rf2=RandomForestRegressor(n_estimators=400,random_state=0,n_jobs=-1).fit(
        np.vstack([Xv,Xm]),np.concatenate([yv,ym]))
    im=rf2.feature_importances_
    order_v=[feats[i] for i in np.argsort(iv)[::-1][:10]]
    order_m=[feats[i] for i in np.argsort(im)[::-1][:10]]
    rho=spearmanr(iv,im).correlation
    overlap=len(set(order_v)&set(order_m))
    print(f"  (3) SAR/feature-importance stability: top-10 overlap {overlap}/10, Spearman(importances) {rho:.3f}")
    print(f"        top5 verified : {order_v[:5]}")
    print(f"        top5 +MED     : {order_m[:5]}")

run("IAJD_master/datasets/IAJD_pKa_v21_final.xlsx",'IAJD','pKa','pKa (full coverage, 17 MED)')
run("IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx",'IAJD_num','log10_flux_total','bioact organ-flux')
