"""
Orthogonality of QM descriptors for pKa, beyond [molgpka_pKa, MolLogP, TPSA].

For each qm_ descriptor: partial Spearman correlation with pKa controlling for
[molgpka_pKa, MolLogP, TPSA] (residualize both on rank-transformed controls via
OLS, then Spearman the residuals). Overall + within PE-Gallic / GA-Tris.

MAE test: LOO XGBoost MAE of [molgpka,MolLogP,TPSA]  vs  + 8 QM cols.
Median-impute NON-QM NaNs; never impute QM (restrict to finite-QM rows).
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

PATH = "IAJD_master/datasets/IAJD_pKa_v21_final.xlsx"
TARGET = "pKa"
QM = ['qm_q_ionizableN','qm_dipole_D','qm_polarizability','qm_homo_lumo_eV',
      'qm_dGsolv_kJmol','qm_dGsolv_head','qm_dGsolv_tail','qm_Ehedup']
CTRL = ['molgpka_pKa','MolLogP','TPSA']

def load():
    df = pd.read_excel(PATH, sheet_name='Sheet1')
    # Exclude UNRESOLVED|FLAG audit rows (defensive; none present but enforce)
    aud = df['audit_status'].astype(str)
    keep_aud = ~aud.str.contains('UNRESOLVED', case=False, na=False) & \
               ~aud.str.contains(r'\bFLAG\b', case=False, na=False, regex=True)
    df = df[keep_aud]
    # Restrict to finite-QM rows (qm_source == xtb)
    df = df[df['qm_source'] == 'xtb'].copy()
    # need finite target
    df = df[np.isfinite(df[TARGET].astype(float))].copy()
    return df

def rankreg_resid(y, X):
    """Residuals of rank(y) regressed on rank(X) columns (partial-Spearman engine)."""
    yr = stats.rankdata(y)
    Xr = np.column_stack([stats.rankdata(X[:, j]) for j in range(X.shape[1])])
    lr = LinearRegression().fit(Xr, yr)
    return yr - lr.predict(Xr)

def partial_spearman(y, qm, ctrl):
    """Partial Spearman of qm vs y controlling ctrl (all 1D/2D numpy, complete-case)."""
    ry = rankreg_resid(y, ctrl)
    rq = rankreg_resid(qm, ctrl)
    rho, p = stats.spearmanr(rq, ry)
    return rho, p

def raw_spearman(y, qm):
    rho, p = stats.spearmanr(qm, y)
    return rho, p

def run_partials(df, label):
    out = []
    for c in QM:
        m = df[[c, TARGET] + CTRL].apply(pd.to_numeric, errors='coerce')
        m = m.dropna()
        n = len(m)
        if n < 4:
            out.append((label, c, n, np.nan, np.nan, np.nan, np.nan))
            continue
        y = m[TARGET].values
        qm = m[c].values
        ctrl = m[CTRL].values
        # guard: zero-variance qm
        if np.ptp(qm) == 0:
            out.append((label, c, n, np.nan, np.nan, np.nan, np.nan))
            continue
        praw, p_raw = raw_spearman(y, qm)
        ppar, p_par = partial_spearman(y, qm, ctrl)
        out.append((label, c, n, praw, p_raw, ppar, p_par))
    return out

def loo_mae(df, feats, target=TARGET):
    """LOO MAE with XGB; median-impute feats (non-QM safe since QM finite on these rows)."""
    X = df[feats].apply(pd.to_numeric, errors='coerce').copy()
    # median-impute (QM cols are already finite on the finite-QM row set, so this only touches controls)
    med = X.median(numeric_only=True)
    X = X.fillna(med)
    y = df[target].astype(float).values
    Xv = X.values
    loo = LeaveOneOut()
    preds = np.empty(len(y))
    for tr, te in loo.split(Xv):
        model = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                 reg_lambda=3, n_jobs=1, verbosity=0)
        model.fit(Xv[tr], y[tr])
        preds[te] = model.predict(Xv[te])
    return mean_absolute_error(y, preds), len(y)

def main():
    df = load()
    print(f"[load] finite-QM (xtb) rows with finite pKa: {len(df)}")
    print(f"[load] family counts:\n{df['family'].value_counts()}\n")

    # ---- Partial correlations: overall ----
    results = []
    results += run_partials(df, 'OVERALL')
    for fam in ['PE-Gallic', 'GA-Tris']:
        sub = df[df['family'] == fam]
        results += run_partials(sub, fam)

    print("=== PARTIAL SPEARMAN (qm vs pKa | molgpka,MolLogP,TPSA) ===")
    print(f"{'group':<11}{'qm':<20}{'n':>4}{'raw_rho':>9}{'p_raw':>9}{'part_rho':>10}{'p_part':>9}")
    rows_struct = []
    for (lab, c, n, praw, p_raw, ppar, p_par) in results:
        praw_s = f"{praw:.3f}" if praw == praw else "nan"
        p_raw_s = f"{p_raw:.3g}" if p_raw == p_raw else "nan"
        ppar_s = f"{ppar:.3f}" if ppar == ppar else "nan"
        p_par_s = f"{p_par:.3g}" if p_par == p_par else "nan"
        print(f"{lab:<11}{c:<20}{n:>4}{praw_s:>9}{p_raw_s:>9}{ppar_s:>10}{p_par_s:>9}")
        rows_struct.append(dict(group=lab, qm=c, n=n, raw_rho=praw, p_raw=p_raw,
                                partial_rho=ppar, p_partial=p_par))

    rs = pd.DataFrame(rows_struct)
    # qm_helps gate: any |partial rho| >= 0.4 at n >= 10
    gate = rs[(rs['n'] >= 10) & (rs['partial_rho'].abs() >= 0.4)]
    print("\n=== GATE: |partial rho| >= 0.4 at n >= 10 ===")
    if len(gate):
        for _, r in gate.iterrows():
            print(f"  {r['group']:<11}{r['qm']:<20} n={int(r['n']):>3}  partial_rho={r['partial_rho']:.3f} (p={r['p_partial']:.3g})")
    else:
        print("  (none)")
    qm_helps = bool(len(gate) > 0)

    # ---- MAE test (minimal): controls vs controls+QM, same LOO row set ----
    # Same rows for both models = finite-QM rows with finite pKa (controls median-imputed).
    base_mae, n_used = loo_mae(df, CTRL)
    qm_mae, _ = loo_mae(df, CTRL + QM)
    print(f"\n=== LOO MAE (XGB; n={n_used}) ===")
    print(f"  baseline [molgpka,MolLogP,TPSA]      MAE = {base_mae:.4f}")
    print(f"  + 8 QM                               MAE = {qm_mae:.4f}")
    print(f"  delta (with_qm - baseline)           = {qm_mae - base_mae:+.4f}")

    # Notable partials for per_family string
    print("\n=== SUMMARY ROWS (|partial_rho|>=0.3 OR notable) ===")
    notable = rs[(rs['partial_rho'].abs() >= 0.3) & (rs['n'] >= 8)].sort_values(
        by='partial_rho', key=lambda s: s.abs(), ascending=False)
    for _, r in notable.iterrows():
        print(f"  {r['group']:<11}{r['qm']:<20} n={int(r['n']):>3}  raw={r['raw_rho']:+.2f}  partial={r['partial_rho']:+.2f} (p={r['p_partial']:.2g})")

    # machine-readable footer
    print("\n###RESULT###")
    print(f"n_used={n_used}")
    print(f"baseline_mae={base_mae:.6f}")
    print(f"with_qm_mae={qm_mae:.6f}")
    print(f"delta_mae={qm_mae - base_mae:.6f}")
    print(f"qm_helps={qm_helps}")
    rs.to_csv('eval/qm_orthogonality_pka_partials.csv', index=False)
    print("wrote eval/qm_orthogonality_pka_partials.csv")

if __name__ == '__main__':
    main()
