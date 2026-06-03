import json
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

PATH = 'IAJD_master/datasets/IAJD_pKa_v21_final.xlsx'
TARGET = 'pKa'
QM = ['qm_q_ionizableN','qm_dipole_D','qm_polarizability','qm_homo_lumo_eV',
      'qm_dGsolv_kJmol','qm_dGsolv_head','qm_dGsolv_tail','qm_Ehedup']

# Families the task asks deltas for (data uses G1-Janus-Dendrimer for G1-Janus)
FAM_REQUEST = ['PE-Gallic','PE-Tris','GA-Tris','sSS-Nonsym','Dialkoxybenzyl','G1-Janus']
FAM_MAP = {'G1-Janus':'G1-Janus-Dendrimer'}

XGB_KW = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
              reg_lambda=3, n_jobs=1, random_state=0)

def load():
    df = pd.read_excel(PATH, sheet_name='Sheet1')
    s = df['audit_status'].astype(str)
    flag = s.str.contains('UNRESOLVED', case=False, na=False) | s.str.contains('FLAG', case=False, na=False)
    qm_ok = (df['qm_source'] == 'xtb')
    use = qm_ok & (~flag) & df[TARGET].notna()
    n_qm_missing = int((~qm_ok).sum())
    return df[use].reset_index(drop=True), n_qm_missing, int((~use).sum())

def structural_cols(df):
    """Genuine RDKit/structural numeric descriptors.
    Exclude: QM, target, target-sd, ID, molgpka (own baseline feat), sp_* (experimental metadata)."""
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    drop = set(QM) | {TARGET, 'pKa_sd', 'IAJD', 'molgpka_pKa'}
    cols = [c for c in num if c not in drop and not c.startswith('sp_') and not c.startswith('qm_')]
    return cols

def all_numeric_feat_cols(df):
    """Robustness variant: all numeric feature cols incl sp_ (~the '~62' hint)."""
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    drop = set(QM) | {TARGET, 'pKa_sd', 'IAJD', 'molgpka_pKa'}
    cols = [c for c in num if c not in drop and not c.startswith('qm_')]
    return cols

def loo_mae(X, y, impute_nonqm_cols, qm_cols):
    """LOO XGB MAE. Median-impute non-QM cols (fit on train fold). QM cols NOT imputed
    (passed with native NaN -> XGBoost sparsity-aware split). Returns mae and abs errors."""
    X = X.copy()
    y = np.asarray(y, float)
    loo = LeaveOneOut()
    preds = np.empty(len(y))
    nonqm = [c for c in impute_nonqm_cols if c in X.columns]
    feat_order = list(X.columns)
    for tr, te in loo.split(X):
        Xtr = X.iloc[tr].copy(); Xte = X.iloc[te].copy()
        if nonqm:
            imp = SimpleImputer(strategy='median')
            Xtr[nonqm] = imp.fit_transform(Xtr[nonqm])
            Xte[nonqm] = imp.transform(Xte[nonqm])
        m = XGBRegressor(**XGB_KW)
        m.fit(Xtr[feat_order].values, y[tr])
        preds[te] = m.predict(Xte[feat_order].values)
    ae = np.abs(preds - y)
    return float(ae.mean()), ae

def per_family_string(fam, ae_base, ae_qm):
    parts = []
    for f in FAM_REQUEST:
        key = FAM_MAP.get(f, f)
        m = (fam == key).values
        n = int(m.sum())
        if n == 0:
            parts.append(f"{f}: n=0 (absent)")
            continue
        d = float(ae_qm[m].mean() - ae_base[m].mean())
        parts.append(f"{f}(n={n}): {d:+.3f}")
    return "; ".join(parts)

def run_variant(df, base_struct_cols, label):
    y = df[TARGET].values
    fam = df['family']
    base_cols = ['molgpka_pKa'] + base_struct_cols
    Xb = df[base_cols]
    Xq = df[base_cols + QM]
    # non-QM impute set = everything except QM
    mae_b, ae_b = loo_mae(Xb, y, impute_nonqm_cols=base_cols, qm_cols=[])
    mae_q, ae_q = loo_mae(Xq, y, impute_nonqm_cols=base_cols, qm_cols=QM)
    delta = mae_q - mae_b
    pf = per_family_string(fam, ae_b, ae_q)
    return dict(label=label, n_feat_base=len(base_cols), n_feat_qm=len(base_cols)+len(QM),
                baseline_mae=mae_b, with_qm_mae=mae_q, delta=delta,
                qm_helps=bool(delta < -0.005), per_family=pf)

def main():
    df, n_qm_missing, n_excluded = load()
    n = len(df)
    struct = structural_cols(df)
    allnum = all_numeric_feat_cols(df)
    print('n_used:', n, '| QM-missing excluded:', n_qm_missing, '| total excluded:', n_excluded)
    print('structural cols:', len(struct))
    print('all-numeric (incl sp_) cols:', len(allnum))
    # QM NaN count among used rows (handled natively, not imputed)
    qm_nan = df[QM].isna().sum().to_dict()
    print('QM NaN among used rows (native-handled, NOT imputed):', qm_nan)

    primary = run_variant(df, struct, 'PRIMARY structural+molgpka')
    robust  = run_variant(df, allnum, 'ROBUST all-numeric(incl sp_)+molgpka')

    out = dict(n_used=n, n_qm_missing_excluded=n_qm_missing,
               primary=primary, robust=robust, qm_nan=qm_nan)
    print(json.dumps(out, indent=2))
    with open('eval/qm_pka_eval_result.json','w') as f:
        json.dump(out, f, indent=2)

if __name__ == '__main__':
    main()
