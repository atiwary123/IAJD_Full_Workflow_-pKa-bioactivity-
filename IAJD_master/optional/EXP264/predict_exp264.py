"""
predict_exp264.py — Run v2.0-MIN-v09 predictions on the 5 EXP264 compounds.
"""
import os, sys, json
sys.path.insert(0, '/home/claude/bioact_v2/build/scripts')
from iajd_bioact_v2 import load_bundle, predict_bioactivity
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import warnings; warnings.filterwarnings('ignore')

# Load v09 bundle
bundle = load_bundle()
print(f'Bundle: {bundle["version"]}, n_train={bundle["n_training"]}')
print(f'Stage B targets: {list(bundle["stage_b_targets"].keys())}')
print()

# Parse the CDXML
mols = Chem.MolsFromCDXMLFile('/mnt/user-data/uploads/EXP_264-1.cdxml')

results = []
for i, m in enumerate(mols, 1):
    smi = Chem.MolToSmiles(m, canonical=True)
    name = f'EXP264-{i}'
    formula = rdMolDescriptors.CalcMolFormula(m)
    mw = Descriptors.ExactMolWt(m)

    print(f'=== {name} ===')
    print(f'  SMILES:  {smi}')
    print(f'  Formula: {formula}, MW: {mw:.2f}')

    r = predict_bioactivity(smi, bundle, return_diagnostics=True)
    if 'error' in r:
        print(f'  ERROR: {r["error"]}')
        continue

    print(f'  Family:  {r["family_assigned"]}    Head: {r["head_group"]}    Chains: {r["chain_pair"]}')
    print(f'  Hierarchy level: {r["similarity_level"]}    Max Tanimoto: {r["max_tanimoto_to_training"]}')
    print(f'  Confidence: {r["confidence_tier"]}')
    print(f'  pKa:      {r["pKa_pred"]}  (PI90 [{r["pKa_PI90"][0]}, {r["pKa_PI90"][1]}])')
    print()
    print(f'  Stage A organ probabilities (calibrated):')
    for organ, p in r['organ_probs'].items():
        print(f'    {organ:22s}  {p:.3f}')
    print(f'  Dominant organ: {r["organ_dominant"]}'
          + (f'  (tied with {r["organ_dominant_tied_with"]})' if r["organ_dominant_tied_with"] else ''))
    print()
    print(f'  Stage B per-organ flux predictions:')
    for tgt in ['log10_flux_total', 'log10_flux_spleen', 'log10_flux_liver', 'log10_flux_lung', 'log10_flux_LN']:
        if tgt in r:
            pi = r[f'{tgt}_PI90']
            print(f'    {tgt:24s}  {r[tgt]:.3f}   PI90: [{pi[0]:.3f}, {pi[1]:.3f}]')
    print()
    print(f'  3-bin posterior on total flux: {r.get("log10_flux_total_bin")}  '
          f'probs={r.get("log10_flux_total_bin_probs")}')
    print(f'  Linear flux: {r.get("flux_total_p_s"):.2e} p/s'
          f'  PI90: [{r.get("flux_total_PI90_p_s")[0]:.2e}, {r.get("flux_total_PI90_p_s")[1]:.2e}]')
    print()
    print(f'  Top-3 nearest training neighbors:')
    for nn in r['nearest_neighbors'][:3]:
        flux_str = f'{nn["log10_flux_total"]:.2f}' if nn["log10_flux_total"] is not None else 'n/a'
        print(f'    {nn["IAJD_id"]:12s}  family={nn["family"]:18s}  Tanimoto={nn["tanimoto"]:.3f}  flux={flux_str}')
    if r.get('warnings'):
        print(f'  Warnings:')
        for w in r['warnings']:
            print(f'    - {w}')
    print()

    # save the prediction
    results.append({
        'name': name, 'SMILES': smi, 'formula': formula, 'mw': mw,
        'prediction': r,
    })

# Pickle for downstream use
import pickle
out_path = '/home/claude/exp264_predictions.pkl'
with open(out_path, 'wb') as f:
    pickle.dump(results, f)
print(f'\nSaved predictions to {out_path}')
