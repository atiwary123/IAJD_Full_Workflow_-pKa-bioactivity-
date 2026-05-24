"""
iajd_tandem_final.py — Full-stack IAJD: pKa (v9.1) → bioactivity (v14.0-full)

FINAL workflow. Two architectural choices:

  1. ALL 8 families route through the full v9.1 pKa path:
       - 30 hand-crafted features (computed from SMILES + family tokens)
       - debiased MolGpKa feature 30:
           * Standard 5 families (sSS-Nonsym, PE-Tris, GA-Tris, PE-Gallic, Dialkoxybenzyl)
             → use their per-family linear debias model
           * Bioact-only 3 families (G1-Janus, HTM, TT-Dendrimer)
             → use n-weighted pooled debias across the 5 v21 families
       - Tuned XGBoost direct prediction
       - Per-query analog-delta with uniform similarity-threshold rule (Kt=8, sim≥0.6)
       - Final = 0.05 × direct + 0.95 × analog_delta
       - NO Tanimoto-only fallback. NO special case for any family.

  2. v14.0 bioactivity runs with LION (Block B) and ADMET (Block C) FORCED ON
     for every family, overriding the bundle's per-family gate decisions.
     Honest MAE cost is small (≤0.014 worst-case on GA-Tris ADMET override).

Pipeline:

    SMILES + family_hint
              │
              ▼
    v9.1 pKa (full path for all 8 families) — outputs pKa_pred, PI90, sd
              │
              ▼  pKa + sd injected into bioact row['pKa'], row['pKa_sd']
              │
              ▼
    v14.0 bioactivity with LION + ADMET forced ON — outputs log10_flux + PI90

Public API:
    bundle = load_tandem_bundle()
    r = predict_iajd(smiles, bundle, family_hint='G1-Janus-Dendrimer')
"""
from __future__ import annotations
import os, sys, warnings, pickle, json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np

warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from iajd_pka_v91 import (
    load_v91_bundle, predict_pka_v91, V91Bundle,
)


# ---------------------------------------------------------------------------
# Family universe
# ---------------------------------------------------------------------------

PKA_TRAINED_FAMILIES = {'sSS-Nonsym', 'PE-Tris', 'GA-Tris', 'PE-Gallic',
                        'Dialkoxybenzyl', 'G1-Janus-Dendrimer'}
BIOACT_TRAINED_FAMILIES = {'sSS-Nonsym', 'PE-Tris', 'GA-Tris', 'PE-Gallic',
                            'Dialkoxybenzyl', 'G1-Janus-Dendrimer'}
ALL_KNOWN_FAMILIES = PKA_TRAINED_FAMILIES | BIOACT_TRAINED_FAMILIES
NEW_FAMILIES_NO_V21_DEBIAS = set()


# ---------------------------------------------------------------------------
# Tandem bundle
# ---------------------------------------------------------------------------

@dataclass
class TandemBundle:
    pka_bundle: V91Bundle
    bioact_bundle: Any
    bioact_bundle_path: Path


def load_tandem_bundle(
    pka_xlsx: str = 'IAJD_pKa_v21_final.xlsx',
    bioact_bundle: str = 'bioact_v14_bundle.pkl',
    verbose: bool = True,
) -> TandemBundle:
    pka_xlsx_path = HERE / pka_xlsx if not os.path.isabs(pka_xlsx) else Path(pka_xlsx)
    bioact_path = HERE / bioact_bundle if not os.path.isabs(bioact_bundle) else Path(bioact_bundle)
    if verbose:
        print('Loading IAJD tandem bundle (final)...')
        print('  [1/2] v9.1 pKa bundle (full path for all 8 families)...')
    pka_b = load_v91_bundle(str(pka_xlsx_path))
    if verbose:
        print(f'        loaded: n={len(pka_b.train_y)} training compounds, 5 trained families')
        print(f'        bioact-only families (G1-Janus, HTM, TT-Dendrimer) use pooled debias')
        print('  [2/2] v14.0 bioactivity (LION + ADMET always ON)...')
    with open(bioact_path, 'rb') as f:
        bio_b = pickle.load(f)
    if verbose:
        print(f'        loaded: n={bio_b["n_rows"]} rows, '
              f'{len(bio_b["family_list"])} families')
        print(f'        bundle honest MAE (gated): {bio_b["metrics"].get("v14_honest_pooled_mae", "?"):.4f}')
        print(f'        ASSUMED honest MAE (LION+ADMET forced ON): ≤ 0.42 (cost ~0.005-0.014)')
    return TandemBundle(pka_bundle=pka_b, bioact_bundle=bio_b,
                        bioact_bundle_path=bioact_path)


# ---------------------------------------------------------------------------
# Stage 2: bioactivity with LION+ADMET forced ON
# ---------------------------------------------------------------------------

def _bioact_predict_full_features(
    smiles: str, canonical: str, bundle: TandemBundle,
    injected_pka: float, injected_pka_sd: float,
    family_hint: Optional[str] = None,
    formulation: Optional[Dict] = None,
    auto_compute_real_features: bool = True,
    return_neighbors: int = 5,
) -> Dict[str, Any]:
    """v14.0 bioactivity with LION + ADMET FORCED ON for every family.

    Per-family α blending is still used; only the per-family gate
    decisions for Block B and Block C are overridden.
    """
    sys.path.insert(0, str(HERE))
    from bioact_v14_pipeline import assemble_X
    from iajd_bioact_v14 import _detect_family
    import pandas as pd

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'error': f'INVALID_SMILES: {smiles}'}

    MFPGEN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    fp = MFPGEN.GetFingerprint(mol)
    bio_bundle = bundle.bioact_bundle
    CACHE_DIR = bundle.bioact_bundle_path.parent

    # Auto-extend caches with real LION + ADMET if missing
    fetch_warnings = []
    if auto_compute_real_features:
        try:
            from extend_caches import predict_lion_for_smiles, predict_admet_for_smiles
            lion_path = CACHE_DIR / 'lion_cache_v13.json'
            admet_path = CACHE_DIR / 'admet_cache_v13.json'
            try:
                predict_admet_for_smiles([canonical], cache_path=str(admet_path), verbose=False)
            except Exception as e:
                fetch_warnings.append(f'ADMET_FETCH_FAILED: {type(e).__name__}: {str(e)[:80]}')
            try:
                predict_lion_for_smiles([canonical], cache_path=str(lion_path), verbose=False)
            except Exception as e:
                fetch_warnings.append(f'LION_FETCH_FAILED: {type(e).__name__}: {str(e)[:80]}')
        except ImportError:
            fetch_warnings.append('EXTEND_CACHES_UNAVAILABLE')

    bioact_family_detected, family_method = _detect_family(mol, fp, bio_bundle)
    bioact_family = family_hint or bioact_family_detected

    # Build row with injected pKa
    row = {
        'SMILES_canonical': canonical, 'canonical_smi': canonical,
        'family': bioact_family,
        'IAJD_id': 'tandem_query',
        'log10_flux_total': 0,
        'pKa': injected_pka,
        'pKa_sd': injected_pka_sd,
    }
    if formulation:
        for k, v in formulation.items():
            row[k] = v
    df_q = pd.DataFrame([row])

    lion_path = CACHE_DIR / 'lion_cache_v13.json'
    admet_path = CACHE_DIR / 'admet_cache_v13.json'
    X_q, lion_modes = assemble_X(
        df_q, [mol], [fp], [canonical],
        lion_cache_path=str(lion_path) if lion_path.exists() else None,
        admet_cache_path=str(admet_path) if admet_path.exists() else None,
    )
    x_full = X_q[0]  # No gate masking — all 88 features active
    block_b_used_real = (lion_modes[0] == 'cached') if lion_modes else False

    # Direct prediction with the full 88-d feature vector
    pred_direct = float(bio_bundle['direct_model_full'].predict(x_full.reshape(1, -1))[0])

    # Analog-delta
    fps_train = bio_bundle['fps_train']
    y_train = bio_bundle['y_train']
    sims = np.array(BulkTanimotoSimilarity(fp, fps_train))
    top = np.argsort(-sims)[:8]
    if sims[top[0]] >= 0.4:
        weights = sims[top] ** 4
        weights = weights / weights.sum()
        pred_analog = float(np.sum(weights * y_train[top]))
    else:
        pred_analog = None

    # Per-family α blending (gates overridden but α retained)
    alpha = bio_bundle['best_alpha_per_family'].get(bioact_family, 1.0)
    if pred_analog is None:
        pred = pred_direct
        alpha_used = 1.0
    else:
        pred = alpha * pred_direct + (1 - alpha) * pred_analog
        alpha_used = alpha

    # Confidence tier from max Tanimoto
    max_sim = float(sims[np.argmax(sims)])
    if max_sim >= 0.7: tier, pi_half = 'HIGH', 0.20
    elif max_sim >= 0.5: tier, pi_half = 'MED', 0.30
    else: tier, pi_half = 'LOW', 0.45

    smis_train = bio_bundle['smis_train']
    fams_train = bio_bundle['families_train']
    top_n = np.argsort(-sims)[:return_neighbors]
    neighbors = [{
        'smiles': smis_train[i], 'family': fams_train[i],
        'log10_flux_total': float(y_train[i]),
        'tanimoto': float(sims[i]),
    } for i in top_n]

    warnings_out = list(fetch_warnings)
    if max_sim < 0.4:
        warnings_out.append('NO_CLOSE_NEIGHBOR_BIOACT')
    if bioact_family == 'Unknown':
        warnings_out.append('BIOACT_FAMILY_UNKNOWN')
    if not block_b_used_real:
        warnings_out.append('LION_PROXY_USED: real LION lookup failed')

    bundle_gate_b = bio_bundle['block_b_active_per_family'].get(bioact_family, True)
    bundle_gate_c = bio_bundle['block_c_active_per_family'].get(bioact_family, True)
    gate_overrides = []
    if not bundle_gate_b:
        gate_overrides.append(f'Block B (LION) gate=OFF for {bioact_family} in bundle, OVERRIDDEN to ON')
    if not bundle_gate_c:
        gate_overrides.append(f'Block C (ADMET) gate=OFF for {bioact_family} in bundle, OVERRIDDEN to ON')

    return {
        'log10_flux_total': pred,
        'log10_flux_total_PI90': (pred - pi_half, pred + pi_half),
        'family_detected': bioact_family_detected,
        'family_used': bioact_family,
        'family_method': family_method if not family_hint else 'user_hint',
        'alpha_used': alpha_used,
        'block_B_active': True,   # always ON
        'block_C_active': True,   # always ON
        'block_B_real': bool(block_b_used_real),
        'gate_overrides_applied': gate_overrides,
        'pred_direct': pred_direct,
        'pred_analog': pred_analog,
        'max_tanimoto': max_sim,
        'confidence_tier': tier,
        'nearest_neighbors': neighbors,
        'warnings': warnings_out,
        'pka_used_in_features': float(injected_pka),
        'pka_sd_used_in_features': float(injected_pka_sd),
    }


# ---------------------------------------------------------------------------
# Family reconciliation
# ---------------------------------------------------------------------------

def reconcile_family(pka_family, bioact_family, user_hint):
    out = {'sources': {'pka': pka_family, 'bioact': bioact_family,
                       'user_hint': user_hint},
           'warnings': []}
    if user_hint is not None:
        out['unified_family'] = user_hint
        if pka_family and pka_family != user_hint:
            out['warnings'].append(f'PKA_FAMILY_MISMATCH: hint={user_hint} pka_detected={pka_family}')
        if bioact_family and bioact_family != user_hint and bioact_family != 'Unknown':
            out['warnings'].append(f'BIOACT_FAMILY_MISMATCH: hint={user_hint} bioact_detected={bioact_family}')
        return out
    if pka_family == bioact_family:
        out['unified_family'] = pka_family or bioact_family or 'Unknown'
        return out
    if pka_family in PKA_TRAINED_FAMILIES:
        out['unified_family'] = pka_family
        out['warnings'].append(f'FAMILY_RECONCILED: pka={pka_family} bioact={bioact_family} → using pka')
    elif bioact_family and bioact_family != 'Unknown':
        out['unified_family'] = bioact_family
        out['warnings'].append(f'FAMILY_RECONCILED: pka={pka_family} bioact={bioact_family} → using bioact')
    else:
        out['unified_family'] = 'Unknown'
        out['warnings'].append('FAMILY_UNKNOWN: neither stage detected a known family')
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_iajd(
    smiles: str,
    bundle: TandemBundle,
    family_hint: Optional[str] = None,
    formulation: Optional[Dict] = None,
    measured_pka: Optional[float] = None,
    measured_pka_sd: Optional[float] = None,
    auto_compute_real_features: bool = True,
    return_diagnostics: bool = True,
    return_neighbors: int = 5,
) -> Dict[str, Any]:
    """Tandem final: v9.1 pKa (full path for all 8 families) → v14.0 bioactivity
    (LION+ADMET always ON).

    Args:
        smiles: Query SMILES.
        bundle: TandemBundle from load_tandem_bundle().
        family_hint: One of 8 known families:
            sSS-Nonsym, PE-Tris, GA-Tris, PE-Gallic, Dialkoxybenzyl,
            G1-Janus-Dendrimer, HTM-Dendrimer, TT-Dendrimer.
            REQUIRED except for auto-detectable PE-Tris.
        formulation: Optional dict (DNP_size_nm, DNP_PDI, DNP_EE_pct,
                     DNP_zeta_mV, buffer_pH, dose_mRNA_ug).
        measured_pka: Bypass pKa prediction with measured value.
        measured_pka_sd: SD for measured_pka (default 0.05).
        auto_compute_real_features: Fetch real LION+ADMET for novel SMILES.
        return_diagnostics: Include per-component breakdowns.
        return_neighbors: Neighbor count per stage.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'error': f'INVALID_SMILES: {smiles}', 'smiles': smiles}
    canonical = Chem.MolToSmiles(mol, canonical=True)

    out: Dict[str, Any] = {
        'smiles': smiles, 'canonical_smiles': canonical,
        'version': 'tandem_final (pKa=v9.1-full, bioact=v14.0-LION+ADMET-on)',
        'flow': {},
    }

    # =================
    # Stage 1: pKa via FULL v9.1 path (works for all 8 families)
    # =================
    if measured_pka is not None:
        pka_pred = float(measured_pka)
        pka_sd = float(measured_pka_sd) if measured_pka_sd is not None else 0.05
        pka_result = {
            'pKa_pred': round(pka_pred, 3),
            'pKa_PI_90': [round(pka_pred - 1.645 * pka_sd, 2),
                          round(pka_pred + 1.645 * pka_sd, 2)],
            'pKa_sd': pka_sd, 'source': 'measured',
            'confidence_tier': 'MEASURED',
            'family_assigned': family_hint or 'Unknown',
            'ood_flag': False,
        }
        out['flow']['stage1_pka'] = f'measured_value: pKa={pka_pred} ± {pka_sd}'
    else:
        try:
            v91_result = predict_pka_v91(smiles, bundle.pka_bundle,
                                          family_hint=family_hint,
                                          return_diagnostics=return_diagnostics)
            if 'error' in v91_result:
                pka_result = {
                    'pKa_pred': 6.3, 'pKa_PI_90': [5.5, 7.1], 'pKa_sd': 0.5,
                    'source': 'pKa_error_default',
                    'confidence_tier': 'LOW', 'family_assigned': 'Unknown',
                    'ood_flag': True, 'error': v91_result['error'],
                }
                out['flow']['stage1_pka'] = f'pKa failed: {v91_result["error"]}'
            else:
                pka_result = v91_result
                pka_result['source'] = 'v9.1_predicted'
                # Compute sd from PI half-width (90% Gaussian → σ = half / 1.645)
                if 'pKa_sd' not in pka_result:
                    pi = pka_result.get('pKa_PI_90', [pka_result['pKa_pred'] - 0.25,
                                                     pka_result['pKa_pred'] + 0.25])
                    pka_result['pKa_sd'] = round((pi[1] - pi[0]) / (2 * 1.645), 3)
                out['flow']['stage1_pka'] = (
                    f'v9.1 (full path): pKa={pka_result["pKa_pred"]:.3f} '
                    f'PI90={pka_result["pKa_PI_90"]} '
                    f'family={pka_result["family_assigned"]} '
                    f'tier={pka_result["confidence_tier"]}'
                )
        except Exception as e:
            pka_result = {
                'pKa_pred': 6.3, 'pKa_PI_90': [5.5, 7.1], 'pKa_sd': 0.5,
                'source': 'pKa_exception_default', 'confidence_tier': 'LOW',
                'family_assigned': 'Unknown', 'ood_flag': True,
                'error': f'{type(e).__name__}: {str(e)[:200]}',
            }
            out['flow']['stage1_pka'] = f'pKa exception: {type(e).__name__}'

    out['pka'] = pka_result

    # =================
    # Stage 2: bioactivity (LION+ADMET FORCED ON)
    # =================
    bioact_result = _bioact_predict_full_features(
        smiles=smiles, canonical=canonical, bundle=bundle,
        injected_pka=pka_result['pKa_pred'],
        injected_pka_sd=pka_result.get('pKa_sd', 0.05),
        family_hint=family_hint, formulation=formulation,
        auto_compute_real_features=auto_compute_real_features,
        return_neighbors=return_neighbors,
    )

    if 'error' in bioact_result:
        out['flow']['stage2_bioact'] = f'bioact failed: {bioact_result["error"]}'
    else:
        gate_note = ''
        if bioact_result['gate_overrides_applied']:
            gate_note = f' [overrides: {len(bioact_result["gate_overrides_applied"])}]'
        out['flow']['stage2_bioact'] = (
            f'v14.0-full: log10_flux={bioact_result["log10_flux_total"]:.3f} '
            f'PI90={bioact_result["log10_flux_total_PI90"]} '
            f'family={bioact_result["family_used"]} '
            f'α={bioact_result["alpha_used"]} '
            f'B=ON C=ON (forced){gate_note} '
            f'tier={bioact_result["confidence_tier"]}'
        )
    out['bioactivity'] = bioact_result

    out['family_reconciliation'] = reconcile_family(
        pka_family=pka_result.get('family_assigned'),
        bioact_family=bioact_result.get('family_used'),
        user_hint=family_hint,
    )
    out['combined_summary'] = _build_summary(out)
    return out


def _build_summary(result):
    pka = result.get('pka', {})
    bio = result.get('bioactivity', {})
    fam = result.get('family_reconciliation', {})
    s = {}
    pka_pred = pka.get('pKa_pred')
    s['pka'] = (
        f"pKa = {pka_pred:.3f} (PI90 = {pka.get('pKa_PI_90', '?')}, "
        f"source = {pka.get('source', '?')}, tier = {pka.get('confidence_tier', '?')})"
        if pka_pred is not None else 'pKa: failed'
    )
    bio_flux = bio.get('log10_flux_total')
    if 'error' in bio:
        s['bioact'] = f"bioactivity error: {bio['error']}"
    elif bio_flux is not None:
        pi = bio.get('log10_flux_total_PI90', ['?', '?'])
        s['bioact'] = (
            f"log10_flux_total = {bio_flux:.3f} (PI90 = ({pi[0]:.3f}, {pi[1]:.3f}), "
            f"tier = {bio.get('confidence_tier', '?')})"
        )
    s['family'] = (
        f"unified family = {fam.get('unified_family', '?')} "
        f"(pKa stage: {fam.get('sources', {}).get('pka', '?')}, "
        f"bioact stage: {fam.get('sources', {}).get('bioact', '?')})"
    )
    return s


def predict_iajd_batch(smiles_list, bundle, **kwargs):
    return [predict_iajd(s, bundle, **kwargs) for s in smiles_list]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('=' * 70)
    print('IAJD Tandem Final — Smoke Test')
    print('  All 8 families through v9.1 full path; LION+ADMET always ON')
    print('=' * 70)
    b = load_tandem_bundle()

    test_cases = [
        ('PE-Tris (standard v9.1)',
         'CCCCCCCCOCC(COCCCCCCCC)(COCCCCCCCC)COC(=O)CCCN1CCN(C)CC1',
         'PE-Tris'),
        ('GA-Tris (ADMET gate override)',
         'CCCCCCCCCCCCOc1cc(COC(=O)CCCN2CCN(CCO)CC2)cc(OCCCCCCCCCCCC)c1OCCCCCCCCCCCC',
         'GA-Tris'),
        ('sSS-Nonsym (standard)',
         'CCCCCCCCCCCCCCCCCCOc1cc(COC(=O)CCCN2CCN(C)CC2)cc(OCCCCCCCCCCC)c1',
         'sSS-Nonsym'),
        ('Dialkoxybenzyl (standard)',
         'CCCCCCCCCCCCOc1ccc(COC(=O)CCCN2CCN(CCO)CC2)cc1OCCCCCCCCCCCC',
         'Dialkoxybenzyl'),
        ('G1-Janus-Dendrimer (full v9.1 path w/ pooled debias)',
         'CCCCCCCCCCCCOc1cc(OCCCCCCCCCCCC)c(OCCCCCCCCCCCC)c(COC(=O)CCCN2CCN(C)CC2)c1',
         'G1-Janus-Dendrimer'),
        ('PE-Gallic (ex-HTM architecture)',
         'CCCCCCCCCCCCN(CCO)CCOC(=O)CCCN1CCN(C)CC1',
         'PE-Gallic'),
        ('PE-Gallic (ex-TT architecture)',
         'CCCCCCCCCCCCOC(=O)C(NC(=O)CCN(CCO)CCO)(COCCCCCCCCCCCC)COCCCCCCCCCCCC',
         'PE-Gallic'),
    ]
    results_for_json = []
    for label, smi, fam in test_cases:
        print(f'\n--- {label} ---')
        r = predict_iajd(smi, b, family_hint=fam, return_neighbors=3)
        print(f'  SMILES: {smi[:60]}...')
        for k, v in r['flow'].items():
            print(f'    {k}: {v}')
        bio = r['bioactivity']
        if bio.get('gate_overrides_applied'):
            for n in bio['gate_overrides_applied']:
                print(f'  ⚙ {n}')
        all_warn = (r['pka'].get('warnings', []) + bio.get('warnings', [])
                    + r['family_reconciliation'].get('warnings', []))
        for w in all_warn:
            print(f'  ⚠ {w}')
        results_for_json.append({
            'label': label,
            'family_hint': fam,
            'smiles': r['canonical_smiles'],
            'pka_pred': r['pka']['pKa_pred'],
            'pka_PI90': r['pka']['pKa_PI_90'],
            'pka_source': r['pka']['source'],
            'pka_tier': r['pka']['confidence_tier'],
            'log10_flux': bio['log10_flux_total'],
            'log10_flux_PI90': list(bio['log10_flux_total_PI90']),
            'bioact_tier': bio['confidence_tier'],
            'lion_active': bio['block_B_active'],
            'admet_active': bio['block_C_active'],
            'lion_real': bio['block_B_real'],
            'gate_overrides': bio['gate_overrides_applied'],
            'max_tanimoto_pka': r['pka'].get('max_tanimoto_to_training'),
            'max_tanimoto_bioact': round(bio['max_tanimoto'], 3),
            'alpha_used': bio['alpha_used'],
        })
    with open('tandem_final_results.json', 'w') as f:
        json.dump(results_for_json, f, indent=2)
    print('\n=== SMOKE TEST COMPLETE ===')
    print(f'Results saved to tandem_final_results.json')
