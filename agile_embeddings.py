"""Extract AGILE pretrained GNN embeddings for IAJD compounds.

Loads the 60k-pretrained AGILE encoder (no prediction head), converts
each IAJD SMILES to a molecular graph, runs it through the GNN, and
returns a 512-dim embedding vector per compound.
"""
from __future__ import annotations
import sys, os, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Data, Batch

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
AGILE_DIR = ROOT / "agile_repo"

sys.path.insert(0, str(AGILE_DIR))
from models.agile_finetune import AGILE

# Same atom/bond featurization as AGILE's dataset.py
ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
from rdkit.Chem.rdchem import BondType as BT
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]


def smiles_to_graph(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    type_idx, chirality_idx = [], []
    for atom in mol.GetAtoms():
        type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
        chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))

    x = torch.tensor(list(zip(type_idx, chirality_idx)), dtype=torch.long)

    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        bf = [BOND_LIST.index(bond.GetBondType()),
              BONDDIR_LIST.index(bond.GetBondDir())]
        edge_feat += [bf, bf]

    if len(row) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 2), dtype=torch.long)
    else:
        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(edge_feat, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def load_agile_encoder(ckpt_dir: str = None) -> AGILE:
    if ckpt_dir is None:
        ckpt_dir = str(AGILE_DIR / "ckpt" / "pretrained_agile_60k" / "checkpoints")

    model = AGILE(
        task="regression",
        num_layer=5,
        emb_dim=300,
        feat_dim=512,
        drop_ratio=0,
        pool="mean",
        pred_additional_feat_dim=0,
        pred_n_layer=2,
        pred_act="softplus",
    )

    state_dict = torch.load(
        os.path.join(ckpt_dir, "model.pth"),
        map_location="cpu",
        weights_only=False,
    )
    model.load_my_state_dict(state_dict)
    model.eval()
    return model


def extract_embeddings(smiles_list: list[str], model: AGILE,
                       batch_size: int = 32) -> np.ndarray:
    embeddings = []
    failed = []

    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i:i+batch_size]
        graphs = []
        valid_indices = []
        for j, smi in enumerate(batch_smiles):
            g = smiles_to_graph(smi)
            if g is not None:
                graphs.append(g)
                valid_indices.append(i + j)
            else:
                failed.append(i + j)

        if not graphs:
            continue

        batch = Batch.from_data_list(graphs)
        with torch.no_grad():
            h, _ = model(batch)
        emb = h.cpu().numpy()

        emb_full = np.full((len(batch_smiles), emb.shape[1]), np.nan)
        for k, vi in enumerate(range(len(graphs))):
            local_idx = valid_indices[k] - i
            emb_full[local_idx] = emb[vi]
        embeddings.append(emb_full)

    result = np.vstack(embeddings) if embeddings else np.zeros((0, 512))
    if failed:
        print(f"  Warning: {len(failed)} SMILES failed graph conversion")
    return result


def main():
    print("Loading AGILE pretrained encoder...")
    model = load_agile_encoder()
    print(f"  Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")

    # Load bioactivity SMILES
    bio = pd.read_excel('IAJD_master/datasets/IAJD_Bioact_v13_clean.xlsx', sheet_name='Sheet1')
    _EXCLUDED_NOVEL_IAJDS = {347, 348, 365, 366, 367, 369, 372, 373}
    if "IAJD_num" in bio.columns:
        bio = bio[~bio["IAJD_num"].isin(_EXCLUDED_NOVEL_IAJDS)].reset_index(drop=True)
    smiles_list = []
    for _, r in bio.iterrows():
        smi = r.get('SMILES_canonical') or r.get('SMILES')
        smiles_list.append(str(smi) if pd.notna(smi) else '')

    print(f"\nExtracting embeddings for {len(smiles_list)} bioactivity compounds...")
    embeddings = extract_embeddings(smiles_list, model)
    print(f"  Shape: {embeddings.shape}")
    print(f"  NaN rows: {np.isnan(embeddings).any(axis=1).sum()}")

    np.save('agile_embeddings_bioact.npy', embeddings)
    print(f"  Saved agile_embeddings_bioact.npy")

    # Also do pKa dataset
    pka = pd.read_excel('IAJD_master/datasets/IAJD_pKa_v21_final.xlsx', sheet_name='Dataset')
    pka_smiles = [str(r['SMILES']) for _, r in pka.iterrows()]
    print(f"\nExtracting embeddings for {len(pka_smiles)} pKa compounds...")
    pka_emb = extract_embeddings(pka_smiles, model)
    print(f"  Shape: {pka_emb.shape}")
    np.save('agile_embeddings_pka.npy', pka_emb)
    print(f"  Saved agile_embeddings_pka.npy")

    # Quick sanity: check embedding variance
    valid = ~np.isnan(embeddings).any(axis=1)
    emb_valid = embeddings[valid]
    print(f"\n  Embedding stats (bioact, {valid.sum()} valid):")
    print(f"    Mean norm: {np.linalg.norm(emb_valid, axis=1).mean():.2f}")
    print(f"    Std per dim: {emb_valid.std(axis=0).mean():.4f}")
    print(f"    Min/max: {emb_valid.min():.4f} / {emb_valid.max():.4f}")


if __name__ == "__main__":
    main()
