"""
nn/agile_embed.py — frozen AGILE 60k-MolCLR graph embeddings for a list of SMILES.

This is the GNN-encoder transfer path (the GPU upgrade over the Morgan fallback in
train_transfer.py). It loads agile_repo's pretrained contrastive encoder (60k ionizable
lipids) + their exact featurizer, and returns an (N, 512) embedding matrix. CPU or GPU.

Honest: this is the closest published encoder to the ionizable-amphiphile space, but the
IAJD scaffold is still OOD (no public dendritic data) — the embedding is a learned prior,
gated downstream by leave-one-family-out in train_transfer.py.
"""
from __future__ import annotations

import io
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
AGILE = ROOT / "agile_repo"
CKPT = AGILE / "ckpt" / "pretrained_agile_60k" / "checkpoints" / "model.pth"


def agile_embeddings(smiles, ckpt: Path = CKPT, batch_size: int = 64, device=None):
    """Return an (N, 512) array of frozen AGILE embeddings for `smiles` (order preserved)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(AGILE) not in sys.path:
        sys.path.insert(0, str(AGILE))
    from models.agile_finetune import AGILE as Model
    from dataset.dataset_test import MolTestDataset
    from torch_geometric.loader import DataLoader

    with redirect_stdout(io.StringIO()):           # the model __init__ prints itself
        model = Model("regression", num_layer=5, emb_dim=300, feat_dim=512,
                      drop_ratio=0.0, pool="mean")
        model.load_my_state_dict(torch.load(str(ckpt), map_location=device))
    model.to(device).eval()

    tmp = Path(tempfile.mkdtemp()) / "smi.csv"
    pd.DataFrame({"smiles": list(smiles), "y": [0.0] * len(smiles)}).to_csv(tmp, index=False)
    ds = MolTestDataset(str(tmp), target="y", task="regression")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    embs = []
    with torch.no_grad():
        for batch in dl:
            batch = batch.to(device)
            h, _ = model(batch)                    # h = pooled feat_lin embedding (512-d)
            embs.append(h.detach().cpu().numpy())
    return np.vstack(embs)


if __name__ == "__main__":
    df = pd.read_csv(ROOT / "nn" / "iajd_transfer_input.csv").dropna(subset=["smiles"]).head(10)
    E = agile_embeddings(df["smiles"].tolist())
    print("AGILE embeddings:", E.shape,
          "| mean L2 norm %.3f" % float(np.linalg.norm(E, axis=1).mean()),
          "| device", "cuda" if torch.cuda.is_available() else "cpu")
