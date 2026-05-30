"""
chemberta_embedder.py — frozen-encoder SMILES embeddings (T3 #7).

Uses DeepChem/ChemBERTa-77M-MTR (a small RoBERTa pretrained on 77M PubChem
molecules) as a frozen feature extractor. For each SMILES we run the model
forward once and pool the last-hidden-states into a 384-d vector.

No-proxy: every embedding comes from a real forward pass. No median fills,
no random vectors. Disk-cached by canonical SMILES for reproducibility and
speed.

The downstream Block F is a PCA projection of these 384-d embeddings into
a tighter, lower-variance subspace — chosen at training time, applied at
inference. Keeping the projection small (~32 dims) avoids overfitting on
our 247-row training set while still capturing the chemistry-pretrained
information that LION + ADMET don't.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent
CACHE_PATH = ROOT / "IAJD_master/bundles_caches/chemberta_embeddings_cache.json"
MODEL_NAME = "DeepChem/ChemBERTa-77M-MTR"
EMBED_DIM = 384

_TOKENIZER = None
_MODEL = None
_CACHE: Optional[dict] = None


def _ensure_model():
    global _TOKENIZER, _MODEL
    if _TOKENIZER is not None and _MODEL is not None:
        return
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    from transformers import AutoTokenizer, AutoModel
    _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
    _MODEL = AutoModel.from_pretrained(MODEL_NAME)
    _MODEL.eval()
    torch.set_grad_enabled(False)


def _load_cache() -> dict:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            _CACHE = json.load(f)
    else:
        _CACHE = {}
    return _CACHE


def _save_cache() -> None:
    if _CACHE is None:
        return
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(_CACHE, f)


def embed_smiles(smiles: str, use_cache: bool = True) -> np.ndarray:
    """Return the 384-d mean-pooled ChemBERTa embedding for one SMILES.

    Returns NaN-filled array if the SMILES is unparseable. Caches by the
    input string (the caller should pass canonical SMILES for reproducibility).
    """
    cache = _load_cache() if use_cache else {}
    if use_cache and smiles in cache:
        return np.asarray(cache[smiles], dtype=np.float32)
    _ensure_model()
    import torch
    batch = _TOKENIZER(
        [smiles], return_tensors="pt", padding=True, truncation=True, max_length=512,
    )
    with torch.no_grad():
        out = _MODEL(**batch)
    # Mean-pool the last hidden state across the SMILES tokens (respecting attention mask)
    mask = batch["attention_mask"].unsqueeze(-1).float()
    summed = (out.last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    emb = (summed / counts).squeeze(0).cpu().numpy().astype(np.float32)
    if use_cache:
        cache[smiles] = emb.tolist()
        _save_cache()
    return emb


def embed_smiles_batch(smiles_list: List[str], use_cache: bool = True,
                        batch_size: int = 16) -> np.ndarray:
    """Vectorized batched embedding.  Returns shape (n, 384) ndarray."""
    cache = _load_cache() if use_cache else {}
    out = np.full((len(smiles_list), EMBED_DIM), np.nan, dtype=np.float32)
    # First pass: hit cache
    todo = []
    for i, s in enumerate(smiles_list):
        if use_cache and s in cache:
            out[i] = np.asarray(cache[s], dtype=np.float32)
        else:
            todo.append(i)
    if not todo:
        return out
    _ensure_model()
    import torch
    for chunk_start in range(0, len(todo), batch_size):
        chunk = todo[chunk_start:chunk_start + batch_size]
        smis = [smiles_list[i] for i in chunk]
        batch = _TOKENIZER(
            smis, return_tensors="pt",
            padding=True, truncation=True, max_length=512,
        )
        with torch.no_grad():
            o = _MODEL(**batch)
        mask = batch["attention_mask"].unsqueeze(-1).float()
        summed = (o.last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        embs = (summed / counts).cpu().numpy().astype(np.float32)
        for i, e in zip(chunk, embs):
            out[i] = e
            if use_cache:
                cache[smiles_list[i]] = e.tolist()
    if use_cache:
        _save_cache()
    return out


if __name__ == "__main__":
    # Smoke test
    test_smis = [
        "CCCCC(CC)CCOc1cc(COC(=O)CN2CCN(CCOCCO)CC2)cc(OCCC(CC)CCCC)c1OCCC(CC)CCCC",
        "CCCCCCCCCCCCOc1cc(COC(=O)CCCN2CCN(C)CC2)cc(OCCCCCCCCCCCC)c1",
        "CN(C)CCN1CCN(C)CC1",
    ]
    import time
    t0 = time.time()
    embs = embed_smiles_batch(test_smis, use_cache=False)
    print(f"Embedded {len(test_smis)} SMILES in {time.time()-t0:.1f}s")
    print(f"shape: {embs.shape}")
    for i, s in enumerate(test_smis):
        print(f"  [{i}] norm={np.linalg.norm(embs[i]):.3f}  first5={embs[i][:5]}")
