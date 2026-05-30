"""
rdkit_compat.py — single-file RDKit API compatibility shim.

Older RDKit builds (the one pinned in this repo's venv at the time of writing)
don't expose `AllChem.GetMorganGenerator`. Many modules in this codebase
reference it at import time, so importing them on a vanilla older RDKit
raises AttributeError before any user code runs.

This module patches `AllChem.GetMorganGenerator` to a thin wrapper around the
legacy `GetMorganFingerprintAsBitVect` / `GetHashedMorganFingerprint` API.
Real fingerprints, real RDKit — no proxies. Just an API rename bridge.

Idempotent: importing this module multiple times is a no-op. Safe to
`import rdkit_compat` at the top of any module that uses
GetMorganGenerator — the shim is applied once on first import.
"""
from __future__ import annotations

try:
    from rdkit.Chem import AllChem as _AC

    if not hasattr(_AC, "GetMorganGenerator"):
        class _ShimGen:
            __slots__ = ("radius", "fpSize")

            def __init__(self, radius: int, fpSize: int):
                self.radius = radius
                self.fpSize = fpSize

            def GetFingerprint(self, mol):
                return _AC.GetMorganFingerprintAsBitVect(
                    mol, self.radius, nBits=self.fpSize
                )

            def GetCountFingerprint(self, mol):
                return _AC.GetHashedMorganFingerprint(
                    mol, radius=self.radius, nBits=self.fpSize
                )

        def _shim_factory(radius: int = 2, fpSize: int = 2048, **_kw):
            return _ShimGen(radius, fpSize)

        _AC.GetMorganGenerator = _shim_factory
except Exception:  # noqa: BLE001
    # Tolerate: if RDKit itself is missing, downstream imports will report it.
    pass
