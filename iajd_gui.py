"""
iajd_gui.py — Minimal Tkinter GUI for STRIDE.

STRIDE = STRuctural Ranking + Informed Design Engine: the pKa + bioactivity
tandem predictor for ionizable amphiphilic Janus dendrimers (IAJDs).

Inputs:  SMILES (required), Family (optional dropdown).
Outputs: pKa with 60% and 90% CIs, bioactivity with both CIs, nearest neighbors
         from both training sets, and any warnings emitted by the pipeline.

Pure stdlib — no external GUI dependencies.

Run:
    python iajd_gui.py
"""
from __future__ import annotations
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk, scrolledtext

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from iajd_predict import ALLOWED_FAMILIES, predict  # noqa: E402

FAMILY_CHOICES = ["(auto)"] + sorted(ALLOWED_FAMILIES)


def _fmt_ci(ci) -> str:
    if not ci:
        return "—"
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _result_text(r: dict) -> str:
    lines = []
    lines.append(f"Canonical SMILES : {r.get('canonical_smiles')}")
    lines.append(f"Family used      : {r.get('family_used')}")
    src = r.get("family_sources", {}) or {}
    lines.append(
        f"  pka stage: {src.get('pka')} | bioact stage: {src.get('bioact')} | "
        f"user hint: {src.get('user_hint')}"
    )
    lines.append("")

    if r.get("error"):
        lines.append(f"ERROR: {r['error']}")
        return "\n".join(lines)

    pka = r.get("pka", {}) or {}
    if "point" in pka:
        lines.append(f"pKa              : {pka['point']:.3f}  (tier={pka.get('tier')}, source={pka.get('source')})")
        lines.append(f"  60% CI         : {_fmt_ci(pka.get('ci_60'))}   sigma={pka.get('sigma')}")
        lines.append(f"  90% CI         : {_fmt_ci(pka.get('ci_90'))}")
        if pka.get("max_tanimoto_to_training") is not None:
            lines.append(
                f"  max Tanimoto    : {pka['max_tanimoto_to_training']:.3f}   OOD={pka.get('ood_flag')}"
            )
    lines.append("")

    bio = r.get("bioactivity", {}) or {}
    if "point" in bio:
        lines.append(
            f"log10 flux total : {bio['point']:.3f}  (tier={bio.get('tier')}, alpha={bio.get('alpha_used')})"
        )
        lines.append(f"  60% CI         : {_fmt_ci(bio.get('ci_60'))}   sigma={bio.get('sigma')}")
        lines.append(f"  90% CI         : {_fmt_ci(bio.get('ci_90'))}")
        lines.append(
            f"  max Tanimoto    : {bio.get('max_tanimoto'):.3f}   LION_real={bio.get('block_B_real')}   "
            f"B={bio.get('block_B_active')} C={bio.get('block_C_active')}"
        )
    elif bio.get("error"):
        lines.append(f"bioactivity ERROR: {bio['error']}")
    lines.append("")

    nb = r.get("neighbors", {}) or {}
    lines.append("Nearest pKa training IAJDs:")
    if isinstance(nb.get("pka"), list):
        for n in nb["pka"]:
            lines.append(
                f"  IAJD #{n['iajd_id']!s:<10}  tan={n['tanimoto']:.3f}  fam={n['family']:<22}  "
                f"pKa={n.get('pKa')}"
            )
    else:
        lines.append(f"  (lookup failed: {nb.get('error')})")

    lines.append("")
    lines.append("Nearest bioactivity training IAJDs:")
    if isinstance(nb.get("bioact"), list):
        for n in nb["bioact"]:
            lines.append(
                f"  IAJD #{n['iajd_id']!s:<10}  tan={n['tanimoto']:.3f}  fam={n['family']:<22}  "
                f"log10_flux={n.get('log10_flux_total')}"
            )
    else:
        lines.append(f"  (lookup failed: {nb.get('error')})")

    warns = r.get("warnings", []) or []
    if warns:
        lines.append("")
        lines.append("Warnings:")
        for w in warns:
            lines.append(f"  - {w}")
    return "\n".join(lines)


class IAJDApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("STRIDE")
        self.geometry("900x700")

        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="x")

        ttk.Label(frm, text="SMILES:").grid(row=0, column=0, sticky="w")
        self.smiles_entry = ttk.Entry(frm, width=80)
        self.smiles_entry.grid(row=0, column=1, columnspan=3, sticky="we", padx=6)
        self.smiles_entry.insert(
            0, "CCCCCCCCOCC(COCCCCCCCC)(COCCCCCCCC)COC(=O)CCCN1CCN(CCO)CC1"
        )

        ttk.Label(frm, text="Family:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.family_var = tk.StringVar(value="PE-Tris")
        self.family_combo = ttk.Combobox(
            frm, textvariable=self.family_var, values=FAMILY_CHOICES, state="readonly", width=22
        )
        self.family_combo.grid(row=1, column=1, sticky="w", padx=6, pady=(6, 0))

        ttk.Label(frm, text="Neighbors:").grid(row=1, column=2, sticky="e", pady=(6, 0))
        self.k_var = tk.IntVar(value=5)
        self.k_spin = ttk.Spinbox(frm, from_=1, to=20, textvariable=self.k_var, width=4)
        self.k_spin.grid(row=1, column=3, sticky="w", pady=(6, 0))

        btn_row = ttk.Frame(self, padding=(10, 0))
        btn_row.pack(fill="x")
        self.run_btn = ttk.Button(btn_row, text="Predict", command=self._on_predict)
        self.run_btn.pack(side="left")
        self.status = ttk.Label(btn_row, text="Ready.")
        self.status.pack(side="left", padx=10)

        self.output = scrolledtext.ScrolledText(
            self, wrap="word", font=("Menlo", 11), height=30
        )
        self.output.pack(fill="both", expand=True, padx=10, pady=10)

        # warm up the bundle on idle so the first Predict isn't slow
        self.after(200, self._warm_bundle)

    def _warm_bundle(self) -> None:
        def _warm():
            try:
                from iajd_predict import _load_bundle as _lb
                _lb()
                self.status.config(text="Bundle loaded. Ready.")
            except Exception as exc:  # noqa: BLE001
                self.status.config(text=f"Bundle load failed: {exc}")
        threading.Thread(target=_warm, daemon=True).start()

    def _on_predict(self) -> None:
        smiles = self.smiles_entry.get().strip()
        if not smiles:
            self.status.config(text="Enter a SMILES.")
            return
        fam = self.family_var.get()
        family = None if fam == "(auto)" else fam
        k = max(1, int(self.k_var.get()))
        self.run_btn.config(state="disabled")
        self.status.config(text="Predicting...")
        self.output.delete("1.0", "end")

        def _run():
            try:
                r = predict(smiles, family=family, neighbors=k)
                text = _result_text(r)
            except Exception as exc:  # noqa: BLE001
                text = f"Prediction failed: {type(exc).__name__}: {exc}"
            self.after(0, lambda: self._finish(text))

        threading.Thread(target=_run, daemon=True).start()

    def _finish(self, text: str) -> None:
        self.output.insert("1.0", text)
        self.status.config(text="Done.")
        self.run_btn.config(state="normal")


if __name__ == "__main__":
    IAJDApp().mainloop()
