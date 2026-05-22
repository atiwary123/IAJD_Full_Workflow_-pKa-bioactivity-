// build_exp264_deck.js — Generate the EXP264 bioactivity prediction deck.
// Style: default Office, white background, black text, academic.

const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const data = JSON.parse(fs.readFileSync("/home/claude/exp264_slides.json", "utf8"));

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";  // 13.3" x 7.5"
pres.author = "Aryaman Singh / Percec Lab";
pres.title  = "EXP264 — Bioactivity Predictions (v2.0-MIN-v09)";

// ---------- Helpers ----------
function fmt(v, digits=2) {
  if (v === null || v === undefined) return "n/a";
  if (typeof v !== "number") return String(v);
  return v.toFixed(digits);
}
function fmtPi(pi, digits=2) {
  if (!pi || pi[0] === null || pi[0] === undefined) return "n/a";
  return `[${pi[0].toFixed(digits)}, ${pi[1].toFixed(digits)}]`;
}
function fmtSci(v) {
  if (v === null || v === undefined) return "n/a";
  return v.toExponential(2);
}

// Wrap long SMILES so it fits in the textbox (~70 chars per line)
function wrapSmiles(smi, width=72) {
  if (smi.length <= width) return smi;
  const out = [];
  for (let i = 0; i < smi.length; i += width) out.push(smi.slice(i, i+width));
  return out.join("\n");
}

// ---------- Title slide ----------
{
  const s = pres.addSlide();
  s.background = { color: "FFFFFF" };
  s.addText("EXP264 — Predicted Bioactivity", {
    x: 0.7, y: 2.4, w: 12, h: 1.0,
    fontSize: 36, fontFace: "Calibri", bold: true, color: "000000", align: "left"
  });
  s.addText("Five novel IAJDs · Bioactivity model v2.0-MIN-v09 · pKa from baked-in surrogate",
    { x: 0.7, y: 3.4, w: 12, h: 0.5,
      fontSize: 18, fontFace: "Calibri", color: "333333", align: "left" });
  s.addText("Percec Lab · University of Pennsylvania", {
    x: 0.7, y: 6.4, w: 12, h: 0.4,
    fontSize: 12, fontFace: "Calibri", color: "555555", italic: true, align: "left"
  });
}

// ---------- Per-compound slides ----------
data.forEach((d, idx) => {
  const s = pres.addSlide();
  s.background = { color: "FFFFFF" };

  // ---- Title row ----
  s.addText(`${d.name} — predicted bioactivity`, {
    x: 0.5, y: 0.30, w: 12.3, h: 0.55,
    fontSize: 24, fontFace: "Calibri", bold: true, color: "000000", align: "left"
  });

  // ---- Structure (left) ----
  const pngPath = `/home/claude/exp264_pngs/EXP264_${idx+1}.png`;
  s.addImage({
    path: pngPath,
    x: 0.5, y: 0.95, w: 6.8, h: 3.6,
    sizing: { type: "contain", w: 6.8, h: 3.6 }
  });

  // ---- Identification block (right of structure) ----
  const idLines = [
    { text: "Identification", options: { bold: true, fontSize: 14, breakLine: true } },
    { text: `Formula:  ${d.formula}`, options: { fontSize: 12, breakLine: true } },
    { text: `MW (exact):  ${fmt(d.mw, 3)}`, options: { fontSize: 12, breakLine: true } },
    { text: `Family (assigned):  ${d.family_predicted}`, options: { fontSize: 12, breakLine: true } },
    { text: `Head group:  ${d.head}`, options: { fontSize: 12, breakLine: true } },
    { text: `Chain pair:  ${d.chains[0]}, ${d.chains[1]}  (alkyl C-count, model-extracted)`, options: { fontSize: 12, breakLine: true } },
    { text: " ", options: { fontSize: 6, breakLine: true } },
    { text: "pKa surrogate", options: { bold: true, fontSize: 14, breakLine: true } },
    { text: `pKa = ${fmt(d.pka, 2)}   PI90 ${fmtPi(d.pka_pi, 2)}`, options: { fontSize: 12, breakLine: true } },
    { text: " ", options: { fontSize: 6, breakLine: true } },
    { text: "Domain", options: { bold: true, fontSize: 14, breakLine: true } },
    { text: `Hierarchy level:  ${d.level}   (1 = exact match, 5 = OOD)`, options: { fontSize: 12, breakLine: true } },
    { text: `Max Tanimoto to training:  ${fmt(d.tanimoto, 3)}`, options: { fontSize: 12, breakLine: true } },
    { text: `Confidence tier:  ${d.confidence}`, options: { fontSize: 12 } },
  ];
  s.addText(idLines, {
    x: 7.5, y: 0.95, w: 5.4, h: 3.6,
    fontFace: "Calibri", color: "000000", valign: "top"
  });

  // ---- SMILES row ----
  s.addText([
    { text: "SMILES: ", options: { bold: true, fontSize: 10 } },
    { text: wrapSmiles(d.smiles, 105), options: { fontSize: 10, fontFace: "Consolas" } }
  ], {
    x: 0.5, y: 4.65, w: 12.3, h: 0.5,
    fontFace: "Calibri", color: "000000", valign: "top"
  });

  // ---- Predicted bioactivity table ----
  const tableHeader = [
    { text: "Endpoint",       options: { bold: true, fontSize: 11 } },
    { text: "log10 flux",     options: { bold: true, fontSize: 11, align: "right" } },
    { text: "PI 90%",         options: { bold: true, fontSize: 11, align: "center" } },
    { text: "Note",           options: { bold: true, fontSize: 11 } },
  ];
  const stage_a = d.organ_probs;
  const tableRows = [
    tableHeader,
    [{ text: "Total flux",  options: { fontSize: 10 } },
     { text: fmt(d.flux_total, 2), options: { fontSize: 10, align: "right" } },
     { text: fmtPi(d.flux_total_pi90, 2), options: { fontSize: 10, align: "center" } },
     { text: `${d.bin || "n/a"} bin (P=${fmt(d.bin_probs ? d.bin_probs[d.bin] : null, 2)})`, options: { fontSize: 10 } }],
    [{ text: "Spleen flux", options: { fontSize: 10 } },
     { text: fmt(d.flux_spleen, 2), options: { fontSize: 10, align: "right" } },
     { text: fmtPi(d.flux_spleen_pi90, 2), options: { fontSize: 10, align: "center" } },
     { text: `Stage A p(spleen-strong)=${fmt(stage_a.spleen_strong, 2)}`, options: { fontSize: 10 } }],
    [{ text: "Liver flux",  options: { fontSize: 10 } },
     { text: fmt(d.flux_liver, 2), options: { fontSize: 10, align: "right" } },
     { text: fmtPi(d.flux_liver_pi90, 2), options: { fontSize: 10, align: "center" } },
     { text: `Stage A p(liver-strong)=${fmt(stage_a.liver_strong, 2)}`, options: { fontSize: 10 } }],
    [{ text: "Lung flux",   options: { fontSize: 10 } },
     { text: fmt(d.flux_lung, 2), options: { fontSize: 10, align: "right" } },
     { text: fmtPi(d.flux_lung_pi90, 2), options: { fontSize: 10, align: "center" } },
     { text: `Stage A p(lung-strong)=${fmt(stage_a.lung_strong, 2)}`, options: { fontSize: 10 } }],
    [{ text: "Lymph node flux", options: { fontSize: 10 } },
     { text: fmt(d.flux_LN, 2), options: { fontSize: 10, align: "right" } },
     { text: fmtPi(d.flux_LN_pi90, 2), options: { fontSize: 10, align: "center" } },
     { text: `Stage A p(LN-strong)=${fmt(stage_a.LN_strong, 2)}`, options: { fontSize: 10 } }],
  ];
  s.addTable(tableRows, {
    x: 0.5, y: 5.20, w: 7.5,
    colW: [1.6, 1.0, 1.6, 3.3],
    border: { type: "solid", pt: 0.5, color: "999999" },
    fontFace: "Calibri",
    rowH: 0.28,
  });

  // ---- Dominant-organ summary box ----
  const dominantTxt = d.organ_dominant + (d.organ_tied ? `  (tied with ${d.organ_tied})` : "");
  const linearTxt = `Linear flux:  ${fmtSci(d.flux_linear)} p/s`;
  s.addText([
    { text: "Stage A call", options: { bold: true, fontSize: 12, breakLine: true } },
    { text: `Dominant organ: ${dominantTxt}`, options: { fontSize: 11, breakLine: true } },
    { text: " ", options: { fontSize: 4, breakLine: true } },
    { text: "Convenience (Stage B)", options: { bold: true, fontSize: 12, breakLine: true } },
    { text: linearTxt, options: { fontSize: 11 } },
  ], {
    x: 8.2, y: 5.20, w: 4.8, h: 1.65,
    fontFace: "Calibri", color: "000000", valign: "top"
  });

  // ---- Explanation paragraph (bottom strip) ----
  // Build a one-paragraph explanation grounded in this compound's facts.
  let explanation = "";
  // Build the explanation contextually
  let nnSummary = "";
  if (d.nearest_neighbors && d.nearest_neighbors.length > 0) {
    const items = d.nearest_neighbors.map(nn => {
      const f = (nn.log10_flux_total !== null && nn.log10_flux_total !== undefined)
                  ? nn.log10_flux_total.toFixed(2) : "n/a";
      return `${nn.IAJD_id} (${nn.family}, T=${nn.tanimoto.toFixed(2)}, flux ${f})`;
    });
    nnSummary = "Nearest training neighbors: " + items.join("; ") + ".";
  }
  // Custom sentences keyed by hierarchy level
  let levelNote = "";
  if (d.level === 1) {
    levelNote = `This SMILES exactly matches a training compound (Tanimoto = 1.0), so the prediction sits on top of measured data; the wide PI reflects per-family experimental variance rather than model uncertainty.`;
  } else if (d.level === 3) {
    levelNote = `Hierarchy level 3 (close in-family analog, Tanimoto ≥ 0.85) — analog-delta lookup carries most of the signal; conformal PI is the per-family quantile.`;
  } else if (d.level === 4) {
    levelNote = `Hierarchy level 4 (in-family analog with Tanimoto ${d.tanimoto.toFixed(2)}) — model interpolates from a small handful of close training compounds.`;
  } else if (d.level === 5) {
    levelNote = `Hierarchy level 5 (no close training neighbor, Tanimoto < 0.50) — prediction is OOD; PI is doubled per spec §5.5.`;
  }

  explanation = `${levelNote}  ${nnSummary}`;

  s.addText(explanation, {
    x: 0.5, y: 6.95, w: 12.3, h: 0.5,
    fontSize: 10, fontFace: "Calibri", color: "333333", align: "left", italic: false, valign: "top"
  });
});

// ---------- Methodology slide ----------
{
  const s = pres.addSlide();
  s.background = { color: "FFFFFF" };
  s.addText("Methodology and caveats", {
    x: 0.5, y: 0.4, w: 12.3, h: 0.6,
    fontSize: 24, fontFace: "Calibri", bold: true, color: "000000"
  });

  const lines = [
    { text: "Model", options: { bold: true, fontSize: 14, breakLine: true } },
    { text: "v2.0-MIN-v09 three-stage cascade. 88-dim feature vector (Block A 50 + B 14 + C 10 + D 6 + Form 8). Stage A: six per-organ XGBoost binary classifiers with isotonic calibration. Stage B: direct-XGBoost + analog-delta blend with per-family α routing, conformal 90% PI. Stage C disabled (formulation completeness < 75% per spec §5.4).", options: { fontSize: 11, breakLine: true } },
    { text: " ", options: { fontSize: 6, breakLine: true } },
    { text: "Training data", options: { bold: true, fontSize: 14, breakLine: true } },
    { text: "v09 dataset: 210 rows, 156 unique IAJDs, 9 papers. TRAIN n=170; X1 stability holdout n=11; X2 isomers n=13; X3 stratified-random n=16. Family balance: PE-Gallic 48 / PE-Tris 50 / sSS-Nonsym 48 / GA-Tris 48 / Dialkoxybenzyl 16.", options: { fontSize: 11, breakLine: true } },
    { text: " ", options: { fontSize: 6, breakLine: true } },
    { text: "Held-out validation (X3, n=16)", options: { bold: true, fontSize: 14, breakLine: true } },
    { text: "log10_flux_total MAE = 0.386, PI90 coverage = 0.875 (in [0.85, 0.92] target). Stage A dominant-organ accuracy 3/4. X1 stability variance 0.041 < 0.20 spec target. All 7 v2.0-MIN production gates pass.", options: { fontSize: 11, breakLine: true } },
    { text: " ", options: { fontSize: 6, breakLine: true } },
    { text: "pKa", options: { bold: true, fontSize: 14, breakLine: true } },
    { text: "Predictions are from the v1.0 baked-in GPR surrogate (LOO MAE 0.146 on 243 v9-pKa training compounds). The current pKa workflow (v7.x with MAE ~0.074) is not deployed as a joblib in this environment; pKa numbers should be treated as ±0.30 log-units rather than the v7.x precision.", options: { fontSize: 11, breakLine: true } },
    { text: " ", options: { fontSize: 6, breakLine: true } },
    { text: "Known caveats for the EXP264 set", options: { bold: true, fontSize: 14, breakLine: true } },
    { text: "(1) EXP264-2 through 5 are 3,4,5-trialkoxy benzyl esters (GA-Tris-family scaffolds). The bundle's family classifier returns sSS-Nonsym for these because the GA-Tris SMARTS matches an amide variant only; this means per-family α routes through 0.05 (delta-leaning) rather than 1.00 (direct-only). The nearest-neighbor analog list is correct (all GA-Tris); the predictions therefore lean on those neighbor fluxes — directionally appropriate but a known mislabeling.  (2) Block B (LiON pretrained inference) and Block C (real ADMET-AI) are inactive — Block C uses an RDKit physicochemical stand-in. Activating them is the v2.0-FULL roadmap step expected to drop pooled MAE by 0.03–0.10.  (3) Confidence tiers cap at MEDIUM bundle-wide while LiON is inactive (the lion_OOD signal is forced neutral).", options: { fontSize: 11 } },
  ];
  s.addText(lines, {
    x: 0.5, y: 1.1, w: 12.3, h: 6.0,
    fontFace: "Calibri", color: "000000", valign: "top"
  });
}

// ---------- Save ----------
const outPath = "/mnt/user-data/outputs/EXP264_predictions.pptx";
pres.writeFile({ fileName: outPath }).then(() => {
  console.log(`Saved: ${outPath}`);
});
