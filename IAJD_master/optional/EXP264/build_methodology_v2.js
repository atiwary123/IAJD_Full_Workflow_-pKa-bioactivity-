// build_methodology_v2.js — Two slides: intuitive workflow narrative + roadmap.

const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";  // 13.3" x 7.5"
pres.author = "Aryaman Singh / Percec Lab";
pres.title  = "IAJD Bioactivity — Methodology & Roadmap";

// =====================================================================
// SLIDE 1 — Workflow
// =====================================================================
const s1 = pres.addSlide();
s1.background = { color: "FFFFFF" };

s1.addText("How the bioactivity predictor works", {
  x: 0.5, y: 0.30, w: 12.3, h: 0.55,
  fontSize: 24, fontFace: "Calibri", bold: true, color: "000000", align: "left"
});

s1.addText("Given a molecule, we answer two questions in parallel: which organ does it target, and how much delivery occurs there.",
  { x: 0.5, y: 0.85, w: 12.3, h: 0.40,
    fontSize: 13, fontFace: "Calibri", color: "333333", italic: true, align: "left" });

// Helper: rounded box with story title + technical caption
function storyBox(x, y, w, h, story, technical) {
  s1.addShape(pres.shapes.RECTANGLE, {
    x: x, y: y, w: w, h: h,
    fill: { color: "FFFFFF" },
    line: { color: "000000", width: 1 }
  });
  // Plain-English line on top
  s1.addText(story, {
    x: x + 0.1, y: y + 0.08, w: w - 0.2, h: 0.45,
    fontSize: 13, fontFace: "Calibri", bold: true, color: "000000",
    align: "center", valign: "middle", margin: 0
  });
  // Technical caption underneath
  s1.addText(technical, {
    x: x + 0.1, y: y + 0.55, w: w - 0.2, h: h - 0.60,
    fontSize: 9, fontFace: "Calibri", italic: true, color: "666666",
    align: "center", valign: "top", margin: 0
  });
}

function arrow(x1, y1, x2, y2) {
  s1.addShape(pres.shapes.LINE, {
    x: x1, y: y1, w: x2-x1, h: y2-y1,
    line: { color: "000000", width: 1.5, endArrowType: "triangle", endArrowSize: 6 }
  });
}

// LEFT COLUMN — Input pipeline (vertical narrative on the left)
const LEFT_X = 0.55;
const LEFT_W = 4.2;
const STEP_H = 0.95;

// Step 1: Draw the molecule
storyBox(LEFT_X, 1.45, LEFT_W, STEP_H,
  "1.  Start with the molecule",
  "Input: SMILES string (chemical structure)"
);
arrow(LEFT_X + LEFT_W/2, 1.45 + STEP_H, LEFT_X + LEFT_W/2, 1.45 + STEP_H + 0.20);

// Step 2: Featurize  (taller — has 4 lines of fine print)
const STEP2_Y = 1.45 + STEP_H + 0.22;
const STEP2_H = 1.50;
storyBox(LEFT_X, STEP2_Y, LEFT_W, STEP2_H,
  "2.  Describe it with 88 numbers",
  "Block A 50 (RDKit + 3D descriptors + pKa)\n+ Block D 6 (membrane geometry, charge density)\n+ Block B 14 + Block C 10  (external pretrained models)\n+ Formulation 8  (DLS, dose, buffer)"
);
arrow(LEFT_X + LEFT_W/2, STEP2_Y + STEP2_H,
      LEFT_X + LEFT_W/2, STEP2_Y + STEP2_H + 0.18);

// Step 3: Find lookalikes in training set
const STEP3_Y = STEP2_Y + STEP2_H + 0.20;
storyBox(LEFT_X, STEP3_Y, LEFT_W, STEP_H + 0.05,
  "3.  Find lookalikes we already tested",
  "Tanimoto on Morgan-2 fingerprints, top K=8\nnearest training compounds.  Used by Stage B."
);

// CENTER — branching arrows from feature box (left col) into right column
const RIGHT_X = 7.50;     // start of right-column boxes
const RIGHT_W = 5.30;

// Right column: two parallel questions
// Top: WHERE
const Y_WHERE = 1.45;
const WHERE_H = 2.30;
s1.addShape(pres.shapes.RECTANGLE, {
  x: RIGHT_X, y: Y_WHERE, w: RIGHT_W, h: WHERE_H,
  fill: { color: "F2F2F2" },
  line: { color: "000000", width: 1 }
});
s1.addText("WHERE does it deliver?", {
  x: RIGHT_X + 0.15, y: Y_WHERE + 0.08, w: RIGHT_W - 0.3, h: 0.40,
  fontSize: 16, fontFace: "Calibri", bold: true, color: "000000",
  align: "left", valign: "middle", margin: 0
});
s1.addText("Stage A — six binary classifiers, one per organ", {
  x: RIGHT_X + 0.15, y: Y_WHERE + 0.46, w: RIGHT_W - 0.3, h: 0.25,
  fontSize: 10, fontFace: "Calibri", italic: true, color: "666666",
  align: "left", valign: "top", margin: 0
});
s1.addText([
  { text: "Each classifier asks: ", options: { fontSize: 11, bold: true } },
  { text: "“is this compound a strong", options: { fontSize: 11, breakLine: true } },
  { text: "  delivery agent for spleen?  for liver?  for lung?  for LN?”", options: { fontSize: 11, breakLine: true } },
  { text: " ", options: { fontSize: 4, breakLine: true } },
  { text: "Output: ", options: { fontSize: 11, bold: true } },
  { text: "calibrated probability per organ.", options: { fontSize: 11, breakLine: true } },
  { text: "Argmax over (spleen, liver, lung) gives the “dominant organ” call.", options: { fontSize: 11, breakLine: true } },
  { text: " ", options: { fontSize: 4, breakLine: true } },
  { text: "Trained on 210 compounds with binary “strong” labels", options: { fontSize: 9, italic: true, color: "666666" } },
], {
  x: RIGHT_X + 0.15, y: Y_WHERE + 0.78, w: RIGHT_W - 0.3, h: WHERE_H - 0.85,
  fontFace: "Calibri", color: "000000", align: "left", valign: "top", margin: 0
});

// Bottom: HOW MUCH
const Y_HOW = Y_WHERE + WHERE_H + 0.20;
const HOW_H = 2.50;
s1.addShape(pres.shapes.RECTANGLE, {
  x: RIGHT_X, y: Y_HOW, w: RIGHT_W, h: HOW_H,
  fill: { color: "F2F2F2" },
  line: { color: "000000", width: 1 }
});
s1.addText("HOW MUCH does it deliver?", {
  x: RIGHT_X + 0.15, y: Y_HOW + 0.08, w: RIGHT_W - 0.3, h: 0.40,
  fontSize: 16, fontFace: "Calibri", bold: true, color: "000000",
  align: "left", valign: "middle", margin: 0
});
s1.addText("Stage B — five regressors, one per organ flux target", {
  x: RIGHT_X + 0.15, y: Y_HOW + 0.46, w: RIGHT_W - 0.3, h: 0.25,
  fontSize: 10, fontFace: "Calibri", italic: true, color: "666666",
  align: "left", valign: "top", margin: 0
});
s1.addText([
  { text: "Two predictions are made and blended:", options: { fontSize: 11, bold: true, breakLine: true } },
  { text: " ", options: { fontSize: 4, breakLine: true } },
  { text: "Direct prediction: ", options: { fontSize: 11, bold: true, bullet: true } },
  { text: "what does the model expect from", options: { fontSize: 11, breakLine: true } },
  { text: "the chemistry alone?  (XGBoost on the 88 features)", options: { fontSize: 11, breakLine: true } },
  { text: "Analog prediction: ", options: { fontSize: 11, bold: true, bullet: true } },
  { text: "average flux of the K=8 closest", options: { fontSize: 11, breakLine: true } },
  { text: "training compounds, weighted by Tanimoto similarity", options: { fontSize: 11, breakLine: true } },
  { text: " ", options: { fontSize: 4, breakLine: true } },
  { text: "Mixing ratio α is set per family.  ", options: { fontSize: 11 } },
  { text: "Output: log10(photons/sec) + 90% PI", options: { fontSize: 11, bold: true, breakLine: true } },
  { text: " ", options: { fontSize: 4, breakLine: true } },
  { text: "Trained on 147 compounds with measured in vivo IVIS flux", options: { fontSize: 9, italic: true, color: "666666" } },
], {
  x: RIGHT_X + 0.15, y: Y_HOW + 0.78, w: RIGHT_W - 0.3, h: HOW_H - 0.85,
  fontFace: "Calibri", color: "000000", align: "left", valign: "top", margin: 0
});

// Two arrows from feature box (Step 2) into right-side boxes
arrow(LEFT_X + LEFT_W, STEP2_Y + STEP2_H * 0.35,
      RIGHT_X, Y_WHERE + WHERE_H/2);
arrow(LEFT_X + LEFT_W, STEP2_Y + STEP2_H * 0.65,
      RIGHT_X, Y_HOW + HOW_H/2);

// Bottom strip: combined output / confidence tier
const Y_FINAL = Y_HOW + HOW_H + 0.18;
s1.addShape(pres.shapes.RECTANGLE, {
  x: 0.55, y: Y_FINAL, w: 12.25, h: 0.55,
  fill: { color: "FFFFFF" }, line: { color: "000000", width: 1.5 }
});
s1.addText([
  { text: "Combined output:  ", options: { bold: true, fontSize: 12 } },
  { text: "organ probabilities + per-organ flux + 90% PI + ", options: { fontSize: 12 } },
  { text: "confidence tier (HIGH / MEDIUM / LOW)", options: { bold: true, fontSize: 12 } },
], {
  x: 0.65, y: Y_FINAL, w: 12.05, h: 0.55,
  fontFace: "Calibri", color: "000000", align: "center", valign: "middle", margin: 0
});

// Footnote
s1.addText(
  "The confidence tier is the minimum of four signals: prediction-interval width, similarity-hierarchy level (1 = exact match, 5 = unfamiliar), Tanimoto to training set, and external-model OOD flag.",
  { x: 0.5, y: 7.20, w: 12.3, h: 0.30,
    fontSize: 9, fontFace: "Calibri", color: "555555", italic: true, align: "left" }
);


// =====================================================================
// SLIDE 2 — Roadmap
// =====================================================================
const s2 = pres.addSlide();
s2.background = { color: "FFFFFF" };

s2.addText("What's next — improvements in the pipeline", {
  x: 0.5, y: 0.30, w: 12.3, h: 0.55,
  fontSize: 24, fontFace: "Calibri", bold: true, color: "000000", align: "left"
});

s2.addText("Each item below has a known expected gain on the model's prediction error. The current model holds the floor at MAE 0.39 on held-out compounds; these are the levers to push it lower.",
  { x: 0.5, y: 0.85, w: 12.3, h: 0.40,
    fontSize: 12, fontFace: "Calibri", color: "333333", italic: true, align: "left" });

// Helper for roadmap row: one box per priority tier
function roadmapBox(x, y, w, h, header, items) {
  s2.addShape(pres.shapes.RECTANGLE, {
    x: x, y: y, w: w, h: h,
    fill: { color: "FFFFFF" },
    line: { color: "000000", width: 1 }
  });
  s2.addText(header, {
    x: x, y: y, w: w, h: 0.42,
    fontSize: 14, fontFace: "Calibri", bold: true, color: "000000",
    fill: { color: "F2F2F2" },
    align: "center", valign: "middle", margin: 0
  });
  // Build the body content
  const body = [];
  items.forEach((it, idx) => {
    body.push({ text: it.title, options: { fontSize: 11, bold: true, breakLine: true } });
    body.push({ text: it.desc,  options: { fontSize: 10, breakLine: true } });
    body.push({ text: it.gain,  options: { fontSize: 9,  italic: true, color: "555555",
                                          breakLine: idx < items.length - 1 } });
    if (idx < items.length - 1) {
      body.push({ text: " ", options: { fontSize: 5, breakLine: true } });
    }
  });
  s2.addText(body, {
    x: x + 0.15, y: y + 0.50, w: w - 0.30, h: h - 0.55,
    fontFace: "Calibri", color: "000000", align: "left", valign: "top", margin: 0
  });
}

const ROW_Y = 1.40;
const ROW_H = 5.50;
const COL_W = 4.05;
const GAP = 0.10;
const COL1_X = 0.55;
const COL2_X = COL1_X + COL_W + GAP;
const COL3_X = COL2_X + COL_W + GAP;

// COLUMN 1 — Easy wins
roadmapBox(COL1_X, ROW_Y, COL_W, ROW_H, "Near-term  (weeks)", [
  {
    title: "Activate Block B (LiON)",
    desc: "Switch on the pretrained LNP-delivery model from Anderson lab (~9,000 LNP measurements across liver-IV, lung-IT, lung-inhaled, lung-nebulized, muscle-IM, nasal). Currently shipping zeros.",
    gain: "Expected ΔMAE: −0.03 to −0.08 log-units"
  },
  {
    title: "Activate Block C (ADMET-AI)",
    desc: "Switch on the Stanford drug-property model (41 pharmacokinetic endpoints). Currently using RDKit-derived stand-ins for 10 endpoints.",
    gain: "Expected ΔMAE: 0 to −0.02"
  },
  {
    title: "Drop in v7.x pKa surrogate",
    desc: "Current pKa from v1.0 baked-in GPR (LOO MAE 0.146). Production v7.x bundle exists at LOO MAE 0.074 — needs to be packaged as a joblib and wired in.",
    gain: "Expected: tighter pKa-derived features in Block D"
  },
]);

// COLUMN 2 — Mid-term: data
roadmapBox(COL2_X, ROW_Y, COL_W, ROW_H, "Mid-term  (months)", [
  {
    title: "Recover the 41 dropped IAJDs",
    desc: "Bioactivity-measured IAJDs lacking v21 SMILES at v07 build time. Each needs SMILES construction + MALDI-TOF MW cross-validation. Biggest structural data improvement available.",
    gain: "Expected: per-family balance + lower variance"
  },
  {
    title: "Digitize qualitative organ labels",
    desc: "Roughly 60 IAJDs in ja2c00273 Table S2 + ja1c05813 Table S7 have strong/weak organ labels but no numbers. Converting to ordinal targets unlocks per-organ Stage B regressors.",
    gain: "Especially helps lymph-node target (current n_pos=8 → ~30+)"
  },
  {
    title: "MAP4 fingerprints for X2 isomers",
    desc: "Current Morgan-2 fingerprints can't distinguish constitutional isomers (3,4 vs 3,5 substitution). MAP4 captures atom-level positional context and would resolve the X2 rank-order weakness (Spearman 0.47 → est. 0.65+).",
    gain: "Expected ΔMAE: −0.01 to −0.03"
  },
]);

// COLUMN 3 — Long-term: publication-grade
roadmapBox(COL3_X, ROW_Y, COL_W, ROW_H, "Long-term  (publication-grade)", [
  {
    title: "Multi-output Gaussian Process",
    desc: "Replace independent XGBoost regressors with a multi-output GP using rank-2 ICM coregionalization. Captures cross-organ correlations (compounds that hit spleen often hit LN). Composite Tanimoto + Matérn kernel.",
    gain: "Expected ΔMAE: −0.02 to −0.05"
  },
  {
    title: "GPU-fine-tune LiON on IAJDs",
    desc: "Once Block B is activated, fine-tune the pretrained LiON checkpoints for 3 epochs on the IAJD bioactivity set. Adapts the LNP-delivery prior to IAJD chemistry without losing the 9,000-LNP transfer learning.",
    gain: "Expected ΔMAE: −0.05 to −0.10  (~12 GPU-hours)"
  },
  {
    title: "CG-MD membrane descriptors",
    desc: "MARTINI 3 simulations of dendrimersome self-assembly. Adds bilayer thickness, area-per-lipid, S_CD order parameters, and tilt angle to Block D. Captures the actual physics of IAJD packing.",
    gain: "Expected ΔMAE: −0.03 to −0.07  (~1 month HPC)"
  },
]);

// Footnote: cumulative target
s2.addText([
  { text: "Cumulative expected MAE if all near-term + mid-term items land:  ", options: { fontSize: 11, bold: true } },
  { text: "0.39  →  ~0.30   (matches v2.0-FULL spec target).  ", options: { fontSize: 11 } },
  { text: "Long-term column is publication-grade and gates the counterfactual / Pareto API.", options: { fontSize: 11, italic: true } },
], {
  x: 0.5, y: 7.05, w: 12.3, h: 0.40,
  fontFace: "Calibri", color: "000000", align: "left", valign: "top", margin: 0
});

// Save
const outPath = "/mnt/user-data/outputs/EXP264_methodology_v2.pptx";
pres.writeFile({ fileName: outPath }).then(() => {
  console.log(`Saved: ${outPath}`);
});
