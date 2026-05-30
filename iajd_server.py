"""
iajd_server.py — Localhost HTTP frontend for STRIDE.

STRIDE = STRuctural Ranking + Informed Design Engine: pKa + bioactivity
tandem predictor with a beam-search proposer for ionizable amphiphilic Janus
dendrimers (IAJDs).

Pure stdlib (no Flask / FastAPI). Serves:
    GET  /              — HTML form (SMILES field, ChemDraw upload, family, neighbors)
    POST /predict       — JSON in / JSON out for a single SMILES query
    POST /predict_batch — multipart form upload (ChemDraw/.cdxml/.mol/.sdf/.smi)
                          or JSON {smiles_text}; runs each parsed structure
                          through the tandem predictor with structural refinement.
    GET  /health        — {"ok": true, "bundle_loaded": bool}

Run:
    .venv/bin/python iajd_server.py [--port 8000] [--host 127.0.0.1]

Then open  http://127.0.0.1:8000  in a browser. The bundle is warmed at startup
so the first prediction doesn't pay the load cost.
"""
from __future__ import annotations
import argparse
import cgi
import io
import json
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from iajd_predict import (  # noqa: E402
    ALLOWED_FAMILIES, predict, predict_batch, _load_bundle,
)

_BUNDLE_READY = threading.Event()
_BUNDLE_ERROR: str | None = None


def _warm():
    global _BUNDLE_ERROR
    try:
        _load_bundle()
    except Exception as exc:  # noqa: BLE001
        _BUNDLE_ERROR = f"{type(exc).__name__}: {exc}"
    finally:
        _BUNDLE_READY.set()


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>STRIDE</title>
<style>
  html, body { background: #ffffff; color: #000000; }
  body { font-family: Helvetica, Arial, sans-serif;
         max-width: 1100px; margin: 24px auto; padding: 0 16px; }
  h1 { margin-bottom: 4px; font-size: 22px; }
  .sub { color: #000; margin-top: 0; font-size: 14px; }
  fieldset { border: 1px solid #000; padding: 12px 14px; margin: 0 0 14px; }
  legend { font-size: 12px; padding: 0 6px; }
  .row { display: grid; gap: 10px; grid-template-columns: 1fr 220px 120px auto;
         align-items: end; }
  .row.file { grid-template-columns: 1fr 220px 120px auto; }
  label { display: block; font-size: 12px; color: #000; margin-bottom: 2px; }
  input[type=text], input[type=file], select, input[type=number], textarea {
    width: 100%; padding: 6px 8px; font: inherit; box-sizing: border-box;
    border: 1px solid #000; background: #fff; color: #000;
  }
  textarea { min-height: 80px; font-family: monospace; font-size: 12px; }
  button { padding: 8px 16px; font: inherit;
           border: 1px solid #000; background: #fff; color: #000; cursor: pointer; }
  button[disabled] { opacity: 0.6; cursor: progress; }
  .status { font-size: 13px; color: #000; margin: 6px 0 16px; }
  .grid { display: grid; gap: 16px; grid-template-columns: 1fr 1fr; }
  .card { border: 1px solid #000; padding: 14px; background: #fff; margin-bottom: 14px; }
  .card h2 { margin: 0 0 8px; font-size: 15px; }
  .card h3 { margin: 0 0 6px; font-size: 14px; }
  .num { font-weight: 600; }
  .ci  { font-size: 13px; color: #000; }
  .delta { font-size: 12px; color: #000; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 6px; }
  th, td { text-align: left; padding: 4px 8px; border-bottom: 1px solid #000; color: #000; }
  th { font-weight: 600; }
  .warn { border: 1px solid #000; padding: 8px 10px; margin-top: 10px; font-size: 13px;
          background: #fff; }
  .err  { border: 1px solid #000; padding: 10px; margin-top: 10px; background: #fff; }
  .badge { display: inline-block; padding: 1px 6px; border: 1px solid #000;
           font-size: 11px; margin-left: 6px; }
  code { background: #fff; color: #000; }
  .mol-header { border-bottom: 1px solid #000; padding-bottom: 8px; margin-bottom: 10px; }
</style>
</head>
<body>
<h1>STRIDE</h1>
<p class="sub">STructural Ranking + Informed Design Engine — pKa (v9.1) + bioactivity (v14.0) + Tanimoto neighbor routing + 2D positional structural refinement</p>

<fieldset>
  <legend>Single SMILES</legend>
  <form id="single_form">
    <div class="row">
      <div>
        <label for="smiles">SMILES</label>
        <input id="smiles" name="smiles" type="text"
          value="CCCCCCCCOCC(COCCCCCCCC)(COCCCCCCCC)COC(=O)CCCN1CCN(CCO)CC1">
      </div>
      <div>
        <label for="family">Family (optional)</label>
        <select id="family" name="family">__FAMILIES__</select>
      </div>
      <div>
        <label for="neighbors">Neighbors</label>
        <input id="neighbors" name="neighbors" type="number" min="1" max="20" value="5">
      </div>
      <button id="go" type="submit">Predict</button>
    </div>
  </form>
</fieldset>

<fieldset>
  <legend>ChemDraw / file upload — multiple IAJDs supported</legend>
  <form id="batch_form" enctype="multipart/form-data">
    <div class="row file" style="margin-bottom:8px">
      <div>
        <label for="upfile">ChemDraw (.cdxml), MOL (.mol), SDF (.sdf), SMILES (.smi). One or many structures per file. Family is <strong>auto-detected per IAJD</strong>.</label>
        <input id="upfile" name="upfile" type="file"
          accept=".cdxml,.cdx,.mol,.molfile,.sdf,.smi,.txt">
      </div>
      <div>
        <label for="family_b">Family (fallback only)</label>
        <select id="family_b" name="family_b">__FAMILIES__</select>
      </div>
      <div>
        <label for="neighbors_b">Neighbors</label>
        <input id="neighbors_b" name="neighbors_b" type="number" min="1" max="20" value="5">
      </div>
      <button id="go_b" type="submit">Predict batch</button>
    </div>
    <label for="smiles_text" style="margin-top:8px">…or paste SMILES (one per line). Tokens: <code>&lt;SMILES&gt; [family=&lt;FAMILY&gt;] [&lt;label&gt;]</code>. Family auto-detected if not given.</label>
    <textarea id="smiles_text" name="smiles_text" placeholder="# Comment lines and blanks are ignored. Per-row family hints override the dropdown.
CCCCCCCCOCC(...)CCN1CCN(CCO)CC1  IAJD-119
CCCCCCCCCCOCC(...)CCN1CCN(CCO)CC1  family=PE-Tris IAJD-120
CCCCC(CC)COc1cc(OCC(CC)CCCC)cc(C(=O)OCCCCN2CCN(C)CC2)c1  IAJD-DAB"></textarea>
  </form>
</fieldset>

<div id="status" class="status"></div>
<div id="out"></div>

<script>
const $ = (s) => document.querySelector(s);

function ci(arr) {
  if (!arr) return '—';
  return `[${arr[0].toFixed(3)}, ${arr[1].toFixed(3)}]`;
}

// log10 -> linear with 2-significant-digit scientific notation.
// Used for the bioactivity figures (log10 flux) so users see both the log
// model output and the linear flux it corresponds to.
function sciFromLog(v) {
  if (v == null || !isFinite(v)) return '';
  const linear = Math.pow(10, v);
  return linear.toExponential(2);
}
function logWithSci(v, decimals) {
  if (v == null) return '—';
  decimals = (decimals == null) ? 3 : decimals;
  return `${v.toFixed(decimals)} (${sciFromLog(v)})`;
}
function ciWithSci(arr) {
  if (!arr) return '—';
  return `[${arr[0].toFixed(3)} (${sciFromLog(arr[0])}), ` +
         `${arr[1].toFixed(3)} (${sciFromLog(arr[1])})]`;
}

function neighborRow(n, valueKey, isLog) {
  const v = n[valueKey];
  let vStr;
  if (v == null) vStr = '—';
  else if (typeof v !== 'number') vStr = v;
  else if (isLog) vStr = logWithSci(v, 3);
  else vStr = v.toFixed(3);
  return `<tr><td>${n.iajd_id}</td><td>${(n.tanimoto||0).toFixed(3)}</td>` +
         `<td>${n.family}</td><td class="num">${vStr}</td></tr>`;
}

function refinementTrace(sr) {
  if (!sr) return '';
  if (!sr.applied) return `<div class="delta">struct refine: skipped (${sr.reason||'—'})</div>`;
  const dlt = sr.delta_from_original;
  const dStr = (dlt == null) ? '—' : (dlt >= 0 ? `+${dlt}` : `${dlt}`);
  return `<div class="delta">struct refine: Δ=${dStr} from v15 (β=${sr.beta}, analog_refined=${sr.analog_refined})</div>`;
}

function renderOne(r, idx) {
  if (r.error) {
    return `<div class="err"><strong>${r.input_label||('mol '+(idx+1))}: </strong>${r.error}</div>`;
  }
  const pka = r.pka || {}, bio = r.bioactivity || {};
  const nb = r.neighbors || {};
  const label = r.input_label || ('mol ' + (idx+1));
  const coordSrc = r.coord_source ? `<span class="badge">${r.coord_source}</span>` : '';
  // Family-resolution trace: how was the family decided?
  const fr = r.family_resolution || {};
  const detection = fr.detection || {};
  const cands = detection.candidates || [];
  const famBadgeClass = fr.source && fr.source.includes('smarts_hard_rule') ? 'badge'
                       : fr.source && fr.source.includes('knn_unanimous') ? 'badge'
                       : fr.source && fr.source.includes('knn_majority') ? 'badge'
                       : 'badge';
  const famSrcShort = (fr.source || '—').replace('auto:', '').replace('user_hint_collapsed_to_pe_gallic', 'user→PE-Gallic');
  const subarchLine = fr.subarch_label
    ? `<div class="delta">subarch label: ${fr.subarch_label} → routed as PE-Gallic for bioact</div>` : '';
  const altsLine = (cands.length > 1)
    ? `<div class="ci">alternatives: ${cands.slice(1).map(c => `${c.family} (${(c.score*100).toFixed(0)}%)`).join(', ')}</div>` : '';
  const confLine = (detection.confidence != null)
    ? `<div class="ci">detect confidence: ${(detection.confidence * 100).toFixed(1)}% · max Tanimoto to training: ${detection.max_tanimoto}</div>` : '';
  let html = `<div class="card">`;
  html += `<div class="mol-header"><strong>${label}</strong> ${coordSrc} ` +
          `<code style="font-size:12px">${r.canonical_smiles||''}</code><br>` +
          `<span class="ci">family: <strong>${fr.family_assigned || r.family_used || '—'}</strong> ` +
          `<span class="${famBadgeClass}">${famSrcShort}</span></span>` +
          subarchLine + confLine + altsLine + `</div>`;
  html += `<div class="grid">`;
  html += `<div>
    <h3>pKa</h3>
    <div class="num" style="font-size:24px">${pka.point!=null?pka.point.toFixed(3):'—'}</div>
    <div class="ci">60% CI: <span class="num">${ci(pka.ci_60)}</span></div>
    <div class="ci">90% CI: <span class="num">${ci(pka.ci_90)}</span></div>
    <div class="ci">σ=${pka.sigma??'—'} · tier ${pka.tier||'—'} · src ${pka.source||'—'}</div>
    <div class="ci">max Tanimoto: ${pka.max_tanimoto_to_training??'—'} · OOD=${pka.ood_flag}</div>
    ${pka.point_v15_original!=null?`<div class="delta">v15 (no struct): ${pka.point_v15_original.toFixed(3)}</div>`:''}
    ${refinementTrace(pka.structural_refinement)}
  </div>`;
  const stk = bio.stacker || {};
  const stackerTrace = stk.applied
    ? `<div class="delta">stacker: pre=${bio.point_pre_stacker?.toFixed?.(3) ?? '—'} · ` +
      `D=${stk.components?.direct_head_pred} A=${stk.components?.analog_pred} ` +
      `L=${stk.components?.lion_head_pred} M=${stk.components?.admet_head_pred} · ` +
      `LOO MAE ${stk.expected_loo_mae} (vs baseline ${stk.baseline_loo_mae})</div>`
    : (stk.reason ? `<div class="delta">stacker: skipped (${stk.reason})</div>` : '');
  html += `<div>
    <h3>log<sub>10</sub> flux total <span class="ci" style="font-weight:normal">(linear flux in parens)</span></h3>
    <div class="num" style="font-size:24px">${bio.point!=null?logWithSci(bio.point, 3):'—'}</div>
    <div class="ci">60% CI: <span class="num">${ciWithSci(bio.ci_60)}</span></div>
    <div class="ci">90% CI: <span class="num">${ciWithSci(bio.ci_90)}</span></div>
    <div class="ci">σ=${bio.sigma??'—'} · tier ${bio.tier||'—'} · α=${bio.alpha_used}</div>
    <div class="ci">max Tanimoto: ${bio.max_tanimoto!=null?bio.max_tanimoto.toFixed(3):'—'} · LION_real=${bio.block_B_real}</div>
    ${bio.point_v15_original!=null?`<div class="delta">v15 (no struct): ${logWithSci(bio.point_v15_original, 3)}</div>`:''}
    ${refinementTrace(bio.structural_refinement)}
    ${stackerTrace}
  </div>`;
  html += `</div>`; // end grid
  const organ = r.organ_delivery || {};
  if (organ.target_organ) {
    const parts = organ.partition_pct || {};
    const keys = Object.keys(parts).sort((a,b) => parts[b] - parts[a]);
    const partRows = keys.map(k => {
      const lv = (organ.log10_flux_by_organ||{})[k];
      const lvCell = (lv == null) ? '—' : logWithSci(lv, 3);
      return `<tr><td>${k}</td><td class="num">${parts[k]}%</td>` +
             `<td class="num">${lvCell}</td></tr>`;
    }).join('');
    html += `<div style="margin-top:12px">
      <h3>Predicted target organ: ${organ.target_organ}</h3>
      <div class="ci">${organ.method} · n=${organ.n_neighbors_used}</div>
      <table style="max-width:560px"><thead><tr><th>organ</th><th>partition</th><th>log10 flux (linear)</th></tr></thead>
      <tbody>${partRows}</tbody></table></div>`;
  }
  html += `<div class="grid" style="margin-top:12px">`;
  html += `<div><h3>Nearest pKa neighbors</h3>
    <table><thead><tr><th>IAJD</th><th>Tan</th><th>family</th><th>pKa</th></tr></thead>
    <tbody>${(nb.pka||[]).map(n => neighborRow(n, 'pKa', false)).join('')}</tbody></table></div>`;
  html += `<div><h3>Nearest bioactivity neighbors</h3>
    <table><thead><tr><th>IAJD</th><th>Tan</th><th>family</th><th>log10 flux (linear)</th></tr></thead>
    <tbody>${(nb.bioact||[]).map(n => neighborRow(n, 'log10_flux_total', true)).join('')}</tbody></table></div>`;
  html += `</div>`;
  if (r.warnings && r.warnings.length) {
    html += `<div class="warn"><strong>Warnings:</strong> ${r.warnings.join('; ')}</div>`;
  }
  html += `</div>`; // end card
  return html;
}

function renderBatch(j) {
  if (j.error) return `<div class="err">${j.error}</div>`;
  let html = `<div class="card"><strong>${j.n_inputs} structure(s) parsed</strong> ` +
             `from source <code>${j.source}</code>.</div>`;
  (j.results||[]).forEach((r, i) => { html += renderOne(r, i); });
  (j.errors||[]).forEach(e => {
    html += `<div class="err"><strong>${e.label||'?'}:</strong> ${e.error}</div>`;
  });
  return html;
}

document.getElementById('single_form').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const go = $('#go'); go.disabled = true;
  $('#status').textContent = 'Predicting…';
  $('#out').innerHTML = '';
  const body = {
    smiles: $('#smiles').value.trim(),
    family: $('#family').value || null,
    neighbors: parseInt($('#neighbors').value, 10),
  };
  try {
    const t0 = performance.now();
    const res = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    const j = await res.json();
    const ms = (performance.now() - t0).toFixed(0);
    $('#status').textContent = `Done in ${ms} ms`;
    $('#out').innerHTML = renderOne(j, 0);
  } catch (exc) {
    $('#status').textContent = '';
    $('#out').innerHTML = `<div class="err">Request failed: ${exc}</div>`;
  } finally { go.disabled = false; }
});

document.getElementById('batch_form').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const go = $('#go_b'); go.disabled = true;
  $('#status').textContent = 'Parsing + predicting batch…';
  $('#out').innerHTML = '';
  const fd = new FormData();
  const f = $('#upfile').files[0];
  if (f) fd.append('upfile', f);
  fd.append('smiles_text', $('#smiles_text').value || '');
  fd.append('family', $('#family_b').value || '');
  fd.append('neighbors', $('#neighbors_b').value || '5');
  try {
    const t0 = performance.now();
    const res = await fetch('/predict_batch', { method: 'POST', body: fd });
    const j = await res.json();
    const ms = (performance.now() - t0).toFixed(0);
    $('#status').textContent = `Done in ${ms} ms · n=${j.n_inputs||0}`;
    $('#out').innerHTML = renderBatch(j);
  } catch (exc) {
    $('#status').textContent = '';
    $('#out').innerHTML = `<div class="err">Request failed: ${exc}</div>`;
  } finally { go.disabled = false; }
});
</script>
</body>
</html>"""


def _family_options() -> str:
    opts = ['<option value="">(auto)</option>']
    for fam in sorted(ALLOWED_FAMILIES):
        sel = " selected" if fam == "PE-Tris" else ""
        opts.append(f'<option value="{fam}"{sel}>{fam}</option>')
    return "\n".join(opts)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter logs
        sys.stderr.write(f"[{self.address_string()}] {fmt % args}\n")

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _wait_bundle(self) -> None:
        if not _BUNDLE_READY.is_set():
            _BUNDLE_READY.wait(timeout=120)

    def do_GET(self):  # noqa: N802
        if self.path == "/" or self.path.startswith("/?"):
            html = INDEX_HTML.replace("__FAMILIES__", _family_options())
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/health":
            self._send_json(200, {
                "ok": True,
                "bundle_ready": _BUNDLE_READY.is_set(),
                "bundle_error": _BUNDLE_ERROR,
            })
            return
        self._send_json(404, {"error": f"Not found: {self.path}"})

    def do_POST(self):  # noqa: N802
        if self.path == "/predict":
            return self._handle_predict_single()
        if self.path == "/predict_batch":
            return self._handle_predict_batch()
        self._send_json(404, {"error": f"Not found: {self.path}"})

    def _handle_predict_single(self):
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except Exception as exc:  # noqa: BLE001
            self._send_json(400, {"error": f"bad JSON: {exc}"})
            return
        smiles = (payload.get("smiles") or "").strip()
        if not smiles:
            self._send_json(400, {"error": "missing 'smiles'"})
            return
        family = payload.get("family") or None
        if family and family not in ALLOWED_FAMILIES:
            self._send_json(400, {"error": f"unknown family: {family}"})
            return
        try:
            k = int(payload.get("neighbors", 5))
        except Exception:  # noqa: BLE001
            k = 5
        k = max(1, min(20, k))
        try:
            self._wait_bundle()
            r = predict(smiles, family=family, neighbors=k)
            self._send_json(200, r)
        except Exception:  # noqa: BLE001
            self._send_json(500, {
                "error": "prediction failed",
                "traceback": traceback.format_exc(),
            })

    def _handle_predict_batch(self):
        ctype = self.headers.get("Content-Type", "")
        try:
            if ctype.startswith("multipart/form-data"):
                content, filename, family, neighbors = self._read_multipart()
            else:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                content = payload.get("smiles_text") or payload.get("content") or ""
                filename = payload.get("filename")
                family = payload.get("family") or None
                neighbors = int(payload.get("neighbors", 5))
        except Exception as exc:  # noqa: BLE001
            self._send_json(400, {"error": f"bad request: {exc}"})
            return
        if not content:
            self._send_json(400, {"error": "no input provided (upload a file or paste SMILES)"})
            return
        if family and family not in ALLOWED_FAMILIES:
            self._send_json(400, {"error": f"unknown family: {family}"})
            return
        neighbors = max(1, min(20, int(neighbors or 5)))
        try:
            self._wait_bundle()
            r = predict_batch(
                content=content, filename=filename,
                family=family, neighbors=neighbors,
            )
            self._send_json(200, r)
        except Exception:  # noqa: BLE001
            self._send_json(500, {
                "error": "batch prediction failed",
                "traceback": traceback.format_exc(),
            })

    def _read_multipart(self):
        """Parse a multipart/form-data POST. Returns
            (content_bytes_or_str, filename, family, neighbors)
        Prefers the uploaded file; falls back to the smiles_text field if no
        file was attached.
        """
        ctype = self.headers.get("Content-Type", "")
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        env = {"REQUEST_METHOD": "POST", "CONTENT_TYPE": ctype, "CONTENT_LENGTH": str(length)}
        fs = cgi.FieldStorage(
            fp=io.BytesIO(raw), headers=self.headers, environ=env,
            keep_blank_values=True,
        )
        family = (fs.getfirst("family") or fs.getfirst("family_b") or "").strip() or None
        try:
            neighbors = int(fs.getfirst("neighbors") or fs.getfirst("neighbors_b") or 5)
        except Exception:  # noqa: BLE001
            neighbors = 5
        # Prefer file upload over pasted text
        file_field = fs["upfile"] if "upfile" in fs else None
        if file_field is not None and getattr(file_field, "filename", None):
            content = file_field.file.read()
            filename = file_field.filename
            return content, filename, family, neighbors
        # Fall back to pasted SMILES
        text = fs.getfirst("smiles_text") or ""
        return text, None, family, neighbors


def main() -> int:
    ap = argparse.ArgumentParser(description="STRIDE localhost HTTP server.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    print(f"Warming bundle in background...", file=sys.stderr)
    threading.Thread(target=_warm, daemon=True).start()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving on http://{args.host}:{args.port}  (Ctrl+C to stop)", file=sys.stderr)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", file=sys.stderr)
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
