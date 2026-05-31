# PROTOCOL — From Expensive Simulations to Optima & Next Candidates

**A generalizable, mechanism-first surrogate-optimization protocol.**
Use it for any campaign where each candidate is **expensive to evaluate** (a
simulation, a synthesis+assay, an experiment), you can compute a **handful of
interpretable descriptors** per candidate, you have a measured **objective for a
small number** of candidates (tens–low hundreds), and you want to **find the
optimum in descriptor space and propose the next candidate(s)** — robustly, with
honest uncertainty.

Motivating example (this repo): IAJD physics descriptors {apparent pKa, spontaneous
curvature c₀, packing parameter CPP, H_II propensity} → mRNA-transfection flux. But
nothing below is IAJD-specific; read `X` = your descriptors, `y` = your objective,
`class` = your family/stratum.

---

## 0. Why this protocol exists (the failure mode it avoids)

The naive move — "compute descriptors, dump into XGBoost/a neural net, read off
importances" — fails predictably on small-n design data:
- It **overfits** and effectively learns *"the traits of the single best-measured
  point = best possible objective."* If that point's measurement is wrong, the model
  is moot.
- It gives **no calibrated uncertainty**, so you can't tell a real optimum from noise.
- Tree/step surfaces **can't resolve a smooth peak**, and "feature importance" is not
  "where is the optimum."
- It tempts you to **extrapolate the extreme** ("push X higher") when the real lever
  is an **optimum** — overshooting it *lowers* the objective.

This protocol replaces that with: **mechanism as the prior → smooth interpretable
response models to locate optima → uncertainty-aware Bayesian optimization to propose
the next candidate → full evaluation to confirm.**

---

## 1. Six principles (read before touching code)

1. **Mechanism is the prior.** Theory/physics tells you *a priori* roughly where optima
   sit and which descriptors should matter. Data *refines* this; it does not replace it.
2. **Levers are optima, not monotones.** Most real design variables have a sweet spot.
   "Push to the extreme" usually overshoots. **Find the peak, not the edge.**
3. **Trends beat anchors.** A fitted trend across many points survives one bad
   measurement; a model anchored to the single best point does not. Prefer methods that
   fit the whole response over ones that memorize the maximum.
4. **Small n demands uncertainty.** With tens of points any "optimum" can be noise. Use
   methods that report **calibrated error bars** so you know if a peak is real.
5. **No black boxes for optima.** High-capacity ML (XGBoost/NN) is the wrong tool at this
   n — overfit, jagged, uncertainty-free.
6. **Honesty / no-proxy.** Every descriptor is a real computed value or `NaN`+audit.
   Every objective value carries a source + error bar. **Surrogate predictions are
   labelled as predictions with uncertainty, never as measurements.**

---

## 2. The method stack

### Step 0 — Data hygiene (do not skip)
- **Stratify by class/family**, or include `class` as a covariate, so a descriptor's
  effect is isolated from class confounds (different architectures move many descriptors
  at once).
- **Replicates + error bars on `y`.** The objective is usually the noisiest thing in the
  pipeline; the surrogate needs the noise level, and a single mismeasured point is your
  worst enemy. If you have no replicates, treat *every* "optimum" as provisional.
- **Standardize `X`**; record ranges and **coverage** (where you have data vs not) —
  flag extrapolation regions explicitly.

### Step 1 — UNDERSTAND: where is each optimum?  (response curves → GAM)
- **Per-variable response curves:** within each stratum, bin or LOESS-smooth `y` vs each
  `Xᵢ`. The peak is visible directly. (This is literally how the ionizable-lipid pKa
  optimum was found — Jayaraman 2012 plotted potency vs pKa and saw ~6.2–6.5.)
- **Generalized Additive Model (GAM):** `y ~ s(X1) + s(X2) + … + class`, with smooth
  additive terms. Read each variable's curve, its **peak**, its **confidence band**, and
  its effective degrees of freedom (is it genuinely non-monotonic, or just flat/noisy?).
- **Sanity vs mechanism:** does each data-peak land where theory predicts? **If not,
  distrust the data-peak** (measurement error / confound), not the mechanism.
- Tools: `pyGAM` (Python), `mgcv` (R). Always LOO/k-fold cross-validate.

### Step 2 — PROPOSE: what to evaluate next?  (Gaussian Process + Bayesian optimization)
- **Gaussian-Process surrogate:** `y ~ GP(mean, kernel)` over `X`, with:
  - an **explicit noise term** (from replicates) so the GP doesn't interpolate noise;
  - a **Matérn (ν=2.5) or RBF kernel** with **ARD length-scales** (per-variable; a long
    length-scale ⇒ that variable barely matters — a free relevance read-out);
  - `class` as a covariate or a **multi-task / hierarchical GP** for stratification.
  GPs are *designed* for small n and give a **calibrated posterior mean ± variance.**
- **Validate the surrogate before trusting it:** LOO-CV; check **calibration** (do the
  error bars cover the truth at the nominal rate?). A miscalibrated GP is worse than none.
- **Bayesian optimization:** define a **candidate pool** (synthetically/physically
  plausible moves on the current best), and score each by an **acquisition function**:
  - **Expected Improvement (EI)** — default; balances *exploit* (near predicted optimum)
    vs *explore* (high uncertainty).
  - **UCB** (`μ + κσ`) — tunable explore/exploit knob `κ`.
  - **Constrained EI** — when feasibility limits apply (toxicity, stability,
    synthesizability): multiply EI by the GP-predicted probability of feasibility.
  - **q-EI / batch BO** — when you can evaluate several candidates per round.
- **Output:** the top-k candidates by acquisition = *what to compute/make next.*
- Tools: `scikit-learn` GaussianProcessRegressor (simple), **`GPyTorch`+`BoTorch`** or
  **`Ax`** or `scikit-optimize` (production BO).

### Step 3 — CONFIRM & ITERATE (active learning)
- Run the **full expensive evaluation** (the real simulation/assay) on the proposed
  candidates — *no surrogate shortcut.*
- Add results back; **refit GAM + GP; repeat.** Each round the surrogate sharpens around
  the optimum and the uncertainty shrinks where it matters. This active-learning loop is
  the efficient way to spend a fixed simulation budget.

---

## 3. The hierarchy — never invert it

```
   mechanism (prior)  →  GAM / response curves (does the data agree?)
        →  GP + Bayesian optimization (quantify + propose next)
             →  full physics/experimental evaluation (confirm)
```
The surrogate **refines** the mechanism; it never overrides it. **A model optimum that
contradicts the mechanism is a red flag for measurement error, not a discovery.**

---

## 4. Why GAM + GP/BO beat XGBoost/NN here

| | XGBoost / NN | GAM | **GP + BO** |
|---|---|---|---|
| Small n (tens) | overfits | OK | **designed for it** |
| Uncertainty | none | per-curve CI | **calibrated posterior** |
| Locates a smooth optimum | jagged steps | smooth peak | **smooth peak** |
| Proposes the *next* candidate | ad-hoc search | — | **acquisition function** |
| Robust to one bad anchor point | poor (memorizes max) | good (fits trend) | **good (+ noise term)** |
| Interpretability | black box (needs SHAP) | per-variable curves | **ARD relevance + slices** |

XGBoost/NN are the right tools when n is large and you only need predictions. For
**small-n, find-the-optimum, propose-the-next-experiment**, they are the wrong tool.

---

## 5. Honest caveats (state these in every report)

- **Small n is small n.** None of these manufacture signal; they *quantify* what's there.
  If the data can't resolve an optimum, say so — don't fabricate one.
- **Garbage objective → garbage optimum.** A noisy/biased `y` poisons every method;
  replicates + error bars are non-negotiable.
- **Correlation ≠ causation.** A descriptor can track a hidden confound (e.g. it just
  correlates with chain length, the real driver). Only the **mechanism** disambiguates;
  prefer descriptors the theory says are causal.
- **The right descriptors must be measured.** If the true lever isn't in `X`, no model
  finds it. Coverage and mechanism guide what to add.
- **Extrapolation is dangerous.** GP uncertainty balloons outside the data; trust
  proposals *near* the data and treat far ones as exploration to *confirm*, not to ship.

---

## 6. Minimal worked template (generalizable pseudocode)

```python
# X: (n, d) descriptors  | y: (n,) objective + yerr (n,) | cls: (n,) class/stratum
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler

# --- Step 0: hygiene ---
Xs = StandardScaler().fit_transform(X)            # standardize
# (carry yerr from replicates; one-hot or target-encode cls)

# --- Step 1: GAM response curves + per-variable optima ---
from pygam import LinearGAM, s, f
gam = LinearGAM(s(0)+s(1)+s(2)+s(3)+f(d)).gridsearch(Xs_with_cls, y)
for i in range(d):
    XX, conf = gam.partial_dependence(term=i, width=0.95)   # curve + CI
    # peak of XX = data-optimum of variable i; compare to mechanistic prior

# --- Step 2: GP surrogate + Bayesian optimization ---
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
kernel = C()*Matern(length_scale=np.ones(d), nu=2.5) + WhiteKernel()   # ARD + noise
gp = GPR(kernel=kernel, alpha=yerr**2, normalize_y=True,
         n_restarts_optimizer=20).fit(Xs, y)
# LOO-CV + calibration check BEFORE trusting gp ...

from scipy.stats import norm
def expected_improvement(Xc, gp, y_best, xi=0.01):
    mu, sd = gp.predict(Xc, return_std=True); sd = np.maximum(sd, 1e-9)
    imp = mu - y_best - xi; z = imp/sd
    return imp*norm.cdf(z) + sd*norm.pdf(z)        # maximize

cand = enumerate_plausible_moves(best)             # your candidate pool, in-class
cand = recompute_descriptors(cand)                 # honor coupling: recompute ALL X
ei = expected_improvement(scaler.transform(cand_X), gp, y.max())
next_to_make = cand[np.argsort(-ei)[:k]]           # top-k by acquisition

# --- Step 3: full evaluation of next_to_make, append, refit, repeat ---
```
(For production BO, swap the hand-rolled EI for **BoTorch/Ax**: analytic/MC acquisition,
batch q-EI, constraints, and multi-task GPs out of the box.)

---

## 7. No-proxy / reproducibility contract

- Every `X` is a real computed value or `NaN`+audit; every `y` has a source + error bar.
- Surrogate outputs are always reported as **prediction ± uncertainty**, never as a
  measured value.
- Seed all stochastic steps; log the fitted model, the CV/calibration result, the
  acquisition choice, and the candidate pool, so any proposal is reproducible and
  auditable.
- A proposed candidate is a **hypothesis to confirm by full evaluation**, not a result.

---

## 8. One-paragraph summary

Don't dump descriptors into XGBoost. **Stratify by class; use a GAM/response curves to
*locate* each variable's optimum (and check it against the mechanism); use a Gaussian
Process + Bayesian-optimization acquisition function to *propose* the next candidate with
calibrated uncertainty; confirm by full evaluation; iterate.** Keep the mechanism as the
prior and the surrogate as its refinement — and remember the levers are optima, so you
chase the peak, never the extreme.
