# Research logbook

## Project

Extending Morimura, Osogami & Idé (NeurIPS 2013), *Solving inverse problem of Markov chain
with partial observations*, to detect and filter out observations corrupted by external
re-routing influence (e.g. sat-navs reacting to congestion), so that the recovered chain
reflects drivers' intrinsic preferences.

## Files

| File | Purpose |
|------|---------|
| `NIPS-2013-...-Paper.pdf` | The reference paper. |
| `morimura.py` | Reproduction of the synthetic experiment in §6.1 of the paper. |
| `congestion_filter.py` | Extension. Detect externally-influenced observations and use the detection to clean an intrinsic-chain fit. |
| `morimura_fig.png` | Output of `morimura.py` — RMAE curves vs. number of observation states. |
| `congestion_filter_fig.png` | Output of `congestion_filter.py` — detection ROC, score histogram, recovery RMAE. |

Both scripts are self-contained; run with `python morimura.py` or `python congestion_filter.py`.

---

## Phase 1 — Reproducing the inverse-MC method (`morimura.py`)

### What the paper poses
A Markov chain with restart on a finite state space $\mathcal{X}$ has parameters $(p_I, p_T, \beta)$:
$p_I$ is the initial-state distribution, $p_T(x'|x)$ is the per-step transition, and $\beta$ is
the continuation probability per step. The paper considers the *inverse* problem: given
observations of visit-frequency $f(x)$ and hitting-rate $g(x, x')$ at a small subset
$X_o \subset \mathcal{X}$ of states, recover the parameters and predict $f$ at the
unobserved states. The approach is to fit a softmax-parametric chain by minimising the
regularised objective (Eq. 7 in the paper):

$$L(\theta) = \gamma L_d(\theta) + (1-\gamma) L_h(\theta) + \lambda R(\theta),$$

where $L_d$ matches log-ratios of stationary probabilities at $X_o$ (Eq. 8) and $L_h$ matches
log hitting probabilities on $X_o \times X_o$ (Eq. 9). Closed-form gradients of
$\log \pi_\theta$ (Eq. 12) and $\log h_\theta$ (Eq. 15) follow from differentiating the
balance equation and the hitting-prob recursion.

### What the script does

`morimura.py` runs the synthetic experiment of §6.1 end-to-end:

1. **Random graph** (`random_graph`): a strongly-connected directed graph on $n$ nodes (mean
   out-degree $\approx 3$, plus a Hamiltonian cycle to guarantee ergodicity).
2. **Synthetic ground truth** (`make_truth`): clean softmax model for $p_I$ and each row of
   $p_T$ with weights drawn from $\mathcal{N}(0,1)$, then mixed 70/30 with Dirichlet $(0.3)$
   noise per the paper. This pushes the truth slightly outside the parametric family —
   robustness test.
3. **Inverter class** (`Inverter`):
   - `forward(theta)` builds $p_I, p_T$ from the parameters.
   - `grad_log_pi` implements Eq. 12 by computing
     $V[:, k] = (\partial P^\top / \partial \theta_k)\,\pi$
     for all $k$ at once (sparse outer-product blocks for the $\nu$ and per-origin $\omega$
     pieces) and then doing one batched solve through $Q = I - P^\top + \pi \mathbf{1}^\top$.
   - `grad_log_h` analogously implements Eq. 15.
   - `loss_grad` assembles $\gamma L_d + (1-\gamma) L_h + \lambda R$ and its analytic gradient,
     vectorised so the antisymmetric difference matrix collapses the $L_d$ gradient to
     $2\,(D \mathbf 1)\cdot J[X_o]$.
   - `fit` calls `scipy.optimize.minimize(..., method='L-BFGS-B')` on the loss plus gradient.
4. **Baseline** (`nwkr_with_cv`): Nadaraya–Watson kernel regression on undirected hop
   distances, bandwidth tuned by leave-one-out CV. This is the comparator the paper uses.
5. **Sweep** (`run`): for each of seven values of $|X_o|$ and several random instances,
   fits all three methods (proposed-with-$g$, proposed-no-$g$, NWKR) and records the
   relative MAE on the held-out states.

### Choices and deviations from the paper

| Choice | Reason |
|--------|--------|
| Local-only parameters (no global features $\phi_I, \phi_T, \psi$). | The paper allows either local or global; local-only keeps the parameter space transparent for the synthetic test. |
| L-BFGS-B optimiser, not the natural-gradient descent of §4.1. | Both consume the same loss/gradient; L-BFGS-B is robust off-the-shelf and we're not chasing per-iteration speed. |
| $\beta$ held at its true value during the inverse fit. | Avoids a known identifiability issue when the only signal is stationary statistics. |
| $n = 50$ instead of paper's 100. | Trades a small loss in absolute RMAE for a shorter sweep; structure of the curves is the same. |
| Prediction rescaling: $\hat f(x) = (\sum_{X_o} f) \cdot \hat\pi(x) / (\sum_{X_o} \hat\pi)$. | The paper writes $\hat c \hat\pi$ with $\hat c = \overline{f}_{X_o}$; this only matches the $f$ scale if $\hat\pi$ averages to 1 over $X_o$, which it doesn't. The given form is the equivalent scale-free version. |

### Result

`morimura_fig.png`: three RMAE-vs-$|X_o|$ curves matching the qualitative pattern of Fig. 2A:

- Proposed (uses both $f$ and $g$) is best, dropping to RMAE ≈ 0.07 at $|X_o|=45$.
- Proposed (no $g$) is intermediate, plateaus around 0.25–0.30.
- NWKR baseline is worst, with high variance.

Total runtime: ≈ 20 s on this laptop.

---

## Phase 2 — Filtering externally-influenced observations (`congestion_filter.py`)

### Motivation

In a real traffic system, drivers' route choices stop being purely intrinsic during
congestion: many switch on a sat-nav, which re-routes them around busy links. Counts logged
in those moments reflect the sat-nav recommendations rather than driver preferences. To
recover an *intrinsic* Markov chain we want to identify and discount such observations.

### Setup

A binary regime $r_t \in \{0,1\}$ indexes each time-snapshot:

- $r_t = 0$ — uncongested. All drivers follow the intrinsic chain $M^{\text{intr}}$.
  $f_t(x) \sim \text{Poisson}(K \pi^{\text{intr}}(x))$.
- $r_t = 1$ — congested. A fraction $\rho$ of drivers use a sat-nav and follow an
  *influenced* chain $M^{\text{infl}}$; the rest stay intrinsic.
  $f_t(x) \sim \text{Poisson}\bigl(K[(1-\rho)\pi^{\text{intr}}(x) + \rho\,\pi^{\text{infl}}(x)]\bigr)$.

`generate_two_regime` builds the two chains and samples $T$ snapshots accordingly.

The influence model is **per-link avoidance**:

$$p_T^{\text{infl}}(x' \mid x) \;\propto\; p_T^{\text{intr}}(x' \mid x) \, \exp\bigl(-\alpha\, \mathrm{cong}(x')\bigr),$$

with $\mathrm{cong}(x) = \pi^{\text{intr}}(x) / \max_y \pi^{\text{intr}}(y) \in [0, 1]$ — the
busiest link is fully congested ($\mathrm{cong}=1$), others scale linearly. The strength
$\alpha$ controls how aggressively sat-nav users avoid busy nodes.

The user supplies a labelled subset of size $T_{\text{label}}$ drawn from the truly
uncongested snapshots; the rest are unlabelled.

### Detection score

For each unlabelled snapshot, compute the KL divergence between the empirical
distribution at $X_o$ and the empirical intrinsic distribution at $X_o$:

$$\text{score}_t \;=\; \mathrm{KL}\bigl(\hat p_t \,\big\|\, \hat p^{\text{intr}}\bigr),
\qquad \hat p_t(x) = \frac{f_t(x)}{\sum_{y \in X_o} f_t(y)}, \qquad
\hat p^{\text{intr}}(x) = \frac{1}{T_{\text{label}}} \sum_{s \in \text{label}} \frac{f_s(x)}{\sum_{y} f_s(y)}.$$

A higher score means the snapshot looks *unlike* the intrinsic regime.

> **Note on iteration.** The first version of the score was the Poisson log-likelihood
> $\sum_x [f_t(x)\log r_x - r_x]$ under intrinsic rate $r$. AUCs came out either ≈ 0 or ≈ 1
> at random across trials. The reason: the dominant term $f_t \cdot \log r$ scales linearly
> with the *total count at $X_o$*, which differs between regimes for nuisance reasons (the
> intrinsic and influenced distributions assign different mass to the observed subset), so
> the score tracked a confound rather than the actual chain difference. Switching to KL on
> normalised distributions fixes this — it directly compares shapes.

### Filter threshold

Rather than a fixed keep-fraction, the script picks the threshold from the labelled
(known-intrinsic) scores:

$$\tau \;=\; \text{quantile}_{1 - \text{fpr\_target}}\bigl(\{\text{score}_s : s \in \text{label}\}\bigr).$$

Unlabelled snapshots with score $\leq \tau$ are kept. Under the null (the snapshot is
intrinsic) the false-positive rate is at most `fpr_target` (default 0.05).

### Three intrinsic-chain recovery strategies

1. **Naive** — aggregate counts across all $T$ snapshots, fit `Inverter` with $\gamma=1$.
   Biased because the aggregate is contaminated by influenced snapshots.
2. **Oracle** — aggregate only true-$r_t=0$ snapshots. Cheats by using the regime label
   and gives the best achievable performance for this data split.
3. **Filtered** — aggregate the labelled set plus the score-passing unlabelled snapshots,
   then fit. Uses no oracle information beyond the labelled set itself.

Each method passes its aggregate vector to the `morimura.Inverter` (with $\gamma=1$, since
hitting rates aren't being modelled here). The recovered $\hat\pi$ is compared to the true
$\pi^{\text{intr}}$ via relative MAE on the unobserved states.

### Result

`congestion_filter_fig.png` — three panels:

1. **ROC for one trial.** AUC = 1.000 at default settings.
2. **Score histograms by true regime.** Labelled and unlabelled-intrinsic snapshots cluster
   tightly near 0; influenced snapshots cluster around KL ≈ 0.08. The chosen threshold
   (vertical dashed line) cleanly separates them.
3. **Recovery RMAE bar chart.** Across 5 trials at default settings:

| Method | RMAE (mean ± std) |
|--------|-------------------|
| naive  | 0.40 ± 0.08 |
| filter | **0.35 ± 0.08** |
| oracle | 0.35 ± 0.08 |

The filter recovers oracle performance using only the labelled set as supervision; naive
fitting (no filtering) is ≈ 14 % worse.

Detection AUC stays near 1.0 even when avoidance strength is reduced from $\alpha = 3$ to
$\alpha = 0.7$ (a much harder regime). The total flow $K = 20{,}000$ gives high SNR per
snapshot, which is part of why detection is easy here. Reducing $K$ would test the small-data
regime.

Total runtime: ≈ 1 s.

---

## Open threads

- **Latent regime.** The current setup assumes we know which snapshots came from the
  uncongested period. A more realistic version would jointly infer $r_t$ and the chain
  parameters via EM, given only the count time-series and an observable congestion proxy.
- **Per-state filtering.** Right now the unit of filtering is a whole snapshot. One could
  instead score each (state, time) pair: a state with a locally large anomaly while its
  neighbours look fine would pinpoint sat-nav-driven re-routing at that link only.
- **Soft re-weighting.** Instead of a hard keep/drop, weight each snapshot by
  $1 - p(\text{influenced} \mid f_t)$ in the aggregation. Strictly better when data is
  scarce.
- **Influence model.** Per-link avoidance with a constant $\alpha$ is a placeholder.
  Alternatives worth trying: sat-nav users following Dijkstra-based shortest paths, an
  $\alpha$ that depends on observed congestion level, or an unrestricted second softmax
  chain trained jointly.
- **Real data.** All experiments so far are synthetic. Hand the system a real flow trace
  with a congestion proxy (clock time, weather, an external traffic API) and check whether
  the same regime separation appears.

## Variables and how they shape the results

The two scripts together expose a fairly large knob-space. The tables below list every
adjustable parameter, where it lives, its default, what it physically represents, and the
direction the main outputs are expected to move when the knob is turned up. Use them to
plan experiments — most are independent of each other to first order, but a few couple
non-trivially (flagged in the notes).

### Graph and chain structure

These parameters control the underlying Markov chain you're trying to learn. They appear
in `morimura.make_truth` and (via that function) in `congestion_filter.generate_two_regime`.

| Variable | Where | Default | What it controls | ↑ effect |
|---|---|---|---|---|
| `n` | both | 50 | Number of states in the chain. | Larger problem. RMAE generally rises at fixed $\lvert X_o\rvert/n$ because there are more states to extrapolate to. Runtime grows like $n^3$ in the inverter (linear solves over $n \times n$ matrices). |
| `mean_out_degree` | both | 3 | Out-edges per state in the random graph. | Denser graph → faster mixing → less informative $f$ pattern (stationary flattens) → harder recovery. Also increases the edge-parameter count, slowing fits. |
| `beta` | both | 0.9 | Per-step continuation probability of the chain with restart. | Closer to 1 means longer trajectories before restart, more weight on $p_T$ over $p_I$ in $\pi$. Recovery is generally easier at higher $\beta$ for the $L_d$ objective (the chain's structure shows through more clearly), but mixing time grows. |
| `mix` | both | 0.7 | Mixing weight on the clean softmax model when generating ground truth (the rest is Dirichlet noise). | Closer to 1 means the truth lies inside the parametric family; recovery is easier. Lower values stress-test the model. |
| `dir_alpha` | both | 0.3 | Dirichlet concentration of the noise term. Values < 1 are spiky (sparse-like). | Lower = more extreme noise spikes; the truth deviates further from the parametric model and RMAE rises. |

### Inverse-MC solver (Phase 1)

`Inverter.__init__` and `Inverter.fit`.

| Variable | Default | What it controls | ↑ effect |
|---|---|---|---|
| `gamma` | 0.1 in solver, 1.0 in the filter pipeline | Weight between the stationary-prob loss $L_d$ and the hitting-prob loss $L_h$ in $L = \gamma L_d + (1-\gamma) L_h$. | At $\gamma=1$ only $L_d$ is used (no hitting-rate information). At $\gamma=0$ only $L_h$ is used. Intermediate values let both kinds of observation contribute. In Phase 1, $\gamma=0.1$ gave the best RMAE; in Phase 2 we use $\gamma=1$ because hitting rates aren't being modelled. |
| `lam` | 1e-3 | Ridge regularisation strength $\lambda$ on $\frac12\lVert\theta\rVert^2$. | Too small ⇒ overfit to the few observed states, especially with small $\lvert X_o\rvert$. Too large ⇒ everything is pulled toward the uniform softmax (zero $\theta$), recovery degrades. Worth a small grid-search if you change `K` or `n` substantially. |
| `maxiter` | 200–300 | L-BFGS-B iteration cap. | Typically converges well below the cap; raise it only if the printed `iters` is hitting the cap. |
| `x0` | zero vector | L-BFGS initialisation. | Zero corresponds to uniform softmax — neutral. Random init occasionally finds better minima for the $L_h$ objective (which is non-convex). |

### Experiment driver (Phase 1)

`morimura.run`.

| Variable | Default | What it controls | ↑ effect |
|---|---|---|---|
| `K` | 1000 | Scale that converts $\pi_{\text{true}}(x)$ into a "count" $f(x)$ in the synthetic test. Acts only as a normaliser for the RMAE denominator $\max(f, 1)$. | Higher = predictions and ground truth both live further from the floor, so the denominator is informative; very small $K$ makes the RMAE numerator dominate (the max-1 floor saturates). |
| `obs_fracs` | (0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90) | Fractions of $\lvert X\rvert$ to use as $\lvert X_o\rvert$. | This is the curve's x-axis. More observation states = better recovery, by definition; the *shape* of the drop-off is what's interesting. |
| `n_trials` | 3 | Random repetitions per $\lvert X_o\rvert$. | More trials = tighter error bars. Runtime scales linearly. |
| `seed` | 42 | RNG seed for reproducibility. | No effect on expected results, only on the realised noise. |

### Two-regime synthesis (Phase 2)

`generate_two_regime` and `run_trial` in `congestion_filter.py`.

| Variable | Default | What it controls | ↑ effect |
|---|---|---|---|
| `alpha` | 3.0 | Per-link avoidance strength in $p_T^{\text{infl}}(x'\!\mid\!x) \propto p_T^{\text{intr}}(x'\!\mid\!x)\exp(-\alpha\,\text{cong}(x'))$. | The single biggest control on detection difficulty. Higher → influenced chain differs more from intrinsic → detection AUC ↑ and naive-recovery RMAE ↑ (more contamination to remove). At $\alpha \to 0$ the two regimes collapse and detection becomes impossible. |
| `rho_c` | 0.7 | Sat-nav adoption fraction during congestion. The mixture rate inside $r_t=1$ snapshots. | Higher → influenced snapshots look more like pure $\pi^{\text{infl}}$ and less like a mix → easier detection. At $\rho_c=0$ the congested regime is indistinguishable from intrinsic. |
| `K` | 20 000 | Per-snapshot Poisson rate scale. Equivalent to "vehicles observed per snapshot". | Higher → empirical distribution $\hat p_t$ is closer to the true regime distribution → KL noise floor drops → detection AUC ↑. With $K$ low (few hundred counts) the noise floor competes with the regime gap and detection degrades. **This is the main lever for testing the small-data regime.** |
| `T` | 200 | Total number of time-snapshots in the dataset. | More data → tighter empirical estimates everywhere. Detection AUC is largely insensitive once $T$ is in the hundreds; recovery RMAE keeps improving (logarithmically) as $T$ grows. |
| `p_congested` | 0.6 | Marginal probability that any given snapshot is congested. | Higher → more contamination in the naive aggregate → naive RMAE worsens; oracle and filter are unaffected. |
| `T_label` | 40 | Labelled (known-uncongested) snapshots provided as supervision. | Higher → the empirical $\hat p^{\text{intr}}$ is more reliable, the threshold quantile from the labelled set is tighter, and the filter trusts the labelled aggregate more. Falls off quickly past ~30–50 in this regime. |
| `n_obs` | 20 | Number of observation states $\lvert X_o\rvert$. | More observed states → more dimensions in the KL score → easier detection; also tighter inverter fit (better extrapolation in the inverse-MC step). |
| `fpr_target` | 0.05 | Target false-positive rate of the filter against the labelled set. The threshold is set at the $(1-\text{fpr\_target})$ quantile of labelled scores. | Higher → looser threshold → more unlabelled snapshots kept, both intrinsic and influenced (more contamination). Lower → tighter threshold → drops genuine intrinsic snapshots from the kept set (less data, but cleaner). Below ~$1/T_{\text{label}}$ the quantile is dominated by the maximum labelled score and becomes unstable. |
| `seed` | 42 | RNG seed. | Same as Phase 1. Note that AUC can swing between trials when the regime gap is small (low $\alpha$); use ≥5 seeds before drawing conclusions. |

### Hard-coded modelling choices

Not surfaced as parameters, but they materially shape the results. Worth keeping in mind
when interpreting outputs, and worth varying if you want a fuller picture.

| Choice | Where | Plausible alternative |
|---|---|---|
| Congestion vector $\text{cong}(x) = \pi^{\text{intr}}(x) / \max \pi^{\text{intr}}$. | `cong_vector` in `congestion_filter.py`. | Use $\pi^{\text{intr}}/\text{mean}(\pi^{\text{intr}})$ (centred at 1) or an externally supplied per-link capacity ratio. |
| KL divergence as the anomaly score. | `kl_score`. | Total-variation distance, $\chi^2$ statistic, Jensen-Shannon divergence, or a learned log-likelihood-ratio if a model for $\pi^{\text{infl}}$ is fitted. |
| Threshold from labelled quantile (a *parametric* null calibrated empirically). | `run_trial`. | Permutation-based threshold, mixture-EM threshold, or a likelihood-ratio test. |
| Hard keep/drop filtering. | `run_trial`. | Soft re-weighting by $1 - p(\text{influenced} \mid f_t)$ — strictly better when data is scarce. |
| Local-only softmax parameters (no global features). | `Inverter`. | Add per-state feature vectors $\phi_I, \phi_T$ and edge features $\psi$ (paper Eqs. 17–18). Helps when many states share structure. |
| L-BFGS-B optimiser. | `Inverter.fit`. | Natural-gradient descent with the Fisher information matrix (paper §4.1). |
| $\beta$ fixed at its true value. | `Inverter` constructor. | Learn $\tilde\beta = \varsigma^{-1}(\beta)$ jointly. Identifiability is fragile without hitting-rate observations. |

### Headline knob-by-knob expectations

If you want a one-line rule of thumb for what to expect when sweeping each knob, this is
the qualitative summary:

- **Make detection harder:** ↓ `alpha`, ↓ `rho_c`, ↓ `K`, ↓ `n_obs`, ↓ `T_label`.
- **Make recovery harder:** ↑ `n`, ↑ `mean_out_degree`, ↓ `mix`, ↓ `dir_alpha`, ↓ `T`, ↑ `p_congested` (naive only).
- **Filter ≈ oracle (the happy regime):** big regime gap (high `alpha`, `rho_c`), strong signal per snapshot (high `K`), enough labelled data (`T_label` ≳ 30).
- **Filter ≈ naive (the failed regime):** small regime gap, low SNR, sparse labelled set.

The most informative single sweep for understanding the system as a whole is `alpha`
held against `K`: it traces a 2D map from "perfect detection, oracle-matching filter" to
"detection at chance, no filtering possible", with the interesting intermediate band
where the filter starts to soft-fail and choice of threshold matters most.

## Reproducing the figures

```bash
# fresh venv
python3 -m venv .venv
.venv/bin/python -m pip install numpy scipy matplotlib

# Phase 1 — Morimura reproduction, ~20 s
.venv/bin/python morimura.py

# Phase 2 — congestion filter, ~1 s
.venv/bin/python congestion_filter.py
```

Both produce a PNG in the working directory and print per-trial summary lines to stdout.
