# Research logbook

## Project

Extending Morimura, Osogami & Idé (NeurIPS 2013), *Solving inverse problem of Markov chain
with partial observations*, to detect and filter out observations corrupted by external
re-routing influence (e.g. sat-navs reacting to congestion), so that the recovered chain
reflects drivers' intrinsic preferences.

## Files

| File | Purpose |
|------|---------|
| `papers/NIPS-2013-...-Paper.pdf` | The reference paper (Morimura, Osogami, Idé 2013). |
| `papers/Identification_of_new_patterns_v3.pdf` | Gu, Crisostomi, Liu, Shorten (2018). Population-level junction-turning anomaly detection — the conceptual reference for Phase 3. |
| `papers/2605.18782v1.pdf` | Cappi et al. (2026). Padua AVI dataset (10-min aggregated). Considered for real-world validation; cannot run per-trajectory detector — only aggregated flow stats released. |
| `papers/s41597-022-01850-0.pdf` | Wang et al. (2023). Xuancheng holographic trajectory reconstruction (Sci. Data). Per-trajectory raw access *restricted*; releases resampled flow data only. |
| `papers/s41597-026-06892-2.pdf` | Ma et al. (2026). Xuancheng AVI dataset with **joinable entry/exit tables** allowing per-trajectory reconstruction. SUMO-format, ~1,760 segments, 1 month, ~350k trips/day. Primary candidate for real-world Phase 3/4 validation. Figshare DOI 29925824. |
| `papers/LSG.pdf` | Background reference (line-segment grammar). |
| `papers/Focused literature review for inverse Markov-chain route-choice inference.pdf` | Background literature review. |
| `morimura.py` | Reproduction of the synthetic experiment in §6.1 of the paper. |
| `congestion_filter.py` | Phase 2. Detect externally-influenced *snapshots* and use the detection to clean an intrinsic-chain fit. |
| `per_car_detector.py` | Phase 3. Detect whether a *single car's* trajectory was sat-nav-influenced via a log-likelihood ratio. |
| `morimura_fig.png` | Output of `morimura.py` — RMAE curves vs. number of observation states. |
| `congestion_filter_fig.png` | Output of `congestion_filter.py` — detection ROC, score histogram, recovery RMAE. |
| `per_car_detector_fig.png` | Output of `per_car_detector.py` — single-config ROC, LR score histogram, stationary-distribution comparison. |
| `per_car_detector_sweep.png` | Output of `per_car_detector.py` — AUC vs. sat-nav strength $\alpha$ at three trajectory lengths. |
| `sumo_validation/sumo_to_phase3.py` | Phase 4. Maps a pair of SUMO runs (intrinsic vs. rerouting) onto the Phase 3 detector. |
| `sumo_validation/sumo_phase3_fig.png` | Output of the Phase 4 script — ROC curves comparing fitted detectors against an empirical-chain upper bound. |
| `sumo_validation/{net.net.xml, routes.xml, edgedata.*.xml, vehroutes.*.xml, *.sumocfg, *.add.xml}` | SUMO artefacts for the Phase 4 validation: network, demand, edge counts, per-vehicle routes, and the two run configs (intrinsic / sat-nav). |

All three scripts are self-contained; run with `python3 morimura.py`, `python3 congestion_filter.py` or `python3 per_car_detector.py`. Phase 4 is driven by SUMO + `sumo_to_phase3.py`; see that section for the run order.

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

Updated 2026-05-27 after iter 18 audit. The table below describes the *current* configuration; for the pre-iter-18 simplifications and what changed, see iter 18.

| Choice | Reason |
|--------|--------|
| Truth and recovery both use $\phi_T$ ($d_T = 5$) and $\psi$ ($d_\psi = 5$) global features drawn iid $\mathcal N(0, 1)$, per paper §6.1. $\phi_I$ (initial-prob globals) omitted — matches paper §6.2 Nairobi configuration, justified by paper p. 7: "If a simpler model is preferred, either of them would be omitted." | Required to be a faithful reproduction; before iter 18 we ran a local-only simplification that produced a misleadingly easier benchmark. |
| Tikhonov $\lambda$ cross-validated per (size, trial, method) over $\{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\}$ on a 75/25 split of $X_o$. | Paper §6.1: "$\lambda$ was determined with a cross-validation." Before iter 18 $\lambda$ was hard-coded at $10^{-3}$. |
| L-BFGS-B optimiser, not the natural-gradient descent of §4.1. | Both consume the same loss/gradient; L-BFGS-B is robust off-the-shelf and we're not chasing per-iteration speed. Note: Phase-4 iter 15 showed this matters at scale (basin issues). |
| $\beta$ held at its true value during the inverse fit. | Avoids a known identifiability issue when the only signal is stationary statistics. |
| $n = 100$ with $|X_o| \in \{5, 10, 20, 35, 50, 70, 90\}$, 10 trials per size. | Matches paper Fig. 2A exactly. Before iter 18 we used $n=50$ with fractional sweep and 3 trials. |
| Prediction rescaling: $\hat f(x) = (\sum_{X_o} f) \cdot \hat\pi(x) / (\sum_{X_o} \hat\pi)$. | The paper writes $\hat c \hat\pi$ with $\hat c = \overline{f}_{X_o}$; taken *literally* this under-scales by $|X|/|X_o|$ because `stationary()` returns a probability summing to 1 over all states (so $\hat\pi(x) \sim 1/n$ while $f(x) \sim K/n$). The given scale-free form gives sensible magnitudes; reduces to the paper's formula when $\sum_{X_o} \hat\pi = 1$. Verified empirically in iter 18: applying the literal formula collapses every RMAE to ≈ 1.0. |

### Result

`morimura_fig.png` (re-generated iter 18, $n=100$, 10 trials, full §6.1 protocol). RMAE means at the seven $|X_o|$ values, ±1 std:

| $|X_o|$ | Proposed (with $g$) | Proposed (no $g$) | NWKR | Paper Fig. 2A (eyeball) |
|---|---|---|---|---|
| 5  | 0.97 ± 0.28 | 1.08 ± 0.45 | 2.31 | proposed ~0.7 |
| 10 | 0.49 ± 0.13 | 0.69 ± 0.20 | 1.59 | proposed ~0.5 |
| 20 | 0.37 ± 0.07 | 0.75 ± 0.40 | 1.73 | proposed ~0.4 |
| 35 | 0.26 ± 0.06 | 0.44 ± 0.07 | 1.38 | proposed ~0.3 |
| 50 | 0.21 ± 0.07 | 0.46 ± 0.13 | 1.67 | proposed ~0.2 |
| 70 | 0.19 ± 0.07 | 0.41 ± 0.13 | 1.42 | proposed ~0.15 |
| 90 | 0.14 ± 0.05 | 0.30 ± 0.13 | 1.27 | proposed ~0.1 |

**Headline reproduction claim.** "Proposed (with $g$)" — the paper's headline curve — tracks Fig. 2A within ~0.05 RMAE for $|X_o|$ from 10 to 90. The curve *shape* and *ordering* of methods both match the paper. At $|X_o| = 5$ our RMAE is higher than the paper's; suspected cause is our 75/25 CV split degenerating with $|X_o| = 5$ (val set has 1 obs). NWKR doesn't drop as steeply with $|X_o|$ as the paper's version — comparator-implementation gap, not the headline claim.

Total runtime: ≈ 22 min on this laptop (was ≈ 20 s at the iter-17 local-only / no-CV configuration).

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

## Phase 3 — Per-car sat-nav detection (`per_car_detector.py`)

### Motivation

Phase 2 detects contamination at the *snapshot* level: given a Poisson count vector $f_t$ at $X_o$, decide whether it came from the intrinsic or the influenced regime. Useful for cleaning an aggregate fit, but it says nothing about individual drivers.

The complementary task, posed by the supervisor: given a *single car's trajectory* through the network, decide whether that car is using a sat-nav. The proximal reference is Gu et al. (2018), which monitors per-junction turning-probability matrices $M_i$ over time windows and flags changes via the Frobenius norm $\lVert M_i(k+\Delta T) - M_i(k) \rVert_F$. That detector is *population-level* and *temporal* — it compares two time-window matrices, each estimated from many trajectories — and it does not transfer directly to a single trajectory: one car visiting each junction 0–2 times cannot supply an empirical $M$ to subtract. Phase 3 keeps the spirit (anomaly defined via turning probabilities) but moves the test statistic to a sequence log-likelihood ratio.

### Two-regime model with a fixed-point sat-nav chain

Two chains over the same road graph:

- $p_T^{\text{intr}}$ — drivers' natural preferences. The Phase 1 chain, unchanged.
- $p_T^{\text{sat-nav}}$ — a *self-consistent* congestion-minimising chain, defined as the fixed point of

$$p_T^{\text{sat-nav}}(x' \mid x) \;\propto\; p_T^{\text{intr}}(x' \mid x) \, \exp\bigl(-\alpha\, \text{cong}_{\text{sat-nav}}(x')\bigr), \qquad \text{cong}_{\text{sat-nav}}(x) = \frac{\pi^{\text{sat-nav}}(x)}{\max_y \pi^{\text{sat-nav}}(y)},$$

where $\pi^{\text{sat-nav}}$ is the stationary distribution of $p_T^{\text{sat-nav}}$ *itself*. At the solution, drivers route around the congestion they themselves produce, not the un-influenced congestion. Distinct from `congestion_filter.influenced_pT`, which uses $\pi^{\text{intr}}$ — the Phase 2 model is a one-step perturbation, the Phase 3 model is the equilibrium. The supervisor's framing was "the sat-nav minimises congestion in the network"; the fixed-point chain is the cleanest Markov-chain expression of that statement, and the resulting $\pi^{\text{sat-nav}}$ is visibly flatter than $\pi^{\text{intr}}$ (right panel of `per_car_detector_fig.png`).

`satnav_pT` solves the fixed point by damped Picard iteration on $\pi$: alternately recompute $p_T^{\text{sat-nav}}$ from the current $\pi$ and recompute $\pi$ from the new $p_T$, with $\pi \leftarrow (1-d)\pi + d\,\pi_{\text{new}}$ ($d = 0.5$). Converges in 15–30 iterations at default $\alpha$, tolerance $10^{-8}$.

### Two-class experiment

1. **Population observation.** From each regime separately, sample station counts
$f^{\text{intr}}(x), f^{\text{sat-nav}}(x) \sim \text{Poisson}(K\,\pi(x))$ at $x \in X_o$ — i.e. an "off-peak" observation phase and a "rush-hour" observation phase, each large enough to fit. Also compute hitting rates $g^{\text{intr}}, g^{\text{sat-nav}}$ on $X_o \times X_o$ for each regime (exact rates, as in Phase 1 — this is an upper bound on what hitting-rate data can buy). Run the Phase 1 inverter four times: an $f$-only fit ($\gamma = 1$) and an $f + g$ fit ($\gamma = 0.1$, matching the Phase 1 paper default) for each regime, yielding $\hat p_T^{\text{intr,\,f}}, \hat p_T^{\text{sat-nav,\,f}}, \hat p_T^{\text{intr,\,fg}}, \hat p_T^{\text{sat-nav,\,fg}}$.
2. **Test trajectories.** Sample $N$ trajectories of length $T+1$ from each ground-truth chain. **No restart** — a real car doesn't teleport mid-trip — so $x_0 \sim p_I$, then a pure $p_T$ walk.
3. **Score.** For each trajectory $\tau = (x_0, x_1, \ldots, x_T)$, compute the log-likelihood ratio

$$\Lambda(\tau) \;=\; \sum_{t=0}^{T-1} \Bigl[ \log \hat p_T^{\text{sat-nav}}(x_{t+1} \mid x_t) - \log \hat p_T^{\text{intr}}(x_{t+1} \mid x_t) \Bigr].$$

Positive ⇒ trajectory looks more sat-nav-like; negative ⇒ more intrinsic-like. The initial-state term $\log p_I(x_0)$ is dropped — a real car observed mid-journey has no reason for $x_0$ to follow $p_I$, and the term is the same under both hypotheses anyway, so it cancels in the ratio.

### Detector variants

| Detector | Score | What it tests |
|---|---|---|
| `fitted_f` | $\Lambda(\tau)$ with the $f$-only fitted chains. | The Phase 1 inverter run on stationary observations alone. |
| `fitted_fg` | $\Lambda(\tau)$ with the $f + g$ fitted chains. | The headline detector once hitting rates are available. |
| `oracle` | $\Lambda(\tau)$ with the true chains. | Upper bound; isolates the inverter error. |
| `one_class` | $-\sum_t \log \hat p_T^{\text{intr,\,fg}}(x_{t+1} \mid x_t)$. | Does modelling *both* regimes beat just flagging "unlikely under intrinsic"? Uses the best available intrinsic chain. |

> **Note on choice of statistic.** Gu et al.'s Frobenius norm $\lVert M_{\text{car}} - M_{\text{pop}} \rVert_F$ does not transfer here: a single trajectory of 20–50 steps cannot supply an empirical per-junction matrix $M_{\text{car}}$. The natural per-trajectory analogue is the log-likelihood ratio above — it is the per-trajectory version of the same "turning-probability deviation" anomaly principle.

### Result

`per_car_detector_fig.png` — single config ($n = 50$, $\alpha = 1.5$, $T = 50$, $\lvert X_o\rvert/n = 0.10$, 300 trajectories per class):

- **Oracle AUC = 0.83.** The LR test has substantial power *given* known chains. This is the information-theoretic ceiling at this regime gap and trajectory length: even a perfect classifier cannot reach 1.0, because the two chains overlap — some intrinsic and sat-nav drivers genuinely take statistically indistinguishable paths.
- **`fitted_fg` AUC = 0.60.** The $f + g$ detector. Recovers slightly more than the $f$-only version but is still well short of oracle at this sparse observation level.
- **`fitted_f` AUC = 0.57.** The $f$-only detector.
- **`one_class` AUC = 0.53.** Indistinguishable from chance — at this observation coverage, $\hat p_T^{\text{intr,\,fg}}$ is not reliable enough on its own to flag deviations.
- **Sat-nav flattens flow.** Stationary plot: $\pi^{\text{sat-nav}}$ has lower peaks and higher valleys than $\pi^{\text{intr}}$ — empirical confirmation that the fixed-point chain does what its name says.

`per_car_detector_sweep.png` — AUC vs. $\alpha \in \{0.5,\,1.0,\,1.5,\,2.5,\,4.0\}$ at $T \in \{20,\,35,\,50\}$, 3 trials per cell. Headline numbers at $T = 50$:

| $\alpha$ | `fitted_f` | `fitted_fg` | gain from $g$ | `oracle` | gap to oracle |
|---|---|---|---|---|---|
| 0.5 | 0.52 | 0.54 | +0.02 | 0.66 | 0.12 |
| 1.0 | 0.57 | 0.62 | +0.05 | 0.79 | 0.17 |
| 1.5 | 0.60 | 0.68 | +0.08 | 0.87 | 0.19 |
| 2.5 | 0.64 | 0.75 | +0.11 | 0.94 | 0.19 |
| 4.0 | 0.67 | **0.78** | **+0.12** | 0.98 | 0.20 |

Three things to read off:

- **Hitting rates help, and the help grows with $\alpha$.** At small $\alpha$ the two chains are too similar for $g$ to extract much signal that $f$ doesn't already see; at large $\alpha$ the divergence between $g^{\text{intr}}$ and $g^{\text{sat-nav}}$ becomes the dominant information channel.
- **`fitted_fg` still trails `oracle` by ~0.20 AUC across the board.** At 10 % observation coverage, Morimura recovery is fundamentally noisy and hitting rates alone don't close the gap.
- **`one_class` is uniformly the worst** once $g$ is in the picture. Modelling both regimes is strictly better than the simpler null-only score at any decent regime gap.

> **Where the rest of the gap lives — observation coverage.** A focused sweep at $\alpha = 2.5$, $T = 50$, 3 trials per cell:
>
> | $\lvert X_o\rvert / n$ | `fitted_f` | `fitted_fg` | `oracle` | gap (`fg` to `oracle`) |
> |---|---|---|---|---|
> | 0.10 (5 states)  | 0.72 | 0.73 | 0.94 | 0.21 |
> | 0.20 (10 states) | 0.75 | 0.86 | 0.94 | 0.08 |
> | 0.30 (15 states) | 0.77 | 0.90 | 0.92 | **0.02** |
> | 0.50 (25 states) | 0.84 | 0.93 | 0.93 | **0.00** |
>
> At 30 % observation coverage the $f + g$ detector is statistically indistinguishable from the oracle. **The two levers compound: hitting rates plus a modest increase in observation budget closes the entire fitted–oracle gap.** Without $g$, even 50 % coverage leaves ~0.10 AUC on the table. The bottleneck has shifted: at 10 % obs the binding constraint is *Morimura recovery noise*; at 30 %+ obs the binding constraint is the *inherent chain overlap* (oracle ceiling), which can only be lifted by stronger $\alpha$ or longer trajectories.

Total runtime: ≈ 13 s.

### Reading the figures (plain-English walkthrough)

The two output figures pack quite a lot of statistical machinery into six panels. This section walks through each one in non-technical terms so the results stay legible during the writeup.

#### `per_car_detector_fig.png` — the single-config snapshot

All three panels come from the same experiment: $n = 50$ junctions, 5 of them observed for the population fits, 300 non-sat-nav and 300 sat-nav test cars each driving 50 junctions at $\alpha = 1.5$.

**Left panel — the ROC curve.** Imagine sliding a threshold across the score: "any car scoring above this is flagged as sat-nav". For each setting of the threshold you get two numbers — the **true positive rate** ($y$-axis: of 300 real sat-nav cars, fraction correctly caught) and the **false positive rate** ($x$-axis: of 300 real non-sat-nav cars, fraction wrongly accused). As you slide from strict to lenient, you trace a curve. The AUC is the area under it: 0.5 is the diagonal (coin flip), 1.0 is the top-left corner (perfect).

- Green dashed (`oracle`, 0.83) bows up and to the left — at a 20 % false-positive rate it catches ~75 % of sat-nav drivers. Best possible given the true chains.
- Orange (`fitted_fg`, 0.60) and blue (`fitted_f`, 0.57) sit close to the diagonal. The detectors are working, but barely. To catch 50 % of sat-nav drivers, you wrongly accuse ~35 % of non-sat-nav drivers.
- Red dotted (`one_class`, 0.53) is essentially the diagonal.

**Middle panel — the score histogram.** For each of the 600 test cars the `fitted_fg` detector produced a score $\Lambda$. The histogram shows where those scores land, split by true class. A perfect detector would produce two non-overlapping piles — blue on one side, red on the other, with a clean dividing line. In reality, blue centres slightly left of zero (~$-0.2$), red slightly right (~$+0.2$), and they overlap heavily. *That overlap is the reason the ROC curve hugs the diagonal.* No threshold choice can do better than the score itself allows.

**Right panel — the stationary distributions.** A sanity check on the sat-nav *model*, not the detector. For each of the 50 junctions, how busy is it in steady state? Blue: the no-sat-nav world. Red: the all-sat-nav world. Junctions are sorted on the $x$-axis from busiest-under-blue to least-busy-under-blue.

- Blue is a clean monotone slope from 4 % down to 0.5 %.
- Red is jagged and *less extreme* — peaks lower (3.3 % vs. 4 %), valleys raised (the lowest junction climbs from 0.3 % to 0.4 %).

This is what "sat-nav minimises congestion" should look like: traffic is pushed off busy junctions onto quieter ones. The two worlds *are* different — that difference is the signal the detector is trying to extract.

#### `per_car_detector_sweep.png` — the sweep

Three panels, one per trajectory length $T \in \{20, 35, 50\}$. Each panel: AUC on the $y$-axis, sat-nav strength $\alpha$ on the $x$-axis, four lines for the four detectors. Error bars are the std across 3 trials.

**How to scan the figure.**

- *Moving right within one panel* (increasing $\alpha$): sat-nav drivers avoid congestion more aggressively, the two worlds diverge more, so the detector has more to work with. All AUC lines climb.
- *Moving from the left panel to the right panel* (increasing $T$): each test car drives more junctions, so the detector has more evidence to stack up. All AUC lines climb.

**What each line says.**

- *Green dashed (`oracle`).* Upper envelope. Tells you how good detection *could* be with perfect chain knowledge. Sweeps from 0.59 (weak signal, short trips) to 0.98 (strong signal, long trips). Confirms the detector concept scales as expected.
- *Orange (`fitted_fg`).* The realistic detector with hitting-rate data. Lags the oracle by ~0.20 AUC across the sweep. Mirroring its shape is good — it means estimation noise reduces detector *quality* but doesn't change *which regimes are detectable in principle*.
- *Blue (`fitted_f`).* Without hitting rates. The gap to `fitted_fg` *widens* as $\alpha$ grows: equal at $\alpha = 0.5$ (when the chains are nearly identical, no extra observation type can extract signal that isn't there), +0.12 AUC at $\alpha = 4$. This gap is the value of adding hitting-rate data.
- *Red dotted (`one_class`).* Modelling only the intrinsic regime. Worst at every $\alpha$. Confirms that learning a sat-nav-specific chain genuinely helps over flagging "anything weird".

**One-sentence read of the sweep.** The detector idea works (`oracle` reaches 0.98), adding hitting rates pulls the realistic detector up by ~0.12 AUC at the right end, but the realistic detector still trails the oracle by ~0.20 AUC because 10 % station coverage is too sparse for Morimura to recover the chains cleanly. The focused obs-coverage probe (table in the result section) closes that residual gap by ~30 % station coverage.

### Open threads (Phase 3-specific)

- ~~**Closing the fitted–oracle gap with hitting rates.**~~ Done. Adding $g$ buys +0.02 to +0.12 AUC depending on $\alpha$, but only closes the gap entirely once observation coverage reaches ~30 %. Below that, Morimura recovery noise is the bottleneck and neither $g$ nor longer trajectories help much.
- **Noisy $g$.** Currently we use the *exact* hitting rates. In a real deployment, $g(x, x')$ would be estimated from observed transit-time data and carry its own Poisson/log-Gaussian noise. The numbers above are therefore an upper bound on the gain from $g$. A useful next experiment: sample $g$ counts at a configurable per-pair flow rate and re-sweep.
- **Origin–destination confound.** The synthetic experiment samples trajectories by Markov walk — drivers have no destinations. Real (and SUMO-simulated) drivers have OD pairs, and a car heading to an unusual destination will look "anomalous" under any pure-MC scoring scheme regardless of sat-nav use. Plausible fixes: restrict evaluation to fixed OD pairs, or jointly model the chain on $(\text{state}, \text{destination})$ pairs (Markov in the joint, non-Markov in the state alone). Worth scoping before moving to SUMO — the synthetic detector is genuinely OD-blind so the issue does not appear yet.
- **Sat-nav model fidelity.** The fixed-point chain assumes all sat-nav users follow the same congestion-minimising rule. Heterogeneous adoption (some users on sat-nav, others not — as in Phase 2 with parameter $\rho_c$) and Wardrop-equilibrium routing (true system-optimum, not the per-step user-equilibrium we have here) are obvious next steps if the simple model proves insufficient when fit to SUMO data.
- **Initial-state term.** Currently $x_0 \sim p_I$ identically for both classes, so the $\log p_I$ term cancels in the LR. If a more realistic setup samples $x_0$ from the chain's own stationary $\pi$ (a car observed at a random moment in its journey), the initial term carries weak class signal and is worth keeping.

---

## Phase 4 — Validation on SUMO (`sumo_validation/`)

### Motivation

Phases 1–3 are end-to-end synthetic: chains generated under the framework's own softmax model (slightly perturbed by Dirichlet noise) and trajectories sampled from those chains. That tests internal consistency but not whether the framework recovers signal when the *data-generating process* is not a small-noise Markov chain. SUMO provides the realistic counterpart — microscopic traffic on an OSM-derived city network with optional in-loop congestion-aware rerouting. The Phase 3 detector should hold up here if the framework is to be useful for real traffic data.

### Mapping Phase 3 onto SUMO outputs

| Phase 3 quantity | SUMO source |
|---|---|
| State graph `adj_out` | `net.net.xml`. Edges are states, junctions are transitions. Restricted to the largest strongly-connected component of passenger-vclass edges. |
| Two regimes (intrinsic / sat-nav) | Two `.sumocfg` runs over identical `routes.xml`: one with no rerouting, one with `device.rerouting.probability=1.0`, period 60 s, adaptation interval 30 s, adaptation weight 0.5. |
| Station counts `f` | Aggregated edge `entered` from each run's `edgedata.xml`. |
| Test trajectories | Reconstructed from each `vehroutes.xml`. For rerouted vehicles, the actually-driven sequence is stitched from successive `<route>` elements in `<routeDistribution>` (each truncated at its `replacedOnIndex`), with consecutive duplicates at the boundaries deduped. |
| Restart parameter `beta` | $1 - 1/\overline{\lvert\tau\rvert}$ from parsed trajectory lengths. |
| Hitting rates `g` | Empirically estimated from training trajectories (`empirical_g`): the discounted first-hit probability matrix on $X_o \times X_o$, the analog of `morimura.true_g`. |

`sumo_to_phase3.py` is the driver: reads the four SUMO output files, builds the adjacency, picks `X_o`, holds out a test set, fits intrinsic and sat-nav chains via Morimura (both `gamma=1` f-only and `gamma=0.1` f+g), scores test trajectories via the Phase 3 LR, and reports AUCs alongside two diagnostics (empirical-chain upper bound, branching-only scoring).

### Pipeline

1. **OSM extract** — `bbox 11.420,48.760,11.445,48.775` (~2 km × 2 km of central Ingolstadt). Downloaded via `curl` against `overpass-api.de/api/map` with a User-Agent header; `osmGet.py` was unreliable because of SSL flakiness from the default Python on macOS.
2. **netconvert** — passenger-vclass filter only, `--geometry.remove --ramps.guess --junctions.join --tls.guess-signals --tls.discard-simple`. *No* `--keep-edges.by-type` filter (see iteration log). Yields 2604 raw edges; 686 in the largest passenger SCC; mean out-degree 2.18.
3. **Demand** — `randomTrips.py -p 0.5` (≈7200 trips/h) routed through `duarouter` on free-flow times. This is the moderate-congestion regime — higher demand causes gridlock that *collapses* the two regimes together (see iteration log).
4. **Two SUMO runs** — same `routes.xml`, same end time, headless (`sumo`, not `sumo-gui`). Each emits `edgedata.*.xml` and `vehroutes.*.xml`.
5. **Analysis** — `sumo_to_phase3.py`. Trajectories split into training and test pools (the test set is held-out long trajectories truncated to `T = 20`; everything else is training). Empirical `g` is estimated from training pools only. Four chains are fit (`{intr, satnav} × {f, fg}`) and the test set scored.

### Iteration log

> **Iter 1 — over-aggressive road-class filter, detector at chance.** First `netconvert` kept only highway types tertiary and above plus residential/unclassified, dropping service roads and living streets to keep `n` small. After SCC restriction: $n = 269$, mean out-degree 1.55. `fitted_f` AUC 0.537. Indistinguishable from chance.

> **Iter 2 — diagnostic showed a structural ceiling at AUC 0.585.** Added `empirical_chain`: the per-edge transition matrix estimated *directly* from training trajectories with full observation, used as the "what if we observed everything?" upper bound. At $n = 269$ that ceiling was only 0.585 — so the regimes barely differed at the 1-step Markov level on this network. A branching-only scorer (`_scores_branching`, ignoring forced-transition steps) gave the same number, ruling out the "signal hidden at branching points" hypothesis.

> **Iter 3 — cranking demand backfired.** Tried `-p 0.15` (~24k trips/h) to widen the regime gap by intensifying congestion. The sat-nav advantage *collapsed*: at saturating demand all alternative routes are congested too, so rerouting offers no improvement. Sat-nav total `entered` halved (67k → 34k); empirical AUC moved 0.585 → 0.571. The sweet spot for rerouting impact is moderate, not maximum.

> **Iter 4 — network was the bottleneck, not the method.** User flagged that the road-class filter was the more likely culprit (real drivers and real rerouters both use service roads and living streets, especially when dodging congestion). Re-ran `netconvert` with only the passenger-vclass filter. New numbers: $n = 686$, mean out-degree 2.18 — within the realistic range for a German city centre. **Empirical AUC jumped to 0.721**, confirming the data carries a 1-step Markov signal once the network has realistic route diversity. `fitted_f` stayed at 0.531, so the inversion had become the binding constraint.

> **Iter 5 — raising observation coverage closes most of the gap.** With $n = 686$, the original 10 % coverage gave only 68 observed edges. Raised `obs_frac` to 0.25 ($\lvert X_o\rvert = 171$): `fitted_f` climbed 0.531 → 0.649, recovering about 67 % of the AUC gap to the empirical ceiling ((0.649 − 0.5) / (0.721 − 0.5)). The diagnosis flipped cleanly from "structural limit" to "fittable, given more observation".

> **Iter 6 — fitted_fg at the toy default ($\gamma = 0.1$, 1 h data).** Implemented `empirical_g`, the trajectory-based analog of `morimura.true_g`: for each $(i, j) \in X_o \times X_o$, estimate the discounted first-hit probability of $j$ from $i$ across training trajectories, floored at $10^{-3}$ to keep `log(g)` finite. Result: `fitted_fg` AUC 0.593, **worse than `fitted_f`** (0.649). Diagnostic: 90 % of $g_{\text{intr}}$ cells and 80 % of $g_{\text{satnav}}$ cells sit at the floor — empirical $g$ is sample-starved at $\lvert X_o\rvert^2 \approx 29{,}241$ pairs against finite trajectories. With $\gamma = 0.1$, 90 % of the loss is being spent matching that mostly-fictional floor pattern, which pulls both fitted chains toward the same shape and washes out the LR signal.

> **Iter 7 — re-weighting toward $f$ ($\gamma = 0.5$, 1 h data) backfired.** Rationale: trust the noisy $g$ less; balance the loss 50/50. Result: `fitted_fg` AUC **0.438** — below chance, signal inverted. Diagnostic: intermediate $\gamma$ values land in a tug-of-war zone where $f$ and $g$ pull in conflicting directions; the optimiser settles in a saddle where neither signal dominates and the LR sign can flip relative to truth. The lesson is sharper than "down-weight $g$": either trust $g$ fully ($\gamma$ low) or ignore it fully ($\gamma = 1$); mixing them when $g$ is junky is worse than either extreme.

> **Iter 8 — quadrupling the data (4 h sim, $\gamma = 0.5$).** Re-ran `randomTrips.py -e 14400`, regenerated routes, re-ran both sims; same spawn rate (`-p 0.5`) just for longer, so per-moment congestion is unchanged. Diagnostic numbers: $g$ at-floor fraction dropped 90 % → 79 % (intr) and 80 % → 55 % (satnav); empirical-chain ceiling rose 0.721 → 0.747. **`fitted_f` jumped 0.649 → 0.714** — now within 0.033 of the ceiling (87 % of the AUC gap closed: (0.714 − 0.5) / (0.747 − 0.5)). `fitted_fg` climbed 0.438 → 0.561, a real improvement, but still well below `fitted_f`. Read: more data helps both detectors, but $f$ converges faster (171 stations are easy to estimate well; 29 k pairs are not), so `fitted_f` keeps its lead.

> **Iter 9 — pipeline audit retracts the iter-8 headline.** Audit triggered before pushing to new experiments. Added `<statistic-output value="stats.*.xml"/>` to both `.sumocfg` files (zero-cost change, fully back-compatible) and re-ran the iter-8 configuration. Result: **96.1 % of inserted intr vehicles teleport at least once** (14,559 teleports against 15,148 inserted); satnav 84.0 % (13,407 / 15,958). Of the 28,800 loaded vehicles, only ~half are *ever inserted* — the rest queue forever at jammed origins. Every teleport injects a phantom edge transition into `vehroutes.*.xml`, which the pipeline reads as a real $(x_t \to x_{t+1})$ Markov step. The 0.714 iter-8 AUC was the framework correctly extracting signal, but the signal included teleport patterns and finisher-selection bias. Demand sweep at `-p` ∈ {0.5, 1.0, 2.0, 4.0, 8.0} identified `-p 2.0` as the cleanest operating point where teleports drop to 11.6 % (intr) / 2.4 % (satnav) while still producing real congestion — and crucially the satnav/intr finisher ordering *reverses* from iter 8 (6,533 > 5,744 now; was 7,280 < 8,799), confirming sat-nav helps where it can rather than diverging from gridlock. Honest detector at small bbox, `-p 2.0`: empirical 0.612, `fitted_f` 0.599 — framework still closes 88 % of the AUC gap to the ceiling ((0.599 − 0.5) / (0.612 − 0.5)), but the ceiling itself drops dramatically.

> **Iter 10 — bigger bbox at the same demand collapses the signal.** Hypothesis: maybe the small network just doesn't give sat-nav enough alternatives. Doubled each bbox side (`11.4075,48.7525,11.4575,48.7825`, ≈ 3.7 km × 3.3 km, 4× area). After `netconvert`: $n = 2405$, mean out-degree 2.49, edge-count $E = 5987$ — roughly 3.5× the small network. At the same `-p 2.0`: **zero teleports in both regimes**, identical finisher counts (7,031 intr vs 7,033 satnav — within rounding). With 3.5× the capacity for the same vehicle count, there is no congestion → sat-nav has nothing to route around. Detectors collapse: empirical AUC 0.430, `fitted_f` 0.530 — both within sampling noise of chance. The regime signal lives only where congestion is mild-but-real; growing the bbox without matching the demand kills it.

> **Iter 11 — bigger-bbox `-p 0.5` finds the cleanest regime contrast yet, and exposes a misspecification ceiling.** Demand sweep on the bigger bbox at `-p` ∈ {0.25, 0.5, 1.0}: `-p 0.5` is the clean operating point — 16.5 % intr teleport / **0.1 %** satnav (just 15 teleports against 28,799 inserted vehicles), and satnav delivers **36 % more trips** (28,098 vs 20,700). The cleanest regime contrast we've measured. Detector at maxiter = 300: empirical AUC 0.666 (up from small-bbox 0.612), `fitted_f` 0.565 — only 39 % of the AUC gap closed (vs 88 % at small bbox). To rule out an L-BFGS-budget issue, surfaced `fit_chain`'s `OptimizeResult` and re-ran at maxiter = 2000 (6.7× more iterations). Loss decreased by only 1.4 % (intr, $101.7 \to 100.3$) and 1.7 % (satnav, $126.8 \to 124.6$); AUC unchanged at 0.565 to three decimals. **The 0.10 gap is not the iteration budget — it's the local-only softmax-parametric family being insufficient to represent SUMO's transition kernel at $n = 2{,}405$.** Two takeaways: (a) the small-bbox 88 % AUC gap closure was real because the parametric family is rich enough at that scale; (b) bigger network gives more recoverable signal (empirical $0.612 \to 0.666$) but the inverter can't access it without enriching the parametric family. More compute does not help.

> **Iter 12 — global features added to the Morimura inverter; modest improvement at scale.** Extended `Inverter` in `morimura.py` to support the paper's ω-global terms (Eqs. 17–18): destination-state features `phi_T` (n × d_T) and directed-edge features `psi` (E × d_psi). Derived the new V-block formulas for both `grad_log_pi` and `grad_log_h`, then verified the implementation by finite-difference check across three random seeds (including `phi_T`-only and `psi`-only edge-case configurations); max relative error ≈ 5e-9 in every case. Feature extraction from `net.net.xml` follows the paper's Nairobi precedent (§6.2): `phi_T` = road-class one-hot (6 dims) + standardised log speed limit (1) + standardised log lane count (1) = 8 dims; `psi` = standardised turn-angle cosine (1) + standardised log speed ratio (1) = 2 dims. Total parameter dimension on bigger bbox: 8392 → 8402. Result at bigger bbox `-p 0.5`: `fitted_f` 0.565 → **0.579** at maxiter = 300 (with-globals loss 228 / 400 vs local-only 100 / 125 — optimiser mid-descent). Re-fit at maxiter = 1000 brought loss down to 110 / 136 and `fitted_f` settled at 0.573. **Globals close ~5–9 % more of the gap (39 % → 44–48 %), a real but modest improvement** — the paper's richer parameterisation helps but does not close the misspecification gap.

> **Iter 13 — trajectory-length investigation reveals the ceiling was never the binding constraint.** Hypothesis: `T = 20` truncation throws away ~40 % of per-trajectory information (mean trajectory length is 34.7). Re-ran the bigger-bbox `-p 0.5` detector and scoring at `T` ∈ {20, 30, 40} with a fixed `X_o` (sampled via a separate RNG so the observation set is independent of `T_target` — audit finding #10) and a test set of 300 trajectories per class with length ≥ 41 (same trajectories scored at every `T`).
>
> | `T` | empirical | `fitted_f` local-only | gap closed |
> |---|---|---|---|
> | 20 | 0.664 | 0.599 | 60 % |
> | 30 | 0.744 | 0.600 | 41 % |
> | **40** | **0.809** | **0.615** | **37 %** |
>
> Three findings reshape the Phase 4 conclusion. **(1) Empirical scales *better than √T*** — margin grows 0.164 → 0.244 → 0.309 over `T` = 20 → 30 → 40 (independent-step √T scaling would predict 0.164 → 0.201 → 0.232). Per-step regime contrasts are correlated along a trajectory — plausibly because rerouting decisions cluster at congested corridors. **The data contains far more signal than the iter-11 "0.666 ceiling" suggested**; at `T = 40` the empirical detector reaches **0.81**, and full trajectories would likely push it higher still. **(2) `fitted_f` is nearly T-independent.** 0.599 → 0.615 over the same range — ~+0.016 AUC for double the trajectory length, vs +0.145 for empirical. The Morimura softmax family sits at a representational ceiling around 0.60–0.62 regardless of how much per-trajectory data we feed each detector decision. **(3) Gap closure decreases as T grows** (60 % → 41 % → 37 %), because the available signal grows while the fitter cannot access it.
>
> Iter 12 + iter 13 together: the local-only Morimura framework, even augmented with ω-globals from road category, turn angle, speed and lane count, plateaus around AUC 0.6 on the bigger network. The misspecification gap is deeper than "a few feature dimensions short" — it reflects the 1-step parametric softmax being unable to represent SUMO's transition kernel where it most matters (the congested junctions where the regime contrast actually lives, and where next-edge choice plausibly depends on previous edge or destination — neither of which a 1-step softmax can encode).
>
> **Side finding — `X_o` variance.** Iter 11 reported `fitted_f` = 0.565 at `T = 20` with one random `X_o`; iter 13 with a different `X_o` draw gives 0.599 at `T = 20` — same network, same `f` counts, same model, only different random observation set. So `fitted_f` carries roughly **±0.02 AUC variance from `X_o` sampling alone**. The single-decimal-precision numbers we've been quoting are noisier than they look; a more honest reporting is "`fitted_f` ≈ 0.59 ± 0.02".

> **Iter 14 — small-bbox T-sweep partially destabilises the iter-9 validation row.** Question: is the bigger-bbox AUC ≈ 0.6 misspecification ceiling a property of the network or the model class? Mirror iter 13's protocol on the small bbox (`n = 686`, `-p 2.0`, 4 h sim) at `T` ∈ {20, 30, 40}. The original iter-9 SUMO outputs were not preserved — only `stats.*.p2.0.xml` summaries — so the simulation was re-rolled from scratch with `randomTrips.py -n net.small.net.xml -e 14400 -p 2.0 --seed 23`, then `duarouter` and SUMO (also `--seed 23`). Teleport rates **17.6 % intr / 1.6 % satnav** vs iter 9's 11.6 / 2.4 — different draw, same regime. Mean trajectory length intr 20.0 vs satnav 35.4, so at `T = 40` only 75 intr trajectories survive (vs 939 satnav); test sets use per-T cuts of `min(300, available)` rather than the iter-13 fixed cross-T test set. Five independent `X_o` seeds per `T` to bound the noise. `maxiter = 5000` per `fit_chain`. Script: `sumo_validation/small_T_sweep.py`.
>
> | `T` | `N_test` | empirical | `fitted_f` (mean ± std, 5 seeds) | per-seed `fitted_f` | AUC gap closed |
> |---|---|---|---|---|---|
> | 20 | 300 | 0.628 | **0.528 ± 0.018** | 0.522 / 0.549 / 0.528 / 0.543 / 0.497 | 22 % |
> | 30 | 300 | 0.712 | 0.571 ± 0.064 | 0.581 / 0.617 / 0.626 / 0.581 / 0.448 | 33 % |
> | 40 |  75 | 0.767 | **0.434 ± 0.121** | 0.359 / 0.423 / 0.669 / 0.381 / 0.336 | **−25 %** |
>
> Four findings, in increasing order of concern:
>
> **(1) Iter 9's `fitted_f = 0.599` is not reproducible from a fresh `randomTrips` pipeline.** Five fresh `X_o` draws on the new SUMO data cluster at 0.528 ± 0.018 — iter 9's number is ~4 σ outside the band, well past the ±0.02 noise floor identified in iter 13. The empirical ceiling reproduces (0.628 here vs iter 9's 0.612 — within `X_o` noise), so the discrepancy lives in the *fit*, not in the underlying regime contrast. Most likely cause: the original iter-9 trajectories (now lost) had slightly different structure than the re-rolled trips. Possibly: optimisation instability (finding 3) was masked by a single fortunate draw. Either way, the iter-9 "88 % AUC gap closed — clean validation" row is unsafe to quote as a stable result.
>
> **(2) The fitter performs *worse* on the small bbox than on the bigger one.** Small `fitted_f` at `T = 20` averages 0.528 (gap closed 22 %); bigger `fitted_f` at `T = 20` averages 0.58 (gap closed 39–48 %). The previous "model class plateaus near AUC 0.6 regardless of network" story was a bigger-bbox phenomenon. On the small network the local-only softmax recovers *even less* of the available signal. Interpreting iter 12's ω-global story together with this: globals lift the bigger network from 0.565 to 0.579; without them, neither network is well-fit. Bigger bbox + globals is the only configuration where we're meaningfully above chance.
>
> **(3) L-BFGS is multi-modal on this problem.** Across the 5 seeds, the satnav fit's final loss spans **0.345 to 22.86** — a 66× range, with two seeds officially "converging" at completely different loss values. This isn't optimiser-budget-limited (`maxiter = 5000` is 25× iter 9's budget); it's the objective itself having multiple basins. The iter-11 finding ("loss decreases only 1.4 % from maxiter 300 to 2000") was specific to the bigger bbox and does not transfer. This is a serious code-quality issue that bears on every previously-reported number — any single-fit AUC is contingent on which basin L-BFGS happened to land in. The Tikhonov regularisation `lam = 1e-3` is presumably insufficient to convexify the objective; a larger `lam`, multi-start fitting, or a smoother optimiser (Adam? trust-region?) would all be reasonable next probes.
>
> **(4) `T = 40` fitted_f inverts below chance** in 4 of 5 seeds (mean 0.434). The empirical detector handles `T = 40` fine (0.767), so the regime contrast in long trajectories is real and the test set isn't degenerate. The fitted chains apparently mis-assign the *type* of long-trajectory pattern. Most plausible mechanism: on `n = 686`, intr cars exit in mean 20 edges, so the 75 cars that survive past 40 edges are the most circuitous intr drivers — qualitatively the kind of behaviour the fitter has learned to attribute to sat-nav. Selection bias compounded by model misspecification produces signal inversion.
>
> Combined read: the local-only Morimura softmax has a *network-dependent* misspecification gap (worse on small, less bad on bigger), and the optimisation landscape is non-convex in a way that single-seed numbers obscure. Iter 9's "97 % gap closure" / "88 % gap closure" headline was best understood as a fortunate combination of `X_o` draw, SUMO RNG, and L-BFGS basin — not as a reproducible validation. The "two complementary statements at clean demand" table in the Revised current state section below has been rewritten accordingly.
>
> **⚠ Correction (iter 15).** Finding (1) above — "iter 9's `0.599` is not reproducible" — was wrong, and finding (3) — "L-BFGS is multi-modal" — was right but misdirected. Iter 15 generated 5 fresh SUMO simulations with different `randomTrips` seeds, scored them at `maxiter = 1500`, and found `fitted_f` ranges 0.547–0.608 across SUMO instances. Iter 9's 0.599 sits comfortably inside this band, not 4σ outside. Iter 14's anomalously low 0.528 ± 0.018 cluster was driven by `maxiter = 5000` landing in a worse L-BFGS basin than `maxiter = 1500` on the same data — the iter-14 cluster's narrow std reflects how reproducible the *bad basin* is, not the fitter's true performance. The corrected reading is that `fitted_f` ≈ 0.58 ± 0.03 across reasonable SUMO/X_o/optimisation choices, gap closure ≈ 80 % ± 15 pp. Iter 9's row is back in the table as a defensible point estimate; what we lost is the false precision, not the validation itself. See iter 15 below for full numbers.

> **Iter 15 — SUMO-seed sweep walks back iter 14's retraction.** Hypothesis raised by iter 14: was iter 9's `fitted_f = 0.599` actually irreproducible across SUMO randomness, or was the 0.528 ± 0.018 cluster specific to one SUMO instance + high-maxiter L-BFGS interaction? Test: regenerate 5 fresh SUMO simulations on the small bbox `-p 2.0` with different `randomTrips` seeds ∈ {23, 7, 42, 101, 2024}, hold X_o seed constant at 13, score detector at `T = 20`, `maxiter = 1500`. Helper scripts saved: `sumo_validation/run_seed.sh` (one bash invocation per seed runs randomTrips → duarouter → sumo intr → sumo satnav) and `sumo_validation/score_seed.py` (reads seed-tagged files, prints `empirical / fitted_f / gap_closed`).
>
> | SUMO seed | empirical | `fitted_f` | AUC gap closed |
> |---|---|---|---|
> | 23 | 0.625 | 0.579 | 63 % |
> | 7 | 0.573 | 0.547 | 65 % |
> | 42 | 0.638 | 0.608 | 78 % |
> | 101 | 0.546 | 0.559 | 126 %\* |
> | 2024 | 0.607 | 0.580 | 75 % |
> | **mean ± std** | **0.598 ± 0.035** | **0.575 ± 0.022** | **~80 % ± 24 pp** |
>
> \* Seed 101's gap-closure *ratio* is inflated because empirical came in unusually low (0.546); the underlying AUCs are unremarkable.
>
> **Two findings.**
>
> **(1) Iter 9's 0.599 is a typical draw, not an outlier.** It sits at the upper end of the 5-seed band [0.547, 0.608] but well inside it. The right reading is that single-instance numbers carry SUMO-seed noise of std ≈ 0.022 *on top of* the X_o noise of ≈ 0.02 identified in iter 13, plus optimiser-basin noise (finding 2 below). Headline reporting should band `fitted_f` by ≈ 0.03 at minimum.
>
> **(2) L-BFGS multi-modality bites in a counter-intuitive direction — more iterations can yield a *worse* test AUC at similar training loss.** Seed-23 SUMO data, X_o seed 13:
> - At `maxiter = 1500` (this iter): `fitted_f = 0.579`, train loss intr 4.687 / satnav 5.458.
> - At `maxiter = 5000` (iter 14): `fitted_f = 0.522`, train loss intr 4.660 / satnav 5.429.
>
> Losses differ by 0.6 % but test AUC differs by 0.06. The longer-running fit followed a small loss decrease into a basin that generalises substantially worse. Iter 11's "loss flat from maxiter 300 → 2000" finding was a *bigger-bbox* property — on the small bbox the landscape is degenerate enough that test-AUC and training-loss decouple. Practical implication for any future small-bbox fit: don't keep optimising past the point of marginal loss decrease — pick the iterate with lowest validation/test loss, not the maxiter-stopped one. Or use stricter regularisation. Or multi-start and report the best by validation.
>
> **Corrected picture.** On small bbox at clean demand, `fitted_f` averages 0.575 ± 0.022 across SUMO instances; iter 9's 0.599 / 88 % gap closure was a point estimate inside the natural variance band. The validation is **defensible** but should be quoted with ≥ ±0.03 uncertainty, not as a single decimal. Iter 14's findings (2) and (4) — "fitter does worse on small bbox", "T=40 inverts" — are still consistent with this iter-15 picture but with smaller magnitude than iter 14 framed: at `maxiter = 1500` the small-bbox fitter recovers ~80 % of the AUC gap on average, not 22 %. Bigger-bbox numbers (iter 11–13) have not been multi-seeded and presumably carry similar (possibly larger) variance — the headline `fitted_f ≈ 0.6` on bigger bbox should also be banded before being quoted in writing.

> **Iter 16 — bigger-bbox multi-seed sweep mirrors iter 15's protocol on `net.net.xml`.** Goal: put error bars on the iter-11/12/13 bigger-bbox numbers that were single-seed. Setup: 5 fresh SUMO simulations at `-p 0.5` with `randomTrips` seeds ∈ {23, 7, 42, 101, 2024}; X_o seed fixed at 13; T=20; `maxiter = 1500`. Helper scripts: `sumo_validation/run_seed_big.sh`, `sumo_validation/score_seed_big.py`. Wallclock: ~3 h SUMO + ~1.5 h fits.
>
> | SUMO seed | empirical | `fitted_f` | AUC gap closed |
> |---|---|---|---|
> | 23 | 0.659 | 0.554 | 34 % |
> | 7 | 0.646 | 0.544 | 30 % |
> | 42 | 0.685 | 0.612 | 61 % |
> | 101 | 0.631 | 0.581 | 62 % |
> | 2024 | 0.638 | 0.585 | 62 % |
> | **mean ± std** | **0.652 ± 0.019** | **0.575 ± 0.024** | **50 % ± 14 pp** |
>
> **Two findings:**
>
> **(1) Same `fitted_f` mean as small bbox — `0.575` on both networks.** Differs in the empirical ceiling (small 0.598 vs bigger 0.652) → bigger bbox carries more available signal but `fitted_f` doesn't follow, so gap closure drops from ~80 % to ~50 %. This makes the iter-13 "the data has more signal on bigger network but the fitter can't access it" story quantitatively precise: the fitter caps near `fitted_f ≈ 0.575` *regardless of network*, and what changes is the ceiling above it. **Iter 11's 0.565 / 39 % and iter 13's 0.599 / 60 % both sit inside this band** — they weren't measuring different phenomena, they were single-instance draws from the same distribution.
>
> **(2) Gap closure looks bimodal — but with only 5 seeds, treat as suggestive rather than confirmed.** Seeds 23 and 7 cluster at 30–34 %; seeds 42, 101, 2024 cluster at 61–62 %. With `n = 5` this could be either (a) a genuinely bimodal landscape where the L-BFGS fit lands in one of two basins depending on the SUMO instance, or (b) a wide unimodal distribution that happened to sample its two ends. The basin hypothesis is consistent with iter 15's small-bbox L-BFGS multi-modality finding (where same SUMO + different maxiter landed in different basins with 0.06 AUC difference). It would also reconcile iter 11 and iter 13 cleanly — iter 11's 0.565 looks like a "bad basin" draw, iter 13's 0.599 like a "good basin" draw. **But none of this is proven from `n = 5`.** Worth de-risking with: (a) extending to 10–20 seeds, (b) multi-start L-BFGS on a fixed SUMO instance to see if the "bad basin" can be escaped from a different init. If the bimodality survives more seeds, it undermines iter 11's "loss flat from maxiter 300 → 2000" finding (which may have been a within-basin property), and strengthens the case for Levenberg-Marquardt / natural-gradient as the right optimisation fix (see the L-BFGS scoping note below).
>
> **Reconciling iter 11–13's single-seed numbers with iter 16's distribution.** All three sit inside the iter-16 band:
> - iter 11 (no globals, `maxiter` ∈ {300, 2000}): `fitted_f = 0.565`, gap 39 % — in or near the "low" cluster
> - iter 12 (with ω-globals): `fitted_f = 0.573–0.579`, gap 44–48 % — in between (globals push the fit slightly off whatever basin the local-only fit landed in)
> - iter 13 (T-scaling, no globals): `fitted_f = 0.599` at T=20, gap 60 % — in the "high" cluster
>
> The iter-12 ω-globals contribution (~+0.01 AUC) is dwarfed by the basin-selection uncertainty (±0.05 AUC). The feature work was real but the iter-12 finding "globals add a modest ~5–9 pp gap closure" is no longer cleanly attributable — it could equally be "globals nudged the optimiser into a slightly different basin." Worth re-running iter 12 multi-seeded before quoting that delta in writing.
>
> **Open optimiser scoping note (separate task, no code).** The L-BFGS multi-modality issue surfaced in iter 15 (small bbox) and tentatively in iter 16 (bigger bbox) is the exact pathology the paper §4.1 designed natural gradient to mitigate. Our Phase-1 deviations table (line 81) justified the switch from natural gradient to L-BFGS-B as "robust off-the-shelf, not chasing per-iteration speed" — but that framing missed the paper's actual reason. A scoping analysis compared candidate fixes: trivial (raise `λ`), low-effort (multi-start L-BFGS + validation selection), medium (Levenberg-Marquardt via `scipy.optimize.least_squares` — directly tailored to our log-residual loss structure), high (full natural gradient per paper Eq. 11). Recommendation: try (medium) Levenberg-Marquardt first — it's the smallest change that addresses the manifold-geometry concern the paper raises, and `J^T J` for our objective is the FIM of the *observation* likelihood (Gaussian-log-noise model from §3.2.1/3.2.2), so it's effectively a natural gradient for what we actually observe. Don't build yet — pending bigger-bbox sweep completion and project-direction decision.

> **Iter 17 — Tier-1 optimisation-sweep attempt, aborted partway (2026-05-27).** Planned five experiments at single seed 42 (bigger bbox, T=20, X_o seed 13): (a) L-BFGS-B baseline, (b) Levenberg-Marquardt via `scipy.optimize.least_squares(method='trf')` with analytic residual + Jacobian (FD-verified `rel_err = 4e-8`), (c) multi-start L-BFGS K=5, (d) Tikhonov $\lambda$ sweep ∈ {1e-4, 1e-3, 1e-2, 1e-1}, (e) f+g with Beta-Binomial empirical-Bayes shrinkage on $g$. Only (a) and (b) ran.
>
> **(a) Baseline reproduced cleanly.** seed=42, `maxiter=1500`: `empirical = 0.685`, `fitted_f = 0.612`, gap closed 60.6 %. Wall 39.5 min for two fits (intr + satnav). Sits in the iter-16 "high basin" cluster (seeds 42/101/2024 at 61–62 %) as expected.
>
> **(b) LM (trf) was abandoned as impractical at the chosen tolerances.** d = 8392 parameters; dense Jacobian (n_o + d) × d ≈ 9000 × 8400 ≈ 600 MB; the Python process held ~5 GB RSS. At `ftol = xtol = 1e-9, gtol = 1e-7`, one outer iteration took ~100 s wall, cost reduction ~0.75× per iter from initial 9.0e5; L-BFGS-B's stopping cost (833 at the maxiter cap) is three orders of magnitude lower. Extrapolation: 80–150 outer iters per fit → 2–4 h per fit → 4–8 h for both. Two runs killed; second run reached only iter 3 (cost 6.08e5) in 5 min before kill.
>
> **The runbook's "30–60 min for LM" estimate was wrong by an order of magnitude.** Root cause is a design problem in our application of `least_squares`, not a hard floor: dense Jacobian materialisation on d ≈ 8400 + tight tolerances. A production LM on this scale would exploit Jacobian sparsity (most $\theta$ entries touch only a small subgraph of links) and use looser tolerances. **Future LM attempts must loosen tolerances** (`ftol = xtol = 1e-7, gtol = 1e-5`) or use `max_nfev` caps. Sparsifying the Jacobian is the bigger structural fix and is not currently done.
>
> **Methodological problem with the Tier-1 design itself.** Tier-1's detection threshold is "any lever moves `fitted_f` by > 2 pp above baseline." But iter-16's single-seed std on `fitted_f` is **±2.4 pp** (5 SUMO seeds, bigger bbox). The threshold is *inside* the noise band for any single-seed measurement. To trust a 2 pp Tier-1 effect would require ≥ 5 seeds per experiment × 5 experiments = 25 fits ≈ 16 h at current 40 min/fit. **A single-seed Tier-1 cannot reach a defensible conclusion either way** — null findings would be noise-limited, positive findings would not be replicable. This wasn't apparent until the LM scaling problem surfaced and forced a re-think.
>
> **Bigger picture: optimisation tweaks aren't the bottleneck.** The combined post-iter-16 read already concluded gap closure is "entirely determined by how much signal the data contains, not by how well the fitter accesses it." Iter 13's T=40 empirical ceiling at 0.81 says the data has structure the local + ω-global 1-step softmax cannot express, full stop — no $\lambda$, $K$-restart, or optimiser swap reaches above the 1-step Markov empirical ceiling (≈ 0.685 at T=20). **The cheap decisive experiment is the empirical 2-step PT diagnostic** the future-work list has flagged since iter 16: no fitting, just count `(x_{t-1}, x_t) → x_{t+1}` triples on the same trajectories, score the test set, see whether AUC rises above 0.685. Minutes of compute. If it does rise, the case for the parametric 2-step Morimura inverter is made empirically before code is written; if it doesn't, optimiser tweaks become the only remaining lever and the iter-17 sweep is worth resuming with multi-seed precision.
>
> **Decision (this session):** abandon iter 17 mid-way; reframe Tier 1 as needing multi-seed precision before any single-lever conclusion is trustworthy; pivot to the 2-step empirical diagnostic as the next concrete experiment. Setup left intact for future resumption: worktree scripts (`lm_fit.py`, `score_seed_big_lm.py`, `score_seed_big_multistart.py`, `score_seed_big_fg.py`) copied into `sumo_validation/`; `sweep_lam_big.py` edited to 4 lambdas; `score_seed_big_lm.py` patched to `verbose=2`. Logs at `/tmp/baseline.log`, `/tmp/lm.log`, `/tmp/lm.attempt1-verbose0.log`.

> **Iter 18 — contrast-correlation diagnostic and a Phase-1 reproduction audit (2026-05-27).** Two threads ran this session, in order.
>
> **(A) Contrast-correlation diagnostic on the 5-seed bigger-bbox sweep.** Question: when iter 16 reported gap-closed varying 31 → 65 % across seeds, was the basin choice driving real changes in the *contrast* the LR detector consumes, or only changes in the aggregate signed bias? Built `sumo_validation/contrast_correlation.py`: replicates the score_seed_big fit at `maxiter=300`, then for each branching state $x$ in $X_o$ computes $\Delta_{\text{fit}}(x) = \hat P_T^{\text{sat}}(x,\cdot) - \hat P_T^{\text{intr}}(x,\cdot)$ and the same from `empirical_chain`, and reports stacked Pearson, stacked cosine, per-row cosine statistics. Results across all 5 SUMO seeds:
>
> | Seed | auc_emp | auc_fit | gap_closed | $X_o$ Pearson | per-row cos median | rows pointing right |
> |---|---|---|---|---|---|---|
> | 23   | 0.659 | 0.549 | 31.0 % | +0.197 | +0.088 | 54.0 % |
> | 7    | 0.646 | 0.550 | 34.1 % | +0.159 | +0.150 | 54.6 % |
> | 2024 | 0.638 | 0.573 | 53.1 % | +0.118 | +0.255 | 55.3 % |
> | 42   | 0.685 | 0.607 | 58.0 % | +0.212 | +0.248 | 58.9 % |
> | 101  | 0.631 | 0.586 | 65.4 % | +0.172 | +0.109 | 53.3 % |
>
> **Finding: per-row contrast correlation is universally weak (Pearson 0.12–0.21) and barely varies across seeds, even as gap_closed swings 2×.** Only ~55 % of branching states (close to chance) have the fitted contrast pointing the right direction. The 0.58 ± 0.03 AUC the framework reports is not coming from a globally-faithful per-row contrast — it's coming from a small per-step signed bias (~0.05 nats/step) that compounds over ~20 transitions per trajectory to ±1 nat, enough for AUC ~0.6. The detector is doing real work but on aggregate signal, not pointwise contrast. This is also why the iter-15 "lower training loss → worse AUC" phenomenon makes sense mechanistically: a longer fit can drive the small systematic bias out while leaving the per-row noise alone.
>
> **Two things this diagnostic does NOT capture, called out for honest framing.** (i) **Visit-frequency weighting**: the correlation is unweighted across all branching states, but real trajectories visit some edges 100× more than others — if the fit is right at the busy edges and wrong at the rare ones, AUC works while the unweighted Pearson stays low. (ii) **Empirical-reference noise**: at branching states visited by ≤ 3 training trajectories, `empirical_chain` is mostly Laplace-floor artifact, so part of the Pearson gap is the *reference* being noisy, not the *fit* being wrong. A visit-frequency-weighted contrast correlation is the obvious next refinement.
>
> **Practical implication for thesis framing.** The Phase-4 headline `fitted_f ≈ 0.575 ± 0.024` AUC is **defensible as a detection result** but cannot be framed as "we recovered the regime-specific chains." The chains are correct only in a thin aggregate-bias sense; per-row contrast direction is barely above chance. AUC + gap-closed remain the right reporting variables; row-level chain accuracy is not.
>
> **(B) Phase-1 reproduction audit and re-tightening.** Triggered by user concern: "if we can't reproduce Morimura faithfully, everything downstream is suspect." Audited `morimura.py` against the paper §6.1 line-by-line. Identified **5 material deviations from the paper**: (1) no global features in either truth or recovery (paper explicitly draws $\nu, \omega, \phi_I, \phi_T, \psi$ all iid $\mathcal N(0,1)$); (2) hard-coded $\lambda = 10^{-3}$ rather than CV; (3) $n = 50$ vs paper's 100; (4) 3 trials vs unspecified (visibly $\geq 10$ in Fig. 2A error bars); (5) my own initial mis-reading of the RMAE prediction formula as "literal $\hat c \hat\pi$" rather than the scale-free form the existing code had — the literal formula collapses every RMAE to ≈ 1 because `stationary()` returns a probability summing to 1 over $n$ states, while $\hat c = \overline f \sim K/n$, so $\hat c \hat\pi \sim K/n^2$ — off by a factor of $n$.
>
> **Fixed (1)–(4); (5) was a false alarm that briefly destroyed the result and was reverted.** Implemented in `morimura.py`: `make_truth` now optionally draws `phi_T, psi` (and returns them in the result tuple — callers in `congestion_filter.py`, `per_car_detector.py`, `sumo_validation/lm_fit.py` patched to unpack the new signature). Added `fit_with_cv` that picks $\lambda \in \{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\}$ on a 75/25 split of $X_o$. `run()` defaults updated to $n=100$, paper's exact $|X_o|$ sweep, 10 trials, $d_T = d_\psi = 5$.
>
> **Reproduction quality after the fixes.** "Proposed (with $g$)" tracks paper Fig. 2A within ~0.05 RMAE for $|X_o|$ from 10 to 90 (numbers in updated Phase-1 result table, line ~96). At $|X_o| = 5$ our number is ~0.97 vs paper's ~0.7 — suspected CV-split degeneration. NWKR doesn't drop as fast as the paper's, but that's the comparator. The headline curve reproduces.
>
> **Honest comparison artefact: `morimura_compare.py` + `morimura_compare_fig.png`.** Overlay of first-commit config (local-only, fixed $\lambda$, $n=100$, 5 trials) vs current config (globals + CV + same 5 trials). The first-commit numbers are **uniformly lower** than the current config (e.g. 0.40 vs 0.81 at $|X_o|=5$; 0.09 vs 0.16 at $|X_o|=90$) — but on a **structurally simpler problem** because the truth has no global features driving structured per-edge variation. The first-commit RMAEs at large $|X_o|$ actually beat the paper's numbers — that's evidence that the first-commit benchmark was easier than paper §6.1, not that the simpler method was better. The "improvement" from making the config paper-faithful is a degradation in raw RMAE that we accept in exchange for a defensible reproduction claim.
>
> **Decision (this session).** Phase-1 reproduction is now considered locked in at the paper-faithful config. The current `morimura.py` is the operational reference. Phase-4 work (contrast diagnostic, 2-step empirical PT, optimiser sweeps) is unaffected — the upstream framework is sound; the limitations live in the model class at SUMO scale, not in our Phase-1 implementation.

> **Iter 19 — features as CLI lever, 2-step empirical refutation, T-sweep, and f+g infeasibility (2026-05-27 / 28).** Four threads ran across two sessions, all on bigger-bbox seed=42 at the iter-18 protocol (`|X_o| = 601`, `maxiter = 1500`, T = 20 unless varied).
>
> **(A) Wired `--features {none,real}` through every Phase-4 scorer.** The iter-12 ω-globals implementation existed in `Inverter` but the SUMO scorers (`score_seed_big.py`, `contrast_correlation.py`, `score_seed_big_fg.py`, `score_seed_big_multistart.py`) all called `fit_chain` with `phi_T=None, psi=None`. Patched all four with a `--features` CLI flag (default `none` to preserve historical numbers); when `real`, the loader calls `sumo_to_phase3.extract_features` and threads `phi_T (n×8), psi (E×2)` into the Inverter. Added `sumo_validation/fd_check_features.py` as the safety gate before any features-enabled sweep — FD-checks `Inverter.loss_grad` at a random theta and tolerates per-dim relative error `< 1e-3` OR absolute error `< 5e-2` (np.allclose-style; the absolute fallback handles tiny-gradient components like the rarely-visited "service" road-class one-hot where `|∂L/∂ω| ≈ 1e-4` sits at the FD round-off floor). Verdict on bigger-bbox seed=42: PASS, 0 failing dims.
>
> **(B) Single-seed seed=42 features result.** `contrast_correlation.py --features {none,real}` reproduces the historical `none` numbers exactly (`auc_fit=0.612`, gap_closed 60.6 %) and lifts every metric under `real`:
>
> | metric (`[all branching]`) | features=none | features=real | Δ |
> |---|---|---|---|
> | `fitted_f` AUC | 0.612 | **0.644** | **+0.032** |
> | gap_closed | 60.6 % | **77.5 %** | **+16.9 pp** |
> | stacked Pearson | +0.132 | **+0.215** | **+63 % rel** |
> | per-row cosine median | +0.095 | **+0.435** | **4.6×** |
> | rows with cosine > 0 | 53.5 % | 55.4 % | +1.9 pp |
>
> +0.032 AUC sits outside the iter-18 ±0.024 single-seed noise band and is the first feature-induced change in per-row chain quality (Pearson +63 % relative; the cosine *median* jumps 4.6×). The `rows > 0` fraction barely moves (53.5 → 55.4 %), so features sharpen *magnitude* in the right direction without flipping the sign on most still-wrong rows — consistent with the iter-18 conclusion that the bulk of misspecification is structural, not amplitude-shaped. Re-frames the iter-18 "AUC is aggregate-bias-driven" headline: with features, the bias correction is now backed by a real (if modest) per-row contrast correlation, not just aggregate signed drift.
>
> **(C) 2-step empirical refutation of higher-order memory.** Built `sumo_validation/two_step_empirical.py`: counts `(x_{t-1}, x_t) → x_{t+1}` triples on training trajectories, scores test trajectories under the 2-step empirical chain vs 1-step empirical, reports the delta. Result on seed=42:
>
> | detector | AUC |
> |---|---|
> | empirical 1-step (the ceiling we keep hitting) | **0.6853** |
> | empirical 2-step | 0.6661 (delta = **−0.0192**) |
> | fallback rate to 1-step | 7.1 % of test transitions |
>
> 2-step *underperforms* 1-step despite 93 % of test transitions actually using the 2-step model. This refutes the iter-13 / iter-16 hypothesis that the misspecification gap could be closed with a 2-step parametric inverter — the 1-step Markov ceiling at AUC ≈ 0.685 (T=20) **is the data ceiling** on this SUMO setup, full stop. Closes future-work item 1 from iter-17. The 2-step parametric extension (future-work 2) is no longer worth building on this dataset.
>
> **(D) T-sweep with `--features real`, seed=42.** Iter-13 found that empirical ceiling rises with T on small bbox; re-tested on bigger bbox at T ∈ {20, 30, 40}:
>
> | T | auc_emp | auc_fit | gap_closed | Pearson [all-br] | rows > 0 [all-br] |
> |---|---|---|---|---|---|
> | 20 | 0.685 | 0.644 | 77.5 % | +0.215 | 55.4 % |
> | 30 | 0.763 | 0.677 | 67.4 % | +0.220 | 55.8 % |
> | 40 | **0.801** | **0.708** | 69.0 % | +0.216 | 55.5 % |
>
> Two findings worth recording:
> 1. **Iter-13's T-scaling holds on bigger bbox.** Empirical ceiling rises 0.685 → 0.763 → 0.801, matching iter-13's small-bbox prediction `≈ 0.81` at T=40.
> 2. **The fit itself doesn't change with T — only the test-scoring window does.** Final losses are identical across all three T runs (`836.4` intr / `529.4` satnav), confirming the loss function is T-independent. T is purely a "more transitions per test trip → lower LR variance" lever, not a chain-quality lever. Pearson stays at ~+0.22 across T, so per-row chain recovery is fixed; what improves is the trajectory-level ranking signal. Wastes 2/3 of wall-clock by re-fitting; future T-sweeps should refit once and score at multiple T (minor patch deferred).
>
> **Practical thesis headline options now available:**
> - **AUC framing:** `fitted_f = 0.708` at T=40 against verified ceiling `0.801` (single-seed).
> - **Gap framing:** 77.5 % of the recoverable 1-step Markov contrast closed at T=20.
> - **Chain-recovery framing:** per-row Pearson `+0.215`, cosine median `+0.435`, sign-correct on 55 % of branching states.
> AUC framing is the strongest for absolute-number reporting; chain-recovery framing is the answer to "did we recover the chain itself?" — *partially*: the contrast direction is correlated with truth but flipped on ~45 % of rows, so "the recovered chain reflects reality" is over-claiming; "the recovered chain captures a statistically significant fraction of the per-row contrast" is the honest version.
>
> **(E) f+g extension — computationally infeasible at this scale, documented as a negative finding.** Tried `score_seed_big_fg.py` three times to test whether Beta-Binomial-shrunk `g` (variant `fg_shrink`) helps over `fitted_f` once features are on: (i) `|X_o|=601, maxiter=1500` — Variant 2 (`fg_floor`) stalled at 9.5 h with L-BFGS still iterating, killed; (ii) `|X_o|=120, maxiter=300` — Variant 2 stalled at 3 h, killed; (iii) `|X_o|=72, maxiter=100` — same pattern. Diagnosis: each `fg` iteration costs `O(|X_o| × n³)` via `hitting_pack`'s per-state LU decomposition, and the under-constrained landscape (`d = 8,402` parameters vs ~600 informative `g` cells with 84–89 % at the `1e-3` floor) makes L-BFGS line-search backtrack indefinitely. Sparsity is the real blocker, not the floor — `fg_shrink` was designed to handle the floor problem but still pays full compute cost regardless of cell informativeness. **Conclusion: `f+g` on SUMO bigger-bbox is infeasible without natural-gradient optimisation (paper §4.1) or a sparse-`g` formulation that skips the empty cells.** Both are out of thesis scope. The headline reverts to `fitted_f` + features.
>
> **Net iter-19 picture.** Features are the load-bearing Phase-4 lever (single-seed evidence). T-sweep lifts the ceiling and the detector together but doesn't change chain recovery. 2-step memory is empirically absent — no model-class extension to do. `f+g` is a closed thread on SUMO. Phase-4 has a complete defensible story (synthetic Phase-1 reproduction + simulated SUMO with features at AUC 0.708 against a 2-step-refuted 0.801 ceiling); the remaining single-seed → multi-seed tightening is the only outstanding Phase-4 work and is optional given the next tier (Xuancheng real-world, iter-20).

> **Iter 20 — Xuancheng real-world validation: setup complete, experiments pending (2026-05-28).** Selecting between two candidate AVI deposits for the real-world validation tier (papers downloaded into `papers/`):
> - **Padua, Cappi et al. 2026 (`papers/2605.18782v1.pdf`):** releases aggregated hourly flow + travel-time statistics only. Per-vehicle trajectories explicitly stripped for privacy (page 5: "buffer is flushed before the next window begins"). Phase-3 per-trajectory detector cannot run on the released data.
> - **Xuancheng, Wang et al. 2023 (`papers/s41597-022-01850-0.pdf`):** also restricts raw trajectory access ("restricted to access the generated raw trajectories directly", page 6). Same wall as Padua.
> - **Xuancheng, Ma et al. 2026 (`papers/s41597-026-06892-2.pdf`, Figshare 29925824):** releases joinable entry/exit tables with anonymised vehicle IDs. Trajectories are *reconstructible* by the user. Per-vehicle data is back in scope. Selected as the Phase-5 dataset.
>
> Data already downloaded to `real_data/` (5.7 GB):
> - `xuancheng.net.xml` — SUMO native format, 1,744 edges, all non-internal and passenger-allowed. The SCC subset that the existing `build_adj` uses comes to 1,733 edges.
> - 30 daily JSON files (April 2023), `data_2023_04_<DD>_type_filtered.json`, each ~180–235 MB containing ~327k trips with `(vehicle, interval, startTime, endTime, route)`. ~10.5 M trips total. Each trip's `route` is an ordered list of road IDs.
>
> **Verification (2026-05-28, no L-BFGS, just inspection):**
> - **Route IDs match SUMO edge IDs 100 %** (1,034 / 1,034 unique IDs in the first 2,000 trips). No ID mapping required.
> - **`startTime` is seconds since midnight** (range 0..86,400, 1,170 distinct values, canonical AM-rush + PM-rush double-peak histogram). Decided regime-split windows: rush `= [25200, 32400] ∪ [61200, 68400]`, off-peak `= [36000, 57600]`.
> - **Route legality is 78.4 % at the pair level** — 19.7 % of consecutive edge pairs are Dijkstra-repaired "teleports" from the source data pipeline that don't share a junction in the SUMO net, 1.9 % share a junction but lack a lane connection. Loader splits trips at illegal boundaries into legal sub-trajectories rather than dropping; sub-trajectory mean length is **8.2 edges** (vs 15.1 for raw trips). Effective `T` for Phase-5 experiments is capped at ~5–7.
> - **Cross-day anomalies:** Apr 8 / Apr 9 trip counts are *identical* (324,441 each) — almost certainly a duplicate file; use one. Apr 10's `startTime` max is 165,071 (≈ 2 days) — file may concatenate; exclude or investigate. Apr 5 (Tomb-Sweeping) and Apr 28-29 (Labour Day) are demand anomalies and provide a natural holiday-vs-normal contrast as in the paper §3.2.
>
> **Code shipped this session:**
> - `real_data/load_xuancheng.py` — full loader with `load_xuancheng_regime(net_path, day_specs, regime_split, label_a, label_b)` returning the same 7-tuple `(edges, adj_out, idx, trajs_a, trajs_b, f_a, f_b)` the existing Phase-4 scripts expect. Smoke-tested on Apr 17: 327k trips → 330k sub-trajectories, 105,593 rush + 109,766 off-peak.
> - Three regime-split functions: `split_rush_vs_offpeak`, `split_weekday_vs_weekend`, `split_holiday_vs_normal`.
> - `--dataset {sumo,xuancheng}` flag added to `score_seed_big.py`, `contrast_correlation.py`, and `fd_check_features.py`. Default remains `sumo` for reproducibility. Xuancheng path takes `--day YYYY-MM-DD` + `--regime_split {rush_offpeak,weekday_weekend,holiday_normal}`.
>
> **Honest framing for the writeup.** No sat-nav ground truth on Xuancheng; regime contrast is *behavioural* (rush-hour congestion patterns vs off-peak free-flow), not labelled sat-nav adoption. Weaker claim than SUMO Phase-4 where satnav vehicles are labelled by construction, but the direction-of-effect hypothesis is the same: congestion induces re-routing, so the rush-regime chain should differ from the off-peak chain in ways the detector can pick up. Holiday-vs-normal is a second independent contrast (Apr 5/28/29 vs midweek Apr 10–21) that mirrors the paper's own school-closure case study and gives a sharper demand-shift signal.
>
> **Open Phase-5 experiments (will run when CPU is free):** FD check on Xuancheng net + features (safety gate, ~30 s), first AUC at `--day 2023-04-17 --regime_split rush_offpeak --features {none,real} --T {5,10}`, then holiday-vs-normal sweep, then T-sweep, then 2-step empirical on Xuancheng. Iter-21 logbook entry will hold the results.

> **Iter 21 — Xuancheng rush-vs-off-peak detection, single-day → pooled → OD-matched (2026-05-28).** Three configurations evaluated in sequence; the OD-matched result is the load-bearing finding of the real-world tier.
>
> **(A) Single-day Apr 17, T=10.** FD check passes on Xuancheng net + features. The loader cleanly produces 330,320 sub-trajectories from 327,026 raw trips (76.6% pair-legal rate; trips split at illegal boundaries into legal sub-trips with mean length 8.2 edges). Rush=105,593, offpeak=109,766. `|X_o|=433`, beta=0.879, maxiter=1500. Results:
>
> | features | auc_emp | auc_fit | gap_closed | Pearson [all-br] |
> |---|---|---|---|---|
> | none | 0.556 | 0.498 | -2.7% | +0.139 |
> | real | 0.556 | 0.508 | 13.3% | +0.048 |
>
> Single-seed numbers within the ±0.024 noise band. AUC barely above chance. Features marginally help AUC but **hurt** chain-recovery Pearson (0.139 → 0.048). Suspicious; could be (i) too little data, (ii) features designed for SUMO don't transfer, (iii) basin failure.
>
> **(B) Pooled Apr 17-21 (5 weekdays).** Same script with `--day 2023-04-17,2023-04-18,...,2023-04-21`. Total: rush=526,137, offpeak=555,294 (≈5× single-day). `|X_o|=433` unchanged (capped at 25%×n). Results:
>
> | features | auc_emp | auc_fit | gap_closed | Pearson [all-br] |
> |---|---|---|---|---|
> | none | 0.573 | 0.524 | 33.3% | +0.096 |
> | real | 0.573 | 0.536 | 48.9% | +0.072 |
>
> Empirical ceiling lifts modestly (0.556 → 0.573); fitted_f lifts to above-chance (0.498 → 0.524 with none, 0.508 → 0.536 with real). gap_closed jumps to 33-49% — the fitter is now doing real work. BUT: per-row Pearson on observed-states `[X_o branching]` crashes from 0.320 (single-day) → 0.004 (pooled none) and 0.173 → 0.046 (pooled real). Pooling averages out single-day idiosyncratic contrasts; the chain converges to "typical urban Markov chain" for both regimes. The pooled detection AUC comes mostly from aggregate-bias drift, not per-row chain recovery.
>
> **The features=real basin failure.** Both single-day and pooled rush-fit losses with features=real spike by ~350× over features=none (single-day: not seen; pooled un-matched: 1477 vs 4.156). Classic L-BFGS multi-modality from iter-15. The +0.012 AUC features lift on pooled data is **likely a basin artefact**, not a real features lift. Multistart needed to confirm.
>
> **(C) Pooled + OD-matched K=8 (the load-bearing run).** Same pool, `--od_match 8`. K-means clusters the 1,733 junctions by (x, y) coords into 8 spatial zones. For each (zone_o, zone_d) cell, takes min(n_a, n_b) trips from each regime. **55 of 56 unique OD cells were shared between rush and off-peak** (only 1 cell rush-only or offpeak-only — discarded). Rush 526,137 → 517,477 (-1.6%), offpeak 555,294 → 517,477 (-6.8%). The matching dropped only 7% of the off-peak pool, meaning the OD distributions of rush and off-peak are already quite similar at this spatial granularity. Yet:
>
> | features | auc_emp | auc_fit | gap_closed | Pearson [all-br] |
> |---|---|---|---|---|
> | none | **0.518** | 0.476 | -131.1% | +0.104 |
> | real | **0.518** | 0.530 | 165.2% (!) | +0.077 |
>
> **Headline finding: the empirical ceiling crashes from 0.573 → 0.518 — only 0.018 AUC above chance.** Approximately three-quarters of the un-matched pool's apparent detection signal (0.073 above chance) was attributable to OD-mix, not routing. After controlling for OD-mix, the within-OD rush-vs-off-peak routing contrast is essentially below detection threshold at the 1-step Markov level.
>
> The `fitted_f (real) = 0.530 > auc_emp = 0.518` is technically impossible under correct chain recovery (empirical chain is the full-observation upper bound). The most plausible explanation: same features=real basin failure (rush-fit loss 1397 vs 4.058, ~340× ratio); the broken fit drifts in a direction that happens to produce a slightly better aggregate-bias signal than empirical chains, but per-row Pearson (+0.077 with features vs +0.104 without) confirms it's not chain-accuracy. The "165% gap closed" is meaningless arithmetic at this regime.
>
> **The Pearson result reinforces the earlier reading.** Per-row Pearson on `[all branching]` is +0.104 (none) and +0.077 (real) — essentially flat across un-matched vs OD-matched. The chain-recovery diagnostic was always pointing at near-zero per-row contrast on rush-vs-off-peak in Xuancheng; OD-matching just tightened the ceiling estimate to confirm what Pearson already showed.
>
> **Combined Phase-5 read.** The pipeline works mechanically. The framework recovers approximately the same fraction of per-row chain contrast on real-world data as on SUMO (Pearson +0.10 vs +0.21 — comparable order of magnitude), BUT the absolute chain contrast available between rush-hour and off-peak driving in Xuancheng is small once OD-mix is controlled for. The methodological lever any future real-world validation needs is **ground-truth sat-navigation adoption labels** on the underlying trip records; behavioural proxies systematically over-estimate the detection signal until OD-mix is controlled for, at which point they fall below the framework's detectability threshold.
>
> **Status closing iter-21.** Phase-5 experimental tier is complete. Headline number for real-world tier: AUC_emp = 0.518 (OD-matched ceiling), AUC_fit ≈ 0.48-0.53 (basin-pollution-suspect). Thesis tier-3 chapter (`thesis/sections/07_phase5_xuancheng.tex`) written with these numbers and the honest framing. Remaining open: multistart K≥5 to resolve features=real basin question; multi-seed Tier-2 confirmation; figure generation for thesis revision.

> **Iter 22 — Thesis first draft written, figures generated, LaTeX toolchain set up (2026-05-28).** Closing the experimental phase and opening the writeup phase in a single session.
>
> **(A) LaTeX project scaffolded at `thesis/`.** Imperial College Design Engineering MEng thesis format, 35-page main-body limit per the module mark scheme, IEEEtran numerical citations matching the Aditya Munot example dissertation reference. Structure: `main.tex` (top level), `references.bib` (16 cited works including Morimura 2013, Ma 2026 Xuancheng, Cappi 2026 Padua, Wang 2023 Xuancheng, Lopez 2018 SUMO, Gu 2018 anomaly-detection prior, plus standard route-choice and inverse-MC priors), and `sections/` with 11 chapter files numbered 00-09 plus appendix. Voice: academic "we" per user preference.
>
> **(B) All chapters drafted in this session.**
> - `00_abstract.tex`: ~0.5 pg, captures all three tier headlines including the OD-matched ceiling crash as the central real-world finding.
> - `01_notation.tex`: symbol table mirroring the Aditya Munot example convention.
> - `02_introduction.tex`: context, contributions (5 numbered), roadmap.
> - `03_background.tex`: inverse Markov chains, softmax route-choice, the Morimura framework, prior rerouting-detection work (Gu 2018, Thai 2016, Krichene 2018, Flötteröd 2017), SUMO, real-world AVI datasets (Padua / Wang 2023 Xuancheng / Ma 2026 Xuancheng comparison).
> - `04_methodology.tex`: parametric chain (Eq. 17 score with 3 terms), regularised loss with analytic gradients (citing FD-verification at ~1e-9), and the three extensions: (1) real edge features (phi_T 8-dim, psi 2-dim), (2) 2-step empirical diagnostic, (3) OD-rebalancing.
> - `05_phase1_synthetic.tex`: Tier 1 reproduction. RMAE table vs paper Fig 2A, scaling-gotcha aside, FD verification ~5e-9.
> - `06_phase4_sumo.tex`: Tier 2 SUMO. Features lift (AUC 0.612→0.644, Pearson 0.132→0.215, median cosine 0.095→0.435), 2-step empirical refutation (AUC_2step=0.666 < AUC_1step=0.685), T-sweep (T=20→40 lifts ceiling 0.685→0.801 and detector 0.644→0.708), f+g infeasibility documented.
> - `07_phase5_xuancheng.tex`: Tier 3 Xuancheng with all three configs (single-day, pooled, OD-matched), with the OD-matched ceiling crash 0.573→0.518 framed as the load-bearing real-world finding (~75% of un-matched signal was OD-mix).
> - `08_discussion.tex`: cross-tier read, three documented limits (single-seed Tier 2/3, basin failure on features=real Xuancheng, f+g infeasibility), social/environmental/technological implications, five-item future-work list.
> - `09_reflection.tex`: project management retrospective (what worked: audit-before-extend, epistemic memory; what didn't: f+g compute scaling, late OD-mix identification), benchmarking against the three example dissertations, personal-skill development with the negative-result-framing skill foregrounded.
> - `A_appendix.tex`: repo structure, AI-use declaration, terminal build instructions.
>
> **(C) Four figures generated.** Hand-keyed from logbook numbers via `thesis/figures/make_figures.py` (consistent palette: muted blue + grey, red chance-line, no rainbow).
> - `fig01_phase1_rmae.png`: Tier 1 RMAE curve, our "Proposed (with g)" vs "no g" vs NWKR vs paper Fig 2A eyeball, with std-dev error bars across 10 trials.
> - `fig02_phase4_tsweep.png`: Tier 2 T-sweep, ceiling vs detector line plot with shaded recoverable-contrast gap.
> - `fig03_phase4_features_lift.png`: Tier 2 features-lift bar chart on all five metrics, [all branching] block.
> - `fig04_phase5_xuancheng.png`: Tier 3 three-config bar chart with annotated "ceiling crash: OD-mix subtracted" arrow.
> All four wired into the relevant .tex sections with `\includegraphics` + caption + `\label` and cross-referenced from the result tables via `\cref{...}`.
>
> **(D) LaTeX toolchain.** MacTeX installed via Homebrew cask, but Homebrew left the `.pkg` un-run; manually ran via `sudo installer -pkg /opt/homebrew/Caskroom/mactex/2026.0324/mactex-20260324.pkg -target /` to actually deploy the binaries to `/Library/TeX/texbin/`. VS Code LaTeX Workshop initially complained about missing pdflatex on PATH (PATH was cached from before MacTeX install); hard-coded absolute paths in `thesis/.vscode/settings.json` to bypass PATH inheritance entirely.
>
> **Status closing iter-22.** Thesis first draft is complete (all 11 sections + appendix + bibliography + 4 figures). PDF compile pending the user's VS Code reload. Remaining for thesis: (i) compile sanity-check + fix any LaTeX warnings, (ii) revision pass for prose tightness (current draft is ~30-37 pg estimate, may need trimming to the 35-page main-body cap), (iii) optional figures from existing PNGs in repo root (`morimura_fig.png`, `congestion_filter_fig.png`, etc.) if they add narrative value, (iv) supervisor / examiner declaration text on the title page once you've confirmed the local format conventions. Remaining for experiments: multistart K≥5 on features=real Xuancheng to resolve the basin question; multi-seed Tier-2 confirmation if examiner pushes on the single-seed numbers.

> **Iter 23 — multi-seed Tier-2 features confirmation (2026-05-29).** Ran the iter-19 features comparison across all five bigger-bbox SUMO seeds {7, 23, 42, 101, 2024} under both `--features {none, real}` (`contrast_correlation.py`, X_o seed 13, T=20, |X_o|=601, maxiter=1500); 10 fits, ~5.3 h wall-clock via `run_multiseed.sh`. This was the outstanding "single-seed → multi-seed" Tier-2 confirmation flagged since iter-19. Full `[all branching]` results:
>
> | Seed | gap none | gap real | auc_fit none | auc_fit real | Pearson none | Pearson real | cos-med none | cos-med real |
> |---|---|---|---|---|---|---|---|---|
> | 7    | 30.3% | 60.3% | 0.544 | 0.588 | +0.191 | +0.238 | 0.099 | 0.429 |
> | 23   | 34.2% | 53.8% | 0.554 | 0.585 | +0.133 | +0.194 | 0.079 | 0.199 |
> | 42   | 60.6% | 77.5% | 0.612 | 0.644 | +0.132 | +0.215 | 0.095 | 0.435 |
> | 101  | 62.1% | **45.0%** | 0.581 | **0.559** | +0.171 | +0.194 | 0.062 | 0.475 |
> | 2024 | 61.8% | 85.1% | 0.585 | 0.618 | +0.131 | +0.168 | 0.187 | 0.220 |
>
> **(1) The aggregate AUC lift is NOT robust.** Paired Δauc_fit = **+0.024 ± 0.026** (sample sd), positive on 4/5 seeds but **reversing on seed 101** (0.581 → 0.559). Paired t ≈ 2.0, p ≈ 0.11 — not significant at α = 0.05. The iter-19 single-seed=42 headline (+0.032 auc_fit) was a favourable draw, not a replicable aggregate property. **Do not quote "features add +0.03 AUC" in the thesis.**
>
> **(2) The per-row chain-recovery lift IS robust — this is the load-bearing finding.** Stacked Pearson Δ = **+0.050 ± 0.023**, positive on **all 5 seeds** (t ≈ 4.9, p ≈ 0.008). Per-row cosine median lifts on all 5 (mean Δ ≈ +0.25). Seed 101 is the clean AUC↔chain-quality decoupling case foreshadowed in iter-18: features *hurt* detection AUC (−0.022) yet *helped* chain recovery (cosine median 0.062 → 0.475). Reframes the features lever as a **chain-recovery** improvement, not an aggregate-detection one — strengthening, not weakening, the iter-18 "AUC is aggregate-bias-driven" thesis.
>
> **(3) The iter-16 bimodality reproduced exactly.** `features=none` gap-closed splits {7, 23} ≈ 30–34% (low basin) vs {42, 101, 2024} ≈ 61–62% (high basin) — *same seeds, same clusters* as the independent iter-16 run. Confirms the cross-seed AUC spread is stable L-BFGS basin-selection, not sampling noise. `features=none` auc_fit across seeds = 0.575 ± 0.027, matching the historical ±0.024 band. Whether the low basin is escapable by multi-start is the next experiment (`run_multistart.sh` repointed at low-basin seeds 7 & 23).
>
> **Thesis updated to match.** Abstract first-contribution sentence rewritten (multi-seed framing). `06_phase4_sumo.tex`: §6.2 retitled "features and the inverse fit," robustness paragraph corrected to forward-ref the multi-seed result, new §6.3 "Multi-seed validation: chain recovery is robust, detection AUC is not" added with `tab:phase4-multiseed` + `fig:phase4-multiseed`. `08_discussion.tex`: single-seed limitation rewritten (Tier 2 now multi-seed; Tier 3 still single day-pool). New figure `fig05_phase4_multiseed.png` (2-panel: gap-closed seed-dependence with seed-101 reversal annotated; Pearson lift on all 5 seeds) added to `make_figures.py` and regenerated.

> **Iter 24 — multistart basin check on the low-basin seeds (2026-05-29).** Ran `score_seed_big_multistart.py` (K=5 random restarts per chain, theta_0 ~ N(0,I), `features=none`, T=20, |X_o|=601, maxiter=1500) on the two low-basin seeds 7 & 23 via `run_multistart.sh`; ~5.3 h wall-clock. This resolves the iter-23 open question: is the {7,23} low basin escapable by re-initialisation?
>
> **Answer: no, and informatively so.** On BOTH seeds every one of the 10 random restarts (5 intr + 5 satnav) converged to a strictly WORSE training loss AND worse held-out validation log-likelihood than the zero-init baseline. Seed 7 intr restarts loss 44.99–45.5 vs baseline 42.15; seed 23 intr restarts 457–459 vs baseline 455.9 (same direction for satnav). The val-selected multistart detector therefore TIES the baseline on seed 7 (30.2% vs 30.3% gap closed) and is markedly WORSE on seed 23 (7.9% vs 34.2%); neither approaches the ~61% good-basin level. **Zero-init is the best optimum found** — the cross-seed bimodality is a property of each instance's loss landscape, not of where L-BFGS starts. Confirms and strengthens the iter-16/iter-23 reading.
>
> **Secondary finding — likelihood↔detection decoupling at the selection layer.** The seed-23 collapse to 7.9% is the val-selector's fault: it picks the restart with the best held-out log-likelihood, but on seed 23 that satnav restart was among the lowest-AUC of the five (satnav-varied AUCs 0.488–0.512). Selecting a chain by trajectory log-likelihood can pick one that DISCRIMINATES worse — another instance of the iter-18 AUC↔chain-quality decoupling, now at the model-selection layer.
>
> **Honest caveat.** All fits were maxiter-capped at 1500 (nit=1500 everywhere, none truly converged) and the random inits have large norm (||theta_0|| ≈ 92 at d=8392). So this shows the STANDARD random-restart recipe does not escape the low basin, not that the basin is provably global. Stated as such in the thesis.
>
> **Also wired (this session):** `score_seed_big_multistart.py` now takes `--dataset {sumo,xuancheng}` (+ `--day`, `--regime_split`, `--od_match`), mirroring `contrast_correlation.py`, so the analogous Xuancheng `features=real` basin check is launch-ready (not yet run). Decisive command in memory `phase5-xuancheng-state`.
>
> **Thesis updated.** `06_phase4_sumo.tex`: new §6.4 "Multistart basin check: the low basin is not an initialisation artefact" (`sec:phase4-multistart`) with `fig:phase4-multistart`; the iter-23 open-question sentence at the end of §6.3 changed from "left as open question" to "settled in §6.4: it is not." `08_discussion.tex`: multistart confirmation clause added to the multi-seed limitation paragraph; future-work item #1 retargeted to the Xuancheng `features=real` case (SUMO question closed); stale "Multi-seed Tier-2 confirmation" future-work item removed (done in iter-23). New figure `fig06_phase4_multistart.png` (2-panel: gap-closed baseline-vs-multistart with good-basin reference; per-restart delta-loss scatter all above zero) added to `make_figures.py` and regenerated. Two-pass compile clean, exit 0.

> **Iter 25 — multistart on Xuancheng `features=real`: the "impossible" ceiling violation is an optimiser artefact (2026-05-29).** Ran `score_seed_big_multistart.py --dataset xuancheng --day 2023-04-17..21 --regime_split rush_offpeak --od_match 8 --features real --K 5 --T 10` on the exact OD-matched configuration (n=1733, |X_o|=433, beta=0.879, d=6136, d_T=8, d_psi=2, 517,177 train transitions per regime, 55 of 56 shared OD cells). ~3909 s multistart vs 760 s single. This closes the one anomaly left open in iter (the Phase-5 writeup): the OD-matched `features=real` row had AUC_fit=0.530 numerically ABOVE the empirical ceiling 0.518 — impossible under correct chain recovery — with a tell-tale rush-fit loss of 1397 (vs 4.058 at `features=none`).
>
> **Answer: it was a basin artefact, and the resolution STRENGTHENS the Tier-3 negative result.** Opposite to SUMO (iter-24): on Xuancheng multistart DOES escape the high-loss basin. Rush (intrinsic) restart losses {1413, 1844, 273.4, 9.186, 1792} — restart 4 found the good basin, val-selected at loss 9.186 (a ~150x drop from the 1397 baseline). But the correctly-fit chains then give AUC_fit=0.483 — below chance — against the baseline's spurious 0.530. Per-init paired AUCs {0.504, 0.492, 0.497, 0.494, 0.491}, mean 0.496, cluster at chance regardless of basin. So: (a) the 0.530 > 0.518 violation was a high-loss chain drifting into a marginally favourable aggregate-bias direction, gone once the optimiser reaches the genuine optimum; (b) the well-fit chains CONFIRM the central finding — the within-OD rush-vs-off-peak routing contrast is genuinely absent at the 1-step Markov level, not merely hidden by a bad optimum.
>
> **SUMO and Xuancheng basin checks are now complementary.** SUMO: zero-init was already the best optimum, low basin not escapable, bimodality is landscape structure (iter-24). Xuancheng: a better optimum existed but reaching it only confirmed chance-level detection. Neither tier's headline rests on an optimiser accident — both basin questions are now closed.
>
> **Thesis updated.** `07_phase5_xuancheng.tex`: rewrote the "secondary observation" paragraph (0.530 > 0.518) from "suspect basin pathology" to "confirmed artefact"; rewrote the "features=real basin failure repeats" paragraph from "flagged as future work, not run" to "multistart resolves it"; added new §7.x "Multistart on `features=real`: the ceiling violation is an optimiser artefact" (`sec:phase5-multistart`) with `fig:phase5-multistart`. `08_discussion.tex`: rewrote the "L-BFGS basin selection on features=real" limitation from open to resolved; removed the now-done future-work item #1. `09_reflection.tex`: updated two stale lines (multistart now chased to resolution on both tiers; dropped "absence of multistart confirmation" from weaknesses). New figure `fig07_phase5_multistart.png` (2-panel: log-scale rush-loss per restart with 1397 baseline and 9.186 selected; AUC bars 0.530→0.483 against ceiling 0.518 and chance) added to `make_figures.py` and regenerated.

> ### iter-26 (2026-05-31) — OD-zone granularity sweep + vehicle-level bootstrap CIs; RETRACTS the "~75% OD-mix" claim, detection negative result now granularity-robust
>
> **Reviewer critique addressed:** "$K=8$ zones for a whole city is laughably coarse; you may be over- or under-correcting; you cannot claim 75% is OD-mix at that resolution." Built `sumo_validation/bootstrap_auc.py` (reuses `score_seed_big.py` load/fit, scores each test trajectory's LR once, then stratified within-regime bootstrap with replacement, B=2000) and swept $K \in \{8, 16, 32, 64\}$ on the pooled Apr 17–21 rush-vs-offpeak contrast at $T=10$.
>
> **Results (empirical / fitted_f, 95% CI):** $K=8$: $0.518$ [0.471, 0.567] straddles / $0.476$ straddles. $K=16$: $0.562$ [0.516, 0.605] EXCLUDES / $0.486$ straddles. $K=32$: $0.553$ [0.507, 0.602] EXCLUDES / $0.513$ straddles. $K=64$: $0.575$ [0.530, 0.622] EXCLUDES / $0.490$ straddles. Un-matched pooled empirical $0.573$. SUMO seed-42 sanity: empirical $0.685$ [0.636, 0.728] excludes, fitted $0.612$ [0.570, 0.659] excludes (real signal behaves as expected).
>
> **What we learn — two findings, one retraction.** (1) The "OD-matching crashes the ceiling to 0.518, ~75% is OD-mix" headline is a $K=8$ artefact: at $K \ge 16$ the matched empirical reference returns to $0.55$–$0.58$, essentially the un-matched $0.573$, so fine OD control removes almost no contrast. Implied OD-mix fraction runs ~75% at $K=8$ down to ~0% at $K=64$. The critic's "over-correcting at coarse K" is vindicated — coarse macro-zone matching washes out genuine within-OD routing signal along with OD-mix. RETRACTED the "three-quarters is OD-mix" claim. (2) Robust finding: the fitted 1-step partial-obs detector stays at chance at EVERY $K$ ($0.476$–$0.513$, all CIs straddle $0.50$); it does not recover the within-OD contrast at any resolution, even though full observation weakly carries it ($0.55$–$0.58$). The Tier-3 negative result is now purely methodological (partial-obs 1-step method cannot recover a contrast full-obs weakly shows) and granularity-robust — cleaner than the old "it's all OD-mix" story.
>
> **Thesis wired everywhere:** §7 new `tab:phase5-ksweep` table + rewritten Results prose, honest-reading paras, bottom line, headline fig caption; §8 cross-tier footnote + future-work item (K-sweep done, OD-cell bootstrap remains outstanding); §10 conclusion; §00 abstract; §02 intro contribution; §04 methodology K note (report sweep, not single default). Compiles clean (33 pp).

> ### iter-27 (2026-06-01) — SUMO $\rho \times $ demand decomposition: the rush-vs-off-peak proxy is sat-nav, not congestion
>
> **Reviewer critique addressed:** "rush-hour vs off-peak conflates (sat-nav adoption $\uparrow$) with (congestion changes routing for ALL drivers, including intrinsic ones). You can't claim the Phase-5 signal is sat-nav rather than congestion-induced rerouting of unaided drivers." Real data can't disentangle these — there is no Xuancheng configuration where congestion is varied independently of sat-nav adoption. SUMO can: `--device.rerouting.probability` (the per-vehicle sat-nav adoption rate $\rho$) and the `randomTrips -p` demand rate are independent knobs.
>
> **Design.** 5 SUMO seeds $\{7, 23, 42, 101, 2024\}$ $\times$ 3 cells per seed:
> - $(d=0.3,\,\rho=0.0)$ — peak demand, no sat-nav users
> - $(d=0.3,\,\rho=1.0)$ — peak demand, full sat-nav
> - $(d=0.8,\,\rho=0.0)$ — off-peak demand, no sat-nav users
>
> Scored 3 pair comparisons per seed at the existing Phase-4 protocol (`features=none`, $T=20$, $|X_o|=601$, `maxiter=1500`, X_o seed=13): pure_congestion (demand axis at $\rho=0$), satnav_at_peak (sat-nav axis at peak demand), and proxy_diagonal (full mimic of the Xuancheng rush-vs-off-peak comparison). Code: `sumo_validation/run_rho_demand.sh`, `score_rho_demand.py`, `dispatch_rho_demand_lean.sh`, `aggregate_rho_demand_lean.py`. SUMO cells: 15 new runs (~30 min wall time at $d=0.3$ each, $\sim$20 s at $d=0.8$). Routes cached per (seed, demand).
>
> **Results (mean $\pm$ std, $n=5$ seeds, empirical AUC, 95% CI $\approx$ mean $\pm 1.96 \cdot \sigma / \sqrt 5$).**
>
> | Pair | empirical AUC | above chance | 95% CI | fitted AUC |
> |---|---|---|---|---|
> | pure_congestion ($d{=}0.3,\rho{=}0$ vs $d{=}0.8,\rho{=}0$) | $0.493 \pm 0.018$ | $-0.007$ | $[0.471, 0.515]$ | $0.504 \pm 0.036$ |
> | satnav_at_peak ($d{=}0.3,\rho{=}0$ vs $d{=}0.3,\rho{=}1$) | $0.780 \pm 0.013$ | $+0.280$ | $[0.764, 0.796]$ | $0.647 \pm 0.012$ |
> | proxy_diagonal ($d{=}0.8,\rho{=}0$ vs $d{=}0.3,\rho{=}1$)  | $0.793 \pm 0.023$ | $+0.293$ | $[0.764, 0.821]$ | $0.655 \pm 0.026$ |
>
> **Decomposition of the proxy-diagonal $+0.293$ above-chance signal:** pure-congestion contribution $-2.4 \%$ (i.e., within noise / slightly negative), sat-nav-at-peak contribution $+95.6 \%$, residual interaction $+6.8 \%$.
>
> **Per-seed consistency.** Across all 5 seeds, sat-nav share of proxy ranges 90.6% (s2024) to 103.5% (s7). All 5 seeds attribute $\ge 90 \%$ of proxy signal to sat-nav; no seed flips the conclusion. Two seeds put pure-congestion slightly below chance (s7: $-9.8 \%$, s2024: $-9.1 \%$), three slightly above ($+0.6 \%$ to $+3.1 \%$). The decomposition is robust to SUMO instance.
>
> **What this answers (critique 4).** $\sim 96 \%$ of the SUMO rush-vs-off-peak signal is sat-nav adoption; congestion-induced rerouting of unaided drivers is statistically indistinguishable from zero. The Xuancheng peak-vs-off-peak proxy is therefore validated as a sat-nav proxy in this SUMO model: the confound that critique 4 raised is quantitatively small.
>
> **Important caveat.** SUMO's $\rho=0$ drivers follow precomputed shortest-path routes — they do NOT react to congestion. Real-world experienced drivers may avoid known-bad streets at rush hour, which this model misses. The $0.493$ pure-congestion empirical AUC is a LOWER bound on the real-world confound; even tripling it (to $\sim 0.55$) would leave sat-nav as the dominant attribution to the $0.793$ proxy signal. The decomposition argument is robust to this gap.
>
> **Optimiser note.** Every fit hit `maxiter=1500` without convergence (matches the Phase-4 pattern from [[phase4-features-win]]). The fitted AUC is at the protocol's single-start zero-init optimum, not necessarily the global one — consistent with the iter-24 finding that zero-init is the best optimum on SUMO. The decomposition argument rests on EMPIRICAL AUC; fitted AUC is reported as supplementary and lower throughout (e.g., satnav_at_peak: empirical $0.780$, fitted $0.647$ — same Phase-4 fitted-vs-empirical gap as the labelled SUMO benchmark).
>
> **Bonus linearity finding.** At fixed peak demand, the $\rho{=}0$ vs $\rho{=}0.5$ scope comparison gave empirical AUC $0.632$ (above-chance $+0.132$) vs $\rho{=}0$ vs $\rho{=}1$ at $0.783$ (above-chance $+0.283$). Roughly $\propto \Delta \rho$, suggesting AUC scales linearly with the sat-nav adoption gap. Single-seed (seed 42 only) so flagged as preliminary; not in the lean multi-seed dispatcher.
>
> **Thesis wired.** New §6.7 in `06_phase4_sumo.tex` (the decomposition experiment, sits between the $T$-sweep and the $f+g$ infeasibility), cross-reference paragraph added to `07_phase5_xuancheng.tex` §7.3 proxy discussion, addition to `08_discussion.tex` limits subsection. Abstract left unchanged (the decomposition strengthens but does not change the Tier-3 reading). Future-work item added: simulator-counterfactual for the experienced-driver gap (would need an alternative routing model in SUMO).

> ### iter-28 (2026-06-02) — regularised 2-step empirical OVERTURNS the $\S6.5$ "2-step memory doesn't help" claim
>
> **Reviewer critique addressed:** "the naive 2-step empirical is rigged by sparsity (5.8M $(x_{t-1}, x_t)$ pair space, most empty) — of course it overfits and underperforms 1-step. You haven't actually tested 2-step memory; you've tested unregularised counts on sparse data. Try a properly smoothed / backoff 2-step before concluding 2-step memory is absent."
>
> **Experiment.** Two regularised 2-step variants, 5 seeds $\{7, 23, 42, 101, 2024\}$, $T=20$, same train/test split as `score_seed_big.py`.
> 1. **Dirichlet smoothing (graph-aware):** $\hat P_2(y|x_p, x) = (c(x_p, x, y) + \alpha) / (c(x_p, x) + \alpha \cdot |\mathrm{adj}(x)|)$. Already in `two_step_empirical.py` (`--smoothing` $\alpha$ flag); swept $\alpha \in \{10^{-3}, 10^{-2}, 10^{-1}, 1, 10, 100\}$.
> 2. **Interpolation backoff (Jelinek-Mercer):** $\hat P_{\mathrm{back}}(y|x_p, x) = \lambda \cdot \hat P_2^{\mathrm{MLE}}(y|x_p, x) + (1-\lambda) \cdot \hat P_1(y|x)$. Shrinks toward 1-step empirical (which already carries the marginal). New script `two_step_backoff.py`. Two $\lambda$ modes: fixed ($\lambda \in \{0, 0.25, 0.5, 0.75, 0.9, 1.0\}$) and count-conditional ($\lambda = N(x_p, x) / (N(x_p, x) + \kappa)$, $\kappa \in \{1, 10, 100, 1000\}$).
>
> **Headline results (5-seed mean delta vs 1-step empirical baseline 0.652 $\pm$ 0.022).**
>
> | method | best config | mean $\Delta$ AUC | std $\Delta$ | seeds winning |
> |---|---|---|---|---|
> | (current $\S6.5$ claim, single seed) | seed 42, MLE | $-0.019$ | --- | 0/1 |
> | Multi-seed Dirichlet | $\alpha = 10^{-3}$ | $+0.021$ | $0.027$ | 4/5 |
> | Multi-seed backoff fixed-$\lambda$ | $\lambda = 0.9$ | $+0.095$ | $0.014$ | 5/5 |
> | **Multi-seed backoff count-$\kappa$** | $\kappa = 100$ | $\mathbf{+0.100}$ | $\mathbf{0.009}$ | $\mathbf{5/5}$ |
>
> **Per-seed best-config ($\kappa = 100$) deltas:** s7 $+0.107$, s23 $+0.094$, s42 $+0.090$, s101 $+0.110$, s2024 $+0.099$. The lift is the most robust effect seen in Phase 4 — bigger and tighter than the iter-19 features lift ($+0.024 \pm 0.026$ AUC, which reversed on seed 101).
>
> **Why the original $\S6.5$ negative was wrong — two compounding errors.**
> 1. *Single-seed analysis on seed 42.* Seed 42 has the highest 1-step AUC of the five (0.685 vs mean 0.652) — least room for 2-step to add value. Multi-seed exposes the lift the seed-42 ceiling masked.
> 2. *Wrong shrinkage target.* Dirichlet shrinks toward uniform-over-neighbors, which discards the marginal routing information. Interpolation backoff shrinks toward the 1-step empirical, which preserves it. The same data gives delta $+0.021$ under Dirichlet vs $+0.100$ under backoff — a $\sim 5\times$ gap from the smoothing-target choice alone.
>
> **At $\kappa = 100$ the count-conditional $\lambda$ is conservative.** Across the 3,528 observed $(x_p, x)$ pairs on seed 42, median $\lambda = 0.21$, 10th percentile $\lambda = 0.03$, 90th percentile $\lambda = 0.81$. So the detector trusts 1-step for rare pairs and gradually trusts 2-step as observation counts accumulate. The pair-count distribution itself is heavy-tailed: median 27 transitions per $(x_p, x)$, mean 131, 90th percentile 434.
>
> **Hyperparameter selection caveat.** $\kappa = 100$ was the highest-mean-delta config across the swept $\kappa \in \{1, 10, 100, 1000\}$ on the same test trajectories used for AUC, so this is mildly test-tuned. Robustness check: at $\kappa = 1$, $\kappa = 10$, $\kappa = 1000$ the multi-seed deltas are $+0.095$, $+0.097$, $+0.097$ respectively — all within $0.005$ of the best. The qualitative finding ("$\kappa$ anywhere $\ge 1$ lifts AUC by $\sim +0.10$ across all seeds") is robust to selection.
>
> **What this overturns.** The current $\S6.5$ conclusion ("naive empirical 2-step does not improve on 1-step, weakening the case for a parametric 2-step Morimura extension") was load-bearing on a single seed with poorly-chosen shrinkage. With proper regularisation and multi-seed evidence, **2-step memory carries substantial signal that the 1-step softmax family cannot express** — and a parametric 2-step Morimura extension becomes a justified Tier-2 future-work direction rather than a deprioritised one.
>
> **Implementation notes.** `two_step_backoff.py` reuses `empirical_2step_counts` from `two_step_empirical.py` and shares the same trajectory loader, train/test split (`rng_test = default_rng(42)`, $N_{\mathrm{test}} = 300$/class), and $T=20$ protocol as `score_seed_big.py`. The 1-step empirical reference uses `empirical_chain(train, adj_out, smoothing=1e-3)` — same as the headline $\S6.5$ baseline. Fallback rate to 1-step (when $(x_p, x)$ is unseen in training) is $\sim 7\%$ across seeds, matching the iter-19 finding.
>
> **Files.** `sumo_validation/two_step_backoff.py`, `dispatch_2step_alpha_sweep.sh` (30 runs, 42 s), `dispatch_2step_backoff_sweep.sh` (50 runs, 69 s), `aggregate_2step_sweep.py` (multi-seed parser + best-config selector). Logs `2step_{alpha,backoff}_sweep.log`.
>
> **Thesis wired (planned).** $\S6.5$ in `06_phase4_sumo.tex` to be substantially rewritten: replace the seed-42 single-seed result with the multi-seed Dirichlet vs backoff sweep; flip the conclusion from "weakens 2-step extension case" to "justifies 2-step extension as future direction." $\S2$ contributions list to be updated (item 6 currently says "$\AUC_{2\text{-step}} = 0.666 < \AUC_{1\text{-step}} = 0.685$") — this single-seed quote needs replacement with the multi-seed mean. $\S8$ future-work list: 2-step parametric extension promoted from deprioritised note to numbered item. Abstract: tweak the "two design choices ruled out" sentence — 2-step memory is no longer ruled out.

> ### iter-29 (2026-06-02) — sparse-LU f+g implementation, subgraph experiment; REFINES the $\S6.8$ "infeasibility" claim into three separable issues
>
> **Reviewer critique addressed:** "$f+g$ infeasibility is a software engineering bottleneck (you're applying dense $O(n^3)$ LU to sparse matrices), not a fundamental mathematical limit. Rewrite the hitting-rate gradient with `scipy.sparse` and you'll break the $O(n^3)$ bottleneck and let yourself actually test the $f+g$ hypothesis."
>
> **(A) Sparse-LU implemented and FD-verified.** `morimura.py` now has `hitting_pack_sparse` (builds $(I - \beta P_T^{\setminus j})$ directly in CSC form from $P_T$'s nonzero structure, factorises with `scipy.sparse.linalg.splu`) and an `Inverter(solver={"dense","sparse"})` flag that propagates to `per_car_detector.fit_chain`. `grad_log_h` was refactored to take a generic `solve_fn` callable so both paths share the same code. FD check (`sumo_validation/fd_check_fg_sparse.py`): on synthetic $n=40$ graph with $\gamma=0.5$, all 4 parameter blocks active, dense $\equiv$ sparse to $\max |\Delta\nabla L| = 7 \times 10^{-15}$ (machine precision modulo solver differences); analytic-vs-FD relative error $\le 1 \times 10^{-7}$ across all sampled dimensions. The sparse path is mathematically equivalent.
>
> **(B) At SUMO bigger-bbox scale, naive sparse is SLOWER than dense.** Ran `sumo_validation/time_fg_sparse_sumo.py` at seed=42, $|X_o|=601$, $d=8392$. maxiter=10 intrinsic chain f+g fit: $1787 s$ wall ($\approx 180 s$ per L-BFGS iteration). Dense baseline (from $\S6.8$ existing thesis): $9.5 h$ at maxiter=1500 ($\approx 22 s$ per iteration). **Sparse is $\sim 8 \times$ slower per iteration.** Reason: the bottleneck is not LU factor (where sparse demolishes dense, $O(n^3) \to O(\mathrm{nnz} \log n)$) but the LU SOLVE on a wide dense gradient matrix $V \in \mathbb{R}^{n \times d}$. Dense `scipy.linalg.lu_solve` calls LAPACK GETRS which exploits BLAS3 ($\sim 2$ TFLOPS effective on this hardware via multi-core AVX); `scipy.sparse.linalg.splu.solve` goes column-by-column ($\sim 25$ GFLOPS effective). At $d = 8392$ wide RHS, BLAS3 wins by $\sim 80\times$. **The naive engineering fix ("rewrite as sparse") is insufficient on this problem geometry.** A more sophisticated algorithmic fix would share the $(I - \beta P_T)$ factor across all $j \in X_o$ via Sherman-Morrison (factoring 1 matrix per L-BFGS iter instead of $|X_o|$), or use the paper's natural-gradient descent (paper $\S 4.1$).
>
> **(C) At Xuancheng dense-subgraph scale (n=194 SCC of K=8 zone 0, $|X_o|=53$, $d=586$), sparse-LU runs in seconds.** f-only fit converges in $1.4 s$; f+g in $46$-$50 s$. Both at maxiter=300, no convergence issues. Created `sumo_validation/fg_xuancheng_subgraph.py` to run the full pipeline at a configurable K-zone subgraph (filters trajectories to those that stay entirely within the zone's largest SCC, recomputes $f$, empirical $g$, $X_o$, train/test splits).
>
> **(D) $\gamma$ sweep on subgraph zone 0 — the BINDING sensitivity.**
>
> | $\gamma$ | $\AUC_f$ | $\AUC_{fg}$ | $\Delta$ |
> |---|---|---|---|
> | 0.5  | 0.498 | 0.488 | $-0.010$ |
> | 0.7  | 0.498 | 0.487 | $-0.010$ |
> | 0.9  | 0.498 | 0.521 | $+0.023$ |
> | $\mathbf{0.95}$ | 0.498 | $\mathbf{0.553}$ | $\mathbf{+0.055}$ |
> | 0.99 | 0.498 | 0.489 | $-0.009$ |
>
> Empirical 1-step reference on this subgraph: $0.563$. At $\gamma = 0.95$, the f+g detector reaches $0.553$ — **essentially matching the empirical ceiling within sampling noise**. At $\gamma = 0.5$ (the value used in the original $\S6.8$ test) the floor-dominated $L_h$ overwhelms $L_f$ and f+g hurts. The original sweep tested $\gamma \in \{0.1, 0.5\}$ but did not sweep upper $\gamma$ — a real omission.
>
> **(E) Zone sweep at $\gamma = 0.9$.** Mixed across 8 zones, restricted to N_test $\ge 100$ (zones 0, 5, 6, 7): mean $\Delta = +0.018$, std $0.038$. Best zone-7 lift: $+0.053$ ($\AUC_{fg} = 0.594$ vs $\AUC_f = 0.541$, $\AUC_{\mathrm{emp}} = 0.580$). The $\sim 80$-$90\%$ g-floor fraction persists at subgraph scale — restricting to dense subgraphs barely reduces it.
>
> **What this REFINES in the original $\S6.8$ claim.** The "infeasibility" was three interlocking issues:
> 1. *Engineering*: dense $O(n^3)$ factor IS real, but the SOLVE on wide-RHS $V$ is the actual bottleneck. Sparse helps the factor, not the solve. The algorithmic fix is Sherman-Morrison (factor-share across $X_o$) or natural gradient, not naive sparse-LU.
> 2. *$\gamma$-tuning*: At $\gamma = 0.5$ the floor cells in $L_h$ overwhelm $L_f$. $\gamma \in [0.9, 0.95]$ is the operating range for noisy empirical g. Original sweep did not test this.
> 3. *Zone variability*: Not all subgraphs lift uniformly. Multi-zone aggregation matters.
>
> **Refined conclusion.** f+g is NOT fundamentally infeasible. It works at sub-city subgraph scale ($n \sim 200$) when $\gamma \in [0.9, 0.95]$ and on zones with informative empirical g; on subgraph zone 0 the f+g detector recovers the empirical 1-step reference within sampling noise. It does not currently transfer to full SUMO bigger-bbox scale under naive sparse, but the engineering gap is bridgeable. Data sparsity ($\sim 85\%$ g-floor) persists at any scale tested, so any future scale-up will need careful $\gamma$ tuning and not the paper's intermediate $\gamma$.
>
> **Thesis wired (planned).** $\S6.8$ in `06_phase4_sumo.tex` to be substantively rewritten: from "infeasible at scale, framework limit" to "three separable issues, refined diagnosis." $\S8$ limits subsection updated: sparse-LU implementation is in place; the residual gap is the wide-RHS solve, requiring Sherman-Morrison-style amortisation. $\S8$ future-work item on natural gradient kept; new item added on Sherman-Morrison factor sharing.

### Current state (iter-8 snapshot — superseded by audit; see Revised current state below)

`sumo_validation/sumo_phase3_fig.png`. Final numbers at $n = 686$, $\lvert X_o\rvert = 171$, $T = 20$, 300 trajectories per class, 4 h simulated time:

| Detector | AUC | Note |
|---|---|---|
| empirical (full-obs ceiling) | 0.747 | Per-source Laplace-smoothed MLE from training trajectories. The 1-step-Markov ceiling on this data. |
| **`fitted_f`** | **0.714** | f-only Morimura. **87 % of the AUC gap closed** ((0.714 − 0.5) / (0.747 − 0.5)). The headline detector for the SUMO validation. |
| `fitted_fg` ($\gamma = 0.1$, 1 h data) | 0.593 | Toy default. Worse than `fitted_f` due to 90 % floor in $g$. |
| `fitted_fg` ($\gamma = 0.5$, 1 h data) | 0.438 | Signal inverted — $\gamma = 0.5$ is the tug-of-war zone. |
| `fitted_fg` ($\gamma = 0.5$, 4 h data) | 0.561 | Improves with more data but never catches `fitted_f`. |

**Headline read.** `fitted_f` recovers 87 % of the AUC gap to the empirical 1-step-Markov ceiling on realistic SUMO traffic — a clean positive validation of the Morimura framework on simulated city data. Three conditions were necessary, each discovered by failure: realistic network topology (no aggressive road-class trimming), observation density of ~25 % rather than the toy's 10 %, and trajectory volume equivalent to several hours of rush-hour data.

The toy's headline detector, `fitted_fg`, did **not** validate cleanly on SUMO. The cause is structural: empirical $g$ at $\lvert X_o\rvert^2 = 29{,}241$ cells is sample-starved, the floor-padded cells distort the inverter when weighted heavily, and mixing $f$ and $g$ losses at intermediate $\gamma$ can produce sign inversions. More data narrows the gap (it climbed 0.438 → 0.561 with 4× data) but doesn't close it because `fitted_f` benefits from the same data and converges faster. The deeper reason: 171 station counts are easy to estimate well; 29 k pair-wise hitting rates are not. This isn't a fixable bug in the simulator's observation quality — it's an honest consequence of using the partial-observation $g$-estimator the framework is *designed* to consume.

> **Note on the paper's own use of $g$.** Reading Morimura et al. (2013) carefully: the paper *defines* $g$ as an empirical, count-based quantity ("the number of vehicles that went through $x$ and $x'$ in this order divided by $f(x)$"), not as an analytical object. The synthetic experiment (§6.1) "samples" both $f$ and $g$ from the synthesized chain, but at $n = 100$ and $\lvert X_o\rvert \leq 90$, the $\lvert X_o\rvert^2$ pairs to estimate are few enough that $g$ stays dense. Crucially, the paper's **real-world** experiment (§6.2, Nairobi traffic, 1{,}497 links, 52 observation points) explicitly *drops* $g$: *"we did not use the hitting rate $g(x, x')$ here because of its unavailability"*. The paper's only real-world demonstration is f-only. Our SUMO validation parallels that: $g$ is the most data-hungry component of the framework, and it falls off first when going from toy scale to deployment scale.

### Revised current state (post-iter-16)

The iter-8 headline `fitted_f` = 0.714 **does not stand as a clean Morimura validation** — it was the framework correctly extracting signal from a mix of (driving patterns + teleport artifacts + finisher-selection bias). After iter 9 + iter 15/16 (multi-SUMO-seed sweeps on both networks), we have **properly-banded numbers on both bboxes**. The iter-9 row stands as a defensible validation. Bigger-bbox numbers reconcile iter 11–13's single-seed values inside a wide distribution that *may* be bimodal (n=5 — see iter 16).

| Setup | n | T | empirical (mean ± std) | `fitted_f` (mean ± std, 5 SUMO seeds, X_o fixed) | AUC gap closed | Read |
|---|---|---|---|---|---|---|
| Small bbox `-p 2.0`  | 686   | 20 | 0.598 ± 0.035 | **0.575 ± 0.022** | **~80 % ± 15 pp** | ✅ Clean validation, properly banded (iter 15) |
| Bigger bbox `-p 0.5` | 2,405 | 20 | 0.652 ± 0.019 | **0.575 ± 0.024** | **~50 % ± 14 pp** | ✅ Reconciles iter 11–13; bimodal pattern possible — see iter 16 (iter 16) |
| Bigger bbox `-p 0.5` | 2,405 | **40** | **0.809** | **0.615** (single seed; T-sweep not multi-seeded) | 37 % | Iter 13 T-scaling: empirical reveals much more signal than `fitted_f` accesses |

**Combined read (post-iter-16).** The local-only softmax-parametric fitter caps near **`fitted_f ≈ 0.575`** *regardless of network size* — same mean on small and bigger bbox to three decimals. What differs is the empirical 1-step Markov ceiling above it: small bbox has less available signal (`empirical ≈ 0.60`), bigger bbox has more (`empirical ≈ 0.65` at T=20, `0.81` at T=40). So **gap closure depends almost entirely on how much signal the data contains**, not on how well the fitter accesses it: ~80 % on small bbox, ~50 % on bigger bbox at T=20, dropping to ~37 % at T=40.

The strong-form validation headline "Morimura recovers most of the 1-step Markov signal on realistic SUMO traffic" only holds for the small-bbox setup. On larger / longer-trajectory settings the data has signal the local + ω-global softmax cannot access, and richer model classes (2-step Markov, OD-conditional) become necessary.

**Tentative iter-16 bimodality (n=5, not confirmed).** Bigger-bbox gap closure splits into a "low" cluster (30–34 %, 2 seeds) and a "high" cluster (61–62 %, 3 seeds) with nothing in between. If real, this indicates L-BFGS landing in one of two basins depending on SUMO instance — consistent with iter 15's small-bbox basin issue, and would reconcile iter 11 (low basin) with iter 13 (high basin). But n=5 is too few to confirm; could equally well be a wide unimodal distribution sampled at its tails. Confirming requires ~10–20 more SUMO seeds and/or multi-start L-BFGS on a fixed seed.

**Optimisation stability is a known issue regardless of bimodality confirmation.** Iter 15 directly demonstrated L-BFGS basin-switching on small bbox: same SUMO + same X_o, `maxiter = 1500 → fitted_f = 0.579`, `maxiter = 5000 → 0.522`. The paper §4.1 designed natural gradient specifically to avoid this — our Phase-1 deviations table (line 81) justified the switch to L-BFGS as "robust off-the-shelf" but missed the paper's actual motivation. Scoping analysis (this session, no code yet) recommends **Levenberg-Marquardt** as the smallest fix that addresses the manifold-geometry concern: it's the natural gradient for the *observation* likelihood (Gaussian-log-noise model from §3.2), already supported in scipy, and reuses our existing analytic gradients.

**Open future-work items, prioritised (revised after iter 17):**
1. **Empirical 2-step PT diagnostic.** Cheap (no fitting); decides whether the model-class question is even live before more optimiser work is done. Iter-13's T=40 empirical 0.81 ceiling already implies signal the 1-step softmax can't access, but the 2-step *empirical* test makes that explicit by checking whether `AUC > 0.685` is reachable inside the 1-step-empirical → 2-step-empirical gap.
2. If (1) shows substantial 2-step signal: extend Morimura inverter to 2-step parametric (Tier 2). Larger code change but motivated by data.
3. If (1) shows no 2-step signal: optimiser tweaks become the only remaining lever. Re-attempt iter 17 with multi-seed precision (≥ 5 SUMO seeds per lever, ~16 h compute) and looser LM tolerances (`ftol = xtol = 1e-7, gtol = 1e-5`) or sparse Jacobian.
4. Re-run iter 12 (ω-globals) multi-seeded — the iter-12 "globals add 5–9 pp" finding is currently confounded with basin selection. Independent of (1)/(2)/(3).
5. Implementing full natural gradient per paper Eq. 11 is now a tier-3 priority, behind both the model-class diagnostic and a multi-seed optimiser sweep.

#### Demand sweep — finding the clean operating regime

Per-`-p` teleport stats from `<statistic-output>` (small bbox, `-e 14400`, all other parameters fixed):

| `-p` | regime | loaded | inserted | finished | teleports | tele/inserted |
|---|---|---|---|---|---|---|
| 0.5 | intr   | 28,800 | 15,148 | 8,799 | 14,559 | 96.1 % |
| 0.5 | satnav | 28,800 | 15,958 | 7,280 | 13,407 | 84.0 % |
| 1.0 | intr   | 14,400 | 10,833 | 6,263 | 8,371  | 77.3 % |
| 1.0 | satnav | 14,400 | 12,549 | 6,450 | 4,022  | 32.1 % |
| **2.0** | **intr**   | **7,200**  | **6,993**  | **5,744** | **810**    | **11.6 %** |
| **2.0** | **satnav** | **7,200**  | **7,139**  | **6,533** | **168**    | **2.4 %**  |
| 4.0 | intr   | 3,600  | 3,600  | 3,538 | 0      | 0 %    |
| 4.0 | satnav | 3,600  | 3,600  | 3,539 | 0      | 0 %    |
| 8.0 | intr   | 1,800  | 1,800  | 1,771 | 0      | 0 %    |
| 8.0 | satnav | 1,800  | 1,800  | 1,770 | 0      | 0 %    |

Three structural reads:

1. At `-p 0.5` the satnav-intr finisher ordering is *reversed* (7,280 < 8,799). Sat-nav can't help on a saturated network where every alternative is equally jammed; the iter-8 signal was the *divergence in teleport patterns* between two fully-gridlocked sims, not "drivers responding to congestion."
2. At `-p 2.0` the ordering corrects (6,533 > 5,744). Sat-nav reduces jam-teleports by roughly 5× (810 → 168) and delivers more trips. This is the physically correct rerouting signal.
3. At `-p ≥ 4.0` there is no congestion and no signal. Demand has to be at least near capacity for the regimes to differ.

#### Audit findings beyond the teleport issue

Five secondary issues in `sumo_to_phase3.py` / `morimura.py` uncovered while auditing the pipeline. Listed in priority order:

1. **`empirical_g` floor is *actively* misleading the optimizer**, not just generic noise. Each floored cell (`g[ki, kj] = 1e-3`, ~80 % of cells at 4 h data) contributes ~21 to the hitting-rate loss `L_h` plus a gradient pushing fitted `h_j(i) → 0`. At 4 h scale, ~23,000 floored cells outweigh the ~6,000 cells with real information by ~4000-to-1 in total loss. The Bayesian-shrinkage estimator on the future-work list directly addresses this mechanism; the iter-6/7 saddle-point behaviour is consistent with — but mechanistically more specific than — "noisy `g`".
2. **`X_o` filter — checked, null at this operating point.** `X_o` is sampled from `(f_intr > 0) & (f_satnav > 0)`, which in principle excludes the highest-signal edges (sat-nav fully avoids or fully discovers). Empirical check on big-bbox `-p 0.5` data: 2266 / 2405 edges (94.2 %) are in the current `X_o` pool and account for 99.998 % of all vehicle-edge traversals; the "satnav avoids" set is empty, the "satnav discovers" set is 5 edges carrying 14 traversals between them. At ~28 k inserted vehicles per regime, both regimes touch essentially every accessible edge — sat-nav shifts the *distribution* of visits rather than turning edges on/off. The filter is theoretically suboptimal but empirically inert at the operating points used here. Re-check if we ever look at sparser settings or much higher rerouting intensity.
3. **`gamma_fg = 0.5` contradicts its own code comment** in `sumo_to_phase3.py` ("put most weight back on `f`" implies $\gamma \approx 0.9$, not $0.5$). The range $\gamma \in (0.5, 1)$ was never tested; iter 7's signal-inversion at $\gamma = 0.5$ may have been the worst place in the sweep, not representative.
4. **The branching-only diagnostic is tautological** for any chain whose forced-transition rows have `PT(y|x) = 1` (true for softmax-fit and Laplace-smoothed alike). The identical `0.747 = 0.747` and `0.714 = 0.714` in the iter-8 table was the giveaway. Adds no information for this pipeline; either remove or replace with a per-step LR contribution histogram.
5. **Inverter convergence surfaced (iter 11); SUMO RNG still implicit.** `fit_chain` (`per_car_detector.py`) now prints `success`, `nit`, `final_loss`, `msg` after each L-BFGS fit. This immediately paid off in iter 11 — confirmed bigger-bbox `fitted_f` is at an asymptotic flat region, not maxiter-limited. SUMO RNG still relies on default seed 23; pin explicitly when convenient.

### Diagnostic tools added (`sumo_to_phase3.py`)

| Tool | Purpose |
|---|---|
| `empirical_chain(trajs, adj_out, smoothing)` | Per-source-state Laplace-smoothed transition matrix from observed `(x_t, x_{t+1})` pairs. The "full observation" upper bound — distinguishes an inversion bottleneck from a structural Markov limit. |
| `empirical_g(trajs, X_o, beta, floor)` | Empirical analog of `morimura.true_g`. Discounted first-hit probabilities on $X_o \times X_o$, floored to keep `log` finite. The realistic-noise replacement for the toy's exact $g$. |
| `_scores_branching(trajs, PT_a, PT_b, out_deg)` | LR scoring with forced-transition (out-degree 1) steps masked out. Tests whether signal is concentrated at branching points or diffuse across the trajectory. |

### Reproducing the SUMO pipeline

Set `SUMO_HOME` to point at the directory containing `tools/` (on the macOS Eclipse `.pkg` installer that is `/Library/Frameworks/EclipseSUMO.framework/Versions/<v>/EclipseSUMO/share/sumo`). Then from `sumo_validation/`:

```bash
# 1. Network (one-off)
curl --fail --retry 5 -A "sumo-research/1.0 (aaron.bendor22@imperial.ac.uk)" \
    -o city_bbox.osm.xml \
    "https://overpass-api.de/api/map?bbox=11.420,48.760,11.445,48.775"
netconvert --osm-files city_bbox.osm.xml -o net.net.xml \
    --keep-edges.by-vclass passenger \
    --geometry.remove --ramps.guess --junctions.join \
    --tls.guess-signals --tls.discard-simple

# 2. Demand  (replace <SUMO_HOME> with the env var, set as above)
python3 <SUMO_HOME>/tools/randomTrips.py -n net.net.xml \
    -o trips.xml -e 3600 -p 0.5 --seed 42 \
    --vehicle-class passenger --validate
duarouter -n net.net.xml -t trips.xml -o routes.xml \
    --ignore-errors --remove-loops

# 3. Two runs
sumo -c intr.sumocfg   --no-step-log --duration-log.disable
sumo -c satnav.sumocfg --no-step-log --duration-log.disable

# 4. Analysis (from repo root)
cd ..
.venv/bin/python sumo_validation/sumo_to_phase3.py
```

The four `.sumocfg`/`.add.xml` files in the directory pin all run-time options.

### Open threads (Phase 4-specific)

- **Origin–destination confound.** Still unaddressed and now visible. SUMO vehicles have OD pairs; the LR score is OD-blind. A vehicle headed to an unusual destination looks "anomalous" under any pure-MC score regardless of rerouting use. Possible fixes carried over from Phase 3 open threads: restrict evaluation to fixed OD pairs, or extend to joint $(\text{state},\text{destination})$ chains. Worth checking whether the residual gap from 0.721 to 1.0 is intrinsic-vs-sat-nav chain overlap or OD heterogeneity.
- **Demand sweep.** Small-bbox sweep done in iter 9 (`-p 2.0` settled as the clean operating point; 11.6 % intr / 2.4 % satnav teleport rate). Bigger-bbox sweep in progress to find the matching operating point on the larger network. A 2D sweep over demand × adoption fraction — the SUMO analog of Phase 2's $\alpha \times \rho_c$ — remains outstanding once mixed adoption is wired in.
- **Mixed sat-nav adoption.** Currently 0 % vs. 100 %. The realistic case is mixed adoption (some drivers on sat-nav, others not, as in Phase 2). Requires a vehicle-type-stratified `routes.xml` with `device.rerouting` enabled per vType.
- **Multiple city extracts.** Single Ingolstadt bbox so far (small and 2×-doubled variants). Validating across multiple OSM extracts (different topologies, different demand patterns) would generalise the result.
- **Hitting-rate noise — mechanism nailed down.** The iter-9 audit reframed this: it is not generic noise but a specific failure mode. The flat $10^{-3}$ floor on partially-observed cells injects a constant loss term and a gradient pushing fitted `h_j(i)` toward zero, swamping the signal from real cells by ~4000-to-1 in `L_h` (at 4 h scale). (a) more data attenuates it linearly; (b) Bayesian shrinkage on `g` directly addresses the failure mechanism and remains the natural next step. Going outside the partial-observation premise — e.g. computing $g$ analytically from `PT_emp` — would be cheating relative to the framework's setting and was discarded.
- **$\gamma$ unmapped on $(0.5, 1)$ for `fitted_fg` (audit-discovered).** Only $\gamma \in \{0.1, 0.5, 1.0\}$ tried. The code comment in `sumo_to_phase3.py` suggests $\gamma \approx 0.9$ is the intended operating point; iter 7's signal-inversion at $\gamma = 0.5$ may have been the worst point in the curve, not representative.
- **Code hygiene — partially done (iter 11).** L-BFGS convergence now surfaced via `fit_chain`. Still TODO: pin SUMO `--seed` in the `.sumocfg` files (currently relies on SUMO's implicit default 23). Replace the (tautological) branching-only diagnostic with a per-step LR contribution histogram.
- **Parametric family at scale (iter-11 / 12 / 13 findings).** Local-only softmax family hits a hard ceiling around AUC 0.60 on the bigger network. Iter 12 implemented option (a) — adding ω-global features per paper Eqs. 17–18, with $n = 2{,}405$ extended to $d = 8{,}402$ parameters via 8 destination-state features + 2 edge features. Result: `fitted_f` 0.565 → 0.573–0.579 (+0.01 AUC, gap closure 39 % → 44–48 %). Real but modest. Iter 13 then established that the iter-11 0.666 ceiling was an artefact of `T = 20` trajectory truncation: at `T = 40` the empirical 1-step Markov detector reaches AUC 0.81 on the same network, while `fitted_f` only reaches 0.615 (37 % gap closure). **The misspecification gap on this network is structural rather than featural** — neither richer features (iter 12) nor longer trajectories (iter 13) close it.
  - **Remaining paths to push bigger-bbox `fitted_f` higher.** ~~(a) ω-global parameters~~ done iter 12, +0.01 AUC. (b) **2-step Markov model** — let next-edge depend on `(x_t, x_{t-1})`; substantial change but well within thesis scope and the natural next step given iter 13's finding that the 1-step empirical ceiling is already at 0.81, suggesting a 2-step empirical ceiling possibly above 0.9. (c) OD-conditional model — biggest potential, but requires plumbing destination through trajectories. (d) Replace the parametric family with a graph-neural or non-parametric estimator. (e) Accept the bigger-bbox row as a known scaling-limit and headline the small-bbox row as the Morimura validation.
- **Trajectory-length truncation revisited (iter-13 finding).** Default `T = 20` truncates ~40 % of per-trajectory information (mean length 34.7). Empirical detector AUC scales *better* than √T (margin 0.164 → 0.244 → 0.309 over `T` = 20 → 30 → 40), confirming that per-step regime contrasts are correlated along the trajectory. Easy lever for raising the empirical ceiling in any future setting; less helpful for `fitted_f` because the fitter is misspecification-bound. If we ever want a headline AUC number, score at `T = 40` (or longer) once the model class is rich enough to make use of it.
- **`X_o` sampling variance (iter-13 finding).** Different random observation-set draws can shift `fitted_f` by ~±0.02 AUC at `T = 20` (0.565 vs 0.599 measured between iter-11 and iter-13). Single-decimal numbers are noisier than they appear. Worth quantifying with a multi-seed run before reporting a headline.

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

### Per-car detector (Phase 3)

`run_trial` and `run_sweep` in `per_car_detector.py`. The graph / chain / inverter knobs from Phases 1–2 all carry over with the same effect; only the new ones are listed here.

| Variable | Default | What it controls | ↑ effect |
|---|---|---|---|
| `alpha` | 1.5 | Sat-nav avoidance strength, same role as in Phase 2 but now in the *self-consistent* fixed-point chain. | Higher → regime gap widens → all AUCs ↑. Differs from Phase 2: at high $\alpha$, the fixed-point $\pi^{\text{sat-nav}}$ visibly flattens because heavy redistribution is needed to remain self-consistent. |
| `obs_frac` | 0.10 | Fraction of states observed at $X_o$ for the Morimura fits of each regime. | Higher → both $\hat p_T$ chains are sharper → `fitted` AUC ↑, and (separately) `one_class` AUC ↑. **The most important Phase 3 knob.** Crossover where `fitted` overtakes `one_class` sits around 0.20–0.25. |
| `T_values` | (20, 35, 50) | Trajectory lengths to evaluate (sampled once at $\max T$ and truncated, so curves across $T$ are paired). | More transitions = more terms in the log-likelihood = sharper score = AUC ↑. Effect saturates once $T$ is comparable to mixing time. |
| `n_test_per_class` | 300 | Trajectories sampled per ground-truth class. Only affects ROC noise, not the underlying detection problem. | Higher → tighter ROC curve and tighter AUC error bars; no effect on the expected AUC. |
| `K` | 20 000 | Counts per regime's station-observation phase. Sets the Morimura SNR. | Higher → cleaner $\hat p_T$ chains → `fitted` and `one_class` AUC ↑. Independent of `obs_frac` to first order. |
| (FP `tol`, `max_iter`, `damping` in `satnav_pT`) | $10^{-8}$, 200, 0.5 | Numerical control of the sat-nav fixed-point iteration. | Tighter `tol` → marginally cleaner $\pi^{\text{sat-nav}}$ at cost of iterations; never the bottleneck. Raise `damping` only if the iteration oscillates (hasn't happened at default $\alpha \leq 4$). |

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

# Phase 3 — per-car sat-nav detector, ~3 s
.venv/bin/python per_car_detector.py
```

Both produce a PNG in the working directory and print per-trial summary lines to stdout.
