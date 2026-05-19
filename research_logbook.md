# Research logbook

## Project

Extending Morimura, Osogami & Idé (NeurIPS 2013), *Solving inverse problem of Markov chain
with partial observations*, to detect and filter out observations corrupted by external
re-routing influence (e.g. sat-navs reacting to congestion), so that the recovered chain
reflects drivers' intrinsic preferences.

## Files

| File | Purpose |
|------|---------|
| `NIPS-2013-...-Paper.pdf` | The reference paper (Morimura, Osogami, Idé 2013). |
| `Identification_of_new_patterns_v3.pdf` | Gu, Crisostomi, Liu, Shorten (2018). Population-level junction-turning anomaly detection — the conceptual reference for Phase 3. |
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

> **Iter 5 — raising observation coverage closes most of the gap.** With $n = 686$, the original 10 % coverage gave only 68 observed edges. Raised `obs_frac` to 0.25 ($\lvert X_o\rvert = 171$): `fitted_f` climbed 0.531 → 0.649, recovering about 60 % of the gap to the empirical ceiling. The diagnosis flipped cleanly from "structural limit" to "fittable, given more observation".

> **Iter 6 (in progress at time of writing) — fitted_fg with empirical hitting rates.** Implemented `empirical_g`, the trajectory-based analog of `morimura.true_g`: for each $(i, j) \in X_o \times X_o$, estimate the discounted first-hit probability of $j$ from $i$ across training trajectories, floored at $10^{-3}$ to keep `log(g)` finite in the inverter. Fitting with $\gamma = 0.1$ mirrors the Phase 3 toy's headline detector. Run time at $\lvert X_o\rvert = 171$ is ~1–2 hours per pair because each L-BFGS gradient step performs 171 LU factorisations of $(I - \beta P_T^{\setminus j})$. Result pending.

### Current state

`sumo_validation/sumo_phase3_fig.png`. Numbers at $n = 686$, $\lvert X_o\rvert = 171$, $T = 20$, 300 trajectories per class:

| Detector | AUC | Note |
|---|---|---|
| empirical (full obs, all steps) | 0.721 | Per-source-state Laplace-smoothed MLE from training trajectories. The 1-step-Markov ceiling on this data. |
| empirical (full obs, branching only) | 0.721 | Forced-transition steps masked out. No gain over all-steps — signal is uniform across the trajectory. |
| `fitted_f` (γ = 1, all steps) | 0.649 | Morimura f-only at 25 % coverage. ~60 % of the gap closed vs. iter 4. |
| `fitted_f` (branching only) | 0.649 | Same as empirical: no separation between branching-step and forced-step contributions. |
| `fitted_fg` (γ = 0.1, all steps) | *pending* | Morimura f+g with empirical $g$. Expected to close most of the remaining 0.07 to the ceiling. |

**Headline read.** The Phase 3 framework recovers a real rerouting signal from SUMO data once two conditions are met: the network has realistic topology (no aggressive road-class trimming) and observation density is on the order of 25 % rather than the 10 % the synthetic toy used. Whether `fitted_fg` matches the empirical ceiling — making the validation positive in the same way the toy was — is the open question being resolved by the current run.

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

# 2. Demand
python3 $SUMO_HOME/tools/randomTrips.py -n net.net.xml \
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
- **Demand sweep.** `-p 0.5` was chosen by trial-and-error (iters 3, 5). A proper sweep over demand × rerouting-fraction — the SUMO analog of Phase 2's $\alpha \times \rho_c$ — would establish the operating regime where the detector is strongest.
- **Mixed sat-nav adoption.** Currently 0 % vs. 100 %. The realistic case is mixed adoption (some drivers on sat-nav, others not, as in Phase 2). Requires a vehicle-type-stratified `routes.xml` with `device.rerouting` enabled per vType.
- **Multiple city extracts.** Single Ingolstadt bbox so far. Validating across multiple OSM extracts (different topologies, different demand patterns) would generalise the result.
- **Hitting-rate noise — answered here.** Phase 3's open thread about $g$ being "too clean" is implicitly addressed by Phase 4: `empirical_g` is finite-sample noisy, and `fitted_fg` here is therefore the realistic-noise version of the toy's exact-$g$ detector. If `fitted_fg` matches the empirical ceiling in iter 6, the framework survives that noise.

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
