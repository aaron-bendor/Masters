# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Master's research extending Morimura, Osogami & Idé (NeurIPS 2013), *Solving inverse problem of Markov chain with partial observations*, to detect and filter observations corrupted by external re-routing influence (sat-navs reacting to congestion) so the recovered chain reflects drivers' intrinsic preferences. Reference PDFs live in `papers/` (Morimura is `papers/NIPS-2013-...-Paper.pdf`); `research_logbook.md` is the running narrative of decisions and results.

## Environment

Python 3.14 in `.venv/`. Dependencies: numpy, scipy, matplotlib (no project-level dependency file — install with `.venv/bin/pip install numpy scipy matplotlib` if recreating).

Each Bash tool call gets a fresh shell, so `source .venv/bin/activate` does not persist. Invoke the venv's interpreter directly:

```
.venv/bin/python morimura.py
.venv/bin/python congestion_filter.py
.venv/bin/python per_car_detector.py
```

Each script's `__main__` runs both a single-config demo and a parameter sweep and writes one or two PNGs next to the source (e.g. `morimura_fig.png`, `congestion_filter_sweep.png`). Runs are slow — minutes for the sweeps — because each cell refits the inverter several times across trials. Phase-1 sweep at the iter-18 paper-faithful config (n=100, globals, λ CV, 10 trials) is ~22 minutes.

## Architecture

Three scripts arranged as a strict dependency chain. Anything that changes the underlying chain model in `morimura.py` propagates to the other two.

### `morimura.py` — Phase 1 reproduction

Self-contained reproduction of §6.1 of the paper. The pieces that the other scripts re-use:

- **`make_truth`** — builds a synthetic ground-truth chain: random strongly-connected graph (`random_graph`), softmax-parametric `pI` and `pT`, then mixed 70/30 with Dirichlet(0.3) noise to push truth slightly outside the parametric family. Optional `d_T, d_psi` parameters add random N(0,1) global features in the truth, matching paper §6.1's recipe; returns `(adj_out, pI, PT, P, pi, phi_T, psi)` (7-tuple). Pass `phi_T, psi` to `Inverter` for matching-capacity recovery.
- **`stationary(P)`** — solves `π^T P = π^T` by replacing the last balance equation with the normalisation constraint.
- **`Inverter`** — fits `theta = [nu_loc, omega_loc, (omega_glo1, omega_glo2)]` by minimising the regularised objective (Eq. 7). Implements analytic gradients of `log π` (Eq. 12, `grad_log_pi`) and `log h_θ(j)` (Eq. 15, `grad_log_h`), then hands `loss_grad` to `scipy.optimize.minimize(method="L-BFGS-B")`. `gamma=1.0` is stationary-only (f-only fit); `gamma<1.0` mixes in the hitting-rate loss. Globals are optional: pass `phi_T` (n × d_T) and/or `psi` (E × d_psi) to enable the paper's Eqs. 17–18 ω-global terms; default `None` reproduces the local-only behaviour exactly. Gradients FD-verified to ~5e-9 relative error in both local-only and globals modes.
- **`true_g`** — exact hitting-rate matrix on `X_o × X_o` used to give the inverter clean `g` observations in the synthetic experiments.

`beta` is held at the true value during inversion to dodge a known identifiability issue when only stationary stats are observed. The local-only configuration is the Phase-1 default (matches the paper's synthetic experiment §6.1); ω-globals are used by Phase 4 on SUMO data per the paper's real-world precedent (§6.2 Nairobi).

### `congestion_filter.py` — Phase 2 (snapshot detection + filtered refit)

Imports `make_truth`, `stationary`, `Inverter`, `softmax` from `morimura`. Generates **T snapshots** under a two-regime mixture (uncongested vs. congested) where the congested rate blends `pi_intr` with `pi_infl` (sat-nav adoption fraction `rho`). The influenced transition kernel in `influenced_pT` uses congestion derived from `pi_intr` (not self-consistent — see Phase 3 for the contrast).

A subset of uncongested snapshots is **labelled**; the rest are unlabelled. Each snapshot gets an anomaly score `kl_score = KL(empirical || empirical-intrinsic-from-labelled)` at `X_o`. KL on distributions (not Poisson rates) so the total-counts nuisance cancels. Compares three intrinsic-chain recovery strategies — `naive`, `oracle`, `filtered` — by RMAE on `pi_intr` at unobserved states.

### `per_car_detector.py` — Phase 3 (per-trajectory log-likelihood ratio)

Imports from both `morimura` and `congestion_filter` (`roc`). Key distinction from Phase 2: the sat-nav chain `pT_satnav` is **self-consistent** — congestion is derived from `pi_satnav` itself, solved by damped Picard iteration in `satnav_pT`. Detects whether a single trajectory came from `pT_intr` vs. `pT_satnav` via `Λ(τ) = sum_t [log pT_satnav_hat(x_{t+1}|x_t) − log pT_intr_hat(x_{t+1}|x_t)]`. Reports four detectors: `fitted_f` (f-only fit), `fitted_fg` (f+g, headline), `oracle` (true chains), `one_class` (anomaly under intrinsic only).

### Conventions worth knowing before editing

- All three scripts use the same softmax-parametric chain and the same `make_truth` seeding contract — keep `np.random.default_rng` plumbing consistent if you add new entry points.
- `Inverter.gamma` defaults differ by use: 1.0 for f-only (`congestion_filter.fit_intrinsic`, the no-g sweep in `morimura.run`), 0.1 when `g` is available (`per_car_detector.fit_chain`).
- Sweep drivers (`run_sweep_2d`, `run_sweep`) are wall-clock dominated by refits; if iterating on plotting only, run the single-config block and skip the sweep.
