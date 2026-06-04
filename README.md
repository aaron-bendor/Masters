# Inverse Markov-Chain Routing Detector

Master's thesis repository for Aaron Bendor's Design Engineering project:
detecting externally influenced route choice from partially observed road-network
traffic, using inverse Markov-chain methods.

The project is organised around the same three-tier structure used in the
thesis:

| Tier | Purpose | Main entry points | Output |
|------|---------|-------------------|--------|
| **Tier 1** | Reproduce Morimura et al.'s synthetic inverse Markov-chain benchmark. | `morimura.py` | `morimura_fig.png`, thesis Fig. 1 |
| **Tier 2** | Validate the detector in labelled SUMO simulation with intrinsic versus sat-nav routing. | `sumo_validation/score_seed_big.py`, `sumo_validation/contrast_correlation.py` | SUMO AUCs, chain-recovery diagnostics, multi-seed logs |
| **Tier 3** | Stress-test the same detector on real Xuancheng AVI trajectories under behavioural proxy splits. | `real_data/load_xuancheng.py`, `sumo_validation/* --dataset xuancheng` | Xuancheng rush/off-peak and Labour-Day results |

The compiled report is in `thesis/main.pdf`; the LaTeX source is in
`thesis/`.

## Start Here

| Path | Role |
|------|------|
| `thesis/` | Final dissertation source, figures, bibliography and compiled PDF. |
| `morimura.py` | Tier 1 estimator implementation and synthetic reproduction driver. |
| `sumo_validation/` | Tier 2 SUMO experiments and Tier 3 scoring scripts. This folder is intentionally flat because the scripts import each other directly. |
| `real_data/` | Xuancheng AVI loader, SUMO network, dataset notes and local data layout. |
| `logs/` | Top-level run transcripts used to support thesis tables and figures. |
| `papers/` | Reference PDFs, including the Morimura et al. inverse-MC paper and the Xuancheng dataset paper. |
| `research_logbook.md` | Chronological project log: decisions, failed paths, retractions and final evidence trail. |
| `HANDOVER.md` | Detailed technical handover for continuing development. |

## Tier 1: Synthetic Reproduction

Tier 1 validates the inverse Markov-chain estimator in isolation. It recreates
the synthetic recovery task from Morimura et al. and compares recovered
stationary mass on unobserved states.

Core files:

| File | Purpose |
|------|---------|
| `morimura.py` | Softmax-parametric Markov-chain model, inverse estimator, analytic gradients, finite-difference-checked fitting and Tier 1 driver. |
| `morimura_compare.py` | One-off comparison/audit script retained for traceability. |
| `morimura_fig.png`, `morimura_compare_fig.png` | Generated plots from the reproduction and audit runs. |

Run:

```bash
.venv/bin/python morimura.py
```

## Tier 2: Controlled SUMO Validation

Tier 2 tests whether the estimator can separate labelled intrinsic routing from
sat-navigation routing in simulation. SUMO provides known regime labels, full
vehicle routes, edge counts and controlled demand/rerouting settings.

Core files:

| File | Purpose |
|------|---------|
| `sumo_validation/sumo_to_phase3.py` | Shared parser and feature module for SUMO networks, edge counts and vehicle trajectories. |
| `sumo_validation/score_seed_big.py` | Headline single-seed scorer for the larger Ingolstadt network. |
| `sumo_validation/contrast_correlation.py` | Adds per-row transition-contrast diagnostics to the fitted detector. |
| `sumo_validation/two_step_backoff.py` | Empirical 2-step Markov diagnostic with interpolation backoff. |
| `sumo_validation/score_rho_demand.py` | Demand-versus-rerouting decomposition used to audit the rush-hour proxy. |
| `run_multiseed.sh`, `run_multistart.sh` | Top-level dispatchers for multi-seed and multistart checks. |

Example run:

```bash
.venv/bin/python sumo_validation/contrast_correlation.py \
    --seed 42 --features real --T 40 --maxiter 1500
```

See `sumo_validation/README.md` for the full script index and file naming
conventions.

## Tier 3: Xuancheng Real-World Stress Test

Tier 3 applies the same scoring pipeline to real reconstructed AVI trajectories
from Xuancheng, China. There are no sat-nav ground-truth labels, so the thesis
uses behavioural proxy contrasts such as rush versus off-peak and Labour-Day
versus normal travel, with origin-destination matching.

Core files:

| File | Purpose |
|------|---------|
| `real_data/load_xuancheng.py` | Dataset loader, legality repair, regime splits and OD matching helpers. |
| `real_data/README.md` | Dataset inventory and verification notes. |
| `sumo_validation/score_seed_big.py --dataset xuancheng` | Reuses the Tier 2 scorer on Xuancheng trajectories. |
| `sumo_validation/bootstrap_auc.py` | Vehicle-level bootstrap confidence intervals. |
| `sumo_validation/xuancheng_2step_backoff.py` | Real-data counterpart to the Tier 2 2-step diagnostic. |

Example rush/off-peak run:

```bash
.venv/bin/python sumo_validation/contrast_correlation.py \
    --dataset xuancheng \
    --day 2023-04-17,2023-04-18,2023-04-19,2023-04-20,2023-04-21 \
    --regime_split rush_offpeak --od_match 8 \
    --features real --T 10 --maxiter 1500
```

Example Labour-Day stress test:

```bash
.venv/bin/python sumo_validation/contrast_correlation.py \
    --dataset xuancheng \
    --day 2023-04-28,2023-04-29,2023-04-30,2023-04-11,2023-04-12,2023-04-13,2023-04-14,2023-04-17,2023-04-18,2023-04-19,2023-04-20,2023-04-21 \
    --regime_split holiday_normal --od_match 64 \
    --features none --T 10 --maxiter 1500
```

The large daily Xuancheng JSON files are intentionally gitignored because they
exceed GitHub's file-size limit. See `real_data/README.md` for the expected
local layout.

## Thesis Build

From the repository root:

```bash
cd thesis
latexmk -pdf -synctex=1 -interaction=nonstopmode main.tex
```

The generated thesis PDF is `thesis/main.pdf`. The thesis appendix also contains
a concise code and reproducibility section.

## Environment

The project uses a local Python virtual environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy scipy matplotlib
```

Additional requirements:

| Tool/data | Needed for |
|-----------|------------|
| SUMO with `SUMO_HOME` set | Regenerating Tier 2 simulations. |
| LaTeX distribution with `latexmk`, `pdflatex` and `bibtex` | Rebuilding the thesis PDF. |
| Xuancheng AVI daily JSON files | Re-running Tier 3 from raw real-world trajectories. |

## What Is Core vs Exploratory

The core thesis evidence path is:

1. `morimura.py`
2. `sumo_validation/score_seed_big.py`
3. `sumo_validation/contrast_correlation.py`
4. `sumo_validation/two_step_backoff.py`
5. `real_data/load_xuancheng.py`
6. `sumo_validation/bootstrap_auc.py`
7. `sumo_validation/xuancheng_2step_backoff.py`
8. `thesis/`

The root-level `congestion_filter.py` and `per_car_detector.py` files are
earlier synthetic detector prototypes. They are retained because they document
how the final three-tier design evolved, but they are not the main evidence
path for the submitted thesis.

## Reproducibility Notes

- Generated Python caches, the virtual environment, macOS metadata, oversized
  Xuancheng JSON files and oversized SUMO route files are ignored by
  `.gitignore`.
- `logs/README.md`, `sumo_validation/README.md` and `real_data/README.md`
  explain where the long-run outputs and data assumptions live.
- `research_logbook.md` is the best place to audit why each experiment was run
  and which earlier interpretations were revised.
