# sumo_validation/ — Phase 4 (simulated) and Phase 5 (real Xuancheng) scripts and data

This directory is one big flat namespace because almost every Python script imports `sumo_to_phase3` at the top, and the `dispatch_*.sh` scripts assume scorers sit next to their data. Don't move files into sub-folders without re-plumbing the imports.

## Scripts — fitting / scoring

| Script | Purpose | Used by |
|--------|---------|---------|
| `sumo_to_phase3.py` | Foundational module: builds `(edges, adj_out, idx)` from `net.net.xml`, parses `edgedata`/`vehroutes`, extracts the 8-D `phi_T` + 2-D `psi` real features, defines `empirical_chain` / `empirical_g` / `_scores_branching`. Every other script imports from this. | All. |
| `score_seed_big.py` | Phase-4 headline scorer. Loads bigger-bbox SUMO data for one `--seed`, fits intr+satnav chains via Morimura, prints `empirical / fitted_f / gap_closed`. Now `--dataset {sumo,xuancheng}` aware. | `run_seed_big.sh`, dispatch scripts. |
| `score_seed.py` | Small-bbox variant (iter-15). | `run_seed.sh`. |
| `score_seed_big_fg.py` | f+g attempt with `γ` and `g`-shrinkage CLI flags. Computationally infeasible at SUMO bigger-bbox scale (iter 19E, iter 29). | manual. |
| `score_seed_big_lm.py` | Levenberg–Marquardt fit via `scipy.optimize.least_squares(method='trf')`. Aborted as impractical at d=8392 (iter 17). | manual. |
| `score_seed_big_multistart.py` | K random restarts per chain, validation-selected. Confirms low basin not escapable on SUMO (iter 24); Xuancheng OD-matched basin IS escapable (iter 25) but result still at chance. | `run_multistart.sh`. |
| `score_rho_demand.py` | Scorer for the (demand × ρ) decomposition cells (iter 27). | `dispatch_rho_demand_lean.sh`. |
| `contrast_correlation.py` | Replicates `score_seed_big` then computes per-row contrast Pearson / cosine — the "is the recovered chain right pointwise?" diagnostic. Load-bearing for the iter-18 / iter-19 / iter-23 chain-recovery framing. | manual + multiseed. |
| `bootstrap_auc.py` | Vehicle-level stratified bootstrap CIs over a scored test set. K-sweep entry point (iter 26). | manual. |
| `two_step_empirical.py` | Counts `(x_{t-1}, x_t) → x_{t+1}` triples, scores LR vs 1-step empirical. The 2-step diagnostic. | iter 19C; iter 28 baseline. |
| `two_step_backoff.py` | Jelinek–Mercer interpolation backoff on 2-step empirical. Best Phase-4 lift (+0.10 AUC, 5/5 seeds). | `dispatch_2step_backoff_sweep.sh` (iter 28). |
| `fg_xuancheng_subgraph.py` | Restricts to one K-zone subgraph then runs f+g sparse-LU. Used for the γ-sweep that found f+g works at subgraph scale with γ ∈ [0.9, 0.95] (iter 29). | `dispatch_subgraph_gamma_zone_sweep.sh`. |

## Scripts — sweeps / safety gates

| Script | Purpose |
|--------|---------|
| `sweep_lam_big.py` | Tikhonov λ sweep — touched briefly in iter 17, not used in headline numbers. |
| `small_T_sweep.py` | Small-bbox T-sweep driver (iter 14). |
| `lm_fit.py`, `lm_fd_real.py` | Levenberg–Marquardt residual + Jacobian helpers; FD-checked. Currently inactive. |
| `fd_check_features.py` | FD gradient check at the SUMO+features parameterisation. Always run before any features-enabled sweep. |
| `fd_check_fg_sparse.py` | FD check on the sparse-LU `hitting_pack_sparse` path. Used to validate the iter-29 implementation. |
| `time_fg_sparse_sumo.py` | One-off timing harness for the iter-29 dense-vs-sparse comparison. |

## Scripts — aggregation

| Script | Purpose |
|--------|---------|
| `aggregate_2step_sweep.py` | Multi-seed parser + best-config selector for the iter-28 sweep. |
| `aggregate_rho_demand_lean.py` | Decomposes the rho×demand cells into sat-nav vs congestion contributions. |

## Shell dispatchers

`dispatch_*.sh` scripts iterate over seeds/configs and invoke a Python scorer per cell, piping `RESULT` lines back to stdout. `run_*.sh` scripts run a single seed end-to-end (randomTrips → duarouter → SUMO ×2 → score). All assume `../.venv/bin/python` is the active interpreter and use `set -euo pipefail`.

## Data — by name

Filename conventions:
- `net.net.xml` — bigger bbox (≈3.7×3.3 km Ingolstadt); 2,405 edges in passenger SCC.
- `net.small.net.xml` — small bbox (≈2×2 km); 686 edges in passenger SCC.
- `city_bbox.osm.xml`, `city_bbox.small.osm.xml` — raw OSM extracts both networks came from.
- `trips.<tag>.xml` → `routes.<tag>.xml` → `vehroutes.<tag>.xml`, `edgedata.<tag>.xml`, `stats.<tag>.xml` — one SUMO run's outputs at tag `<tag>`.
- `*.alt.xml`, `*.add.xml`, `*.sumocfg` — SUMO config files (alternates and additional-files for edgedata aggregation).

Tags you'll see:
- `intr` / `satnav` — original Phase-4 small-bbox runs (iter 1–9).
- `intr.big.seedN` / `satnav.big.seedN` — bigger-bbox 5-seed sweep (iter 16). N ∈ {7, 23, 42, 101, 2024}.
- `intr.bigbox.pX` / `satnav.bigbox.pX` — early bigger-bbox demand sweep (iter 10–11). X ∈ {0.25, 0.5, 1.0}.
- `intr.small.seedN` / `satnav.small.seedN` — small-bbox 5-seed sweep (iter 15).
- `d<DEM>.r<RHO>.seed<N>` — the iter-27 ρ×demand grid. `d` ∈ {0.3 = peak, 0.8 = off-peak}; `r` ∈ {0.0, 0.5, 1.0} = sat-nav adoption fraction. 15 cells in the lean sweep, ~30 with scoping extras.

The vehroutes files for `d0.3.r1.0` are gigabytes each (saturating peak demand with full rerouting generates huge `routeDistribution` blocks) and are explicitly gitignored.

## Where the numbers in the thesis live

- Tier-2 headline AUC numbers → `score_seed_big.py` / `contrast_correlation.py` output.
- Tier-2 multi-seed table → `multiseed_features.log` (in repo-root `logs/`).
- Tier-2 multistart → `multistart_lowbasin.log`.
- Tier-2 2-step backoff (the iter-28 OVERTURN) → `logs/2step_backoff_sweep.log` here.
- Tier-2 ρ×demand decomposition → `logs/rho_demand_lean.log` here.
- Tier-3 Xuancheng → `xuancheng_*.log` in repo-root `logs/`.
- Tier-3 K-sweep RETRACTION → `logs/xuan_Ksweep.log` here.
- Tier-3 f+g subgraph win → `logs/subgraph_gamma_zone_sweep.log` here.

For full prose context, every iter is logged in `../research_logbook.md`.
