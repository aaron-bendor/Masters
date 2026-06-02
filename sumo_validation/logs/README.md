# sumo_validation/logs/ — sweep transcripts

Same idea as `../logs/` but for sweeps that ran inside `sumo_validation/` (most often via a `dispatch_*.sh` script that calls a Python scorer per seed/config and prints `RESULT` rows).

| File | Dispatcher / source | Headline | Logbook iter |
|------|---------------------|----------|--------------|
| `2step_alpha_sweep.log` | `dispatch_2step_alpha_sweep.sh` | Dirichlet-smoothing α sweep on `two_step_empirical.py`. Best Δ = +0.021 (α=1e-3). | iter 28 |
| `2step_backoff_sweep.log` | `dispatch_2step_backoff_sweep.sh` | Interpolation backoff for 2-step. **Best Δ = +0.100** (κ=100, all 5 seeds win). Overturns §6.5 negative. | iter 28 |
| `rho_demand_lean.log` | `dispatch_rho_demand_lean.sh` | ρ×demand decomposition: ~96% of rush-vs-off-peak signal attributable to sat-nav, not congestion. | iter 27 |
| `rho_demand_scope.log` | `dispatch_rho_demand_scope.sh` | Single-seed (s42) scoping run for ρ-linearity check. | iter 27 (bonus) |
| `rho_demand_scope_scores.log` | `dispatch_score_rho_demand_scope.sh` | Score-only re-run of `rho_demand_scope`. | iter 27 (bonus) |
| `subgraph_gamma_zone_sweep.log` | `dispatch_subgraph_gamma_zone_sweep.sh` | γ × zone sweep on Xuancheng subgraph; γ=0.95 lifts f+g to 0.553 ≈ empirical 0.563. | iter 29 (D, E) |
| `fg_xuancheng_subgraph.log` | `fg_xuancheng_subgraph.py` | Single-zone f+g sanity-check (Xuancheng zone 0, n=194 SCC). | iter 29 (C) |
| `time_fg_sparse_sumo.log` | `time_fg_sparse_sumo.py` | Timing: sparse-LU on SUMO bigger bbox is ~8× SLOWER than dense (wide-RHS solve, not factorisation, is the bottleneck). | iter 29 (B) |
| `boot_xuan_odmatch.log` | `bootstrap_auc.py` | Vehicle-level bootstrap CIs on the Xuancheng OD-matched run. | iter 26 |
| `xuan_Ksweep.log` | `bootstrap_auc.py` K-sweep | K ∈ {8,16,32,64} OD-zone granularity sweep. **Retracts the "~75% OD-mix" framing** — K=8 artefact. Robust finding: fitted detector at chance for all K. | iter 26 |
| `sumo.intr.log`, `sumo.satnav.log` | raw SUMO simulator output | Generated when running `sumo -c intr.sumocfg` (or `satnav.sumocfg`) headless. Kept once for sanity-check; not used downstream. | iter 1+ |

Re-creating any of these: cd into `sumo_validation/`, run the matching `dispatch_*.sh` or Python script, redirect to `logs/<name>.log`. Most sweeps take 1 min – 5 h depending on whether they refit chains.
