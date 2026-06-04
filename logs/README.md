# logs/ — captured stdout from long tier and exploratory runs

These are run transcripts, not data. Each was produced by piping a sweep dispatcher or a single fit to `tee` while it ran. The numbers they contain have been distilled into `research_logbook.md`; the raw logs are kept for traceability.

| File | Source | What's in it | Logbook iter |
|------|--------|--------------|--------------|
| `T_sweep_s42.log` | `contrast_correlation.py --features real --T {20,30,40}` on bigger-bbox seed 42 | Per-T fit losses + AUC + Pearson; the T-sweep table in iter-19. | iter 19 (D) |
| `fg_features_s42.log` | `score_seed_big_fg.py` | First f+g attempt at \|X_o\|=601, killed at 9.5 h. | iter 19 (E) |
| `fg_light_s42.log` | `score_seed_big_fg.py --obs_frac` lighter | Second f+g attempt at \|X_o\|=120, also killed. | iter 19 (E) |
| `phase4_features_ab_` | stray FD-check redirect | FD verification of `Inverter.loss_grad` with features on. Output has no extension because the redirect target was truncated. PASS, 0 failing dims. | iter 19 (A) |
| `phase4_features_ab_s42.log` | `contrast_correlation.py --features {none,real}` seed 42 v1 | First single-seed features comparison. | iter 19 (B) |
| `phase4_features_ab_s42_v2.log` | same, rerun with FD gate | Same numbers, fuller context. The reported iter-19 table uses these. | iter 19 (B) |
| `two_step_s42.log` | `two_step_empirical.py` seed 42 | 2-step empirical refutation: AUC_2step=0.666 < AUC_1step=0.685. | iter 19 (C) |
| `xuancheng_first_s17.log` | `score_seed_big.py --dataset xuancheng --day 2023-04-17` | First Xuancheng single-day fit. | iter 21 (A) |
| `xuancheng_pooled_5d.log` | same, `--day 2023-04-17,...,2023-04-21` | Pooled 5-day fit. | iter 21 (B) |
| `xuancheng_od8_pooled.log` | same, `--od_match 8` | The headline OD-matched run. Ceiling crash 0.573 → 0.518. | iter 21 (C) |
| `multiseed_features.log` | `run_multiseed.sh` | 5-seed × 2-features sweep, ~5.3 h. Loaded the iter-23 table. | iter 23 |
| `multistart_lowbasin.log` | `run_multistart.sh` | K=5 restart sweep on seeds 7 & 23. Confirms low basin not escapable. | iter 24 |
| `multistart_xuancheng_odmatch.log` | `score_seed_big_multistart.py --dataset xuancheng --od_match 8 --features real` | K=5 multistart on Xuancheng. Confirms 0.530 > 0.518 ceiling violation was a basin artefact. | iter 25 |

To re-create a log, rerun the corresponding dispatcher/script from `sumo_validation/` and pipe to `tee logs/<name>.log`. The dispatchers don't hard-code log paths; the user supplies the redirect.
