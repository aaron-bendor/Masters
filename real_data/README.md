# real_data/ — Xuancheng AVI dataset (Ma et al. 2026)

Source: Figshare DOI 29925824 (companion to `../papers/s41597-026-06892-2.pdf`).

## Layout

| Path | What it is |
|------|------------|
| `xuancheng.net.xml` | SUMO network. 1,744 edges (1,733 in passenger SCC used by the loader). |
| `Data/cfg/Xuancheng/` | SUMO config templates from the dataset authors. |
| `Data/data/Xuancheng/` | 30 daily JSON files `data_2023_04_<DD>_type_filtered.json`, each 150–225 MB, ~327k trips/day, ~10.5M trips total. Each trip carries `(vehicle, interval, startTime, endTime, route)` with `route` as an ordered list of edge IDs. **Gitignored** (over GitHub's 100 MB hard limit). |
| `Data/{cfg,data}/{Hangzhou,Manhattan,Jinan,Nanchang}/` | Other cities in the same release — kept for completeness, not used. |
| `load_xuancheng.py` | The loader. Entry point: `load_xuancheng_regime(net_path, day_specs, regime_split, label_a, label_b)` returns the same 7-tuple `(edges, adj_out, idx, trajs_a, trajs_b, f_a, f_b)` that the Tier 2 SUMO scripts already consume. Also exposes `split_rush_vs_offpeak`, `split_weekday_vs_weekend`, `split_holiday_vs_normal`. |
| `61700545_Documentation.pdf`, `61742701_Documentation.pdf` | Figshare-side dataset documentation. |

## Verified properties of the dataset (iter 20 audit)

- Route IDs match SUMO edge IDs 1:1 (1,034/1,034 unique IDs in first 2k trips). No ID mapping required.
- `startTime` is **seconds since midnight**, range 0..86,400, canonical AM/PM double peak. Tier 3 rush/off-peak split: rush = [25200, 32400] ∪ [61200, 68400], off-peak = [36000, 57600].
- 78.4% of consecutive edge pairs are legal SUMO transitions. The remaining 21.6% are Dijkstra-repaired teleports from the dataset's source pipeline — the loader splits trips at illegal boundaries into legal sub-trajectories rather than dropping.
- Sub-trajectory mean length 8.2 edges (vs 15.1 raw). Effective `T` for Tier 3 experiments is therefore ~5–7 transitions.
- **Apr 8 and Apr 9 are duplicate files** (identical 324,441 trip counts) — use only one.
- **Apr 10 has `startTime` max ≈ 165,071 s** (~2 days) — file may concatenate; exclude or investigate.
- Apr 5 (Tomb-Sweeping), Apr 28–30 (Labour Day period) are holiday-vs-normal contrast candidates. The loader excludes Apr 10 from the normal-day set because of the `startTime` anomaly.

## How Tier 3 scripts use this

The `--dataset xuancheng` CLI flag in `sumo_validation/score_seed_big.py`, `contrast_correlation.py`, `score_seed_big_multistart.py`, and `bootstrap_auc.py` routes through `load_xuancheng_regime`. Required args: `--day YYYY-MM-DD[,YYYY-MM-DD,...]`, `--regime_split {rush_offpeak,weekday_weekend,holiday_normal}`, optional `--od_match K` for spatial OD-zone matching with K-means.
