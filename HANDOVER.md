# HANDOVER.md

A pickup-where-the-last-one-left-off briefing for an AI continuing this Master's project. Read this first; everything else is reachable from here.

---

## 1. What this project is

Aaron Bendor's MEng Design Engineering thesis at Imperial College London. The thesis extends **Morimura, Osogami & Idé (NeurIPS 2013), *Solving inverse problem of Markov chain with partial observations*** to a new question: *can we detect, on a road network, the fraction of route choices that were rerouted by an external influence (sat-navs reacting to congestion)?* The thesis is in five experimental tiers; the first reproduces the paper, the next four are new contributions.

The thesis itself is in `thesis/`. First draft was complete on 2026-05-28; revisions and final compile have been happening since.

### The five tiers, with one-line current state

| Tier | Subject | Script(s) | Headline | Logbook |
|------|---------|-----------|----------|---------|
| **Phase 1** | Reproduce paper §6.1 synthetic experiment | `morimura.py` | RMAE tracks paper Fig. 2A within ~0.05 at `\|X_o\| ∈ [10, 90]`, `n=100`, 10 trials, full globals + CV. **Locked in.** | iter 18 |
| **Phase 2** | Snapshot-level detection + filtered intrinsic-chain refit (synthetic) | `congestion_filter.py` | Filter ≈ oracle when regime gap is big and SNR good; collapses to naive in low-SNR. Not the central finding. | n/a (one-shot) |
| **Phase 3** | Per-car LR detector (synthetic, self-consistent sat-nav fixed point) | `per_car_detector.py` | `fitted_fg` is the headline; AUC up with `α` and `obs_frac`. One-shot. | n/a |
| **Phase 4** | Validation on SUMO simulated traffic | `sumo_validation/sumo_to_phase3.py` and successors | `fitted_f` AUC ≈ 0.58 ± 0.03 across 5 seeds on bigger bbox (∼50% gap closure). Features add per-row contrast (Pearson +0.05, all 5 seeds, p≈0.008) but **not** robust AUC. 2-step memory carries +0.10 AUC with interpolation backoff (5/5 seeds, iter 28 — OVERTURNS the original §6.5 negative). ρ×demand decomposition: 96% of rush-vs-off-peak SUMO signal is sat-nav, 0% is congestion-rerouting of unaided drivers. | iters 1–19, 23–24, 27–29 |
| **Phase 5** | Validation on real-world Xuancheng AVI dataset (Ma et al. 2026) | `sumo_validation/score_seed_big.py --dataset xuancheng` | OD-matched empirical ceiling **0.518** (only +0.018 above chance); fitted detector at chance for K ∈ {8, 16, 32, 64} (iter 26 K-sweep retracted the "75% OD-mix" framing). Tier-3 negative result is granularity-robust. | iters 20–22, 25–26 |

The narrative of how the project got here is in `research_logbook.md` — 1100 lines, written contemporaneously, contains the surprises and reversals (iter 9 teleport artefact, iter 14 retracted by iter 15, iter 21 "75% OD-mix" retracted by iter 26, iter 28 overturning §6.5).

---

## 2. Repo map

```
DESENG/Masters/
├── HANDOVER.md              ← you are here
├── CLAUDE.md                project instructions for the Claude Code harness (terser than this; partly overlapping)
├── research_logbook.md      THE running narrative. 5 phases × every iteration. Read this for "why" and "what we tried".
├── README equivalents in every subfolder (logs/, papers/, real_data/, sumo_validation/, sumo_validation/logs/, thesis/)
│
├── morimura.py              Phase 1.  ~640 lines.
├── congestion_filter.py     Phase 2.  Imports from morimura.
├── per_car_detector.py      Phase 3.  Imports from morimura + congestion_filter.
├── morimura_compare.py      One-off comparison script (iter 18); produces morimura_compare_fig.png. Throwaway.
├── morimura_fig.png         current Phase-1 plot
├── congestion_filter_fig.png, congestion_filter_sweep.png   Phase-2 plots
├── per_car_detector_fig.png, per_car_detector_sweep.png      Phase-3 plots
├── run_multiseed.sh, run_multistart.sh   Top-level Phase-4 dispatch
│
├── sumo_validation/         Phase 4 + 5.  See sumo_validation/README.md for the script index.
│   ├── README.md            ← read this before editing scripts in here
│   ├── logs/                sweep transcripts; logs/README.md indexes them
│   ├── sumo_to_phase3.py    foundational module (everything imports it)
│   ├── score_seed_big.py    headline Phase-4 scorer (also Phase-5 via --dataset xuancheng)
│   ├── contrast_correlation.py   per-row chain-recovery diagnostic
│   ├── ... 19 other Python scripts (see sumo_validation/README.md)
│   ├── dispatch_*.sh        sweep dispatchers
│   ├── run_*.sh             single-seed end-to-end runners (randomTrips → duarouter → SUMO×2 → score)
│   ├── net.net.xml (bigger bbox, 2,405-edge SCC), net.small.net.xml (small bbox, 686-edge SCC)
│   └── ~250 SUMO data files (trips/routes/vehroutes/edgedata/stats per seed/config — see naming convention in sumo_validation/README.md)
│
├── real_data/               Xuancheng AVI dataset (Ma et al. 2026); real_data/README.md has the inventory + dataset audit findings
│   ├── load_xuancheng.py    the loader; exposes load_xuancheng_regime + 3 split functions
│   ├── xuancheng.net.xml    SUMO network for the dataset
│   └── Data/data/Xuancheng/data_2023_04_*_type_filtered.json   30 daily JSON files (~10.5M trips total). Gitignored over 100MB.
│
├── thesis/                  LaTeX project. thesis/README.md has build instructions.
│   ├── main.tex             top-level
│   ├── sections/            11 chapter files (00_abstract.tex … 10_conclusion.tex, plus A_appendix.tex)
│   ├── figures/             4 hand-keyed PNGs + make_figures.py
│   ├── references.bib       IEEEtran-style bibliography, 16 citations
│   └── main.pdf             current compiled output (tracked)
│
├── papers/                  Reference PDFs; papers/README.md indexes them
├── logs/                    Top-level run transcripts (T sweep, features comparison, Xuancheng days, multiseed, multistart); logs/README.md indexes them
├── masters_thesis_examples/ Three example dissertations (Aditya Munot, Antanas Zilinskas, Nora Luo) — formatting/voice/length references
│
├── .venv/                   Python 3.14 venv with numpy/scipy/matplotlib (the only deps)
└── .gitignore               oversized SUMO outputs, Xuancheng JSON, __pycache__, .DS_Store, .venv, LaTeX intermediates
```

---

## 3. The script dependency chain

```
   morimura.py
        ├── softmax, make_truth, stationary, Inverter, hitting_pack, hitting_pack_sparse, true_g
        │
        ├──► congestion_filter.py     uses make_truth + Inverter + softmax + stationary
        │         └── fit_intrinsic, generate_two_regime, kl_score, roc
        │
        └──► per_car_detector.py      uses make_truth + Inverter + softmax + roc
                  ├── satnav_pT (damped Picard FP iteration)
                  ├── fit_chain (gamma=0.1 default, f+g)
                  └── _scores_for_chains, _scores_one_class

   sumo_validation/sumo_to_phase3.py  uses per_car_detector.fit_chain + congestion_filter.roc
        └── build_adj, extract_features (phi_T 8-D + psi 2-D), read_edge_counts, read_trajectories,
            empirical_chain, empirical_g, _scores_branching

   sumo_validation/<everything else>.py  imports sumo_to_phase3 at the top

   real_data/load_xuancheng.py  used by sumo_validation/* via --dataset xuancheng
```

**Anything that changes `Inverter` in `morimura.py` propagates everywhere.** FD-check before touching it (`sumo_validation/fd_check_features.py` for SUMO-features mode; `sumo_validation/fd_check_fg_sparse.py` for the sparse-LU f+g path).

---

## 4. Where to find what

### "What does Morimura's paper actually say?"
`papers/NIPS-2013-...-Paper.pdf`. §6.1 = the synthetic experiment we reproduce; §6.2 = the Nairobi real-world example (f-only because g unavailable; the precedent we cite when defending our own f-only choice on SUMO).

### "What is the current Phase-1 reproduction quality?"
`research_logbook.md` line ~96 — RMAE table at `n=100`, 10 trials, full globals + CV. Headline: "Proposed (with g)" tracks paper Fig. 2A within ~0.05 RMAE for `|X_o|` 10–90. `morimura_fig.png` shows the plot. The reproduction was re-audited and re-locked in iter 18 (logbook line ~503).

### "What is the current Phase-4 SUMO headline?"
- Defensible single number: `fitted_f` AUC = 0.575 ± 0.024 across 5 SUMO seeds on bigger bbox (`research_logbook.md` line ~463 — iter 16 table; expanded in iter 23 line ~670).
- Real features lift: +0.05 Pearson per-row contrast (5/5 seeds, p≈0.008), but NOT robust AUC (reverses on seed 101). Iter 23 line ~670.
- 2-step backoff: +0.100 AUC, 5/5 seeds. **The largest robust Phase-4 effect.** Iter 28 line ~752.
- ρ×demand decomposition: 96% of rush-vs-off-peak signal is sat-nav. Iter 27 line ~718.

### "What is the current Phase-5 Xuancheng headline?"
OD-matched empirical ceiling = 0.518 (only 0.018 above chance). Fitted detector at chance for K ∈ {8, 16, 32, 64}. The original "~75% OD-mix" framing from iter 21 was **retracted in iter 26** as a K=8 artefact. Robust finding: at all K, the 1-step partial-obs detector fails to recover the within-OD rush-vs-off-peak contrast that full observation weakly shows. Iter 26 line ~709.

### "Where is the f+g feasibility story?"
- **Negative on full SUMO**: iter 19 (E), `score_seed_big_fg.py` runs killed at 9.5h / 3h / etc. Logs in `logs/fg_features_s42.log` and `logs/fg_light_s42.log`.
- **Diagnosis refined**: iter 29 (line ~786). Three separable issues: (i) wide-RHS solve is the bottleneck, not factorisation — sparse-LU is 8× SLOWER than dense on bigger bbox; (ii) γ ∈ [0.9, 0.95] needed for noisy empirical g, not the paper's 0.5; (iii) zone variability matters.
- **Positive at subgraph scale**: γ=0.95 lifts f+g AUC to 0.553 (≈ empirical ceiling 0.563) on Xuancheng zone-0 subgraph (n=194). `logs/subgraph_gamma_zone_sweep.log` (in `sumo_validation/logs/`).

### "Where are the iter-by-iter sweep results?"
`research_logbook.md` is the canonical source. `> Iter N` blocks read like email threads — each contains motivation, design, results table, findings, and how it changes the picture.

### "Where are the raw log files those iters reference?"
`logs/` for top-level runs (Phase-4 multi-seed, multistart, T-sweep, features, Xuancheng days). `sumo_validation/logs/` for sweep transcripts produced by `dispatch_*.sh`. Both have README.md indexes mapping log → script → iter.

### "How do I configure / extend / re-run X?"
Look up X in `research_logbook.md` "Variables and how they shape the results" (line ~970) for the synthetic knobs, and in `sumo_validation/README.md` for the SUMO scripts.

### "How is the data layout in `sumo_validation/`?"
`sumo_validation/README.md` § Data — filename conventions, tag glossary (intr/satnav/big/small/d/r/seed), what each tag refers to in the iteration log.

### "How is the Xuancheng dataset structured / what's been verified about it?"
`real_data/README.md` § Verified properties of the dataset — the iter-20 audit findings (route IDs match SUMO 1:1, startTime convention, 78.4% legal pair rate, Apr 8/9 duplicate, Apr 10 anomaly, holiday dates).

### "What's in the user's memory?"
`/Users/aaronbendor/.claude/projects/-Users-aaronbendor-DESENG-Masters/memory/MEMORY.md` is the index — auto-loaded into every Claude Code session in this project. Most useful entries: `phase4_current_state.md`, `phase5_xuancheng_state.md`, `phase4_features_win.md`, `phase4_rho_demand.md`, `phase4_2step_backoff.md`, `phase4_fg_sparse_refined.md`, `thesis_writeup_state.md`. See user instructions at the top of CLAUDE.md.

---

## 5. Decisions, conventions, and traps worth knowing

These are non-obvious choices you might be tempted to "fix" — don't, without reading first.

### Phase-1 specific

1. **`predict_f` uses the scale-free rescaling `(Σ_{X_o} f) · π̂(x) / (Σ_{X_o} π̂)`, NOT Morimura's literal `ĉ · π̂`.** The literal formula collapses every RMAE to ≈ 1.0 because `stationary()` returns a probability summing to 1 over `n` states. Iter 18 verified this empirically; do not "fix" it back to the literal form. Memory: `feedback_predict_f_scaling.md`.
2. **`β` is held at its true value during inversion.** Avoids a known identifiability issue with stationary-only observations. Paper §6.1 does the same.
3. **L-BFGS-B is used instead of the paper's natural gradient.** Justified for Phase-1 robustness. Phase-4 iter-15 found this matters at SUMO scale (multi-modality); see Open issues below.
4. **Global features (`phi_T`, `psi`) are required for the paper-faithful reproduction.** They're on in the current `make_truth` and `Inverter`. Pre-iter-18 we ran a simpler local-only setup that gave misleadingly low RMAE on a structurally easier problem.

### Phase-4 specific

1. **`maxiter=1500` is the operating point.** Phase-4 iter-15 found that going to `maxiter=5000` on the same data can *worsen* test AUC at marginally lower training loss (basin switching). Loss and AUC decouple on this non-convex objective; don't equate convergence with quality.
2. **The empirical 1-step Markov chain (`empirical_chain`) is the *ceiling reference*, not a baseline.** It tells you how much signal the data carries; `fitted_f` tells you how much the model class can access. Gap closure = `(fitted - 0.5) / (empirical - 0.5)`.
3. **`gamma=1.0` on SUMO (f-only) is the operating default.** The toy default in `per_car_detector.py` is `gamma=0.1` (f+g). Phase-4 iter 19E demonstrated f+g is infeasible at SUMO bigger-bbox scale; iter 29 refined this.
4. **`X_o` filter (`f_intr > 0 & f_satnav > 0`) is empirically inert** at current operating points (94.2% of edges qualify; the excluded edges carry < 0.002% of traversals). Iter-9 audit confirmed this; don't worry about it unless going to sparser regimes.
5. **SUMO teleport pollution.** Pre-iter-9 runs at `-p 0.5` had 96% intr-vehicle teleport rates, polluting trajectories. `-p 2.0` is the small-bbox clean operating point (11.6% / 2.4% teleports). The iter-8 "0.714 / 87% gap closed" headline was *retracted in iter 9* as artefact-driven.

### Phase-5 specific

1. **No sat-nav ground-truth labels on Xuancheng.** Regime contrast is behavioural (rush vs off-peak). The SUMO ρ×demand decomposition (iter 27) is what validates that this proxy is meaningful: ~96% of the SUMO peak-vs-off-peak signal is sat-nav, ~0% is congestion-rerouting of unaided drivers. The proxy is honest *in this SUMO model*.
2. **`features=real` triggers L-BFGS basin failures on Xuancheng OD-matched** — iter 21 noted `fitted_f = 0.530 > empirical = 0.518` (impossible under correct chain recovery). Iter 25 multistart resolved it: a good basin DOES exist (loss drops 1397 → 9.2) but the well-fit chains give AUC ≈ 0.483, BELOW chance. Both the iter-21 violation and the iter-25 resolution confirm: the within-OD rush-vs-off-peak routing contrast is genuinely absent at the 1-step Markov level, not hidden by a bad optimum.

### Workflow conventions

1. **Use `python3` not bare `python`** in shell commands outside the venv. Memory: `feedback_python_commands.md`.
2. **Long runs**: hand the user a paste-ready `caffeinate -i nohup ... &` command rather than spawning a Bash call. Memory: `feedback_caffeinated_commands.md`.
3. **Tier-1 sweeps need ≥5 SUMO seeds.** Single-seed std on `fitted_f` is ±2.4 pp; the original Tier-1 detection threshold of +2 pp lives inside the noise band. Memory: `feedback_multiseed_tier1.md`.
4. **Verify before extending.** Run FD gradient checks on `Inverter` and any new analytic-gradient code before trusting it. Memory: `feedback_verify_before_extending.md`.
5. **Markdown dollar parity.** Keep `$` count even in `research_logbook.md` — bare `$VAR` in code blocks breaks VS Code preview rendering of math sections. Memory: `feedback_markdown_dollars.md`.

---

## 6. Current open issues / future work

Listed in roughly the order the user thinks about them.

1. **Thesis revision.** First draft complete (2026-05-28). Open: §6.5 needs a rewrite to reflect iter-28's 2-step backoff overturn (currently still has the single-seed "2-step doesn't help" framing); §6.8 needs the iter-29 refined diagnosis; §6.7 ρ×demand decomposition was added at iter 27. Compile + page-budget check pending. See `thesis/README.md` and memory `thesis_writeup_state.md`.
2. **Sherman–Morrison amortisation for f+g**. Iter 29 (B) identified this as the algorithmic fix that would make f+g feasible at SUMO scale: factor `(I − β P_T)` once per L-BFGS iter and re-use across all `j ∈ X_o`, instead of `|X_o|` factorisations.
3. **OD-cell-level bootstrap** on Xuancheng (rather than vehicle-level), to cleanly attribute residual signal to inter- vs intra-OD variance. Flagged in iter 26 as outstanding.
4. **Experienced-driver SUMO counterfactual.** Iter 27 caveat: `ρ=0` drivers in SUMO follow precomputed shortest-path routes and don't react to congestion. Real-world experienced drivers may avoid known-bad streets at rush hour, which the iter-27 model misses. Would need an alternative routing model.
5. **Parametric 2-step Morimura inverter.** Iter 28 overturned the original negative, so this is now a Tier-2 future-work candidate rather than deprioritised. The 2-step empirical with backoff already lifts AUC by +0.10; a parametric extension would let the same lever work in the partial-observation regime.

The `08_discussion.tex` chapter has the formal future-work list; this section is the working version.

---

## 7. How to run things

All Python commands use the project venv at `.venv/bin/python` (Python 3.14, deps: numpy, scipy, matplotlib).

```bash
# Phase 1 — Morimura reproduction. ~22 min on this laptop at n=100 + CV.
.venv/bin/python morimura.py

# Phase 2 — congestion filter (synthetic, snapshot detection).  ~1 s single-config, sweep slower.
.venv/bin/python congestion_filter.py

# Phase 3 — per-car LR detector (synthetic).  ~3 s single, sweep slower.
.venv/bin/python per_car_detector.py

# Phase 4 — score one SUMO seed (bigger bbox, features=none, T=20).  ~40 min/fit.
.venv/bin/python sumo_validation/score_seed_big.py --seed 42

# Phase 4 — features comparison + contrast diagnostic.
.venv/bin/python sumo_validation/contrast_correlation.py --seed 42 --features real

# Phase 5 — Xuancheng pooled rush vs off-peak, OD-matched.
.venv/bin/python sumo_validation/score_seed_big.py \
    --dataset xuancheng --day 2023-04-17,2023-04-18,2023-04-19,2023-04-20,2023-04-21 \
    --regime_split rush_offpeak --od_match 8 --features none

# Multi-seed Phase-4 sweep (~5.3 h).
./run_multiseed.sh

# 2-step backoff sweep (~3 min, 50 runs).
cd sumo_validation && ./dispatch_2step_backoff_sweep.sh | tee logs/2step_backoff_sweep.log
```

For a fresh SUMO simulation (Phase 4 from scratch), see `research_logbook.md` § "Reproducing the SUMO pipeline" (line ~903) — that block has the OSM extract command, `netconvert` flags, `randomTrips.py` invocation, the two `sumo` runs, and the scoring step.

---

## 8. Quick triage if a future request comes in

| Request | Read this first |
|---------|-----------------|
| "Reproduce / extend the paper's synthetic experiment" | `morimura.py` + `research_logbook.md` §Phase 1 |
| "Run a new SUMO experiment / change demand / change ρ" | `sumo_validation/README.md` + `research_logbook.md` line ~903 |
| "Score a new Xuancheng day / regime split" | `real_data/README.md` + `score_seed_big.py --dataset xuancheng` CLI |
| "Modify `Inverter` (Phase 1 solver)" | `morimura.py` `Inverter` class, FD-check via `sumo_validation/fd_check_features.py` |
| "Add a new detector" | `per_car_detector.py` for synthetic, `sumo_to_phase3.py` for SUMO |
| "Touch the thesis" | `thesis/README.md` + memory `thesis_writeup_state.md` |
| "Why did we do X / not do Y?" | Search `research_logbook.md` for the relevant iter — every reversal is documented |
| "What's the user's preferred workflow / style?" | `/Users/aaronbendor/.claude/projects/-Users-aaronbendor-DESENG-Masters/memory/` |
