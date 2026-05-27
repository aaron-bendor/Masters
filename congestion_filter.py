"""
congestion_filter.py - Detect externally-influenced observations and
recover the intrinsic Markov chain by filtering them out.

Builds on morimura.py.

Setup
-----
  * Intrinsic chain  M_intr   - drivers' natural preferences (the chain
                                we fit in the original paper).
  * Influenced chain M_infl   - per-link avoidance under congestion:
        p_T^infl(x'|x)  =  p_T^intr(x'|x) * exp(-alpha * cong(x'))
        with  cong(x) = pi^intr(x) / max(pi^intr).
  * T time snapshots indexed by t. A binary regime r_t is observed:
      r_t = 0  uncongested  ->  rate(x) = K * pi^intr(x).
      r_t = 1  congested    ->  rate(x) = K * [(1-rho) pi^intr +
                                                 rho     pi^infl ](x),
                                with rho ~ 0.7 (sat-nav adoption).
    Counts:  f_t(x) ~ Poisson(rate(x))  for x in X_o.

A subset of the uncongested snapshots come pre-labelled (the user's
trusted reference set). The rest are unlabelled and may be in either
regime.

Three intrinsic-chain recovery strategies
-----------------------------------------
  1. naive     - aggregate all snapshots, fit the inverter on the sum.
  2. oracle    - aggregate only true-r=0 snapshots (cheating baseline).
  3. filtered  - score each unlabelled snapshot by its Poisson log-
                 likelihood under the empirical intrinsic rate from the
                 labelled set, keep the most-intrinsic-looking fraction,
                 aggregate labelled + kept and refit.

Detection task
--------------
For each unlabelled snapshot, compute an anomaly score; classify
high-score ones as influenced. ROC against the (held-out for evaluation
only) ground-truth regime r_t.

Run:  python congestion_filter.py
"""

from __future__ import annotations

import time
import numpy as np
import matplotlib.pyplot as plt

from morimura import make_truth, stationary, Inverter, softmax


# ===============================================================
#  Influenced chain (per-link avoidance, model A from the proposal)
# ===============================================================

def cong_vector(pi_intr):
    """Per-state congestion proxy: normalised intrinsic flow in [0, 1]."""
    return pi_intr / pi_intr.max()


def influenced_pT(adj_out, PT_intr, alpha, cong):
    """p_T^infl(x'|x)  proportional to  p_T^intr(x'|x) * exp(-alpha cong(x'))"""
    PT = np.zeros_like(PT_intr)
    for x, nbrs in enumerate(adj_out):
        s = (np.log(np.clip(PT_intr[x, nbrs], 1e-15, None))
             - alpha * cong[nbrs])
        PT[x, nbrs] = softmax(s)
    return PT


# ===============================================================
#  Two-regime data generation
# ===============================================================

def generate_two_regime(n, mean_out_degree, beta, alpha, K,
                        T, p_congested, rho_c,
                        mix=0.7, dir_alpha=0.3, rng=None):
    rng = rng or np.random.default_rng()
    adj, pI, PT_intr, _, pi_intr, _, _ = make_truth(
        n, mean_out_degree, beta, mix, dir_alpha, rng,
    )
    cong = cong_vector(pi_intr)
    PT_infl = influenced_pT(adj, PT_intr, alpha, cong)
    pi_infl = stationary(beta * PT_infl + (1.0 - beta) * pI[None, :])

    regimes = (rng.random(T) < p_congested).astype(int)
    snapshots = np.zeros((T, n), dtype=int)
    for t, r in enumerate(regimes):
        rho = rho_c if r == 1 else 0.0
        rate = K * ((1.0 - rho) * pi_intr + rho * pi_infl)
        snapshots[t] = rng.poisson(rate)

    return dict(
        adj=adj, pI=pI, PT_intr=PT_intr, PT_infl=PT_infl,
        pi_intr=pi_intr, pi_infl=pi_infl, cong=cong,
        snapshots=snapshots, regimes=regimes, beta=beta, K=K,
    )


# ===============================================================
#  Detection
# ===============================================================

def kl_score(counts, intr_rate):
    """KL(empirical distribution at X_o  ||  intrinsic distribution at X_o).

    Comparing distributions rather than Poisson rates removes the
    counts-scale dependency: total mass at X_o differs between regimes
    for reasons unrelated to which regime generated the snapshot, so a
    raw Poisson likelihood ends up tracking that nuisance instead of
    the actual chain difference."""
    f = counts.astype(float)
    total = f.sum()
    if total <= 0:
        return 0.0
    p_emp = f / total
    p_intr = intr_rate / intr_rate.sum()
    p_intr = np.clip(p_intr, 1e-12, None)
    mask = p_emp > 0
    return float(np.sum(p_emp[mask] * (np.log(p_emp[mask]) -
                                       np.log(p_intr[mask]))))


def roc(scores, labels):
    """ROC for a 1-D score array against binary labels (1 = positive).
    Higher score => more 'positive' (here: more 'congested-looking')."""
    labels = labels.astype(float)
    order = np.argsort(-scores)
    s = labels[order]
    P, N = s.sum(), len(s) - s.sum()
    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.5
    tpr = np.cumsum(s) / P
    fpr = np.cumsum(1.0 - s) / N
    fpr = np.concatenate([[0.0], fpr])
    tpr = np.concatenate([[0.0], tpr])
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


# ===============================================================
#  Recovery (wraps the morimura inverter)
# ===============================================================

def fit_intrinsic(adj, beta, X_o, f_aggr, lam=1e-3, maxiter=300):
    """Fit the intrinsic chain to a single aggregated count vector at X_o."""
    inv = Inverter(adj, beta, gamma=1.0, lam=lam)
    theta, res = inv.fit(X_o, f_aggr, None, maxiter=maxiter)
    pI_h, PT_h = inv.forward(theta)
    pi_h = stationary(beta * PT_h + (1.0 - beta) * pI_h[None, :])
    return pi_h, res


def relative_mae(pi_true, pi_pred, X_o):
    """Relative MAE on stationary probabilities at unobserved states."""
    n = len(pi_true)
    mask = np.ones(n, dtype=bool)
    mask[X_o] = False
    return float(np.mean(np.abs(pi_true[mask] - pi_pred[mask]) /
                         np.clip(pi_true[mask], 1e-12, None)))


# ===============================================================
#  One end-to-end trial
# ===============================================================

def run_trial(n=50, mean_out_degree=3, beta=0.9, alpha=3.0,
              T=200, T_label=40, p_congested=0.6, rho_c=0.7,
              K=20000, n_obs=20, fpr_target=0.05, seed=42):
    """fpr_target controls the filter's threshold: pick it so that at most
    fpr_target of the labelled (known-intrinsic) snapshots would be
    falsely flagged as anomalous.  The same threshold is then applied to
    the unlabelled set."""
    rng = np.random.default_rng(seed)
    truth = generate_two_regime(
        n, mean_out_degree, beta, alpha, K,
        T, p_congested, rho_c, rng=rng,
    )
    adj = truth["adj"]
    pi_intr = truth["pi_intr"]
    snapshots = truth["snapshots"]
    regimes = truth["regimes"]

    X_o = np.sort(rng.choice(n, size=n_obs, replace=False))

    uncong = np.where(regimes == 0)[0]
    n_label = min(T_label, len(uncong))
    labelled = rng.choice(uncong, size=n_label, replace=False)
    is_lab = np.zeros(T, dtype=bool)
    is_lab[labelled] = True
    unlabelled = np.where(~is_lab)[0]
    unl_truth = regimes[unlabelled]

    # Empirical intrinsic distribution at X_o, from the labelled set
    intr_rate_o = snapshots[labelled][:, X_o].mean(axis=0)
    intr_rate_o = np.clip(intr_rate_o, 0.5, None)

    # Anomaly score: KL(emp || intrinsic) at X_o.  High => looks unlike
    # the intrinsic regime  =>  probably influenced.
    def score_one(t):
        return kl_score(snapshots[t, X_o], intr_rate_o)
    scores_unl = np.array([score_one(t) for t in unlabelled])
    scores_lab = np.array([score_one(t) for t in labelled])

    fpr, tpr, auc = roc(scores_unl, unl_truth)

    # Threshold = (1 - fpr_target) quantile of labelled scores, so that
    # under the null (intrinsic) at most fpr_target fraction is flagged.
    threshold = float(np.quantile(scores_lab, 1.0 - fpr_target))
    keep_unl = unlabelled[scores_unl <= threshold]

    F_naive = snapshots[:, X_o].sum(axis=0)
    pi_naive, _ = fit_intrinsic(adj, beta, X_o, F_naive)

    F_oracle = snapshots[regimes == 0][:, X_o].sum(axis=0)
    pi_oracle, _ = fit_intrinsic(adj, beta, X_o, F_oracle)

    selected = np.concatenate([labelled, keep_unl])
    F_filt = snapshots[selected][:, X_o].sum(axis=0)
    pi_filt, _ = fit_intrinsic(adj, beta, X_o, F_filt)

    return dict(
        scores=scores_unl, scores_lab=scores_lab, threshold=threshold,
        unl_truth=unl_truth, fpr=fpr, tpr=tpr, auc=auc,
        rmaes=dict(
            naive=relative_mae(pi_intr, pi_naive, X_o),
            oracle=relative_mae(pi_intr, pi_oracle, X_o),
            filter=relative_mae(pi_intr, pi_filt, X_o),
        ),
        truth=truth, X_o=X_o,
        labelled=labelled, unlabelled=unlabelled,
        n_kept=len(keep_unl),
    )


# ===============================================================
#  Multi-trial driver and plotting
# ===============================================================

def run(n_trials=5, base_seed=42, **kwargs):
    aucs = []
    rmaes = {"naive": [], "oracle": [], "filter": []}
    one_trial = None
    for t in range(n_trials):
        out = run_trial(seed=base_seed + t, **kwargs)
        aucs.append(out["auc"])
        for k in rmaes:
            rmaes[k].append(out["rmaes"][k])
        if one_trial is None:
            one_trial = out
        print(f"  trial {t}: AUC={out['auc']:.3f}  "
              f"naive={out['rmaes']['naive']:.3f}  "
              f"oracle={out['rmaes']['oracle']:.3f}  "
              f"filter={out['rmaes']['filter']:.3f}")
    return aucs, rmaes, one_trial


def plot(aucs, rmaes, one_trial, out_path="congestion_filter_fig.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # ROC for one representative trial
    ax = axes[0]
    ax.plot(one_trial["fpr"], one_trial["tpr"], color="C0", lw=2,
            label=f"AUC = {one_trial['auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k:", alpha=0.5, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(
        f"Detection ROC  (one trial; AUC over {len(aucs)} trials: "
        f"{np.mean(aucs):.3f} ± {np.std(aucs):.3f})"
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    # Score histogram split by true regime, with the chosen threshold
    ax = axes[1]
    sc = one_trial["scores"]
    lbl = one_trial["unl_truth"]
    sc_all = np.concatenate([sc, one_trial["scores_lab"]])
    bins = np.linspace(sc_all.min(), sc_all.max(), 30)
    ax.hist(one_trial["scores_lab"], bins=bins, alpha=0.6, color="C0",
            label=f"labelled intrinsic (n={len(one_trial['scores_lab'])})")
    ax.hist(sc[lbl == 0], bins=bins, alpha=0.6, color="C2",
            label=f"unlabelled & intrinsic (n={int((lbl == 0).sum())})")
    ax.hist(sc[lbl == 1], bins=bins, alpha=0.6, color="C3",
            label=f"unlabelled & influenced (n={int((lbl == 1).sum())})")
    ax.axvline(one_trial["threshold"], color="k", linestyle="--",
               label=f"threshold ({one_trial['n_kept']} kept)")
    ax.set_xlabel(r"anomaly score  KL$(\hat p_t \,\|\, \hat p^{\mathrm{intr}})$")
    ax.set_ylabel("# snapshots")
    ax.set_title("Score distribution by true regime")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # RMAE bar chart
    ax = axes[2]
    methods = ["naive", "filter", "oracle"]
    means = [np.mean(rmaes[m]) for m in methods]
    stds = [np.std(rmaes[m]) for m in methods]
    colors = {"naive": "C3", "filter": "C0", "oracle": "C2"}
    bars = ax.bar(methods, means, yerr=stds,
                  color=[colors[m] for m in methods],
                  alpha=0.8, capsize=6)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(r"relative MAE  on  $\hat\pi^{\mathrm{intr}}$")
    ax.set_title(f"Intrinsic-chain recovery error  ({len(aucs)} trials)")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"saved figure: {out_path}")


# ===============================================================
#  2D sweep over (alpha, K) - difficulty map
# ===============================================================

def run_sweep_2d(K_values, alpha_values, n_trials=5, base_seed=42, **kwargs):
    """Per-cell mean AUC and recovery RMAEs over (alpha, K)."""
    nA, nK = len(alpha_values), len(K_values)
    out = {
        "auc":    np.zeros((nA, nK)),
        "naive":  np.zeros((nA, nK)),
        "filter": np.zeros((nA, nK)),
        "oracle": np.zeros((nA, nK)),
        "K_values":     np.asarray(K_values),
        "alpha_values": np.asarray(alpha_values),
    }
    total = nA * nK
    cell = 0
    for i, alpha in enumerate(alpha_values):
        for j, K in enumerate(K_values):
            cell += 1
            aucs, n_, f_, o_ = [], [], [], []
            for t in range(n_trials):
                r = run_trial(K=K, alpha=alpha, seed=base_seed + t, **kwargs)
                aucs.append(r["auc"])
                n_.append(r["rmaes"]["naive"])
                f_.append(r["rmaes"]["filter"])
                o_.append(r["rmaes"]["oracle"])
            out["auc"][i, j]    = np.mean(aucs)
            out["naive"][i, j]  = np.mean(n_)
            out["filter"][i, j] = np.mean(f_)
            out["oracle"][i, j] = np.mean(o_)
            print(f"  [{cell:2d}/{total}] alpha={alpha:>4.2f}  K={K:>6d}  "
                  f"AUC={out['auc'][i, j]:.3f}  "
                  f"naive={out['naive'][i, j]:.3f}  "
                  f"filter={out['filter'][i, j]:.3f}  "
                  f"oracle={out['oracle'][i, j]:.3f}")
    return out


def _annotate(ax, arr, text_thresh):
    """Write each cell's value over the heatmap, choosing text colour for
    legibility."""
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            ax.text(j, i, f"{v:.2f}" if abs(v) >= 0.01 else f"{v:.0e}",
                    ha="center", va="center", fontsize=8,
                    color="white" if v < text_thresh else "black")


def plot_sweep_2d(sweep, out_path="congestion_filter_sweep.png"):
    K = sweep["K_values"]
    A = sweep["alpha_values"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    def _heat(ax, data, title, cmap, vmin, vmax, text_thresh, cbar_label=None):
        im = ax.imshow(data, origin="lower", aspect="auto",
                       cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(K)))
        ax.set_xticklabels([f"{k}" for k in K])
        ax.set_yticks(range(len(A)))
        ax.set_yticklabels([f"{a:.2f}" for a in A])
        ax.set_xlabel("K  (counts per snapshot)")
        ax.set_ylabel(r"$\alpha$  (avoidance strength)")
        ax.set_title(title)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                ax.text(j, i, f"{v:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if v < text_thresh else "black")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046)
        if cbar_label:
            cbar.set_label(cbar_label)

    _heat(axes[0], sweep["auc"], "Detection AUC",
          cmap="viridis", vmin=0.5, vmax=1.0, text_thresh=0.75)

    gap_filt = sweep["filter"] - sweep["oracle"]
    vmax = max(0.05, float(np.abs(gap_filt).max()))
    _heat(axes[1], gap_filt,
          "Filter RMAE − Oracle RMAE\n(higher = filter underperforms oracle)",
          cmap="Reds", vmin=0.0, vmax=vmax, text_thresh=vmax * 0.5)

    gap_naive = sweep["naive"] - sweep["oracle"]
    vmax2 = max(0.05, float(np.abs(gap_naive).max()))
    _heat(axes[2], gap_naive,
          "Naive RMAE − Oracle RMAE\n(higher = more contamination to fix)",
          cmap="Reds", vmin=0.0, vmax=vmax2, text_thresh=vmax2 * 0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"saved figure: {out_path}")


if __name__ == "__main__":
    t0 = time.time()

    print("Phase 2 single-config (5 trials at defaults)")
    aucs, rmaes, one_trial = run(n_trials=5, base_seed=42)
    plot(aucs, rmaes, one_trial)

    print("\n2D sweep over (alpha, K)")
    sweep = run_sweep_2d(
        K_values=[200, 500, 1000, 3000, 10000, 30000],
        alpha_values=[0.3, 0.6, 1.0, 1.5, 3.0],
        n_trials=5, base_seed=42,
    )
    plot_sweep_2d(sweep)
    print(f"total time: {time.time() - t0:.1f}s")
