"""
per_car_detector.py - Per-trajectory sat-nav anomaly detection via
log-likelihood ratio.

Builds on morimura.py. Separate from congestion_filter.py, which scores
*snapshots* of aggregate flow; here we score *one car's trajectory*.

Setup
-----
  * Intrinsic chain  pT_intr    - drivers' natural preferences (same as
                                  morimura.py).
  * Sat-nav chain    pT_satnav  - self-consistent congestion-minimising
                                  variant:
        pT_satnav(x'|x)  prop  pT_intr(x'|x) * exp(-alpha cong(x'))
        cong(x)          =     pi_satnav(x) / max(pi_satnav)
    where pi_satnav is the stationary of pT_satnav ITSELF. Fixed-point in
    pi, solved by damped Picard iteration. Distinct from
    congestion_filter.py's influenced_pT, which uses pi_intr instead of
    pi_satnav.

Two-class experiment
--------------------
  * Sample station counts f at X_o from K * pi_intr  and from K * pi_satnav
    (separate "off-peak" and "rush-hour" observation phases).
  * Fit the Morimura inverter twice  ->  pT_intr_hat, pT_satnav_hat.
  * Sample N test trajectories of length T from each regime (no restart -
    a real car doesn't teleport mid-trip).
  * Score with the log-likelihood ratio
        Lambda(tau) = sum_t [ log pT_satnav_hat(x_{t+1}|x_t)
                            - log pT_intr_hat (x_{t+1}|x_t) ]
    positive => sat-nav-like, negative => intrinsic-like.
  * ROC of Lambda against ground-truth class.

Baselines
---------
  * oracle    - same Lambda using true pT_intr, pT_satnav. Upper bound.
  * one_class - score = -sum_t log pT_intr_hat(x_{t+1}|x_t). Tests whether
                modelling both regimes beats just flagging "unlikely
                under intrinsic".

Run:  python per_car_detector.py
"""

from __future__ import annotations

import time
import numpy as np
import matplotlib.pyplot as plt

from morimura import make_truth, stationary, Inverter, softmax
from congestion_filter import roc


# ===============================================================
#  Self-consistent sat-nav chain
# ===============================================================

def satnav_pT_given_pi(adj_out, PT_intr, alpha, pi):
    """One step of the fixed point: pT_satnav(x'|x) prop
    pT_intr(x'|x) exp(-alpha * pi(x') / max pi). Matches the functional
    form of congestion_filter.influenced_pT, but with congestion derived
    from `pi` (which will be pi_satnav at the fixed point) rather than
    pi_intr."""
    cong = pi / pi.max()
    PT = np.zeros_like(PT_intr)
    for x, nbrs in enumerate(adj_out):
        s = (np.log(np.clip(PT_intr[x, nbrs], 1e-15, None))
             - alpha * cong[nbrs])
        PT[x, nbrs] = softmax(s)
    return PT


def satnav_pT(adj_out, PT_intr, pI, beta, alpha,
              tol=1e-8, max_iter=200, damping=0.5):
    """Solve  pi = stationary( beta * pT_satnav(pi) + (1-beta) pI 1^T ).
    Damped Picard iteration: pi <- (1-d) pi + d pi_new."""
    pi = stationary(beta * PT_intr + (1.0 - beta) * pI[None, :])
    delta = np.inf
    for it in range(max_iter):
        PT_new = satnav_pT_given_pi(adj_out, PT_intr, alpha, pi)
        pi_new = stationary(beta * PT_new + (1.0 - beta) * pI[None, :])
        delta = float(np.max(np.abs(pi_new - pi)))
        if delta < tol:
            pi = pi_new
            break
        pi = (1.0 - damping) * pi + damping * pi_new
    PT_fp = satnav_pT_given_pi(adj_out, PT_intr, alpha, pi)
    return PT_fp, pi, it + 1, delta


# ===============================================================
#  Two-regime population observation + Morimura double-fit
# ===============================================================

def fit_chain(adj, beta, X_o, f_obs, lam=1e-3, maxiter=300):
    """Fit pT (and pI) to a single count vector at X_o via the Morimura
    inverter, stationary-only (gamma=1.0). Returns the fitted pI, pT and
    the resulting stationary."""
    inv = Inverter(adj, beta, gamma=1.0, lam=lam)
    theta, _ = inv.fit(X_o, f_obs, None, maxiter=maxiter)
    pI_h, PT_h = inv.forward(theta)
    pi_h = stationary(beta * PT_h + (1.0 - beta) * pI_h[None, :])
    return pI_h, PT_h, pi_h


# ===============================================================
#  Trajectory simulator + scorers (no restart)
# ===============================================================

def sample_trajectory(pI, PT, T, rng):
    """Length T+1 state sequence with T transitions. x_0 ~ pI; thereafter
    pure pT walk (no restart). Returned as int array."""
    n = len(pI)
    xs = np.zeros(T + 1, dtype=int)
    xs[0] = rng.choice(n, p=pI)
    for t in range(T):
        xs[t + 1] = rng.choice(n, p=PT[xs[t]])
    return xs


def log_lik(traj, PT, eps=1e-15):
    """sum_t log PT(x_{t+1} | x_t). The softmax fit has all-positive
    entries on edges, so the clip is only insurance for the oracle
    chain (Dirichlet noise can yield very small but nonzero entries)."""
    p = np.clip(PT[traj[:-1], traj[1:]], eps, None)
    return float(np.log(p).sum())


# ===============================================================
#  Detection scores
# ===============================================================

def _scores_for_chains(trajs, PT_intr, PT_satnav):
    return np.array([log_lik(t, PT_satnav) - log_lik(t, PT_intr)
                     for t in trajs])


def _scores_one_class(trajs, PT_intr):
    """Anomaly = trajectory unlikely under intrinsic. Sign convention:
    high score => positive class (= sat-nav)."""
    return np.array([-log_lik(t, PT_intr) for t in trajs])


# ===============================================================
#  Single trial
# ===============================================================

def run_trial(n=50, mean_out_degree=3, beta=0.9, alpha=1.5,
              K=20000, obs_frac=0.10,
              T_values=(20, 30, 50), n_test_per_class=300,
              seed=42):
    """One trial = one truth + one X_o + one pair of fitted chains, then
    ROC at each T in T_values. Test trajectories are sampled at the max T
    and truncated for shorter T, so curves across T are paired."""
    rng = np.random.default_rng(seed)
    adj, pI, PT_intr, _, pi_intr = make_truth(
        n, mean_out_degree, beta, mix=0.7, dir_alpha=0.3, rng=rng,
    )
    PT_satnav, pi_satnav, fp_iter, fp_delta = satnav_pT(
        adj, PT_intr, pI, beta, alpha,
    )

    n_obs = max(2, int(round(obs_frac * n)))
    X_o = np.sort(rng.choice(n, size=n_obs, replace=False))

    f_intr   = rng.poisson(K * pi_intr  )[X_o]
    f_satnav = rng.poisson(K * pi_satnav)[X_o]
    _, PT_intr_h,   _ = fit_chain(adj, beta, X_o, f_intr)
    _, PT_satnav_h, _ = fit_chain(adj, beta, X_o, f_satnav)

    T_max = max(T_values)
    trajs_intr   = [sample_trajectory(pI, PT_intr,   T_max, rng)
                    for _ in range(n_test_per_class)]
    trajs_satnav = [sample_trajectory(pI, PT_satnav, T_max, rng)
                    for _ in range(n_test_per_class)]
    labels = np.concatenate([np.zeros(n_test_per_class, dtype=int),
                             np.ones (n_test_per_class, dtype=int)])

    per_T = {}
    for T in T_values:
        cut_intr   = [t[: T + 1] for t in trajs_intr]
        cut_satnav = [t[: T + 1] for t in trajs_satnav]
        cut_all = cut_intr + cut_satnav
        scores = {
            "fitted":    _scores_for_chains(cut_all, PT_intr_h, PT_satnav_h),
            "oracle":    _scores_for_chains(cut_all, PT_intr,   PT_satnav),
            "one_class": _scores_one_class (cut_all, PT_intr_h),
        }
        rocs = {name: roc(s, labels) for name, s in scores.items()}
        per_T[T] = dict(scores=scores, roc=rocs,
                        aucs={k: r[2] for k, r in rocs.items()})

    return dict(
        adj=adj, X_o=X_o, n=n,
        alpha=alpha, beta=beta, K=K,
        pi_intr=pi_intr, pi_satnav=pi_satnav,
        fp_iter=fp_iter, fp_delta=fp_delta,
        labels=labels, per_T=per_T, T_values=tuple(T_values),
    )


# ===============================================================
#  Plotting (single config)
# ===============================================================

def plot_single(trial, T_show=None,
                out_path="per_car_detector_fig.png"):
    if T_show is None:
        T_show = max(trial["T_values"])
    info = trial["per_T"][T_show]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # ROC
    ax = axes[0]
    styles = dict(fitted=dict(color="C0", lw=2),
                  oracle=dict(color="C2", lw=2, linestyle="--"),
                  one_class=dict(color="C3", lw=1.6, linestyle=":"))
    for name, r in info["roc"].items():
        fpr, tpr, auc = r
        ax.plot(fpr, tpr, label=f"{name}  (AUC = {auc:.3f})", **styles[name])
    ax.plot([0, 1], [0, 1], "k:", alpha=0.4)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(rf"ROC  ($\alpha$={trial['alpha']}, T={T_show}, "
                 rf"$|X_o|$={len(trial['X_o'])}/{trial['n']})")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    # Score histogram (fitted)
    ax = axes[1]
    s = info["scores"]["fitted"]
    lbl = trial["labels"]
    bins = np.linspace(s.min(), s.max(), 35)
    ax.hist(s[lbl == 0], bins=bins, alpha=0.6, color="C0",
            label=f"intrinsic (n={int((lbl == 0).sum())})")
    ax.hist(s[lbl == 1], bins=bins, alpha=0.6, color="C3",
            label=f"sat-nav (n={int((lbl == 1).sum())})")
    ax.axvline(0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel(r"$\Lambda(\tau) = \log L_{\mathrm{sat\,nav}} - \log L_{\mathrm{intr}}$")
    ax.set_ylabel("# trajectories")
    ax.set_title("LR score distribution (fitted chains)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Stationaries: how flat is sat-nav?
    ax = axes[2]
    order = np.argsort(-trial["pi_intr"])
    ax.plot(trial["pi_intr"][order],   color="C0", lw=2, label=r"$\pi_{\mathrm{intr}}$")
    ax.plot(trial["pi_satnav"][order], color="C3", lw=2,
            label=r"$\pi_{\mathrm{sat\,nav}}$")
    ax.set_xlabel(r"states (sorted by $\pi_{\mathrm{intr}}$)")
    ax.set_ylabel("stationary probability")
    ax.set_title(
        rf"Sat-nav flattens flow  (FP iters: {trial['fp_iter']}, "
        rf"$\Delta_\pi$: {trial['fp_delta']:.1e})"
    )
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"saved figure: {out_path}")


# ===============================================================
#  Sweep over (alpha, T_traj)
# ===============================================================

def run_sweep(alpha_values=(0.5, 1.0, 1.5, 2.5, 4.0),
              T_values=(20, 35, 50),
              n_trials=3, base_seed=42, **kwargs):
    methods = ("fitted", "oracle", "one_class")
    auc = {m: np.zeros((len(alpha_values), len(T_values)))
           for m in methods}
    auc_std = {m: np.zeros_like(auc["fitted"]) for m in methods}
    for i, a in enumerate(alpha_values):
        per = {(m, j): [] for m in methods for j in range(len(T_values))}
        for t in range(n_trials):
            r = run_trial(alpha=a, T_values=T_values,
                          seed=base_seed + t, **kwargs)
            for j, T in enumerate(T_values):
                for m in methods:
                    per[(m, j)].append(r["per_T"][T]["aucs"][m])
        for j, T in enumerate(T_values):
            for m in methods:
                auc[m][i, j]     = float(np.mean(per[(m, j)]))
                auc_std[m][i, j] = float(np.std (per[(m, j)]))
            print(f"  alpha={a:>4.2f}  T={T:>3d}  "
                  f"fitted={auc['fitted'][i, j]:.3f}  "
                  f"oracle={auc['oracle'][i, j]:.3f}  "
                  f"one_class={auc['one_class'][i, j]:.3f}")
    return dict(alpha_values=np.asarray(alpha_values),
                T_values=np.asarray(T_values),
                auc=auc, auc_std=auc_std, n_trials=n_trials)


def plot_sweep(sweep, out_path="per_car_detector_sweep.png"):
    Ts = sweep["T_values"]
    A  = sweep["alpha_values"]
    fig, axes = plt.subplots(1, len(Ts),
                             figsize=(4.6 * len(Ts), 4.5), sharey=True)
    if len(Ts) == 1:
        axes = [axes]
    methods = [("fitted",    "C0", "o", "-"),
               ("oracle",    "C2", "s", "--"),
               ("one_class", "C3", "^", ":")]
    for j, T in enumerate(Ts):
        ax = axes[j]
        for name, color, marker, ls in methods:
            ax.errorbar(A, sweep["auc"][name][:, j],
                        yerr=sweep["auc_std"][name][:, j],
                        color=color, marker=marker, linestyle=ls,
                        capsize=3, label=name)
        ax.set_xlabel(r"$\alpha$  (sat-nav avoidance strength)")
        if j == 0:
            ax.set_ylabel("ROC AUC")
            ax.legend(loc="lower right")
        ax.set_title(f"T = {T}")
        ax.axhline(0.5, color="k", linestyle=":", alpha=0.4)
        ax.set_ylim(0.45, 1.02)
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"Per-trajectory detector AUC vs sat-nav strength  "
        f"({sweep['n_trials']} trials/cell)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"saved figure: {out_path}")


# ===============================================================
#  Driver
# ===============================================================

if __name__ == "__main__":
    t0 = time.time()

    print("Single config")
    trial = run_trial(alpha=1.5, T_values=(20, 35, 50),
                      n_test_per_class=300, seed=42)
    for T in trial["T_values"]:
        a = trial["per_T"][T]["aucs"]
        print(f"  T={T:>3d}  fitted={a['fitted']:.3f}  "
              f"oracle={a['oracle']:.3f}  one_class={a['one_class']:.3f}")
    plot_single(trial)

    print("\nSweep over (alpha, T)")
    sweep = run_sweep(
        alpha_values=(0.5, 1.0, 1.5, 2.5, 4.0),
        T_values=(20, 35, 50),
        n_trials=3, base_seed=42,
        n_test_per_class=300,
    )
    plot_sweep(sweep)

    print(f"\ntotal time: {time.time() - t0:.1f}s")
