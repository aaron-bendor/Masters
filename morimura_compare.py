"""Side-by-side comparison: first-commit-style configuration vs current
(paper-faithful) configuration. Both at n=100, paper's |X_o| sweep, 10
trials, same seed. The contrast isolates the effect of adding globals +
lambda cross-validation, holding everything else fixed.

Run:  .venv/bin/python morimura_compare.py
Writes:  morimura_compare_fig.png
"""
import time
import numpy as np
import matplotlib.pyplot as plt

from morimura import (
    make_truth, true_g, Inverter, stationary, nwkr_with_cv,
    predict_f, rmae, fit_with_cv,
)


def sweep(n=100, sizes=(5, 10, 20, 35, 50, 70, 90), n_trials=10, seed=42,
          *, use_globals: bool, use_cv: bool, d_T=5, d_psi=5,
          fixed_lam=1e-3, label=""):
    """One sweep at the requested configuration."""
    rng = np.random.default_rng(seed)
    methods = ("proposed", "proposed_no_g", "nwkr")
    results = {m: np.full((len(sizes), n_trials), np.nan) for m in methods}
    dT = d_T if use_globals else 0
    dpsi = d_psi if use_globals else 0
    print(f"[{label}]  use_globals={use_globals}  use_cv={use_cv}  "
          f"d_T={dT} d_psi={dpsi}  fixed_lam={fixed_lam}")
    t0 = time.time()
    for s_i, n_obs in enumerate(sizes):
        for t in range(n_trials):
            sub = np.random.default_rng(rng.integers(2**31))
            adj, pI, PT, P, pi, phi_T, psi = make_truth(
                n, 3, 0.9, mix=0.7, dir_alpha=0.3, rng=sub,
                d_T=dT, d_psi=dpsi,
            )
            f_full = 1000.0 * pi
            X_o = np.sort(sub.choice(n, size=n_obs, replace=False))
            f_o = f_full[X_o]
            g_o = true_g(PT, 0.9, X_o)

            phi_T_arg = phi_T if use_globals else None
            psi_arg   = psi   if use_globals else None

            if use_cv:
                pi_a, _, _ = fit_with_cv(
                    adj, 0.9, X_o, f_o, g_o, gamma=0.1,
                    phi_T=phi_T_arg, psi=psi_arg, rng=sub,
                )
                pi_b, _, _ = fit_with_cv(
                    adj, 0.9, X_o, f_o, None, gamma=1.0,
                    phi_T=phi_T_arg, psi=psi_arg, rng=sub,
                )
            else:
                inv = Inverter(adj, 0.9, gamma=0.1, lam=fixed_lam,
                               phi_T=phi_T_arg, psi=psi_arg)
                theta_a, _ = inv.fit(X_o, f_o, g_o, maxiter=200)
                pI_a, PT_a = inv.forward(theta_a)
                pi_a = stationary(0.9 * PT_a + 0.1 * pI_a[None, :])

                inv2 = Inverter(adj, 0.9, gamma=1.0, lam=fixed_lam,
                                phi_T=phi_T_arg, psi=psi_arg)
                theta_b, _ = inv2.fit(X_o, f_o, None, maxiter=200)
                pI_b, PT_b = inv2.forward(theta_b)
                pi_b = stationary(0.9 * PT_b + 0.1 * pI_b[None, :])

            results["proposed"][s_i, t] = rmae(
                f_full, predict_f(pi_a, X_o, f_o), X_o)
            results["proposed_no_g"][s_i, t] = rmae(
                f_full, predict_f(pi_b, X_o, f_o), X_o)
            f_nwkr, _ = nwkr_with_cv(adj, X_o, f_o)
            results["nwkr"][s_i, t] = rmae(f_full, f_nwkr, X_o)
        print(f"  size={n_obs:3d}  done")
    print(f"  [{label}] total {time.time()-t0:.1f}s")
    return list(sizes), results


def plot_overlay(sizes, results_old, results_new, out_path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    method_styles = {
        "proposed":      dict(color="C0", marker="o"),
        "proposed_no_g": dict(color="C2", marker="s"),
        "nwkr":          dict(color="C3", marker="^"),
    }
    label_map = {
        "proposed":      "proposed (with g)",
        "proposed_no_g": "proposed (no g)",
        "nwkr":          "NWKR",
    }
    for name, st in method_styles.items():
        m_old = np.nanmean(results_old[name], axis=1)
        s_old = np.nanstd(results_old[name], axis=1)
        m_new = np.nanmean(results_new[name], axis=1)
        s_new = np.nanstd(results_new[name], axis=1)
        ax.errorbar(sizes, m_old, yerr=s_old, capsize=3,
                    linestyle=":", alpha=0.65,
                    label=f"{label_map[name]} — first-commit config", **st)
        ax.errorbar(sizes, m_new, yerr=s_new, capsize=3,
                    linestyle="-", lw=2,
                    label=f"{label_map[name]} — current config", **st)
    ax.set_xlabel("# of observation states  |X_o|")
    ax.set_ylabel("RMAE  (lower is better)")
    ax.set_title("Phase-1 reproduction: first-commit config vs current  "
                 "(n=100, 10 trials)")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"saved figure: {out_path}")


if __name__ == "__main__":
    NTR = 5  # trials per cell (kept small for speed; std bars wide but shape clear)
    sizes, res_old = sweep(use_globals=False, use_cv=False, n_trials=NTR,
                            label="first-commit config")
    sizes, res_new = sweep(use_globals=True,  use_cv=True,  n_trials=NTR,
                            label="current config")
    print("\nSize  | first-commit (proposed)  | current (proposed)")
    for i, s in enumerate(sizes):
        m_o = np.nanmean(res_old["proposed"][i])
        m_n = np.nanmean(res_new["proposed"][i])
        print(f"  {s:3d} | {m_o:.3f}                    | {m_n:.3f}")
    plot_overlay(sizes, res_old, res_new, "morimura_compare_fig.png")
