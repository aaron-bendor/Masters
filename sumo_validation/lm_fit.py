"""LM (trust-region-reflective least-squares) variant of the Morimura
inverter for gamma=1.0 (f-only) fits.

Replaces L-BFGS-B with scipy.optimize.least_squares using analytic
residuals + Jacobian. The Morimura objective with gamma=1.0 is

    L(theta) = 0.5 * sum_{i, j in X_o} D[i,j]^2  +  0.5 * lam * ||theta||^2

with D[i,j] = (log_pi[i] - log_pi[j]) - (log_f[i] - log_f[j])
            = r[i] - r[j],
where r[i] = log_pi[i] - log_f[i].

Define centred residuals s[i] = r[i] - mean(r). Then

    0.5 * sum_{i,j} (r_i - r_j)^2  =  n_o * sum_i s_i^2

so an equivalent residual vector for least_squares is

    [sqrt(2 * n_o) * s_i  for i in X_o]
    [sqrt(lam) * theta_k  for k in 1..d]

which is length n_o + d (much smaller than the n_o^2 raw layout).
This module only handles gamma=1.0 (the f-only fit used by
score_seed_big.py).
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import scipy.optimize as sopt

# Make sure we import morimura.py from the worktree root (one level up).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from morimura import Inverter, stationary


def _residuals_and_J(theta, inv, X_o, log_f, want_J=True):
    """Compact residual vector r(theta) (length n_o + d) and Jacobian.

    Residual layout:
        [0 : n_o]                -> sqrt(2 n_o) * (r_i - mean(r))
                                    where r_i = log_pi_i - log_f_i
        [n_o : n_o + d]          -> sqrt(lam) * theta_k

    Total sum-of-squares matches the original loss exactly
    (verified analytically above and FD-checked numerically).
    """
    pI, PT = inv.forward(theta)
    P = inv.beta * PT + (1.0 - inv.beta) * pI[None, :]
    pi = stationary(P)
    log_pi = np.log(pi)

    n_o = len(X_o)
    r_vec = log_pi[X_o] - log_f                     # (n_o,)
    s_vec = r_vec - r_vec.mean()                    # centred
    w = np.sqrt(2.0 * n_o)
    r_d = w * s_vec                                  # length n_o
    r_reg = np.sqrt(inv.lam) * theta                # length d

    r = np.concatenate([r_d, r_reg])

    if not want_J:
        return r, None

    # Jacobian of s_i = r_i - mean(r) wrt theta:
    #   ds_i/dtheta = J_pi[i] - mean_over_X_o(J_pi)
    J_pi_full = inv.grad_log_pi(pI, PT, pi)         # (n, d)
    J_pi = J_pi_full[X_o]                           # (n_o, d)
    J_pi_mean = J_pi.mean(axis=0, keepdims=True)    # (1, d)
    J_s = J_pi - J_pi_mean                          # (n_o, d)
    J_d = w * J_s                                    # (n_o, d)

    # Ridge Jacobian: sqrt(lam) * I
    J_reg = np.sqrt(inv.lam) * np.eye(inv.d)

    J = np.vstack([J_d, J_reg])
    return r, J


def fit_lm(adj, beta, X_o, f_obs, lam=1e-3, x0=None,
           max_nfev=None, ftol=1e-9, xtol=1e-9, gtol=1e-7,
           verbose=0):
    """gamma=1.0 (f-only) Morimura fit via scipy.optimize.least_squares
    with method='trf' (trust-region-reflective, LM-style).

    Returns (theta_hat, PT_hat, pi_hat, info_dict)."""
    inv = Inverter(adj, beta, gamma=1.0, lam=lam)
    if x0 is None:
        x0 = np.zeros(inv.d)
    log_f = np.log(np.asarray(f_obs, dtype=float))
    X_o = np.asarray(X_o)

    def resid(th):
        r, _ = _residuals_and_J(th, inv, X_o, log_f, want_J=False)
        return r

    def jac(th):
        _, J = _residuals_and_J(th, inv, X_o, log_f, want_J=True)
        return J

    t0 = time.time()
    res = sopt.least_squares(
        resid, x0, jac=jac, method="trf",
        max_nfev=max_nfev, ftol=ftol, xtol=xtol, gtol=gtol,
        verbose=verbose,
    )
    elapsed = time.time() - t0

    theta = res.x
    pI_h, PT_h = inv.forward(theta)
    pi_h = stationary(beta * PT_h + (1.0 - beta) * pI_h[None, :])
    final_loss = float(res.cost)  # scipy reports 0.5 * sum(r^2)
    return theta, PT_h, pi_h, dict(
        result=res, elapsed=elapsed, final_loss=final_loss,
        nfev=res.nfev, njev=res.njev, status=res.status,
        message=res.message,
    )


def loss_lbfgs_equivalent(theta, inv, X_o, log_f):
    """Re-evaluate the *original* L_d + ridge loss at theta, for comparison
    with L-BFGS-B's scalar reported loss (which uses the n_o x n_o pairwise
    form). Used only for cross-checking the compact representation."""
    pI, PT = inv.forward(theta)
    P = inv.beta * PT + (1.0 - inv.beta) * pI[None, :]
    pi = stationary(P)
    log_pi = np.log(pi)
    D = ((log_pi[X_o, None] - log_pi[None, X_o]) -
         (log_f[:, None] - log_f[None, :]))
    L_d = 0.5 * (D ** 2).sum()
    L_R = 0.5 * (theta ** 2).sum()
    return L_d + inv.lam * L_R


def fd_check_jacobian(adj, beta, X_o, f_obs, lam=1e-3, eps=1e-6,
                      seed=0, max_cols=None):
    """Central-difference Jacobian check.

    Returns (rel_err, abs_err, J_analytic, J_fd) for `max_cols` random
    parameter columns (defaults to all d)."""
    inv = Inverter(adj, beta, gamma=1.0, lam=lam)
    rng = np.random.default_rng(seed)
    theta = 0.1 * rng.standard_normal(inv.d)
    log_f = np.log(np.asarray(f_obs, dtype=float))
    X_o = np.asarray(X_o)

    r0, J = _residuals_and_J(theta, inv, X_o, log_f, want_J=True)
    d = inv.d
    cols = (np.arange(d) if max_cols is None
            else rng.choice(d, size=min(max_cols, d), replace=False))

    J_fd = np.zeros((len(r0), len(cols)))
    for k, j in enumerate(cols):
        th_p = theta.copy(); th_p[j] += eps
        th_m = theta.copy(); th_m[j] -= eps
        r_p, _ = _residuals_and_J(th_p, inv, X_o, log_f, want_J=False)
        r_m, _ = _residuals_and_J(th_m, inv, X_o, log_f, want_J=False)
        J_fd[:, k] = (r_p - r_m) / (2.0 * eps)

    J_an = J[:, cols]
    abs_err = float(np.max(np.abs(J_an - J_fd)))
    scale = float(np.max(np.abs(J_fd))) + 1e-30
    rel_err = abs_err / scale
    return rel_err, abs_err, J_an, J_fd


if __name__ == "__main__":
    # Tiny synthetic smoke test (uses make_truth from morimura).
    from morimura import make_truth  # noqa: F401
    rng = np.random.default_rng(0)
    adj, pI, PT, P, pi, _, _ = make_truth(
        n=30, mean_out_degree=3, beta=0.9, mix=0.7, dir_alpha=0.3, rng=rng,
    )
    K = 1000.0
    f_full = K * pi
    X_o = np.sort(rng.choice(30, size=10, replace=False))
    f_o = f_full[X_o]

    rel_err, abs_err, _, _ = fd_check_jacobian(adj, 0.9, X_o, f_o, seed=1)
    print(f"FD check: rel_err={rel_err:.2e}  abs_err={abs_err:.2e}")
    theta, PT_h, pi_h, info = fit_lm(adj, 0.9, X_o, f_o, verbose=0)
    print(f"LM fit: final_loss={info['final_loss']:.6g}  "
          f"nfev={info['nfev']}  njev={info['njev']}  "
          f"elapsed={info['elapsed']:.2f}s")

    # Cross-check: original L_d + ridge at theta matches scipy.cost (within
    # floating point). The compact residual representation should give the
    # same scalar.
    inv = Inverter(adj, 0.9, gamma=1.0, lam=1e-3)
    log_f = np.log(f_o)
    L_orig = loss_lbfgs_equivalent(theta, inv, X_o, log_f)
    print(f"Cross-check: original L_d + lam*R = {L_orig:.6g}  "
          f"(LM cost = {info['final_loss']:.6g}, ratio = "
          f"{L_orig / info['final_loss']:.6f})")
