"""
morimura.py - Reproduction of the synthetic experiment in Section 6.1 of:

  Morimura, Osogami, Ide. "Solving inverse problem of Markov chain with
  partial observations." NeurIPS 2013.

Setup: an n-state Markov chain with restart is generated from softmax-based
parametric models (paper Eqs. 16, 18) and then perturbed with Dirichlet
noise (mix=0.7, alpha=0.3) so that the data-generating chain lies slightly
outside the parametric family. Visit-frequencies f(x) and (optionally)
hitting-rates g(x, x') are observed on a small subset X_o of states. The
solver fits the same parametric family by minimising the regularised
objective (Eq. 7) using explicit analytic gradients for log-stationary
(Eq. 12) and log-hitting (Eq. 15) probabilities.

Differences from the paper:
  - Local-only parameters (no global features). The paper says either the
    local or the global term in Eqs. 17-18 may be omitted.
  - L-BFGS-B optimiser in place of the natural-gradient descent of Sec. 4.1
    - same loss, robust quasi-Newton step on the analytic gradients.
  - Continuation rate beta is held at its true value to avoid a known
    identifiability issue with stationary-only observations.

Score (paper):
  RMAE = (1 / |X\\X_o|) sum_{x notin X_o} |f(x) - c_hat * pi_hat(x)|
                                                     / max(f(x), 1),
  c_hat = (1 / |X_o|) sum_{x in X_o} f(x).

Run:  python morimura.py
"""

from __future__ import annotations

import time
import numpy as np
import scipy.linalg as sla
import scipy.optimize as sopt
import scipy.sparse as ssp
import scipy.sparse.csgraph as sgr
import matplotlib.pyplot as plt


# ===============================================================
#  Random connected directed graph
# ===============================================================

def random_graph(n, mean_out_degree, rng):
    """n nodes, each with `mean_out_degree` random out-neighbours plus one
    extra edge from a Hamiltonian cycle so the graph is strongly connected
    (hence the chain is ergodic for any positive softmax weights)."""
    adj = [set() for _ in range(n)]
    nodes = np.arange(n)
    for x in range(n):
        cands = nodes[nodes != x]
        for y in rng.choice(cands, size=mean_out_degree, replace=False):
            adj[x].add(int(y))
    cycle = rng.permutation(n)
    for i in range(n):
        adj[int(cycle[i])].add(int(cycle[(i + 1) % n]))
    return [sorted(s) for s in adj]


# ===============================================================
#  Markov-chain primitives
# ===============================================================

def softmax(s):
    s = s - s.max()
    e = np.exp(s)
    return e / e.sum()


def build_pT(adj_out, om_loc, edge_idx):
    n = len(adj_out)
    PT = np.zeros((n, n))
    for x, nbrs in enumerate(adj_out):
        idx = [edge_idx[(x, y)] for y in nbrs]
        PT[x, nbrs] = softmax(om_loc[idx])
    return PT


def stationary(P):
    """Solve pi^T P = pi^T with the normalisation constraint by replacing
    the last balance equation."""
    n = P.shape[0]
    A = P.T - np.eye(n)
    A[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    pi = sla.solve(A, b)
    pi = np.clip(pi, 1e-15, None)
    return pi / pi.sum()


def hitting_pack(PT, beta, j):
    """h_theta(j) and the LU factor of (I - beta P_T^{\\j}) (Eq. 14).
    Returned h_theta(j) is the vector of hitting probabilities of j from
    each starting state."""
    n = PT.shape[0]
    Px = PT.copy()
    Px[j, :] = 0.0
    A = np.eye(n) - beta * Px
    lu, piv = sla.lu_factor(A)
    e = np.zeros(n)
    e[j] = 1.0
    h = sla.lu_solve((lu, piv), e)
    return np.clip(h, 1e-15, 1.0), lu, piv


# ===============================================================
#  Inverse solver  (paper Sections 3-4)
# ===============================================================

class Inverter:
    """Fit theta = [nu^loc, omega^loc] to match observed f and g on X_o.
    gamma in [0, 1] balances stationary-prob and hitting-prob losses."""

    def __init__(self, adj_out, beta, gamma=0.1, lam=1e-3):
        self.n = len(adj_out)
        self.adj_out = adj_out
        self.beta = beta
        self.gamma = gamma
        self.lam = lam
        self.edges = [(x, y) for x, ys in enumerate(adj_out) for y in ys]
        self.E = len(self.edges)
        self.edge_idx = {e: i for i, e in enumerate(self.edges)}
        self.d = self.n + self.E

    def forward(self, theta):
        nu = theta[: self.n]
        om = theta[self.n:]
        pI = softmax(nu)
        PT = build_pT(self.adj_out, om, self.edge_idx)
        return pI, PT

    # ----- Jacobian of log pi wrt theta  (Eq. 12) -----
    def grad_log_pi(self, pI, PT, pi):
        n, d, b = self.n, self.d, self.beta
        P = b * PT + (1.0 - b) * pI[None, :]
        # Q = I - P^T + pi 1^T  (always invertible by Prop. 1)
        Q = np.eye(n) - P.T + np.outer(pi, np.ones(n))
        # V[:, k] = (dP^T / dtheta_k) pi
        V = np.zeros((n, d))
        # nu^loc block:  (1-beta) * (diag(pI) - pI pI^T)
        V[:, :n] = (1.0 - b) * (np.diag(pI) - np.outer(pI, pI))
        # omega^loc block: per origin a, sparse on rows in X_a
        for a, nbrs in enumerate(self.adj_out):
            pa = PT[a, nbrs]
            block = b * pi[a] * (np.diag(pa) - np.outer(pa, pa))
            cols = [n + self.edge_idx[(a, c)] for c in nbrs]
            V[np.ix_(nbrs, cols)] = block
        MV = sla.solve(Q, V)
        return MV / pi[:, None]

    # ----- Jacobian of log h_theta(j) wrt theta  (Eq. 15) -----
    def grad_log_h(self, PT, h_j, lu, piv, j):
        n, d, b = self.n, self.d, self.beta
        V = np.zeros((n, d))
        # only omega^loc contributes (initial-prob params don't enter P_T)
        for a, nbrs in enumerate(self.adj_out):
            if a == j:
                continue
            pa = PT[a, nbrs]
            h_a = h_j[nbrs]
            scalars = pa * (h_a - pa @ h_a)
            cols = [n + self.edge_idx[(a, c)] for c in nbrs]
            V[a, cols] = scalars
        MV = sla.lu_solve((lu, piv), V)
        return b * MV / h_j[:, None]

    # ----- combined loss + gradient  (Eqs. 7-9) -----
    def loss_grad(self, theta, X_o, log_f, log_g):
        d = self.d
        pI, PT = self.forward(theta)
        P = self.beta * PT + (1.0 - self.beta) * pI[None, :]
        pi = stationary(P)
        log_pi = np.log(pi)

        # L_d  (Eq. 8): log-ratio matching on stationary probs
        D = ((log_pi[X_o, None] - log_pi[None, X_o]) -
             (log_f[:, None] - log_f[None, :]))
        L_d = 0.5 * (D ** 2).sum()
        if self.gamma > 0:
            J = self.grad_log_pi(pI, PT, pi)
            grad_d = 2.0 * D.sum(axis=1) @ J[X_o]
        else:
            grad_d = np.zeros(d)

        # L_h  (Eq. 9): log-error on hitting probs
        L_h = 0.0
        grad_h = np.zeros(d)
        if self.gamma < 1.0 and log_g is not None:
            for kj, j in enumerate(X_o):
                h_j, lu, piv = hitting_pack(PT, self.beta, j)
                log_h = np.log(h_j)
                r = log_h[X_o] - log_g[:, kj]   # log h(j|i) - log g(i, j)
                L_h += 0.5 * (r ** 2).sum()
                Jh = self.grad_log_h(PT, h_j, lu, piv, j)
                grad_h += r @ Jh[X_o]

        # ridge regulariser  R(theta) = 1/2 ||theta||^2  ->  grad = theta
        L_R = 0.5 * (theta ** 2).sum()
        L = self.gamma * L_d + (1 - self.gamma) * L_h + self.lam * L_R
        g = self.gamma * grad_d + (1 - self.gamma) * grad_h + self.lam * theta
        return L, g

    def fit(self, X_o, f_obs, g_obs=None, x0=None, maxiter=300):
        log_f = np.log(np.asarray(f_obs, dtype=float))
        log_g = (np.log(np.clip(g_obs, 1e-15, None))
                 if g_obs is not None else None)
        X_o = np.asarray(X_o)
        if x0 is None:
            x0 = np.zeros(self.d)
        res = sopt.minimize(
            self.loss_grad, x0, args=(X_o, log_f, log_g),
            jac=True, method="L-BFGS-B",
            options=dict(maxiter=maxiter, ftol=1e-9, gtol=1e-7),
        )
        return res.x, res


# ===============================================================
#  Nadaraya-Watson kernel regression baseline
# ===============================================================

def _hop_distance(adj_out):
    n = len(adj_out)
    rows, cols = [], []
    for x, ys in enumerate(adj_out):
        for y in ys:
            rows.append(x)
            cols.append(y)
    A = ssp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n, n)
    )
    A_und = A + A.T
    return sgr.shortest_path(A_und, directed=False, unweighted=True)


def nwkr_predict(dist, X_o, f_obs, bandwidth):
    K = np.exp(-dist[:, X_o] ** 2 / (2.0 * bandwidth ** 2))
    K_sum = K.sum(axis=1, keepdims=True)
    K_sum = np.where(K_sum == 0, 1.0, K_sum)
    return (K @ f_obs) / K_sum.ravel()


def nwkr_with_cv(adj_out, X_o, f_obs, bandwidths=None):
    """Tune bandwidth by leave-one-out CV on the observed states."""
    dist = _hop_distance(adj_out)
    if bandwidths is None:
        bandwidths = np.linspace(0.5, 5.0, 10)
    n_o = len(X_o)
    best_err, best_bw = np.inf, bandwidths[0]
    X_o_arr = np.asarray(X_o)
    f_obs = np.asarray(f_obs)
    for bw in bandwidths:
        err = 0.0
        for i in range(n_o):
            mask = np.ones(n_o, dtype=bool)
            mask[i] = False
            p = nwkr_predict(dist, X_o_arr[mask], f_obs[mask], bw)
            err += abs(p[X_o_arr[i]] - f_obs[i])
        if err < best_err:
            best_err = err
            best_bw = bw
    return nwkr_predict(dist, X_o_arr, f_obs, best_bw), best_bw


# ===============================================================
#  Synthetic data
# ===============================================================

def make_truth(n, mean_out_degree, beta, mix, dir_alpha, rng):
    """Build a 'true' Markov chain: clean softmax model, then mixed with
    Dirichlet noise on both pI and each row of pT."""
    adj_out = random_graph(n, mean_out_degree, rng)
    edges = [(x, y) for x, ys in enumerate(adj_out) for y in ys]
    edge_idx = {e: i for i, e in enumerate(edges)}

    nu_loc = rng.normal(size=n)
    pI_clean = softmax(nu_loc)
    sigma = rng.dirichlet(dir_alpha * np.ones(n))
    pI = mix * pI_clean + (1.0 - mix) * sigma
    pI = pI / pI.sum()

    om_loc = rng.normal(size=len(edges))
    PT_clean = build_pT(adj_out, om_loc, edge_idx)
    PT = np.zeros_like(PT_clean)
    for x, nbrs in enumerate(adj_out):
        tau = rng.dirichlet(dir_alpha * np.ones(len(nbrs)))
        row = mix * PT_clean[x, nbrs] + (1.0 - mix) * tau
        PT[x, nbrs] = row / row.sum()

    P = beta * PT + (1.0 - beta) * pI[None, :]
    pi = stationary(P)
    return adj_out, pI, PT, P, pi


def true_g(PT, beta, X_o):
    no = len(X_o)
    g = np.zeros((no, no))
    for kj, j in enumerate(X_o):
        h, _, _ = hitting_pack(PT, beta, j)
        g[:, kj] = h[X_o]
    return g


# ===============================================================
#  Score and experiment driver
# ===============================================================

def rmae(f_true, f_pred, X_o):
    n = len(f_true)
    mask = np.ones(n, dtype=bool)
    mask[X_o] = False
    return np.mean(np.abs(f_true[mask] - f_pred[mask])
                   / np.maximum(f_true[mask], 1.0))


def predict_f(pi_hat, X_o, f_o):
    """Rescale stationary estimate to f's units. The paper writes
    c_hat = mean(f) and f_hat = c_hat * pi_hat, but that only gives the
    right magnitude if pi_hat sums to |X_o| over X_o. Equivalent
    scale-free form: f_hat(x) = (sum f_o) * pi_hat(x) / sum(pi_hat[X_o])."""
    s = float(np.sum(pi_hat[X_o]))
    if s <= 0:
        s = 1e-15
    return float(np.sum(f_o)) * pi_hat / s


def run(n=50, mean_out_degree=3, beta=0.9, K=1000.0,
        obs_fracs=(0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90),
        n_trials=3, seed=42):
    """Sweep |X_o|, average RMAE across n_trials random instances."""
    rng = np.random.default_rng(seed)
    sizes = [max(2, int(round(f * n))) for f in obs_fracs]
    methods = ("proposed", "proposed_no_g", "nwkr")
    results = {m: np.full((len(sizes), n_trials), np.nan) for m in methods}
    print(f"|X|={n}  out_deg={mean_out_degree}  beta={beta}  "
          f"sizes={sizes}  n_trials={n_trials}")

    for s_i, n_obs in enumerate(sizes):
        for t in range(n_trials):
            sub = np.random.default_rng(rng.integers(2**31))
            adj, pI, PT, P, pi = make_truth(
                n, mean_out_degree, beta,
                mix=0.7, dir_alpha=0.3, rng=sub,
            )
            f_full = K * pi
            X_o = np.sort(sub.choice(n, size=n_obs, replace=False))
            f_o = f_full[X_o]
            g_o = true_g(PT, beta, X_o)

            # (a) proposed - uses both f and g
            inv = Inverter(adj, beta, gamma=0.1, lam=1e-3)
            theta_a, _ = inv.fit(X_o, f_o, g_o, maxiter=200)
            pI_a, PT_a = inv.forward(theta_a)
            pi_a = stationary(beta * PT_a + (1 - beta) * pI_a[None, :])
            results["proposed"][s_i, t] = rmae(
                f_full, predict_f(pi_a, X_o, f_o), X_o)

            # (b) proposed (no g)
            inv2 = Inverter(adj, beta, gamma=1.0, lam=1e-3)
            theta_b, _ = inv2.fit(X_o, f_o, None, maxiter=200)
            pI_b, PT_b = inv2.forward(theta_b)
            pi_b = stationary(beta * PT_b + (1 - beta) * pI_b[None, :])
            results["proposed_no_g"][s_i, t] = rmae(
                f_full, predict_f(pi_b, X_o, f_o), X_o)

            # (c) NWKR baseline
            f_nwkr, bw = nwkr_with_cv(adj, X_o, f_o)
            results["nwkr"][s_i, t] = rmae(f_full, f_nwkr, X_o)

            print(f"  size={n_obs:3d}  trial={t}  "
                  f"proposed={results['proposed'][s_i, t]:.3f}  "
                  f"no_g={results['proposed_no_g'][s_i, t]:.3f}  "
                  f"nwkr={results['nwkr'][s_i, t]:.3f}  (bw={bw:.2f})")
    return sizes, results


def plot(sizes, results, out_path="morimura_fig.png"):
    fig, ax = plt.subplots(figsize=(7, 5))
    styles = dict(
        proposed=dict(color="C0", marker="o",
                      label="Proposed (uses f and g)"),
        proposed_no_g=dict(color="C2", marker="s", linestyle="--",
                           label="Proposed (no g)"),
        nwkr=dict(color="C3", marker="^", linestyle=":",
                  label="NWKR (baseline)"),
    )
    for name, st in styles.items():
        m = np.nanmean(results[name], axis=1)
        s = np.nanstd(results[name], axis=1)
        ax.errorbar(sizes, m, yerr=s, capsize=3, **st)
    ax.set_xlabel("# of observation states  |X_o|")
    ax.set_ylabel("RMAE")
    ax.set_title("Reproducing Morimura et al. (2013), Fig. 2A")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"saved figure: {out_path}")


if __name__ == "__main__":
    t0 = time.time()
    sizes, results = run(
        n=50, mean_out_degree=3, beta=0.9,
        obs_fracs=(0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90),
        n_trials=3, seed=42,
    )
    plot(sizes, results)
    print(f"total time: {time.time() - t0:.1f}s")
