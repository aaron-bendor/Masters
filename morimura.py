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
    """Fit theta = [nu^loc, omega^loc, omega^glo1, omega^glo2] to match
    observed f and g on X_o. gamma in [0, 1] balances stationary-prob and
    hitting-prob losses.

    Globals are optional: pass `phi_T` (shape n x d_T) for destination-state
    features (paper's phi_T(x') in Eq. 17), and `psi` (shape E x d_psi) for
    directed-edge features (paper's psi(x, x') in Eq. 17). Either or both
    may be omitted (paper p. 7: "If a simpler model is preferred, either of
    them would be omitted."); with phi_T=None and psi=None the behaviour
    reduces exactly to the local-only configuration."""

    def __init__(self, adj_out, beta, gamma=0.1, lam=1e-3,
                 phi_T=None, psi=None):
        self.n = len(adj_out)
        self.adj_out = adj_out
        self.beta = beta
        self.gamma = gamma
        self.lam = lam
        self.edges = [(x, y) for x, ys in enumerate(adj_out) for y in ys]
        self.E = len(self.edges)
        self.edge_idx = {e: i for i, e in enumerate(self.edges)}
        self._src = np.array([x for (x, y) in self.edges], dtype=int)
        self._dst = np.array([y for (x, y) in self.edges], dtype=int)
        # Global feature matrices
        self.phi_T = (np.asarray(phi_T, dtype=float)
                      if phi_T is not None else np.zeros((self.n, 0)))
        self.psi   = (np.asarray(psi,   dtype=float)
                      if psi   is not None else np.zeros((self.E, 0)))
        assert self.phi_T.shape[0] == self.n, \
            f"phi_T first dim must be n={self.n}, got {self.phi_T.shape}"
        assert self.psi.shape[0]   == self.E, \
            f"psi first dim must be E={self.E}, got {self.psi.shape}"
        self.d_T   = int(self.phi_T.shape[1])
        self.d_psi = int(self.psi.shape[1])
        # Pre-index phi_T at the destination of each edge for fast forward
        self._phi_T_dst = (self.phi_T[self._dst] if self.d_T > 0
                           else np.zeros((self.E, 0)))
        self.d = self.n + self.E + self.d_T + self.d_psi

    def _split_theta(self, theta):
        n, E, dT = self.n, self.E, self.d_T
        nu      = theta[:n]
        om_loc  = theta[n : n + E]
        om_glo1 = theta[n + E : n + E + dT]
        om_glo2 = theta[n + E + dT : self.d]
        return nu, om_loc, om_glo1, om_glo2

    def forward(self, theta):
        nu, om_loc, om_glo1, om_glo2 = self._split_theta(theta)
        pI = softmax(nu)
        # Score per edge: omega^loc + phi_T(dst).om_glo1 + psi.om_glo2
        s = om_loc.copy()
        if self.d_T > 0:
            s = s + self._phi_T_dst @ om_glo1
        if self.d_psi > 0:
            s = s + self.psi @ om_glo2
        # Build PT by per-origin softmax over outgoing edges
        PT = np.zeros((self.n, self.n))
        for x, nbrs in enumerate(self.adj_out):
            idx = [self.edge_idx[(x, y)] for y in nbrs]
            PT[x, nbrs] = softmax(s[idx])
        return pI, PT

    # ----- Jacobian of log pi wrt theta  (Eq. 12) -----
    def grad_log_pi(self, pI, PT, pi):
        n, d, b = self.n, self.d, self.beta
        E, dT, dpsi = self.E, self.d_T, self.d_psi
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
        # omega^glo1 block: globals on phi_T(x') (destination feature)
        # V_glo1[y, k] = beta * [tilde_pi[y] phi_T[y,k] - (P_T^T (pi . bar_phi_T))[y,k]]
        # where tilde_pi = P_T^T pi, bar_phi_T = P_T phi_T
        if dT > 0:
            bar_phi_T = PT @ self.phi_T               # (n, d_T)
            tilde_pi  = PT.T @ pi                     # (n,)
            V_glo1 = b * (tilde_pi[:, None] * self.phi_T
                          - PT.T @ (pi[:, None] * bar_phi_T))
            V[:, n + E : n + E + dT] = V_glo1
        # omega^glo2 block: globals on psi(x, x') (edge feature)
        # V_glo2[y, k] = beta * [sum_{e: dst(e)=y} pi[src(e)] P_T[src,dst] psi_e[k]
        #                        - (P_T^T (pi . bar_psi))[y, k]]
        # bar_psi[x, k] = sum_{e: src(e)=x} P_T[src,dst] psi_e[k]
        if dpsi > 0:
            P_T_e = PT[self._src, self._dst]          # (E,)
            weighted_psi = P_T_e[:, None] * self.psi  # (E, d_psi)
            bar_psi = np.zeros((n, dpsi))
            np.add.at(bar_psi, self._src, weighted_psi)
            u = (pi[self._src] * P_T_e)[:, None] * self.psi  # (E, d_psi)
            V_first = np.zeros((n, dpsi))
            np.add.at(V_first, self._dst, u)
            V_glo2 = b * (V_first - PT.T @ (pi[:, None] * bar_psi))
            V[:, n + E + dT :] = V_glo2
        MV = sla.solve(Q, V)
        return MV / pi[:, None]

    # ----- Jacobian of log h_theta(j) wrt theta  (Eq. 15) -----
    def grad_log_h(self, PT, h_j, lu, piv, j):
        n, d, b = self.n, self.d, self.beta
        E, dT, dpsi = self.E, self.d_T, self.d_psi
        V = np.zeros((n, d))
        # nu params don't enter P_T -> their columns stay zero.
        # omega^loc block (existing)
        for a, nbrs in enumerate(self.adj_out):
            if a == j:
                continue
            pa = PT[a, nbrs]
            h_a = h_j[nbrs]
            scalars = pa * (h_a - pa @ h_a)
            cols = [n + self.edge_idx[(a, c)] for c in nbrs]
            V[a, cols] = scalars
        # omega^glo1 block: for a != j,
        # V[a, k] = sum_y P_T(y|a) phi_T(y,k) h(y) - bar_phi_T(a,k) (P_T h)(a)
        # row j stays zero (absorbed). No beta inside V; applied at return.
        if dT > 0:
            Mh = PT @ (h_j[:, None] * self.phi_T)     # (n, d_T)
            bar_phi_T = PT @ self.phi_T               # (n, d_T)
            r = PT @ h_j                              # (n,)
            V_glo1 = Mh - bar_phi_T * r[:, None]
            V_glo1[j, :] = 0
            V[:, n + E : n + E + dT] = V_glo1
        # omega^glo2 block: for a != j,
        # V[a, k] = sum_{e: src(e)=a} P_T[src,dst] psi_e[k] h[dst]
        #          - bar_psi(a, k) (P_T h)(a)
        if dpsi > 0:
            P_T_e = PT[self._src, self._dst]          # (E,)
            weighted_psi = P_T_e[:, None] * self.psi  # (E, d_psi)
            bar_psi = np.zeros((n, dpsi))
            np.add.at(bar_psi, self._src, weighted_psi)
            u = (P_T_e * h_j[self._dst])[:, None] * self.psi  # (E, d_psi)
            V_first_h = np.zeros((n, dpsi))
            np.add.at(V_first_h, self._src, u)
            r = PT @ h_j                              # (n,)
            V_glo2 = V_first_h - bar_psi * r[:, None]
            V_glo2[j, :] = 0
            V[:, n + E + dT :] = V_glo2
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

def make_truth(n, mean_out_degree, beta, mix, dir_alpha, rng,
               d_T=0, d_psi=0):
    """Build a 'true' Markov chain: softmax-parametric model with optional
    global features, then mixed with Dirichlet noise on both pI and each
    row of pT.

    Paper §6.1 protocol: "every element of nu, omega, phi_I(x), phi_T(x),
    psi_T(x, x') was drawn independently from N(0, 1^2)." Setting both
    d_T and d_psi to 0 reduces to the local-only variant (still a valid
    paper configuration per the "either of them would be omitted" remark
    on p. 7). With d_T, d_psi > 0 the truth is the paper's §6.2-style
    configuration (transition globals only).

    Returns (adj_out, pI, PT, P, pi, phi_T, psi) where phi_T (n, d_T) and
    psi (E, d_psi) are the feature matrices used in the truth — pass these
    to Inverter to use the same parametric family for recovery."""
    adj_out = random_graph(n, mean_out_degree, rng)
    edges = [(x, y) for x, ys in enumerate(adj_out) for y in ys]
    edge_idx = {e: i for i, e in enumerate(edges)}
    E = len(edges)
    dst = np.array([y for (_, y) in edges], dtype=int)

    # Features (drawn iid N(0, 1))
    phi_T = (rng.normal(size=(n, d_T)) if d_T > 0
             else np.zeros((n, 0), dtype=float))
    psi = (rng.normal(size=(E, d_psi)) if d_psi > 0
           else np.zeros((E, 0), dtype=float))

    # Initial prob (local only — Inverter doesn't take phi_I)
    nu_loc = rng.normal(size=n)
    pI_clean = softmax(nu_loc)
    sigma = rng.dirichlet(dir_alpha * np.ones(n))
    pI = mix * pI_clean + (1.0 - mix) * sigma
    pI = pI / pI.sum()

    # Transition probs: per-edge score om_loc + phi_T(dst).om_glo1 + psi.om_glo2
    om_loc = rng.normal(size=E)
    om_glo1 = rng.normal(size=d_T) if d_T > 0 else np.zeros(0)
    om_glo2 = rng.normal(size=d_psi) if d_psi > 0 else np.zeros(0)
    s = om_loc.copy()
    if d_T > 0:
        s = s + phi_T[dst] @ om_glo1
    if d_psi > 0:
        s = s + psi @ om_glo2
    PT_clean = np.zeros((n, n))
    for x, nbrs in enumerate(adj_out):
        idx = [edge_idx[(x, y)] for y in nbrs]
        PT_clean[x, nbrs] = softmax(s[idx])

    PT = np.zeros_like(PT_clean)
    for x, nbrs in enumerate(adj_out):
        tau = rng.dirichlet(dir_alpha * np.ones(len(nbrs)))
        row = mix * PT_clean[x, nbrs] + (1.0 - mix) * tau
        PT[x, nbrs] = row / row.sum()

    P = beta * PT + (1.0 - beta) * pI[None, :]
    pi = stationary(P)
    return adj_out, pI, PT, P, pi, phi_T, psi


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
    """Convert recovered stationary probability to count units.

    The paper writes c_hat = mean(f over X_o) and f_hat(x) = c_hat * pi_hat(x).
    Taken literally that under-scales by a factor of |X|/|X_o|, because our
    `stationary()` returns a probability summing to 1 over all states, so
    pi_hat(x) ~ 1/n while f(x) ~ K/n. The scale-free reading that gives
    sensible magnitudes — and the one we use — is:
       f_hat(x) = (sum f_o) * pi_hat(x) / sum(pi_hat[X_o])
    which normalises pi_hat so its sum over X_o matches the empirical sum
    of f over X_o. Reduces to the paper's formula when sum(pi_hat[X_o]) = 1."""
    s = float(np.sum(pi_hat[X_o]))
    if s <= 0:
        s = 1e-15
    return float(np.sum(f_o)) * pi_hat / s


def fit_with_cv(adj, beta, X_o, f_obs, g_obs=None, gamma=0.1,
                phi_T=None, psi=None, lam_grid=None, val_frac=0.25,
                maxiter=200, rng=None):
    """Pick lambda by one train/val split on X_o (paper §6.1: 'lambda was
    determined with a cross-validation'), then refit on full X_o.

    Returns (pi_hat, best_lam, val_err_best)."""
    rng = np.random.default_rng() if rng is None else rng
    if lam_grid is None:
        lam_grid = [1e-4, 1e-3, 1e-2, 1e-1]
    X_o = np.asarray(X_o)
    f_obs = np.asarray(f_obs, dtype=float)
    n_o = len(X_o)
    n_val = max(1, int(round(val_frac * n_o)))
    perm = rng.permutation(n_o)
    val_local = np.sort(perm[:n_val])
    tr_local = np.sort(perm[n_val:])
    X_train, X_val = X_o[tr_local], X_o[val_local]
    f_train, f_val = f_obs[tr_local], f_obs[val_local]
    g_train = (g_obs[np.ix_(tr_local, tr_local)] if g_obs is not None else None)

    best_lam, best_err = lam_grid[0], np.inf
    for lam in lam_grid:
        inv = Inverter(adj, beta, gamma=gamma, lam=lam,
                       phi_T=phi_T, psi=psi)
        theta, _ = inv.fit(X_train, f_train, g_train, maxiter=maxiter)
        pI_h, PT_h = inv.forward(theta)
        pi_h = stationary(beta * PT_h + (1 - beta) * pI_h[None, :])
        # Same scaling as predict_f, applied with train-only info.
        s_train = float(np.sum(pi_h[X_train]))
        scale = float(np.sum(f_train)) / (s_train if s_train > 0 else 1e-15)
        f_pred_val = scale * pi_h[X_val]
        err = float(np.mean(np.abs(f_pred_val - f_val)
                            / np.maximum(f_val, 1.0)))
        if err < best_err:
            best_err, best_lam = err, lam
    inv = Inverter(adj, beta, gamma=gamma, lam=best_lam,
                   phi_T=phi_T, psi=psi)
    theta, _ = inv.fit(X_o, f_obs, g_obs, maxiter=maxiter)
    pI_h, PT_h = inv.forward(theta)
    pi_h = stationary(beta * PT_h + (1 - beta) * pI_h[None, :])
    return pi_h, best_lam, best_err


def run(n=100, mean_out_degree=3, beta=0.9, K=1000.0,
        sizes=(5, 10, 20, 35, 50, 70, 90),
        d_T=5, d_psi=5,
        n_trials=10, seed=42):
    """Reproduce paper Fig. 2A. n=100 with paper's |X_o| sweep, transition
    globals on (d_T=d_psi=5), lambda picked by CV per (size, trial, method)."""
    rng = np.random.default_rng(seed)
    methods = ("proposed", "proposed_no_g", "nwkr")
    results = {m: np.full((len(sizes), n_trials), np.nan) for m in methods}
    lams = {m: np.full((len(sizes), n_trials), np.nan) for m in methods[:2]}
    print(f"|X|={n}  out_deg={mean_out_degree}  beta={beta}  "
          f"d_T={d_T} d_psi={d_psi}  "
          f"sizes={list(sizes)}  n_trials={n_trials}")

    for s_i, n_obs in enumerate(sizes):
        for t in range(n_trials):
            sub = np.random.default_rng(rng.integers(2**31))
            adj, pI, PT, P, pi, phi_T, psi = make_truth(
                n, mean_out_degree, beta,
                mix=0.7, dir_alpha=0.3, rng=sub,
                d_T=d_T, d_psi=d_psi,
            )
            f_full = K * pi
            X_o = np.sort(sub.choice(n, size=n_obs, replace=False))
            f_o = f_full[X_o]
            g_o = true_g(PT, beta, X_o)

            # (a) proposed - uses both f and g, paper's gamma=0.1
            pi_a, lam_a, _ = fit_with_cv(
                adj, beta, X_o, f_o, g_o, gamma=0.1,
                phi_T=phi_T, psi=psi, rng=sub,
            )
            results["proposed"][s_i, t] = rmae(
                f_full, predict_f(pi_a, X_o, f_o), X_o)
            lams["proposed"][s_i, t] = lam_a

            # (b) proposed (no g) - gamma=1.0
            pi_b, lam_b, _ = fit_with_cv(
                adj, beta, X_o, f_o, g_obs=None, gamma=1.0,
                phi_T=phi_T, psi=psi, rng=sub,
            )
            results["proposed_no_g"][s_i, t] = rmae(
                f_full, predict_f(pi_b, X_o, f_o), X_o)
            lams["proposed_no_g"][s_i, t] = lam_b

            # (c) NWKR baseline
            f_nwkr, bw = nwkr_with_cv(adj, X_o, f_o)
            results["nwkr"][s_i, t] = rmae(f_full, f_nwkr, X_o)

            print(f"  size={n_obs:3d}  trial={t}  "
                  f"proposed={results['proposed'][s_i, t]:.3f} (lam={lam_a:.0e})  "
                  f"no_g={results['proposed_no_g'][s_i, t]:.3f} (lam={lam_b:.0e})  "
                  f"nwkr={results['nwkr'][s_i, t]:.3f} (bw={bw:.2f})")
    return list(sizes), results, lams


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
    sizes, results, lams = run(
        n=100, mean_out_degree=3, beta=0.9,
        sizes=(5, 10, 20, 35, 50, 70, 90),
        d_T=5, d_psi=5,
        n_trials=10, seed=42,
    )
    plot(sizes, results)
    print(f"total time: {time.time() - t0:.1f}s")
