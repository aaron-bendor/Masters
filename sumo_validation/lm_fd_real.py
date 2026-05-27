"""Standalone FD check of the LM residual Jacobian on the real bigger-bbox
SUMO instance. Compact: 3 epsilons, 5 columns each.
"""

import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

from sumo_to_phase3 import build_adj, read_edge_counts
from lm_fit import fd_check_jacobian

S = 42
edges, adj_out, idx = build_adj("net.net.xml")
n = len(edges)
print(f"n={n}", flush=True)
f_intr = read_edge_counts(f"edgedata.intr.big.seed{S}.xml", idx)
cands = np.where(f_intr > 0)[0]
n_obs = min(max(20, int(0.25 * n)), len(cands))
rng_xo = np.random.default_rng(13)
X_o = np.sort(rng_xo.choice(cands, size=n_obs, replace=False))
print(f"|X_o|={len(X_o)}", flush=True)

beta = 0.972

for eps in (1e-3, 1e-5, 1e-7):
    rel_err, abs_err, _, _ = fd_check_jacobian(
        adj_out, beta, X_o, f_intr[X_o],
        lam=1e-3, eps=eps, seed=1, max_cols=5,
    )
    print(f"  eps={eps:.0e}  rel_err={rel_err:.3e}  abs_err={abs_err:.3e}",
          flush=True)
