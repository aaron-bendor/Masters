"""
sumo_to_phase3.py - Map a pair of SUMO runs (intrinsic vs. rerouting) onto
the Phase 3 per-trajectory log-likelihood-ratio detector.

Inputs (in this directory):
    net.net.xml
    edgedata.intr.xml      edgedata.satnav.xml
    vehroutes.intr.xml     vehroutes.satnav.xml

Output:
    sumo_phase3_fig.png

Run from anywhere with SUMO_HOME set:
    .venv/bin/python sumo_validation/sumo_to_phase3.py

Notes
-----
Oracle and one_class detectors from Phase 3 aren't available here - SUMO is
the ground-truth process but doesn't expose a parametric pT we can plug in.
We run the realistic detector only: `fitted_f` (f-only Morimura fit, gamma=1).
Adding `fitted_fg` later means estimating empirical hitting rates on X_o.
"""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt

# Pull in Phase 3 code from the parent (Masters) directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from per_car_detector import fit_chain, _scores_for_chains
from congestion_filter import roc

# sumolib (network parser) ships with SUMO under $SUMO_HOME/tools.
sumo_home = os.environ.get("SUMO_HOME")
if sumo_home:
    sys.path.insert(0, os.path.join(sumo_home, "tools"))
import sumolib  # noqa: E402


# ===============================================================
#  State graph from net.net.xml
# ===============================================================

def build_adj(net_path):
    """Return (edges, adj_out, idx) for the passenger-allowed, non-internal
    subgraph restricted to its largest strongly-connected component."""
    net = sumolib.net.readNet(net_path)
    raw = [e for e in net.getEdges()
           if not e.getID().startswith(":") and e.allows("passenger")]
    raw_set = {e.getID() for e in raw}
    adj_raw = {e.getID(): [s.getID() for s in e.getOutgoing()
                           if s.getID() in raw_set]
               for e in raw}
    scc = _largest_scc(adj_raw)
    edges = sorted(scc)
    idx = {e: i for i, e in enumerate(edges)}
    adj_out = [[idx[s] for s in adj_raw[e] if s in idx] for e in edges]
    return edges, adj_out, idx


def _largest_scc(adj_dict):
    """Iterative Tarjan - safe for deep graphs."""
    index = {}
    lowlink = {}
    on_stack = {}
    sccs = []
    counter = [0]

    def strongconnect(start):
        index[start] = counter[0]
        lowlink[start] = counter[0]
        counter[0] += 1
        on_stack[start] = True
        call_stack = [start]
        work = [(start, iter(adj_dict.get(start, [])))]
        while work:
            node, it = work[-1]
            nbr = next(it, None)
            if nbr is None:
                if lowlink[node] == index[node]:
                    comp = set()
                    while True:
                        w = call_stack.pop()
                        on_stack[w] = False
                        comp.add(w)
                        if w == node:
                            break
                    sccs.append(comp)
                work.pop()
                if work:
                    parent = work[-1][0]
                    lowlink[parent] = min(lowlink[parent], lowlink[node])
            else:
                if nbr not in index:
                    index[nbr] = counter[0]
                    lowlink[nbr] = counter[0]
                    counter[0] += 1
                    on_stack[nbr] = True
                    call_stack.append(nbr)
                    work.append((nbr, iter(adj_dict.get(nbr, []))))
                elif on_stack.get(nbr, False):
                    lowlink[node] = min(lowlink[node], index[nbr])

    for n in adj_dict:
        if n not in index:
            strongconnect(n)
    return max(sccs, key=len)


# ===============================================================
#  Global features from net.net.xml  (paper Eqs. 17-18)
# ===============================================================

_ROAD_CLASSES = ("motorway", "primary", "tertiary", "residential", "service", "other")


def _road_class(sumo_edge):
    """Map SUMO/OSM road type to a coarser class."""
    t = sumo_edge.getType() or ""
    if t.startswith("highway."):
        t = t[len("highway."):]
    if t in ("motorway", "motorway_link", "trunk", "trunk_link"):
        return "motorway"
    if t in ("primary", "primary_link", "secondary", "secondary_link"):
        return "primary"
    if t in ("tertiary", "tertiary_link"):
        return "tertiary"
    if t in ("residential", "living_street"):
        return "residential"
    if t in ("service", "unclassified", "road"):
        return "service"
    return "other"


def _headings(shape):
    """Unit headings at the start and end of an edge polyline."""
    if len(shape) < 2:
        v = np.array([1.0, 0.0])
        return v, v
    p0, p1 = np.array(shape[0]), np.array(shape[1])
    h0 = p1 - p0
    n = np.linalg.norm(h0)
    h0 = h0 / n if n > 0 else np.array([1.0, 0.0])
    pm, pn = np.array(shape[-2]), np.array(shape[-1])
    h1 = pn - pm
    n = np.linalg.norm(h1)
    h1 = h1 / n if n > 0 else np.array([1.0, 0.0])
    return h0, h1


def extract_features(net_path, edges, adj_out):
    """Build (phi_T, psi) feature matrices matching the paper's Eqs. 17-18.

    phi_T[i, :]  (per-destination state i): one-hot road class (6 dims) +
        standardised log speed limit (1) + standardised log lane count (1)
        => shape (n, 8).

    psi[e, :]    (per-directed-edge e = (x, y) in adj_out's order): turn-angle
        cosine between edge x's end-heading and edge y's start-heading (1),
        plus standardised log speed-limit difference log s(y) - log s(x) (1)
        => shape (E, 2).

    Continuous features are zero-mean unit-variance. Returns (phi_T, psi)."""
    net = sumolib.net.readNet(net_path)
    n = len(edges)
    sumo_edges = [net.getEdge(eid) for eid in edges]

    road_oh = np.zeros((n, len(_ROAD_CLASSES)))
    raw_log_speed = np.zeros(n)
    raw_log_lanes = np.zeros(n)
    head = []
    for i, se in enumerate(sumo_edges):
        road_oh[i, _ROAD_CLASSES.index(_road_class(se))] = 1.0
        raw_log_speed[i] = np.log(max(1.0, se.getSpeed()))
        raw_log_lanes[i] = np.log(max(1, se.getLaneNumber()))
        head.append(_headings(se.getShape()))

    def _std(x):
        m, s = float(x.mean()), float(x.std())
        return (x - m) / (s + 1e-9)

    phi_T = np.column_stack([
        road_oh,
        _std(raw_log_speed)[:, None],
        _std(raw_log_lanes)[:, None],
    ])

    edge_list = [(x, y) for x, ys in enumerate(adj_out) for y in ys]
    E = len(edge_list)
    raw_turn_cos = np.zeros(E)
    raw_log_speed_ratio = np.zeros(E)
    for i, (x, y) in enumerate(edge_list):
        _, hx_end = head[x]
        hy_start, _ = head[y]
        raw_turn_cos[i] = float(np.clip(np.dot(hx_end, hy_start), -1.0, 1.0))
        raw_log_speed_ratio[i] = raw_log_speed[y] - raw_log_speed[x]
    psi = np.column_stack([
        _std(raw_turn_cos)[:, None],
        _std(raw_log_speed_ratio)[:, None],
    ])

    return phi_T, psi


# ===============================================================
#  Edge counts (f) from edgedata.*.xml
# ===============================================================

def read_edge_counts(edgedata_path, idx):
    n = len(idx)
    counts = np.zeros(n, dtype=float)
    root = ET.parse(edgedata_path).getroot()
    for interval in root.findall("interval"):
        for e in interval.findall("edge"):
            eid = e.attrib["id"]
            if eid in idx:
                counts[idx[eid]] += float(e.attrib.get("entered", "0"))
    return counts


# ===============================================================
#  Trajectories from vehroutes.*.xml
# ===============================================================

def _traversed_edges(vehicle_elem):
    """Reconstruct the actually-driven edge sequence. For non-rerouted
    vehicles the single <route> is the answer. For rerouted ones, walk
    through <routeDistribution>: each non-last <route> contributes its
    edges up to `replacedOnIndex`; the last contributes everything.
    Dedupe consecutive duplicates that occur when plan-i's tail and
    plan-(i+1)'s head share an edge."""
    rd = vehicle_elem.find("routeDistribution")
    if rd is None:
        route = vehicle_elem.find("route")
        if route is None:
            return []
        return route.attrib.get("edges", "").split()

    routes = rd.findall("route")
    raw = []
    for r in routes[:-1]:
        idx_str = r.attrib.get("replacedOnIndex", "0")
        idx_int = int(idx_str) if idx_str else 0
        plan = r.attrib.get("edges", "").split()
        raw.extend(plan[:idx_int + 1])
    raw.extend(routes[-1].attrib.get("edges", "").split())

    dedup = []
    for e in raw:
        if not dedup or dedup[-1] != e:
            dedup.append(e)
    return dedup


def read_trajectories(vehroutes_path, idx, min_len=5):
    trajs = []
    n_total = 0
    n_outside = 0
    n_short = 0
    root = ET.parse(vehroutes_path).getroot()
    for v in root.findall("vehicle"):
        n_total += 1
        edges = _traversed_edges(v)
        if any(e not in idx for e in edges):
            n_outside += 1
            continue
        if len(edges) < min_len:
            n_short += 1
            continue
        trajs.append(np.array([idx[e] for e in edges], dtype=int))
    return trajs, dict(total=n_total, outside_scc=n_outside,
                       too_short=n_short, kept=len(trajs))


def _scores_branching(trajs, PT_a, PT_b, out_deg, eps=1e-15):
    """LR scores with steps from forced (out-degree 1) source states masked
    out. Isolates the "real choices" signal from forced-transition noise."""
    scores = np.zeros(len(trajs), dtype=float)
    for i, t in enumerate(trajs):
        x = t[:-1]
        y = t[1:]
        mask = out_deg[x] > 1
        if not mask.any():
            continue
        pa = np.clip(PT_a[x, y], eps, None)
        pb = np.clip(PT_b[x, y], eps, None)
        scores[i] = float((np.log(pb) - np.log(pa))[mask].sum())
    return scores


def empirical_g(trajs, X_o, beta, floor=1e-3):
    """Empirical analog of morimura.true_g. Estimates the discounted
    first-hit probability matrix g[ki, kj] = E[beta^(t_hit_j - t_at_i)],
    where t_hit_j is the first time the trajectory visits X_o[kj] after
    being at X_o[ki]. Mirrors the analytical (I - beta P_T^{\\j})^{-1}
    construction in morimura.hitting_pack."""
    no = len(X_o)
    Xset = {int(x): k for k, x in enumerate(X_o)}
    hit_sum = np.zeros((no, no), dtype=float)
    visit_count = np.zeros(no, dtype=float)

    for traj in trajs:
        T = len(traj)
        for t_i in range(T):
            ki = Xset.get(int(traj[t_i]))
            if ki is None:
                continue
            visit_count[ki] += 1
            seen = set()
            for t_j in range(t_i + 1, T):
                kj = Xset.get(int(traj[t_j]))
                if kj is None or kj in seen:
                    continue
                hit_sum[ki, kj] += beta ** (t_j - t_i)
                seen.add(kj)
                if len(seen) == no:
                    break

    h = hit_sum / np.maximum(visit_count[:, None], 1.0)
    return np.clip(h, floor, 1.0)


def empirical_chain(trajs, adj_out, smoothing=1e-3):
    """Per-source-state empirical transition matrix from observed
    (x_t, x_{t+1}) counts. Laplace-smoothed over legal out-neighbors so
    that log-likelihoods are finite for unseen transitions. Self-loop on
    dead-end states (shouldn't occur within an SCC)."""
    n = len(adj_out)
    counts = np.zeros((n, n), dtype=float)
    for t in trajs:
        np.add.at(counts, (t[:-1], t[1:]), 1.0)
    PT = np.zeros((n, n), dtype=float)
    for x in range(n):
        nbrs = adj_out[x]
        if not nbrs:
            PT[x, x] = 1.0
            continue
        row = counts[x, nbrs] + smoothing
        PT[x, nbrs] = row / row.sum()
    return PT


# ===============================================================
#  Driver
# ===============================================================

def main(seed=42, out_path="sumo_phase3_fig.png", T_target=20):
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    print("=== State graph ===")
    edges, adj_out, idx = build_adj("net.net.xml")
    n = len(edges)
    out_deg = float(np.mean([len(a) for a in adj_out]))
    print(f"  n = {n}, mean out-degree = {out_deg:.2f}")

    print("\n=== Edge counts ===")
    f_intr = read_edge_counts("edgedata.intr.xml", idx)
    f_satnav = read_edge_counts("edgedata.satnav.xml", idx)
    print(f"  intr   total entered = {f_intr.sum():.0f}, "
          f"nonzero = {int((f_intr > 0).sum())}/{n}")
    print(f"  satnav total entered = {f_satnav.sum():.0f}, "
          f"nonzero = {int((f_satnav > 0).sum())}/{n}")

    print("\n=== Trajectories ===")
    trajs_intr, stats_i = read_trajectories("vehroutes.intr.xml", idx)
    trajs_satnav, stats_s = read_trajectories("vehroutes.satnav.xml", idx)
    print(f"  intr   {stats_i}")
    print(f"  satnav {stats_s}")
    if not trajs_intr or not trajs_satnav:
        raise RuntimeError("No usable trajectories after filtering.")
    mean_len = float(np.mean([len(t) for t in trajs_intr + trajs_satnav]))
    print(f"  combined mean length = {mean_len:.1f} edges")

    long_intr = [t for t in trajs_intr if len(t) >= T_target + 1]
    long_satnav = [t for t in trajs_satnav if len(t) >= T_target + 1]
    print(f"  trajectories with len >= {T_target + 1}: "
          f"intr={len(long_intr)}, satnav={len(long_satnav)}")
    if not long_intr or not long_satnav:
        raise RuntimeError(f"Not enough long trajectories at T={T_target}.")
    n_test = min(300, len(long_intr), len(long_satnav))
    rng = np.random.default_rng(seed)
    rng.shuffle(long_intr)
    rng.shuffle(long_satnav)
    test_intr_full = long_intr[:n_test]
    test_satnav_full = long_satnav[:n_test]
    test_intr = [t[: T_target + 1] for t in test_intr_full]
    test_satnav = [t[: T_target + 1] for t in test_satnav_full]
    test_intr_ids = {id(t) for t in test_intr_full}
    test_satnav_ids = {id(t) for t in test_satnav_full}
    train_intr = [t for t in trajs_intr if id(t) not in test_intr_ids]
    train_satnav = [t for t in trajs_satnav if id(t) not in test_satnav_ids]
    print(f"  test set: {n_test} per class at T = {T_target}")
    print(f"  train pool: intr={len(train_intr)}, satnav={len(train_satnav)}")

    beta = float(max(0.5, min(0.99, 1.0 - 1.0 / mean_len)))
    print(f"\n=== Inversion ===")
    print(f"  beta = {beta:.3f}")

    cands = np.where((f_intr > 0) & (f_satnav > 0))[0]
    obs_frac = 0.25  # was 0.10 in Phase 3 toy; raised after empirical AUC 0.72 vs fitted 0.53
    n_obs = min(max(20, int(obs_frac * n)), len(cands))
    X_o = np.sort(rng.choice(cands, size=n_obs, replace=False))
    print(f"  |X_o| = {len(X_o)} ({100 * obs_frac:.0f}% of n, from edges with traffic in both regimes)")

    print("  fitting intrinsic chain (f-only) ...")
    _, PT_intr_h, _ = fit_chain(adj_out, beta, X_o, f_intr[X_o])
    print("  fitting sat-nav chain (f-only) ...")
    _, PT_satnav_h, _ = fit_chain(adj_out, beta, X_o, f_satnav[X_o])

    print("  estimating empirical g (hitting rates) ...")
    g_intr_emp = empirical_g(train_intr, X_o, beta)
    g_satnav_emp = empirical_g(train_satnav, X_o, beta)
    floor = 1e-3
    frac_floor_intr = float((g_intr_emp <= floor + 1e-12).mean())
    frac_floor_satnav = float((g_satnav_emp <= floor + 1e-12).mean())
    print(f"    g_intr   range: {g_intr_emp.min():.3f} .. {g_intr_emp.max():.3f}, "
          f"mean = {g_intr_emp.mean():.3f}, "
          f"{100 * frac_floor_intr:.0f}% at floor")
    print(f"    g_satnav range: {g_satnav_emp.min():.3f} .. {g_satnav_emp.max():.3f}, "
          f"mean = {g_satnav_emp.mean():.3f}, "
          f"{100 * frac_floor_satnav:.0f}% at floor")
    # Empirical g is sparse: at |X_o|=171, T~22, most cells see <2 hits and
    # both regimes' g look near-identical at the floor. The toy used
    # gamma=0.1 (90% weight on g) because true_g is analytic and clean;
    # for noisy empirical g, raise gamma to put most of the weight back on f.
    gamma_fg = 0.5
    print(f"  fitting intrinsic chain (f+g, gamma={gamma_fg}) ...")
    _, PT_intr_fg, _ = fit_chain(adj_out, beta, X_o, f_intr[X_o],
                                 g_intr_emp, gamma=gamma_fg)
    print(f"  fitting sat-nav chain (f+g, gamma={gamma_fg}) ...")
    _, PT_satnav_fg, _ = fit_chain(adj_out, beta, X_o, f_satnav[X_o],
                                   g_satnav_emp, gamma=gamma_fg)

    print("\n=== Detection ===")
    all_test = test_intr + test_satnav
    scores = _scores_for_chains(all_test, PT_intr_h, PT_satnav_h)
    scores_fg = _scores_for_chains(all_test, PT_intr_fg, PT_satnav_fg)
    labels = np.concatenate([np.zeros(n_test, dtype=int),
                             np.ones(n_test, dtype=int)])
    fpr, tpr, auc = roc(scores, labels)
    fpr_fg, tpr_fg, auc_fg = roc(scores_fg, labels)
    print(f"  fitted_f  AUC = {auc:.3f}")
    print(f"  fitted_fg AUC = {auc_fg:.3f}")

    print("\n=== Diagnostic: empirical-chain upper bound ===")
    print("  building empirical PT from training trajectories ...")
    PT_intr_emp = empirical_chain(train_intr, adj_out)
    PT_satnav_emp = empirical_chain(train_satnav, adj_out)
    scores_emp = _scores_for_chains(all_test, PT_intr_emp, PT_satnav_emp)
    fpr_emp, tpr_emp, auc_emp = roc(scores_emp, labels)
    print(f"  empirical AUC = {auc_emp:.3f}")

    print("\n=== Diagnostic: branching-only scoring ===")
    out_deg = np.array([len(a) for a in adj_out])
    n_branching = int((out_deg > 1).sum())
    print(f"  branching states (out-deg > 1): "
          f"{n_branching}/{n} ({100 * n_branching / n:.0f}%)")
    avg_branch_steps = float(np.mean(
        [(out_deg[t[:-1]] > 1).sum() for t in all_test]))
    print(f"  avg branching steps per test traj: "
          f"{avg_branch_steps:.1f} / {T_target}")
    scores_b_emp = _scores_branching(all_test, PT_intr_emp, PT_satnav_emp, out_deg)
    scores_b_fit = _scores_branching(all_test, PT_intr_h, PT_satnav_h, out_deg)
    fpr_be, tpr_be, auc_be = roc(scores_b_emp, labels)
    fpr_bf, tpr_bf, auc_bf = roc(scores_b_fit, labels)
    print(f"  empirical (branching-only) AUC = {auc_be:.3f}")
    print(f"  fitted_f  (branching-only) AUC = {auc_bf:.3f}")

    if max(auc_emp, auc_be) > 0.8 and auc < 0.7:
        diag = "inversion is the bottleneck (raise |X_o| or add g)"
    elif max(auc_emp, auc_be) < 0.65:
        diag = "regimes barely differ at the 1-step Markov level"
    else:
        diag = "intermediate - regimes differ but signal is modest"
    print(f"  diagnosis: {diag}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(fpr_emp, tpr_emp, color="C2", lw=2, linestyle="-",
            label=f"empirical (ceiling)  AUC = {auc_emp:.3f}")
    ax.plot(fpr_fg, tpr_fg, color="C1", lw=2.5, linestyle="-",
            label=f"fitted_fg            AUC = {auc_fg:.3f}")
    ax.plot(fpr, tpr, color="C0", lw=2, linestyle="-",
            label=f"fitted_f             AUC = {auc:.3f}")
    ax.plot(fpr_be, tpr_be, color="C2", lw=1.4, linestyle="--",
            label=f"empirical (branching)  AUC = {auc_be:.3f}")
    ax.plot(fpr_bf, tpr_bf, color="C0", lw=1.4, linestyle="--",
            label=f"fitted_f  (branching)  AUC = {auc_bf:.3f}")
    ax.plot([0, 1], [0, 1], "k:", alpha=0.4)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(
        f"SUMO per-trajectory detector\n"
        f"n={n}, |X_o|={len(X_o)}, T={T_target}, {n_test} trajs/class"
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    bins = np.linspace(scores_fg.min(), scores_fg.max(), 35)
    ax.hist(scores_fg[labels == 0], bins=bins, alpha=0.6, color="C0",
            label=f"intrinsic (n={n_test})")
    ax.hist(scores_fg[labels == 1], bins=bins, alpha=0.6, color="C3",
            label=f"sat-nav (n={n_test})")
    ax.axvline(0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel(
        r"$\Lambda(\tau) = \log L_{\mathrm{sat\,nav}} - \log L_{\mathrm{intr}}$"
    )
    ax.set_ylabel("# trajectories")
    ax.set_title("LR score distribution (fitted_fg)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"\nsaved figure: {out_path}")


if __name__ == "__main__":
    main()
