"""
load_xuancheng.py — Loader for the Xuancheng AVI dataset (Ma et al. 2026,
Scientific Data) that produces inputs compatible with the existing Phase-4
pipeline (sumo_to_phase3 / score_seed_big / contrast_correlation).

Data layout under real_data/:
  xuancheng.net.xml                                  -- SUMO road network
  Data/data/Xuancheng/data_2023_04_<DD>_type_filtered.json
                                                     -- one file per day in
                                                        April 2023, list of
                                                        trip dicts with keys
                                                        {vehicle, interval,
                                                         startTime, endTime,
                                                         route}.

Key contract: ``load_xuancheng_regime(net_path, day_specs, regime_split)``
returns the same 7-tuple shape that sumo_to_phase3-based scripts consume:

    (edges, adj_out, idx, trajs_a, trajs_b, f_a, f_b)

so the existing per_car_detector / Inverter pipeline runs unchanged.

Three design decisions worth knowing before reading the code:

1. We reuse ``sumo_to_phase3.build_adj`` for the network — it already
   filters non-internal + passenger-allowed and restricts to the largest
   SCC, which is exactly what we want here too.

2. Trip JSON ``route`` arrays contain ~22% illegal consecutive pairs
   (Dijkstra repair from the source pipeline imputed teleports that don't
   respect SUMO's lane-level connectivity). We **split trips at illegal
   boundaries** into legal-only sub-trajectories rather than dropping the
   trip wholesale; this preserves the genuine observed transitions while
   discarding the imputed cross-network jumps.

3. ``startTime`` is seconds since midnight on the trip's day (verified by
   distribution check: range 0..86400 with the canonical AM-rush /
   PM-rush double-peak pattern). All regime-split functions assume this.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
from typing import Callable, Iterable

import numpy as np

# Re-use the SUMO-Phase-3 graph builder.  Xuancheng's .net.xml has the same
# format as the SUMO bigger-bbox net used in Phase 4 (verified: 1,744 edges,
# all non-internal, all passenger-allowed).
_HERE = os.path.dirname(os.path.abspath(__file__))
_MASTERS_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_MASTERS_ROOT, "sumo_validation"))
from sumo_to_phase3 import build_adj  # noqa: E402


# ============================================================================
#  Trip reader
# ============================================================================

def read_trips_with_meta(json_path: str,
                         idx: dict,
                         adj_out: list,
                         min_segment_len: int = 5):
    """Parse one daily JSON, split each route at illegal-pair boundaries.

    Returns a list of dicts, one per legal sub-trajectory:
        {"traj":       np.ndarray of edge indices into idx (length >= min_segment_len),
         "start_time": int seconds since midnight,
         "vehicle":    raw vehicle attribute dict from the JSON,
         "trip_id":    int index of the source trip in the JSON file}

    Sub-trajectories are produced by splitting the original route at any
    consecutive (a, b) pair where b is not in adj_out[a].
    """
    with open(json_path) as f:
        trips_raw = json.load(f)

    # Pre-compute a set of legal (src_idx, dst_idx) pairs for O(1) lookup.
    legal = {(x, y) for x, ys in enumerate(adj_out) for y in ys}

    sub_trips = []
    n_raw_pairs = 0
    n_legal_pairs = 0
    for trip_id, trip in enumerate(trips_raw):
        route_ids = trip["route"]
        # Map to edge indices, dropping IDs not in idx (non-SCC or non-existent).
        route_idx = []
        for rid in route_ids:
            i = idx.get(rid)
            route_idx.append(i)   # None for unknown IDs; handled below

        # Walk the route, splitting at illegal / missing transitions.
        segment = []
        for k in range(len(route_idx)):
            cur = route_idx[k]
            if cur is None:
                # unknown edge -> flush current segment and skip
                if len(segment) >= min_segment_len:
                    sub_trips.append({
                        "traj":       np.asarray(segment, dtype=int),
                        "start_time": int(trip["startTime"]),
                        "vehicle":    trip.get("vehicle"),
                        "trip_id":    trip_id,
                    })
                segment = []
                continue
            if segment and (segment[-1], cur) not in legal:
                # illegal transition: flush, start fresh from `cur`
                n_raw_pairs += 1
                if len(segment) >= min_segment_len:
                    sub_trips.append({
                        "traj":       np.asarray(segment, dtype=int),
                        "start_time": int(trip["startTime"]),
                        "vehicle":    trip.get("vehicle"),
                        "trip_id":    trip_id,
                    })
                segment = [cur]
                continue
            if segment:
                n_raw_pairs += 1
                n_legal_pairs += 1
            segment.append(cur)
        # Flush trailing segment.
        if len(segment) >= min_segment_len:
            sub_trips.append({
                "traj":       np.asarray(segment, dtype=int),
                "start_time": int(trip["startTime"]),
                "vehicle":    trip.get("vehicle"),
                "trip_id":    trip_id,
            })

    stats = {
        "raw_trips":           len(trips_raw),
        "consecutive_pairs":   n_raw_pairs,
        "legal_pairs":         n_legal_pairs,
        "legal_pair_frac":     (n_legal_pairs / n_raw_pairs) if n_raw_pairs else 0.0,
        "sub_trajectories":    len(sub_trips),
        "min_segment_len":     min_segment_len,
    }
    return sub_trips, stats


# ============================================================================
#  Regime-split functions
# ============================================================================

def split_rush_vs_offpeak(
    start_time: int,
    rush_windows: Iterable[tuple] = ((25200, 32400), (61200, 68400)),
    offpeak_window: tuple = (36000, 57600),
) -> str | None:
    """Return 'rush' if start_time is in any rush window,
       'offpeak' if it is in the off-peak window, else None.

    Defaults match the verified Xuancheng AM-rush (7-9 AM = 25200-32400s),
    PM-rush (5-7 PM = 61200-68400s), and off-peak (10 AM-4 PM = 36000-57600s)
    windows derived from the startTime histogram.
    """
    for lo, hi in rush_windows:
        if lo <= start_time < hi:
            return "rush"
    lo, hi = offpeak_window
    if lo <= start_time < hi:
        return "offpeak"
    return None


def split_weekday_vs_weekend(date: datetime.date) -> str | None:
    """date: a python datetime.date.  weekday() returns 0=Mon..6=Sun.
    Returns 'weekday' for Mon-Fri, 'weekend' for Sat-Sun."""
    return "weekday" if date.weekday() < 5 else "weekend"


# Per paper §3.2: Apr 5 (Tomb-Sweeping holiday, 264k trips) and Apr 28-29
# (Labour Day eve, 426k / 424k trips) are clearly anomalous in the daily
# demand profile.  Treat those as 'holiday' and contrast against typical
# midweek workdays.
_HOLIDAY_DATES = {
    datetime.date(2023, 4, 5),
    datetime.date(2023, 4, 28),
    datetime.date(2023, 4, 29),
    datetime.date(2023, 4, 30),  # Labour Day
}
_NORMAL_DATES = {
    datetime.date(2023, 4, d) for d in (10, 11, 12, 13, 14, 17, 18, 19, 20, 21)
}


def split_holiday_vs_normal(date: datetime.date) -> str | None:
    if date in _HOLIDAY_DATES:
        return "holiday"
    if date in _NORMAL_DATES:
        return "normal"
    return None


# ============================================================================
#  OD-matching: cluster junctions, bin trips by (zone_o, zone_d), subsample
# ============================================================================

def compute_edge_endpoint_zones(net_path: str, edges: list, k_zones: int = 8,
                                random_state: int = 42):
    """K-means cluster the network's junctions by (x, y) coordinates, then map
    each edge to (from_zone, to_zone) using its endpoint nodes.

    Returns (from_zones, to_zones) — two int arrays of length len(edges).
    ``edges[i]`` is the SUMO edge ID at index ``i`` in the loader's frame.
    """
    import sys as _sys
    _sumo_home = os.environ.get("SUMO_HOME")
    if _sumo_home:
        _sys.path.insert(0, os.path.join(_sumo_home, "tools"))
    import sumolib  # noqa: E402
    from scipy.cluster.vq import kmeans2  # noqa: E402

    net = sumolib.net.readNet(net_path)
    nodes = list(net.getNodes())
    node_ids = [n.getID() for n in nodes]
    coords = np.array([list(n.getCoord()) for n in nodes], dtype=float)
    # Standardise so kmeans treats x/y on equal footing (coord scale is meters
    # but bounding box may be slightly non-square).
    coords_norm = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-9)
    _, labels = kmeans2(coords_norm, k_zones, seed=random_state, minit="++")
    node_to_zone = {nid: int(lbl) for nid, lbl in zip(node_ids, labels)}

    from_zones = np.zeros(len(edges), dtype=int)
    to_zones = np.zeros(len(edges), dtype=int)
    for i, eid in enumerate(edges):
        e = net.getEdge(eid)
        from_zones[i] = node_to_zone[e.getFromNode().getID()]
        to_zones[i] = node_to_zone[e.getToNode().getID()]
    return from_zones, to_zones


def od_match_subsample(trajs_a, trajs_b, from_zones, to_zones,
                       rng_seed: int = 42, verbose: bool = True,
                       label_a: str = "a", label_b: str = "b"):
    """Bin trajectories by (origin_zone, destination_zone) where origin =
    from-zone of the first edge and destination = to-zone of the last edge.

    For each OD cell, take ``min(n_a, n_b)`` trajectories from each regime
    (sampled without replacement). Returns the matched (trajs_a, trajs_b).

    The marginal (zone_o, zone_d) distribution of the returned lists is then
    identical between the two regimes — any chain contrast between them is
    no longer confounded by an OD-mix difference (the same concern that
    distinguishes SUMO's controlled experiment, where intrinsic and sat-nav
    runs share OD pairs by construction, from the real-world Xuancheng pool).
    """
    from collections import defaultdict
    rng = np.random.default_rng(rng_seed)

    def _bin(trajs):
        b = defaultdict(list)
        for traj in trajs:
            if len(traj) < 1:
                continue
            o = int(from_zones[traj[0]])
            d = int(to_zones[traj[-1]])
            b[(o, d)].append(traj)
        return b

    bins_a = _bin(trajs_a)
    bins_b = _bin(trajs_b)
    common = sorted(set(bins_a) & set(bins_b))

    out_a, out_b = [], []
    cell_counts = []  # for diagnostics
    for od in common:
        ta = bins_a[od]
        tb = bins_b[od]
        n = min(len(ta), len(tb))
        if n == 0:
            continue
        ia = rng.choice(len(ta), size=n, replace=False)
        ib = rng.choice(len(tb), size=n, replace=False)
        out_a.extend([ta[k] for k in ia])
        out_b.extend([tb[k] for k in ib])
        cell_counts.append((od, len(ta), len(tb), n))

    rng.shuffle(out_a)
    rng.shuffle(out_b)

    if verbose:
        only_a = set(bins_a) - set(bins_b)
        only_b = set(bins_b) - set(bins_a)
        dropped_a = sum(len(bins_a[od]) for od in only_a)
        dropped_b = sum(len(bins_b[od]) for od in only_b)
        print(f"  od-match: {len(common)} shared cells "
              f"(of {len(set(bins_a) | set(bins_b))} unique OD pairs)")
        print(f"  od-match: {label_a} {len(trajs_a):,} → {len(out_a):,} "
              f"(dropped {dropped_a:,} in non-shared cells, "
              f"plus per-cell excess)")
        print(f"  od-match: {label_b} {len(trajs_b):,} → {len(out_b):,} "
              f"(dropped {dropped_b:,} in non-shared cells, "
              f"plus per-cell excess)")
    return out_a, out_b


# ============================================================================
#  High-level entry point
# ============================================================================

def load_xuancheng_regime(
    net_path: str,
    day_specs: list[tuple[str, datetime.date]],
    regime_split: Callable[[dict, datetime.date], str | None],
    label_a: str,
    label_b: str,
    min_segment_len: int = 5,
    verbose: bool = True,
    od_match_zones: int | None = None,
    od_match_seed: int = 42,
):
    """Load multiple daily JSONs and partition sub-trajectories into two regimes.

    Parameters
    ----------
    net_path : str
        Path to xuancheng.net.xml.
    day_specs : list of (json_path, date) tuples
        One entry per day's JSON to include.  ``date`` is a python ``date``.
    regime_split : callable
        Takes (sub_trip_dict, date) and returns ``label_a`` / ``label_b`` / None.
        The sub_trip dict has keys 'traj', 'start_time', 'vehicle', 'trip_id'.
    label_a, label_b : str
        The two regime labels that ``regime_split`` returns.  Sub-trajectories
        with any other label (including None) are discarded.
    min_segment_len : int
        Minimum legal-segment length to keep (passed through to read_trips_with_meta).
    verbose : bool
        If True, prints per-day diagnostics.

    Returns
    -------
    (edges, adj_out, idx, trajs_a, trajs_b, f_a, f_b)
        Same shape as the Phase-4 SUMO pipeline expects.
        ``trajs_a`` / ``trajs_b`` are lists of np.ndarray (edge-index sequences).
        ``f_a`` / ``f_b`` are np.ndarray edge-count vectors over the full edge set.
    """
    if verbose:
        print(f"  building adj from {net_path} ...")
    edges, adj_out, idx = build_adj(net_path)
    n = len(edges)
    if verbose:
        print(f"  n = {n}, mean out-degree = "
              f"{np.mean([len(a) for a in adj_out]):.2f}")

    trajs_a, trajs_b = [], []
    f_a = np.zeros(n, dtype=float)
    f_b = np.zeros(n, dtype=float)

    for json_path, date in day_specs:
        sub_trips, stats = read_trips_with_meta(
            json_path, idx, adj_out, min_segment_len=min_segment_len,
        )
        n_a, n_b, n_other = 0, 0, 0
        for st in sub_trips:
            r = regime_split(st, date)
            if r == label_a:
                trajs_a.append(st["traj"])
                np.add.at(f_a, st["traj"], 1.0)
                n_a += 1
            elif r == label_b:
                trajs_b.append(st["traj"])
                np.add.at(f_b, st["traj"], 1.0)
                n_b += 1
            else:
                n_other += 1
        if verbose:
            print(f"  {date}  raw_trips={stats['raw_trips']:>7,}  "
                  f"legal_frac={100*stats['legal_pair_frac']:.1f}%  "
                  f"sub_trajs={stats['sub_trajectories']:>7,}  "
                  f"{label_a}={n_a:>6,}  {label_b}={n_b:>6,}  "
                  f"other={n_other:>6,}")

    if verbose:
        print(f"  total: {label_a}={len(trajs_a):,}, {label_b}={len(trajs_b):,}")
        print(f"  edges with traffic ({label_a}): "
              f"{int((f_a > 0).sum())}/{n}")
        print(f"  edges with traffic ({label_b}): "
              f"{int((f_b > 0).sum())}/{n}")

    # Optional OD-rebalancing: clusters junctions into K zones, then per
    # (zone_o, zone_d) cell takes min(n_a, n_b) trajectories from each regime.
    # The marginal OD distribution is then identical between the two regimes,
    # so any chain contrast is the routing-given-OD effect — analogous to
    # SUMO's controlled-OD comparison.
    if od_match_zones is not None:
        if verbose:
            print(f"  OD-matching: clustering junctions into K={od_match_zones}"
                  f" zones, then subsampling per (zone_o, zone_d) cell ...")
        from_zones, to_zones = compute_edge_endpoint_zones(
            net_path, edges, k_zones=od_match_zones,
            random_state=od_match_seed,
        )
        trajs_a, trajs_b = od_match_subsample(
            trajs_a, trajs_b, from_zones, to_zones,
            rng_seed=od_match_seed, verbose=verbose,
            label_a=label_a, label_b=label_b,
        )
        # Recompute f counts on the matched pools.
        f_a = np.zeros(n, dtype=float)
        f_b = np.zeros(n, dtype=float)
        for traj in trajs_a:
            np.add.at(f_a, traj, 1.0)
        for traj in trajs_b:
            np.add.at(f_b, traj, 1.0)
        if verbose:
            print(f"  after OD-matching: edges with traffic "
                  f"({label_a}): {int((f_a > 0).sum())}/{n}, "
                  f"({label_b}): {int((f_b > 0).sum())}/{n}")

    return edges, adj_out, idx, trajs_a, trajs_b, f_a, f_b


# ============================================================================
#  Convenience: pick day_specs from a directory + date list
# ============================================================================

DATA_ROOT_DEFAULT = os.path.join(_HERE, "Data", "data", "Xuancheng")
NET_PATH_DEFAULT = os.path.join(_HERE, "xuancheng.net.xml")


def day_spec(date: datetime.date, data_root: str = DATA_ROOT_DEFAULT) -> tuple[str, datetime.date]:
    """Convenience: (path, date) for a given April-2023 date."""
    fname = f"data_2023_{date.month:02d}_{date.day:02d}_type_filtered.json"
    return (os.path.join(data_root, fname), date)


# ============================================================================
#  __main__: smoke test
# ============================================================================

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--day", type=str, default="2023-04-17",
                   help="YYYY-MM-DD, defaults to mid-week 2023-04-17 (Monday).")
    p.add_argument("--split", choices=("rush_offpeak", "weekday_weekend"),
                   default="rush_offpeak")
    args = p.parse_args()

    date = datetime.date.fromisoformat(args.day)
    if args.split == "rush_offpeak":
        def rs(st, d):
            return split_rush_vs_offpeak(st["start_time"])
        la, lb = "rush", "offpeak"
    else:
        def rs(st, d):
            return split_weekday_vs_weekend(d)
        la, lb = "weekday", "weekend"

    edges, adj_out, idx, trajs_a, trajs_b, f_a, f_b = load_xuancheng_regime(
        net_path=NET_PATH_DEFAULT,
        day_specs=[day_spec(date)],
        regime_split=rs,
        label_a=la, label_b=lb,
    )
    print(f"\n  trajs_{la}: {len(trajs_a):,}, mean len = "
          f"{np.mean([len(t) for t in trajs_a]):.1f}")
    print(f"  trajs_{lb}: {len(trajs_b):,}, mean len = "
          f"{np.mean([len(t) for t in trajs_b]):.1f}")
    print(f"  f_{la}.sum() = {f_a.sum():.0f}")
    print(f"  f_{lb}.sum() = {f_b.sum():.0f}")
