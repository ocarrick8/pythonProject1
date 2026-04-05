from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np


# ============================================================
# PARAMETERS
# ============================================================

N = 1000
K_MEAN = 6.0
P_EDGE = K_MEAN / (N - 1)

# Dynamic model
BETA_DYNAMIC = 1.0
MU = 1.0

# Repeats on the SAME graph realisation
N_DYNAMIC_RUNS = 500

# Beta search grid for static fit
BETA_MIN = 0.43
BETA_MAX = 0.47
BETA_STEP = 0.002
BETA_GRID = np.arange(BETA_MIN, BETA_MAX + 0.5 * BETA_STEP, BETA_STEP)

# For each cutoff, repeat optimisation several times and average/median beta_eff
N_OPT_REPEATS_PER_CUTOFF = 3

# Static runs used inside EACH optimisation repeat
N_STATIC_RUNS_PER_OPT = 200

# Final static runs at median beta_eff
N_STATIC_RUNS_FINAL = 1000

# Output time grid for <I(t)>
T_MAX = 12.0
DT_OUT = 0.05
T_GRID = np.arange(0.0, T_MAX + 0.5 * DT_OUT, DT_OUT)

THRESHOLDS = {
    "T80": 0.80,
    "T90": 0.90,
    "T95": 0.95,
    "T97": 0.97,
    "T99": 0.99,
}

# Fix seed node if desired; otherwise random each run
FIXED_SEED_NODE = None

MASTER_SEED = 123456789


# ============================================================
# FENWICK TREE
# ============================================================

class FenwickTree:
    """
    0-indexed wrapper around a 1-indexed internal Fenwick tree.
    Stores nonnegative weights and supports:
      - add(i, delta)
      - sum_all()
      - find_by_cumulative(target)
    """
    def __init__(self, size: int):
        self.n = size
        self.bit = np.zeros(size + 1, dtype=np.int64)

    def reset(self) -> None:
        self.bit.fill(0)

    def build_from_int_array(self, arr: np.ndarray) -> None:
        self.reset()
        for i, val in enumerate(arr):
            if val:
                self.add(i, int(val))

    def add(self, idx: int, delta: int) -> None:
        i = idx + 1
        n = self.n
        while i <= n:
            self.bit[i] += delta
            i += i & -i

    def prefix_sum(self, idx: int) -> int:
        s = 0
        i = idx + 1
        while i > 0:
            s += self.bit[i]
            i -= i & -i
        return int(s)

    def sum_all(self) -> int:
        return self.prefix_sum(self.n - 1) if self.n > 0 else 0

    def find_by_cumulative(self, target_1_based: int) -> int:
        """
        Return smallest idx such that prefix_sum(idx) >= target_1_based,
        assuming 1 <= target_1_based <= sum_all().
        """
        idx = 0
        bitmask = 1 << (self.n.bit_length() - 1)
        while bitmask:
            t = idx + bitmask
            if t <= self.n and self.bit[t] < target_1_based:
                idx = t
                target_1_based -= int(self.bit[t])
            bitmask >>= 1
        return idx  # already 0-indexed


# ============================================================
# GRAPH BUILDING
# ============================================================

@dataclass
class Graph:
    n: int
    m: int
    edge_u: np.ndarray              # shape (m,)
    edge_v: np.ndarray              # shape (m,)
    incident_edges: List[np.ndarray]  # incident edge ids per node


def make_er_graph(n: int, p: float, rng: np.random.Generator) -> Graph:
    edge_u = []
    edge_v = []
    incident = [[] for _ in range(n)]

    eid = 0
    for i in range(n - 1):
        draws = rng.random(n - i - 1) < p
        js = np.nonzero(draws)[0] + i + 1
        for j in js:
            edge_u.append(i)
            edge_v.append(j)
            incident[i].append(eid)
            incident[j].append(eid)
            eid += 1

    return Graph(
        n=n,
        m=eid,
        edge_u=np.asarray(edge_u, dtype=np.int32),
        edge_v=np.asarray(edge_v, dtype=np.int32),
        incident_edges=[np.asarray(lst, dtype=np.int32) for lst in incident],
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def choose_seed_node(rng: np.random.Generator, n: int, fixed_seed_node) -> int:
    if fixed_seed_node is None:
        return int(rng.integers(0, n))
    return int(fixed_seed_node)


def mean_curve_from_sorted_times(
    sorted_time_runs: List[np.ndarray],
    t_grid: np.ndarray,
    n: int,
) -> np.ndarray:
    out = np.zeros_like(t_grid, dtype=float)
    for times in sorted_time_runs:
        out += np.searchsorted(times, t_grid, side="right") / n
    return out / len(sorted_time_runs)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = y_true - y_pred
    return float(np.mean(d * d))


def l1_area(y_true: np.ndarray, y_pred: np.ndarray, t_grid: np.ndarray) -> float:
    return float(np.trapezoid(np.abs(y_true - y_pred), x=t_grid))


def max_abs_diff(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.max(np.abs(y_true - y_pred)))


def cutoff_index_from_mean_curve(curve: np.ndarray, threshold_fraction: float) -> int:
    idx = int(np.searchsorted(curve, threshold_fraction, side="left"))
    return min(idx, len(curve) - 1)


def truncated_curve(curve: np.ndarray, cutoff_index: int) -> np.ndarray:
    return curve[:cutoff_index + 1]


def truncated_grid(t_grid: np.ndarray, cutoff_index: int) -> np.ndarray:
    return t_grid[:cutoff_index + 1]


# ============================================================
# STATIC SI VIA GILLESPIE + FENWICK TREE
# ============================================================

def simulate_static_si_times(
    graph: Graph,
    beta_eff: float,
    rng: np.random.Generator,
    fixed_seed_node=None,
) -> np.ndarray:
    """
    Exact continuous-time SI on a static graph.
    Each SI edge transmits at rate beta_eff.
    Fenwick tree stores weights 1 for current SI edges, 0 otherwise.
    """
    n, m = graph.n, graph.m
    edge_u, edge_v = graph.edge_u, graph.edge_v
    incident = graph.incident_edges

    infected = np.zeros(n, dtype=np.bool_)
    inf_time = np.full(n, np.inf, dtype=float)

    # edge weight = 1 if edge is currently SI, else 0
    si_edge_weight = np.zeros(m, dtype=np.int8)
    ft = FenwickTree(m)

    seed = choose_seed_node(rng, n, fixed_seed_node)
    infected[seed] = True
    inf_time[seed] = 0.0
    t = 0.0
    infected_count = 1

    # initialise SI edges adjacent to seed
    for e in incident[seed]:
        si_edge_weight[e] = 1
        ft.add(int(e), 1)

    while infected_count < n:
        n_si = ft.sum_all()
        if n_si == 0:
            # disconnected component case
            break

        t += rng.exponential(1.0 / (beta_eff * n_si))

        # choose transmitting SI edge uniformly
        r = int(rng.integers(1, n_si + 1))
        e = ft.find_by_cumulative(r)

        u = int(edge_u[e])
        v = int(edge_v[e])

        # exactly one endpoint should be infected
        if infected[u] and not infected[v]:
            new_node = v
        elif infected[v] and not infected[u]:
            new_node = u
        else:
            # defensive: should never happen if weights are maintained correctly
            continue

        infected[new_node] = True
        inf_time[new_node] = t
        infected_count += 1

        # update all incident edges of newly infected node
        for ee in incident[new_node]:
            a = int(edge_u[ee])
            b = int(edge_v[ee])
            is_si_now = int(infected[a] ^ infected[b])  # XOR
            old = int(si_edge_weight[ee])

            if is_si_now != old:
                si_edge_weight[ee] = is_si_now
                ft.add(int(ee), is_si_now - old)

    return np.sort(inf_time)


# ============================================================
# DYNAMIC SI VIA GILLESPIE + FENWICK TREE
# ============================================================

def simulate_dynamic_si_times(
    graph: Graph,
    beta_dynamic: float,
    mu: float,
    rng: np.random.Generator,
    fixed_seed_node=None,
) -> np.ndarray:
    """
    Exact dynamic on/off-edge SI process on a fixed underlying graph.

    Edge state:
      active <-> inactive at rate mu

    Infection:
      each ACTIVE SI edge transmits at rate beta_dynamic

    Key fact:
      every edge toggles at rate mu regardless of current state,
      so total edge-toggle rate is always mu * m.
    """
    n, m = graph.n, graph.m
    edge_u, edge_v = graph.edge_u, graph.edge_v
    incident = graph.incident_edges

    infected = np.zeros(n, dtype=np.bool_)
    inf_time = np.full(n, np.inf, dtype=float)

    # initial active/inactive states
    active = (rng.random(m) < 0.5)

    # edge weight = 1 if currently ACTIVE and SI, else 0
    active_si_weight = np.zeros(m, dtype=np.int8)
    ft = FenwickTree(m)

    seed = choose_seed_node(rng, n, fixed_seed_node)
    infected[seed] = True
    inf_time[seed] = 0.0
    infected_count = 1
    t = 0.0

    # initialise ACTIVE SI edges adjacent to seed
    for e in incident[seed]:
        if active[e]:
            active_si_weight[e] = 1
            ft.add(int(e), 1)

    toggle_rate_total = mu * m

    while infected_count < n:
        n_active_si = ft.sum_all()
        total_rate = toggle_rate_total + beta_dynamic * n_active_si

        if total_rate <= 0.0:
            break

        t += rng.exponential(1.0 / total_rate)

        # choose event type
        if rng.random() < (beta_dynamic * n_active_si) / total_rate:
            # ------------------------------------------------
            # infection event: choose active SI edge uniformly
            # ------------------------------------------------
            if n_active_si == 0:
                continue

            r = int(rng.integers(1, n_active_si + 1))
            e = ft.find_by_cumulative(r)

            u = int(edge_u[e])
            v = int(edge_v[e])

            if infected[u] and not infected[v]:
                new_node = v
            elif infected[v] and not infected[u]:
                new_node = u
            else:
                # defensive
                continue

            infected[new_node] = True
            inf_time[new_node] = t
            infected_count += 1

            # update all incident edges of newly infected node
            for ee in incident[new_node]:
                a = int(edge_u[ee])
                b = int(edge_v[ee])
                is_active_si_now = int(active[ee] and (infected[a] ^ infected[b]))
                old = int(active_si_weight[ee])

                if is_active_si_now != old:
                    active_si_weight[ee] = is_active_si_now
                    ft.add(int(ee), is_active_si_now - old)

        else:
            # ------------------------------------------------
            # edge toggle event: choose one edge uniformly
            # ------------------------------------------------
            e = int(rng.integers(0, m))
            active[e] = ~active[e]

            u = int(edge_u[e])
            v = int(edge_v[e])
            is_active_si_now = int(active[e] and (infected[u] ^ infected[v]))
            old = int(active_si_weight[e])

            if is_active_si_now != old:
                active_si_weight[e] = is_active_si_now
                ft.add(e, is_active_si_now - old)

    return np.sort(inf_time)


# ============================================================
# MEAN-CURVE RUNNERS
# ============================================================

def run_dynamic_mean_curve(
    graph: Graph,
    n_runs: int,
    t_grid: np.ndarray,
    beta_dynamic: float,
    mu: float,
    base_seed: int,
    fixed_seed_node=None,
) -> np.ndarray:
    master = np.random.default_rng(base_seed)
    runs = []

    for _ in range(n_runs):
        rng = np.random.default_rng(int(master.integers(0, 2**63 - 1)))
        times = simulate_dynamic_si_times(
            graph=graph,
            beta_dynamic=beta_dynamic,
            mu=mu,
            rng=rng,
            fixed_seed_node=fixed_seed_node,
        )
        runs.append(times)

    return mean_curve_from_sorted_times(runs, t_grid, graph.n)


def run_static_mean_curve(
    graph: Graph,
    n_runs: int,
    t_grid: np.ndarray,
    beta_eff: float,
    base_seed: int,
    fixed_seed_node=None,
) -> np.ndarray:
    master = np.random.default_rng(base_seed)
    runs = []

    for _ in range(n_runs):
        rng = np.random.default_rng(int(master.integers(0, 2**63 - 1)))
        times = simulate_static_si_times(
            graph=graph,
            beta_eff=beta_eff,
            rng=rng,
            fixed_seed_node=fixed_seed_node,
        )
        runs.append(times)

    return mean_curve_from_sorted_times(runs, t_grid, graph.n)


# ============================================================
# OPTIMISATION
# ============================================================

def optimise_beta_for_cutoff_once(
    graph: Graph,
    t_grid: np.ndarray,
    dynamic_mean_curve: np.ndarray,
    cutoff_index: int,
    beta_grid: np.ndarray,
    n_static_runs_per_opt: int,
    base_seed: int,
    fixed_seed_node=None,
) -> float:
    dyn_trunc = truncated_curve(dynamic_mean_curve, cutoff_index)

    best_beta = None
    best_obj = np.inf

    master = np.random.default_rng(base_seed)

    for beta_eff in beta_grid:
        stat_seed = int(master.integers(0, 2**63 - 1))
        stat_curve = run_static_mean_curve(
            graph=graph,
            n_runs=n_static_runs_per_opt,
            t_grid=t_grid,
            beta_eff=float(beta_eff),
            base_seed=stat_seed,
            fixed_seed_node=fixed_seed_node,
        )
        stat_trunc = truncated_curve(stat_curve, cutoff_index)
        obj = mse(dyn_trunc, stat_trunc)

        if obj < best_obj:
            best_obj = obj
            best_beta = float(beta_eff)

    return float(best_beta)


def repeated_beta_optimisation_for_cutoff(
    graph: Graph,
    t_grid: np.ndarray,
    dynamic_mean_curve: np.ndarray,
    cutoff_index: int,
    beta_grid: np.ndarray,
    n_opt_repeats: int,
    n_static_runs_per_opt: int,
    base_seed: int,
    fixed_seed_node=None,
) -> np.ndarray:
    master = np.random.default_rng(base_seed)
    betas = np.empty(n_opt_repeats, dtype=float)

    for r in range(n_opt_repeats):
        rep_seed = int(master.integers(0, 2**63 - 1))
        betas[r] = optimise_beta_for_cutoff_once(
            graph=graph,
            t_grid=t_grid,
            dynamic_mean_curve=dynamic_mean_curve,
            cutoff_index=cutoff_index,
            beta_grid=beta_grid,
            n_static_runs_per_opt=n_static_runs_per_opt,
            base_seed=rep_seed,
            fixed_seed_node=fixed_seed_node,
        )

    return betas


# ============================================================
# ANALYSIS
# ============================================================

@dataclass
class CutoffResult:
    name: str
    threshold_fraction: float
    cutoff_time: float
    cutoff_index: int
    beta_eff_values: np.ndarray
    beta_eff_mean: float
    beta_eff_median: float
    beta_eff_std: float
    mse_at_median_beta: float
    l1_area_at_median_beta: float
    max_abs_diff_at_median_beta: float


def analyse_cutoff(
    name: str,
    threshold_fraction: float,
    graph: Graph,
    t_grid: np.ndarray,
    dynamic_mean_curve: np.ndarray,
    beta_grid: np.ndarray,
    n_opt_repeats: int,
    n_static_runs_per_opt: int,
    n_static_runs_final: int,
    base_seed: int,
    fixed_seed_node=None,
) -> CutoffResult:
    cidx = cutoff_index_from_mean_curve(dynamic_mean_curve, threshold_fraction)
    ctime = float(t_grid[cidx])

    beta_values = repeated_beta_optimisation_for_cutoff(
        graph=graph,
        t_grid=t_grid,
        dynamic_mean_curve=dynamic_mean_curve,
        cutoff_index=cidx,
        beta_grid=beta_grid,
        n_opt_repeats=n_opt_repeats,
        n_static_runs_per_opt=n_static_runs_per_opt,
        base_seed=base_seed,
        fixed_seed_node=fixed_seed_node,
    )

    beta_mean = float(np.mean(beta_values))
    beta_median = float(np.median(beta_values))
    beta_std = float(np.std(beta_values, ddof=1)) if len(beta_values) > 1 else 0.0

    final_static_curve = run_static_mean_curve(
        graph=graph,
        n_runs=n_static_runs_final,
        t_grid=t_grid,
        beta_eff=beta_median,
        base_seed=base_seed + 999999,
        fixed_seed_node=fixed_seed_node,
    )

    dyn_trunc = truncated_curve(dynamic_mean_curve, cidx)
    stat_trunc = truncated_curve(final_static_curve, cidx)
    grid_trunc = truncated_grid(t_grid, cidx)

    return CutoffResult(
        name=name,
        threshold_fraction=threshold_fraction,
        cutoff_time=ctime,
        cutoff_index=cidx,
        beta_eff_values=beta_values,
        beta_eff_mean=beta_mean,
        beta_eff_median=beta_median,
        beta_eff_std=beta_std,
        mse_at_median_beta=mse(dyn_trunc, stat_trunc),
        l1_area_at_median_beta=l1_area(dyn_trunc, stat_trunc, grid_trunc),
        max_abs_diff_at_median_beta=max_abs_diff(dyn_trunc, stat_trunc),
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 90)
    print("BUILDING ONE ER NETWORK REALISATION")
    print("=" * 90)
    graph_rng = np.random.default_rng(MASTER_SEED)
    graph = make_er_graph(N, P_EDGE, graph_rng)

    mean_degree = 2.0 * graph.m / graph.n
    print(f"N                     = {graph.n}")
    print(f"Number of edges       = {graph.m}")
    print(f"Mean degree           = {mean_degree:.6f}")
    print(f"Fixed seed node       = {FIXED_SEED_NODE}")
    print()

    print("=" * 90)
    print("RUNNING DYNAMIC PROCESS TO ESTIMATE <I(t)>")
    print("=" * 90)
    print(f"N_DYNAMIC_RUNS        = {N_DYNAMIC_RUNS}")
    dynamic_mean_curve = run_dynamic_mean_curve(
        graph=graph,
        n_runs=N_DYNAMIC_RUNS,
        t_grid=T_GRID,
        beta_dynamic=BETA_DYNAMIC,
        mu=MU,
        base_seed=MASTER_SEED + 1,
        fixed_seed_node=FIXED_SEED_NODE,
    )
    print("Done.")
    print()

    results: Dict[str, CutoffResult] = {}

    for i, (name, frac) in enumerate(THRESHOLDS.items(), start=1):
        print("=" * 90)
        print(f"ANALYSING {name} (fraction = {frac:.2f})")
        print("=" * 90)

        res = analyse_cutoff(
            name=name,
            threshold_fraction=frac,
            graph=graph,
            t_grid=T_GRID,
            dynamic_mean_curve=dynamic_mean_curve,
            beta_grid=BETA_GRID,
            n_opt_repeats=N_OPT_REPEATS_PER_CUTOFF,
            n_static_runs_per_opt=N_STATIC_RUNS_PER_OPT,
            n_static_runs_final=N_STATIC_RUNS_FINAL,
            base_seed=MASTER_SEED + 10000 * i,
            fixed_seed_node=FIXED_SEED_NODE,
        )
        results[name] = res

        print(f"dynamic cutoff time                 = {res.cutoff_time:.6f}")
        print(f"mean(beta_eff)                      = {res.beta_eff_mean:.6f}")
        print(f"median(beta_eff)                    = {res.beta_eff_median:.6f}")
        print(f"std(beta_eff)                       = {res.beta_eff_std:.6f}")
        print(f"MSE at median(beta_eff)             = {res.mse_at_median_beta:.10f}")
        print(f"L1 area at median(beta_eff)         = {res.l1_area_at_median_beta:.10f}")
        print(f"max abs diff at median(beta_eff)    = {res.max_abs_diff_at_median_beta:.10f}")
        print()

    print("=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)
    for name in THRESHOLDS:
        r = results[name]
        print(
            f"{name}: cutoff={r.cutoff_time:.6f}, "
            f"mean(beta_eff)={r.beta_eff_mean:.6f}, "
            f"median(beta_eff)={r.beta_eff_median:.6f}, "
            f"std(beta_eff)={r.beta_eff_std:.6f}, "
            f"MSE={r.mse_at_median_beta:.10f}, "
            f"L1={r.l1_area_at_median_beta:.10f}, "
            f"MaxAbs={r.max_abs_diff_at_median_beta:.10f}"
        )


if __name__ == "__main__":
    main()