from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    import networkx as nx
except ImportError as e:
    raise ImportError(
        "This script uses networkx for BA graphs. Install it with: pip install networkx"
    ) from e


# ============================================================
# PARAMETERS
# ============================================================

RNG_SEED = 12345

# Network size / target mean degree
N = 400
K_MEAN = 11.0

# Dynamic SI model on a flickering-edge network
BETA_DYNAMIC = 1.0
OMEGA_ON = 0.2
OMEGA_OFF = 0.2

# Fitting / evaluation
Q_MIN_FIT = 0.05
Q_MAX_GRID = np.array([0.80, 0.90, 0.95, 1.00])
Q_MAX_REPORT = 0.90

# Original single-grid optimisation
BETA_GRID = np.arange(0.4, 0.4901, 0.0025)

# Replication
N_REALISATIONS = 10
N_DYNAMIC_RUNS = 60
N_STATIC_RUNS_PER_BETA = 60

# Time grid for comparing mean I(t)
N_TIME_GRID = 350

# 2D geometric graph
USE_PERIODIC_GEOMETRY = False


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class GraphData:
    n: int
    edges_u: np.ndarray
    edges_v: np.ndarray
    nbrs: List[np.ndarray]
    inc_edges: List[np.ndarray]


@dataclass
class FitResult:
    q_max_fit: float
    q_max_report: float
    ignored_tail_fraction_fit: float
    best_beta: float
    mse_fit_window: float
    l1_fit_window: float
    max_abs_fit_window: float
    mse_report_window: float
    l1_report_window: float
    max_abs_report_window: float


# ============================================================
# UTILS
# ============================================================

def make_rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(seed)


def trapz_l1(y1: np.ndarray, y2: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapz(np.abs(y1 - y2), x))


def mse(y1: np.ndarray, y2: np.ndarray) -> float:
    d = y1 - y2
    return float(np.mean(d * d))


def max_abs(y1: np.ndarray, y2: np.ndarray) -> float:
    return float(np.max(np.abs(y1 - y2)))


def build_graphdata_from_edge_list(n: int, edge_list: List[Tuple[int, int]]) -> GraphData:
    edge_list = [(min(u, v), max(u, v)) for (u, v) in edge_list if u != v]
    edge_list = sorted(set(edge_list))

    m = len(edge_list)
    edges_u = np.empty(m, dtype=np.int32)
    edges_v = np.empty(m, dtype=np.int32)

    nbrs_tmp = [[] for _ in range(n)]
    inc_tmp = [[] for _ in range(n)]

    for ei, (u, v) in enumerate(edge_list):
        edges_u[ei] = u
        edges_v[ei] = v

        nbrs_tmp[u].append(v)
        nbrs_tmp[v].append(u)

        inc_tmp[u].append(ei)
        inc_tmp[v].append(ei)

    nbrs = [np.array(lst, dtype=np.int32) for lst in nbrs_tmp]
    inc_edges = [np.array(lst, dtype=np.int32) for lst in inc_tmp]

    return GraphData(
        n=n,
        edges_u=edges_u,
        edges_v=edges_v,
        nbrs=nbrs,
        inc_edges=inc_edges,
    )


# ============================================================
# GRAPH GENERATORS
# ============================================================

def generate_er_graph(n: int, k_mean: float, rng: np.random.Generator) -> GraphData:
    p = k_mean / (n - 1)
    edge_list: List[Tuple[int, int]] = []

    for i in range(n - 1):
        r = rng.random(n - 1 - i)
        js = np.where(r < p)[0] + i + 1
        edge_list.extend((i, int(j)) for j in js)

    return build_graphdata_from_edge_list(n, edge_list)


def generate_ba_graph(n: int, k_mean: float, rng: np.random.Generator) -> GraphData:
    m_attach = max(1, int(round(k_mean / 2.0)))
    seed = int(rng.integers(0, 2**31 - 1))
    g = nx.barabasi_albert_graph(n, m_attach, seed=seed)
    edge_list = list(g.edges())
    return build_graphdata_from_edge_list(n, edge_list)


def generate_2d_geometric_graph(
    n: int,
    k_mean: float,
    rng: np.random.Generator,
    periodic: bool = False,
) -> GraphData:
    r = math.sqrt(k_mean / (math.pi * n))
    pts = rng.random((n, 2))

    edge_list: List[Tuple[int, int]] = []
    for i in range(n - 1):
        dx = pts[i + 1:, 0] - pts[i, 0]
        dy = pts[i + 1:, 1] - pts[i, 1]

        if periodic:
            dx = np.minimum(np.abs(dx), 1.0 - np.abs(dx))
            dy = np.minimum(np.abs(dy), 1.0 - np.abs(dy))
        else:
            dx = np.abs(dx)
            dy = np.abs(dy)

        dist2 = dx * dx + dy * dy
        js = np.where(dist2 <= r * r)[0] + i + 1
        edge_list.extend((i, int(j)) for j in js)

    return build_graphdata_from_edge_list(n, edge_list)


# ============================================================
# STATIC SI ON FIXED GRAPH
# ============================================================

def simulate_static_si(
    graph: GraphData,
    beta: float,
    rng: np.random.Generator,
    seed_node: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    n = graph.n
    if seed_node is None:
        seed_node = int(rng.integers(0, n))

    infected = np.zeros(n, dtype=bool)
    infected[seed_node] = True

    inf_nbr_count = np.zeros(n, dtype=np.int32)
    for nb in graph.nbrs[seed_node]:
        inf_nbr_count[nb] += 1

    total_si = int(np.sum(inf_nbr_count[~infected]))
    if total_si <= 0:
        return np.array([0.0]), np.array([1.0 / n])

    times = [0.0]
    fracs = [1.0 / n]
    t = 0.0
    infected_count = 1

    while infected_count < n and total_si > 0:
        t += rng.exponential(1.0 / (beta * total_si))

        weights = inf_nbr_count.astype(float)
        weights[infected] = 0.0
        csum = np.cumsum(weights)
        r = rng.random() * csum[-1]
        x = int(np.searchsorted(csum, r, side="right"))

        infected[x] = True
        infected_count += 1

        total_si -= int(inf_nbr_count[x])

        for nb in graph.nbrs[x]:
            if not infected[nb]:
                inf_nbr_count[nb] += 1
                total_si += 1

        times.append(t)
        fracs.append(infected_count / n)

    return np.array(times, dtype=float), np.array(fracs, dtype=float)


# ============================================================
# O(1) DENSE SET
# ============================================================

class IndexSet:
    def __init__(self, size: int):
        self.items = np.empty(size, dtype=np.int32)
        self.pos = -np.ones(size, dtype=np.int32)
        self.count = 0

    def add(self, x: int) -> None:
        if self.pos[x] != -1:
            return
        self.items[self.count] = x
        self.pos[x] = self.count
        self.count += 1

    def remove(self, x: int) -> None:
        p = self.pos[x]
        if p == -1:
            return
        last = self.items[self.count - 1]
        self.items[p] = last
        self.pos[last] = p
        self.pos[x] = -1
        self.count -= 1

    def contains(self, x: int) -> bool:
        return self.pos[x] != -1

    def random_choice(self, rng: np.random.Generator) -> int:
        idx = int(rng.integers(0, self.count))
        return int(self.items[idx])


# ============================================================
# FAST DYNAMIC SI: TRACK ONLY SI EDGES
# ============================================================

def simulate_dynamic_si(
    graph: GraphData,
    beta: float,
    omega_on: float,
    omega_off: float,
    rng: np.random.Generator,
    seed_node: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact SI simulation for an underlying graph with independent edge ON/OFF switching,
    assuming the edge-state process is in stationarity.

    Only current SI edges are tracked.
    When a new SI edge is created, it is ON with probability
    rho = omega_on / (omega_on + omega_off).
    """
    n = graph.n
    m = len(graph.edges_u)

    if seed_node is None:
        seed_node = int(rng.integers(0, n))

    rho = omega_on / (omega_on + omega_off)

    infected = np.zeros(n, dtype=bool)
    infected[seed_node] = True

    si_on = IndexSet(m)
    si_off = IndexSet(m)

    # initial SI frontier from seed
    for e in graph.inc_edges[seed_node]:
        u = int(graph.edges_u[e])
        v = int(graph.edges_v[e])
        other = v if u == seed_node else u

        if infected[other]:
            continue

        if rng.random() < rho:
            si_on.add(int(e))
        else:
            si_off.add(int(e))

    times = [0.0]
    fracs = [1.0 / n]
    t = 0.0
    infected_count = 1

    while infected_count < n:
        n_on = si_on.count
        n_off = si_off.count

        rate_inf = beta * n_on
        rate_on = omega_on * n_off
        rate_off = omega_off * n_on
        total_rate = rate_inf + rate_on + rate_off

        if total_rate <= 0.0:
            break

        t += rng.exponential(1.0 / total_rate)
        r = rng.random() * total_rate

        # infection along ON SI edge
        if r < rate_inf:
            e = si_on.random_choice(rng)

            u = int(graph.edges_u[e])
            v = int(graph.edges_v[e])

            x = u if not infected[u] else v
            infected[x] = True
            infected_count += 1

            # all SI edges touching x disappear
            for ee in graph.inc_edges[x]:
                si_on.remove(int(ee))
                si_off.remove(int(ee))

            # new SI edges from x to susceptible neighbours
            for ee in graph.inc_edges[x]:
                uu = int(graph.edges_u[ee])
                vv = int(graph.edges_v[ee])
                y = vv if uu == x else uu

                if infected[y]:
                    continue

                if rng.random() < rho:
                    si_on.add(int(ee))
                else:
                    si_off.add(int(ee))

            times.append(t)
            fracs.append(infected_count / n)
            continue

        r -= rate_inf

        # OFF -> ON
        if r < rate_on:
            e = si_off.random_choice(rng)
            si_off.remove(e)
            si_on.add(e)
            continue

        # ON -> OFF
        e = si_on.random_choice(rng)
        si_on.remove(e)
        si_off.add(e)

    return np.array(times, dtype=float), np.array(fracs, dtype=float)


# ============================================================
# CURVE PROCESSING
# ============================================================

def step_curve_on_grid(times: np.ndarray, fracs: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(times, t_grid, side="right") - 1
    idx = np.clip(idx, 0, len(fracs) - 1)
    return fracs[idx]


def average_curves_on_grid(
    runs: List[Tuple[np.ndarray, np.ndarray]],
    n_grid: int = N_TIME_GRID,
) -> Tuple[np.ndarray, np.ndarray]:
    t_end = max(times[-1] for times, _ in runs)
    t_grid = np.linspace(0.0, t_end, n_grid)

    vals = np.zeros((len(runs), n_grid), dtype=float)
    for i, (times, fracs) in enumerate(runs):
        vals[i] = step_curve_on_grid(times, fracs, t_grid)

    return t_grid, vals.mean(axis=0)


def align_two_mean_curves(
    t1: np.ndarray,
    y1: np.ndarray,
    t2: np.ndarray,
    y2: np.ndarray,
    n_grid: int = N_TIME_GRID,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_end = max(t1[-1], t2[-1])
    t_grid = np.linspace(0.0, t_end, n_grid)

    y1a = np.interp(t_grid, t1, y1, left=y1[0], right=y1[-1])
    y2a = np.interp(t_grid, t2, y2, left=y2[0], right=y2[-1])
    return t_grid, y1a, y2a


def metric_bundle_window(
    t_grid: np.ndarray,
    y_dyn: np.ndarray,
    y_stat: np.ndarray,
    q_min: float,
    q_max: float,
) -> Tuple[float, float, float]:
    mask = (y_dyn >= q_min) & (y_dyn <= q_max)

    if not np.any(mask):
        return float("inf"), float("inf"), float("inf")

    x = t_grid[mask]
    yd = y_dyn[mask]
    ys = y_stat[mask]

    return mse(yd, ys), trapz_l1(yd, ys, x), max_abs(yd, ys)


# ============================================================
# EXPERIMENT HELPERS
# ============================================================

def run_many_dynamic(
    graph: GraphData,
    n_runs: int,
    beta_dynamic: float,
    omega_on: float,
    omega_off: float,
    rng: np.random.Generator,
    seed_node: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    return [
        simulate_dynamic_si(
            graph=graph,
            beta=beta_dynamic,
            omega_on=omega_on,
            omega_off=omega_off,
            rng=rng,
            seed_node=seed_node,
        )
        for _ in range(n_runs)
    ]


def run_many_static(
    graph: GraphData,
    n_runs: int,
    beta_eff: float,
    rng: np.random.Generator,
    seed_node: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    return [
        simulate_static_si(
            graph=graph,
            beta=beta_eff,
            rng=rng,
            seed_node=seed_node,
        )
        for _ in range(n_runs)
    ]


def make_graph(topology: str, n: int, k_mean: float, rng: np.random.Generator) -> GraphData:
    topology = topology.upper()
    if topology == "ER":
        return generate_er_graph(n, k_mean, rng)
    if topology == "BA":
        return generate_ba_graph(n, k_mean, rng)
    if topology in ("2DG", "2D", "GEOM"):
        return generate_2d_geometric_graph(n, k_mean, rng, periodic=USE_PERIODIC_GEOMETRY)
    raise ValueError(f"Unknown topology: {topology}")


def summarise_results(results: List[FitResult]) -> None:
    print("\nPer-cutoff results", flush=True)
    print("=" * 124, flush=True)
    print(
        f"{'q_fit':>8} {'q_rep':>8} {'ignore%':>8} {'beta*':>8} "
        f"{'MSE(fit)':>14} {'L1(fit)':>14} {'MaxAbs(fit)':>14} "
        f"{'MSE(rep)':>14} {'L1(rep)':>14} {'MaxAbs(rep)':>14}",
        flush=True,
    )
    print("-" * 124, flush=True)

    for r in results:
        print(
            f"{r.q_max_fit:8.2f} "
            f"{r.q_max_report:8.2f} "
            f"{100.0 * r.ignored_tail_fraction_fit:8.1f} "
            f"{r.best_beta:8.3f} "
            f"{r.mse_fit_window:14.8f} "
            f"{r.l1_fit_window:14.8f} "
            f"{r.max_abs_fit_window:14.8f} "
            f"{r.mse_report_window:14.8f} "
            f"{r.l1_report_window:14.8f} "
            f"{r.max_abs_report_window:14.8f}",
            flush=True,
        )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_topology_experiment(topology: str, master_rng: np.random.Generator) -> Dict[str, List[FitResult]]:
    all_realisation_results: List[List[FitResult]] = []

    print(f"\n\n{'#' * 90}", flush=True)
    print(f"TOPOLOGY: {topology}", flush=True)
    print(f"{'#' * 90}", flush=True)

    for real_idx in range(N_REALISATIONS):
        print(f"\n--- Realisation {real_idx + 1}/{N_REALISATIONS} ---", flush=True)
        rng = make_rng(int(master_rng.integers(0, 2**31 - 1)))

        # ONE graph for this realisation
        graph = make_graph(topology, N, K_MEAN, rng)
        # ONE seed for this realisation
        seed_node = int(rng.integers(0, graph.n))

        print(
            f"Graph edges = {len(graph.edges_u)}, seed = {seed_node}, seed degree = {len(graph.nbrs[seed_node])}",
            flush=True,
        )

        # ONE dynamic reference batch for this realisation
        dynamic_runs = run_many_dynamic(
            graph=graph,
            n_runs=N_DYNAMIC_RUNS,
            beta_dynamic=BETA_DYNAMIC,
            omega_on=OMEGA_ON,
            omega_off=OMEGA_OFF,
            rng=rng,
            seed_node=seed_node,
        )
        t_dyn, y_dyn = average_curves_on_grid(dynamic_runs)

        results_this_realisation: List[FitResult] = []

        for q_max in Q_MAX_GRID:
            best_beta = None
            best_mse_fit = float("inf")
            best_l1_fit = float("inf")
            best_ma_fit = float("inf")

            best_mse_report = float("inf")
            best_l1_report = float("inf")
            best_ma_report = float("inf")

            for beta_eff in BETA_GRID:
                static_runs = run_many_static(
                    graph=graph,
                    n_runs=N_STATIC_RUNS_PER_BETA,
                    beta_eff=float(beta_eff),
                    rng=rng,
                    seed_node=seed_node,
                )
                t_stat, y_stat = average_curves_on_grid(static_runs)

                t_grid, yd, ys = align_two_mean_curves(t_dyn, y_dyn, t_stat, y_stat)

                # metrics on FIT window
                mse_fit, l1_fit, ma_fit = metric_bundle_window(
                    t_grid=t_grid,
                    y_dyn=yd,
                    y_stat=ys,
                    q_min=Q_MIN_FIT,
                    q_max=float(q_max),
                )

                # metrics on common REPORT window
                mse_report, l1_report, ma_report = metric_bundle_window(
                    t_grid=t_grid,
                    y_dyn=yd,
                    y_stat=ys,
                    q_min=Q_MIN_FIT,
                    q_max=float(Q_MAX_REPORT),
                )

                # optimise on fit-window MSE only
                if mse_fit < best_mse_fit:
                    best_beta = float(beta_eff)

                    best_mse_fit = float(mse_fit)
                    best_l1_fit = float(l1_fit)
                    best_ma_fit = float(ma_fit)

                    best_mse_report = float(mse_report)
                    best_l1_report = float(l1_report)
                    best_ma_report = float(ma_report)

            assert best_beta is not None

            results_this_realisation.append(
                FitResult(
                    q_max_fit=float(q_max),
                    q_max_report=float(Q_MAX_REPORT),
                    ignored_tail_fraction_fit=float(1.0 - q_max),
                    best_beta=best_beta,
                    mse_fit_window=best_mse_fit,
                    l1_fit_window=best_l1_fit,
                    max_abs_fit_window=best_ma_fit,
                    mse_report_window=best_mse_report,
                    l1_report_window=best_l1_report,
                    max_abs_report_window=best_ma_report,
                )
            )

        summarise_results(results_this_realisation)
        all_realisation_results.append(results_this_realisation)

    # Aggregate across realisations
    q_to_results: Dict[float, List[FitResult]] = {}
    for rr in all_realisation_results:
        for r in rr:
            q_to_results.setdefault(r.q_max_fit, []).append(r)

    aggregate: List[FitResult] = []
    for q in sorted(q_to_results):
        rs = q_to_results[q]
        aggregate.append(
            FitResult(
                q_max_fit=q,
                q_max_report=Q_MAX_REPORT,
                ignored_tail_fraction_fit=1.0 - q,
                best_beta=float(np.mean([r.best_beta for r in rs])),
                mse_fit_window=float(np.mean([r.mse_fit_window for r in rs])),
                l1_fit_window=float(np.mean([r.l1_fit_window for r in rs])),
                max_abs_fit_window=float(np.mean([r.max_abs_fit_window for r in rs])),
                mse_report_window=float(np.mean([r.mse_report_window for r in rs])),
                l1_report_window=float(np.mean([r.l1_report_window for r in rs])),
                max_abs_report_window=float(np.mean([r.max_abs_report_window for r in rs])),
            )
        )

    print(f"\n\n{'=' * 90}", flush=True)
    print(f"AGGREGATED OVER {N_REALISATIONS} REALISATIONS: {topology}", flush=True)
    print(f"{'=' * 90}", flush=True)
    summarise_results(aggregate)

    return {"aggregate": aggregate}


def main() -> None:
    print("STARTED MAIN", flush=True)
    master_rng = make_rng(RNG_SEED)

    for topology in ["ER", "BA", "2DG"]:
        print(f"\nABOUT TO RUN {topology}", flush=True)
        run_topology_experiment(topology, master_rng)
        print(f"FINISHED {topology}", flush=True)


if __name__ == "__main__":
    main()