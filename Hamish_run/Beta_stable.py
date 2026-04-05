import math
import multiprocessing as mp
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

#test, i like penis

# ============================================================
# PARAMETERS
# ============================================================

# ER network
N = 1000
K_MEAN = 6.0
P_EDGE = K_MEAN / (N - 1)

# Dynamic model
BETA_DYNAMIC = 1.0
MU = 1.0

# Time grid
DT_OUT = 0.05
T_MAX = 14.0
T_GRID = np.arange(0.0, T_MAX + DT_OUT, DT_OUT)

# Match <I(t)> only up to this mean prevalence cutoff
MATCH_CUTOFF = 0.95

# How many graph realisations to analyse
N_GRAPH_REALISATIONS = 5

# Dynamic repeats per graph
N_DYNAMIC_RUNS = 200

# Beta search
BETA_MIN = 0.410
BETA_MAX = 0.481
BETA_COARSE_STEP = 0.02
BETA_FINE_STEP = 0.002
FINE_HALF_WIDTH = 0.02  # +/- around coarse optimum

# Static repeats used during optimisation
N_STATIC_RUNS_COARSE = 20
N_STATIC_RUNS_FINE = 30

# Repeat the whole optimisation several times on the SAME graph
# then average the resulting optimum betas
N_OPT_REPEATS_PER_GRAPH = 2

# Final verification at averaged beta
N_STATIC_RUNS_VERIFY = 200

# Objective
OBJECTIVE = "mse"   # "mse" or "mae"

# Randomness / parallel
MASTER_SEED = 12345
N_JOBS = max(1, mp.cpu_count() - 1)
FORCE_SERIAL = False


# ============================================================
# FENWICK TREE
# ============================================================

class FenwickTree:
    def __init__(self, n: int):
        self.n = n
        self.bit = np.zeros(n + 1, dtype=np.float64)

    def build_from(self, arr: np.ndarray):
        self.bit[:] = 0.0
        for i, val in enumerate(arr, start=1):
            j = i
            while j <= self.n:
                self.bit[j] += val
                j += j & -j

    def update(self, idx0: int, delta: float):
        i = idx0 + 1
        while i <= self.n:
            self.bit[i] += delta
            i += i & -i

    def total(self) -> float:
        s = 0.0
        i = self.n
        while i > 0:
            s += self.bit[i]
            i -= i & -i
        return s

    def find_prefix_gt(self, x: float) -> int:
        idx = 0
        bitmask = 1 << (self.n.bit_length() - 1)
        cur = 0.0
        while bitmask:
            nxt = idx + bitmask
            if nxt <= self.n and cur + self.bit[nxt] <= x:
                idx = nxt
                cur += self.bit[nxt]
            bitmask >>= 1
        return idx


# ============================================================
# GRAPH DATA
# ============================================================

@dataclass
class GraphData:
    n: int
    m: int
    edges_u: np.ndarray
    edges_v: np.ndarray
    incident_edges: list


def build_connected_er_graph(n: int, p: float, rng: np.random.Generator) -> GraphData:
    trial = 0
    while True:
        trial += 1
        edges_u = []
        edges_v = []
        incident = [[] for _ in range(n)]

        for i in range(n):
            r = rng.random(n - i - 1)
            js = np.where(r < p)[0] + i + 1
            for j in js:
                eidx = len(edges_u)
                edges_u.append(i)
                edges_v.append(j)
                incident[i].append(eidx)
                incident[j].append(eidx)

        m = len(edges_u)
        if m == 0:
            continue

        seen = np.zeros(n, dtype=bool)
        stack = [0]
        seen[0] = True
        while stack:
            u = stack.pop()
            for eidx in incident[u]:
                a = edges_u[eidx]
                b = edges_v[eidx]
                v = b if a == u else a
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)

        if seen.all():
            print(f"Connected ER graph found after {trial} trial(s).")
            return GraphData(
                n=n,
                m=m,
                edges_u=np.array(edges_u, dtype=np.int32),
                edges_v=np.array(edges_v, dtype=np.int32),
                incident_edges=incident,
            )


# ============================================================
# UTILITIES
# ============================================================

def choose_seed_nodes(n_runs: int, n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=n_runs, endpoint=False)


def infection_times_to_curve(infection_times: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    sorted_times = np.sort(infection_times)
    counts = np.searchsorted(sorted_times, t_grid, side="right")
    return counts.astype(np.float64)


def mean_curve_from_runs(infection_time_runs: list, t_grid: np.ndarray, n: int) -> np.ndarray:
    curves = np.empty((len(infection_time_runs), len(t_grid)), dtype=np.float64)
    for i, times in enumerate(infection_time_runs):
        curves[i] = infection_times_to_curve(times, t_grid)
    return curves.mean(axis=0) / n


def first_time_mean_reaches(mean_curve: np.ndarray, t_grid: np.ndarray, q: float) -> float:
    idx = np.where(mean_curve >= q)[0]
    if len(idx) == 0:
        return t_grid[-1]
    i = idx[0]
    if i == 0:
        return t_grid[0]
    x0, x1 = t_grid[i - 1], t_grid[i]
    y0, y1 = mean_curve[i - 1], mean_curve[i]
    if y1 == y0:
        return x1
    w = (q - y0) / (y1 - y0)
    return x0 + w * (x1 - x0)


def truncated_objective(dynamic_mean: np.ndarray, static_mean: np.ndarray,
                        t_grid: np.ndarray, t_cut: float, mode: str) -> float:
    mask = t_grid <= t_cut
    d = dynamic_mean[mask] - static_mean[mask]
    if mode == "mse":
        return np.mean(d * d)
    elif mode == "mae":
        return np.mean(np.abs(d))
    else:
        raise ValueError("mode must be 'mse' or 'mae'")


# ============================================================
# EDGE WEIGHTS
# ============================================================

def compute_static_weight(u: int, v: int, infected: np.ndarray) -> float:
    return 1.0 if infected[u] != infected[v] else 0.0


def compute_dynamic_weight(u: int, v: int, infected: np.ndarray, active: np.ndarray, eidx: int) -> float:
    return 1.0 if active[eidx] and (infected[u] != infected[v]) else 0.0


# ============================================================
# SINGLE RUN SIMULATORS
# ============================================================

def run_static_si_once(graph: GraphData, beta_eff: float, seed_node: int,
                       rng: np.random.Generator) -> np.ndarray:
    n = graph.n
    m = graph.m

    infected = np.zeros(n, dtype=bool)
    infected[seed_node] = True

    infection_times = np.full(n, np.inf, dtype=np.float64)
    infection_times[seed_node] = 0.0

    weights = np.zeros(m, dtype=np.float64)
    for eidx in graph.incident_edges[seed_node]:
        u = graph.edges_u[eidx]
        v = graph.edges_v[eidx]
        weights[eidx] = compute_static_weight(u, v, infected)

    ft = FenwickTree(m)
    ft.build_from(weights)

    t = 0.0
    n_inf = 1

    while n_inf < n:
        si_count = ft.total()
        if si_count <= 0.0:
            break

        t += rng.exponential(1.0 / (beta_eff * si_count))
        x = rng.random() * si_count
        eidx = ft.find_prefix_gt(x)

        u = graph.edges_u[eidx]
        v = graph.edges_v[eidx]

        if infected[u] and not infected[v]:
            new_node = v
        elif infected[v] and not infected[u]:
            new_node = u
        else:
            continue

        infected[new_node] = True
        infection_times[new_node] = t
        n_inf += 1

        for ee in graph.incident_edges[new_node]:
            a = graph.edges_u[ee]
            b = graph.edges_v[ee]
            old_w = weights[ee]
            new_w = compute_static_weight(a, b, infected)
            if new_w != old_w:
                ft.update(ee, new_w - old_w)
                weights[ee] = new_w

    return infection_times


def run_dynamic_si_once(graph: GraphData, beta_dynamic: float, mu: float,
                        seed_node: int, rng: np.random.Generator) -> np.ndarray:
    n = graph.n
    m = graph.m

    infected = np.zeros(n, dtype=bool)
    infected[seed_node] = True

    infection_times = np.full(n, np.inf, dtype=np.float64)
    infection_times[seed_node] = 0.0

    active = rng.random(m) < 0.5

    weights = np.zeros(m, dtype=np.float64)
    for eidx in range(m):
        u = graph.edges_u[eidx]
        v = graph.edges_v[eidx]
        weights[eidx] = compute_dynamic_weight(u, v, infected, active, eidx)

    ft = FenwickTree(m)
    ft.build_from(weights)

    t = 0.0
    n_inf = 1

    while n_inf < n:
        active_si = ft.total()
        total_rate = mu * m + beta_dynamic * active_si
        if total_rate <= 0.0:
            break

        t += rng.exponential(1.0 / total_rate)

        if rng.random() < (mu * m) / total_rate:
            eidx = int(rng.integers(0, m))
            active[eidx] = ~active[eidx]

            u = graph.edges_u[eidx]
            v = graph.edges_v[eidx]
            old_w = weights[eidx]
            new_w = compute_dynamic_weight(u, v, infected, active, eidx)
            if new_w != old_w:
                ft.update(eidx, new_w - old_w)
                weights[eidx] = new_w
        else:
            if active_si <= 0.0:
                continue

            x = rng.random() * active_si
            eidx = ft.find_prefix_gt(x)

            u = graph.edges_u[eidx]
            v = graph.edges_v[eidx]

            if infected[u] and not infected[v]:
                new_node = v
            elif infected[v] and not infected[u]:
                new_node = u
            else:
                continue

            infected[new_node] = True
            infection_times[new_node] = t
            n_inf += 1

            for ee in graph.incident_edges[new_node]:
                a = graph.edges_u[ee]
                b = graph.edges_v[ee]
                old_w = weights[ee]
                new_w = compute_dynamic_weight(a, b, infected, active, ee)
                if new_w != old_w:
                    ft.update(ee, new_w - old_w)
                    weights[ee] = new_w

    return infection_times


# ============================================================
# MULTIPROCESS BATCH RUNNERS
# ============================================================

def _dynamic_worker(run_idx, seed_node, graph, beta_dynamic, mu, base_seed):
    rng = np.random.default_rng(base_seed + 1000003 * run_idx)
    return run_dynamic_si_once(graph, beta_dynamic, mu, seed_node, rng)


def _static_worker(run_idx, seed_node, graph, beta_eff, base_seed):
    rng = np.random.default_rng(base_seed + 1000003 * run_idx)
    return run_static_si_once(graph, beta_eff, seed_node, rng)


def run_dynamic_batch(graph: GraphData, n_runs: int, beta_dynamic: float, mu: float,
                      seed_nodes: np.ndarray, base_seed: int):
    if FORCE_SERIAL or N_JOBS == 1:
        return [
            _dynamic_worker(i, int(seed_nodes[i]), graph, beta_dynamic, mu, base_seed)
            for i in range(n_runs)
        ]

    args = [(i, int(seed_nodes[i]), graph, beta_dynamic, mu, base_seed)
            for i in range(n_runs)]

    with mp.Pool(N_JOBS) as pool:
        runs = pool.starmap(_dynamic_worker, args)

    return runs


def run_static_batch(graph: GraphData, n_runs: int, beta_eff: float,
                     seed_nodes: np.ndarray, base_seed: int):
    if FORCE_SERIAL or N_JOBS == 1:
        return [
            _static_worker(i, int(seed_nodes[i]), graph, beta_eff, base_seed)
            for i in range(n_runs)
        ]

    args = [(i, int(seed_nodes[i]), graph, beta_eff, base_seed)
            for i in range(n_runs)]

    with mp.Pool(N_JOBS) as pool:
        runs = pool.starmap(_static_worker, args)

    return runs


# ============================================================
# OPTIMISATION RESULTS
# ============================================================

@dataclass
class SingleOptResult:
    best_beta: float
    best_obj: float
    coarse_best_beta: float
    fine_betas: np.ndarray
    fine_objs: np.ndarray


@dataclass
class GraphResult:
    graph_index: int
    m_edges: int
    mean_degree: float
    t_cut: float
    opt_betas: np.ndarray
    opt_beta_mean: float
    opt_beta_std: float
    verify_obj: float


# ============================================================
# BETA SEARCH PER GRAPH
# ============================================================

def optimise_once_for_graph(graph: GraphData,
                            dynamic_mean_curve: np.ndarray,
                            t_cut: float,
                            rep_seed: int) -> SingleOptResult:
    seed_rng = np.random.default_rng(rep_seed)

    coarse_grid = np.arange(BETA_MIN, BETA_MAX + 0.5 * BETA_COARSE_STEP, BETA_COARSE_STEP)
    static_seed_nodes_coarse = choose_seed_nodes(N_STATIC_RUNS_COARSE, graph.n, seed_rng)

    coarse_objs = []
    for beta in coarse_grid:
        static_runs = run_static_batch(
            graph=graph,
            n_runs=N_STATIC_RUNS_COARSE,
            beta_eff=float(beta),
            seed_nodes=static_seed_nodes_coarse,
            base_seed=rep_seed + int(round(beta * 1_000_000))
        )
        static_mean = mean_curve_from_runs(static_runs, T_GRID, graph.n)
        obj = truncated_objective(dynamic_mean_curve, static_mean, T_GRID, t_cut, OBJECTIVE)
        coarse_objs.append(obj)

    coarse_objs = np.array(coarse_objs)
    i_coarse = int(np.argmin(coarse_objs))
    coarse_best_beta = float(coarse_grid[i_coarse])

    fine_lo = max(BETA_MIN, coarse_best_beta - FINE_HALF_WIDTH)
    fine_hi = min(BETA_MAX, coarse_best_beta + FINE_HALF_WIDTH)
    fine_grid = np.arange(fine_lo, fine_hi + 0.5 * BETA_FINE_STEP, BETA_FINE_STEP)

    static_seed_nodes_fine = choose_seed_nodes(N_STATIC_RUNS_FINE, graph.n, seed_rng)

    fine_objs = []
    for beta in fine_grid:
        static_runs = run_static_batch(
            graph=graph,
            n_runs=N_STATIC_RUNS_FINE,
            beta_eff=float(beta),
            seed_nodes=static_seed_nodes_fine,
            base_seed=rep_seed + 5000000 + int(round(beta * 1_000_000))
        )
        static_mean = mean_curve_from_runs(static_runs, T_GRID, graph.n)
        obj = truncated_objective(dynamic_mean_curve, static_mean, T_GRID, t_cut, OBJECTIVE)
        fine_objs.append(obj)

    fine_objs = np.array(fine_objs)
    best_idx = int(np.argmin(fine_objs))
    best_beta = float(fine_grid[best_idx])
    best_obj = float(fine_objs[best_idx])

    return SingleOptResult(
        best_beta=best_beta,
        best_obj=best_obj,
        coarse_best_beta=coarse_best_beta,
        fine_betas=fine_grid,
        fine_objs=fine_objs,
    )


def analyse_one_graph(graph_index: int, master_seed: int) -> GraphResult:
    print("\n" + "=" * 80)
    print(f"GRAPH REALISATION {graph_index + 1} / {N_GRAPH_REALISATIONS}")
    print("=" * 80)

    graph_rng = np.random.default_rng(master_seed + 100000 * graph_index)
    graph = build_connected_er_graph(N, P_EDGE, graph_rng)

    mean_degree = 2.0 * graph.m / graph.n
    print(f"N = {graph.n}, M = {graph.m}, <k> = {mean_degree:.4f}")

    # Dynamic runs on this graph
    dyn_rng = np.random.default_rng(master_seed + 100000 * graph_index + 11111)
    dynamic_seed_nodes = choose_seed_nodes(N_DYNAMIC_RUNS, graph.n, dyn_rng)

    print("Running dynamic simulations...")
    dynamic_runs = run_dynamic_batch(
        graph=graph,
        n_runs=N_DYNAMIC_RUNS,
        beta_dynamic=BETA_DYNAMIC,
        mu=MU,
        seed_nodes=dynamic_seed_nodes,
        base_seed=master_seed + 100000 * graph_index + 22222
    )
    dynamic_mean_curve = mean_curve_from_runs(dynamic_runs, T_GRID, graph.n)

    t_cut = first_time_mean_reaches(dynamic_mean_curve, T_GRID, MATCH_CUTOFF)
    print(f"Dynamic mean reaches q={MATCH_CUTOFF:.2f} at t={t_cut:.4f}")

    # Repeat optimisation several times on same graph
    opt_results = []
    for rep in range(N_OPT_REPEATS_PER_GRAPH):
        print(f"  optimisation repeat {rep + 1} / {N_OPT_REPEATS_PER_GRAPH}")
        rep_seed = master_seed + 100000 * graph_index + 5000000 * (rep + 1)
        opt = optimise_once_for_graph(graph, dynamic_mean_curve, t_cut, rep_seed)
        opt_results.append(opt)
        print(
            f"    best beta = {opt.best_beta:.4f} | "
            f"best obj = {opt.best_obj:.8e}"
        )

    opt_betas = np.array([r.best_beta for r in opt_results], dtype=float)
    opt_beta_mean = float(np.mean(opt_betas))
    opt_beta_std = float(np.std(opt_betas, ddof=1)) if len(opt_betas) > 1 else 0.0

    # Final verification at averaged beta
    verify_rng = np.random.default_rng(master_seed + 100000 * graph_index + 88888)
    verify_seed_nodes = choose_seed_nodes(N_STATIC_RUNS_VERIFY, graph.n, verify_rng)
    verify_runs = run_static_batch(
        graph=graph,
        n_runs=N_STATIC_RUNS_VERIFY,
        beta_eff=opt_beta_mean,
        seed_nodes=verify_seed_nodes,
        base_seed=master_seed + 100000 * graph_index + 99999
    )
    verify_mean_curve = mean_curve_from_runs(verify_runs, T_GRID, graph.n)
    verify_obj = truncated_objective(dynamic_mean_curve, verify_mean_curve, T_GRID, t_cut, OBJECTIVE)

    print(f"  mean optimum beta = {opt_beta_mean:.6f}")
    print(f"  std optimum beta  = {opt_beta_std:.6f}")
    print(f"  verification obj  = {verify_obj:.8e}")

    return GraphResult(
        graph_index=graph_index,
        m_edges=graph.m,
        mean_degree=mean_degree,
        t_cut=t_cut,
        opt_betas=opt_betas,
        opt_beta_mean=opt_beta_mean,
        opt_beta_std=opt_beta_std,
        verify_obj=verify_obj,
    )


# ============================================================
# SUMMARY AND PLOTTING
# ============================================================

def summarise_results(results: list[GraphResult]):
    betas_all = np.array([r.opt_beta_mean for r in results], dtype=float)
    within_sds = np.array([r.opt_beta_std for r in results], dtype=float)

    print("\n" + "=" * 80)
    print("SUMMARY ACROSS GRAPH REALISATIONS")
    print("=" * 80)

    print(f"Total graphs analysed     = {len(results)}")
    print(f"mean(beta_eff)            = {np.mean(betas_all):.6f}")
    print(f"std(beta_eff)             = {np.std(betas_all, ddof=1):.6f}" if len(betas_all) > 1 else "std(beta_eff)             = 0.0")
    print(f"median(beta_eff)          = {np.median(betas_all):.6f}")
    print(f"mean within-graph sd      = {np.mean(within_sds):.6f}")

    print("\nPer-graph results:")
    for r in results:
        print(
            f"Graph {r.graph_index + 1:02d} | "
            f"<k>={r.mean_degree:.4f} | "
            f"t_cut={r.t_cut:.4f} | "
            f"beta_mean={r.opt_beta_mean:.6f} | "
            f"beta_sd={r.opt_beta_std:.6f} | "
            f"verify_obj={r.verify_obj:.8e}"
        )


def plot_results(results: list[GraphResult]):
    x = np.arange(1, len(results) + 1)
    beta_means = np.array([r.opt_beta_mean for r in results], dtype=float)
    beta_sds = np.array([r.opt_beta_std for r in results], dtype=float)

    plt.figure(figsize=(10, 5.5))
    plt.errorbar(
        x,
        beta_means,
        yerr=beta_sds,
        fmt="o",
        capsize=4,
        label="graph-wise mean optimum beta"
    )
    plt.xlabel("Graph realisation")
    plt.ylabel("Optimum beta_eff")
    plt.title(f"Variation of optimum beta_eff across ER graph realisations (cutoff q={MATCH_CUTOFF:.2f})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.hist(beta_means, bins=min(10, len(results)))
    plt.xlabel("Graph-wise mean optimum beta_eff")
    plt.ylabel("Count")
    plt.title("Distribution of optimum beta_eff across graph realisations")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    print("Starting ER graph-to-graph beta_eff variability analysis")
    print(f"N = {N}, <k> = {K_MEAN}, p = {P_EDGE:.6f}")
    print(f"beta_dynamic = {BETA_DYNAMIC}, mu = {MU}")
    print(f"match cutoff q = {MATCH_CUTOFF:.2f}")
    print(f"graph realisations = {N_GRAPH_REALISATIONS}")
    print(f"dynamic runs per graph = {N_DYNAMIC_RUNS}")
    print(f"optimisation repeats per graph = {N_OPT_REPEATS_PER_GRAPH}")
    print(f"verify static runs per graph = {N_STATIC_RUNS_VERIFY}")
    print(f"parallel jobs = {N_JOBS}, force_serial = {FORCE_SERIAL}")

    results = []
    for g in range(N_GRAPH_REALISATIONS):
        res = analyse_one_graph(g, MASTER_SEED)
        results.append(res)

    summarise_results(results)
    plot_results(results)


if __name__ == "__main__":
    mp.freeze_support()
    main()