import math
import multiprocessing as mp
from dataclasses import dataclass, asdict

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from scipy.stats import ks_2samp, anderson_ksamp, wasserstein_distance
from matplotlib.colors import ListedColormap, BoundaryNorm


# ============================================================
# USER PARAMETERS
# ============================================================

SEED = 12345

# Network size
N = 300

# Dynamic model
BETA_DYNAMIC = 1.0
ON_FRACTION = 0.5   # stationary fraction of time an edge is ON

# Thresholds to compare
THRESHOLDS = [0.20, 0.50, 0.80]

# Only fit / analyse up to this infected fraction
FIT_MAX_FRACTION = 0.90

# Heat-map axes
MU_VALUES = np.array([0.1, 0.3, 0.6, 1.0, 2.0])
K_VALUES = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

# Repeats
N_GRAPH_REALISATIONS = 5
N_DYNAMIC_RUNS_PER_GRAPH = 120
N_STATIC_RUNS_PER_BETA_PER_GRAPH = 60
N_STATIC_RUNS_FINAL_PER_GRAPH = 120

# Beta search grid
BETA_EFF_GRID = np.arange(0.30, 0.71, 0.04)

# Time grids
DT_OUT = 0.1
T_MAX_DEFAULT = 15.0
T_MAX_2DG = 40.0

# AD significance level
AD_ALPHA_PERCENT = 5.0

# Parallelism
USE_MULTIPROCESSING = True
N_PROCESSES = max(1, mp.cpu_count() - 1)

# 2D geometric graph boundary condition
PERIODIC_2DG = True


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class ThresholdStats:
    q: float
    median_dynamic: float
    median_static: float
    delta_median: float
    std_dynamic: float
    std_static: float
    delta_std: float
    wasserstein: float
    ks_stat: float
    ks_pvalue: float
    ad_stat: float
    ad_significance_level_percent: float
    ad_pass: bool


@dataclass
class GridPointResult:
    graph_type: str
    mu: float
    k_mean_target: float
    k_mean_realised_avg: float
    beta_eff_mean: float
    ad_fail_count: int
    threshold_stats: list


# ============================================================
# GRAPH GENERATORS
# ============================================================

def make_er_graph(n: int, k_mean: float, rng: np.random.Generator) -> nx.Graph:
    p = min(1.0, k_mean / max(1, n - 1))
    seed = int(rng.integers(0, 2**31 - 1))
    g = nx.fast_gnp_random_graph(n, p, seed=seed)
    g.remove_edges_from(nx.selfloop_edges(g))
    return g


def make_ba_graph(n: int, k_mean: float, rng: np.random.Generator) -> nx.Graph:
    m = max(1, int(round(k_mean / 2.0)))
    m = min(m, n - 1)
    seed = int(rng.integers(0, 2**31 - 1))
    g = nx.barabasi_albert_graph(n, m, seed=seed)
    return g


def torus_distance(x1, y1, x2, y2):
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    dx = min(dx, 1.0 - dx)
    dy = min(dy, 1.0 - dy)
    return math.hypot(dx, dy)


def make_rgg_graph_periodic(n: int, radius: float, rng: np.random.Generator) -> nx.Graph:
    pts = rng.random((n, 2))
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            if torus_distance(xi, yi, xj, yj) <= radius:
                g.add_edge(i, j)
    return g


def make_rgg_graph_hard(n: int, radius: float, rng: np.random.Generator) -> nx.Graph:
    pts = rng.random((n, 2))
    g = nx.random_geometric_graph(n, radius, pos={i: pts[i] for i in range(n)})
    return nx.Graph(g)


def rgg_radius_from_kmean(n: int, k_mean: float) -> float:
    val = k_mean / max(1.0, (n - 1) * math.pi)
    return math.sqrt(max(0.0, val))


def make_2dg_graph(n: int, k_mean: float, rng: np.random.Generator, periodic: bool) -> nx.Graph:
    r = rgg_radius_from_kmean(n, k_mean)
    if periodic:
        return make_rgg_graph_periodic(n, r, rng)
    return make_rgg_graph_hard(n, r, rng)


def make_graph(graph_type: str, n: int, k_mean: float, rng: np.random.Generator) -> nx.Graph:
    if graph_type == "ER":
        return make_er_graph(n, k_mean, rng)
    if graph_type == "BA":
        return make_ba_graph(n, k_mean, rng)
    if graph_type == "2DG":
        return make_2dg_graph(n, k_mean, rng, periodic=PERIODIC_2DG)
    raise ValueError(f"Unknown graph_type: {graph_type}")


# ============================================================
# TIME GRID
# ============================================================

def get_t_max(graph_type: str) -> float:
    if graph_type == "2DG":
        return T_MAX_2DG
    return T_MAX_DEFAULT


def make_t_grid(graph_type: str) -> np.ndarray:
    t_max = get_t_max(graph_type)
    return np.arange(0.0, t_max + 1e-12, DT_OUT)


# ============================================================
# PREPROCESS GRAPH
# ============================================================

def preprocess_graph(g: nx.Graph):
    n = g.number_of_nodes()
    edges = np.array(list(g.edges()), dtype=np.int32)
    m = len(edges)

    neighbors = [[] for _ in range(n)]
    edge_ids_by_node = [[] for _ in range(n)]

    for eid, (u, v) in enumerate(edges):
        neighbors[u].append(v)
        neighbors[v].append(u)
        edge_ids_by_node[u].append(eid)
        edge_ids_by_node[v].append(eid)

    neighbors = [np.array(x, dtype=np.int32) for x in neighbors]
    edge_ids_by_node = [np.array(x, dtype=np.int32) for x in edge_ids_by_node]

    return {
        "n": n,
        "edges": edges,
        "m": m,
        "neighbors": neighbors,
        "edge_ids_by_node": edge_ids_by_node,
        "k_mean_realised": 0.0 if n == 0 else (2.0 * m / n),
    }


# ============================================================
# UTILITIES
# ============================================================

def infection_curve_from_event_history(infection_times, t_grid, n_total):
    infection_times = np.sort(np.asarray(infection_times, dtype=float))
    infected_counts = np.searchsorted(infection_times, t_grid, side="right")
    return infected_counts / n_total


def extract_threshold_times_from_curve(curve, t_grid, thresholds, max_fraction=0.90):
    out = {}
    for q in thresholds:
        if q > max_fraction:
            out[q] = np.nan
            continue

        idx = np.searchsorted(curve, q, side="left")
        if idx >= len(t_grid):
            out[q] = np.nan
        else:
            out[q] = float(t_grid[idx])
    return out


def choose_seed(pre, rng):
    return int(rng.integers(0, pre["n"]))


def safe_std(x):
    return float(np.std(x, ddof=1)) if len(x) > 1 else np.nan


def ad_test_pass(x, y, alpha_percent=5.0):
    res = anderson_ksamp([x, y])
    passed = float(res.significance_level) > alpha_percent
    return passed, float(res.statistic), float(res.significance_level)


# ============================================================
# SI SIMULATORS
# ============================================================

def run_static_si_once(pre, beta, t_grid, thresholds, rng):
    n = pre["n"]
    neighbors = pre["neighbors"]
    t_max = t_grid[-1]

    infected = np.zeros(n, dtype=bool)
    infected_time = np.full(n, np.inf, dtype=float)

    seed = choose_seed(pre, rng)
    infected[seed] = True
    infected_time[seed] = 0.0
    infected_count = 1

    t = 0.0

    while infected_count < n and t < t_max:
        si_edges = []
        infected_nodes = np.where(infected)[0]

        for u in infected_nodes:
            for v in neighbors[u]:
                if not infected[v]:
                    si_edges.append((u, v))

        rate_inf = beta * len(si_edges)
        if rate_inf <= 0.0:
            break

        t += rng.exponential(1.0 / rate_inf)
        if t > t_max:
            break

        _, v = si_edges[int(rng.integers(0, len(si_edges)))]
        if not infected[v]:
            infected[v] = True
            infected_time[v] = t
            infected_count += 1

    curve = infection_curve_from_event_history(
        infected_time[np.isfinite(infected_time)],
        t_grid,
        n_total=n
    )
    th = extract_threshold_times_from_curve(
        curve,
        t_grid,
        thresholds,
        max_fraction=FIT_MAX_FRACTION
    )
    return curve, th


def run_dynamic_si_once(pre, beta, mu, t_grid, thresholds, rng):
    n = pre["n"]
    edges = pre["edges"]
    m = pre["m"]
    edge_ids_by_node = pre["edge_ids_by_node"]
    t_max = t_grid[-1]

    infected = np.zeros(n, dtype=bool)
    infected_time = np.full(n, np.inf, dtype=float)

    # initial active states: each edge ON with probability 0.5
    edge_on = rng.random(m) < ON_FRACTION

    seed = choose_seed(pre, rng)
    infected[seed] = True
    infected_time[seed] = 0.0
    infected_count = 1

    t = 0.0

    while infected_count < n and t < t_max:
        active_si_edges = []

        infected_nodes = np.where(infected)[0]
        for u in infected_nodes:
            for eid in edge_ids_by_node[u]:
                if not edge_on[eid]:
                    continue
                a, b = edges[eid]
                v = b if a == u else a
                if not infected[v]:
                    active_si_edges.append((eid, v))

        rate_inf = beta * len(active_si_edges)
        rate_toggle = mu * m
        total_rate = rate_inf + rate_toggle

        if total_rate <= 0.0:
            break

        t += rng.exponential(1.0 / total_rate)
        if t > t_max:
            break

        if rng.random() < (rate_inf / total_rate if total_rate > 0 else 0.0):
            eid, v = active_si_edges[int(rng.integers(0, len(active_si_edges)))]
            if not infected[v]:
                infected[v] = True
                infected_time[v] = t
                infected_count += 1
        else:
            eid = int(rng.integers(0, m))
            edge_on[eid] = ~edge_on[eid]

    curve = infection_curve_from_event_history(
        infected_time[np.isfinite(infected_time)],
        t_grid,
        n_total=n
    )
    th = extract_threshold_times_from_curve(
        curve,
        t_grid,
        thresholds,
        max_fraction=FIT_MAX_FRACTION
    )
    return curve, th


# ============================================================
# BATCH RUNNERS
# ============================================================

def run_many_dynamic(pre, beta, mu, n_runs, t_grid, thresholds, seed):
    rng = np.random.default_rng(seed)
    curves = []
    threshold_times = {q: [] for q in thresholds}

    for _ in range(n_runs):
        curve, th = run_dynamic_si_once(pre, beta, mu, t_grid, thresholds, rng)
        curves.append(curve)
        for q in thresholds:
            if np.isfinite(th[q]):
                threshold_times[q].append(th[q])

    return np.asarray(curves), threshold_times


def run_many_static(pre, beta, n_runs, t_grid, thresholds, seed):
    rng = np.random.default_rng(seed)
    curves = []
    threshold_times = {q: [] for q in thresholds}

    for _ in range(n_runs):
        curve, th = run_static_si_once(pre, beta, t_grid, thresholds, rng)
        curves.append(curve)
        for q in thresholds:
            if np.isfinite(th[q]):
                threshold_times[q].append(th[q])

    return np.asarray(curves), threshold_times


# ============================================================
# FIT BETA_EFF ONLY UP TO I/N = 0.9
# ============================================================

def fit_beta_eff_for_graph(pre, graph_type, mu, beta_dynamic, beta_grid, seed_base):
    t_grid = make_t_grid(graph_type)

    dyn_curves, _ = run_many_dynamic(
        pre=pre,
        beta=beta_dynamic,
        mu=mu,
        n_runs=N_DYNAMIC_RUNS_PER_GRAPH,
        t_grid=t_grid,
        thresholds=THRESHOLDS,
        seed=seed_base + 1
    )
    mean_dyn = dyn_curves.mean(axis=0)

    idx_cut = np.searchsorted(mean_dyn, FIT_MAX_FRACTION, side="left")

    if idx_cut >= len(t_grid):
        fit_mask = np.ones_like(t_grid, dtype=bool)
    else:
        fit_mask = np.zeros_like(t_grid, dtype=bool)
        fit_mask[:idx_cut + 1] = True

    t_fit = t_grid[fit_mask]
    mean_dyn_fit = mean_dyn[fit_mask]

    best_beta = None
    best_obj = np.inf

    for i, beta_eff in enumerate(beta_grid):
        st_curves, _ = run_many_static(
            pre=pre,
            beta=beta_eff,
            n_runs=N_STATIC_RUNS_PER_BETA_PER_GRAPH,
            t_grid=t_grid,
            thresholds=THRESHOLDS,
            seed=seed_base + 1000 + i
        )
        mean_st = st_curves.mean(axis=0)
        mean_st_fit = mean_st[fit_mask]

        obj = np.trapz(np.abs(mean_dyn_fit - mean_st_fit), t_fit)

        if obj < best_obj:
            best_obj = obj
            best_beta = beta_eff

    return float(best_beta), mean_dyn


# ============================================================
# GRID POINT ANALYSIS
# ============================================================

def analyse_one_grid_point(graph_type, mu, k_mean, global_seed):
    rng = np.random.default_rng(global_seed)
    t_grid = make_t_grid(graph_type)

    all_thresholds_dyn = {q: [] for q in THRESHOLDS}
    all_thresholds_st = {q: [] for q in THRESHOLDS}
    beta_effs = []
    realised_k_list = []

    for g_idx in range(N_GRAPH_REALISATIONS):
        g = make_graph(graph_type, N, k_mean, rng)
        pre = preprocess_graph(g)
        realised_k_list.append(pre["k_mean_realised"])

        beta_eff, _ = fit_beta_eff_for_graph(
            pre=pre,
            graph_type=graph_type,
            mu=mu,
            beta_dynamic=BETA_DYNAMIC,
            beta_grid=BETA_EFF_GRID,
            seed_base=global_seed + 10000 * g_idx
        )
        beta_effs.append(beta_eff)

        _, dyn_th = run_many_dynamic(
            pre=pre,
            beta=BETA_DYNAMIC,
            mu=mu,
            n_runs=N_DYNAMIC_RUNS_PER_GRAPH,
            t_grid=t_grid,
            thresholds=THRESHOLDS,
            seed=global_seed + 200000 + 10000 * g_idx
        )

        _, st_th = run_many_static(
            pre=pre,
            beta=beta_eff,
            n_runs=N_STATIC_RUNS_FINAL_PER_GRAPH,
            t_grid=t_grid,
            thresholds=THRESHOLDS,
            seed=global_seed + 400000 + 10000 * g_idx
        )

        for q in THRESHOLDS:
            all_thresholds_dyn[q].extend(dyn_th[q])
            all_thresholds_st[q].extend(st_th[q])

    threshold_stats = []
    ad_fail_count = 0

    for q in THRESHOLDS:
        x = np.asarray(all_thresholds_dyn[q], dtype=float)
        y = np.asarray(all_thresholds_st[q], dtype=float)

        if len(x) < 2 or len(y) < 2:
            ad_pass = False
            ad_stat = np.nan
            ad_sig = np.nan
            ks_stat = np.nan
            ks_pvalue = np.nan
            med_x = np.nan
            med_y = np.nan
            std_x = np.nan
            std_y = np.nan
            wdist = np.nan
            ad_fail_count += 1
        else:
            ad_pass, ad_stat, ad_sig = ad_test_pass(x, y, alpha_percent=AD_ALPHA_PERCENT)
            if not ad_pass:
                ad_fail_count += 1

            ks = ks_2samp(x, y)
            ks_stat = float(ks.statistic)
            ks_pvalue = float(ks.pvalue)
            med_x = float(np.median(x))
            med_y = float(np.median(y))
            std_x = safe_std(x)
            std_y = safe_std(y)
            wdist = float(wasserstein_distance(x, y))

        row = ThresholdStats(
            q=q,
            median_dynamic=med_x,
            median_static=med_y,
            delta_median=(med_x - med_y) if np.isfinite(med_x) and np.isfinite(med_y) else np.nan,
            std_dynamic=std_x,
            std_static=std_y,
            delta_std=(std_x - std_y) if np.isfinite(std_x) and np.isfinite(std_y) else np.nan,
            wasserstein=wdist,
            ks_stat=ks_stat,
            ks_pvalue=ks_pvalue,
            ad_stat=ad_stat,
            ad_significance_level_percent=ad_sig,
            ad_pass=bool(ad_pass),
        )
        threshold_stats.append(asdict(row))

    return GridPointResult(
        graph_type=graph_type,
        mu=float(mu),
        k_mean_target=float(k_mean),
        k_mean_realised_avg=float(np.mean(realised_k_list)),
        beta_eff_mean=float(np.mean(beta_effs)),
        ad_fail_count=int(ad_fail_count),
        threshold_stats=threshold_stats,
    )


# ============================================================
# HEAT-MAP PLOTTING
# ============================================================

def make_failcount_matrix(results, mu_values, k_values):
    mat = np.full((len(mu_values), len(k_values)), np.nan)
    for r in results:
        i = np.where(np.isclose(mu_values, r.mu))[0][0]
        j = np.where(np.isclose(k_values, r.k_mean_target))[0][0]
        mat[i, j] = r.ad_fail_count
    return mat


def plot_heatmap(fail_mat, mu_values, k_values, title):
    cmap = ListedColormap(["#2ca25f", "#ffd92f", "#fd8d3c", "#de2d26"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(fail_mat, origin="lower", aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(np.arange(len(k_values)))
    ax.set_xticklabels([f"{k:.1f}" for k in k_values])
    ax.set_yticks(np.arange(len(mu_values)))
    ax.set_yticklabels([f"{mu:.2f}" for mu in mu_values])

    ax.set_xlabel(r"Target $\langle k \rangle$")
    ax.set_ylabel(r"$\mu$")
    ax.set_title(title)

    for i in range(len(mu_values)):
        for j in range(len(k_values)):
            val = fail_mat[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{int(val)}", ha="center", va="center", fontsize=10)

    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels([
        "0 fails (green)",
        "1 fail (yellow)",
        "2 fails (orange)",
        "3 fails (red)"
    ])

    plt.tight_layout()
    plt.show()


# ============================================================
# DRIVER
# ============================================================

def run_for_graph_type(graph_type):
    tasks = []
    task_id = 0

    for mu in MU_VALUES:
        for k_mean in K_VALUES:
            seed = SEED + 10_000_000 * task_id + hash(graph_type) % 100000
            tasks.append((graph_type, float(mu), float(k_mean), int(seed)))
            task_id += 1

    if USE_MULTIPROCESSING:
        with mp.Pool(processes=N_PROCESSES) as pool:
            results = pool.starmap(analyse_one_grid_point, tasks)
    else:
        results = [analyse_one_grid_point(*task) for task in tasks]

    fail_mat = make_failcount_matrix(results, MU_VALUES, K_VALUES)

    t_max_used = get_t_max(graph_type)
    plot_heatmap(
        fail_mat,
        MU_VALUES,
        K_VALUES,
        title=f"{graph_type}: AD failures across T20, T50, T80\nfit only up to I/N = 0.9, t_max = {t_max_used:g}s"
    )

    return results, fail_mat


if __name__ == "__main__":
    all_results = {}

    for graph_type in ["ER", "BA", "2DG"]:
        print(f"\nRunning graph type: {graph_type}")
        print(f"Using t_max = {get_t_max(graph_type):.1f} s")
        results, fail_mat = run_for_graph_type(graph_type)
        all_results[graph_type] = {
            "results": results,
            "fail_mat": fail_mat,
        }

        print(f"\nSummary for {graph_type}")
        for r in results:
            print(
                f"mu={r.mu:>4.2f}, <k>={r.k_mean_target:>4.1f}, "
                f"realised<k>={r.k_mean_realised_avg:>5.2f}, "
                f"beta_eff_mean={r.beta_eff_mean:>5.3f}, "
                f"AD fails={r.ad_fail_count}"
            )
            for s in r.threshold_stats:
                print(
                    f"   T{int(100*s['q'])}: "
                    f"dmed={s['delta_median']:+.4f}, "
                    f"dstd={s['delta_std']:+.4f}, "
                    f"W={s['wasserstein']:.4f}, "
                    f"KS={s['ks_stat']:.4f}, "
                    f"p={s['ks_pvalue']:.4g}, "
                    f"AD={s['ad_stat']:.4f}, "
                    f"AD_pass={s['ad_pass']}"
                )