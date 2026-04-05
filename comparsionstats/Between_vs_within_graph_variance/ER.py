import math
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, anderson_ksamp, wasserstein_distance


# ============================================================
# USER PARAMETERS
# ============================================================

# -------------------------
# graph ensemble
# -------------------------
N = 1000
K_MEAN = 6.0
P_ER = K_MEAN / (N - 1)

N_GRAPHS = 10
GRAPH_MASTER_SEED = 123456

# -------------------------
# epidemic model
# -------------------------
MU = 1.0
BETA_DYNAMIC = 1.0

# -------------------------
# nested repeat counts
# -------------------------
DYNAMIC_RUNS_PER_GRAPH = 200
FINAL_STATIC_RUNS_PER_GRAPH = 200

# calibration
COARSE_BETA_GRID = np.linspace(0.40, 0.50, 9)
COARSE_RUNS_PER_BETA = 20

FINE_HALF_WIDTH = 0.01
FINE_STEP = 0.002
FINE_RUNS_PER_BETA = 50

# -------------------------
# output grid for <I(t)>
# -------------------------
T_MAX = 10.0
DT_CAL = 0.10
DT_FINAL = 0.05

# -------------------------
# thresholds
# -------------------------
THRESHOLD_FRACS = [0.10, 0.20, 0.30, 0.40, 0.50,
                   0.60, 0.70, 0.80, 0.90, 0.99]

# -------------------------
# parallel
# -------------------------
N_JOBS = max(1, mp.cpu_count() - 1)
MASTER_SEED = 987654321

# -------------------------
# output folder
# -------------------------
OUTDIR = Path("nested_er_static_dynamic_results_fast")
OUTDIR.mkdir(exist_ok=True)


# ============================================================
# FENWICK TREE
# ============================================================

class FenwickTree:
    def __init__(self, n):
        self.n = n
        self.tree = np.zeros(n + 1, dtype=np.int64)

    def add(self, idx0, delta):
        i = idx0 + 1
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i

    def prefix_sum(self, idx0):
        s = 0
        i = idx0 + 1
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s

    def total(self):
        return self.prefix_sum(self.n - 1)

    def find_by_cumulative(self, target):
        idx = 0
        bit = 1 << (self.n.bit_length() - 1)
        while bit:
            nxt = idx + bit
            if nxt <= self.n and self.tree[nxt] < target:
                target -= self.tree[nxt]
                idx = nxt
            bit >>= 1
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
    neighbors: list
    edge_ids: list


GLOBAL_GRAPH = None
GLOBAL_T_GRID_FINAL = None
GLOBAL_THRESH_COUNTS = None
GLOBAL_CAL_STRIDE = None


def init_worker(graph, t_grid_final, thresh_counts, cal_stride):
    global GLOBAL_GRAPH, GLOBAL_T_GRID_FINAL, GLOBAL_THRESH_COUNTS, GLOBAL_CAL_STRIDE
    GLOBAL_GRAPH = graph
    GLOBAL_T_GRID_FINAL = t_grid_final
    GLOBAL_THRESH_COUNTS = thresh_counts
    GLOBAL_CAL_STRIDE = cal_stride


# ============================================================
# HELPERS
# ============================================================

def make_t_grid(dt):
    return np.arange(0.0, T_MAX + dt, dt)


def threshold_counts(n, fracs):
    return np.array([math.ceil(f * n) for f in fracs], dtype=np.int32)


def threshold_label(frac):
    return f"T{int(round(100 * frac))}"


def fill_I_grid_piecewise(event_times, event_counts, t_grid, n):
    out = np.empty_like(t_grid, dtype=np.float64)
    j = 0
    current = event_counts[0]

    for i, t in enumerate(t_grid):
        while j + 1 < len(event_times) and event_times[j + 1] <= t:
            j += 1
            current = event_counts[j]
        out[i] = current / n
    return out


def curve_metrics(mean_dyn, mean_stat, dt):
    diff = mean_dyn - mean_stat
    mse = np.mean(diff ** 2)
    l1_area = np.sum(np.abs(diff)) * dt
    max_abs = np.max(np.abs(diff))
    return {"mse": mse, "l1_area": l1_area, "max_abs": max_abs}


def scalarise_curve_metrics(metrics):
    return metrics["mse"] + 0.2 * metrics["l1_area"] + 0.2 * metrics["max_abs"]


# ============================================================
# FAST NEIGHBOUR CHOICE
# ============================================================

def choose_random_sus_neighbor_static(node, infected, neighbors, rng):
    nbrs = neighbors[node]
    while True:
        nb = int(nbrs[rng.integers(len(nbrs))])
        if not infected[nb]:
            return nb


def choose_random_active_sus_neighbor_dynamic(node, infected, neighbors, edge_ids, edge_state, rng):
    nbrs = neighbors[node]
    eids = edge_ids[node]
    while True:
        k = int(rng.integers(len(nbrs)))
        nb = int(nbrs[k])
        eid = int(eids[k])
        if edge_state[eid] and (not infected[nb]):
            return nb


# ============================================================
# GRAPH GENERATION
# ============================================================

def is_connected(neighbors):
    n = len(neighbors)
    seen = np.zeros(n, dtype=bool)
    stack = [0]
    seen[0] = True
    count = 1
    while stack:
        u = stack.pop()
        for v in neighbors[u]:
            if not seen[v]:
                seen[v] = True
                count += 1
                stack.append(v)
    return count == n


def make_connected_er_graph(n, p, seed):
    rng = np.random.default_rng(seed)

    while True:
        edges_u = []
        edges_v = []

        for i in range(n):
            r = rng.random(n - i - 1)
            js = np.nonzero(r < p)[0] + i + 1
            for j in js:
                edges_u.append(i)
                edges_v.append(j)

        edges_u = np.array(edges_u, dtype=np.int32)
        edges_v = np.array(edges_v, dtype=np.int32)
        m = len(edges_u)

        neighbors = [[] for _ in range(n)]
        edge_ids = [[] for _ in range(n)]

        for eid, (u, v) in enumerate(zip(edges_u, edges_v)):
            neighbors[u].append(v)
            edge_ids[u].append(eid)
            neighbors[v].append(u)
            edge_ids[v].append(eid)

        # convert to numpy arrays once
        neighbors = [np.array(x, dtype=np.int32) for x in neighbors]
        edge_ids = [np.array(x, dtype=np.int32) for x in edge_ids]

        if is_connected(neighbors):
            return GraphData(
                n=n,
                m=m,
                edges_u=edges_u,
                edges_v=edges_v,
                neighbors=neighbors,
                edge_ids=edge_ids
            )


# ============================================================
# STATIC SI
# ============================================================

def simulate_static_once(seed, beta, want_thresholds=True):
    G = GLOBAL_GRAPH
    t_grid = GLOBAL_T_GRID_FINAL
    thresh_counts = GLOBAL_THRESH_COUNTS
    rng = np.random.default_rng(seed)

    n = G.n
    neighbors = G.neighbors

    infected = np.zeros(n, dtype=bool)
    sus_deg = np.zeros(n, dtype=np.int32)
    fenw = FenwickTree(n)

    seed_node = int(rng.integers(n))
    infected[seed_node] = True

    sus_deg[seed_node] = len(neighbors[seed_node])
    fenw.add(seed_node, sus_deg[seed_node])

    t = 0.0
    infected_count = 1

    if want_thresholds:
        threshold_times = np.full(len(thresh_counts), np.nan, dtype=np.float64)
        next_thr_idx = 0
        while next_thr_idx < len(thresh_counts) and infected_count >= thresh_counts[next_thr_idx]:
            threshold_times[next_thr_idx] = t
            next_thr_idx += 1
    else:
        threshold_times = None
        next_thr_idx = 0

    event_times = [0.0]
    event_counts = [1]

    while infected_count < n:
        total_is = fenw.total()
        if total_is <= 0:
            break

        t += rng.exponential(1.0 / (beta * total_is))

        target = int(rng.integers(1, total_is + 1))
        src = fenw.find_by_cumulative(target)
        new_node = choose_random_sus_neighbor_static(src, infected, neighbors, rng)

        infected[new_node] = True
        infected_count += 1

        new_w = 0
        for nb in neighbors[new_node]:
            if infected[nb]:
                sus_deg[nb] -= 1
                fenw.add(int(nb), -1)
            else:
                new_w += 1

        sus_deg[new_node] = new_w
        fenw.add(new_node, new_w)

        event_times.append(t)
        event_counts.append(infected_count)

        if want_thresholds:
            while next_thr_idx < len(thresh_counts) and infected_count >= thresh_counts[next_thr_idx]:
                threshold_times[next_thr_idx] = t
                next_thr_idx += 1

    I_grid_final = fill_I_grid_piecewise(
        np.array(event_times, dtype=np.float64),
        np.array(event_counts, dtype=np.int32),
        t_grid,
        n
    )

    return I_grid_final, threshold_times


# ============================================================
# DYNAMIC SI
# ============================================================

def simulate_dynamic_once(seed):
    G = GLOBAL_GRAPH
    t_grid = GLOBAL_T_GRID_FINAL
    thresh_counts = GLOBAL_THRESH_COUNTS
    rng = np.random.default_rng(seed)

    n = G.n
    m = G.m
    neighbors = G.neighbors
    edge_ids = G.edge_ids
    edges_u = G.edges_u
    edges_v = G.edges_v

    infected = np.zeros(n, dtype=bool)
    edge_state = rng.random(m) < 0.5

    active_sus_deg = np.zeros(n, dtype=np.int32)
    fenw = FenwickTree(n)

    seed_node = int(rng.integers(n))
    infected[seed_node] = True

    w0 = 0
    for nb, eid in zip(neighbors[seed_node], edge_ids[seed_node]):
        if edge_state[eid] and (not infected[nb]):
            w0 += 1
    active_sus_deg[seed_node] = w0
    fenw.add(seed_node, w0)

    t = 0.0
    infected_count = 1

    threshold_times = np.full(len(thresh_counts), np.nan, dtype=np.float64)
    next_thr_idx = 0
    while next_thr_idx < len(thresh_counts) and infected_count >= thresh_counts[next_thr_idx]:
        threshold_times[next_thr_idx] = t
        next_thr_idx += 1

    event_times = [0.0]
    event_counts = [1]

    toggle_rate_total = MU * m

    while infected_count < n:
        total_active_is = fenw.total()
        rate_inf = BETA_DYNAMIC * total_active_is
        rate_total = rate_inf + toggle_rate_total

        if rate_total <= 0:
            break

        t += rng.exponential(1.0 / rate_total)

        if rng.random() < rate_inf / rate_total:
            if total_active_is <= 0:
                continue

            target = int(rng.integers(1, total_active_is + 1))
            src = fenw.find_by_cumulative(target)
            new_node = choose_random_active_sus_neighbor_dynamic(
                src, infected, neighbors, edge_ids, edge_state, rng
            )

            infected[new_node] = True
            infected_count += 1

            new_w = 0
            for nb, eid in zip(neighbors[new_node], edge_ids[new_node]):
                if infected[nb]:
                    if edge_state[eid]:
                        active_sus_deg[nb] -= 1
                        fenw.add(int(nb), -1)
                else:
                    if edge_state[eid]:
                        new_w += 1

            active_sus_deg[new_node] = new_w
            fenw.add(new_node, new_w)

            event_times.append(t)
            event_counts.append(infected_count)

            while next_thr_idx < len(thresh_counts) and infected_count >= thresh_counts[next_thr_idx]:
                threshold_times[next_thr_idx] = t
                next_thr_idx += 1

        else:
            eid = int(rng.integers(m))
            u = int(edges_u[eid])
            v = int(edges_v[eid])

            edge_state[eid] = not edge_state[eid]
            delta = 1 if edge_state[eid] else -1

            iu = infected[u]
            iv = infected[v]

            if iu and (not iv):
                active_sus_deg[u] += delta
                fenw.add(u, delta)
            elif iv and (not iu):
                active_sus_deg[v] += delta
                fenw.add(v, delta)

    I_grid_final = fill_I_grid_piecewise(
        np.array(event_times, dtype=np.float64),
        np.array(event_counts, dtype=np.int32),
        t_grid,
        n
    )

    return I_grid_final, threshold_times


# ============================================================
# WORKER WRAPPERS
# ============================================================

def dynamic_task(seed):
    return simulate_dynamic_once(seed)


def static_final_task(args):
    seed, beta = args
    return simulate_static_once(seed, beta, want_thresholds=True)


def static_cal_task(args):
    seed, beta = args
    I_grid_final, _ = simulate_static_once(seed, beta, want_thresholds=False)
    return I_grid_final[::GLOBAL_CAL_STRIDE]


# ============================================================
# ACCUMULATION HELPERS
# ============================================================

def collect_mean_and_thresholds(result_iter, n_runs, grid_len, n_thresh):
    sum_curve = np.zeros(grid_len, dtype=np.float64)
    thresholds = np.empty((n_runs, n_thresh), dtype=np.float64)

    for i, (I_grid, T) in enumerate(result_iter):
        sum_curve += I_grid
        thresholds[i, :] = T

    mean_curve = sum_curve / n_runs
    return mean_curve, thresholds


def collect_mean_only(result_iter, n_runs, grid_len):
    sum_curve = np.zeros(grid_len, dtype=np.float64)
    for I_grid in result_iter:
        sum_curve += I_grid
    return sum_curve / n_runs


# ============================================================
# CALIBRATION
# ============================================================

def evaluate_beta_with_pool(pool, beta, n_runs, mean_dyn_curve, seed_base, coarse_len):
    tasks = [(seed_base + i, beta) for i in range(n_runs)]
    mean_stat_curve = collect_mean_only(
        pool.imap(static_cal_task, tasks, chunksize=8),
        n_runs=n_runs,
        grid_len=coarse_len
    )
    metrics = curve_metrics(mean_dyn_curve, mean_stat_curve, dt=DT_CAL)
    objective = scalarise_curve_metrics(metrics)
    return {
        "beta": beta,
        "mean_curve": mean_stat_curve,
        "metrics": metrics,
        "objective": objective
    }


def coarse_to_fine_beta_scan(pool, mean_dyn_curve, seed_offset):
    coarse_results = []
    coarse_len = len(mean_dyn_curve)

    for k, beta in enumerate(COARSE_BETA_GRID):
        res = evaluate_beta_with_pool(
            pool=pool,
            beta=float(beta),
            n_runs=COARSE_RUNS_PER_BETA,
            mean_dyn_curve=mean_dyn_curve,
            seed_base=seed_offset + 10000 * k,
            coarse_len=coarse_len
        )
        coarse_results.append(res)

    coarse_betas = np.array([r["beta"] for r in coarse_results])
    coarse_objectives = np.array([r["objective"] for r in coarse_results])
    best_coarse_beta = float(coarse_betas[np.argmin(coarse_objectives)])

    lo = max(0.0, best_coarse_beta - FINE_HALF_WIDTH)
    hi = best_coarse_beta + FINE_HALF_WIDTH
    fine_grid = np.arange(lo, hi + 0.5 * FINE_STEP, FINE_STEP)

    fine_results = []
    for k, beta in enumerate(fine_grid):
        res = evaluate_beta_with_pool(
            pool=pool,
            beta=float(beta),
            n_runs=FINE_RUNS_PER_BETA,
            mean_dyn_curve=mean_dyn_curve,
            seed_base=seed_offset + 500000 + 10000 * k,
            coarse_len=coarse_len
        )
        fine_results.append(res)

    fine_betas = np.array([r["beta"] for r in fine_results])
    fine_objectives = np.array([r["objective"] for r in fine_results])
    best_idx = int(np.argmin(fine_objectives))

    return {
        "best_beta": float(fine_betas[best_idx]),
        "best_result": fine_results[best_idx],
        "coarse_results": coarse_results,
        "fine_results": fine_results
    }


# ============================================================
# STATS
# ============================================================

def compare_threshold_samples(x_dyn, x_stat):
    ks = ks_2samp(x_dyn, x_stat)
    ad = anderson_ksamp([x_dyn, x_stat])

    mean_dyn = np.mean(x_dyn)
    mean_stat = np.mean(x_stat)
    med_dyn = np.median(x_dyn)
    med_stat = np.median(x_stat)
    std_dyn = np.std(x_dyn, ddof=1)
    std_stat = np.std(x_stat, ddof=1)

    return {
        "mean_dyn": mean_dyn,
        "mean_stat": mean_stat,
        "delta_mean": mean_dyn - mean_stat,
        "median_dyn": med_dyn,
        "median_stat": med_stat,
        "delta_median": med_dyn - med_stat,
        "std_dyn": std_dyn,
        "std_stat": std_stat,
        "delta_std": std_dyn - std_stat,
        "ks_stat": ks.statistic,
        "ks_pvalue": ks.pvalue,
        "ad_stat": ad.statistic,
        "ad_significance_level": ad.significance_level,
        "wasserstein": wasserstein_distance(x_dyn, x_stat),
    }


def variance_decomposition(samples_by_graph):
    graph_means = np.mean(samples_by_graph, axis=1)
    within_vars = np.var(samples_by_graph, axis=1, ddof=1)

    total_var = np.var(samples_by_graph.ravel(), ddof=1)
    mean_within = np.mean(within_vars)
    between = np.var(graph_means, ddof=1)

    return {
        "total_var": total_var,
        "mean_within_var": mean_within,
        "between_graph_var": between,
        "sum_within_plus_between": mean_within + between,
    }


# ============================================================
# PER-GRAPH ANALYSIS
# ============================================================

def analyse_one_graph(graph_idx):
    graph_seed = GRAPH_MASTER_SEED + 100000 * graph_idx
    graph = make_connected_er_graph(N, P_ER, graph_seed)

    t_grid_final = make_t_grid(DT_FINAL)
    t_grid_cal = make_t_grid(DT_CAL)
    cal_stride = int(round(DT_CAL / DT_FINAL))
    thresh_counts = threshold_counts(N, THRESHOLD_FRACS)

    with mp.Pool(
        processes=N_JOBS,
        initializer=init_worker,
        initargs=(graph, t_grid_final, thresh_counts, cal_stride)
    ) as pool:

        # ------------------------------------------------
        # DYNAMIC: run ONCE on fine grid, then downsample
        # ------------------------------------------------
        dyn_seeds = [MASTER_SEED + 1_000_000 * graph_idx + i for i in range(DYNAMIC_RUNS_PER_GRAPH)]

        mean_dyn_final, T_dyn_final = collect_mean_and_thresholds(
            pool.imap(dynamic_task, dyn_seeds, chunksize=8),
            n_runs=DYNAMIC_RUNS_PER_GRAPH,
            grid_len=len(t_grid_final),
            n_thresh=len(thresh_counts)
        )

        mean_dyn_cal = mean_dyn_final[::cal_stride]

        # ------------------------------------------------
        # STATIC calibration
        # ------------------------------------------------
        scan = coarse_to_fine_beta_scan(
            pool=pool,
            mean_dyn_curve=mean_dyn_cal,
            seed_offset=MASTER_SEED + 3_000_000 * graph_idx
        )
        beta_best = scan["best_beta"]

        # ------------------------------------------------
        # FINAL STATIC runs
        # ------------------------------------------------
        stat_tasks = [
            (MASTER_SEED + 4_000_000 * graph_idx + i, beta_best)
            for i in range(FINAL_STATIC_RUNS_PER_GRAPH)
        ]

        mean_stat_final, T_stat_final = collect_mean_and_thresholds(
            pool.imap(static_final_task, stat_tasks, chunksize=8),
            n_runs=FINAL_STATIC_RUNS_PER_GRAPH,
            grid_len=len(t_grid_final),
            n_thresh=len(thresh_counts)
        )

    final_curve = curve_metrics(mean_dyn_final, mean_stat_final, DT_FINAL)

    graph_summary = {
        "graph_idx": graph_idx,
        "graph_seed": graph_seed,
        "M": graph.m,
        "mean_degree": 2 * graph.m / graph.n,
        "beta_best": beta_best,
        "curve_mse": final_curve["mse"],
        "curve_l1_area": final_curve["l1_area"],
        "curve_max_abs": final_curve["max_abs"],
    }

    threshold_rows = []
    for j, frac in enumerate(THRESHOLD_FRACS):
        stats = compare_threshold_samples(T_dyn_final[:, j], T_stat_final[:, j])
        threshold_rows.append({
            "graph_idx": graph_idx,
            "threshold": threshold_label(frac),
            **stats
        })

    return {
        "graph_summary": graph_summary,
        "threshold_rows": threshold_rows,
        "T_dyn_final": T_dyn_final,
        "T_stat_final": T_stat_final,
        "mean_dyn_final": mean_dyn_final,
        "mean_stat_final": mean_stat_final,
        "t_grid_final": t_grid_final,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("Starting nested ER experiment (fast version)")
    print(f"N_GRAPHS = {N_GRAPHS}")
    print(f"DYNAMIC_RUNS_PER_GRAPH = {DYNAMIC_RUNS_PER_GRAPH}")
    print(f"FINAL_STATIC_RUNS_PER_GRAPH = {FINAL_STATIC_RUNS_PER_GRAPH}")
    print(f"N_JOBS = {N_JOBS}")

    all_graph_summaries = []
    all_threshold_rows = []

    all_dyn_thresholds = {threshold_label(f): [] for f in THRESHOLD_FRACS}
    all_stat_thresholds = {threshold_label(f): [] for f in THRESHOLD_FRACS}

    example_plot_done = False

    for g in range(N_GRAPHS):
        print(f"\nGraph {g + 1} / {N_GRAPHS}")
        res = analyse_one_graph(g)

        all_graph_summaries.append(res["graph_summary"])
        all_threshold_rows.extend(res["threshold_rows"])

        for j, frac in enumerate(THRESHOLD_FRACS):
            label = threshold_label(frac)
            all_dyn_thresholds[label].append(res["T_dyn_final"][:, j])
            all_stat_thresholds[label].append(res["T_stat_final"][:, j])

        if not example_plot_done:
            plt.figure(figsize=(9, 6))
            plt.plot(res["t_grid_final"], res["mean_dyn_final"], label="Dynamic")
            plt.plot(
                res["t_grid_final"],
                res["mean_stat_final"],
                label=f"Static best beta={res['graph_summary']['beta_best']:.4f}"
            )
            plt.xlabel("t")
            plt.ylabel("<I(t)> / N")
            plt.title("Example graph: dynamic vs calibrated static")
            plt.legend()
            plt.tight_layout()
            plt.savefig(OUTDIR / "example_mean_curve.png", dpi=200)
            plt.close()
            example_plot_done = True

    df_graph = pd.DataFrame(all_graph_summaries)
    df_graph.to_csv(OUTDIR / "graph_level_summary.csv", index=False)

    df_threshold = pd.DataFrame(all_threshold_rows)
    df_threshold.to_csv(OUTDIR / "per_graph_threshold_stats.csv", index=False)

    ensemble_rows = []
    for frac in THRESHOLD_FRACS:
        label = threshold_label(frac)

        dyn_mat = np.array(all_dyn_thresholds[label])
        stat_mat = np.array(all_stat_thresholds[label])

        dyn_decomp = variance_decomposition(dyn_mat)
        stat_decomp = variance_decomposition(stat_mat)

        ensemble_rows.append({
            "threshold": label,
            "dynamic_total_var": dyn_decomp["total_var"],
            "dynamic_mean_within_var": dyn_decomp["mean_within_var"],
            "dynamic_between_graph_var": dyn_decomp["between_graph_var"],
            "dynamic_sum_within_plus_between": dyn_decomp["sum_within_plus_between"],
            "static_total_var": stat_decomp["total_var"],
            "static_mean_within_var": stat_decomp["mean_within_var"],
            "static_between_graph_var": stat_decomp["between_graph_var"],
            "static_sum_within_plus_between": stat_decomp["sum_within_plus_between"],
            "dynamic_grand_mean": dyn_mat.mean(),
            "static_grand_mean": stat_mat.mean(),
            "delta_grand_mean": dyn_mat.mean() - stat_mat.mean(),
        })

    df_ensemble = pd.DataFrame(ensemble_rows)
    df_ensemble.to_csv(OUTDIR / "variance_decomposition_summary.csv", index=False)

    print("\nFinished.")
    print(f"Results written to: {OUTDIR.resolve()}")
    print("\nGraph-level beta_eff summary:")
    print(df_graph[["beta_best", "curve_mse", "curve_l1_area", "curve_max_abs"]].describe())

    print("\nVariance decomposition summary:")
    print(df_ensemble)

    plt.figure(figsize=(8, 5))
    plt.hist(df_graph["beta_best"], bins=12)
    plt.xlabel("best beta_eff across graphs")
    plt.ylabel("count")
    plt.title("Distribution of fitted beta_eff over graph realisations")
    plt.tight_layout()
    plt.savefig(OUTDIR / "beta_eff_histogram.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df_ensemble["threshold"], df_ensemble["dynamic_mean_within_var"], marker="o", label="Dynamic within-graph var")
    plt.plot(df_ensemble["threshold"], df_ensemble["dynamic_between_graph_var"], marker="o", label="Dynamic between-graph var")
    plt.plot(df_ensemble["threshold"], df_ensemble["static_mean_within_var"], marker="o", label="Static within-graph var")
    plt.plot(df_ensemble["threshold"], df_ensemble["static_between_graph_var"], marker="o", label="Static between-graph var")
    plt.ylabel("variance contribution")
    plt.title("Within-graph vs between-graph variance contributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "variance_decomposition_plot.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    mp.freeze_support()
    main()