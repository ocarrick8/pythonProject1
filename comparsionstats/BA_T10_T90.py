import math
import multiprocessing as mp
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, anderson_ksamp, wasserstein_distance


# ============================================================
# PARAMETERS
# ============================================================

N = 1000
M_ATTACH = 3                 # BA => average degree ~ 2m = 6
N_RUNS = 2000

BETA_DYNAMIC = 1.0
MU = 1.0

BETA_GRID = np.linspace(0.42, 0.47, 21)

T_MAX = 10.0                 # make sure this is long enough to reach 90%
DT_OUT = 0.05
T_GRID = np.arange(0.0, T_MAX + DT_OUT, DT_OUT)

MASTER_SEED = 12345
N_JOBS = max(1, mp.cpu_count() - 1)

THRESHOLDS = [0.10, 0.30, 0.50, 0.70, 0.90]


# ============================================================
# FENWICK TREE
# ============================================================

class FenwickTree:
    """
    1-indexed Fenwick tree for weighted sampling.
    Stores node infection propensities.
    """
    def __init__(self, n):
        self.n = n
        self.tree = np.zeros(n + 1, dtype=np.float64)
        self.values = np.zeros(n, dtype=np.float64)

    def add(self, idx, delta):
        self.values[idx] += delta
        i = idx + 1
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i

    def set(self, idx, value):
        delta = value - self.values[idx]
        if delta != 0.0:
            self.add(idx, delta)

    def total(self):
        s = 0.0
        i = self.n
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s

    def find_prefix_index(self, target):
        idx = 0
        bit = 1 << (self.n.bit_length() - 1)
        while bit:
            nxt = idx + bit
            if nxt <= self.n and self.tree[nxt] <= target:
                target -= self.tree[nxt]
                idx = nxt
            bit >>= 1
        return idx

    def sample(self, rng):
        tot = self.total()
        if tot <= 0.0:
            raise ValueError("Cannot sample from empty Fenwick tree.")
        u = rng.random() * tot
        return self.find_prefix_index(u)


# ============================================================
# GRAPH
# ============================================================

@dataclass
class GraphData:
    n: int
    edges_u: np.ndarray
    edges_v: np.ndarray
    neighbors: list
    edge_ids: list


def generate_ba_graph(n, m, rng):
    """
    Generate a BA graph with attachment parameter m.
    """
    edges_u = []
    edges_v = []
    degree = np.zeros(n, dtype=np.int32)

    # initial complete graph on m+1 nodes
    for i in range(m + 1):
        for j in range(i):
            edges_u.append(i)
            edges_v.append(j)
            degree[i] += 1
            degree[j] += 1

    repeated_nodes = []
    for node in range(m + 1):
        repeated_nodes.extend([node] * degree[node])

    for new_node in range(m + 1, n):
        targets = set()
        while len(targets) < m:
            t = repeated_nodes[rng.integers(len(repeated_nodes))]
            targets.add(t)

        for t in targets:
            edges_u.append(new_node)
            edges_v.append(t)
            degree[new_node] += 1
            degree[t] += 1

            repeated_nodes.append(new_node)
            repeated_nodes.append(t)

    edges_u = np.array(edges_u, dtype=np.int32)
    edges_v = np.array(edges_v, dtype=np.int32)

    neighbors = [[] for _ in range(n)]
    edge_ids = [[] for _ in range(n)]

    for e, (u, v) in enumerate(zip(edges_u, edges_v)):
        neighbors[u].append(v)
        edge_ids[u].append(e)
        neighbors[v].append(u)
        edge_ids[v].append(e)

    return GraphData(
        n=n,
        edges_u=edges_u,
        edges_v=edges_v,
        neighbors=neighbors,
        edge_ids=edge_ids
    )


# ============================================================
# HELPERS
# ============================================================

def threshold_label(frac):
    return f"T{int(round(100 * frac))}"


def init_threshold_times(thresholds):
    return {thr: None for thr in thresholds}


def update_threshold_times(infected_fraction_prev, infected_fraction_new, t_new, threshold_times, thresholds):
    """
    Record threshold times the first time each threshold is crossed.
    We assign the crossing to the event time t_new.
    """
    for thr in thresholds:
        if threshold_times[thr] is None and infected_fraction_prev < thr <= infected_fraction_new:
            threshold_times[thr] = t_new
    return threshold_times


def fill_curve_segment(curve, t_grid, pointer, t_start, t_end, value):
    while pointer < len(t_grid) and t_grid[pointer] < t_end:
        if t_grid[pointer] >= t_start:
            curve[pointer] = value
        pointer += 1
    return pointer


# ============================================================
# DYNAMIC SI
# ============================================================

def simulate_dynamic_si_fenwick(graph, beta, mu, t_grid, rng, thresholds):
    """
    Dynamic SI on a BA graph with edge flickering.
    Each edge flips ON/OFF at rate mu.
    Infection uses only currently active infected-susceptible edges.
    """
    n = graph.n
    edges_u = graph.edges_u
    edges_v = graph.edges_v
    neighbors = graph.neighbors
    edge_ids = graph.edge_ids
    m_edges = len(edges_u)

    infected = np.zeros(n, dtype=np.bool_)
    infected_active_nbrs = np.zeros(n, dtype=np.int32)

    # stationarity: half the edges on, on average
    active = rng.random(m_edges) < 0.5

    seed = rng.integers(n)
    infected[seed] = True
    infected_count = 1

    threshold_times = init_threshold_times(thresholds)

    ft = FenwickTree(n)

    # initialise infection propensities
    for nbr, e in zip(neighbors[seed], edge_ids[seed]):
        if not infected[nbr] and active[e]:
            infected_active_nbrs[nbr] += 1
            ft.add(nbr, beta)

    curve = np.empty(len(t_grid), dtype=np.float64)
    t = 0.0
    pointer = 0

    while t < t_grid[-1] and infected_count < n:
        lambda_inf = ft.total()
        lambda_flip = mu * m_edges
        lambda_tot = lambda_inf + lambda_flip

        if lambda_tot <= 0.0:
            break

        dt = -math.log(rng.random()) / lambda_tot
        t_next = t + dt

        current_I = infected_count / n
        pointer = fill_curve_segment(curve, t_grid, pointer, t, min(t_next, t_grid[-1] + 1e-12), current_I)

        t = t_next
        if t > t_grid[-1]:
            break

        if rng.random() < (lambda_inf / lambda_tot):
            # infection event
            node = ft.sample(rng)

            if infected[node]:
                ft.set(node, 0.0)
                continue

            prev_I = infected_count / n

            infected[node] = True
            infected_count += 1
            ft.set(node, 0.0)

            for nbr, e in zip(neighbors[node], edge_ids[node]):
                if active[e] and not infected[nbr]:
                    infected_active_nbrs[nbr] += 1
                    ft.add(nbr, beta)

            new_I = infected_count / n
            threshold_times = update_threshold_times(prev_I, new_I, t, threshold_times, thresholds)

        else:
            # edge flip event
            e = rng.integers(m_edges)
            u = edges_u[e]
            v = edges_v[e]

            if active[e]:
                active[e] = False
                if infected[u] and not infected[v]:
                    infected_active_nbrs[v] -= 1
                    ft.add(v, -beta)
                elif infected[v] and not infected[u]:
                    infected_active_nbrs[u] -= 1
                    ft.add(u, -beta)
            else:
                active[e] = True
                if infected[u] and not infected[v]:
                    infected_active_nbrs[v] += 1
                    ft.add(v, beta)
                elif infected[v] and not infected[u]:
                    infected_active_nbrs[u] += 1
                    ft.add(u, beta)

    final_I = infected_count / n
    while pointer < len(t_grid):
        curve[pointer] = final_I
        pointer += 1

    return curve, threshold_times


# ============================================================
# STATIC SI
# ============================================================

def simulate_static_si_fenwick(graph, beta, t_grid, rng, thresholds):
    """
    Static SI on the BA graph.
    """
    n = graph.n
    neighbors = graph.neighbors

    infected = np.zeros(n, dtype=np.bool_)
    infected_in_nbrs = np.zeros(n, dtype=np.int32)

    seed = rng.integers(n)
    infected[seed] = True
    infected_count = 1

    threshold_times = init_threshold_times(thresholds)

    ft = FenwickTree(n)

    for nbr in neighbors[seed]:
        if not infected[nbr]:
            infected_in_nbrs[nbr] += 1
            ft.add(nbr, beta)

    curve = np.empty(len(t_grid), dtype=np.float64)
    t = 0.0
    pointer = 0

    while t < t_grid[-1] and infected_count < n:
        lambda_inf = ft.total()
        if lambda_inf <= 0.0:
            break

        dt = -math.log(rng.random()) / lambda_inf
        t_next = t + dt

        current_I = infected_count / n
        pointer = fill_curve_segment(curve, t_grid, pointer, t, min(t_next, t_grid[-1] + 1e-12), current_I)

        t = t_next
        if t > t_grid[-1]:
            break

        node = ft.sample(rng)

        if infected[node]:
            ft.set(node, 0.0)
            continue

        prev_I = infected_count / n

        infected[node] = True
        infected_count += 1
        ft.set(node, 0.0)

        for nbr in neighbors[node]:
            if not infected[nbr]:
                infected_in_nbrs[nbr] += 1
                ft.add(nbr, beta)

        new_I = infected_count / n
        threshold_times = update_threshold_times(prev_I, new_I, t, threshold_times, thresholds)

    final_I = infected_count / n
    while pointer < len(t_grid):
        curve[pointer] = final_I
        pointer += 1

    return curve, threshold_times


# ============================================================
# WORKERS
# ============================================================

def dynamic_worker(args):
    graph_seed, sim_seed = args
    rng_graph = np.random.default_rng(graph_seed)
    rng_sim = np.random.default_rng(sim_seed)

    graph = generate_ba_graph(N, M_ATTACH, rng_graph)
    curve, threshold_times = simulate_dynamic_si_fenwick(
        graph=graph,
        beta=BETA_DYNAMIC,
        mu=MU,
        t_grid=T_GRID,
        rng=rng_sim,
        thresholds=THRESHOLDS
    )
    return curve, threshold_times


def static_worker(args):
    graph_seed, sim_seed, beta_static = args
    rng_graph = np.random.default_rng(graph_seed)
    rng_sim = np.random.default_rng(sim_seed)

    graph = generate_ba_graph(N, M_ATTACH, rng_graph)
    curve, threshold_times = simulate_static_si_fenwick(
        graph=graph,
        beta=beta_static,
        t_grid=T_GRID,
        rng=rng_sim,
        thresholds=THRESHOLDS
    )
    return curve, threshold_times


# ============================================================
# ENSEMBLE RUNNERS
# ============================================================

def run_ensemble(worker_fn, arglist, n_jobs, thresholds):
    curves = []
    threshold_lists = {thr: [] for thr in thresholds}

    with mp.Pool(processes=n_jobs) as pool:
        for k, out in enumerate(pool.imap(worker_fn, arglist, chunksize=10), start=1):
            curve, threshold_times = out
            curves.append(curve)

            for thr in thresholds:
                val = threshold_times.get(thr, None)
                threshold_lists[thr].append(np.nan if val is None else val)

            if k % 100 == 0:
                print(f"Completed {k}/{len(arglist)}")

    curves = np.array(curves)
    avg_curve = np.mean(curves, axis=0)

    threshold_arrays = {
        thr: np.array(vals, dtype=float)
        for thr, vals in threshold_lists.items()
    }

    return avg_curve, threshold_arrays


# ============================================================
# DISTRIBUTION COMPARISON
# ============================================================

def clean_nan_pair(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    return x, y


def summarise_distribution(x):
    x = np.asarray(x, dtype=float)
    return {
        "n": len(x),
        "mean": np.mean(x),
        "std": np.std(x, ddof=1),
        "median": np.median(x),
        "q25": np.percentile(x, 25),
        "q75": np.percentile(x, 75),
        "iqr": np.percentile(x, 75) - np.percentile(x, 25),
        "min": np.min(x),
        "max": np.max(x),
    }


def compare_distributions(x_dyn, x_stat):
    x_dyn, x_stat = clean_nan_pair(x_dyn, x_stat)

    dyn = summarise_distribution(x_dyn)
    stat = summarise_distribution(x_stat)

    ks = ks_2samp(x_dyn, x_stat)
    ad = anderson_ksamp([x_dyn, x_stat])
    wdist = wasserstein_distance(x_dyn, x_stat)

    return {
        "n_dynamic": dyn["n"],
        "n_static": stat["n"],

        "mean_dynamic": dyn["mean"],
        "mean_static": stat["mean"],
        "delta_mean": dyn["mean"] - stat["mean"],

        "std_dynamic": dyn["std"],
        "std_static": stat["std"],
        "delta_std": dyn["std"] - stat["std"],

        "median_dynamic": dyn["median"],
        "median_static": stat["median"],
        "delta_median": dyn["median"] - stat["median"],

        "iqr_dynamic": dyn["iqr"],
        "iqr_static": stat["iqr"],
        "delta_iqr": dyn["iqr"] - stat["iqr"],

        "ks_statistic": ks.statistic,
        "ks_pvalue": ks.pvalue,

        "ad_statistic": ad.statistic,
        "ad_significance_level_percent": ad.significance_level,

        "wasserstein": wdist,
    }


def make_results_table(dynamic_thresholds, static_thresholds, thresholds):
    rows = []

    for thr in thresholds:
        row = compare_distributions(dynamic_thresholds[thr], static_thresholds[thr])
        row["threshold"] = threshold_label(thr)
        rows.append(row)

    df = pd.DataFrame(rows)
    cols = ["threshold"] + [c for c in df.columns if c != "threshold"]
    return df[cols]


# ============================================================
# MAIN
# ============================================================

def main():
    rng_master = np.random.default_rng(MASTER_SEED)

    # same graph realisations used throughout
    graph_seeds = rng_master.integers(0, 2**32 - 1, size=N_RUNS, dtype=np.uint64)
    sim_seeds_dynamic = rng_master.integers(0, 2**32 - 1, size=N_RUNS, dtype=np.uint64)
    sim_seeds_static = rng_master.integers(0, 2**32 - 1, size=N_RUNS, dtype=np.uint64)

    # --------------------------------------------------------
    # Dynamic ensemble
    # --------------------------------------------------------
    print("Running dynamic BA ensemble...")
    dyn_args = list(zip(graph_seeds, sim_seeds_dynamic))
    I_dynamic_avg, thresholds_dynamic = run_ensemble(dynamic_worker, dyn_args, N_JOBS, THRESHOLDS)

    # --------------------------------------------------------
    # Static beta=1 baseline
    # --------------------------------------------------------
    print("\nRunning static BA ensemble with beta=1...")
    static_args_beta1 = [(g, s, 1.0) for g, s in zip(graph_seeds, sim_seeds_static)]
    I_static_beta1_avg, thresholds_static_beta1 = run_ensemble(
        static_worker, static_args_beta1, N_JOBS, THRESHOLDS
    )

    # --------------------------------------------------------
    # Search beta_eff using average I(t) curve
    # --------------------------------------------------------
    print("\nSearching for best effective beta...")
    mse_values = []
    static_results_by_beta = {}

    for beta_eff in BETA_GRID:
        print(f"\nTesting beta_eff = {beta_eff:.4f}")
        args = [(g, s, float(beta_eff)) for g, s in zip(graph_seeds, sim_seeds_static)]
        I_static_avg, thresholds_static = run_ensemble(static_worker, args, N_JOBS, THRESHOLDS)

        mse = np.mean((I_static_avg - I_dynamic_avg) ** 2)
        mse_values.append(mse)
        static_results_by_beta[beta_eff] = (I_static_avg, thresholds_static)

        print(f"beta_eff = {beta_eff:.4f}, MSE = {mse:.8e}")

    mse_values = np.array(mse_values)
    best_idx = np.argmin(mse_values)
    best_beta = float(BETA_GRID[best_idx])

    I_static_best_avg, thresholds_static_best = static_results_by_beta[best_beta]

    print("\n===================================================")
    print(f"Best beta_eff = {best_beta:.4f}")
    print(f"Best MSE      = {mse_values[best_idx]:.8e}")
    print("===================================================")

    # --------------------------------------------------------
    # Compare T10/T30/T50/T70/T90 distributions
    # --------------------------------------------------------
    results_df = make_results_table(
        thresholds_dynamic,
        thresholds_static_best,
        THRESHOLDS
    )

    print("\nDistribution comparison table:")
    print(results_df.to_string(index=False))

    # --------------------------------------------------------
    # Plot infection curves
    # --------------------------------------------------------
    plt.figure(figsize=(9, 6))
    plt.plot(T_GRID, I_dynamic_avg, label="Dynamic, beta=1, mu=1", linewidth=2)
    plt.plot(T_GRID, I_static_beta1_avg, label="Static, beta=1", linestyle="--", linewidth=2)
    plt.plot(T_GRID, I_static_best_avg, label=f"Static, beta_eff={best_beta:.4f}", linewidth=2)
    plt.xlabel("t")
    plt.ylabel("Average infected fraction I(t)")
    plt.title("BA network: average infection curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Plot beta_eff search
    # --------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(BETA_GRID, mse_values, marker="o")
    plt.axvline(best_beta, linestyle="--", linewidth=1)
    plt.xlabel(r"Static $\beta_{\mathrm{eff}}$")
    plt.ylabel("MSE against dynamic average curve")
    plt.title("Effective beta search on BA networks")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Histograms for T10/T30/T50/T70/T90
    # --------------------------------------------------------
    def finite_only(x):
        x = np.asarray(x, dtype=float)
        return x[np.isfinite(x)]

    for thr in THRESHOLDS:
        label = threshold_label(thr)
        td = finite_only(thresholds_dynamic[thr])
        ts = finite_only(thresholds_static_best[thr])

        plt.figure(figsize=(8, 5))
        plt.hist(td, bins=40, density=True, alpha=0.5, label=f"Dynamic {label}")
        plt.hist(ts, bins=40, density=True, alpha=0.5, label=f"Static {label}, beta_eff={best_beta:.4f}")
        plt.xlabel(label)
        plt.ylabel("Density")
        plt.title(f"BA network: {label} distributions")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()