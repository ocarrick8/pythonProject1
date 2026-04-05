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
TARGET_MEAN_DEGREE = 10.0

# For a 2D random geometric graph in the unit square:
# mean degree ~ (N-1) * pi * R^2   (ignoring boundary effects)
R_RADIUS = math.sqrt(TARGET_MEAN_DEGREE / ((N - 1) * math.pi))

N_RUNS = 500

BETA_DYNAMIC = 1.0
MU = 0.5

# Grid of static beta values to test
BETA_GRID = np.linspace(0.39, 0.44, 21)

MASTER_SEED = 12345
N_JOBS = max(1, mp.cpu_count() - 1)

# Thresholds: T10, T20, ..., T90
THRESHOLDS = [i / 10 for i in range(1, 10)]

# Optional safety cutoff for very slow runs
T_MAX = 100.0

# Which objective to use to choose beta_eff at each threshold:
#   "wasserstein"  -> distributional fit
#   "mean_abs"     -> fit by |mean_dynamic - mean_static|
FIT_OBJECTIVE = "wasserstein"


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


def generate_rgg_2d_graph(n, radius, rng):
    """
    Generate a 2D random geometric graph in the unit square.
    Nodes are placed uniformly in [0,1]^2.
    Two nodes are connected if Euclidean distance <= radius.

    Uses cell-list spatial hashing to avoid O(N^2) pair checking.
    """
    xy = rng.random((n, 2))
    r2 = radius * radius

    cell_size = radius
    n_cells = max(1, int(math.ceil(1.0 / cell_size)))

    cells = {}
    cell_coords = np.floor(xy / cell_size).astype(int)
    cell_coords = np.clip(cell_coords, 0, n_cells - 1)

    for i in range(n):
        cx, cy = cell_coords[i]
        cells.setdefault((cx, cy), []).append(i)

    edges_u = []
    edges_v = []

    for i in range(n):
        xi, yi = xy[i]
        cx, cy = cell_coords[i]

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx = cx + dx
                ny = cy + dy
                if nx < 0 or nx >= n_cells or ny < 0 or ny >= n_cells:
                    continue

                for j in cells.get((nx, ny), []):
                    if j <= i:
                        continue

                    xj, yj = xy[j]
                    ddx = xi - xj
                    ddy = yi - yj
                    if ddx * ddx + ddy * ddy <= r2:
                        edges_u.append(i)
                        edges_v.append(j)

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


def update_threshold_times(infected_fraction_prev, infected_fraction_new, t_new,
                           threshold_times, thresholds):
    """
    Record threshold times the first time each threshold is crossed.
    """
    for thr in thresholds:
        if threshold_times[thr] is None and infected_fraction_prev < thr <= infected_fraction_new:
            threshold_times[thr] = t_new
    return threshold_times


def finite_only(x):
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def summarise_distribution(x):
    x = finite_only(x)
    if len(x) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "iqr": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    std = np.std(x, ddof=1) if len(x) > 1 else 0.0
    return {
        "n": len(x),
        "mean": np.mean(x),
        "std": std,
        "median": np.median(x),
        "q25": np.percentile(x, 25),
        "q75": np.percentile(x, 75),
        "iqr": np.percentile(x, 75) - np.percentile(x, 25),
        "min": np.min(x),
        "max": np.max(x),
    }


def ks_critical_values(n1, n2):
    """
    Two-sample KS critical values for common alpha levels:
        alpha = 0.10, 0.05, 0.025, 0.01
    Using asymptotic constants:
        1.22, 1.36, 1.48, 1.63
    D_crit = c(alpha) * sqrt((n1+n2)/(n1*n2))
    """
    if n1 <= 0 or n2 <= 0:
        return {
            "ks_crit_10pct": np.nan,
            "ks_crit_5pct": np.nan,
            "ks_crit_2p5pct": np.nan,
            "ks_crit_1pct": np.nan,
        }

    factor = math.sqrt((n1 + n2) / (n1 * n2))
    return {
        "ks_crit_10pct": 1.22 * factor,
        "ks_crit_5pct": 1.36 * factor,
        "ks_crit_2p5pct": 1.48 * factor,
        "ks_crit_1pct": 1.63 * factor,
    }


def compare_distributions(x_dyn, x_stat):
    x_dyn = finite_only(x_dyn)
    x_stat = finite_only(x_stat)

    if len(x_dyn) < 2 or len(x_stat) < 2:
        return {
            "n_dynamic": len(x_dyn),
            "n_static": len(x_stat),
            "mean_dynamic": np.nan,
            "mean_static": np.nan,
            "delta_mean": np.nan,
            "std_dynamic": np.nan,
            "std_static": np.nan,
            "delta_std": np.nan,
            "median_dynamic": np.nan,
            "median_static": np.nan,
            "delta_median": np.nan,
            "wasserstein": np.nan,
            "ks_statistic": np.nan,
            "ks_pvalue": np.nan,
            "ks_crit_10pct": np.nan,
            "ks_crit_5pct": np.nan,
            "ks_crit_2p5pct": np.nan,
            "ks_crit_1pct": np.nan,
            "ad_statistic": np.nan,
            "ad_significance_level_percent": np.nan,
            "ad_crit_25pct": np.nan,
            "ad_crit_10pct": np.nan,
            "ad_crit_5pct": np.nan,
            "ad_crit_2p5pct": np.nan,
            "ad_crit_1pct": np.nan,
            "ad_crit_0p5pct": np.nan,
            "ad_crit_0p1pct": np.nan,
        }

    dyn = summarise_distribution(x_dyn)
    stat = summarise_distribution(x_stat)

    ks = ks_2samp(x_dyn, x_stat)
    ks_crits = ks_critical_values(len(x_dyn), len(x_stat))

    ad = anderson_ksamp([x_dyn, x_stat])

    # SciPy returns these in order:
    # significance levels = [25, 10, 5, 2.5, 1, 0.5, 0.1]
    ad_cv = ad.critical_values

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

        "wasserstein": wasserstein_distance(x_dyn, x_stat),

        "ks_statistic": ks.statistic,
        "ks_pvalue": ks.pvalue,
        **ks_crits,

        "ad_statistic": ad.statistic,
        "ad_significance_level_percent": ad.significance_level,
        "ad_crit_25pct": ad_cv[0],
        "ad_crit_10pct": ad_cv[1],
        "ad_crit_5pct": ad_cv[2],
        "ad_crit_2p5pct": ad_cv[3],
        "ad_crit_1pct": ad_cv[4],
        "ad_crit_0p5pct": ad_cv[5],
        "ad_crit_0p1pct": ad_cv[6],
    }


# ============================================================
# THRESHOLD-ONLY SI SIMULATORS
# ============================================================

def simulate_dynamic_si_thresholds(graph, beta, mu, rng, thresholds, t_max=np.inf):
    """
    Dynamic SI on an RGG with edge flickering.
    Returns only threshold crossing times, not full I(t).
    """
    n = graph.n
    edges_u = graph.edges_u
    edges_v = graph.edges_v
    neighbors = graph.neighbors
    edge_ids = graph.edge_ids
    m_edges = len(edges_u)

    infected = np.zeros(n, dtype=np.bool_)
    infected_active_nbrs = np.zeros(n, dtype=np.int32)

    # stationary initialisation: each edge ON with probability 1/2
    active = rng.random(m_edges) < 0.5

    seed = int(rng.integers(n))
    infected[seed] = True
    infected_count = 1

    threshold_times = init_threshold_times(thresholds)
    ft = FenwickTree(n)

    # initialise infection propensities
    for nbr, e in zip(neighbors[seed], edge_ids[seed]):
        if not infected[nbr] and active[e]:
            infected_active_nbrs[nbr] += 1
            ft.add(nbr, beta)

    t = 0.0

    while infected_count < n and t < t_max:
        lambda_inf = ft.total()
        lambda_flip = mu * m_edges
        lambda_tot = lambda_inf + lambda_flip

        if lambda_tot <= 0.0:
            break

        dt = -math.log(rng.random()) / lambda_tot
        t += dt

        if t > t_max:
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

            if all(threshold_times[thr] is not None for thr in thresholds):
                break

        else:
            # edge flip event
            if m_edges == 0:
                continue

            e = int(rng.integers(m_edges))
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

    return threshold_times


def simulate_static_si_thresholds(graph, beta, rng, thresholds, t_max=np.inf):
    """
    Static SI on the RGG.
    Returns only threshold crossing times.
    """
    n = graph.n
    neighbors = graph.neighbors

    infected = np.zeros(n, dtype=np.bool_)
    infected_in_nbrs = np.zeros(n, dtype=np.int32)

    seed = int(rng.integers(n))
    infected[seed] = True
    infected_count = 1

    threshold_times = init_threshold_times(thresholds)
    ft = FenwickTree(n)

    for nbr in neighbors[seed]:
        if not infected[nbr]:
            infected_in_nbrs[nbr] += 1
            ft.add(nbr, beta)

    t = 0.0

    while infected_count < n and t < t_max:
        lambda_inf = ft.total()
        if lambda_inf <= 0.0:
            break

        dt = -math.log(rng.random()) / lambda_inf
        t += dt

        if t > t_max:
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

        if all(threshold_times[thr] is not None for thr in thresholds):
            break

    return threshold_times


# ============================================================
# WORKERS
# ============================================================

def dynamic_worker(args):
    graph_seed, sim_seed = args

    rng_graph = np.random.default_rng(int(graph_seed))
    rng_sim = np.random.default_rng(int(sim_seed))

    graph = generate_rgg_2d_graph(N, R_RADIUS, rng_graph)
    threshold_times = simulate_dynamic_si_thresholds(
        graph=graph,
        beta=BETA_DYNAMIC,
        mu=MU,
        rng=rng_sim,
        thresholds=THRESHOLDS,
        t_max=T_MAX
    )
    return threshold_times


def static_worker(args):
    graph_seed, sim_seed, beta_static = args

    rng_graph = np.random.default_rng(int(graph_seed))
    rng_sim = np.random.default_rng(int(sim_seed))

    graph = generate_rgg_2d_graph(N, R_RADIUS, rng_graph)
    threshold_times = simulate_static_si_thresholds(
        graph=graph,
        beta=beta_static,
        rng=rng_sim,
        thresholds=THRESHOLDS,
        t_max=T_MAX
    )
    return threshold_times


# ============================================================
# ENSEMBLE RUNNER
# ============================================================

def run_threshold_ensemble(worker_fn, arglist, n_jobs, thresholds):
    threshold_lists = {thr: [] for thr in thresholds}

    with mp.Pool(processes=n_jobs) as pool:
        for k, threshold_times in enumerate(pool.imap(worker_fn, arglist, chunksize=10), start=1):
            for thr in thresholds:
                val = threshold_times.get(thr, None)
                threshold_lists[thr].append(np.nan if val is None else val)

            if k % 100 == 0:
                print(f"Completed {k}/{len(arglist)}")

    threshold_arrays = {
        thr: np.array(vals, dtype=float)
        for thr, vals in threshold_lists.items()
    }
    return threshold_arrays


# ============================================================
# FITTING
# ============================================================

def objective_value(x_dyn, x_stat, objective):
    x_dyn = finite_only(x_dyn)
    x_stat = finite_only(x_stat)

    if len(x_dyn) == 0 or len(x_stat) == 0:
        return np.inf

    if objective == "wasserstein":
        return wasserstein_distance(x_dyn, x_stat)
    elif objective == "mean_abs":
        return abs(np.mean(x_dyn) - np.mean(x_stat))
    else:
        raise ValueError(f"Unknown FIT_OBJECTIVE: {objective}")


def fit_beta_per_threshold(dynamic_thresholds, static_results_by_beta, thresholds, objective):
    rows = []

    for thr in thresholds:
        best_beta = None
        best_obj = np.inf
        best_stats = None

        for beta_eff, static_thresholds in static_results_by_beta.items():
            x_dyn = dynamic_thresholds[thr]
            x_stat = static_thresholds[thr]

            obj = objective_value(x_dyn, x_stat, objective)

            if obj < best_obj:
                best_obj = obj
                best_beta = beta_eff
                best_stats = compare_distributions(x_dyn, x_stat)

        row = {
            "threshold": threshold_label(thr),
            "best_beta_eff": best_beta,
            "fit_objective": best_obj,
        }
        row.update(best_stats)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ============================================================
# PLOTTING
# ============================================================

def plot_t90_distribution(dynamic_thresholds, static_thresholds_best_t90, best_beta_t90):
    thr = 0.90
    td = finite_only(dynamic_thresholds[thr])
    ts = finite_only(static_thresholds_best_t90[thr])

    plt.figure(figsize=(8, 5))
    plt.hist(td, bins=40, density=True, alpha=0.5, label="Dynamic T90")
    plt.hist(ts, bins=40, density=True, alpha=0.5, label=f"Static T90, beta_eff={best_beta_t90:.4f}")
    plt.xlabel("T90")
    plt.ylabel("Density")
    plt.title("2D random geometric graph: T90 threshold-time distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    rng_master = np.random.default_rng(MASTER_SEED)

    # Same graph realisations reused everywhere
    graph_seeds = rng_master.integers(0, 2**32 - 1, size=N_RUNS, dtype=np.uint64)
    sim_seeds_dynamic = rng_master.integers(0, 2**32 - 1, size=N_RUNS, dtype=np.uint64)
    sim_seeds_static = rng_master.integers(0, 2**32 - 1, size=N_RUNS, dtype=np.uint64)

    print(f"Using 2D RGG radius R = {R_RADIUS:.6f}")
    print(f"N = {N}, target mean degree = {TARGET_MEAN_DEGREE}")
    print(f"N_RUNS = {N_RUNS}")
    print(f"Dynamic beta = {BETA_DYNAMIC}, mu = {MU}")
    print(f"Thresholds = {[threshold_label(t) for t in THRESHOLDS]}")
    print(f"Fit objective = {FIT_OBJECTIVE}")

    # --------------------------------------------------------
    # Dynamic ensemble
    # --------------------------------------------------------
    print("\nRunning dynamic threshold ensemble...")
    dyn_args = list(zip(graph_seeds, sim_seeds_dynamic))
    thresholds_dynamic = run_threshold_ensemble(
        dynamic_worker, dyn_args, N_JOBS, THRESHOLDS
    )

    # --------------------------------------------------------
    # Static ensembles for all beta values
    # --------------------------------------------------------
    print("\nRunning static threshold ensembles over beta grid...")
    static_results_by_beta = {}

    for beta_eff in BETA_GRID:
        print(f"\nTesting beta_eff = {beta_eff:.4f}")
        args = [(g, s, float(beta_eff)) for g, s in zip(graph_seeds, sim_seeds_static)]
        thresholds_static = run_threshold_ensemble(
            static_worker, args, N_JOBS, THRESHOLDS
        )
        static_results_by_beta[float(beta_eff)] = thresholds_static

    # --------------------------------------------------------
    # Fit beta separately for each threshold
    # --------------------------------------------------------
    print("\nFinding optimum beta_eff for each threshold...")
    results_df = fit_beta_per_threshold(
        dynamic_thresholds=thresholds_dynamic,
        static_results_by_beta=static_results_by_beta,
        thresholds=THRESHOLDS,
        objective=FIT_OBJECTIVE
    )

    # --------------------------------------------------------
    # Print compact results table
    # --------------------------------------------------------
    display_cols = [
        "threshold",
        "best_beta_eff",
        "fit_objective",
        "n_dynamic",
        "n_static",
        "mean_dynamic",
        "mean_static",
        "delta_mean",
        "std_dynamic",
        "std_static",
        "delta_std",
        "median_dynamic",
        "median_static",
        "delta_median",
        "wasserstein",
        "ks_statistic",
        "ks_pvalue",
        "ks_crit_5pct",
        "ad_statistic",
        "ad_significance_level_percent",
        "ad_crit_5pct",
    ]

    print("\n========================================================")
    print("BEST beta_eff FOR EACH THRESHOLD")
    print("========================================================")
    print(results_df[display_cols].to_string(index=False))

    # --------------------------------------------------------
    # Print detailed per-threshold diagnostics
    # --------------------------------------------------------
    print("\n========================================================")
    print("DETAILED DIAGNOSTICS")
    print("========================================================")

    for _, row in results_df.iterrows():
        print(f"\n{row['threshold']}:")
        print(f"  best_beta_eff = {row['best_beta_eff']:.4f}")
        print(f"  objective ({FIT_OBJECTIVE}) = {row['fit_objective']:.6f}")
        print(f"  n_dynamic = {int(row['n_dynamic'])}, n_static = {int(row['n_static'])}")
        print(f"  mean_dynamic = {row['mean_dynamic']:.6f}, mean_static = {row['mean_static']:.6f}, delta_mean = {row['delta_mean']:.6f}")
        print(f"  std_dynamic = {row['std_dynamic']:.6f}, std_static = {row['std_static']:.6f}, delta_std = {row['delta_std']:.6f}")
        print(f"  median_dynamic = {row['median_dynamic']:.6f}, median_static = {row['median_static']:.6f}, delta_median = {row['delta_median']:.6f}")

        print(f"  KS statistic = {row['ks_statistic']:.6f}, p-value = {row['ks_pvalue']:.6f}")
        print(
            "  KS critical values: "
            f"10%={row['ks_crit_10pct']:.6f}, "
            f"5%={row['ks_crit_5pct']:.6f}, "
            f"2.5%={row['ks_crit_2p5pct']:.6f}, "
            f"1%={row['ks_crit_1pct']:.6f}"
        )

        print(f"  AD statistic = {row['ad_statistic']:.6f}, significance_level_percent = {row['ad_significance_level_percent']:.6f}")
        print(
            "  AD critical values: "
            f"25%={row['ad_crit_25pct']:.6f}, "
            f"10%={row['ad_crit_10pct']:.6f}, "
            f"5%={row['ad_crit_5pct']:.6f}, "
            f"2.5%={row['ad_crit_2p5pct']:.6f}, "
            f"1%={row['ad_crit_1pct']:.6f}, "
            f"0.5%={row['ad_crit_0p5pct']:.6f}, "
            f"0.1%={row['ad_crit_0p1pct']:.6f}"
        )

        print(f"  Wasserstein = {row['wasserstein']:.6f}")

    # --------------------------------------------------------
    # Plot T90 distribution using its own best beta
    # --------------------------------------------------------
    t90_row = results_df.loc[results_df["threshold"] == "T90"].iloc[0]
    best_beta_t90 = float(t90_row["best_beta_eff"])
    static_thresholds_best_t90 = static_results_by_beta[best_beta_t90]

    print("\n========================================================")
    print(f"T90 best beta_eff = {best_beta_t90:.4f}")
    print("Plotting T90 distribution...")
    print("========================================================")

    plot_t90_distribution(
        dynamic_thresholds=thresholds_dynamic,
        static_thresholds_best_t90=static_thresholds_best_t90,
        best_beta_t90=best_beta_t90
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()