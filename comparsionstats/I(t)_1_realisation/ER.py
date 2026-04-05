import math
import multiprocessing as mp
from dataclasses import dataclass
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, anderson_ksamp, wasserstein_distance


# ============================================================
# PARAMETERS
# ============================================================

N = 1000
K_MEAN = 6.0
P_ER = K_MEAN / (N - 1)

MU = 1.0                 # edge toggle rate
BETA_DYNAMIC = 1.0       # infection rate in dynamic model

DYNAMIC_RUNS = 3000
STATIC_CAL_RUNS = 100
STATIC_EXTRA_RUNS = 2900

BETA_GRID = np.linspace(0.40, 0.50, 21)

T_MAX = 10.0
DT_OUT = 0.05
T_GRID = np.arange(0.0, T_MAX + DT_OUT, DT_OUT)

THRESHOLD_FRACS = [0.10, 0.20, 0.30, 0.40, 0.50,
                   0.60, 0.70, 0.80, 0.90, 0.99]

MASTER_SEED = 123456
N_JOBS = max(1, mp.cpu_count() - 1)


# ============================================================
# FENWICK TREE
# ============================================================

class FenwickTree:
    """
    Fenwick tree / Binary Indexed Tree for non-negative integer weights.
    Supports:
      - point updates
      - prefix sums
      - total sum
      - inverse prefix lookup for weighted random sampling
    """
    def __init__(self, n):
        self.n = n
        self.tree = np.zeros(n + 1, dtype=np.int64)

    def build(self, arr):
        self.tree[:] = 0
        for i, val in enumerate(arr, start=1):
            self.tree[i] += val
            j = i + (i & -i)
            if j <= self.n:
                self.tree[j] += self.tree[i]

    def add(self, idx0, delta):
        """idx0 is 0-based."""
        i = idx0 + 1
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i

    def sum_prefix(self, idx0):
        """sum of [0, idx0], idx0 is 0-based."""
        s = 0
        i = idx0 + 1
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s

    def total(self):
        return self.sum_prefix(self.n - 1)

    def find_by_cumulative(self, target):
        """
        Smallest idx such that prefix_sum(idx) >= target,
        with target in {1, ..., total}.
        Returns 0-based index.
        """
        idx = 0
        bit = 1 << (self.n.bit_length() - 1)
        while bit:
            nxt = idx + bit
            if nxt <= self.n and self.tree[nxt] < target:
                target -= self.tree[nxt]
                idx = nxt
            bit >>= 1
        return idx  # 0-based because internal idx ends one below answer


# ============================================================
# GRAPH STORAGE
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


def init_worker(graph):
    global GLOBAL_GRAPH
    GLOBAL_GRAPH = graph


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
    trial = 0

    while True:
        trial += 1

        edges_u = []
        edges_v = []

        for i in range(n):
            # sample upper-triangle edges i < j
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

        if is_connected(neighbors):
            print(f"Connected ER graph found after {trial} trial(s).")
            print(f"N = {n}, M = {m}, <k> = {2*m/n:.4f}")
            return GraphData(
                n=n,
                m=m,
                edges_u=edges_u,
                edges_v=edges_v,
                neighbors=neighbors,
                edge_ids=edge_ids
            )


# ============================================================
# HELPERS
# ============================================================

def threshold_counts(n, fracs):
    # ceil so T10 means first time reaching at least 10% infected
    return np.array([math.ceil(f * n) for f in fracs], dtype=np.int32)


THRESH_COUNTS = threshold_counts(N, THRESHOLD_FRACS)


def fill_I_grid_piecewise(event_times, event_counts, t_grid, n):
    """
    event_times[k] = time immediately after infection count became event_counts[k]
    event_counts is monotone increasing.
    Return fraction infected on t_grid using left-continuous piecewise-constant path.
    """
    out = np.empty_like(t_grid, dtype=np.float64)

    j = 0
    current_count = event_counts[0]

    for i, t in enumerate(t_grid):
        while j + 1 < len(event_times) and event_times[j + 1] <= t:
            j += 1
            current_count = event_counts[j]
        out[i] = current_count / n

    return out


def choose_random_sus_neighbor_static(node, infected, neighbors, rng):
    cand = []
    for nb in neighbors[node]:
        if not infected[nb]:
            cand.append(nb)
    return cand[rng.integers(len(cand))]


def choose_random_active_sus_neighbor_dynamic(node, infected, neighbors, edge_ids, edge_state, rng):
    cand = []
    for nb, eid in zip(neighbors[node], edge_ids[node]):
        if edge_state[eid] and (not infected[nb]):
            cand.append(nb)
    return cand[rng.integers(len(cand))]


# ============================================================
# STATIC SI SIMULATION
# ============================================================

def simulate_static_once(seed, beta):
    """
    Static SI on the fixed graph realization.
    Infection events only.
    Uses Fenwick tree over infected nodes weighted by number of susceptible neighbors.
    """
    G = GLOBAL_GRAPH
    rng = np.random.default_rng(seed)

    n = G.n
    neighbors = G.neighbors

    infected = np.zeros(n, dtype=bool)
    sus_deg = np.zeros(n, dtype=np.int32)

    fenw = FenwickTree(n)

    # random seed node
    seed_node = int(rng.integers(n))
    infected[seed_node] = True

    sus_deg[seed_node] = len(neighbors[seed_node])
    fenw.add(seed_node, sus_deg[seed_node])

    t = 0.0
    infected_count = 1

    threshold_times = np.full(len(THRESH_COUNTS), np.nan, dtype=np.float64)
    next_thr_idx = 0
    while next_thr_idx < len(THRESH_COUNTS) and infected_count >= THRESH_COUNTS[next_thr_idx]:
        threshold_times[next_thr_idx] = t
        next_thr_idx += 1

    event_times = [0.0]
    event_counts = [1]

    while infected_count < n:
        total_is = fenw.total()
        if total_is <= 0:
            # Should not happen on a connected graph before full infection,
            # but keep a safe break.
            break

        rate_total = beta * total_is
        t += rng.exponential(1.0 / rate_total)

        # choose infected source node with weight = susceptible neighbor count
        target = int(rng.integers(1, total_is + 1))
        src = fenw.find_by_cumulative(target)

        # choose a susceptible neighbor uniformly among src's susceptible neighbors
        new_node = choose_random_sus_neighbor_static(src, infected, neighbors, rng)

        # infect new node
        infected[new_node] = True
        infected_count += 1

        # new infected node gets as weight the number of still-susceptible neighbors
        new_w = 0

        for nb in neighbors[new_node]:
            if infected[nb]:
                # nb was infected already: it loses new_node as susceptible neighbor
                sus_deg[nb] -= 1
                fenw.add(nb, -1)
            else:
                new_w += 1

        sus_deg[new_node] = new_w
        fenw.add(new_node, new_w)

        event_times.append(t)
        event_counts.append(infected_count)

        while next_thr_idx < len(THRESH_COUNTS) and infected_count >= THRESH_COUNTS[next_thr_idx]:
            threshold_times[next_thr_idx] = t
            next_thr_idx += 1

    I_grid = fill_I_grid_piecewise(np.array(event_times), np.array(event_counts), T_GRID, n)
    return I_grid, threshold_times


# ============================================================
# DYNAMIC SI SIMULATION
# ============================================================

def simulate_dynamic_once(seed):
    """
    Dynamic SI on fixed allowed ER graph.
    Each allowed edge flips ON <-> OFF at rate mu.
    With symmetric flip rate, stationary ON probability is 1/2.
    Infection can only pass along ON edges, at beta = 1.
    Uses Fenwick tree over infected nodes weighted by number of ACTIVE susceptible neighbors.
    """
    G = GLOBAL_GRAPH
    rng = np.random.default_rng(seed)

    n = G.n
    m = G.m
    neighbors = G.neighbors
    edge_ids = G.edge_ids
    edges_u = G.edges_u
    edges_v = G.edges_v

    infected = np.zeros(n, dtype=bool)

    # initial ON/OFF states, stationary Bernoulli(1/2)
    edge_state = rng.random(m) < 0.5

    active_sus_deg = np.zeros(n, dtype=np.int32)
    fenw = FenwickTree(n)

    seed_node = int(rng.integers(n))
    infected[seed_node] = True

    # weight of seed = number of active susceptible neighbors
    w0 = 0
    for nb, eid in zip(neighbors[seed_node], edge_ids[seed_node]):
        if edge_state[eid] and (not infected[nb]):
            w0 += 1
    active_sus_deg[seed_node] = w0
    fenw.add(seed_node, w0)

    t = 0.0
    infected_count = 1

    threshold_times = np.full(len(THRESH_COUNTS), np.nan, dtype=np.float64)
    next_thr_idx = 0
    while next_thr_idx < len(THRESH_COUNTS) and infected_count >= THRESH_COUNTS[next_thr_idx]:
        threshold_times[next_thr_idx] = t
        next_thr_idx += 1

    event_times = [0.0]
    event_counts = [1]

    toggle_rate_total = MU * m

    while infected_count < n:
        total_is_active = fenw.total()
        rate_inf = BETA_DYNAMIC * total_is_active
        rate_total = rate_inf + toggle_rate_total

        if rate_total <= 0:
            break

        t += rng.exponential(1.0 / rate_total)

        if rng.random() < (rate_inf / rate_total if rate_total > 0 else 0.0):
            # --------------------------
            # infection event
            # --------------------------
            if total_is_active <= 0:
                continue

            target = int(rng.integers(1, total_is_active + 1))
            src = fenw.find_by_cumulative(target)
            new_node = choose_random_active_sus_neighbor_dynamic(
                src, infected, neighbors, edge_ids, edge_state, rng
            )

            infected[new_node] = True
            infected_count += 1

            new_w = 0
            for nb, eid in zip(neighbors[new_node], edge_ids[new_node]):
                if infected[nb]:
                    # previously infected nb loses new_node as susceptible neighbor if edge is ON
                    if edge_state[eid]:
                        active_sus_deg[nb] -= 1
                        fenw.add(nb, -1)
                else:
                    # susceptible neighbor of new_node contributes if edge is ON
                    if edge_state[eid]:
                        new_w += 1

            active_sus_deg[new_node] = new_w
            fenw.add(new_node, new_w)

            event_times.append(t)
            event_counts.append(infected_count)

            while next_thr_idx < len(THRESH_COUNTS) and infected_count >= THRESH_COUNTS[next_thr_idx]:
                threshold_times[next_thr_idx] = t
                next_thr_idx += 1

        else:
            # --------------------------
            # edge toggle event
            # choose one edge uniformly
            # --------------------------
            eid = int(rng.integers(m))
            u = int(edges_u[eid])
            v = int(edges_v[eid])

            old_state = edge_state[eid]
            edge_state[eid] = not old_state
            delta = 1 if edge_state[eid] else -1

            # Only matters if one endpoint infected and the other susceptible
            iu = infected[u]
            iv = infected[v]

            if iu and (not iv):
                active_sus_deg[u] += delta
                fenw.add(u, delta)
            elif iv and (not iu):
                active_sus_deg[v] += delta
                fenw.add(v, delta)

    I_grid = fill_I_grid_piecewise(np.array(event_times), np.array(event_counts), T_GRID, n)
    return I_grid, threshold_times


# ============================================================
# BATCH RUNNERS
# ============================================================

def run_many_dynamic(seeds, n_jobs):
    with mp.Pool(processes=n_jobs, initializer=init_worker, initargs=(GLOBAL_GRAPH,)) as pool:
        results = pool.map(simulate_dynamic_once, seeds)

    I_all = np.array([r[0] for r in results], dtype=np.float64)
    T_all = np.array([r[1] for r in results], dtype=np.float64)
    return I_all, T_all


def run_many_static(beta, seeds, n_jobs):
    func = partial(simulate_static_once, beta=beta)
    with mp.Pool(processes=n_jobs, initializer=init_worker, initargs=(GLOBAL_GRAPH,)) as pool:
        results = pool.map(func, seeds)

    I_all = np.array([r[0] for r in results], dtype=np.float64)
    T_all = np.array([r[1] for r in results], dtype=np.float64)
    return I_all, T_all


# ============================================================
# STATISTICS
# ============================================================

def threshold_label(frac):
    return f"T{int(round(100 * frac))}"


def compare_samples(x_dyn, x_stat):
    out = {}

    out["mean_dyn"] = np.mean(x_dyn)
    out["mean_stat"] = np.mean(x_stat)
    out["delta_mean"] = out["mean_dyn"] - out["mean_stat"]

    out["median_dyn"] = np.median(x_dyn)
    out["median_stat"] = np.median(x_stat)
    out["delta_median"] = out["median_dyn"] - out["median_stat"]

    out["std_dyn"] = np.std(x_dyn, ddof=1)
    out["std_stat"] = np.std(x_stat, ddof=1)
    out["delta_std"] = out["std_dyn"] - out["std_stat"]

    ks = ks_2samp(x_dyn, x_stat)
    out["ks_stat"] = ks.statistic
    out["ks_pvalue"] = ks.pvalue

    ad = anderson_ksamp([x_dyn, x_stat])
    out["ad_stat"] = ad.statistic
    out["ad_significance_level"] = ad.significance_level
    out["ad_critical_values"] = np.array(ad.critical_values)

    out["wasserstein"] = wasserstein_distance(x_dyn, x_stat)
    return out


def print_threshold_stats(dynamic_thresholds, static_thresholds):
    print("\n" + "=" * 70)
    print("THRESHOLD-TIME COMPARISON STATISTICS")
    print("=" * 70)

    for j, frac in enumerate(THRESHOLD_FRACS):
        label = threshold_label(frac)
        x_dyn = dynamic_thresholds[:, j]
        x_stat = static_thresholds[:, j]

        stats = compare_samples(x_dyn, x_stat)

        print(f"\n{label}:")
        print(f"  n_dynamic = {len(x_dyn)}, n_static = {len(x_stat)}")
        print(f"  mean_dynamic   = {stats['mean_dyn']:.6f}")
        print(f"  mean_static    = {stats['mean_stat']:.6f}")
        print(f"  delta_mean     = {stats['delta_mean']:.6f}")
        print(f"  median_dynamic = {stats['median_dyn']:.6f}")
        print(f"  median_static  = {stats['median_stat']:.6f}")
        print(f"  delta_median   = {stats['delta_median']:.6f}")
        print(f"  std_dynamic    = {stats['std_dyn']:.6f}")
        print(f"  std_static     = {stats['std_stat']:.6f}")
        print(f"  delta_std      = {stats['delta_std']:.6f}")
        print(f"  KS statistic   = {stats['ks_stat']:.6f}")
        print(f"  KS p-value     = {stats['ks_pvalue']:.6g}")
        print(f"  AD statistic   = {stats['ad_stat']:.6f}")
        print(f"  AD signif lvl  = {stats['ad_significance_level']:.6g}%")
        print(f"  AD crit values = {np.array2string(stats['ad_critical_values'], precision=3)}")
        print(f"  Wasserstein    = {stats['wasserstein']:.6f}")


# ============================================================
# PLOTTING
# ============================================================

def plot_mean_curves(mean_I_dyn, mean_I_stat, beta_eff):
    plt.figure(figsize=(9, 6))
    plt.plot(T_GRID, mean_I_dyn, label="Dynamic")
    plt.plot(T_GRID, mean_I_stat, label=f"Static (beta_eff = {beta_eff:.4f})")
    plt.xlabel("t")
    plt.ylabel("<I(t)> / N")
    plt.title("Mean infection curves")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_threshold_pdfs(dynamic_thresholds, static_thresholds, beta_eff):
    fig, axes = plt.subplots(5, 2, figsize=(12, 16))
    axes = axes.ravel()

    for j, frac in enumerate(THRESHOLD_FRACS):
        ax = axes[j]
        x_dyn = dynamic_thresholds[:, j]
        x_stat = static_thresholds[:, j]

        lo = min(np.min(x_dyn), np.min(x_stat))
        hi = max(np.max(x_dyn), np.max(x_stat))
        bins = np.linspace(lo, hi, 40)

        ax.hist(x_dyn, bins=bins, density=True, histtype="step", linewidth=1.8, label="Dynamic")
        ax.hist(x_stat, bins=bins, density=True, histtype="step", linewidth=1.8,
                label=f"Static ({beta_eff:.4f})")

        ax.set_title(threshold_label(frac))
        ax.set_xlabel("threshold time")
        ax.set_ylabel("PDF")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_beta_scan(beta_grid, mse_vals):
    plt.figure(figsize=(8, 5))
    plt.plot(beta_grid, mse_vals, marker="o")
    plt.xlabel("beta_eff")
    plt.ylabel("MSE between mean I(t) curves")
    plt.title("Static beta calibration")
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    global GLOBAL_GRAPH

    # --------------------------------------------------------
    # Build one fixed connected ER graph
    # --------------------------------------------------------
    graph_seed = MASTER_SEED + 11
    GLOBAL_GRAPH = make_connected_er_graph(N, P_ER, graph_seed)

    # --------------------------------------------------------
    # Dynamic runs: 3000
    # --------------------------------------------------------
    print("\nRunning dynamic simulations...")
    dyn_seeds = [MASTER_SEED + 100000 + i for i in range(DYNAMIC_RUNS)]
    I_dyn_all, T_dyn_all = run_many_dynamic(dyn_seeds, N_JOBS)
    mean_I_dyn = I_dyn_all.mean(axis=0)

    # --------------------------------------------------------
    # Static calibration scan: 100 runs per beta
    # --------------------------------------------------------
    print("\nScanning static beta_eff values...")
    mse_vals = []
    static_scan_cache = {}

    for k, beta_eff in enumerate(BETA_GRID):
        print(f"  beta_eff = {beta_eff:.4f}")
        seeds = [MASTER_SEED + 200000 + 10000 * k + i for i in range(STATIC_CAL_RUNS)]
        I_stat_100, T_stat_100 = run_many_static(beta_eff, seeds, N_JOBS)
        mean_I_stat_100 = I_stat_100.mean(axis=0)

        mse = np.mean((mean_I_stat_100 - mean_I_dyn) ** 2)
        mse_vals.append(mse)

        static_scan_cache[beta_eff] = {
            "I": I_stat_100,
            "T": T_stat_100,
            "mean_I": mean_I_stat_100,
            "mse": mse
        }

    mse_vals = np.array(mse_vals)
    best_idx = int(np.argmin(mse_vals))
    beta_best = float(BETA_GRID[best_idx])

    print("\n" + "=" * 70)
    print("BEST STATIC BETA_EFF")
    print("=" * 70)
    print(f"best beta_eff = {beta_best:.4f}")
    print(f"MSE           = {mse_vals[best_idx]:.8f}")

    plot_beta_scan(BETA_GRID, mse_vals)

    # --------------------------------------------------------
    # Extra 2900 static runs at best beta
    # --------------------------------------------------------
    print(f"\nRunning extra static simulations at beta_eff = {beta_best:.4f}...")
    extra_seeds = [MASTER_SEED + 300000 + i for i in range(STATIC_EXTRA_RUNS)]
    I_stat_extra, T_stat_extra = run_many_static(beta_best, extra_seeds, N_JOBS)

    I_stat_100 = static_scan_cache[beta_best]["I"]
    T_stat_100 = static_scan_cache[beta_best]["T"]

    I_stat_all = np.vstack([I_stat_100, I_stat_extra])
    T_stat_all = np.vstack([T_stat_100, T_stat_extra])

    mean_I_stat = I_stat_all.mean(axis=0)

    # --------------------------------------------------------
    # Final plots
    # --------------------------------------------------------
    plot_mean_curves(mean_I_dyn, mean_I_stat, beta_best)
    plot_threshold_pdfs(T_dyn_all, T_stat_all, beta_best)

    # --------------------------------------------------------
    # Print threshold-time comparison stats
    # --------------------------------------------------------
    print_threshold_stats(T_dyn_all, T_stat_all)


if __name__ == "__main__":
    main()