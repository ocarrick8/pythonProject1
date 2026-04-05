import math
import multiprocessing as mp
from dataclasses import dataclass
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import (
    ks_2samp,
    anderson_ksamp,
    wasserstein_distance,
    mannwhitneyu,
)

# ============================================================
# USER PARAMETERS
# ============================================================

# -------------------------
# Graph
# -------------------------
N = 1000
K_MEAN = 6.0
P_ER = K_MEAN / (N - 1)

# keep = 1 for one fixed graph
N_GRAPH_REALISATIONS = 1

# -------------------------
# Dynamic model
# -------------------------
MU = 1.0
BETA_DYNAMIC = 1.0

# -------------------------
# Simulation counts
# -------------------------
DYNAMIC_RUNS = 1000

COARSE_BETA_GRID = np.linspace(0.40, 0.50, 11)
COARSE_RUNS_PER_BETA = 100

FINE_STEP = 0.002
FINE_RUNS_PER_BETA = 300
BETA_MIN = 0.35
BETA_MAX = 0.55

NEAR_OPT_REL_TOL = 0.05
NEAR_OPT_ABS_TOL = None
REFINEMENT_MARGIN_STEPS = 1

BOUNDARY_BUFFER = 1
EXPANSION_STEPS = 4
MAX_EXPANSIONS = 3

FINAL_STATIC_RUNS = 1000

# Optional verification stage:
# re-evaluate the best few fine-grid betas with more runs before choosing final optimum
DO_VERIFY_TOP_CANDIDATES = True
VERIFY_TOP_K = 5
VERIFY_RUNS_PER_BETA = 1000

# -------------------------
# Time grid for <I(t)>
# -------------------------
T_MAX = 10.0
DT_CAL = 0.10
DT_FINAL = 0.05

# -------------------------
# Thresholds
# -------------------------
THRESHOLD_FRACS = [0.10, 0.20, 0.30, 0.40, 0.50,
                   0.60, 0.70, 0.80, 0.90, 0.99]

MODE_ANALYSIS_FRACS = {
    "global": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99],
    "bulk":   [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
    "tail":   [0.90, 0.99],
}

# Only plot histograms and ECDFs for T99 for global and tail
PLOT_SINGLE_THRESHOLD_ONLY = 0.99
PLOT_THRESHOLD_MODES = ["global", "tail"]

# -------------------------
# Bootstrap
# -------------------------
DO_BOOTSTRAP = False
BOOTSTRAP_REPS = 300

# -------------------------
# Parallelism / RNG
# -------------------------
MASTER_SEED = 123456789
N_JOBS = max(1, mp.cpu_count() - 1)

# Set True to debug serially
FORCE_SERIAL = False

# -------------------------
# Compare 3 optimisation modes
# -------------------------
OPTIMISATION_MODES = ["global", "bulk", "tail"]
PLOT_EACH_STAGE = False

# ============================================================
# FENWICK TREE
# ============================================================

class FenwickTree:
    """
    Fenwick tree over nonnegative integer weights.
    Used to sample infected source nodes proportional to current
    infection-capable degree.
    """
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
        """
        Returns smallest 0-based index idx such that prefix_sum(idx) >= target,
        assuming 1 <= target <= total().
        """
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
GLOBAL_T_GRID = None
GLOBAL_THRESH_COUNTS = None


def init_worker(graph, t_grid, thresh_counts):
    global GLOBAL_GRAPH, GLOBAL_T_GRID, GLOBAL_THRESH_COUNTS
    GLOBAL_GRAPH = graph
    GLOBAL_T_GRID = t_grid
    GLOBAL_THRESH_COUNTS = thresh_counts


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
                edge_ids=edge_ids,
            )


# ============================================================
# HELPERS
# ============================================================

def make_t_grid(dt):
    return np.arange(0.0, T_MAX + dt, dt)


def threshold_counts(n, fracs):
    return np.array([math.ceil(f * n) for f in fracs], dtype=np.int32)


def threshold_label(frac):
    return f"T{int(round(100 * frac))}"


def frac_to_index(frac):
    return THRESHOLD_FRACS.index(frac)


def fill_I_grid_piecewise(event_times, event_counts, t_grid, n):
    """
    event_times[j] is the time immediately after count became event_counts[j].
    Produces left-continuous piecewise-constant I(t)/N on t_grid.
    """
    out = np.empty_like(t_grid, dtype=np.float64)
    j = 0
    current = event_counts[0]

    for i, t in enumerate(t_grid):
        while j + 1 < len(event_times) and event_times[j + 1] <= t:
            j += 1
            current = event_counts[j]
        out[i] = current / n

    return out


def downsample_mean_curve(mean_curve_fine, dt_fine, dt_coarse):
    ratio = dt_coarse / dt_fine
    k = int(round(ratio))
    if not np.isclose(ratio, k):
        raise ValueError("DT_CAL must be an integer multiple of DT_FINAL for this downsampling.")
    return mean_curve_fine[::k]


def choose_random_sus_neighbor_static(node, infected, neighbors, rng):
    cand = [nb for nb in neighbors[node] if not infected[nb]]
    return cand[rng.integers(len(cand))]


def choose_random_active_sus_neighbor_dynamic(node, infected, neighbors, edge_ids, edge_state, rng):
    cand = []
    for nb, eid in zip(neighbors[node], edge_ids[node]):
        if edge_state[eid] and (not infected[nb]):
            cand.append(nb)
    return cand[rng.integers(len(cand))]


def standardised_wasserstein(x_dyn, x_stat, eps=1e-12):
    scale = np.std(x_dyn, ddof=1)
    if not np.isfinite(scale) or scale < eps:
        scale = max(np.std(x_stat, ddof=1), eps)
    return wasserstein_distance(x_dyn, x_stat) / scale


def objective_from_thresholds(T_dyn, T_stat, fracs):
    vals = []
    per_thr = {}
    for frac in fracs:
        j = frac_to_index(frac)
        x_dyn = T_dyn[:, j]
        x_stat = T_stat[:, j]
        w = standardised_wasserstein(x_dyn, x_stat)
        vals.append(w)
        per_thr[threshold_label(frac)] = w
    return float(np.mean(vals)), per_thr


# ============================================================
# STATIC SI
# ============================================================

def simulate_static_once(seed, beta):
    G = GLOBAL_GRAPH
    t_grid = GLOBAL_T_GRID
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

    threshold_times = np.full(len(thresh_counts), np.nan, dtype=np.float64)
    next_thr_idx = 0
    while next_thr_idx < len(thresh_counts) and infected_count >= thresh_counts[next_thr_idx]:
        threshold_times[next_thr_idx] = t
        next_thr_idx += 1

    event_times = [0.0]
    event_counts = [1]

    while infected_count < n:
        total_is = fenw.total()
        if total_is <= 0:
            break

        rate_total = beta * total_is
        t += rng.exponential(1.0 / rate_total)

        target = int(rng.integers(1, total_is + 1))
        src = fenw.find_by_cumulative(target)
        new_node = choose_random_sus_neighbor_static(src, infected, neighbors, rng)

        infected[new_node] = True
        infected_count += 1

        new_w = 0
        for nb in neighbors[new_node]:
            if infected[nb]:
                sus_deg[nb] -= 1
                fenw.add(nb, -1)
            else:
                new_w += 1

        sus_deg[new_node] = new_w
        fenw.add(new_node, new_w)

        event_times.append(t)
        event_counts.append(infected_count)

        while next_thr_idx < len(thresh_counts) and infected_count >= thresh_counts[next_thr_idx]:
            threshold_times[next_thr_idx] = t
            next_thr_idx += 1

    I_grid = fill_I_grid_piecewise(np.array(event_times), np.array(event_counts), t_grid, n)
    return I_grid, threshold_times


# ============================================================
# DYNAMIC SI
# ============================================================

def simulate_dynamic_once(seed):
    """
    Allowed edges are fixed by the ER base graph.
    Each allowed edge flips ON <-> OFF at rate mu.
    Infection can pass only along ON edges, with beta = 1.
    """
    G = GLOBAL_GRAPH
    t_grid = GLOBAL_T_GRID
    thresh_counts = GLOBAL_THRESH_COUNTS
    rng = np.random.default_rng(seed)

    n = G.n
    m = G.m
    neighbors = G.neighbors
    edge_ids = G.edge_ids
    edges_u = G.edges_u
    edges_v = G.edges_v

    infected = np.zeros(n, dtype=bool)

    # stationary ON distribution for symmetric toggling
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

        if rng.random() < (rate_inf / rate_total):
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
                        fenw.add(nb, -1)
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

            old_state = edge_state[eid]
            edge_state[eid] = not old_state
            delta = 1 if edge_state[eid] else -1

            iu = infected[u]
            iv = infected[v]

            if iu and (not iv):
                active_sus_deg[u] += delta
                fenw.add(u, delta)
            elif iv and (not iu):
                active_sus_deg[v] += delta
                fenw.add(v, delta)

    I_grid = fill_I_grid_piecewise(np.array(event_times), np.array(event_counts), t_grid, n)
    return I_grid, threshold_times


# ============================================================
# PARALLEL RUNNERS
# ============================================================

def run_many_dynamic(graph, t_grid, thresh_counts, seeds, n_jobs):
    if FORCE_SERIAL or n_jobs == 1:
        init_worker(graph, t_grid, thresh_counts)
        results = [simulate_dynamic_once(seed) for seed in seeds]
    else:
        with mp.Pool(
            processes=n_jobs,
            initializer=init_worker,
            initargs=(graph, t_grid, thresh_counts),
        ) as pool:
            results = pool.map(simulate_dynamic_once, seeds)

    I_all = np.array([r[0] for r in results], dtype=np.float64)
    T_all = np.array([r[1] for r in results], dtype=np.float64)
    return I_all, T_all


def run_many_static(graph, t_grid, thresh_counts, beta, seeds, n_jobs):
    func = partial(simulate_static_once, beta=beta)
    if FORCE_SERIAL or n_jobs == 1:
        init_worker(graph, t_grid, thresh_counts)
        results = [func(seed) for seed in seeds]
    else:
        with mp.Pool(
            processes=n_jobs,
            initializer=init_worker,
            initargs=(graph, t_grid, thresh_counts),
        ) as pool:
            results = pool.map(func, seeds)

    I_all = np.array([r[0] for r in results], dtype=np.float64)
    T_all = np.array([r[1] for r in results], dtype=np.float64)
    return I_all, T_all


# ============================================================
# CURVE METRICS / OBJECTIVES
# ============================================================

def curve_metrics(mean_dyn, mean_stat, dt):
    diff = mean_dyn - mean_stat
    mse = np.mean(diff ** 2)
    l1_area = np.sum(np.abs(diff)) * dt
    max_abs = np.max(np.abs(diff))
    return {
        "mse": mse,
        "l1_area": l1_area,
        "max_abs": max_abs,
    }


def scalarise_curve_metrics(metrics, weights=None):
    if weights is None:
        weights = {"mse": 1.0, "l1_area": 0.2, "max_abs": 0.2}
    return (
        weights["mse"] * metrics["mse"]
        + weights["l1_area"] * metrics["l1_area"]
        + weights["max_abs"] * metrics["max_abs"]
    )


# ============================================================
# THRESHOLD STATS
# ============================================================

def empirical_quantiles(x, probs=(0.10, 0.25, 0.50, 0.75, 0.90)):
    q = np.quantile(x, probs)
    return dict(zip(probs, q))


def compare_samples(x_dyn, x_stat):
    out = {}

    out["n_dynamic"] = len(x_dyn)
    out["n_static"] = len(x_stat)

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

    try:
        mw = mannwhitneyu(x_dyn, x_stat, alternative="two-sided")
        out["mw_stat"] = mw.statistic
        out["mw_pvalue"] = mw.pvalue
    except Exception:
        out["mw_stat"] = np.nan
        out["mw_pvalue"] = np.nan

    q_dyn = empirical_quantiles(x_dyn)
    q_stat = empirical_quantiles(x_stat)
    out["quantiles_dyn"] = q_dyn
    out["quantiles_stat"] = q_stat
    out["quantile_deltas"] = {p: q_dyn[p] - q_stat[p] for p in q_dyn.keys()}

    return out


# ============================================================
# BOOTSTRAP
# ============================================================

def bootstrap_ci_two_sample_stat(x, y, stat_func, reps=300, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.empty(reps, dtype=np.float64)
    n = len(x)
    m = len(y)

    for b in range(reps):
        xb = x[rng.integers(0, n, n)]
        yb = y[rng.integers(0, m, m)]
        vals[b] = stat_func(xb, yb)

    lo = np.quantile(vals, alpha / 2)
    hi = np.quantile(vals, 1 - alpha / 2)
    return lo, hi, vals


def print_bootstrap_summary(dynamic_thresholds, static_thresholds, beta_eff, fracs_to_use, label=""):
    suffix = f", {label}" if label else ""
    print("\n" + "=" * 80)
    print(f"BOOTSTRAP 95% CIs FOR DELTA_MEAN, DELTA_MEDIAN, WASSERSTEIN  (beta_eff = {beta_eff:.4f}{suffix})")
    print("=" * 80)

    for frac in fracs_to_use:
        j = frac_to_index(frac)
        lbl = threshold_label(frac)
        x_dyn = dynamic_thresholds[:, j]
        x_stat = static_thresholds[:, j]

        dm_lo, dm_hi, _ = bootstrap_ci_two_sample_stat(
            x_dyn, x_stat, lambda a, b: np.mean(a) - np.mean(b),
            reps=BOOTSTRAP_REPS, seed=1000 + j
        )
        dmed_lo, dmed_hi, _ = bootstrap_ci_two_sample_stat(
            x_dyn, x_stat, lambda a, b: np.median(a) - np.median(b),
            reps=BOOTSTRAP_REPS, seed=2000 + j
        )
        w_lo, w_hi, _ = bootstrap_ci_two_sample_stat(
            x_dyn, x_stat, lambda a, b: wasserstein_distance(a, b),
            reps=BOOTSTRAP_REPS, seed=3000 + j
        )

        print(f"\n{lbl}:")
        print(f"  delta_mean   95% CI = [{dm_lo:.6f}, {dm_hi:.6f}]")
        print(f"  delta_median 95% CI = [{dmed_lo:.6f}, {dmed_hi:.6f}]")
        print(f"  Wasserstein  95% CI = [{w_lo:.6f}, {w_hi:.6f}]")


# ============================================================
# PLOTTING
# ============================================================

def plot_scan(beta_grid, objective_vals, title):
    plt.figure(figsize=(8, 5))
    plt.plot(beta_grid, objective_vals, marker="o")
    plt.xlabel(r"$\beta_{\mathrm{eff}}$")
    plt.ylabel("scan objective")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_mean_curves_multiple(t_grid, mean_dyn, mean_curves_dict):
    plt.figure(figsize=(10, 6))
    plt.plot(t_grid, mean_dyn, label="Dynamic", linewidth=2.0)
    for label, (beta, curve) in mean_curves_dict.items():
        plt.plot(t_grid, curve, label=f"{label} static (beta_eff={beta:.4f})")
    plt.xlabel("t")
    plt.ylabel(r"$\langle I(t)\rangle / N$")
    plt.title("Mean infection curves: dynamic vs fitted static models")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_mean_curve_difference_multiple(t_grid, mean_dyn, mean_curves_dict):
    plt.figure(figsize=(10, 6))
    plt.axhline(0.0, linewidth=1.0)
    for label, (_, curve) in mean_curves_dict.items():
        plt.plot(t_grid, mean_dyn - curve, label=f"Dynamic - {label}")
    plt.xlabel("t")
    plt.ylabel(r"$\langle I_{dyn}(t)\rangle/N - \langle I_{stat}(t)\rangle/N$")
    plt.title("Difference of mean curves")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_single_threshold_pdf(dynamic_thresholds, static_thresholds, beta_eff, frac, title_prefix=""):
    j = frac_to_index(frac)
    x_dyn = dynamic_thresholds[:, j]
    x_stat = static_thresholds[:, j]

    plt.figure(figsize=(8, 5))
    lo = min(np.min(x_dyn), np.min(x_stat))
    hi = max(np.max(x_dyn), np.max(x_stat))
    bins = np.linspace(lo, hi, 40)

    plt.hist(x_dyn, bins=bins, density=True, histtype="step", linewidth=1.7, label="Dynamic")
    plt.hist(x_stat, bins=bins, density=True, histtype="step", linewidth=1.7,
             label=fr"Static ($\beta_{{eff}}={beta_eff:.4f}$)")
    plt.title(f"{title_prefix}{threshold_label(frac)} PDF")
    plt.xlabel("threshold time")
    plt.ylabel("PDF")
    plt.legend()
    plt.tight_layout()
    plt.show()


def ecdf_xy(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


def plot_single_threshold_ecdf(dynamic_thresholds, static_thresholds, beta_eff, frac, title_prefix=""):
    j = frac_to_index(frac)
    x_dyn = dynamic_thresholds[:, j]
    x_stat = static_thresholds[:, j]

    xs_d, ys_d = ecdf_xy(x_dyn)
    xs_s, ys_s = ecdf_xy(x_stat)

    plt.figure(figsize=(8, 5))
    plt.step(xs_d, ys_d, where="post", label="Dynamic")
    plt.step(xs_s, ys_s, where="post", label=fr"Static ($\beta_{{eff}}={beta_eff:.4f}$)")
    plt.title(f"{title_prefix}{threshold_label(frac)} ECDF")
    plt.xlabel("threshold time")
    plt.ylabel("ECDF")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# PRINTING
# ============================================================

def print_curve_metrics(metrics, header):
    print("\n" + "=" * 80)
    print(header)
    print("=" * 80)
    if "mse" in metrics:
        print(f"MSE          = {metrics['mse']:.10f}")
        print(f"L1 area      = {metrics['l1_area']:.10f}")
        print(f"Max abs diff = {metrics['max_abs']:.10f}")
    else:
        for k, v in metrics.items():
            if isinstance(v, dict):
                print(f"{k}:")
                for kk, vv in v.items():
                    print(f"  {kk} = {vv:.10f}")
            else:
                print(f"{k} = {v:.10f}")


def print_threshold_stats(dynamic_thresholds, static_thresholds, fracs_to_use, header="THRESHOLD-TIME COMPARISON STATISTICS"):
    print("\n" + "=" * 80)
    print(header)
    print("=" * 80)

    for frac in fracs_to_use:
        j = frac_to_index(frac)
        label = threshold_label(frac)
        x_dyn = dynamic_thresholds[:, j]
        x_stat = static_thresholds[:, j]

        s = compare_samples(x_dyn, x_stat)

        print(f"\n{label}:")
        print(f"  n_dynamic = {s['n_dynamic']}, n_static = {s['n_static']}")
        print(f"  mean_dynamic   = {s['mean_dyn']:.6f}")
        print(f"  mean_static    = {s['mean_stat']:.6f}")
        print(f"  delta_mean     = {s['delta_mean']:.6f}")
        print(f"  median_dynamic = {s['median_dyn']:.6f}")
        print(f"  median_static  = {s['median_stat']:.6f}")
        print(f"  delta_median   = {s['delta_median']:.6f}")
        print(f"  std_dynamic    = {s['std_dyn']:.6f}")
        print(f"  std_static     = {s['std_stat']:.6f}")
        print(f"  delta_std      = {s['delta_std']:.6f}")
        print(f"  KS statistic   = {s['ks_stat']:.6f}")
        print(f"  KS p-value     = {s['ks_pvalue']:.6g}")
        print(f"  AD statistic   = {s['ad_stat']:.6f}")
        print(f"  AD signif lvl  = {s['ad_significance_level']:.6g}%")
        print(f"  AD crit values = {np.array2string(s['ad_critical_values'], precision=3)}")
        print(f"  Wasserstein    = {s['wasserstein']:.6f}")
        print(f"  MW U stat      = {s['mw_stat']:.6f}")
        print(f"  MW p-value     = {s['mw_pvalue']:.6g}")

        print("  Quantiles dynamic:")
        for p, q in s["quantiles_dyn"].items():
            print(f"    q{int(100*p):02d} = {q:.6f}")

        print("  Quantiles static:")
        for p, q in s["quantiles_stat"].items():
            print(f"    q{int(100*p):02d} = {q:.6f}")

        print("  Quantile deltas (dynamic - static):")
        for p, dq in s["quantile_deltas"].items():
            print(f"    dq{int(100*p):02d} = {dq:.6f}")


# ============================================================
# BETA EVALUATION
# ============================================================

def evaluate_beta(
    graph,
    beta,
    t_grid,
    thresh_counts,
    dyn_mean_curve,
    dyn_thresholds,
    mode,
    static_seeds
):
    I_stat, T_stat = run_many_static(graph, t_grid, thresh_counts, beta, static_seeds, N_JOBS)
    mean_stat = I_stat.mean(axis=0)

    if mode == "global":
        metrics = curve_metrics(dyn_mean_curve, mean_stat, dt=t_grid[1] - t_grid[0])
        objective = scalarise_curve_metrics(metrics)

    elif mode == "bulk":
        objective, per_thr = objective_from_thresholds(
            dyn_thresholds, T_stat, MODE_ANALYSIS_FRACS["bulk"]
        )
        metrics = {"bulk_mean_standardised_wasserstein": objective, "per_threshold": per_thr}

    elif mode == "tail":
        objective, per_thr = objective_from_thresholds(
            dyn_thresholds, T_stat, MODE_ANALYSIS_FRACS["tail"]
        )
        metrics = {"tail_mean_standardised_wasserstein": objective, "per_threshold": per_thr}

    else:
        raise ValueError(f"Unknown optimisation mode: {mode}")

    return {
        "beta": beta,
        "I": I_stat,
        "T": T_stat,
        "mean_curve": mean_stat,
        "metrics": metrics,
        "objective": objective,
    }


def evaluate_beta_grid(
    graph,
    beta_grid,
    t_grid,
    thresh_counts,
    dyn_mean_curve,
    dyn_thresholds,
    mode,
    static_seeds,
):
    results = []
    for beta in beta_grid:
        print(f"  beta = {beta:.4f}")
        res = evaluate_beta(
            graph=graph,
            beta=float(beta),
            t_grid=t_grid,
            thresh_counts=thresh_counts,
            dyn_mean_curve=dyn_mean_curve,
            dyn_thresholds=dyn_thresholds,
            mode=mode,
            static_seeds=static_seeds,
        )
        results.append(res)
    return results


def best_result_from_grid(results):
    objs = np.array([r["objective"] for r in results], dtype=float)
    idx = int(np.argmin(objs))
    return idx, results[idx]


# ============================================================
# ADAPTIVE BETA SEARCH
# ============================================================

def build_refinement_grid_from_coarse(
    coarse_results,
    step,
    margin_steps=2,
    rel_tol=0.03,
    abs_tol=None,
    beta_min=0.0,
    beta_max=None,
):
    coarse_betas = np.array([r["beta"] for r in coarse_results], dtype=float)
    coarse_objs = np.array([r["objective"] for r in coarse_results], dtype=float)

    best_obj = np.min(coarse_objs)

    mask = coarse_objs <= best_obj * (1.0 + rel_tol)
    if abs_tol is not None:
        mask |= coarse_objs <= best_obj + abs_tol

    candidate_betas = coarse_betas[mask]
    lo = np.min(candidate_betas) - margin_steps * step
    hi = np.max(candidate_betas) + margin_steps * step

    lo = max(beta_min, lo)
    if beta_max is not None:
        hi = min(beta_max, hi)

    n_steps = int(round((hi - lo) / step))
    grid = lo + step * np.arange(n_steps + 1)
    grid = np.unique(np.round(grid, 12))
    return grid


def boundary_direction(best_idx, n_points, boundary_buffer=1):
    if best_idx <= boundary_buffer:
        return -1
    if best_idx >= n_points - 1 - boundary_buffer:
        return +1
    return 0


def expand_grid(beta_grid, direction, step, growth_steps=6, beta_min=0.0, beta_max=None):
    lo = float(np.min(beta_grid))
    hi = float(np.max(beta_grid))

    if direction < 0:
        new_lo = max(beta_min, lo - growth_steps * step)
        extra = new_lo + step * np.arange(int(round((lo - new_lo) / step)))
        new_grid = np.unique(np.round(np.concatenate([extra, beta_grid]), 12))
    elif direction > 0:
        new_hi = hi + growth_steps * step if beta_max is None else min(beta_max, hi + growth_steps * step)
        extra = hi + step * np.arange(1, int(round((new_hi - hi) / step)) + 1)
        new_grid = np.unique(np.round(np.concatenate([beta_grid, extra]), 12))
    else:
        new_grid = beta_grid.copy()

    return np.unique(np.round(new_grid, 12))


def merge_result_lists(old_results, new_results):
    by_beta = {}
    for r in old_results + new_results:
        by_beta[round(float(r["beta"]), 12)] = r
    merged = [by_beta[k] for k in sorted(by_beta.keys())]
    return merged


def adaptive_beta_scan(
    graph,
    dyn_mean_curve,
    dyn_thresholds,
    t_grid_cal,
    thresh_counts,
    coarse_beta_grid,
    coarse_runs_per_beta,
    fine_step,
    fine_runs_per_beta,
    seed_offset,
    mode,
    beta_min=0.0,
    beta_max=None,
    near_opt_rel_tol=0.03,
    near_opt_abs_tol=None,
    refinement_margin_steps=2,
    boundary_buffer=1,
    expansion_steps=6,
    max_expansions=6,
    plot_each_stage=True,
):
    """
    Optimisation modes:
      - global : fit whole mean curve
      - bulk   : fit threshold-time distributions T10..T80
      - tail   : fit threshold-time distributions T90 and T99
    """

    coarse_static_seeds = [seed_offset + i for i in range(coarse_runs_per_beta)]
    fine_static_seeds = [seed_offset + 1_000_000 + i for i in range(fine_runs_per_beta)]
    verify_static_seeds = [seed_offset + 2_000_000 + i for i in range(VERIFY_RUNS_PER_BETA)]

    print(f"\nRunning coarse beta scan [{mode}]...")
    coarse_results = evaluate_beta_grid(
        graph=graph,
        beta_grid=coarse_beta_grid,
        t_grid=t_grid_cal,
        thresh_counts=thresh_counts,
        dyn_mean_curve=dyn_mean_curve,
        dyn_thresholds=dyn_thresholds,
        mode=mode,
        static_seeds=coarse_static_seeds,
    )

    coarse_betas = np.array([r["beta"] for r in coarse_results], dtype=float)
    coarse_objs = np.array([r["objective"] for r in coarse_results], dtype=float)
    _, coarse_best = best_result_from_grid(coarse_results)

    print("\nBest coarse beta:")
    print(f"  beta = {coarse_best['beta']:.4f}")
    print(f"  objective = {coarse_best['objective']:.10f}")

    if plot_each_stage:
        plot_scan(coarse_betas, coarse_objs, f"Coarse scan objective vs beta_eff [{mode}]")

    fine_grid = build_refinement_grid_from_coarse(
        coarse_results=coarse_results,
        step=fine_step,
        margin_steps=refinement_margin_steps,
        rel_tol=near_opt_rel_tol,
        abs_tol=near_opt_abs_tol,
        beta_min=beta_min,
        beta_max=beta_max,
    )

    print("\nInitial fine grid:")
    print(f"  from {fine_grid.min():.4f} to {fine_grid.max():.4f} with step {fine_step:.4f}")

    fine_results = evaluate_beta_grid(
        graph=graph,
        beta_grid=fine_grid,
        t_grid=t_grid_cal,
        thresh_counts=thresh_counts,
        dyn_mean_curve=dyn_mean_curve,
        dyn_thresholds=dyn_thresholds,
        mode=mode,
        static_seeds=fine_static_seeds,
    )

    expansion_count = 0

    while True:
        fine_betas = np.array([r["beta"] for r in fine_results], dtype=float)
        fine_objs = np.array([r["objective"] for r in fine_results], dtype=float)
        best_idx, best_res = best_result_from_grid(fine_results)

        if plot_each_stage:
            plot_scan(fine_betas, fine_objs, f"Fine scan objective vs beta_eff [{mode}, expansion {expansion_count}]")

        print("\nCurrent best fine beta:")
        print(f"  beta = {best_res['beta']:.4f}")
        print(f"  objective = {best_res['objective']:.10f}")

        direction = boundary_direction(best_idx, len(fine_results), boundary_buffer=boundary_buffer)

        if direction == 0:
            print("  Optimum appears bracketed: best beta is interior to the fine grid.")
            break

        if expansion_count >= max_expansions:
            print("  Reached max expansions before fully bracketing the optimum.")
            break

        side = "upper" if direction > 0 else "lower"
        print(f"  Warning: best fine beta is too close to the {side} boundary.")
        print("  Expanding grid and continuing search...")

        expanded_grid = expand_grid(
            beta_grid=fine_betas,
            direction=direction,
            step=fine_step,
            growth_steps=expansion_steps,
            beta_min=beta_min,
            beta_max=beta_max,
        )

        old_set = set(np.round(fine_betas, 12))
        new_points = np.array([b for b in expanded_grid if round(float(b), 12) not in old_set], dtype=float)

        if len(new_points) == 0:
            print("  No new beta points available after attempted expansion.")
            break

        print(f"  New interval: {expanded_grid.min():.4f} to {expanded_grid.max():.4f}")
        print(f"  Evaluating {len(new_points)} new beta values...")

        new_results = evaluate_beta_grid(
            graph=graph,
            beta_grid=new_points,
            t_grid=t_grid_cal,
            thresh_counts=thresh_counts,
            dyn_mean_curve=dyn_mean_curve,
            dyn_thresholds=dyn_thresholds,
            mode=mode,
            static_seeds=fine_static_seeds,
        )

        fine_results = merge_result_lists(fine_results, new_results)
        expansion_count += 1

    fine_betas = np.array([r["beta"] for r in fine_results], dtype=float)

    if DO_VERIFY_TOP_CANDIDATES:
        fine_sorted = sorted(fine_results, key=lambda r: r["objective"])
        verify_betas = np.array([r["beta"] for r in fine_sorted[:VERIFY_TOP_K]], dtype=float)

        print("\nRunning verification stage on top fine candidates...")
        print(f"  candidates = {[round(float(b), 4) for b in verify_betas]}")
        print(f"  verification runs/beta = {VERIFY_RUNS_PER_BETA}")

        verify_results = evaluate_beta_grid(
            graph=graph,
            beta_grid=verify_betas,
            t_grid=t_grid_cal,
            thresh_counts=thresh_counts,
            dyn_mean_curve=dyn_mean_curve,
            dyn_thresholds=dyn_thresholds,
            mode=mode,
            static_seeds=verify_static_seeds,
        )

        _, final_best = best_result_from_grid(verify_results)

        print("\nVerification ranking:")
        for r in sorted(verify_results, key=lambda rr: rr["objective"]):
            print(f"  beta = {r['beta']:.4f}, objective = {r['objective']:.10f}")

    else:
        _, final_best = best_result_from_grid(fine_results)

    print("\n" + "=" * 80)
    print(f"ADAPTIVE BETA SEARCH RESULT [{mode}]")
    print("=" * 80)
    print(f"best beta_eff = {final_best['beta']:.4f}")
    print(f"objective     = {final_best['objective']:.10f}")
    print(f"search range  = [{fine_betas.min():.4f}, {fine_betas.max():.4f}]")
    print(f"grid points   = {len(fine_betas)}")
    print(f"expansions    = {expansion_count}")

    fine_sorted = sorted(fine_results, key=lambda r: r["objective"])
    print("\nTop 10 beta values by fine-stage objective:")
    for r in fine_sorted[:10]:
        print(f"  beta = {r['beta']:.4f}, objective = {r['objective']:.10f}")

    return {
        "coarse_results": coarse_results,
        "fine_results": fine_results,
        "best_beta": float(final_best["beta"]),
        "best_scan_result": final_best,
        "expansions_used": expansion_count,
        "final_grid_min": float(fine_betas.min()),
        "final_grid_max": float(fine_betas.max()),
        "mode": mode,
    }


# ============================================================
# FINAL FIT COMPARISON HELPERS
# ============================================================

def run_final_static_fit(graph, t_grid_final, thresh_counts, beta, seed_base):
    seeds = [seed_base + i for i in range(FINAL_STATIC_RUNS)]
    I_stat, T_stat = run_many_static(graph, t_grid_final, thresh_counts, beta, seeds, N_JOBS)
    mean_stat = I_stat.mean(axis=0)
    return mean_stat, T_stat


def print_beta_comparison(beta_dict):
    print("\n" + "=" * 80)
    print("BETA COMPARISON")
    print("=" * 80)
    for label, beta in beta_dict.items():
        print(f"{label:>8s} beta_eff = {beta:.4f}")


def quick_compare_key_thresholds(T_dyn, fits_dict):
    print("\n" + "=" * 80)
    print("KEY THRESHOLD DELTA-MEAN COMPARISON")
    print("=" * 80)
    for fit_label, fit_data in fits_dict.items():
        T_stat = fit_data["T_stat"]
        print(f"\n--- {fit_label} ---")
        for frac in MODE_ANALYSIS_FRACS[fit_label]:
            idx = frac_to_index(frac)
            delta = np.mean(T_dyn[:, idx]) - np.mean(T_stat[:, idx])
            print(f"  {threshold_label(frac)} delta mean = {delta:.6f}")


# ============================================================
# SINGLE GRAPH ANALYSIS
# ============================================================

def analyse_one_graph(graph_index, graph_seed):
    print("\n" + "#" * 90)
    print(f"GRAPH REALISATION {graph_index + 1} / {N_GRAPH_REALISATIONS}")
    print("#" * 90)

    graph = make_connected_er_graph(N, P_ER, graph_seed)

    t_grid_final = make_t_grid(DT_FINAL)
    t_grid_cal = make_t_grid(DT_CAL)
    thresh_counts_arr = threshold_counts(N, THRESHOLD_FRACS)

    # --------------------------------------------------------
    # Dynamic benchmark: ONE fixed dynamic ensemble used everywhere
    # --------------------------------------------------------
    print("\nRunning dynamic simulations...")
    dyn_seeds = [MASTER_SEED + 10_000_000 * (graph_index + 1) + i for i in range(DYNAMIC_RUNS)]
    I_dyn_final, T_dyn = run_many_dynamic(graph, t_grid_final, thresh_counts_arr, dyn_seeds, N_JOBS)
    mean_dyn_final = I_dyn_final.mean(axis=0)
    mean_dyn_cal = downsample_mean_curve(mean_dyn_final, DT_FINAL, DT_CAL)

    # --------------------------------------------------------
    # Run all 3 optimisation modes
    # --------------------------------------------------------
    scan_results = {}

    mode_seed_offsets = {
        "global": MASTER_SEED + 30_000_000 * (graph_index + 1),
        "bulk":   MASTER_SEED + 40_000_000 * (graph_index + 1),
        "tail":   MASTER_SEED + 50_000_000 * (graph_index + 1),
    }

    for mode in OPTIMISATION_MODES:
        scan_results[mode] = adaptive_beta_scan(
            graph=graph,
            dyn_mean_curve=mean_dyn_cal,
            dyn_thresholds=T_dyn,
            t_grid_cal=t_grid_cal,
            thresh_counts=thresh_counts_arr,
            coarse_beta_grid=COARSE_BETA_GRID,
            coarse_runs_per_beta=COARSE_RUNS_PER_BETA,
            fine_step=FINE_STEP,
            fine_runs_per_beta=FINE_RUNS_PER_BETA,
            seed_offset=mode_seed_offsets[mode],
            mode=mode,
            beta_min=BETA_MIN,
            beta_max=BETA_MAX,
            near_opt_rel_tol=NEAR_OPT_REL_TOL,
            near_opt_abs_tol=NEAR_OPT_ABS_TOL,
            refinement_margin_steps=REFINEMENT_MARGIN_STEPS,
            boundary_buffer=BOUNDARY_BUFFER,
            expansion_steps=EXPANSION_STEPS,
            max_expansions=MAX_EXPANSIONS,
            plot_each_stage=PLOT_EACH_STAGE,
        )

    beta_dict = {mode: scan_results[mode]["best_beta"] for mode in OPTIMISATION_MODES}
    print_beta_comparison(beta_dict)

    # --------------------------------------------------------
    # Final static runs for each fitted beta
    # --------------------------------------------------------
    fits = {}
    final_seed_offsets = {
        "global": MASTER_SEED + 60_000_000 * (graph_index + 1),
        "bulk":   MASTER_SEED + 70_000_000 * (graph_index + 1),
        "tail":   MASTER_SEED + 80_000_000 * (graph_index + 1),
    }

    for mode in OPTIMISATION_MODES:
        beta_best = beta_dict[mode]
        print(f"\nRunning final static simulations for [{mode}] at beta_eff = {beta_best:.4f} ...")
        mean_stat, T_stat = run_final_static_fit(
            graph=graph,
            t_grid_final=t_grid_final,
            thresh_counts=thresh_counts_arr,
            beta=beta_best,
            seed_base=final_seed_offsets[mode],
        )
        final_metrics = curve_metrics(mean_dyn_final, mean_stat, dt=DT_FINAL)
        print_curve_metrics(final_metrics, f"Final curve metrics: Dynamic vs Static(best beta) [{mode}]")

        fits[mode] = {
            "beta": beta_best,
            "mean_stat": mean_stat,
            "T_stat": T_stat,
            "final_metrics": final_metrics,
        }

    # --------------------------------------------------------
    # Comparison plots
    # --------------------------------------------------------
    mean_curves_dict = {
        mode: (fits[mode]["beta"], fits[mode]["mean_stat"])
        for mode in OPTIMISATION_MODES
    }
    plot_mean_curves_multiple(t_grid_final, mean_dyn_final, mean_curves_dict)
    plot_mean_curve_difference_multiple(t_grid_final, mean_dyn_final, mean_curves_dict)

    # --------------------------------------------------------
    # Per-fit threshold analysis
    # --------------------------------------------------------
    for mode in OPTIMISATION_MODES:
        beta = fits[mode]["beta"]
        T_stat = fits[mode]["T_stat"]
        fracs_to_use = MODE_ANALYSIS_FRACS[mode]

        print_threshold_stats(
            T_dyn,
            T_stat,
            fracs_to_use=fracs_to_use,
            header=f"THRESHOLD-TIME COMPARISON STATISTICS [{mode}]"
        )

        if DO_BOOTSTRAP:
            print_bootstrap_summary(
                T_dyn,
                T_stat,
                beta_eff=beta,
                fracs_to_use=fracs_to_use,
                label=mode
            )

        if mode in PLOT_THRESHOLD_MODES:
            plot_single_threshold_pdf(
                T_dyn, T_stat, beta, frac=PLOT_SINGLE_THRESHOLD_ONLY, title_prefix=f"[{mode}] "
            )
            plot_single_threshold_ecdf(
                T_dyn, T_stat, beta, frac=PLOT_SINGLE_THRESHOLD_ONLY, title_prefix=f"[{mode}] "
            )

    quick_compare_key_thresholds(T_dyn, fits)

    return {
        "graph": graph,
        "beta_dict": beta_dict,
        "mean_dyn_final": mean_dyn_final,
        "T_dyn": T_dyn,
        "fits": fits,
    }


# ============================================================
# MULTI-GRAPH SUMMARY
# ============================================================

def print_multi_graph_summary(results):
    if len(results) <= 1:
        return

    print("\n" + "=" * 90)
    print("MULTI-GRAPH SUMMARY")
    print("=" * 90)

    for mode in OPTIMISATION_MODES:
        betas = np.array([r["beta_dict"][mode] for r in results], dtype=np.float64)
        mses = np.array([r["fits"][mode]["final_metrics"]["mse"] for r in results], dtype=np.float64)
        l1s = np.array([r["fits"][mode]["final_metrics"]["l1_area"] for r in results], dtype=np.float64)
        maxabs = np.array([r["fits"][mode]["final_metrics"]["max_abs"] for r in results], dtype=np.float64)

        print(f"\nMode: {mode}")
        print(f"  beta_eff mean  = {np.mean(betas):.6f}")
        print(f"  beta_eff std   = {np.std(betas, ddof=1):.6f}")
        print(f"  beta_eff min   = {np.min(betas):.6f}")
        print(f"  beta_eff max   = {np.max(betas):.6f}")
        print(f"  MSE mean       = {np.mean(mses):.8f}")
        print(f"  MSE std        = {np.std(mses, ddof=1):.8f}")
        print(f"  L1 area mean   = {np.mean(l1s):.8f}")
        print(f"  L1 area std    = {np.std(l1s, ddof=1):.8f}")
        print(f"  Max abs mean   = {np.mean(maxabs):.8f}")
        print(f"  Max abs std    = {np.std(maxabs, ddof=1):.8f}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("\nStarting ER dynamic-vs-static SI comparison")
    print(f"N = {N}, <k> = {K_MEAN}, mu = {MU}, beta_dynamic = {BETA_DYNAMIC}")
    print(f"dynamic runs = {DYNAMIC_RUNS}")
    print(f"coarse scan betas = {COARSE_BETA_GRID}")
    print(f"coarse runs/beta = {COARSE_RUNS_PER_BETA}")
    print(f"fine step = {FINE_STEP}")
    print(f"fine runs/beta = {FINE_RUNS_PER_BETA}")
    print(f"verify top candidates = {DO_VERIFY_TOP_CANDIDATES}, top_k = {VERIFY_TOP_K}, verify runs/beta = {VERIFY_RUNS_PER_BETA}")
    print(f"beta search bounds = [{BETA_MIN}, {BETA_MAX}]")
    print(f"near-optimal rel tol = {NEAR_OPT_REL_TOL}")
    print(f"boundary buffer = {BOUNDARY_BUFFER}")
    print(f"expansion steps = {EXPANSION_STEPS}")
    print(f"max expansions = {MAX_EXPANSIONS}")
    print(f"final static runs = {FINAL_STATIC_RUNS}")
    print(f"thresholds = {[threshold_label(f) for f in THRESHOLD_FRACS]}")
    print(f"analysis thresholds by mode = {MODE_ANALYSIS_FRACS}")
    print(f"plot threshold-only modes = {PLOT_THRESHOLD_MODES}, threshold = {threshold_label(PLOT_SINGLE_THRESHOLD_ONLY)}")
    print(f"graph realisations = {N_GRAPH_REALISATIONS}")
    print(f"bootstrap enabled = {DO_BOOTSTRAP}, reps = {BOOTSTRAP_REPS}")
    print(f"parallel jobs = {N_JOBS}, force_serial = {FORCE_SERIAL}")
    print(f"optimisation modes = {OPTIMISATION_MODES}")

    all_results = []

    for g in range(N_GRAPH_REALISATIONS):
        graph_seed = MASTER_SEED + 1_000_000 * (g + 1)
        res = analyse_one_graph(g, graph_seed)
        all_results.append(res)

    print_multi_graph_summary(all_results)


if __name__ == "__main__":
    main()