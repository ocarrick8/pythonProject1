from __future__ import annotations

import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from scipy.optimize import minimize_scalar
from scipy.stats import ks_2samp, wasserstein_distance, anderson_ksamp


# ============================================================
# Fenwick tree
# ============================================================
class FenwickTree:
    """
    Fenwick tree supporting:
      - point updates
      - prefix sums
      - total sum
      - sampling an index proportional to weights
    We use 0-based indexing externally.
    """

    def __init__(self, n: int):
        self.n = n
        self.tree = np.zeros(n + 1, dtype=float)
        self.arr = np.zeros(n, dtype=float)

    def build(self, values: np.ndarray) -> None:
        self.arr[:] = values
        self.tree[:] = 0.0
        for i in range(self.n):
            self._add_internal(i + 1, values[i])

    def _add_internal(self, i1: int, delta: float) -> None:
        n = self.n
        while i1 <= n:
            self.tree[i1] += delta
            i1 += i1 & -i1

    def set(self, i: int, value: float) -> None:
        delta = value - self.arr[i]
        if delta != 0.0:
            self.arr[i] = value
            self._add_internal(i + 1, delta)

    def total(self) -> float:
        return float(self.prefix_sum(self.n - 1)) if self.n > 0 else 0.0

    def prefix_sum(self, i: int) -> float:
        if i < 0:
            return 0.0
        s = 0.0
        i1 = i + 1
        while i1 > 0:
            s += self.tree[i1]
            i1 -= i1 & -i1
        return float(s)

    def find_by_cumulative(self, x: float) -> int:
        """
        Smallest index i such that prefix_sum(i) > x,
        assuming 0 <= x < total().
        """
        idx = 0
        bit = 1 << (self.n.bit_length() - 1)
        curr = 0.0

        while bit:
            nxt = idx + bit
            if nxt <= self.n and curr + self.tree[nxt] <= x:
                idx = nxt
                curr += self.tree[nxt]
            bit >>= 1

        return idx  # 0-based externally because idx is count of prefix items


# ============================================================
# BA graph generation
# ============================================================
def sample_ba_graph(N: int, m: int, rng: np.random.Generator) -> nx.Graph:
    return nx.barabasi_albert_graph(N, m, seed=int(rng.integers(0, 2**32 - 1)))


def build_edge_data(G: nx.Graph) -> tuple[np.ndarray, list[list[int]]]:
    """
    Returns:
      edges: (E,2) integer array
      adj_edge_ids: list of incident edge ids for each node
    """
    edges = np.array(list(G.edges()), dtype=int)
    E = len(edges)

    # canonical ordering
    u = np.minimum(edges[:, 0], edges[:, 1])
    v = np.maximum(edges[:, 0], edges[:, 1])
    edges = np.stack([u, v], axis=1)

    N = G.number_of_nodes()
    adj_edge_ids = [[] for _ in range(N)]
    for eid, (a, b) in enumerate(edges):
        adj_edge_ids[a].append(eid)
        adj_edge_ids[b].append(eid)

    return edges, adj_edge_ids


# ============================================================
# Static SI T90 using Fenwick-tree SSA
# ============================================================
def si_t90_static_fenwick(
    edges: np.ndarray,
    adj_edge_ids: list[list[int]],
    beta: float,
    rng: np.random.Generator,
    target_frac: float = 0.9,
    seed_node: int | None = None,
) -> float:
    N = len(adj_edge_ids)
    E = len(edges)
    target = math.ceil(target_frac * N)

    if E == 0:
        return float("inf")

    infected = np.zeros(N, dtype=bool)
    if seed_node is None:
        seed_node = int(rng.integers(0, N))
    infected[seed_node] = True
    I = 1

    def is_si_edge(eid: int) -> bool:
        a, b = edges[eid]
        return bool(infected[a] ^ infected[b])

    weights = np.zeros(E, dtype=float)
    for eid in adj_edge_ids[seed_node]:
        if is_si_edge(eid):
            weights[eid] = beta

    ft = FenwickTree(E)
    ft.build(weights)

    t = 0.0

    while I < target:
        total_rate = ft.total()
        if total_rate <= 0.0:
            return float("inf")

        t += rng.exponential(1.0 / total_rate)

        r = rng.random() * total_rate
        eid = ft.find_by_cumulative(r)

        a, b = edges[eid]
        sa, sb = infected[a], infected[b]

        if sa and not sb:
            new_inf = b
        elif sb and not sa:
            new_inf = a
        else:
            ft.set(eid, 0.0)
            continue

        infected[new_inf] = True
        I += 1

        # only edges touching the newly infected node can change SI status
        for feid in adj_edge_ids[new_inf]:
            ft.set(feid, beta if is_si_edge(feid) else 0.0)

    return t


# ============================================================
# Dynamic SI T90 using Fenwick-tree SSA
# ============================================================
def si_t90_dynamic_fenwick(
    edges: np.ndarray,
    adj_edge_ids: list[list[int]],
    beta: float,
    mu: float,
    rng: np.random.Generator,
    target_frac: float = 0.9,
    seed_node: int | None = None,
) -> float:
    """
    Exact SSA with a Fenwick tree over 2E channels:
      - channels [0, E): infection along ON SI edges, weight beta or 0
      - channels [E, 2E): edge flips, each weight mu
    """
    N = len(adj_edge_ids)
    E = len(edges)
    target = math.ceil(target_frac * N)

    if E == 0:
        return float("inf")

    infected = np.zeros(N, dtype=bool)
    if seed_node is None:
        seed_node = int(rng.integers(0, N))
    infected[seed_node] = True
    I = 1

    on = rng.random(E) < 0.5

    def is_si_edge(eid: int) -> bool:
        a, b = edges[eid]
        return bool(infected[a] ^ infected[b])

    weights = np.zeros(2 * E, dtype=float)

    # infection channels
    for eid in range(E):
        if on[eid] and is_si_edge(eid):
            weights[eid] = beta

    # flip channels: every edge flips at rate mu
    weights[E:] = mu

    ft = FenwickTree(2 * E)
    ft.build(weights)

    t = 0.0

    while I < target:
        total_rate = ft.total()
        if total_rate <= 0.0:
            return float("inf")

        t += rng.exponential(1.0 / total_rate)

        r = rng.random() * total_rate
        idx = ft.find_by_cumulative(r)

        # -------------------------
        # Infection event
        # -------------------------
        if idx < E:
            eid = idx
            a, b = edges[eid]
            sa, sb = infected[a], infected[b]

            if sa and not sb:
                new_inf = b
            elif sb and not sa:
                new_inf = a
            else:
                ft.set(eid, 0.0)
                continue

            infected[new_inf] = True
            I += 1

            for feid in adj_edge_ids[new_inf]:
                new_w = beta if (on[feid] and is_si_edge(feid)) else 0.0
                ft.set(feid, new_w)

        # -------------------------
        # Flip event
        # -------------------------
        else:
            eid = idx - E
            on[eid] = not on[eid]
            new_w = beta if (on[eid] and is_si_edge(eid)) else 0.0
            ft.set(eid, new_w)

    return t


# ============================================================
# Histogram helper
# ============================================================
def shared_hist_pdfs(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    bins: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xmin = float(min(a.min(), b.min(), c.min()))
    xmax = float(max(a.max(), b.max(), c.max()))

    ha, edges = np.histogram(a, bins=bins, range=(xmin, xmax), density=True)
    hb, _ = np.histogram(b, bins=bins, range=(xmin, xmax), density=True)
    hc, _ = np.histogram(c, bins=bins, range=(xmin, xmax), density=True)

    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, ha, hb, hc


# ============================================================
# Stats printing
# ============================================================
def print_dynamic_vs_effective_stats(
    mu: float,
    t_dynamic: np.ndarray,
    t_static_eff: np.ndarray,
    beta_eff: float,
) -> None:
    ks = ks_2samp(t_dynamic, t_static_eff, alternative="two-sided")
    w = float(wasserstein_distance(t_dynamic, t_static_eff))

    ad = anderson_ksamp([t_dynamic, t_static_eff])

    print("\n" + "=" * 78)
    print(f"mu = {mu}")
    print(f"beta_eff = {beta_eff:.6g}")
    print("--- Dynamic vs Static(beta_eff) at T90 ---")
    print(f"Median(Dynamic) = {np.median(t_dynamic):.6g}")
    print(f"Median(Static(beta_eff)) = {np.median(t_static_eff):.6g}")
    print(f"Δmedian (Dynamic-Static(beta_eff)) = {(np.median(t_dynamic) - np.median(t_static_eff)):.6g}")
    print(f"Mean(Dynamic) = {np.mean(t_dynamic):.6g}")
    print(f"Mean(Static(beta_eff)) = {np.mean(t_static_eff):.6g}")
    print(f"Δmean (Dynamic-Static(beta_eff)) = {(np.mean(t_dynamic) - np.mean(t_static_eff)):.6g}")
    print(f"KS: D = {ks.statistic:.6g}, p = {ks.pvalue:.6g}")
    print(f"Wasserstein distance = {w:.6g}")

    print("Anderson-Darling k-sample:")
    print(f"  statistic = {ad.statistic:.6g}")
    print(f"  significance_level ≈ {ad.significance_level:.6g}%")
    print(f"  critical_values = {np.array2string(ad.critical_values, precision=6)}")


# ============================================================
# Calibration for one mu
# ============================================================
def calibrate_beta_eff_for_mu(
    edge_data: list[tuple[np.ndarray, list[list[int]]]],
    seeds: list[int],
    *,
    beta0: float,
    mu: float,
    target_frac: float,
    mu_tag: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    n_runs = len(edge_data)

    # -------------------------
    # Dynamic samples
    # -------------------------
    rng_dyn = np.random.default_rng(10_000 + mu_tag)
    t_dynamic = np.empty(n_runs, dtype=float)

    for i, ((edges, adj_edge_ids), seed) in enumerate(zip(edge_data, seeds)):
        t_dynamic[i] = si_t90_dynamic_fenwick(
            edges=edges,
            adj_edge_ids=adj_edge_ids,
            beta=beta0,
            mu=mu,
            rng=rng_dyn,
            target_frac=target_frac,
            seed_node=seed,
        )

    if not np.all(np.isfinite(t_dynamic)):
        raise RuntimeError(f"Non-finite dynamic T90 values encountered for mu={mu}")

    # -------------------------
    # Baseline static(beta0)
    # -------------------------
    rng_static0 = np.random.default_rng(20_000 + mu_tag)
    t_static0 = np.empty(n_runs, dtype=float)

    for i, ((edges, adj_edge_ids), seed) in enumerate(zip(edge_data, seeds)):
        t_static0[i] = si_t90_static_fenwick(
            edges=edges,
            adj_edge_ids=adj_edge_ids,
            beta=beta0,
            rng=rng_static0,
            target_frac=target_frac,
            seed_node=seed,
        )

    if not np.all(np.isfinite(t_static0)):
        raise RuntimeError(f"Non-finite baseline static T90 values encountered for mu={mu}")

    # -------------------------
    # Loss function
    # -------------------------
    def loss(beta: float) -> float:
        if beta <= 0.0:
            return float("inf")

        local_seed = (1234567 + 7919 * mu_tag) ^ int(1_000_000 * beta)
        local_rng = np.random.default_rng(local_seed)

        t_stat = np.empty(n_runs, dtype=float)

        for i, ((edges, adj_edge_ids), seed) in enumerate(zip(edge_data, seeds)):
            t_stat[i] = si_t90_static_fenwick(
                edges=edges,
                adj_edge_ids=adj_edge_ids,
                beta=beta,
                rng=local_rng,
                target_frac=target_frac,
                seed_node=seed,
            )

        if not np.all(np.isfinite(t_stat)):
            return float("inf")

        return float(wasserstein_distance(t_stat, t_dynamic))

    # -------------------------
    # Coarse scan + refine
    # -------------------------
    beta_grid = np.linspace(0.05, 1.2, 24)
    grid_losses = np.array([loss(b) for b in beta_grid])

    j = int(np.argmin(grid_losses))
    lo = float(beta_grid[max(j - 1, 0)])
    hi = float(beta_grid[min(j + 1, len(beta_grid) - 1)])

    if lo == hi:
        lo = max(0.01, beta_grid[j] / 2.0)
        hi = beta_grid[j] * 2.0

    res = minimize_scalar(
        loss,
        bounds=(lo, hi),
        method="bounded",
        options={"maxiter": 30},
    )

    beta_eff = float(res.x)

    # -------------------------
    # Static samples at beta_eff
    # -------------------------
    rng_eff = np.random.default_rng(30_000 + mu_tag)
    t_static_eff = np.empty(n_runs, dtype=float)

    for i, ((edges, adj_edge_ids), seed) in enumerate(zip(edge_data, seeds)):
        t_static_eff[i] = si_t90_static_fenwick(
            edges=edges,
            adj_edge_ids=adj_edge_ids,
            beta=beta_eff,
            rng=rng_eff,
            target_frac=target_frac,
            seed_node=seed,
        )

    if not np.all(np.isfinite(t_static_eff)):
        raise RuntimeError(f"Non-finite optimised static T90 values encountered for mu={mu}")

    print_dynamic_vs_effective_stats(
        mu=mu,
        t_dynamic=t_dynamic,
        t_static_eff=t_static_eff,
        beta_eff=beta_eff,
    )

    return beta_eff, t_dynamic, t_static0, t_static_eff


# ============================================================
# Main
# ============================================================
def main():
    # -------------------------
    # Settings
    # -------------------------
    N = 1000
    m = 3
    n_runs = 2000

    beta0 = 1.0
    target_frac = 0.9   # T90
    bins = 35

    mu_values = [
        ("slow", 0.1),
        ("medium", 1.0),
    ]

    rng = np.random.default_rng(123)

    print("=== Effective-beta calibration on STATIC BA networks ===")
    print(f"N = {N}")
    print(f"m = {m}")
    print(f"runs = {n_runs}")
    print(f"approx mean degree ≈ {2*m}")
    print(f"beta0 = {beta0}")
    print(f"target_frac = {target_frac} (T90)")
    print("mu set:", ", ".join([f"{name}={mu}" for name, mu in mu_values]))

    # -------------------------
    # Freeze graphs + seed nodes across all mu
    # -------------------------
    edge_data: list[tuple[np.ndarray, list[list[int]]]] = []
    seeds: list[int] = []

    for _ in range(n_runs):
        G = sample_ba_graph(N, m, rng)
        edges, adj_edge_ids = build_edge_data(G)
        edge_data.append((edges, adj_edge_ids))
        seeds.append(int(rng.integers(0, N)))

    # -------------------------
    # Run all mu values
    # -------------------------
    results = []

    for mu_tag, (mu_name, mu) in enumerate(mu_values):
        print("\n" + "-" * 78)
        print(f"Running regime '{mu_name}' with mu = {mu}")

        beta_eff, t_dynamic, t_static0, t_static_eff = calibrate_beta_eff_for_mu(
            edge_data=edge_data,
            seeds=seeds,
            beta0=beta0,
            mu=mu,
            target_frac=target_frac,
            mu_tag=mu_tag,
        )

        results.append({
            "mu_name": mu_name,
            "mu": mu,
            "beta_eff": beta_eff,
            "t_dynamic": t_dynamic,
            "t_static0": t_static0,
            "t_static_eff": t_static_eff,
        })

    # -------------------------
    # One figure, 3 stacked subplots
    # -------------------------
    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(9, 13),
        sharex=False,
    )

    for ax, result in zip(axes, results):
        centers, pdf_dyn, pdf_stat0, pdf_stateff = shared_hist_pdfs(
            result["t_dynamic"],
            result["t_static0"],
            result["t_static_eff"],
            bins=bins,
        )

        mu_name = result["mu_name"]
        mu = result["mu"]
        beta_eff = result["beta_eff"]

        ax.plot(centers, pdf_dyn, label=f"Dynamic (β={beta0}, μ={mu})")
        ax.plot(centers, pdf_stat0, label=f"Static original (β={beta0})")
        ax.plot(centers, pdf_stateff, label=f"Static optimised (β_eff={beta_eff:.3f})")

        ax.set_title(f"{mu_name.capitalize()} switching: μ = {mu}")
        ax.set_xlabel("T90")
        ax.set_ylabel("PDF")
        ax.legend()

    fig.suptitle(
        f"BA T90 distribution comparison (N={N}, m={m}, runs={n_runs})",
        fontsize=14,
        y=0.995
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()