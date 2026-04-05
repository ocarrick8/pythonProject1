"""
Static edge-percolation approximation for a dynamic BA network.

Goal:
Approximate a dynamic SI process on a BA backbone by a static percolated BA graph.

Dynamic case:
- underlying BA graph with parameters (N, m)
- edges flicker ON/OFF with rate mu
- infection spreads with rate beta0 across currently ON SI edges

Static approximation:
- keep beta fixed at beta0
- instead of flickering edges, create a static percolated graph:
      each edge of the original BA graph is kept independently with probability p_keep
- optimise p_keep so the static percolated half-life distribution matches
  the dynamic half-life distribution as closely as possible

Only one dynamic case is used:
- mu = 0.1
"""

from __future__ import annotations
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp, wasserstein_distance
from scipy.optimize import minimize_scalar


# ============================================================
# O(1) set with random sampling
# ============================================================
class IndexedSet:
    def __init__(self):
        self.items: list[int] = []
        self.pos: dict[int, int] = {}

    def __len__(self) -> int:
        return len(self.items)

    def add(self, x: int) -> None:
        if x in self.pos:
            return
        self.pos[x] = len(self.items)
        self.items.append(x)

    def discard(self, x: int) -> None:
        i = self.pos.pop(x, None)
        if i is None:
            return
        last = self.items.pop()
        if i < len(self.items):
            self.items[i] = last
            self.pos[last] = i

    def sample(self, rng: np.random.Generator) -> int:
        return self.items[int(rng.integers(0, len(self.items)))]


# ============================================================
# Static SI half-life on a fixed graph
# ============================================================
def si_half_life_static_gillespie(
    G: nx.Graph,
    beta: float,
    rng: np.random.Generator,
    target_frac: float = 0.5,
    seed_node: int | None = None,
) -> float:
    N = G.number_of_nodes()
    target = math.ceil(target_frac * N)

    infected = np.zeros(N, dtype=bool)
    if seed_node is None:
        seed_node = int(rng.integers(0, N))
    infected[seed_node] = True
    I = 1

    edges = np.array(list(G.edges()), dtype=int)
    E = len(edges)
    if E == 0:
        return float("inf")

    u = np.minimum(edges[:, 0], edges[:, 1])
    v = np.maximum(edges[:, 0], edges[:, 1])
    edges = np.stack([u, v], axis=1)

    adj_edge_ids = [[] for _ in range(N)]
    for eid, (a, b) in enumerate(edges):
        adj_edge_ids[a].append(eid)
        adj_edge_ids[b].append(eid)

    def is_si_edge(eid: int) -> bool:
        a, b = edges[eid]
        return bool(infected[a] ^ infected[b])

    si_edges = IndexedSet()
    for eid in adj_edge_ids[seed_node]:
        if is_si_edge(eid):
            si_edges.add(eid)

    t = 0.0
    while I < target:
        M = len(si_edges)
        if M == 0:
            return float("inf")

        rate = beta * M
        t += rng.exponential(1.0 / rate)

        eid = si_edges.sample(rng)
        a, b = edges[eid]
        sa, sb = infected[a], infected[b]

        if sa and not sb:
            new_inf = b
        elif sb and not sa:
            new_inf = a
        else:
            si_edges.discard(eid)
            continue

        infected[new_inf] = True
        I += 1

        for feid in adj_edge_ids[new_inf]:
            if is_si_edge(feid):
                si_edges.add(feid)
            else:
                si_edges.discard(feid)

    return t


# ============================================================
# Dynamic SI half-life with flickering edges
# ============================================================
def si_half_life_dynamic_gillespie(
    G: nx.Graph,
    beta: float,
    mu: float,
    rng: np.random.Generator,
    target_frac: float = 0.5,
    seed_node: int | None = None,
) -> float:
    N = G.number_of_nodes()
    target = math.ceil(target_frac * N)

    infected = np.zeros(N, dtype=bool)
    if seed_node is None:
        seed_node = int(rng.integers(0, N))
    infected[seed_node] = True
    I = 1

    edges = np.array(list(G.edges()), dtype=int)
    E = len(edges)
    if E == 0:
        return float("inf")

    adj_edge_ids = [[] for _ in range(N)]
    for eid, (u, v) in enumerate(edges):
        adj_edge_ids[u].append(eid)
        adj_edge_ids[v].append(eid)

    on = rng.random(E) < 0.5

    def is_si_edge(eid: int) -> bool:
        u, v = edges[eid]
        return bool(infected[u] ^ infected[v])

    si_on = IndexedSet()
    for eid in adj_edge_ids[seed_node]:
        if on[eid] and is_si_edge(eid):
            si_on.add(eid)

    t = 0.0
    while I < target:
        M = len(si_on)
        R_inf = beta * M
        R_flip = mu * E
        R_tot = R_inf + R_flip

        if R_tot == 0.0:
            return float("inf")

        t += rng.exponential(1.0 / R_tot)

        if rng.random() < (R_inf / R_tot):
            if M == 0:
                continue

            eid = si_on.sample(rng)
            u, v = edges[eid]
            su, sv = infected[u], infected[v]

            if su and not sv:
                new_inf = v
            elif sv and not su:
                new_inf = u
            else:
                si_on.discard(eid)
                continue

            infected[new_inf] = True
            I += 1

            for feid in adj_edge_ids[new_inf]:
                if not on[feid]:
                    si_on.discard(feid)
                elif is_si_edge(feid):
                    si_on.add(feid)
                else:
                    si_on.discard(feid)
        else:
            eid = int(rng.integers(0, E))
            on[eid] = not on[eid]
            if on[eid]:
                if is_si_edge(eid):
                    si_on.add(eid)
            else:
                si_on.discard(eid)

    return t


# ============================================================
# BA graph generation
# ============================================================
def sample_ba_graph(N: int, m: int, rng: np.random.Generator) -> nx.Graph:
    return nx.barabasi_albert_graph(N, m, seed=int(rng.integers(0, 2**32 - 1)))


# ============================================================
# Static edge percolation: keep each edge independently with p_keep
# ============================================================
def percolate_graph(
    G: nx.Graph,
    p_keep: float,
    rng: np.random.Generator,
) -> nx.Graph:
    H = nx.Graph()
    H.add_nodes_from(G.nodes())

    for u, v in G.edges():
        if rng.random() < p_keep:
            H.add_edge(u, v)

    return H


# ============================================================
# Histogram helper
# ============================================================
def shared_hist_pdfs(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    bins: int = 35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xmin = float(min(a.min(), b.min(), c.min()))
    xmax = float(max(a.max(), b.max(), c.max()))

    ha, edges = np.histogram(a, bins=bins, range=(xmin, xmax), density=True)
    hb, _ = np.histogram(b, bins=bins, range=(xmin, xmax), density=True)
    hc, _ = np.histogram(c, bins=bins, range=(xmin, xmax), density=True)

    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, ha, hb, hc


# ============================================================
# Reporting
# ============================================================
def report(label_a: str, a: np.ndarray, label_b: str, b: np.ndarray) -> None:
    ks = ks_2samp(a, b, alternative="two-sided")
    w = float(wasserstein_distance(a, b))

    print(f"\n--- {label_a} vs {label_b} ---")
    print(f"Median({label_a}) = {np.median(a):.6g}")
    print(f"Median({label_b}) = {np.median(b):.6g}")
    print(f"Δmedian ({label_a}-{label_b}) = {(np.median(a) - np.median(b)):.6g}")
    print(f"KS: D = {ks.statistic:.6g}, p = {ks.pvalue:.6g}")
    print(f"Wasserstein distance = {w:.6g}")


# ============================================================
# Main calibration
# ============================================================
def main():
    # -------------------------
    # Settings
    # -------------------------
    N = 500
    m = 4
    n_runs = 200

    beta0 = 1.0
    mu = 0.1
    target_frac = 0.5
    bins = 35

    rng = np.random.default_rng(12345)

    print("=== Static edge-percolation approximation for dynamic BA network ===")
    print(f"N = {N}")
    print(f"m = {m}")
    print(f"runs = {n_runs}")
    print(f"beta0 = {beta0}")
    print(f"mu = {mu}")
    print(f"target_frac = {target_frac}")

    # -------------------------
    # Freeze BA graphs + seeds
    # -------------------------
    graphs: list[nx.Graph] = []
    seeds: list[int] = []

    for _ in range(n_runs):
        G = sample_ba_graph(N, m, rng)
        seed = int(rng.integers(0, N))
        graphs.append(G)
        seeds.append(seed)

    # -------------------------
    # Dynamic samples
    # -------------------------
    rng_dyn = np.random.default_rng(10_000)
    t_dynamic = np.empty(n_runs, dtype=float)

    for i, (G, seed) in enumerate(zip(graphs, seeds)):
        t_dynamic[i] = si_half_life_dynamic_gillespie(
            G=G,
            beta=beta0,
            mu=mu,
            rng=rng_dyn,
            target_frac=target_frac,
            seed_node=seed,
        )

    if not np.all(np.isfinite(t_dynamic)):
        raise RuntimeError("Dynamic samples contained non-finite values.")

    # -------------------------
    # Original static backbone, no percolation
    # -------------------------
    rng_static0 = np.random.default_rng(20_000)
    t_static0 = np.empty(n_runs, dtype=float)

    for i, (G, seed) in enumerate(zip(graphs, seeds)):
        t_static0[i] = si_half_life_static_gillespie(
            G=G,
            beta=beta0,
            rng=rng_static0,
            target_frac=target_frac,
            seed_node=seed,
        )

    if not np.all(np.isfinite(t_static0)):
        raise RuntimeError("Original static samples contained non-finite values.")

    # -------------------------
    # Loss function: optimise p_keep
    # -------------------------
    def loss(p_keep: float) -> float:
        if p_keep <= 0.0 or p_keep > 1.0:
            return float("inf")

        local_seed = 987654 + int(1_000_000 * p_keep)
        local_rng = np.random.default_rng(local_seed)

        t_perc = np.empty(n_runs, dtype=float)

        for i, (G, seed) in enumerate(zip(graphs, seeds)):
            H = percolate_graph(G, p_keep=p_keep, rng=local_rng)

            t_perc[i] = si_half_life_static_gillespie(
                G=H,
                beta=beta0,
                rng=local_rng,
                target_frac=target_frac,
                seed_node=seed,
            )

        # If percolation disconnects the graph too much, infection may not reach 50%
        # That should be penalised heavily.
        if not np.all(np.isfinite(t_perc)):
            return float("inf")

        return float(wasserstein_distance(t_perc, t_dynamic))

    # -------------------------
    # Coarse grid + refinement
    # -------------------------
    p_grid = np.linspace(0.2, 1.0, 17)
    grid_losses = np.array([loss(p) for p in p_grid])

    j = int(np.argmin(grid_losses))
    p_best = float(p_grid[j])

    lo = float(p_grid[max(j - 1, 0)])
    hi = float(p_grid[min(j + 1, len(p_grid) - 1)])

    if lo == hi:
        lo, hi = max(0.01, p_best - 0.1), min(1.0, p_best + 0.1)

    res = minimize_scalar(
        loss,
        bounds=(lo, hi),
        method="bounded",
        options={"maxiter": 25},
    )

    p_eff = float(res.x)

    # -------------------------
    # Evaluate best static-percolated approximation
    # -------------------------
    rng_eff = np.random.default_rng(30_000)
    t_static_perc = np.empty(n_runs, dtype=float)

    for i, (G, seed) in enumerate(zip(graphs, seeds)):
        H = percolate_graph(G, p_keep=p_eff, rng=rng_eff)

        t_static_perc[i] = si_half_life_static_gillespie(
            G=H,
            beta=beta0,
            rng=rng_eff,
            target_frac=target_frac,
            seed_node=seed,
        )

    if not np.all(np.isfinite(t_static_perc)):
        raise RuntimeError("Optimised percolated static samples contained non-finite values.")

    # -------------------------
    # Print summary
    # -------------------------
    print("\n" + "=" * 70)
    print(f"Optimised p_keep = {p_eff:.6g}")
    print(f"Coarse-grid best p_keep ~ {p_best:.6g} (loss = {grid_losses[j]:.6g})")
    print(f"Optimiser loss at p_eff = {loss(p_eff):.6g}")

    report("Dynamic", t_dynamic, "Static(percolated)", t_static_perc)

    # -------------------------
    # Plot PDFs
    # -------------------------
    centers, pdf_dyn, pdf_stat0, pdf_perc = shared_hist_pdfs(
        t_dynamic, t_static0, t_static_perc, bins=bins
    )

    plt.figure(figsize=(8.5, 5.2))
    plt.plot(centers, pdf_dyn, label=f"Dynamic (beta={beta0}, mu={mu})")
    plt.plot(centers, pdf_stat0, label=f"Static original BA (beta={beta0})")
    plt.plot(centers, pdf_perc, label=f"Static percolated BA (p_keep={p_eff:.3g})")
    plt.xlabel("Half-life $t_{1/2}$")
    plt.ylabel("PDF (density)")
    plt.title(f"Dynamic BA vs static percolated BA (N={N}, m={m}, runs={n_runs})")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()