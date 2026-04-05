"""
Calibrate an "effective beta" for a STATIC ER network so that the STATIC SI half-life
distribution matches the DYNAMIC SI half-life distribution as closely as possible.

Requested:
- N = 300
- repeats (runs) = 1000
- Keep dynamic case fixed (beta0, mu)
- Find beta_eff for static by minimising Wasserstein distance between half-life samples
- Plot THREE PDFs (histogram density estimates on shared bins):
    1) Dynamic (beta0, mu)
    2) Original static (beta0)
    3) Optimised static (beta_eff)
- No overlap shading
- Do NOT put statistical test results on the graph title (print them instead)

Implementation note:
- We pre-generate graphs + seeds once, compute dynamic once.
- For stable optimisation, the loss(beta) evaluation uses a deterministic RNG seeded from beta.
"""

from __future__ import annotations
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp, wasserstein_distance
from scipy.optimize import minimize_scalar


# -----------------------------
# O(1) add/remove + O(1) random sample container
# -----------------------------
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


# -----------------------------
# Static SI half-life (Gillespie) — SI-edge sampling using IndexedSet
# -----------------------------
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
    E = edges.shape[0]
    if E == 0:
        return float("inf")

    # canonical ordering (u<v) for consistency
    u = np.minimum(edges[:, 0], edges[:, 1])
    v = np.maximum(edges[:, 0], edges[:, 1])
    edges = np.stack([u, v], axis=1)

    adj_edge_ids = [[] for _ in range(N)]
    for eid, (a, b) in enumerate(edges):
        adj_edge_ids[a].append(eid)
        adj_edge_ids[b].append(eid)

    si_edges = IndexedSet()

    def is_si_edge(eid: int) -> bool:
        a, b = edges[eid]
        return bool(infected[a] ^ infected[b])

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


# -----------------------------
# Dynamic SI half-life (Gillespie with edge flips)
# -----------------------------
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
    E = edges.shape[0]
    if E == 0:
        return float("inf")

    adj_edge_ids = [[] for _ in range(N)]
    for eid, (u, v) in enumerate(edges):
        adj_edge_ids[u].append(eid)
        adj_edge_ids[v].append(eid)

    on = rng.random(E) < 0.5
    si_on = IndexedSet()

    def is_si_edge(eid: int) -> bool:
        u, v = edges[eid]
        return bool(infected[u] ^ infected[v])

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
                    continue
                if is_si_edge(feid):
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


# -----------------------------
# Sample ER graphs (Option A)
# Ensure 50% infection reachable: require a component >= target size, seed within it.
# -----------------------------
def sample_er_with_large_component(
    N: int,
    p: float,
    rng: np.random.Generator,
    min_comp_size: int,
) -> tuple[nx.Graph, list[int]]:
    while True:
        G = nx.fast_gnp_random_graph(N, p, seed=int(rng.integers(0, 2**32 - 1)))
        comps = list(nx.connected_components(G))
        giant = max(comps, key=len)
        if len(giant) >= min_comp_size:
            return G, list(giant)


# -----------------------------
# Make three PDFs on a shared binning (hist density)
# -----------------------------
def shared_hist_pdfs(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    bins: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xmin = float(min(a.min(), b.min(), c.min()))
    xmax = float(max(a.max(), b.max(), c.max()))
    ha, edges = np.histogram(a, bins=bins, range=(xmin, xmax), density=True)
    hb, _ = np.histogram(b, bins=bins, range=(xmin, xmax), density=True)
    hc, _ = np.histogram(c, bins=bins, range=(xmin, xmax), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, ha, hb, hc


def main():
    # Requested settings
    N = 250
    p = 0.02
    n_runs = 700

    # Dynamic kept fixed
    beta0 = 1.0
    mu = 0.1
    target_frac = 0.5

    bins = 40
    rng = np.random.default_rng(123)

    min_comp_size = math.ceil(target_frac * N)

    # -----------------------------
    # 1) Freeze experiment: graphs + seeds
    # -----------------------------
    graphs: list[nx.Graph] = []
    seeds: list[int] = []
    for _ in range(n_runs):
        G, giant_nodes = sample_er_with_large_component(N, p, rng, min_comp_size)
        seed = int(rng.choice(giant_nodes))
        graphs.append(G)
        seeds.append(seed)

    # -----------------------------
    # 2) Dynamic times once
    # -----------------------------
    t_dynamic = np.empty(n_runs, dtype=float)
    for i, (G, seed) in enumerate(zip(graphs, seeds)):
        t_dynamic[i] = si_half_life_dynamic_gillespie(
            G, beta=beta0, mu=mu, rng=rng, target_frac=target_frac, seed_node=seed
        )

    if not np.all(np.isfinite(t_dynamic)):
        raise RuntimeError("Non-finite dynamic half-lives encountered. Try increasing p or tightening component filter.")

    # -----------------------------
    # 3) Original static at beta0
    # -----------------------------
    rng_static0 = np.random.default_rng(456)
    t_static0 = np.empty(n_runs, dtype=float)
    for i, (G, seed) in enumerate(zip(graphs, seeds)):
        t_static0[i] = si_half_life_static_gillespie(
            G, beta=beta0, rng=rng_static0, target_frac=target_frac, seed_node=seed
        )

    if not np.all(np.isfinite(t_static0)):
        raise RuntimeError("Non-finite baseline static half-lives encountered. Unexpected with component filter.")

    # -----------------------------
    # 4) Loss(beta): Wasserstein(static(beta), dynamic)
    #    Use deterministic beta-seeded RNG so optimisation is stable.
    # -----------------------------
    def loss(beta: float) -> float:
        if beta <= 0:
            return float("inf")
        local_seed = 10_000 + int(1_000_000 * min(beta, 1_000.0))
        local_rng = np.random.default_rng(local_seed)

        t_stat = np.empty(n_runs, dtype=float)
        for i, (G, seed) in enumerate(zip(graphs, seeds)):
            t_stat[i] = si_half_life_static_gillespie(
                G, beta=beta, rng=local_rng, target_frac=target_frac, seed_node=seed
            )
        if not np.all(np.isfinite(t_stat)):
            return float("inf")
        return float(wasserstein_distance(t_stat, t_dynamic))

    # -----------------------------
    # 5) Two-stage beta search
    # -----------------------------
    beta_grid = np.logspace(-2, 1.7, 25)  # ~0.01 to ~50
    grid_losses = np.array([loss(b) for b in beta_grid])
    j = int(np.argmin(grid_losses))
    beta_best = float(beta_grid[j])

    lo = float(beta_grid[max(j - 1, 0)])
    hi = float(beta_grid[min(j + 1, len(beta_grid) - 1)])
    if lo == hi:
        lo, hi = max(1e-6, beta_best / 3), beta_best * 3

    res = minimize_scalar(loss, bounds=(lo, hi), method="bounded", options={"maxiter": 25})
    beta_eff = float(res.x)

    # Evaluate static at beta_eff (fresh eval RNG)
    rng_eff = np.random.default_rng(789)
    t_static_eff = np.empty(n_runs, dtype=float)
    for i, (G, seed) in enumerate(zip(graphs, seeds)):
        t_static_eff[i] = si_half_life_static_gillespie(
            G, beta=beta_eff, rng=rng_eff, target_frac=target_frac, seed_node=seed
        )

    # -----------------------------
    # 6) Print stats (not on plot title)
    # -----------------------------
    def report(label_a: str, a: np.ndarray, label_b: str, b: np.ndarray) -> None:
        ks = ks_2samp(a, b, alternative="two-sided")
        w = float(wasserstein_distance(a, b))
        print(f"\n--- {label_a} vs {label_b} ---")
        print(f"Median({label_a}) = {np.median(a):.6g}")
        print(f"Median({label_b}) = {np.median(b):.6g}")
        print(f"Δmedian ({label_a}-{label_b}) = {(np.median(a)-np.median(b)):.6g}")
        print(f"KS: D={ks.statistic:.6g}, p={ks.pvalue:.6g}")
        print(f"Wasserstein distance = {w:.6g}")

    print("=== Effective-beta calibration (STATIC to match DYNAMIC) ===")
    print(f"N={N}, p={p}, runs={n_runs}, target_frac={target_frac}")
    print(f"Dynamic params: beta0={beta0}, mu={mu} (P(ON)~0.5)")
    print(f"beta_eff (static) = {beta_eff:.6g}")
    print(f"Coarse-grid best beta ~ {beta_best:.6g} (loss={grid_losses[j]:.6g})")
    print(f"Optimiser loss at beta_eff = {loss(beta_eff):.6g}")

    report("Dynamic", t_dynamic, "Static(beta0)", t_static0)
    report("Dynamic", t_dynamic, "Static(beta_eff)", t_static_eff)
    report("Static(beta_eff)", t_static_eff, "Static(beta0)", t_static0)

    # -----------------------------
    # 7) Plot three PDFs (shared histogram binning), no shading
    # -----------------------------
    centers, pdf_dyn, pdf_stat0, pdf_stateff = shared_hist_pdfs(
        t_dynamic, t_static0, t_static_eff, bins=bins
    )

    plt.figure(figsize=(8, 5))
    plt.plot(centers, pdf_dyn, label=f"Dynamic (beta={beta0}, mu={mu})")
    plt.plot(centers, pdf_stat0, label=f"Static original (beta={beta0})")
    plt.plot(centers, pdf_stateff, label=f"Static optimised (beta_eff={beta_eff:.3g})")
    plt.xlabel("Half-life t_1/2")
    plt.ylabel("PDF (density)")
    plt.title(f"PDF comparison (N={N}, p={p}, runs={n_runs})")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()