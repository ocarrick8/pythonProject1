"""
Option A: New ER graph each run.
- Measure SI half-life t_1/2 (time to reach 50% infected) for:
  (1) Static ER: edges always live
  (2) Dynamic ER: same underlying edges, but each edge toggles ON/OFF (stationary ON fraction = 0.5)

Requires: networkx, numpy, scipy, matplotlib
"""

from __future__ import annotations
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu, ks_2samp, gaussian_kde
from scipy.stats import wasserstein_distance


# -----------------------------
# Small helper: O(1) add/remove + O(1) random sampling
# -----------------------------
class IndexedSet:
    def __init__(self):
        self.items = []
        self.pos = {}

    def __len__(self):
        return len(self.items)

    def add(self, x):
        if x in self.pos:
            return
        self.pos[x] = len(self.items)
        self.items.append(x)

    def discard(self, x):
        i = self.pos.pop(x, None)
        if i is None:
            return
        last = self.items.pop()
        if i < len(self.items):
            self.items[i] = last
            self.pos[last] = i

    def sample(self, rng: np.random.Generator):
        return self.items[rng.integers(0, len(self.items))]


# -----------------------------
# SI half-life on a STATIC graph (Gillespie)
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

    # active SI edges = edges with one infected endpoint and one susceptible endpoint
    si_edges = IndexedSet()
    # store edges as (min(u,v), max(u,v)) tuples for stable keys
    for v in G.neighbors(seed_node):
        a, b = (seed_node, v) if seed_node < v else (v, seed_node)
        si_edges.add((a, b))

    t = 0.0
    while I < target:
        M = len(si_edges)
        if M == 0:
            return float("inf")  # cannot reach target (seed stuck in its component)

        rate = beta * M
        t += rng.exponential(1.0 / rate)

        # choose a random SI edge, infect the susceptible endpoint
        u, v = si_edges.sample(rng)
        su, sv = infected[u], infected[v]
        if su and not sv:
            new_inf = v
        elif sv and not su:
            new_inf = u
        else:
            # stale edge (should be rare); remove and continue
            si_edges.discard((u, v))
            continue

        # infect
        infected[new_inf] = True
        I += 1

        # update SI edge set: edges incident to new_inf can change status
        for w in G.neighbors(new_inf):
            a, b = (new_inf, w) if new_inf < w else (w, new_inf)
            if infected[w]:
                # edge became II -> not SI
                si_edges.discard((a, b))
            else:
                # edge became IS -> SI
                si_edges.add((a, b))

    return t


# -----------------------------
# SI half-life on a DYNAMIC graph:
# - Underlying edges fixed (same ER realisation)
# - Each edge toggles ON/OFF as a Poisson process with rate mu per edge
#   (symmetric toggling => stationary ON fraction = 0.5)
# - Infection can occur only across ON edges
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
    E = G.number_of_edges()
    target = math.ceil(target_frac * N)

    infected = np.zeros(N, dtype=bool)
    if seed_node is None:
        seed_node = int(rng.integers(0, N))
    infected[seed_node] = True
    I = 1

    # edge indexing
    edges = [(u, v) for u, v in G.edges()]
    # initial ON/OFF states with P(ON)=0.5
    on = rng.random(E) < 0.5

    # active SI edges among ON edges
    si_on = IndexedSet()

    # build adjacency map to edge indices for fast updates
    adj_edge_ids = [[] for _ in range(N)]
    for eid, (u, v) in enumerate(edges):
        adj_edge_ids[u].append(eid)
        adj_edge_ids[v].append(eid)

    def edge_key(u, v):
        return (u, v) if u < v else (v, u)

    # initialize si_on for edges incident to seed_node that are ON
    for eid in adj_edge_ids[seed_node]:
        if not on[eid]:
            continue
        u, v = edges[eid]
        other = v if u == seed_node else u
        if not infected[other]:
            si_on.add((eid, edge_key(u, v)))

    t = 0.0
    while I < target:
        M = len(si_on)          # number of ON edges that are currently SI
        R_inf = beta * M
        R_flip = mu * E
        R_tot = R_inf + R_flip

        if R_tot == 0.0:
            return float("inf")

        t += rng.exponential(1.0 / R_tot)

        if rng.random() < (R_inf / R_tot):
            # infection event: pick a random ON SI edge
            eid, (a, b) = si_on.sample(rng)
            u, v = edges[eid]
            su, sv = infected[u], infected[v]
            if su and not sv:
                new_inf = v
            elif sv and not su:
                new_inf = u
            else:
                # stale; remove
                si_on.discard((eid, (a, b)))
                continue

            infected[new_inf] = True
            I += 1

            # update all incident edges for new_inf
            for feid in adj_edge_ids[new_inf]:
                fu, fv = edges[feid]
                k = edge_key(fu, fv)
                if not on[feid]:
                    # OFF edges can't be in si_on
                    si_on.discard((feid, k))
                    continue

                other = fv if fu == new_inf else fu
                if infected[other]:
                    # II -> not SI
                    si_on.discard((feid, k))
                else:
                    # IS -> SI
                    si_on.add((feid, k))
        else:
            # flip event: pick a random edge uniformly and toggle it
            eid = int(rng.integers(0, E))
            on[eid] = not on[eid]
            u, v = edges[eid]
            k = edge_key(u, v)

            if on[eid]:
                # turning ON: add if it is SI
                if infected[u] ^ infected[v]:
                    si_on.add((eid, k))
            else:
                # turning OFF: remove if present
                si_on.discard((eid, k))

    return t


# -----------------------------
# Run Option A experiments
# -----------------------------
def sample_er_graph_with_giant_component(
    N: int,
    p: float,
    rng: np.random.Generator,
    min_comp_size: int,
) -> tuple[nx.Graph, list[int]]:
    """
    Draw ER graphs until the largest connected component has size >= min_comp_size.
    Returns (G, largest_component_nodes).
    """
    while True:
        G = nx.fast_gnp_random_graph(N, p, seed=int(rng.integers(0, 2**32 - 1)))
        comps = list(nx.connected_components(G))
        giant = max(comps, key=len)
        if len(giant) >= min_comp_size:
            return G, list(giant)


def bootstrap_ci_median_ratio(x, y, rng, n_boot=5000, alpha=0.05):
    x = np.asarray(x); y = np.asarray(y)
    n, m = len(x), len(y)
    ratios = np.empty(n_boot)
    for i in range(n_boot):
        xb = x[rng.integers(0, n, size=n)]
        yb = y[rng.integers(0, m, size=m)]
        ratios[i] = np.median(yb) / np.median(xb)
    lo = np.quantile(ratios, alpha/2)
    hi = np.quantile(ratios, 1 - alpha/2)
    return float(np.median(ratios)), (float(lo), float(hi))


def main():
    rng = np.random.default_rng(123)

    N = 500
    p = 0.01
    beta = 1.0          # infection rate per live SI edge
    mu = 5.0            # edge flip rate per edge (higher => faster flicker)
    q_stationary = 0.5  # implied by symmetric toggling (documented in text)
    n_runs = 1000
    target_frac = 0.5

    min_comp_size = math.ceil(target_frac * N)  # ensure half-life is reachable

    t_static = []
    t_dynamic = []

    for _ in range(n_runs):
        G, giant_nodes = sample_er_graph_with_giant_component(N, p, rng, min_comp_size)

        # choose seed inside the giant component so reaching 50% is feasible
        seed = int(rng.choice(giant_nodes))

        tS = si_half_life_static_gillespie(G, beta=beta, rng=rng, target_frac=target_frac, seed_node=seed)
        tD = si_half_life_dynamic_gillespie(G, beta=beta, mu=mu, rng=rng, target_frac=target_frac, seed_node=seed)

        t_static.append(tS)
        t_dynamic.append(tD)

    t_static = np.array(t_static, dtype=float)
    t_dynamic = np.array(t_dynamic, dtype=float)

    # -----------------------------
    # Stats: shift + shape + distance
    # -----------------------------
    # One-sided MWU: H1 = dynamic > static (slower spread)
    mwu = mannwhitneyu(t_dynamic, t_static, alternative="greater")
    ks = ks_2samp(t_dynamic, t_static, alternative="two-sided")
    wdist = wasserstein_distance(t_dynamic, t_static)

    medS, medD = np.median(t_static), np.median(t_dynamic)
    slowdown = medD / medS

    boot_med_ratio, boot_ci = bootstrap_ci_median_ratio(t_static, t_dynamic, rng=rng)

    print(f"Static median t_1/2:  {medS:.4g}")
    print(f"Dynamic median t_1/2: {medD:.4g}")
    print(f"Median slowdown factor (dyn/static): {slowdown:.4g}")
    print(f"Bootstrap median ratio ~ {boot_med_ratio:.4g} with 95% CI {boot_ci}")
    print(f"Mann–Whitney U (dyn > static): U={mwu.statistic:.4g}, p={mwu.pvalue:.4g}")
    print(f"KS two-sample: D={ks.statistic:.4g}, p={ks.pvalue:.4g}")
    print(f"Wasserstein distance: {wdist:.4g} (time units)")

    # -----------------------------
    # Plot "pdfs" (KDE) + hist overlay
    # -----------------------------
    xmin = min(t_static.min(), t_dynamic.min())
    xmax = max(t_static.max(), t_dynamic.max())
    xs = np.linspace(xmin, xmax, 500)

    kdeS = gaussian_kde(t_static)
    kdeD = gaussian_kde(t_dynamic)

    plt.figure(figsize=(8, 5))
    plt.hist(t_static, bins=40, density=True, alpha=0.35, label="Static (hist)")
    plt.hist(t_dynamic, bins=40, density=True, alpha=0.35, label="Dynamic (hist)")
    plt.plot(xs, kdeS(xs), label="Static (KDE)")
    plt.plot(xs, kdeD(xs), label="Dynamic (KDE)")
    plt.xlabel("Half-life t_1/2")
    plt.ylabel("Estimated pdf")
    plt.title(f"ER(N={N}, p={p}) half-life distributions | beta={beta}, mu={mu}, ON~{q_stationary}")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()