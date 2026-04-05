import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

TARGET_FRACTION = 0.7
infection_rate = 1.0
REPEATS = 8  # << repeat each experiment 5 times

random.seed(0)

# -------------------------------------------------------
# Gillespie time to 70% infection for a given seed node
# -------------------------------------------------------
def gillespie_critical_time(G, seed):
    N = len(G)
    infected = {seed}
    t = 0.0
    neighbors = {n: list(G.neighbors(n)) for n in G.nodes()}

    while len(infected) < TARGET_FRACTION * N:
        active_edges = []
        for i in infected:
            for j in neighbors[i]:
                if j not in infected:
                    active_edges.append((i, j))

        if not active_edges:
            return np.inf  # cannot reach 70%

        rate_total = infection_rate * len(active_edges)
        dt = np.random.exponential(1 / rate_total)
        t += dt

        _, j = random.choice(active_edges)
        infected.add(j)

    return t

# -------------------------------------------------------
# Compute correlations for ONE graph
# -------------------------------------------------------
def compute_correlations_single(N, p):
    G = nx.erdos_renyi_graph(N, p)
    degrees = dict(G.degree())
    farness = {i: sum(nx.shortest_path_length(G, i).values()) for i in G.nodes()}

    crit_times = {}
    for seed in G.nodes():
        t = gillespie_critical_time(G, seed)
        crit_times[seed] = t

    seeds = list(G.nodes())
    deg_vals = np.array([degrees[s] for s in seeds])
    far_vals = np.array([farness[s] for s in seeds])
    time_vals = np.array([crit_times[s] for s in seeds])

    # Filter failed seeds
    mask = np.isfinite(time_vals)
    if mask.sum() < 3:
        return np.nan, np.nan

    deg_vals = deg_vals[mask]
    far_vals = far_vals[mask]
    time_vals = time_vals[mask]

    corr_deg = np.corrcoef(deg_vals, time_vals)[0, 1]
    corr_far = np.corrcoef(far_vals, time_vals)[0, 1]
    return corr_deg, corr_far

# -------------------------------------------------------
# Repeat correlation calculation REPEATS times
# -------------------------------------------------------
def compute_correlations(N, p):
    deg_corrs = []
    far_corrs = []

    for r in range(REPEATS):
        cdeg, cfar = compute_correlations_single(N, p)
        if np.isfinite(cdeg):
            deg_corrs.append(cdeg)
        if np.isfinite(cfar):
            far_corrs.append(cfar)

    if len(deg_corrs) == 0:
        return np.nan, np.nan

    return np.mean(deg_corrs), np.mean(far_corrs)

# -------------------------------------------------------
# EXPERIMENT 1 — vary N
# -------------------------------------------------------
Ns = list(range(25, 126, 10))
corr_deg_vs_N = []
corr_far_vs_N = []

for N in Ns:
    cdeg, cfar = compute_correlations(N, p=0.1)
    corr_deg_vs_N.append(cdeg)
    corr_far_vs_N.append(cfar)

# -------------------------------------------------------
# EXPERIMENT 2 — vary p
# -------------------------------------------------------
ps = np.arange(0.1, 0.4501, 0.025)
corr_deg_vs_p = []
corr_far_vs_p = []


for p in ps:

    cdeg, cfar = compute_correlations(N=50, p=p)
    corr_deg_vs_p.append(cdeg)
    corr_far_vs_p.append(cfar)


# -------------------------------------------------------
# PLOTS
# -------------------------------------------------------
plt.figure(figsize=(14,5))

# --- Correlation vs N ---
plt.subplot(1,2,1)
plt.plot(Ns, corr_deg_vs_N, "-o", label="Corr(degree, time)")
plt.plot(Ns, corr_far_vs_N, "-o", label="Corr(farness, time)")
plt.xlabel("Network size N")
plt.ylabel("Correlation")
plt.title("Avg Correlation vs N      (p = 0.1)")
plt.legend()
plt.grid(True)

# --- Correlation vs p ---
plt.subplot(1,2,2)
plt.plot(ps, corr_deg_vs_p, "-o", label="Corr(degree, time)")
plt.plot(ps, corr_far_vs_p, "-o", label="Corr(farness, time)")
plt.xlabel("Edge probability p")
plt.ylabel("Correlation")
plt.title("Avg Correlation vs p      (N = 50)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
