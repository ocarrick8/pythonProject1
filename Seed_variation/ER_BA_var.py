import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random, math
from scipy.stats import pearsonr

# --------------------------------------------------
# Parameters
# --------------------------------------------------
N = 100
ER_ps = [0.05, 0.10, 0.15]
BA_ms = [2, 3, 4]

TARGET_FRACTION = 0.7
INFECTION_RATE = 1.0
REPEATS = 3

random.seed(0)
np.random.seed(0)


# --------------------------------------------------
# Gillespie spreading simulation
# --------------------------------------------------
def simulate_time(G, seed):
    target = math.ceil(TARGET_FRACTION * G.number_of_nodes())
    infected = {seed}
    susceptible = set(G.nodes()) - infected
    t = 0.0

    neigh = {n: list(G.neighbors(n)) for n in G.nodes()}

    while len(infected) < target:
        edges = []
        for i in infected:
            for j in neigh[i]:
                if j in susceptible:
                    edges.append((i, j))

        if not edges:
            return np.inf

        dt = -math.log(random.random()) / (INFECTION_RATE * len(edges))
        t += dt

        _, j = random.choice(edges)
        infected.add(j)
        susceptible.remove(j)

    return t


# --------------------------------------------------
# Compute seed-node structural metrics
# --------------------------------------------------
def compute_node_metrics(G):
    degree = dict(G.degree())
    core = nx.core_number(G)
    clustering = nx.clustering(G)
    betweenness = nx.betweenness_centrality(G, normalized=True)
    farness = {n: nx.closeness_centrality(G) for n in G.nodes()}

    # Participation coefficient (requires community detection)
    com = nx.algorithms.community.louvain_communities(G, seed=0)
    comm_map = {}
    for cid, C in enumerate(com):
        for n in C:
            comm_map[n] = cid

    participation = {}
    for n in G.nodes():
        k = G.degree(n)
        if k == 0:
            participation[n] = 0
            continue

        links = np.zeros(len(com))
        for nbr in G.neighbors(n):
            links[comm_map[nbr]] += 1

        participation[n] = 1 - np.sum((links / k) ** 2)

    return {
        "degree": degree,
        "core": core,
        "clustering": clustering,
        "betweenness": betweenness,
        "participation": participation,
        "farness": farness
    }


# --------------------------------------------------
# Correlation
# --------------------------------------------------
def correlation_dict(metrics, times):
    corr = {}
    for name, values in metrics.items():
        x = np.array([values[n] for n in times.keys()])
        y = np.array([times[n] for n in times.keys()])
        if len(x) > 1 and np.isfinite(y).all():
            corr[name] = pearsonr(x, y)[0]
        else:
            corr[name] = np.nan
    return corr


# --------------------------------------------------
# Run ER experiment (vary p)
# --------------------------------------------------
ER_results = {k: [] for k in ["degree", "core", "clustering", "betweenness", "participation", "farness"]}

for p in ER_ps:
    G = nx.erdos_renyi_graph(N, p)

    metrics = compute_node_metrics(G)
    times = {n: np.mean([simulate_time(G, n) for _ in range(REPEATS)]) for n in G.nodes()}

    corr = correlation_dict(metrics, times)
    for key in ER_results:
        ER_results[key].append(corr[key])

# --------------------------------------------------
# Run BA experiment (vary m)
# --------------------------------------------------
BA_results = {k: [] for k in ["degree", "core", "clustering", "betweenness", "participation", "farness"]}

for m in BA_ms:
    G = nx.barabasi_albert_graph(N, m)

    metrics = compute_node_metrics(G)
    times = {n: np.mean([simulate_time(G, n) for _ in range(REPEATS)]) for n in G.nodes()}

    corr = correlation_dict(metrics, times)
    for key in BA_results:
        BA_results[key].append(corr[key])

# --------------------------------------------------
# Plot ER correlations vs p
# --------------------------------------------------
plt.figure(figsize=(12, 7))
for metric in ER_results:
    plt.plot(ER_ps, ER_results[metric], "-o", label=metric)

plt.xlabel("p (ER, N = 100)")
plt.ylabel("Correlation with spreading time")
plt.title("ER Model: Metric vs Spreading-Time Correlation (N = 100)")
plt.grid(True)
plt.legend()
plt.show()

# --------------------------------------------------
# Plot BA correlations vs m
# --------------------------------------------------
plt.figure(figsize=(12, 7))
for metric in BA_results:
    plt.plot(BA_ms, BA_results[metric], "-s", label=metric)

plt.xlabel("m (BA, N = 50)")
plt.ylabel("Correlation with spreading time")
plt.title("BA Model: Metric vs Spreading-Time Correlation (N = 50)")
plt.grid(True)
plt.legend()
plt.show()

