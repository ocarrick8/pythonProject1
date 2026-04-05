import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

# ---------------------------
# PARAMETERS
# ---------------------------
N = 40
p = 0.1
infection_rate = 1.0
TARGET_FRACTION = 0.7   # 70% infection
random.seed(0)

# ---------------------------
# Build ER network
# ---------------------------
G = nx.erdos_renyi_graph(N, p)

# Precompute degrees and farness for all nodes
degrees = dict(G.degree())
farness = {i: sum(nx.shortest_path_length(G, i).values()) for i in G.nodes()}

# ---------------------------
# Gillespie simulation function
# ---------------------------
def gillespie_critical_time(seed):
    infected = {seed}
    t = 0.0

    # Create adjacency lookup for speed
    neighbors = {node: list(G.neighbors(node)) for node in G.nodes()}

    while len(infected) < TARGET_FRACTION * N:
        # Find all active edges (infected → susceptible)
        active_edges = []
        for i in infected:
            for j in neighbors[i]:
                if j not in infected:
                    active_edges.append((i, j))

        # If no more edges, infection stuck
        if not active_edges:
            return np.inf

        # Gillespie time step
        rate_total = infection_rate * len(active_edges)
        dt = np.random.exponential(scale=1/rate_total)
        t += dt

        # Choose one transmission uniformly from active edges
        _, j = random.choice(active_edges)
        infected.add(j)

    return t

# ---------------------------
# Run simulation for all seeds
# ---------------------------
critical_times = {}


for seed in G.nodes():
    t = gillespie_critical_time(seed)
    critical_times[seed] = t


# Convert to arrays for plotting
seeds = list(G.nodes())
deg_vals = np.array([degrees[i] for i in seeds])
far_vals = np.array([farness[i] for i in seeds])
crit_vals = np.array([critical_times[i] for i in seeds])

# ---------------------------
# PLOTS
# ---------------------------

plt.figure(figsize=(15,4))

# Degree vs time
plt.subplot(1,3,1)
plt.scatter(deg_vals, crit_vals)
plt.xlabel("Degree of seed node")
plt.ylabel("Critical time to 70% infection")
plt.title("Degree vs Critical Time")

# Farness vs time
plt.subplot(1,3,2)
plt.scatter(far_vals, crit_vals)
plt.xlabel("Farness of seed node")
plt.ylabel("Critical time to 70% infection")
plt.title("Farness vs Critical Time")

# Degree vs farness
plt.subplot(1,3,3)
plt.scatter(deg_vals, far_vals)
plt.xlabel("Degree of seed node")
plt.ylabel("Farness of seed node")
plt.title("Degree vs Farness")

plt.tight_layout()
plt.show()

# ---------------------------
# Print correlation values
# ---------------------------
corr_deg_time = np.corrcoef(deg_vals, crit_vals)[0,1]
corr_far_time = np.corrcoef(far_vals, crit_vals)[0,1]
corr_deg_far  = np.corrcoef(deg_vals, far_vals)[0,1]

print("\nCorrelation results:")
print(f"Correlation: degree vs critical time = {corr_deg_time:.3f}")
print(f"Correlation: farness vs critical time = {corr_far_time:.3f}")
print(f"Correlation: degree vs farness = {corr_deg_far:.3f}")
