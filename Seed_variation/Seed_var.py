import networkx as nx
import numpy as np
import pandas as pd
import random

# Parameters
Ns = [20, 40, 60, 80, 100]             # Network sizes
ps = [0.05, 0.1, 0.15, 0.2, 0.25]      # ER edge probabilities
infection_prob = 0.8
critical_fraction = 0.7
num_repeats = 50  # Repeats per seed node

# Function: SI infection until critical fraction
def run_infection(G, seed, beta=0.3, critical_frac=0.5):
    infected = set([seed])
    t = 0
    while len(infected) < critical_frac * G.number_of_nodes():
        new_infected = set()
        for node in infected:
            for nbr in G.neighbors(node):
                if nbr not in infected and random.random() < beta:
                    new_infected.add(nbr)
        infected.update(new_infected)
        t += 1
        if not new_infected:  # Infection stalled
            break
    return t

# Prepare matrix to store variance of critical times
variance_matrix = pd.DataFrame(index=Ns, columns=ps)

# Main loop over N and p
for N in Ns:
    for p in ps:
        # Generate one ER network
        G = nx.erdos_renyi_graph(N, p)
        critical_times = []
        for seed in G.nodes():
            times = []
            for _ in range(num_repeats):
                T = run_infection(G, seed, infection_prob, critical_fraction)
                times.append(T)
            avg_time = np.mean(times)
            critical_times.append(avg_time)
        critical_times = np.array(critical_times)
        variance_matrix.at[N, p] = np.var(critical_times, ddof=1)

print("Variance of critical times matrix (rows=N, columns=p):")
print(variance_matrix)


