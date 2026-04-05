#!/usr/bin/env python3
"""
Compare between-seed variability of spreading time for ER vs BA graphs.

- For each node we run `REPEATS_PER_NODE` independent Gillespie SI simulations
  and compute mean_time_to_fraction (per-node).
- We then compare half_life_distributions across nodes for ER and BA, compute summary
  stats and tests, and correlate per-node mean time with structural features.
"""

import math, random, time
from collections import defaultdict
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Optional imports (if available)
try:
    from community import community_louvain  # python-louvain
except Exception:
    community_louvain = None

try:
    from scipy.stats import ks_2samp, mannwhitneyu, spearmanr
except Exception:
    ks_2samp = mannwhitneyu = spearmanr = None

# -------------------------
# PARAMETERS (change me)
# -------------------------
N = 100               # nodes
p = 0.05              # ER parameter (so avg deg ~ p*(N-1))
m_param = 2           # BA attachment parameter (avg deg ~ 2*m)
TARGET_FRACTION = 0.7
REPEATS_PER_NODE = 30  # repeats to average per node (balances cost vs noise)
infection_rate = 1.0
RANDOM_SEED = 0
# -------------------------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def simulate_gillespie_time(G, seed_node, target_fraction=TARGET_FRACTION, beta=infection_rate, rng=None):
    """Return time to reach target_fraction infected starting from seed_node.
    If the infection cannot reach target_fraction (graph disconnected), returns np.inf.
    Continuous-time Gillespie SI with per-edge rate beta."""
    if rng is None:
        rng = random
    Nn = G.number_of_nodes()
    target_count = math.ceil(target_fraction * Nn)
    infected = set([seed_node])
    susceptible = set(G.nodes()) - infected
    t = 0.0

    # Precompute neighbors lists for speed
    nbrs = {n: list(G.neighbors(n)) for n in G.nodes()}

    while len(infected) < target_count:
        # active directed edges infected -> susceptible
        active = []
        for i in infected:
            for j in nbrs[i]:
                if j in susceptible:
                    active.append((i, j))
        if not active:
            return np.inf
        rate_total = beta * len(active)
        # exponential waiting time
        u = rng.random() if hasattr(rng, "random") else random.random()
        dt = -math.log(u) / rate_total
        t += dt
        # choose one active edge uniformly
        _, j = active[rng.randrange(len(active))]
        infected.add(j)
        susceptible.remove(j)
    return t

def per_node_mean_times(G, repeats=REPEATS_PER_NODE, target_fraction=TARGET_FRACTION, beta=infection_rate):
    """For each node run 'repeats' simulations and return arrays:
       nodes, mean_times, std_times, fail_counts (infs)."""
    nodes = list(G.nodes())
    mean_times = np.zeros(len(nodes))
    std_times = np.zeros(len(nodes))
    fail_counts = np.zeros(len(nodes), dtype=int)

    for idx, node in enumerate(nodes):
        times = []
        # use independent RNG per run for reproducibility
        for r in range(repeats):
            # use numpy RNG for speed but python RNG ok too
            t = simulate_gillespie_time(G, node, target_fraction=target_fraction, beta=beta, rng=random)
            times.append(t)
        arr = np.array(times, dtype=float)
        fail_counts[idx] = np.isinf(arr).sum()
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            mean_times[idx] = finite.mean()
            std_times[idx] = finite.std(ddof=1) if finite.size > 1 else 0.0
        else:
            mean_times[idx] = np.inf
            std_times[idx] = np.nan
    return np.array(nodes), mean_times, std_times, fail_counts

# ----- structural features -----
def node_features(G):
    deg = dict(G.degree())
    # k-core number
    core = nx.core_number(G)
    # clustering
    clust = nx.clustering(G)
    # betweenness (can be expensive for large graphs; try approximate if needed)
    betw = nx.betweenness_centrality(G, normalized=True)
    # community partition via python-louvain if available, else greedy modularity
    if community_louvain is not None:
        partition = community_louvain.best_partition(G)
    else:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(G))
        partition = {}
        for com_id, com in enumerate(comms):
            for n in com:
                partition[n] = com_id
    # compute participation coefficient and within-module z-score
    # participation p_i = 1 - sum_s (k_i,s / k_i)^2
    # where k_i,s is edges of i to community s
    # within-module z-score: (k_i_in - mean_k_in_s)/std_k_in_s
    # prepare community-members mapping
    comm_nodes = defaultdict(list)
    for n, c in partition.items():
        comm_nodes[c].append(n)
    # degrees within module
    part_coeff = {}
    within_z = {}
    for n in G.nodes():
        k_i = deg[n]
        counts = defaultdict(int)
        for nbr in G.neighbors(n):
            counts[partition[nbr]] += 1
        sum_sq = 0.0
        for s, kis in counts.items():
            frac = kis / k_i if k_i > 0 else 0.0
            sum_sq += frac * frac
        part_coeff[n] = 1.0 - sum_sq
        # within-module z
        com = partition[n]
        kin = counts[com]
        kin_list = [ sum(1 for nbr in G.neighbors(m) if partition[nbr] == com) for m in comm_nodes[com] ]
        if len(kin_list) > 1:
            mean_kin = np.mean(kin_list)
            std_kin = np.std(kin_list, ddof=1)
            within_z[n] = (kin - mean_kin) / std_kin if std_kin > 0 else 0.0
        else:
            within_z[n] = 0.0

    # return features dicts
    feats = {
        "degree": deg,
        "core": core,
        "clustering": clust,
        "betweenness": betw,
        "participation": part_coeff,
        "within_z": within_z,
        "community": partition
    }
    return feats

# ----- summary stats helpers -----
def gini_coefficient(x):
    """Compute Gini coefficient for 1D array x (ignores inf/NaN)."""
    a = np.array(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    a = a.flatten()
    if np.amin(a) < 0:
        a = a - np.amin(a)  # shift to non-negative
    mean = a.mean()
    if mean == 0:
        return 0.0
    n = a.size
    idx = np.arange(1, n+1)
    a_sorted = np.sort(a)
    return ((2.0 * np.sum(idx * a_sorted)) / (n * np.sum(a_sorted)) - (n + 1) / n)

def summary_stats(arr):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {}
    return {
        "n": arr.size,
        "n_finite": finite.size,
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite, ddof=1)),
        "IQR": float(np.percentile(finite, 75) - np.percentile(finite, 25)),
        "gini": float(gini_coefficient(finite)),
        "cv": float(np.std(finite) / np.mean(finite)) if np.mean(finite) != 0 else np.nan
    }

# ----- main experiment -----
def run_experiment(N, p, m_param, repeats_per_node=REPEATS_PER_NODE, target_fraction=TARGET_FRACTION):
    # Build ER and BA graphs with same N
    G_er = nx.erdos_renyi_graph(N, p, seed=42)
    G_ba = nx.barabasi_albert_graph(N, m_param, seed=42)

    print("Computing per-node mean times for ER...")
    nodes_er, mean_er, std_er, fail_er = per_node_mean_times(G_er, repeats=repeats_per_node, target_fraction=target_fraction)
    print("Computing per-node mean times for BA...")
    nodes_ba, mean_ba, std_ba, fail_ba = per_node_mean_times(G_ba, repeats=repeats_per_node, target_fraction=target_fraction)

    # compute structural features
    print("Computing structural features (ER)...")
    feats_er = node_features(G_er)
    print("Computing structural features (BA)...")
    feats_ba = node_features(G_ba)

    # Summaries
    s_er = summary_stats(mean_er)
    s_ba = summary_stats(mean_ba)
    print("\nER per-node mean time summary:", s_er)
    print("BA per-node mean time summary:", s_ba)

    # Statistical comparison (KS / Mann-Whitney)
    if ks_2samp is not None:
        finite_er = mean_er[np.isfinite(mean_er)]
        finite_ba = mean_ba[np.isfinite(mean_ba)]
        try:
            ks_res = ks_2samp(finite_er, finite_ba)
            mw_res = mannwhitneyu(finite_er, finite_ba, alternative="two-sided")
            print("\nKS two-sample test ER vs BA: stat=%.4f p=%.4g" % (ks_res.statistic, ks_res.pvalue))
            print("Mann-Whitney U ER vs BA: U=%.4f p=%.4g" % (mw_res.statistic, mw_res.pvalue))
        except Exception as e:
            print("Stat tests failed:", e)
    else:
        print("\nscipy not available — skipping KS / Mann-Whitney tests")

    # Plot half_life_distributions: hist/CDF/box
    finite_er = mean_er[np.isfinite(mean_er)]
    finite_ba = mean_ba[np.isfinite(mean_ba)]

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.hist(finite_er, bins=30, alpha=0.7, label='ER')
    plt.hist(finite_ba, bins=30, alpha=0.7, label='BA')
    plt.xlabel("Per-node mean time")
    plt.legend()
    plt.title("Histogram of per-node mean times")

    plt.subplot(1,3,2)
    # CDF
    for arr, lab in [(finite_er, "ER"), (finite_ba, "BA")]:
        xs = np.sort(arr)
        ys = np.arange(1, xs.size+1) / xs.size
        plt.step(xs, ys, where='post', label=lab)
    plt.xlabel("Per-node mean time")
    plt.ylabel("CDF")
    plt.legend()
    plt.title("CDF of per-node mean times")

    plt.subplot(1,3,3)
    plt.boxplot([finite_er, finite_ba], labels=["ER", "BA"])
    plt.title("Boxplots")
    plt.tight_layout()
    plt.show()

    # Correlate per-node mean times with features (show Pearson and Spearman if available)
    def correlate_feature(arr_mean, feats_dict, feat_name):
        vec = np.array([feats_dict[feat_name][n] for n in range(len(arr_mean))], dtype=float)
        mask = np.isfinite(arr_mean) & np.isfinite(vec)
        if mask.sum() < 3:
            return np.nan, np.nan
        pear = np.corrcoef(arr_mean[mask], vec[mask])[0,1]
        if spearmanr is not None:
            sp = spearmanr(arr_mean[mask], vec[mask]).correlation
        else:
            sp = np.nan
        return pear, sp

    feat_list = ["degree", "core", "clustering", "betweenness", "participation", "within_z"]
    print("\nCorrelations (Pearson, Spearman if available) between per-node mean time and structural features")
    print("ER:")
    for f in feat_list:
        p_pear, p_spe = correlate_feature(mean_er, feats_er, f)
        print(f"  {f:12s}: pear={p_pear:6.3f}, spearman={p_spe if not np.isnan(p_spe) else 'NA'}")
    print("BA:")
    for f in feat_list:
        p_pear, p_spe = correlate_feature(mean_ba, feats_ba, f)
        print(f"  {f:12s}: pear={p_pear:6.3f}, spearman={p_spe if not np.isnan(p_spe) else 'NA'}")

    # Output per-node table (small sample) for inspection
    import pandas as pd
    df_er = pd.DataFrame({
        "node": nodes_er,
        "mean_time": mean_er,
        "std_time": std_er,
        "fail_count": fail_er,
        "degree": [feats_er["degree"][n] for n in nodes_er],
        "core": [feats_er["core"][n] for n in nodes_er],
        "clustering": [feats_er["clustering"][n] for n in nodes_er],
        "betweenness": [feats_er["betweenness"][n] for n in nodes_er],
        "participation": [feats_er["participation"][n] for n in nodes_er],
        "within_z": [feats_er["within_z"][n] for n in nodes_er],
        "community": [feats_er["community"][n] for n in nodes_er],
    })
    df_ba = pd.DataFrame({
        "node": nodes_ba,
        "mean_time": mean_ba,
        "std_time": std_ba,
        "fail_count": fail_ba,
        "degree": [feats_ba["degree"][n] for n in nodes_ba],
        "core": [feats_ba["core"][n] for n in nodes_ba],
        "clustering": [feats_ba["clustering"][n] for n in nodes_ba],
        "betweenness": [feats_ba["betweenness"][n] for n in nodes_ba],
        "participation": [feats_ba["participation"][n] for n in nodes_ba],
        "within_z": [feats_ba["within_z"][n] for n in nodes_ba],
        "community": [feats_ba["community"][n] for n in nodes_ba],
    })

    print("\nER sample rows (sorted by mean_time):")
    print(df_er.sort_values("mean_time").head(8).to_string(index=False))

    print("\nBA sample rows (sorted by mean_time):")
    print(df_ba.sort_values("mean_time").head(8).to_string(index=False))

    return {
        "G_er": G_er, "G_ba": G_ba,
        "df_er": df_er, "df_ba": df_ba,
        "summary_er": s_er, "summary_ba": s_ba
    }

if __name__ == "__main__":
    out = run_experiment(N=N, p=p, m_param=m_param, repeats_per_node=REPEATS_PER_NODE)
