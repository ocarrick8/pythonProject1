import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple

# ============================================================
# 1) "Normal BE graph" = Binomial/Erdős–Rényi G(N, p)
#    (often abbreviated as B–E / binomial random graph)
# ============================================================

def generate_BE_graph_ER(N: int, p: float, rng: np.random.Generator) -> nx.Graph:
    """
    Generate a Binomial/Erdős–Rényi random graph G(N, p).

    Parameters
    ----------
    N : int
        Number of nodes.
    p : float
        Edge probability in [0, 1].
    rng : np.random.Generator
        Random generator.

    Returns
    -------
    G : nx.Graph
    """
    if N < 1:
        raise ValueError("N must be >= 1.")
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1].")

    # NetworkX supports a NumPy Generator directly for reproducibility
    G = nx.erdos_renyi_graph(n=N, p=p, seed=rng, directed=False)
    return G


# ============================================================
# 2) Gillespie SI / SIR with first-passage times to thresholds
# ============================================================

S, I, R = 0, 1, 2


def gillespie_first_passage_times(
    G: nx.Graph,
    model: str,
    beta: float,
    gamma: float,
    thresholds: List[float],
    rng: np.random.Generator,
    seed_node: Optional[int] = None,
    t_max: float = np.inf,
) -> Dict[float, float]:
    """
    Run Gillespie (continuous-time) SI or SIR from a single infected seed,
    and record the first time that I/N reaches each threshold (upward crossing).

    For SIR: if the process dies out before reaching a threshold, its time is NaN.

    Parameters
    ----------
    G : nx.Graph
    model : {"SI","SIR"}
    beta : float
        Infection rate per SI edge.
    gamma : float
        Recovery rate per infected (used only for SIR).
    thresholds : list of float
        Fractions of infected I/N to hit (e.g. [0.1, 0.3, 0.5]).
    rng : np.random.Generator
    seed_node : int or None
        Which node is infected at t=0. If None, chosen uniformly at random.
    t_max : float
        Optional safety cutoff time.

    Returns
    -------
    times : dict {threshold: time}
        First passage times. Unreached thresholds are NaN.
    """
    model = model.upper().strip()
    if model not in {"SI", "SIR"}:
        raise ValueError("model must be 'SI' or 'SIR'")
    if beta < 0 or (model == "SIR" and gamma < 0):
        raise ValueError("beta and gamma must be non-negative.")

    N = G.number_of_nodes()
    if N == 0:
        raise ValueError("Graph has no nodes.")

    thr_sorted = sorted(thresholds)
    if any((thr <= 0 or thr >= 1) for thr in thr_sorted):
        raise ValueError("thresholds must be in (0, 1).")

    # states: 0=S, 1=I, 2=R
    state = np.zeros(N, dtype=np.int8)

    if seed_node is None:
        seed_node = int(rng.integers(0, N))
    state[seed_node] = I

    # inf_nbrs[s] = number of infected neighbors of susceptible node s
    inf_nbrs = np.zeros(N, dtype=np.int32)
    for nbr in G.neighbors(seed_node):
        if state[nbr] == S:
            inf_nbrs[nbr] += 1

    infected_nodes = {seed_node}

    # Total number of SI edges = sum_{susceptible s} inf_nbrs[s]
    SI_edge_count = int(inf_nbrs.sum())

    # First-passage recording
    times = {thr: np.nan for thr in thr_sorted}
    next_thr_idx = 0
    t = 0.0

    def maybe_record_times(current_I: int, current_t: float):
        nonlocal next_thr_idx
        frac = current_I / N
        while next_thr_idx < len(thr_sorted) and frac >= thr_sorted[next_thr_idx]:
            thr = thr_sorted[next_thr_idx]
            if np.isnan(times[thr]):
                times[thr] = current_t
            next_thr_idx += 1

    maybe_record_times(len(infected_nodes), t)

    while t < t_max:
        I_count = len(infected_nodes)

        # stop conditions & rates
        if model == "SI":
            if next_thr_idx >= len(thr_sorted):
                break
            if SI_edge_count == 0:
                break
            rate_infect = beta * SI_edge_count
            rate_total = rate_infect
        else:
            if I_count == 0:
                break
            rate_infect = beta * SI_edge_count
            rate_recover = gamma * I_count
            rate_total = rate_infect + rate_recover
            if rate_total <= 0:
                break

        # time step
        dt = rng.exponential(1.0 / rate_total)
        t += dt

        # choose event
        if model == "SI":
            do_infection = True
        else:
            do_infection = (rng.random() < (rate_infect / rate_total))

        if do_infection:
            if SI_edge_count == 0:
                continue

            # choose susceptible node to infect with weight proportional to its infected neighbors count
            candidates = np.flatnonzero((state == S) & (inf_nbrs > 0))
            weights = inf_nbrs[candidates].astype(float)
            wsum = weights.sum()
            if wsum <= 0:
                continue
            probs = weights / wsum
            newly_infected = int(rng.choice(candidates, p=probs))

            # S -> I
            state[newly_infected] = I
            infected_nodes.add(newly_infected)

            # remove this node's contribution from SI_edge_count
            SI_edge_count -= int(inf_nbrs[newly_infected])
            inf_nbrs[newly_infected] = 0

            # its susceptible neighbors gain +1 infected neighbor, hence SI_edge_count +1 each such edge
            for nbr in G.neighbors(newly_infected):
                if state[nbr] == S:
                    inf_nbrs[nbr] += 1
                    SI_edge_count += 1

            maybe_record_times(len(infected_nodes), t)

        else:
            # recovery event: choose infected uniformly (each has rate gamma)
            rec = int(rng.choice(list(infected_nodes)))

            # I -> R
            infected_nodes.remove(rec)
            state[rec] = R

            # susceptible neighbors lose one infected neighbor
            for nbr in G.neighbors(rec):
                if state[nbr] == S:
                    inf_nbrs[nbr] -= 1
                    SI_edge_count -= 1

    return times


# ============================================================
# 3) Run SI vs SIR comparison on an ER (Binomial/BE) graph
# ============================================================

def run_comparison_BE_ER(
    N: int = 3000,
    p_edge: float = 0.004,
    beta: float = 0.7,
    gamma: float = 0.25,
    thresholds: List[float] = [0.1, 0.3, 0.5],
    n_runs: int = 30,
    seed: int = 42,
    reuse_same_graph: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Build a Binomial/Erdős–Rényi graph G(N, p_edge) and estimate first-passage times
    to I/N thresholds for SI and SIR using Gillespie.

    If reuse_same_graph=True:
      - one fixed graph, multiple epidemic runs (different random seeds/seed nodes)
    If False:
      - new ER graph each run (includes graph-to-graph variability)

    Returns a dict with raw arrays so you can plot / do stats.
    """
    rng_master = np.random.default_rng(seed)
    thr_sorted = sorted(thresholds)

    si_times = np.full((n_runs, len(thr_sorted)), np.nan, dtype=float)
    sir_times = np.full((n_runs, len(thr_sorted)), np.nan, dtype=float)

    G_fixed = None
    if reuse_same_graph:
        G_fixed = generate_BE_graph_ER(N=N, p=p_edge, rng=rng_master)

    for r in range(n_runs):
        G = G_fixed if reuse_same_graph else generate_BE_graph_ER(
            N=N, p=p_edge, rng=np.random.default_rng(rng_master.integers(0, 2**63 - 1))
        )

        # SI run
        rng_si = np.random.default_rng(rng_master.integers(0, 2**63 - 1))
        t_si = gillespie_first_passage_times(
            G=G, model="SI", beta=beta, gamma=gamma,
            thresholds=thr_sorted, rng=rng_si, seed_node=None
        )
        for j, thr in enumerate(thr_sorted):
            si_times[r, j] = t_si[thr]

        # SIR run
        rng_sir = np.random.default_rng(rng_master.integers(0, 2**63 - 1))
        t_sir = gillespie_first_passage_times(
            G=G, model="SIR", beta=beta, gamma=gamma,
            thresholds=thr_sorted, rng=rng_sir, seed_node=None
        )
        for j, thr in enumerate(thr_sorted):
            sir_times[r, j] = t_sir[thr]

    # print a clean summary
    def summarize(arr: np.ndarray):
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        reach = np.mean(~np.isnan(arr), axis=0)  # fraction of runs that reached threshold
        return mean, std, reach

    si_mean, si_std, si_reach = summarize(si_times)
    sir_mean, sir_std, sir_reach = summarize(sir_times)

    G_for_stats = G_fixed if reuse_same_graph else None
    if G_for_stats is not None:
        avg_k = 2 * G_for_stats.number_of_edges() / N
        print(f"BE (Binomial/ER) graph: N={N}, p_edge={p_edge}, <k>≈{avg_k:.3f}")
    else:
        print(f"BE (Binomial/ER) graphs: N={N}, p_edge={p_edge} (new graph each run)")

    print(f"Rates: beta={beta}, gamma={gamma} (gamma ignored for SI)")
    print(f"Runs: {n_runs}\n")
    print("Time to reach I/N threshold (first upward crossing):")
    print("Threshold | SI mean ± std (reach%)      | SIR mean ± std (reach%)")
    print("-" * 72)
    for j, thr in enumerate(thr_sorted):
        print(
            f"{thr:8.2f} | "
            f"{si_mean[j]:8.4f} ± {si_std[j]:8.4f} ({100*si_reach[j]:5.1f}%) | "
            f"{sir_mean[j]:8.4f} ± {sir_std[j]:8.4f} ({100*sir_reach[j]:5.1f}%)"
        )

    return {
        "thresholds": np.array(thr_sorted, dtype=float),
        "si_times": si_times,
        "sir_times": sir_times,
    }


if __name__ == "__main__":
    # Example usage:
    out = run_comparison_BE_ER(
        N=600,
        p_edge=0.01,
        beta=0.4,
        gamma=0.1,
        thresholds=[0.05, 0.1, 0.3, 0.5],
        n_runs=60,
        seed=42,
        reuse_same_graph=True,
    )
