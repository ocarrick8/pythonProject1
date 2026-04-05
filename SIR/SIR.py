import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def sir_gillespie_on_network(G, beta, gamma, i0=None, rng=None, t_max=np.inf):
    """
    Continuous-time SIR on a fixed network using a Gillespie (SSA) algorithm.

    Model:
      - Along each S-I edge: infection at rate beta
      - Each infected node: recovery at rate gamma

    Args:
        G: networkx.Graph (assumed undirected, unweighted)
        beta: float, infection rate per S-I edge
        gamma: float, recovery rate per infected node
        i0: optional int, initial infected node (random if None)
        rng: optional int or np.random.Generator, randomness control
        t_max: float, optional maximum simulation time

    Returns:
        times, S_series, I_series, R_series
    """
    # RNG setup
    if isinstance(rng, np.random.Generator):
        gen = rng
    else:
        gen = np.random.default_rng(rng)

    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {u: i for i, u in enumerate(nodes)}
    idx_to_node = {i: u for u, i in node_to_idx.items()}

    # States: 0=S, 1=I, 2=R (store by node index 0..n-1)
    state = np.zeros(n, dtype=np.int8)

    # Initial infected
    if i0 is None:
        i0 = idx_to_node[int(gen.integers(0, n))]
    i0i = node_to_idx[i0]
    state[i0i] = 1

    infected = {i0i}
    recovered = set()

    # Track: for each susceptible node, how many infected neighbours it currently has
    inf_nbr_count = np.zeros(n, dtype=np.int32)
    for nbr in G.neighbors(i0):
        j = node_to_idx[nbr]
        inf_nbr_count[j] += 1

    # Current total number of S-I edges = sum_{s in S} (infected neighbours of s)
    # (each S-I edge counted once)
    E_SI = int(inf_nbr_count[state == 0].sum())

    # Time series
    t = 0.0
    S = int(np.sum(state == 0))
    I = int(np.sum(state == 1))
    R = int(np.sum(state == 2))

    times = [t]
    S_series = [S]
    I_series = [I]
    R_series = [R]

    while infected and t < t_max:
        rate_inf = beta * E_SI
        rate_rec = gamma * len(infected)
        rate_tot = rate_inf + rate_rec

        if rate_tot <= 0:
            break

        # Next event time
        t += float(gen.exponential(1.0 / rate_tot))
        if t > t_max:
            break

        # Choose event type
        if gen.random() < rate_inf / rate_tot:
            # ----------------------------
            # Infection event: choose a susceptible node with prob ∝ (# infected neighbours)
            # ----------------------------
            sus_idx = np.where(state == 0)[0]
            weights = inf_nbr_count[sus_idx].astype(float)
            wsum = weights.sum()
            if wsum <= 0:
                # Shouldn't happen if E_SI>0, but safe-guard.
                continue

            # Sample susceptible proportional to weights
            r = gen.random() * wsum
            cum = 0.0
            chosen = None
            for s_i, w in zip(sus_idx, weights):
                cum += w
                if cum >= r:
                    chosen = int(s_i)
                    break
            if chosen is None:
                chosen = int(sus_idx[-1])

            # Infect chosen
            state[chosen] = 1
            infected.add(chosen)
            S -= 1
            I += 1

            # Update E_SI:
            # - The newly infected node was susceptible with inf_nbr_count[chosen] infected neighbours,
            #   so those S-I edges become I-I => remove that many from E_SI
            E_SI -= int(inf_nbr_count[chosen])

            # - For each susceptible neighbour of the newly infected node, we add 1 new infected neighbour,
            #   which creates one new S-I edge (for each such neighbour).
            u = idx_to_node[chosen]
            for nbr in G.neighbors(u):
                j = node_to_idx[nbr]
                if state[j] == 0:
                    inf_nbr_count[j] += 1
                    E_SI += 1

            # (Recovered neighbours don't matter; infected neighbours don't change E_SI.)
        else:
            # ----------------------------
            # Recovery event: choose an infected node uniformly at random
            # ----------------------------
            inf_list = tuple(infected)
            k = int(gen.integers(0, len(inf_list)))
            chosen = inf_list[k]

            state[chosen] = 2
            infected.remove(chosen)
            recovered.add(chosen)
            I -= 1
            R += 1

            # Update E_SI:
            # When an infected node recovers, all S-I edges incident to it disappear.
            # That equals the number of susceptible neighbours it currently has.
            u = idx_to_node[chosen]
            removed_edges = 0
            for nbr in G.neighbors(u):
                j = node_to_idx[nbr]
                if state[j] == 0:
                    # This neighbour loses one infected neighbour
                    inf_nbr_count[j] -= 1
                    removed_edges += 1
            E_SI -= removed_edges

        times.append(t)
        S_series.append(S)
        I_series.append(I)
        R_series.append(R)

    return np.array(times), np.array(S_series), np.array(I_series), np.array(R_series)


def demo_plot_er_sir(n=1000, p=0.01, beta=0.6, gamma=0.2, seed_graph=0, seed_rng=1):
    """
    Build an ER network and plot S, I, R over time from one Gillespie run.
    """
    G = nx.erdos_renyi_graph(n, p, seed=seed_graph)
    times, S, I, R = sir_gillespie_on_network(G, beta=beta, gamma=gamma, rng=seed_rng)

    plt.figure(figsize=(9, 5))
    plt.plot(times, S, label="S(t)")
    plt.plot(times, I, label="I(t)")
    plt.plot(times, R, label="R(t)")
    plt.xlabel("time")
    plt.ylabel("number of nodes")
    plt.title(f"SIR on ER(n={n}, p={p}), Gillespie (beta={beta}, gamma={gamma})")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo_plot_er_sir(
        n=1500,     # try bigger/smaller
        p=0.01,     # ER edge probability
        beta=0.15,   # infection rate per S-I edge
        gamma=0.1,  # recovery rate per infected node
        seed_graph=42,
        seed_rng=123
    )
