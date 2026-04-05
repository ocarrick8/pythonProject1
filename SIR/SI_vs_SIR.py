import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Gillespie simulators (counts)
# -----------------------------
def si_gillespie_on_network(G, beta, i0=None, rng=None, t_max=np.inf):
    """
    Continuous-time SI on a fixed network (Gillespie).
    Infection along each S-I edge at rate beta. No recovery.
    Returns: times, I_series (counts)
    """
    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {u: i for i, u in enumerate(nodes)}
    idx_to_node = {i: u for u, i in node_to_idx.items()}

    # state: 0=S, 1=I
    state = np.zeros(n, dtype=np.int8)

    if i0 is None:
        i0 = idx_to_node[int(gen.integers(0, n))]
    i0i = node_to_idx[i0]
    state[i0i] = 1

    # infected-neighbour counts for susceptibles
    inf_nbr_count = np.zeros(n, dtype=np.int32)
    for nbr in G.neighbors(i0):
        j = node_to_idx[nbr]
        inf_nbr_count[j] += 1

    # total S-I edges
    E_SI = int(inf_nbr_count[state == 0].sum())

    t = 0.0
    I = 1
    times = [t]
    I_series = [I]

    while E_SI > 0 and I < n and t < t_max:
        rate_tot = beta * E_SI
        if rate_tot <= 0:
            break

        t += float(gen.exponential(1.0 / rate_tot))
        if t > t_max:
            break

        # choose susceptible to infect with prob ∝ (# infected neighbours)
        sus_idx = np.where(state == 0)[0]
        weights = inf_nbr_count[sus_idx].astype(float)
        wsum = weights.sum()
        if wsum <= 0:
            break

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

        # infect
        state[chosen] = 1
        I += 1

        # update E_SI:
        E_SI -= int(inf_nbr_count[chosen])  # S-I -> I-I
        u = idx_to_node[chosen]
        for nbr in G.neighbors(u):
            j = node_to_idx[nbr]
            if state[j] == 0:
                inf_nbr_count[j] += 1
                E_SI += 1

        times.append(t)
        I_series.append(I)

    return np.array(times), np.array(I_series)


def sir_gillespie_on_network(G, beta, gamma, i0=None, rng=None, t_max=np.inf):
    """
    Continuous-time SIR on a fixed network (Gillespie).
    Infection along each S-I edge at rate beta. Recovery of each infected at rate gamma.
    Returns: times, I_series (counts)
    """
    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {u: i for i, u in enumerate(nodes)}
    idx_to_node = {i: u for u, i in node_to_idx.items()}

    # state: 0=S, 1=I, 2=R
    state = np.zeros(n, dtype=np.int8)

    if i0 is None:
        i0 = idx_to_node[int(gen.integers(0, n))]
    i0i = node_to_idx[i0]
    state[i0i] = 1
    infected = {i0i}

    # infected-neighbour counts for susceptibles
    inf_nbr_count = np.zeros(n, dtype=np.int32)
    for nbr in G.neighbors(i0):
        j = node_to_idx[nbr]
        inf_nbr_count[j] += 1

    # total S-I edges
    E_SI = int(inf_nbr_count[state == 0].sum())

    t = 0.0
    I = 1
    times = [t]
    I_series = [I]

    while infected and t < t_max:
        rate_inf = beta * E_SI
        rate_rec = gamma * len(infected)
        rate_tot = rate_inf + rate_rec
        if rate_tot <= 0:
            break

        t += float(gen.exponential(1.0 / rate_tot))
        if t > t_max:
            break

        if gen.random() < rate_inf / rate_tot:
            # infection
            sus_idx = np.where(state == 0)[0]
            weights = inf_nbr_count[sus_idx].astype(float)
            wsum = weights.sum()
            if wsum <= 0:
                continue

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

            state[chosen] = 1
            infected.add(chosen)
            I += 1

            E_SI -= int(inf_nbr_count[chosen])  # S-I -> I-I

            u = idx_to_node[chosen]
            for nbr in G.neighbors(u):
                j = node_to_idx[nbr]
                if state[j] == 0:
                    inf_nbr_count[j] += 1
                    E_SI += 1

        else:
            # recovery
            inf_list = tuple(infected)
            chosen = inf_list[int(gen.integers(0, len(inf_list)))]
            state[chosen] = 2
            infected.remove(chosen)
            I -= 1

            u = idx_to_node[chosen]
            removed_edges = 0
            for nbr in G.neighbors(u):
                j = node_to_idx[nbr]
                if state[j] == 0:
                    inf_nbr_count[j] -= 1
                    removed_edges += 1
            E_SI -= removed_edges

        times.append(t)
        I_series.append(I)

    return np.array(times), np.array(I_series)


# -----------------------------
# Averaging helpers
# -----------------------------
def value_at_times_post(times, values, grid):
    """Piecewise-constant (post) evaluation of a Gillespie path at grid times."""
    idx = np.searchsorted(times, grid, side="right") - 1
    idx = np.clip(idx, 0, len(values) - 1)
    return values[idx]


def average_I_trajectory(sim_fn, repeats, t_end, grid_n, rng_master, *sim_args, **sim_kwargs):
    """
    Run 'repeats' simulations, evaluate I(t) on a common grid [0,t_end], return mean.
    If a trajectory ends early, it is held constant at its final value (post-hold).
    """
    grid = np.linspace(0.0, float(t_end), int(grid_n))
    Ys = []
    for _ in range(int(repeats)):
        rng = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))
        times, I = sim_fn(*sim_args, rng=rng, **sim_kwargs)
        Ys.append(value_at_times_post(times, I, grid))
    return grid, np.mean(np.vstack(Ys), axis=0)


def mean_sir_extinction_time(G, beta, gamma, repeats, rng_master, i0=None):
    """
    Mean time until I(t)=0 for SIR (i.e. simulation end when infected set empties).
    """
    t_exts = []
    for _ in range(int(repeats)):
        rng = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))
        t, I = sir_gillespie_on_network(G, beta=beta, gamma=gamma, i0=i0, rng=rng, t_max=np.inf)
        t_exts.append(float(t[-1]))  # when infection dies out
    return float(np.mean(t_exts))


# -----------------------------
# Main comparison plot
# -----------------------------
def compare_si_vs_sir_I_longer(
    n=1500,
    p=0.01,
    betas_sir=(0.15, 0.25, 0.4, 0.6),
    gamma=0.2,
    repeats=3,
    si_beta_multipliers=(0.5, 1.0, 1.5),
    include_beta_minus_gamma=True,
    grid_n=800,
    seed_graph=1,
    seed_rng=7,
    share_seed_node=True,
):
    """
    For each beta in betas_sir (one subplot each):
      - pick a time horizon t_end = mean SIR extinction time (I -> 0) over 'repeats'
      - plot mean I(t)/N for SIR with (beta=beta_sir, gamma)
      - also plot mean I(t)/N for SI for a range of betas (beta_sir*multipliers + optionally beta_sir-gamma)
      - make the SIR curve visually stand out (thicker, higher zorder)
    """
    G = nx.erdos_renyi_graph(n, p, seed=seed_graph)
    rng_master = np.random.default_rng(seed_rng)

    #i0 = int(rng_master.integers(0, n)) if share_seed_node else None
    i0 = 20
    fig, axes = plt.subplots(len(betas_sir), 1, figsize=(9, 3.2 * len(betas_sir)), sharex=False)
    if len(betas_sir) == 1:
        axes = [axes]

    for ax, beta_sir in zip(axes, betas_sir):
        # --- time horizon: until SIR dies out (mean over repeats) ---
        t_end = mean_sir_extinction_time(G, beta=beta_sir, gamma=gamma, repeats=repeats, rng_master=rng_master, i0=i0)

        # --- SI beta range for this subplot ---
        betas_si = [float(beta_sir) * float(m) for m in si_beta_multipliers]
        if include_beta_minus_gamma:
            bmg = float(beta_sir) - float(gamma)
            if bmg > 0:
                betas_si.append(bmg)
        betas_si = sorted(set(betas_si))

        # --- Plot SI curves first (thin + faint) ---
        for beta_si in betas_si:
            grid, I_mean = average_I_trajectory(
                si_gillespie_on_network,
                repeats=repeats,
                t_end=t_end,
                grid_n=grid_n,
                rng_master=rng_master,
                G=G,
                beta=beta_si,
                i0=i0,
                t_max=np.inf,
            )
            label = f"SI: β={beta_si:.3g}"
            if include_beta_minus_gamma and abs(beta_si - (beta_sir - gamma)) < 1e-12:
                label += " (=β_SIR−γ)"
            ax.plot(grid, I_mean / n, label=label, linewidth=1.0, alpha=0.5, zorder=1)

        # --- Plot SIR curve last (thicker + on top) ---
        grid, I_mean = average_I_trajectory(
            sir_gillespie_on_network,
            repeats=repeats,
            t_end=t_end,
            grid_n=grid_n,
            rng_master=rng_master,
            G=G,
            beta=beta_sir,
            gamma=gamma,
            i0=i0,
            t_max=np.inf,
        )
        ax.plot(
            grid, I_mean / n,
            label=f"SIR: β={beta_sir:.3g}, γ={gamma:.3g}",
            linewidth=3.0,
            linestyle="-",
            zorder=5
        )

        ax.set_xlim(0, t_end)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("I(t) / N")
        ax.set_title(f"ER(n={n}, p={p}), (β={beta_sir:.3g}, γ={gamma:.3g})")
        ax.legend(ncols=2)

    axes[-1].set_xlabel("time")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_si_vs_sir_I_longer(
        n= 300,
        p=0.02,
        betas_sir=(0.15, 0.2, 0.3),
        gamma=0.1,
        repeats=10,
        si_beta_multipliers=(1.0, 1.5),
        include_beta_minus_gamma=True,
        grid_n=900,
        seed_graph=1,
        seed_rng=7,
        share_seed_node=True,
    )
