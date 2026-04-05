import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# ----------------------------
# Robust Gillespie SI (recompute SI edges each event)
# ----------------------------
def si_half_life_gillespie(G, beta=1.0, rng=None, seed_node=None):
    """
    SI model with Gillespie algorithm.
    Infection happens along SI edges at rate beta per SI edge.

    Returns:
        t_half: time to reach ceil(N/2) infected
    """
    if rng is None:
        rng = np.random.default_rng()

    N = G.number_of_nodes()
    target = (N + 1) // 2  # ceil(N/2)

    if seed_node is None:
        seed_node = int(rng.integers(0, N))

    infected = np.zeros(N, dtype=bool)
    infected[seed_node] = True
    I = 1
    t = 0.0

    while I < target:
        # Build SI edge list
        infected_nodes = np.flatnonzero(infected)
        SI_edges = []
        for v in infected_nodes:
            for u in G.neighbors(v):
                if not infected[u]:
                    SI_edges.append((v, u))

        m_SI = len(SI_edges)
        if m_SI == 0:
            return np.inf

        rate = beta * m_SI
        t += rng.exponential(1.0 / rate)

        # Choose which SI edge transmits uniformly among SI edges
        v, u = SI_edges[int(rng.integers(0, m_SI))]
        infected[u] = True
        I += 1

    return t


# ----------------------------
# Experiment: fixed m, varying N
# ----------------------------
def half_life_vs_N(
    N_values,
    m=3,
    beta=1.0,
    n_networks=5,
    n_runs_per_network=5,
    rng_seed=123,
):
    rng = np.random.default_rng(rng_seed)

    means, stds = [], []
    all_samples = {}

    for N in N_values:
        samples = []

        for _ in range(n_networks):
            G = nx.barabasi_albert_graph(N, m, seed=int(rng.integers(0, 2**31 - 1)))

            for _ in range(n_runs_per_network):
                t_half = si_half_life_gillespie(G, beta=beta, rng=rng)
                samples.append(t_half)

        samples = np.array(samples, dtype=float)
        all_samples[N] = samples

        means.append(np.mean(samples))
        stds.append(np.std(samples, ddof=1) if len(samples) > 1 else 0.0)

        print(f"N={N:5d}  mean t_half={means[-1]:.5g}  std={stds[-1]:.5g}  (n={len(samples)})")

    return np.array(means), np.array(stds), all_samples


# ----------------------------
# Fitting helpers
# ----------------------------
def fit_log_linear(N, t):
    """
    Fit t = a ln N + b (least squares).
    Returns a, b
    """
    x = np.log(N)
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, t, rcond=None)[0]
    return a, b


def fit_power_law(N, t):
    """
    Fit t = c N^alpha via log transform:
        ln t = ln c + alpha ln N
    Returns c, alpha
    """
    x = np.log(N)
    y = np.log(t)
    A = np.vstack([x, np.ones_like(x)]).T
    alpha, ln_c = np.linalg.lstsq(A, y, rcond=None)[0]
    c = np.exp(ln_c)
    return c, alpha


# ----------------------------
# Run + Plot
# ----------------------------
if __name__ == "__main__":
    # Many more N values (log-spaced, unique ints, reasonably bounded)
    N_values = np.unique(np.round(np.logspace(np.log10(80), np.log10(2000), 10)).astype(int))

    # Parameters
    m = 3
    beta = 1.0

    # Repeats: tune these for runtime vs smoothness
    n_networks = 10
    n_runs_per_network = 5

    means, stds, samples = half_life_vs_N(
        N_values=N_values,
        m=m,
        beta=beta,
        n_networks=n_networks,
        n_runs_per_network=n_runs_per_network,
        rng_seed=42,
    )

    # Remove any infs (in case a graph got stuck, rare in BA)
    mask = np.isfinite(means)
    N_fit = N_values[mask]
    t_fit = means[mask]
    s_fit = stds[mask]

    # Fits
    a, b = fit_log_linear(N_fit, t_fit)          # t = a ln N + b
    c, alpha = fit_power_law(N_fit, t_fit)       # t = c N^alpha

    print("\nFITS (using mean t_half values):")
    print(f"  Log-linear: t = a ln N + b   with a={a:.6g}, b={b:.6g}")
    print(f"  Power-law:  t = c N^alpha    with c={c:.6g}, alpha={alpha:.6g}")

    # Smooth curve for fits
    N_smooth = np.linspace(N_fit.min(), N_fit.max(), 400)
    t_loglin = a * np.log(N_smooth) + b
    t_power = c * (N_smooth ** alpha)

    # Plot 1: linear axes (often fine for log-linear expectation)
    plt.figure(figsize=(8, 5))
    plt.errorbar(N_values, means, yerr=stds, fmt="o", capsize=3, label="Sim mean ± std")
    plt.plot(N_smooth, t_loglin, "-", label=f"Fit: a ln N + b (a={a:.3g})")
    plt.plot(N_smooth, t_power, "--", label=f"Fit: c N^α (α={alpha:.3g})")
    plt.xlabel("Network size N")
    plt.ylabel("Half-life time t½ (to reach 50% infected)")
    plt.title(f"SI Gillespie on BA networks (m={m}, beta={beta})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 2: log-x (useful if expecting ~ ln N scaling)
    plt.figure(figsize=(8, 5))
    plt.errorbar(N_values, means, yerr=stds, fmt="o", capsize=3, label="Sim mean ± std")
    plt.plot(N_smooth, t_loglin, "-", label="Log-linear fit")
    plt.xscale("log")
    plt.xlabel("Network size N (log scale)")
    plt.ylabel("Half-life time t½")
    plt.title(f"Half-life vs N (log-x) — BA (m={m})")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 3: log-log (useful if suspecting power-law)
    plt.figure(figsize=(8, 5))
    plt.errorbar(N_values, means, yerr=stds, fmt="o", capsize=3, label="Sim mean ± std")
    plt.plot(N_smooth, t_power, "--", label="Power-law fit")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Network size N (log scale)")
    plt.ylabel("Half-life time t½ (log scale)")
    plt.title(f"Half-life vs N (log-log) — BA (m={m})")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
