import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# -----------------------------
# SI Gillespie half-life
# -----------------------------
def si_half_life_gillespie(G, beta=1.0, initial_infected=1, rng=None):
    """
    Continuous-time SI via Gillespie.
    Infection along each S-I edge at rate beta.
    Returns t_half: first time I(t) >= ceil(N/2). Returns inf if spread gets stuck.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = G.number_of_nodes()
    target = (n + 1) // 2

    infected = np.zeros(n, dtype=bool)
    seed = rng.choice(n, size=initial_infected, replace=False)
    infected[seed] = True
    I = infected.sum()
    if I >= target:
        return 0.0

    # neighbour lists (fast)
    nbrs = [list(G.neighbors(i)) for i in range(n)]

    t = 0.0
    while I < target:
        # build list of infected nodes with count of susceptible neighbours
        inf_nodes = np.flatnonzero(infected)

        total_SI = 0
        blocks = []  # (u, sus_list, count)
        for u in inf_nodes:
            sus_list = [v for v in nbrs[u] if not infected[v]]
            c = len(sus_list)
            if c:
                blocks.append((u, sus_list, c))
                total_SI += c

        if total_SI == 0:
            return np.inf

        rate = beta * total_SI
        t += rng.exponential(1.0 / rate)

        # choose which S-I edge transmits: pick block proportional to c
        r = rng.integers(total_SI)
        cum = 0
        chosen_sus_list = None
        for _, sus_list, c in blocks:
            if cum + c > r:
                chosen_sus_list = sus_list
                break
            cum += c

        v = rng.choice(chosen_sus_list)
        infected[v] = True
        I += 1

    return t


def mean_degree(G):
    return 2.0 * G.number_of_edges() / G.number_of_nodes()


# -----------------------------
# Graph generators
# -----------------------------
def make_ba(n, target_k, rng):
    """
    BA: <k> ~ 2m => pick m = round(target_k/2), at least 1.
    """
    m = max(1, int(round(target_k / 2)))
    return nx.barabasi_albert_graph(n=n, m=m, seed=int(rng.integers(1_000_000_000)))


def make_er(n, target_k, rng):
    """
    ER G(n,p): <k> ~ p*(n-1) => p = target_k/(n-1)
    """
    p = min(1.0, max(0.0, target_k / (n - 1)))
    return nx.erdos_renyi_graph(n=n, p=p, seed=int(rng.integers(1_000_000_000)))


def make_geometric(n, target_k, rng, dim=2):
    """
    Random geometric graph in unit square/cube.
    For n large: <k> ~ n * V_d(r)  => r ~ (target_k/(n*V_d(1)))^(1/d)
    where V_d(r) = c_d r^d, c_2=pi, c_3=4pi/3.
    This is approximate (boundary effects), but close enough for comparisons.
    """
    if dim == 2:
        c_d = np.pi
    elif dim == 3:
        c_d = 4.0 * np.pi / 3.0
    else:
        raise ValueError("dim must be 2 or 3")

    r = (target_k / (n * c_d)) ** (1.0 / dim)
    # cap radius so it stays sensible
    r = float(min(max(r, 1e-6), 1.0))
    pos = {i: rng.random(dim) for i in range(n)}
    return nx.random_geometric_graph(n=n, radius=r, pos=pos)


# -----------------------------
# Experiment runner
# -----------------------------
def half_life_curve(
    make_graph_fn,
    k_values,
    n=200,
    beta=1.0,
    trials=20,
    initial_infected=1,
    seed=0,
):
    rng = np.random.default_rng(seed)
    ks_out, t_mean, t_std = [], [], []

    for k_target in k_values:
        half_lives = []
        # generate a fresh network each trial (includes network + process randomness)
        for _ in range(trials):
            G = make_graph_fn(n, k_target, rng)
            half_lives.append(
                si_half_life_gillespie(G, beta=beta, initial_infected=initial_infected, rng=rng)
            )

        half_lives = np.array(half_lives, float)
        finite = np.isfinite(half_lives)
        half_lives = half_lives[finite]

        # actual <k> realised (take from last G is crude, so recompute by regenerating once)
        # better: estimate <k> by averaging across trials via quick regen:
        k_real = []
        for _ in range(5):
            Gtmp = make_graph_fn(n, k_target, rng)
            k_real.append(mean_degree(Gtmp))
        ks_out.append(np.mean(k_real))

        t_mean.append(np.mean(half_lives))
        t_std.append(np.std(half_lives, ddof=1) if len(half_lives) > 1 else 0.0)

    return np.array(ks_out), np.array(t_mean), np.array(t_std)


def fit_power_law(k, t, k_min=None, k_max=None):
    """
    Fit t = A * k^{-alpha} by linear regression on logs.
    Returns alpha, A.
    """
    mask = np.isfinite(k) & np.isfinite(t) & (k > 0) & (t > 0)
    if k_min is not None:
        mask &= (k >= k_min)
    if k_max is not None:
        mask &= (k <= k_max)

    kf = k[mask]
    tf = t[mask]
    x = np.log(kf)
    y = np.log(tf)

    # y = b + m x  where m = -alpha, b = log A
    m, b = np.polyfit(x, y, 1)
    alpha = -m
    A = np.exp(b)
    return alpha, A, mask


# -----------------------------
# Main: compare BA vs ER vs geometric
# -----------------------------
if __name__ == "__main__":
    n = 200
    beta = 1.0
    trials = 20
    initial_infected = 1
    seed = 42

    # choose target mean degrees (avoid too close to 1 where geometry may disconnect)
    k_targets = np.array([2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30], dtype=float)

    # run curves
    k_ba, t_ba, s_ba = half_life_curve(make_ba, k_targets, n=n, beta=beta, trials=trials,
                                      initial_infected=initial_infected, seed=seed)
    k_er, t_er, s_er = half_life_curve(make_er, k_targets, n=n, beta=beta, trials=trials,
                                      initial_infected=initial_infected, seed=seed + 1)
    k_g2, t_g2, s_g2 = half_life_curve(lambda n, k, rng: make_geometric(n, k, rng, dim=2),
                                      k_targets, n=n, beta=beta, trials=trials,
                                      initial_infected=initial_infected, seed=seed + 2)

    # pick a fitting window where your plot looks roughly straight on log-log
    # (you can change these)
    fit_k_min, fit_k_max = 4, 20

    alpha_ba, A_ba, _ = fit_power_law(k_ba, t_ba, k_min=fit_k_min, k_max=fit_k_max)
    alpha_er, A_er, _ = fit_power_law(k_er, t_er, k_min=fit_k_min, k_max=fit_k_max)
    alpha_g2, A_g2, _ = fit_power_law(k_g2, t_g2, k_min=fit_k_min, k_max=fit_k_max)

    print("Power-law fits on log-log (t = A * k^{-alpha})")
    print(f"  Fit window: k in [{fit_k_min}, {fit_k_max}]")
    print(f"  BA:  alpha = {alpha_ba:.3f}, A = {A_ba:.3f}")
    print(f"  ER:  alpha = {alpha_er:.3f}, A = {A_er:.3f}")
    print(f"  Geo2D: alpha = {alpha_g2:.3f}, A = {A_g2:.3f}")

    # ---- Plot 1: linear axes with error bars ----
    plt.figure()
    plt.errorbar(k_ba, t_ba, yerr=s_ba, fmt="o-", capsize=3, label="BA")
    plt.errorbar(k_er, t_er, yerr=s_er, fmt="s-", capsize=3, label="ER")
    plt.errorbar(k_g2, t_g2, yerr=s_g2, fmt="^-", capsize=3, label="Geometric (2D)")

    plt.xlabel(r"Average degree $\langle k \rangle$")
    plt.ylabel(r"Half-life $t_{1/2}$")
    plt.title(f"SI half-life vs mean degree (n={n}, beta={beta}, trials={trials})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # ---- Plot 2: log-log + fitted lines ----
    plt.figure()
    plt.loglog(k_ba, t_ba, "o-", label=f"BA (alpha~{alpha_ba:.2f})")
    plt.loglog(k_er, t_er, "s-", label=f"ER (alpha~{alpha_er:.2f})")
    plt.loglog(k_g2, t_g2, "^-", label=f"Geo2D (alpha~{alpha_g2:.2f})")

    # overlay fit curves (same window)
    k_fit = np.linspace(fit_k_min, fit_k_max, 200)
    plt.loglog(k_fit, A_ba * k_fit ** (-alpha_ba), "--")
    plt.loglog(k_fit, A_er * k_fit ** (-alpha_er), "--")
    plt.loglog(k_fit, A_g2 * k_fit ** (-alpha_g2), "--")

    plt.xlabel(r"Average degree $\langle k \rangle$")
    plt.ylabel(r"Half-life $t_{1/2}$")
    plt.title("Log–log scaling and power-law fit window")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.show()
