"""
Compare dynamic vs static SI spreading using the FULL infection trajectory I(t)/N.

Goal
----
For each network family (ER, BA, 2D geometric):
1. Build one underlying static graph.
2. Simulate a dynamic-edge SI model where each edge switches ON/OFF with rate mu,
   and is ON 50% of the time on average.
3. Simulate static SI on the SAME graph for a dense range of beta_eff values.
4. Compare the mean infection trajectory <I(t)/N>.
5. Choose the beta_eff that best fits the dynamic mean trajectory.
6. Plot each network separately.

Designed to stay computationally manageable:
- only one dynamic regime: mu = beta0
- modest N
- modest repeats
- separate figures only for the infection curves
- no extra objective-vs-beta plots
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# ============================================================
# USER SETTINGS
# ============================================================

beta0 = 1.0
mu = beta0
repeats = 100                 # reduce to 30 if slow
n_time = 120                 # reduce to 100 if slow
beta_grid = np.linspace(0.35, 0.60, 51)   # dense trial beta_eff grid
master_seed = 123456

NETWORK_SPECS = {
    "ER":  {"type": "ER",  "N": 200, "k_target": 5.0},
    "BA":  {"type": "BA",  "N": 200, "m": 3},
    "RGG": {"type": "RGG", "N": 200, "k_target": 5.0},
}

show_std_band = True

# ============================================================
# GRAPH BUILDERS
# ============================================================

def build_connected_er(N, k_target, rng):
    p = k_target / (N - 1)
    for _ in range(200):
        G = nx.erdos_renyi_graph(N, p, seed=int(rng.integers(1, 10**9)))
        if nx.is_connected(G):
            return G
    G = nx.erdos_renyi_graph(N, p, seed=int(rng.integers(1, 10**9)))
    cc = max(nx.connected_components(G), key=len)
    return G.subgraph(cc).copy()

def build_connected_ba(N, m, rng):
    return nx.barabasi_albert_graph(N, m, seed=int(rng.integers(1, 10**9)))

def build_connected_rgg(N, k_target, rng):
    r0 = math.sqrt(k_target / (math.pi * N)) * 1.20
    for factor in [1.00, 1.05, 1.10, 1.15, 1.20, 1.30]:
        r = r0 * factor
        for _ in range(100):
            pos = {i: (rng.random(), rng.random()) for i in range(N)}
            G = nx.random_geometric_graph(N, r, pos=pos)
            if nx.is_connected(G):
                return G
    pos = {i: (rng.random(), rng.random()) for i in range(N)}
    G = nx.random_geometric_graph(N, r0 * 1.4, pos=pos)
    cc = max(nx.connected_components(G), key=len)
    return G.subgraph(cc).copy()

def build_graph(spec, rng):
    if spec["type"] == "ER":
        return build_connected_er(spec["N"], spec["k_target"], rng)
    elif spec["type"] == "BA":
        return build_connected_ba(spec["N"], spec["m"], rng)
    elif spec["type"] == "RGG":
        return build_connected_rgg(spec["N"], spec["k_target"], rng)
    else:
        raise ValueError(f"Unknown graph type {spec['type']}")

# ============================================================
# PREP GRAPH FOR FAST SIMULATION
# ============================================================

def prepare_graph_data(G):
    nodes = list(G.nodes())
    mapping = {node: i for i, node in enumerate(nodes)}
    H = nx.relabel_nodes(G, mapping, copy=True)

    N = H.number_of_nodes()
    edges = list(H.edges())
    M = len(edges)

    u = np.empty(M, dtype=np.int32)
    v = np.empty(M, dtype=np.int32)
    incident_edges = [[] for _ in range(N)]

    for eid, (a, b) in enumerate(edges):
        u[eid] = a
        v[eid] = b
        incident_edges[a].append(eid)
        incident_edges[b].append(eid)

    return {
        "G": H,
        "N": N,
        "M": M,
        "u": u,
        "v": v,
        "incident_edges": incident_edges,
    }

# ============================================================
# UTILITIES
# ============================================================

def sample_trajectory_on_grid(event_times, infected_counts, t_grid, N):
    idx = np.searchsorted(event_times, t_grid, side="right") - 1
    out = np.empty_like(t_grid, dtype=float)

    before_mask = idx < 0
    out[before_mask] = 1.0 / N

    valid_mask = ~before_mask
    out[valid_mask] = infected_counts[idx[valid_mask]] / N
    return out

def first_time_mean_reaches(curve, t_grid, threshold):
    idx = np.where(curve >= threshold)[0]
    if len(idx) == 0:
        return t_grid[-1]
    return t_grid[idx[0]]

# ============================================================
# DYNAMIC SI TRAJECTORY
# ============================================================

def simulate_dynamic_SI_trajectory(graph_data, beta, mu, seed_node, rng):
    N = graph_data["N"]
    M = graph_data["M"]
    u = graph_data["u"]
    v = graph_data["v"]
    incident_edges = graph_data["incident_edges"]

    infected = np.zeros(N, dtype=bool)
    infected[seed_node] = True
    infected_count = 1

    active = rng.random(M) < 0.5

    active_SI = set()
    for eid in range(M):
        a, b = u[eid], v[eid]
        if active[eid] and (infected[a] ^ infected[b]):
            active_SI.add(eid)

    t = 0.0
    switch_rate = mu * M

    event_times = []
    infected_counts = []

    while infected_count < N:
        inf_rate = beta * len(active_SI)
        total_rate = inf_rate + switch_rate

        if total_rate <= 0:
            break

        t += rng.exponential(1.0 / total_rate)

        if rng.random() < inf_rate / total_rate:
            if len(active_SI) == 0:
                continue

            eid = tuple(active_SI)[int(rng.integers(len(active_SI)))]
            a, b = u[eid], v[eid]

            if infected[a] and (not infected[b]):
                new_node = b
            elif infected[b] and (not infected[a]):
                new_node = a
            else:
                continue

            infected[new_node] = True
            infected_count += 1

            event_times.append(t)
            infected_counts.append(infected_count)

            for e2 in incident_edges[new_node]:
                x, y = u[e2], v[e2]
                if active[e2] and (infected[x] ^ infected[y]):
                    active_SI.add(e2)
                else:
                    active_SI.discard(e2)

        else:
            eid = int(rng.integers(M))
            active[eid] = ~active[eid]
            a, b = u[eid], v[eid]
            if active[eid] and (infected[a] ^ infected[b]):
                active_SI.add(eid)
            else:
                active_SI.discard(eid)

    return np.array(event_times, dtype=float), np.array(infected_counts, dtype=np.int32)

# ============================================================
# STATIC SI TRAJECTORY
# ============================================================

def simulate_static_SI_trajectory(graph_data, beta, seed_node, rng):
    N = graph_data["N"]
    u = graph_data["u"]
    v = graph_data["v"]
    incident_edges = graph_data["incident_edges"]

    infected = np.zeros(N, dtype=bool)
    infected[seed_node] = True
    infected_count = 1

    SI_edges = set(incident_edges[seed_node])

    t = 0.0
    event_times = []
    infected_counts = []

    while infected_count < N:
        inf_rate = beta * len(SI_edges)
        if inf_rate <= 0:
            break

        t += rng.exponential(1.0 / inf_rate)

        eid = tuple(SI_edges)[int(rng.integers(len(SI_edges)))]
        a, b = u[eid], v[eid]

        if infected[a] and (not infected[b]):
            new_node = b
        elif infected[b] and (not infected[a]):
            new_node = a
        else:
            SI_edges.discard(eid)
            continue

        infected[new_node] = True
        infected_count += 1

        event_times.append(t)
        infected_counts.append(infected_count)

        for e2 in incident_edges[new_node]:
            x, y = u[e2], v[e2]
            if infected[x] ^ infected[y]:
                SI_edges.add(e2)
            else:
                SI_edges.discard(e2)

    return np.array(event_times, dtype=float), np.array(infected_counts, dtype=np.int32)

# ============================================================
# REPEATED TRAJECTORIES
# ============================================================

def run_dynamic_trajectories(graph_data, beta, mu, repeats, base_seed):
    rng = np.random.default_rng(base_seed)
    N = graph_data["N"]

    seeds = rng.integers(0, N, size=repeats)
    trajectories = []
    final_times = np.empty(repeats)

    for i in range(repeats):
        local_rng = np.random.default_rng(int(rng.integers(1, 10**9)))
        event_times, infected_counts = simulate_dynamic_SI_trajectory(
            graph_data=graph_data,
            beta=beta,
            mu=mu,
            seed_node=int(seeds[i]),
            rng=local_rng
        )
        trajectories.append((event_times, infected_counts))
        final_times[i] = event_times[-1] if len(event_times) > 0 else 0.0

    return seeds, trajectories, final_times

def run_static_trajectories(graph_data, beta, seeds, base_seed):
    rng = np.random.default_rng(base_seed)
    repeats = len(seeds)

    trajectories = []
    final_times = np.empty(repeats)

    for i in range(repeats):
        local_rng = np.random.default_rng(int(rng.integers(1, 10**9)))
        event_times, infected_counts = simulate_static_SI_trajectory(
            graph_data=graph_data,
            beta=beta,
            seed_node=int(seeds[i]),
            rng=local_rng
        )
        trajectories.append((event_times, infected_counts))
        final_times[i] = event_times[-1] if len(event_times) > 0 else 0.0

    return trajectories, final_times

# ============================================================
# EVALUATE TRAJECTORIES ON COMMON GRID
# ============================================================

def trajectories_to_matrix(trajectories, t_grid, N):
    mat = np.empty((len(trajectories), len(t_grid)), dtype=float)
    for i, (event_times, infected_counts) in enumerate(trajectories):
        mat[i] = sample_trajectory_on_grid(event_times, infected_counts, t_grid, N)
    return mat

# ============================================================
# OBJECTIVE FUNCTION
# ============================================================

def trajectory_fit_metrics(dynamic_mean, static_mean):
    diff = static_mean - dynamic_mean
    mse = np.mean(diff**2)
    mae = np.mean(np.abs(diff))
    max_abs = np.max(np.abs(diff))
    return mse, mae, max_abs

# ============================================================
# MAIN EXPERIMENT
# ============================================================

master_rng = np.random.default_rng(master_seed)
results = {}

for family, spec in NETWORK_SPECS.items():
    print(f"\n=== {family} ===")

    G = build_graph(spec, master_rng)
    gd = prepare_graph_data(G)

    N = gd["N"]
    M = gd["M"]
    avg_k = 2 * M / N
    print(f"N={N}, M={M}, <k>={avg_k:.3f}")

    print("Running dynamic trajectories...")
    seeds, dyn_trajs, dyn_final_times = run_dynamic_trajectories(
        graph_data=gd,
        beta=beta0,
        mu=mu,
        repeats=repeats,
        base_seed=int(master_rng.integers(1, 10**9))
    )

    t_max = np.quantile(dyn_final_times, 0.95)
    if t_max <= 0:
        t_max = 1.0
    t_grid = np.linspace(0.0, t_max, n_time)

    dyn_mat = trajectories_to_matrix(dyn_trajs, t_grid, N)
    dyn_mean = dyn_mat.mean(axis=0)
    dyn_std = dyn_mat.std(axis=0)

    t90_mean = first_time_mean_reaches(dyn_mean, t_grid, 0.90)
    fit_mask = t_grid <= t90_mean
    if fit_mask.sum() < 10:
        fit_mask = np.ones_like(t_grid, dtype=bool)

    print("Optimising beta_eff on full I(t)/N trajectory...")
    beta_summaries = []

    for beta_eff in beta_grid:
        print(f"  static beta_eff = {beta_eff:.3f}")

        st_trajs, st_final_times = run_static_trajectories(
            graph_data=gd,
            beta=beta_eff,
            seeds=seeds,
            base_seed=int(master_rng.integers(1, 10**9))
        )

        st_mat = trajectories_to_matrix(st_trajs, t_grid, N)
        st_mean = st_mat.mean(axis=0)
        st_std = st_mat.std(axis=0)

        mse, mae, max_abs = trajectory_fit_metrics(
            dynamic_mean=dyn_mean[fit_mask],
            static_mean=st_mean[fit_mask]
        )

        beta_summaries.append({
            "beta_eff": float(beta_eff),
            "mse": float(mse),
            "mae": float(mae),
            "max_abs": float(max_abs),
            "mean_curve": st_mean,
            "std_curve": st_std,
        })

    best = min(beta_summaries, key=lambda d: d["mse"])

    results[family] = {
        "graph_data": gd,
        "N": N,
        "M": M,
        "avg_k": avg_k,
        "t_grid": t_grid,
        "fit_mask": fit_mask,
        "dyn_mean": dyn_mean,
        "dyn_std": dyn_std,
        "beta_summaries": beta_summaries,
        "best_beta_eff": best["beta_eff"],
        "best_mse": best["mse"],
        "best_mae": best["mae"],
        "best_max_abs": best["max_abs"],
        "best_static_mean": best["mean_curve"],
        "best_static_std": best["std_curve"],
        "fit_tmax": t_grid[fit_mask][-1],
    }

# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("SUMMARY OF BEST FITS")
print("=" * 80)

for family, res in results.items():
    print(f"\n--- {family} ---")
    print(f"N={res['N']}, M={res['M']}, <k>={res['avg_k']:.3f}")
    print(f"Best beta_eff = {res['best_beta_eff']:.4f}")
    print(f"MSE           = {res['best_mse']:.6f}")
    print(f"MAE           = {res['best_mae']:.6f}")
    print(f"Max |diff|    = {res['best_max_abs']:.6f}")
    print(f"Fit window    = [0, {res['fit_tmax']:.4f}]")

# ============================================================
# PLOTS: one separate figure per network
# ============================================================

for family, res in results.items():
    t_grid = res["t_grid"]
    dyn_mean = res["dyn_mean"]
    dyn_std = res["dyn_std"]
    st_mean = res["best_static_mean"]
    st_std = res["best_static_std"]
    best_beta = res["best_beta_eff"]

    plt.figure(figsize=(7.4, 5.3))

    plt.plot(
        t_grid, dyn_mean,
        linewidth=2.5,
        label=f"Dynamic (mu = beta0 = {beta0:.2f})"
    )
    if show_std_band:
        plt.fill_between(
            t_grid,
            np.clip(dyn_mean - dyn_std, 0, 1),
            np.clip(dyn_mean + dyn_std, 0, 1),
            alpha=0.18
        )

    plt.plot(
        t_grid, st_mean,
        linewidth=2.3,
        linestyle="--",
        label=f"Static best fit (beta_eff = {best_beta:.3f})"
    )
    if show_std_band:
        plt.fill_between(
            t_grid,
            np.clip(st_mean - st_std, 0, 1),
            np.clip(st_mean + st_std, 0, 1),
            alpha=0.18
        )

    plt.axvline(
        res["fit_tmax"],
        linestyle=":",
        linewidth=1.5,
        alpha=0.8,
        label="Fit window end"
    )

    plt.xlabel("Time")
    plt.ylabel("Mean infected fraction, <I(t)/N>")
    plt.title(
        f"{family}: dynamic vs static trajectory fit\n"
        f"MSE={res['best_mse']:.4g}, MAE={res['best_mae']:.4g}"
    )
    plt.ylim(-0.02, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()