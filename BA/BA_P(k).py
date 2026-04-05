import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Parameters
# -----------------------
N_values = [500, 900, 1200]
m_values = [3, 7, 12]
n_runs = 20  # << increase: reduces noise a lot

# -----------------------
# BA theoretical P(k): exact
# P(k)=2m(m+1)/[k(k+1)(k+2)] for k>=m, else 0
# -----------------------
def ba_theory_pk(k_vals, m):
    k = np.asarray(k_vals, dtype=float)
    pk = 2 * m * (m + 1) / (k * (k + 1) * (k + 2))
    pk[k < m] = 0.0
    # normalise over the provided k grid (useful if you truncate at k_max)
    s = pk.sum()
    return pk / s if s > 0 else pk

# -----------------------
# Store summary results
# -----------------------
results = {}

# -----------------------
# Loop over N
# -----------------------
for N in N_values:

    fig, axes = plt.subplots(
        1, len(m_values),
        figsize=(5 * len(m_values), 4),
        squeeze=False
    )

    for im, m in enumerate(m_values):

        deg_arrays = []
        kmax_list = []

        # --- Multiple runs ---
        for r in range(n_runs):
            G = nx.barabasi_albert_graph(N, m, seed=r)  # seed for reproducibility
            degrees = np.fromiter((d for _, d in G.degree()), dtype=int)
            deg_arrays.append(degrees)
            kmax_list.append(int(degrees.max()))

        # --- Use a clean contiguous support from m up to max observed degree ---
        kmax_overall = max(kmax_list)
        k_support = np.arange(m, kmax_overall + 1, dtype=int)

        # --- Average empirical P(k) over runs on this support ---
        pk_runs = []
        for degrees in deg_arrays:
            counts = np.bincount(degrees, minlength=kmax_overall + 1)  # counts[k]
            pk = counts / counts.sum()
            pk_runs.append(pk[m:kmax_overall + 1])  # align with k_support

        pk_mean = np.mean(pk_runs, axis=0)

        # --- Theory on same support ---
        p_theory = ba_theory_pk(k_support, m)

        # --- Deviation metrics (only where empirical > 0 to avoid log(0)) ---
        mask = (pk_mean > 0) & (p_theory > 0)
        log_rmse = np.sqrt(np.mean((np.log10(pk_mean[mask]) - np.log10(p_theory[mask])) ** 2))

        dev_L1 = float(np.sum(np.abs(pk_mean - p_theory)))

        # --- k_max stats ---
        kmax_mean = float(np.mean(kmax_list))
        kmax_std = float(np.std(kmax_list))

        results[(N, m)] = {
            "kmax_runs": kmax_list,
            "kmax_mean": kmax_mean,
            "kmax_std": kmax_std,
            "dev_L1": dev_L1,
            "log_rmse": float(log_rmse),
        }

        # -----------------------
        # Plot
        # -----------------------
        ax = axes[0, im]
        ax.scatter(k_support, pk_mean, s=18, label=f"Empirical (avg over {n_runs} runs)")
        ax.plot(k_support, p_theory, linewidth=2, label=r"BA theory: $\frac{2m(m+1)}{k(k+1)(k+2)}$")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("k (log-scale)")
        ax.set_ylabel("P(k) (log-scale)")


        ax.set_title(
            f"m={m}\n"
        )
        ax.legend(fontsize=8)

    fig.suptitle(f"BA Degree Distribution (N={N})", fontsize=14)
    plt.tight_layout()
    plt.show()

# -----------------------
# Print summary table
# -----------------------
for (N, m), d in results.items():
    print(
        f"(N={N}, m={m})  "
        f"k_max mean={d['kmax_mean']:.2f}±{d['kmax_std']:.2f}  "
        f"logRMSE={d['log_rmse']:.4f}  "
    )
