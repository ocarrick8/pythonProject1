#!/usr/bin/env python3
"""
ER three-seed side-by-side Gillespie animations

Produces one window with 3 panels (highest-degree seed, median-degree seed, lowest-degree seed).
"""

import argparse
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# -----------------------
# Parameters (you can change)
# -----------------------
N = 40
p = 0.1
infection_rate = 1.0      # beta0 (λ)
k_layout = 1.5
scale_layout = 8.0
rotation_speed_deg_per_frame = 1.0
node_size = 40
interval_ms = 250         # ms per frame (visual)
# -----------------------

@dataclass
class Event:
    t: float
    i: int
    j: int

@dataclass
class History:
    times: List[float]
    infected_sets: List[set]

# Gillespie simulation (matching your earlier logic)
def simulate_infection(G: nx.Graph, beta0: float = 1.0, seed_node: Optional[int] = None, rng_seed: Optional[int] = None) -> Tuple[List[Event], History]:
    if rng_seed is not None:
        random.seed(rng_seed)
    Nn = G.number_of_nodes()
    if Nn == 0:
        return [], History([], [])
    nodes = list(G.nodes())
    if seed_node is None:
        s0 = random.choice(nodes)
    else:
        s0 = seed_node
        if s0 not in G:
            raise ValueError(f"seed_node {s0} not in graph")

    S = set(nodes)
    I = set()
    S.remove(s0); I.add(s0)
    t = 0.0
    events: List[Event] = []
    times: List[float] = [t]
    infected_sets: List[set] = [set(I)]

    while True:
        sus_counts: Dict[int, int] = {}
        total_directed = 0
        for i in I:
            k = 0
            for v in G.neighbors(i):
                if v in S:
                    k += 1
            if k > 0:
                sus_counts[i] = k
                total_directed += k
        if total_directed == 0:
            break

        R = beta0 * total_directed
        u = random.random() or 1e-12
        dt = -math.log(u) / R
        t += dt

        ticket = random.randrange(total_directed)
        chosen_i = None
        acc = 0
        for i in sus_counts:
            acc += sus_counts[i]
            if ticket < acc:
                chosen_i = i
                break
        assert chosen_i is not None

        cand = [v for v in G.neighbors(chosen_i) if v in S]
        j = random.choice(cand)

        S.remove(j); I.add(j)
        events.append(Event(t=t, i=chosen_i, j=j))
        times.append(t); infected_sets.append(set(I))

    return events, History(times=times, infected_sets=infected_sets)


# Build graph and layout once
def build_graph_and_layout(seed_graph: int = 42):
    G = nx.erdos_renyi_graph(N, p, seed=seed_graph)
    pos3d = nx.spring_layout(G, dim=3, seed=1, k=k_layout, scale=scale_layout)
    return G, pos3d

# Pick seeds: highest, median, lowest by degree
def pick_seeds(G: nx.Graph) -> Tuple[int, int, int]:
    degree_list = sorted(G.degree(), key=lambda x: x[1])
    lowest = degree_list[0][0]
    median = degree_list[len(degree_list)//2][0]
    highest = degree_list[-1][0]
    return highest, median, lowest
def show_initial_states(G, pos3d, seeds):
    """Display 3 side-by-side initial states of the network."""
    fig = plt.figure(figsize=(20, 7))
    axs = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
    labels = ["highest degree", "median degree", "lowest degree"]

    for ax, s, lbl in zip(axs, seeds, labels):
        ax.set_axis_off()
        ax.set_xlim(-scale_layout, scale_layout)
        ax.set_ylim(-scale_layout, scale_layout)
        ax.set_zlim(-scale_layout, scale_layout)

        # Get degree
        deg = G.degree[s]

        # Draw nodes
        xs = [pos3d[i][0] for i in G.nodes()]
        ys = [pos3d[i][1] for i in G.nodes()]
        zs = [pos3d[i][2] for i in G.nodes()]
        colors = ["red" if i == s else "#66b2ff" for i in G.nodes()]
        ax.scatter(xs, ys, zs, c=colors, s=80, edgecolor='k', linewidth=0.5)

        # Draw edges
        for u, v in G.edges():
            x = [pos3d[u][0], pos3d[v][0]]
            y = [pos3d[u][1], pos3d[v][1]]
            z = [pos3d[u][2], pos3d[v][2]]

            # Highlight the seed’s edges
            if u == s or v == s:
                ax.plot(x, y, z, color=(0.0, 1.0, 0.0, 0.9) , linewidth=2.5)  # red + opaque
            else:
                ax.plot(x, y, z, color=(0.5, 0.5, 0.5, 0.15), linewidth=1)

        ax.set_title(f"Seed {s} — degree {deg}\n({lbl})")

    plt.tight_layout()
    plt.show()

# Draw frame function (used by FuncAnimation)
def animate_all_three(G, pos3d, seeds, sim_results,
                      save_path=None, interval=interval_ms,
                      rotation_speed=rotation_speed_deg_per_frame):

    fig = plt.figure(figsize=(20, 7))
    axs = [fig.add_subplot(1, 3, i + 1, projection='3d') for i in range(3)]

    titles = [
        f"Seed {seeds[0]} (degree {G.degree[seeds[0]]}) — highest",
        f"Seed {seeds[1]} (degree {G.degree[seeds[1]]}) — median",
        f"Seed {seeds[2]} (degree {G.degree[seeds[2]]}) — lowest",
    ]

    # ---------------------------
    # Compute the global max time
    # ---------------------------
    max_time = max(sim_results[s]["history"].times[-1] for s in seeds)

    # Number of animation frames (visual only)
    frames = 200
    dt = max_time / frames  # physical time step per frame

    def get_state_at_time(s, t):
        """Return infected_set and highlight_edge at physical time t."""
        hist = sim_results[s]["history"]
        evs = sim_results[s]["events"]

        times = hist.times
        infected = hist.infected_sets

        # If time exceeds last event: fully infected
        if t >= times[-1]:
            return infected[-1], None

        # Find largest index with time <= t
        idx = max(i for i, T in enumerate(times) if T <= t)

        # If idx>0 then highlight the event that occurred at idx
        he = None
        if idx > 0 and idx - 1 < len(evs):
            ev = evs[idx - 1]
            he = (ev.i, ev.j)

        return infected[idx], he

    def draw_panel(ax, infected_set, highlight_edge):
        ax.clear()
        ax.set_axis_off()
        ax.set_xlim(-scale_layout, scale_layout)
        ax.set_ylim(-scale_layout, scale_layout)
        ax.set_zlim(-scale_layout, scale_layout)

        # draw nodes
        xs = [pos3d[i][0] for i in G.nodes()]
        ys = [pos3d[i][1] for i in G.nodes()]
        zs = [pos3d[i][2] for i in G.nodes()]
        colors = ['red' if i in infected_set else '#66b2ff' for i in G.nodes()]
        ax.scatter(xs, ys, zs, c=colors, s=node_size)

        # draw edges
        for u, v in G.edges():
            x = [pos3d[u][0], pos3d[v][0]]
            y = [pos3d[u][1], pos3d[v][1]]
            z = [pos3d[u][2], pos3d[v][2]]

            if highlight_edge and ((u, v) == highlight_edge or (v, u) == highlight_edge):
                ax.plot(x, y, z, color=(1,1,0,1), linewidth=4)
            else:
                su = u in infected_set
                sv = v in infected_set
                if su and sv:
                    col = (1, 0, 0, 0.6)
                elif not su and not sv:
                    col = (0, 0, 1, 0.12)
                else:
                    col = (0, 1, 0, 0.8)
                ax.plot(x, y, z, color=col, linewidth=1.5)

    def draw_frame(frame):
        t = frame * dt  # convert frame index → physical time

        for ax, s, title in zip(axs, seeds, titles):
            infected_set, highlight_edge = get_state_at_time(s, t)
            draw_panel(ax, infected_set, highlight_edge)

            ax.set_title(title + f"\nTime = {t:.2f}  |  I = {len(infected_set)}/{N}")

            # spin camera
            ax.view_init(elev=20, azim=(frame * rotation_speed) % 360)

        return axs

    ani = FuncAnimation(fig, draw_frame, frames=frames, interval=interval, blit=False)

    if save_path:
        ani.save(save_path, dpi=150)

    plt.tight_layout()
    plt.show()



def main(save_path: Optional[str] = None):
    # Build graph/layout
    G, pos3d = build_graph_and_layout(seed_graph=42)
    seeds = pick_seeds(G)

    # ---- NEW: show initial state comparison ----
    show_initial_states(G, pos3d, seeds)

    # Precompute simulations
    sim_results = {}
    for s in seeds:
        events, history = simulate_infection(G, beta0=infection_rate, seed_node=s, rng_seed=1000 + s)
        sim_results[s] = {"events": events, "history": history}

    # Run animations
    animate_all_three(G, pos3d, seeds, sim_results, save_path=save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ER three-seed side-by-side Gillespie animations")
    parser.add_argument("--save", type=str, default=None, help="optional path to save animation (gif/mp4)")
    args = parser.parse_args()
    main(save_path=args.save)
