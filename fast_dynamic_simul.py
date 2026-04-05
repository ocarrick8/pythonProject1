from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class GraphData:
    n: int
    edges_u: np.ndarray
    edges_v: np.ndarray
    nbrs: List[np.ndarray]
    inc_edges: List[np.ndarray]


# ============================================================
# O(1) dense set
# ============================================================

class IndexSet:
    def __init__(self, size: int):
        self.items = np.empty(size, dtype=np.int32)
        self.pos = -np.ones(size, dtype=np.int32)
        self.count = 0

    def add(self, x: int) -> None:
        if self.pos[x] != -1:
            return
        self.items[self.count] = x
        self.pos[x] = self.count
        self.count += 1

    def remove(self, x: int) -> None:
        p = self.pos[x]
        if p == -1:
            return
        last = self.items[self.count - 1]
        self.items[p] = last
        self.pos[last] = p
        self.pos[x] = -1
        self.count -= 1

    def contains(self, x: int) -> bool:
        return self.pos[x] != -1

    def random_choice(self, rng: np.random.Generator) -> int:
        idx = int(rng.integers(0, self.count))
        return int(self.items[idx])


# ============================================================
# FAST DYNAMIC SI: track only SI edges
# ============================================================

def simulate_dynamic_si_frontier_only(
    graph: GraphData,
    beta: float,
    omega_on: float,
    omega_off: float,
    rng: np.random.Generator,
    seed_node: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast exact SI simulation for an underlying graph with independent edge ON/OFF switching.

    Key idea:
    - Only SI edges matter for future infection.
    - SS and II edges are never explicitly simulated.
    - When an SS edge becomes SI, its current ON/OFF state is sampled from the
      stationary distribution rho = omega_on / (omega_on + omega_off).

    This is exact for the model where each edge switches independently and the
    edge-state process is started in stationarity.
    """
    n = graph.n
    m = len(graph.edges_u)

    if seed_node is None:
        seed_node = int(rng.integers(0, n))

    rho = omega_on / (omega_on + omega_off)

    infected = np.zeros(n, dtype=bool)
    infected[seed_node] = True

    # Track only SI edges, split into ON and OFF
    si_on = IndexSet(m)
    si_off = IndexSet(m)

    # Initialise frontier from seed to its susceptible neighbours.
    # Each new SI edge is ON independently with probability rho.
    for e in graph.inc_edges[seed_node]:
        u = int(graph.edges_u[e])
        v = int(graph.edges_v[e])
        other = v if u == seed_node else u

        if infected[other]:
            continue

        if rng.random() < rho:
            si_on.add(int(e))
        else:
            si_off.add(int(e))

    times = [0.0]
    fracs = [1.0 / n]
    t = 0.0
    infected_count = 1

    while infected_count < n:
        n_on = si_on.count
        n_off = si_off.count

        rate_inf = beta * n_on
        rate_on = omega_on * n_off
        rate_off = omega_off * n_on
        total_rate = rate_inf + rate_on + rate_off

        if total_rate <= 0.0:
            break

        t += rng.exponential(1.0 / total_rate)
        r = rng.random() * total_rate

        # ----------------------------------------------------
        # 1) Infection along a currently ON SI edge
        # ----------------------------------------------------
        if r < rate_inf:
            e = si_on.random_choice(rng)

            u = int(graph.edges_u[e])
            v = int(graph.edges_v[e])

            # exactly one endpoint must be infected
            x = u if not infected[u] else v
            infected[x] = True
            infected_count += 1

            # Any SI edge touching x now ceases to be SI
            # (because x is now infected).
            for ee in graph.inc_edges[x]:
                si_on.remove(int(ee))
                si_off.remove(int(ee))

            # New SI edges are created from x to susceptible neighbours.
            # These edges were previously SS, so we sample their current
            # ON/OFF state from the stationary law.
            for ee in graph.inc_edges[x]:
                uu = int(graph.edges_u[ee])
                vv = int(graph.edges_v[ee])
                y = vv if uu == x else uu

                if infected[y]:
                    continue

                if rng.random() < rho:
                    si_on.add(int(ee))
                else:
                    si_off.add(int(ee))

            times.append(t)
            fracs.append(infected_count / n)
            continue

        r -= rate_inf

        # ----------------------------------------------------
        # 2) OFF -> ON on an SI edge
        # ----------------------------------------------------
        if r < rate_on:
            e = si_off.random_choice(rng)
            si_off.remove(e)
            si_on.add(e)
            continue

        # ----------------------------------------------------
        # 3) ON -> OFF on an SI edge
        # ----------------------------------------------------
        e = si_on.random_choice(rng)
        si_on.remove(e)
        si_off.add(e)

    return np.array(times, dtype=float), np.array(fracs, dtype=float)