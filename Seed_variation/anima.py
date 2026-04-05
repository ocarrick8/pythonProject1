import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# -------------------------------
# PARAMETERS
# -------------------------------
N = 40
p = 0.1
seed_node = 0
infection_rate = 1  # λ
rotation_speed = 1.0  # degrees per frame
# -------------------------------

# Generate graph
G = nx.erdos_renyi_graph(N, p, seed=42)

# Generate a 3D layout
pos3d = nx.spring_layout(G, dim=3, seed=1, k=1.5, scale=8)


# Infection state: 0 = susceptible, 1 = infected
state = np.zeros(N, dtype=int)
state[seed_node] = 1

infected = {seed_node}

# Infection curve storage
times = [0.0]
infected_counts = [1]

# Highlighted edge
highlight_edge = None
frame_index = 0


def get_infectious_edges():
    """Edges where infection can spread (u infected, v susceptible)."""
    return [(u, v) for u in infected for v in G.neighbors(u) if state[v] == 0]


# -----------------------------------
# FIGURE LAYOUT
# -----------------------------------
fig = plt.figure(figsize=(16, 6))

# Large 3D plot: left 80%
ax3d = fig.add_axes([0.02, 0.05, 0.70, 0.90], projection='3d')

# Infection curve: right 20%
ax_curve = fig.add_axes([0.75, 0.15, 0.23, 0.70])

ax_curve.set_title("Infection Curve")
ax_curve.set_xlabel("Time")
ax_curve.set_ylabel("Number Infected")
line_curve, = ax_curve.plot([], [], lw=2)


def draw_graph_3d():
    global frame_index
    ax3d.clear()

    # -----------------------------------
    # Remove ALL axis components (safe)
    # -----------------------------------
    ax3d.set_axis_off()
    ax3d.grid(False)

    # Remove panes / background
    ax3d.xaxis.pane.set_visible(False)
    ax3d.yaxis.pane.set_visible(False)
    ax3d.zaxis.pane.set_visible(False)

    # Remove axis lines
    ax3d.xaxis.line.set_visible(False)
    ax3d.yaxis.line.set_visible(False)
    ax3d.zaxis.line.set_visible(False)

    # Remove ticks
    ax3d.set_xticks([])
    ax3d.set_yticks([])
    ax3d.set_zticks([])

    ax3d.set_xlim(-8, 8)
    ax3d.set_ylim(-8, 8)
    ax3d.set_zlim(-8, 8)

    # -----------------------------------
    # Draw nodes + edges
    # -----------------------------------
    colors = ["red" if state[i] == 1 else "blue" for i in range(N)]
    xs = [pos3d[i][0] for i in range(N)]
    ys = [pos3d[i][1] for i in range(N)]
    zs = [pos3d[i][2] for i in range(N)]

    ax3d.scatter(xs, ys, zs, c=colors, s=45)

    # Edges
    # -----------------------------------
    # Draw edges with custom color logic
    # -----------------------------------
    for u, v in G.edges():

        # Coordinates
        x = [pos3d[u][0], pos3d[v][0]]
        y = [pos3d[u][1], pos3d[v][1]]
        z = [pos3d[u][2], pos3d[v][2]]

        # Highlighted infection edge (overrides all)
        if highlight_edge and (u, v) == highlight_edge:
            ax3d.plot(x, y, z, color=(1.0, 1.0, 0.0, 1.0), linewidth=4)
            continue

        # Node states
        su, sv = state[u], state[v]

        # RED → RED (infected to infected)
        if su == 1 and sv == 1:
            color = (1.0, 0.0, 0.0, 0.8)   # red, medium opacity
            lw = 1.4

        # BLUE → BLUE (susceptible to susceptible)
        elif su == 0 and sv == 0:
            color = (0.0, 0.0, 1.0, 0.15)  # blue, very faint
            lw = 0.5

        # RED ↔ BLUE (infectious edge)
        else:
            color = (0.0, 1.0, 0.0, 0.4)   # green, strong but not opaque
            lw = 1.0

        ax3d.plot(x, y, z, color=color, linewidth=lw)


    # Spin camera
    ax3d.view_init(elev=20, azim=frame_index * rotation_speed)
    frame_index += 1



# -----------------------------------
# GILLESPIE SIMULATION
# -----------------------------------
def gillespie_steps():
    global highlight_edge

    time = 0.0
    draw_graph_3d()
    yield state

    while sum(state) < N:
        infectious_edges = get_infectious_edges()
        if not infectious_edges:
            break

        rate = infection_rate * len(infectious_edges)
        dt = np.random.exponential(1 / rate)
        time += dt

        # Choose infection
        u, v = infectious_edges[np.random.randint(len(infectious_edges))]
        state[v] = 1
        infected.add(v)

        # Highlight edge
        highlight_edge = (u, v)

        # Update infection curve
        times.append(time)
        infected_counts.append(sum(state))

        draw_graph_3d()
        line_curve.set_data(times, infected_counts)
        ax_curve.set_xlim(0, times[-1] * 1.05)
        ax_curve.set_ylim(0, N)

        yield state

        # Remove highlight after one frame
        highlight_edge = None


# -----------------------------------
# RUN ANIMATION
# -----------------------------------
ani = FuncAnimation(fig, lambda frame: None,
                    frames=gillespie_steps,
                    interval=700)

plt.show()

