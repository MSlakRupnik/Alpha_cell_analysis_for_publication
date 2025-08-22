# --- Setup ---
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

from islets.Regions import Regions, load_regions
import islets
from islets.utils import *

# --- Preprocessing ---
max_t_end = regions.protocol['t_end'].max()
rounded_max_t_end = int((max_t_end // 50) * 50)

print("The maximal t_end value is:", max_t_end)
print("Rounded down to the nearest multiple of 50:", rounded_max_t_end)


# --- Helper functions ---
def make_graph(g, pos, t, scale, th):
    """Build graph from connectivity matrix with threshold."""
    m = g[t][scale]
    N = m.shape[0]
    int_mat = np.zeros((N, N))
    node_weights = [sum(m[node, j] for j in range(N) if j != node)
                    for node in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            p1, p2 = node_weights[i], node_weights[j]
            if p1 > 0 and p2 > 0:
                int_mat[i, j] = int_mat[j, i] = p1 * p2 / (N - 1) ** 2
    norm_int_mat = int_mat / int_mat.max()
    adj_mat = np.where(norm_int_mat > th, 1, 0)
    return nx.from_numpy_array(adj_mat)


def swap_axes(pos_dict):
    return {n: (y, x) for n, (x, y) in pos_dict.items()}


def flip_y(pos_dict):
    return {n: (x, -y) for n, (x, y) in pos_dict.items()}


def plot_network_highlight_connected(G, pos, title=None):
    """Plot network with blue edges, red connected nodes, navy isolated nodes."""
    connected_nodes = [n for n, d in G.degree() if d > 0]
    isolated_nodes = [n for n, d in G.degree() if d == 0]

    plt.figure(figsize=(6, 5))
    # draw edges (uniform blue)
    nx.draw_networkx_edges(G, pos, edge_color='blue', width=0.5)

    # draw nodes: isolated navy, connected red
    if isolated_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=isolated_nodes, node_color='navy', node_size=30)
    if connected_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=connected_nodes, node_color='red', node_size=30)

    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# --- Analysis ---
time_ranges = [{'range': (start, start+400), 'label': f"{start}-{start+400}"}
               for start in range(100, rounded_max_t_end, 600)]

T = 200
data = []

raw_events = islets.EventDistillery.sequential_filtering(
    regions, timescales=2.**np.arange(10), verbose=False)

for tr in time_ranges:
    start, end = tr['range']
    label = tr['label']

    g = islets.utils.get_ccs(
        regions,
        skip_sequential_filtering=True,
        time_ranges={i: (j, j+T) for i, j in zip(range(100), range(start, end, 10))},
        mode="cross",
    )

    coords = regions.df.peak.to_list()
    G = make_graph(g, coords, 0, 6, 0.70)

    mean_degree = np.mean([G.degree()[node] for node in G.nodes()])
    clustering_coeff = nx.average_clustering(G)

    print(f"Time Range: {label}, Mean Degree: {mean_degree}, "
          f"Clustering Coefficient: {clustering_coeff}")

    data.append([label, mean_degree, clustering_coeff])


# --- Results ---
df_result = pd.DataFrame(data, columns=['time_range', 'mean_degree', 'clustering_coeff'])

pathToNetwork = pathToRois.split("_rois")[0] + "_network.csv"
df_result.to_csv(pathToNetwork, index=False)
df_result = pd.read_csv(pathToNetwork)


# --- Visualization of metrics ---
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))

# Plot Mean Degree
sns.lineplot(x='time_range', y='mean_degree', data=df_result,
             color='red', marker='o', ax=axes[0])
axes[0].set_ylabel("Mean Degree")
axes[0].set_title("Mean Degree by Time Range")

# Plot Clustering Coefficient
sns.lineplot(x='time_range', y='clustering_coeff', data=df_result,
             color='blue', marker='o', ax=axes[1])
axes[1].set_ylabel("Clustering Coefficient")
axes[1].set_title("Clustering Coefficient by Time Range")

# Fix x-axis labels (show every 5th for readability)
for ax in axes:
    current_labels = [label.get_text() for label in ax.get_xticklabels()]
    new_labels = current_labels[::5]
    ax.set_xticks(range(0, len(current_labels), 5))
    ax.set_xticklabels(new_labels, rotation=90, ha='right')

plt.tight_layout()
plt.show()


# --- Visualization of networks ---
specific_time_ranges = [100, 700, 1300, 1900, 2500, 3100]
T_window = 10  # window length

for start in specific_time_ranges:
    end = start + 500
    time_ranges = {i: (j, j + T_window) for i, j in zip(range(100), range(start, end, 10))}

    g = islets.utils.get_ccs(
        regions,
        skip_sequential_filtering=True,
        time_ranges=time_ranges,
        mode="cross"
    )

    coords = regions.df.peak.to_list()
    G = make_graph(g, coords, 0, 6, 0.7)

    # make pos dictionary from coords
    pos_dict = dict(zip(G.nodes(), coords))

    # swap axes then flip y-axis
    pos_swapped = swap_axes(pos_dict)
    pos_swapped_flipped_y = flip_y(pos_swapped)

    plot_network_highlight_connected(
        G,
        pos_swapped_flipped_y,
        title=f"α-cells Network, Time Range {start}-{end}"
    )
