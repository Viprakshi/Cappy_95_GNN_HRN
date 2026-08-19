from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt


def visualize_graph(graph, players=None, ball=None, title: str = "Volleyball frame graph"):
    """Visualize a single-frame volleyball graph using graph.x coordinates.

    The graph stores normalized node positions in graph.x[:, 2] and graph.x[:, 3], and
    node types are encoded in graph.node_type. Player labels come from graph.player_ids.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("x_norm")
    ax.set_ylabel("y_norm")

    node_x = graph.x[:, 2].detach().cpu().numpy()
    node_y = graph.x[:, 3].detach().cpu().numpy()
    node_type = graph.node_type.detach().cpu().numpy()

    player_mask = node_type[:, 0] == 1
    ball_mask = node_type[:, 1] == 1

    for idx in np.where(player_mask)[0]:
        x = float(node_x[idx])
        y = float(node_y[idx])
        player_label = graph.player_ids[idx] if idx < len(graph.player_ids) else idx
        ax.scatter(x, y, color="tab:blue", s=90)
        ax.text(x + 0.01, y + 0.01, str(player_label), fontsize=9)

    for idx in np.where(ball_mask)[0]:
        x = float(node_x[idx])
        y = float(node_y[idx])
        ax.scatter(x, y, color="tab:orange", s=120, marker="o")
        ax.text(x + 0.01, y + 0.01, "BALL", fontsize=9)

    if graph.edge_index is not None and graph.edge_attr is not None:
        for edge_idx in range(graph.edge_index.size(1)):
            src = int(graph.edge_index[0, edge_idx].item())
            dst = int(graph.edge_index[1, edge_idx].item())
            if src >= graph.num_nodes or dst >= graph.num_nodes:
                continue

            edge_type = int(graph.edge_type[edge_idx].item())
            color = "tab:green" if edge_type == 0 else "tab:red"
            x_src = float(graph.x[src, 2])
            y_src = float(graph.x[src, 3])
            x_dst = float(graph.x[dst, 2])
            y_dst = float(graph.x[dst, 3])

            ax.plot([x_src, x_dst], [y_src, y_dst], color=color, alpha=0.6, linewidth=1)

    plt.tight_layout()
    return fig, ax
