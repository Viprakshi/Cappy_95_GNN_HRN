from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt


def visualize_graph(graph, players=None, ball=None, title: str = "Volleyball frame graph"):
    """Create a simple 2D plot of the volleyball graph for debugging."""
    if players is None and hasattr(graph, "player_ids"):
        players = [
            {"id": player_id, "x": 0.0, "y": 0.0}
            for player_id in graph.player_ids
        ]

    if ball is None:
        ball = {"x": 0.0, "y": 0.0}

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("x_norm")
    ax.set_ylabel("y_norm")

    if players:
        for idx, player in enumerate(players):
            x = player.get("x_norm", 0.0)
            y = player.get("y_norm", 0.0)
            ax.scatter(x, y, color="tab:blue", s=90)
            ax.text(x + 0.01, y + 0.01, str(player.get("id", idx)), fontsize=9)

    if ball:
        bx = ball.get("x_norm", 0.0)
        by = ball.get("y_norm", 0.0)
        ax.scatter(bx, by, color="tab:orange", s=120, marker="o")
        ax.text(bx + 0.01, by + 0.01, "BALL", fontsize=9)

    if graph.edge_index is not None and graph.edge_attr is not None:
        for edge_idx in range(graph.edge_index.size(1)):
            src = int(graph.edge_index[0, edge_idx].item())
            dst = int(graph.edge_index[1, edge_idx].item())
            if src >= graph.num_nodes or dst >= graph.num_nodes:
                continue

            x_src = float(graph.x[src, 2])
            y_src = float(graph.x[src, 3])
            x_dst = float(graph.x[dst, 2])
            y_dst = float(graph.x[dst, 3])

            ax.plot([x_src, x_dst], [y_src, y_dst], color="gray", alpha=0.5, linewidth=1)

    plt.tight_layout()
    return fig, ax
