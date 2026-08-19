from data.sample_data import create_sample_data
from graph.graph_builder import (
    PLAYER_BALL_EDGE_TYPE,
    PLAYER_PLAYER_EDGE_TYPE,
    VolleyballGraphBuilder,
)


def main():
    players, ball, court = create_sample_data()
    builder = VolleyballGraphBuilder()
    graph = builder.build_graph(players, ball, court)

    player_player_edges = int((graph.edge_type == PLAYER_PLAYER_EDGE_TYPE).sum().item())
    player_ball_edges = int((graph.edge_type == PLAYER_BALL_EDGE_TYPE).sum().item())

    print("\n========== GRAPH INFORMATION ==========")

    print("Number of nodes:", graph.num_nodes)
    print("Number of edges:", graph.num_edges)

    print("Node feature shape:", graph.x.shape)
    print("Edge index shape:", graph.edge_index.shape)
    print("Edge feature shape:", graph.edge_attr.shape)
    print("Edge type shape:", graph.edge_type.shape)

    num_players = int(
        (graph.node_type == 0).sum()
    )

    num_balls = int(
        (graph.node_type == 1).sum()
    )

    num_player_player_edges = int(
        (graph.edge_type == 0).sum()
    )

    num_player_ball_edges = int(
        (graph.edge_type == 1).sum()
    )

    print("Player nodes:", num_players)
    print("Ball nodes:", num_balls)

    print(
        "Player-player edges:",
        num_player_player_edges
    )

    print(
        "Player-ball edges:",
        num_player_ball_edges
    )

    print("\nExample edge:")

    source = graph.edge_index[0, 0].item()
    target = graph.edge_index[1, 0].item()
    edge_kind = graph.edge_type[0].item()
    attributes = graph.edge_attr[0].tolist()

    print("Source:", source)
    print("Target:", target)
    print("Edge type:", edge_kind)
    print("Attributes:", attributes)

    print("\n========================================")

if __name__ == "__main__":
    main()