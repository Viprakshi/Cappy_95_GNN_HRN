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
    print(f"Number of nodes: {graph.num_nodes}")
    print(f"Number of edges: {graph.num_edges}")
    print(f"Node feature shape: {graph.x.shape}")
    print(f"Edge index shape: {graph.edge_index.shape}")
    print(f"Edge feature shape: {graph.edge_attr.shape}")
    print(f"Edge type shape: {graph.edge_type.shape}")
    print(f"Player nodes: {int((graph.node_type == 0).sum().item())}")
    print(f"Ball nodes: {int((graph.node_type == 1).sum().item())}")
    print(f"Player-player edges: {player_player_edges}")
    print(f"Player-ball edges: {player_ball_edges}")

    print("\nExample edge:")
    for idx in range(min(2, graph.edge_index.size(1))):
        src = int(graph.edge_index[0, idx].item())
        dst = int(graph.edge_index[1, idx].item())
        edge_kind = (
            "PLAYER_PLAYER"
            if int(graph.edge_type[idx].item()) == PLAYER_PLAYER_EDGE_TYPE
            else "PLAYER_BALL"
        )
        attrs = graph.edge_attr[idx].tolist()
        print(f"source={src} target={dst} type={edge_kind} attributes={attrs}")

    print("\n========================================")


if __name__ == "__main__":
    main()