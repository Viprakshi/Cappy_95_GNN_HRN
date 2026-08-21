from data.sample_data import create_sample_data

from graph.graph_builder import (
    PLAYER_BALL_EDGE_TYPE,
    PLAYER_PLAYER_EDGE_TYPE,
    VolleyballGraphBuilder,
)

from visualization.visualize_graph import (
    visualize_feature_graph,
    visualize_graph,
    visualize_interaction_graph,
)


def main():

    # ========================================================
    # CREATE INPUT DATA
    # ========================================================

    players, ball, court = (
        create_sample_data()
    )

    # ========================================================
    # BUILD GRAPH
    # ========================================================

    builder = (
        VolleyballGraphBuilder()
    )

    graph = builder.build_graph(
        players,
        ball,
        court
    )

    # ========================================================
    # GRAPH INFORMATION
    # ========================================================

    print(
        "\n========== GRAPH INFORMATION =========="
    )

    print(
        "Number of nodes:",
        graph.num_nodes
    )

    print(
        "Number of edges:",
        graph.num_edges
    )

    print(
        "Node feature shape:",
        graph.x.shape
    )

    print(
        "Edge index shape:",
        graph.edge_index.shape
    )

    print(
        "Edge feature shape:",
        graph.edge_attr.shape
    )

    print(
        "Edge type shape:",
        graph.edge_type.shape
    )

    # --------------------------------------------------------
    # Node counts
    # --------------------------------------------------------

    num_players = int(
        (
            graph.node_type
            == 0
        ).sum().item()
    )

    num_balls = int(
        (
            graph.node_type
            == 1
        ).sum().item()
    )

    # --------------------------------------------------------
    # Edge counts
    # --------------------------------------------------------

    num_player_player_edges = int(
        (
            graph.edge_type
            == PLAYER_PLAYER_EDGE_TYPE
        ).sum().item()
    )

    num_player_ball_edges = int(
        (
            graph.edge_type
            == PLAYER_BALL_EDGE_TYPE
        ).sum().item()
    )

    print(
        "Player nodes:",
        num_players
    )

    print(
        "Ball nodes:",
        num_balls
    )

    print(
        "Player-player edges:",
        num_player_player_edges
    )

    print(
        "Player-ball edges:",
        num_player_ball_edges
    )

    # ========================================================
    # FEATURE GRAPH INFORMATION
    # ========================================================

    print(
        "\n========== FEATURE GRAPH =========="
    )

    print(
        "Node feature dimension:",
        graph.x.size(1)
    )

    print(
        "Expected feature dimension:",
        49
    )

    print(
        "Feature graph valid:",
        graph.x.size(1) == 49
    )

    # ========================================================
    # INTERACTION GRAPH INFORMATION
    # ========================================================

    print(
        "\n========== INTERACTION GRAPH =========="
    )

    print(
        "Total interactions:",
        graph.num_edges
    )

    print(
        "Player-player interactions:",
        num_player_player_edges
    )

    print(
        "Player-ball interactions:",
        num_player_ball_edges
    )

    print(
        "Edge feature dimension:",
        graph.edge_attr.size(1)
    )

    # ========================================================
    # EXAMPLE EDGE
    # ========================================================

    print(
        "\n========== EXAMPLE EDGE =========="
    )

    source = (
        graph.edge_index[
            0,
            0
        ].item()
    )

    target = (
        graph.edge_index[
            1,
            0
        ].item()
    )

    edge_kind = (
        graph.edge_type[
            0
        ].item()
    )

    attributes = (
        graph.edge_attr[
            0
        ].tolist()
    )

    print(
        "Source:",
        source
    )

    print(
        "Target:",
        target
    )

    print(
        "Edge type:",
        edge_kind
    )

    print(
        "Attributes:",
        attributes
    )

    print(
        "\n========================================"
    )

    # ========================================================
    # VISUALIZATIONS
    # ========================================================

    visualize_graph(
        graph,
        title="Volleyball Feature + Interaction Graph"
    )

    visualize_feature_graph(
        graph
    )

    visualize_interaction_graph(
        graph
    )


if __name__ == "__main__":
    main()