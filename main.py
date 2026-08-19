from data.sample_data import create_sample_data
from graph.graph_builder import VolleyballGraphBuilder


def main():

    players, ball, court = create_sample_data()

    builder = VolleyballGraphBuilder()

    graph = builder.build_graph(
        players,
        ball,
        court
    )

    print("\n========== GRAPH INFORMATION ==========")

    print("Number of nodes:", graph.num_nodes)
    print("Number of edges:", graph.num_edges)

    print("Node feature shape:", graph.x.shape)

    print("Edge index shape:", graph.edge_index.shape)

    print("Edge feature shape:", graph.edge_attr.shape)

    print("\nNode features:")
    print(graph.x)

    print("\nEdge index:")
    print(graph.edge_index)

    print("\nEdge attributes:")
    print(graph.edge_attr)

    print("\n========================================")


if __name__ == "__main__":
    main()