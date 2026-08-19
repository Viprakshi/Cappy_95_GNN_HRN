import io
import runpy
from contextlib import redirect_stdout

import numpy as np
import torch

from data.sample_data import create_sample_data
from graph.graph_builder import (
    NODE_FEATURE_DIM,
    PLAYER_BALL_EDGE_TYPE,
    PLAYER_PLAYER_EDGE_TYPE,
    VolleyballGraphBuilder,
)


def test_six_players_and_ball_create_seven_nodes():
    players, ball, court = create_sample_data()
    graph = VolleyballGraphBuilder().build_graph(players, ball, court)
    assert graph.num_nodes == 7


def test_full_graph_has_expected_edge_count():
    players, ball, court = create_sample_data()
    graph = VolleyballGraphBuilder().build_graph(players, ball, court)
    assert graph.num_edges == 42
    assert int((graph.edge_type == PLAYER_PLAYER_EDGE_TYPE).sum().item()) == 30
    assert int((graph.edge_type == PLAYER_BALL_EDGE_TYPE).sum().item()) == 12


def test_node_feature_dimension_is_49_for_all_nodes():
    players, ball, court = create_sample_data()
    graph = VolleyballGraphBuilder().build_graph(players, ball, court)
    assert graph.x.shape == (7, NODE_FEATURE_DIM)
    assert graph.x.size(1) == 49


def test_edge_attr_matches_edge_index_rows():
    players, ball, court = create_sample_data()
    graph = VolleyballGraphBuilder().build_graph(players, ball, court)
    assert graph.edge_attr.shape[0] == graph.edge_index.shape[1]
    assert graph.edge_type.shape[0] == graph.edge_index.shape[1]


def test_normalized_coordinates_are_in_range():
    players, ball, court = create_sample_data()
    graph = VolleyballGraphBuilder().build_graph(players, ball, court)
    assert torch.all(graph.x[:, 2] >= 0.0)
    assert torch.all(graph.x[:, 2] <= 1.0)
    assert torch.all(graph.x[:, 3] >= 0.0)
    assert torch.all(graph.x[:, 3] <= 1.0)


def test_edge_features_are_normalized_and_non_negative_distance():
    players, ball, court = create_sample_data()
    graph = VolleyballGraphBuilder().build_graph(players, ball, court)
    assert torch.all(graph.edge_attr[:, 0].abs() <= 1.0)
    assert torch.all(graph.edge_attr[:, 1].abs() <= 1.0)
    assert torch.all(graph.edge_attr[:, 2] >= 0.0)


def test_opposite_edges_have_opposite_dx_dy():
    players, ball, court = create_sample_data()
    graph = VolleyballGraphBuilder().build_graph(players, ball, court)
    for idx in range(graph.edge_index.size(1)):
        src = int(graph.edge_index[0, idx].item())
        dst = int(graph.edge_index[1, idx].item())
        if src != dst and src < len(players) and dst < len(players):
            reverse_idx = None
            for j in range(graph.edge_index.size(1)):
                if int(graph.edge_index[0, j].item()) == dst and int(graph.edge_index[1, j].item()) == src:
                    reverse_idx = j
                    break
            assert reverse_idx is not None
            assert torch.allclose(graph.edge_attr[idx, :2], -graph.edge_attr[reverse_idx, :2])


def test_missing_pose_is_handled():
    player = {
        "id": 99,
        "x": 300.0,
        "y": 400.0,
        "team": 0,
        "confidence": 0.9,
        "pose": None,
    }
    players = [player]
    ball = {"x": 600.0, "y": 500.0, "confidence": 0.8}
    court = {"width": 1920, "height": 1080, "net_x": 960}
    graph = VolleyballGraphBuilder().build_graph(players, ball, court)
    assert graph.x.shape[1] == NODE_FEATURE_DIM
    assert graph.x[0, 42].item() == 0.0


def test_missing_velocity_is_handled():
    players, ball, court = create_sample_data()
    players[0].pop("vx", None)
    players[0].pop("vy", None)
    graph = VolleyballGraphBuilder().build_graph(players, ball, court)
    assert graph.x.shape[1] == NODE_FEATURE_DIM
    assert torch.isfinite(graph.x[:, 4:6]).all()


def test_variable_player_count_is_supported():
    players, ball, court = create_sample_data()
    reduced_players = players[:4]
    graph = VolleyballGraphBuilder().build_graph(reduced_players, ball, court)
    assert graph.num_nodes == 5
    assert graph.num_edges == 4 * 3 + 8


def test_player_ids_are_metadata_not_node_features():
    players, ball, court = create_sample_data()
    graph = VolleyballGraphBuilder().build_graph(players, ball, court)
    assert hasattr(graph, "player_ids")
    assert len(graph.player_ids) == len(players)
    assert not any(isinstance(v, list) for v in graph.player_ids)


def test_no_nan_or_inf_values():
    players, ball, court = create_sample_data()
    graph = VolleyballGraphBuilder().build_graph(players, ball, court)
    assert torch.isfinite(graph.x).all()
    assert torch.isfinite(graph.edge_attr).all()
    assert torch.isfinite(graph.edge_type).all()


def test_module_runs_graph_summary_when_executed_as_script():
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        runpy.run_module("main", run_name="__main__")
    output = stdout.getvalue()
    assert "GRAPH INFORMATION" in output
    assert "Number of nodes:" in output
    assert "Number of edges:" in output
