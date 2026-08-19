import math

import numpy as np
import torch
from torch_geometric.data import Data

NUM_POSE_KEYPOINTS = 17
POSE_FEATURE_DIM = NUM_POSE_KEYPOINTS * 2
NODE_TYPE_DIM = 2
POSITION_DIM = 2
VELOCITY_DIM = 2
TEAM_DIM = 2
POSE_AVAILABLE_DIM = 1
CONFIDENCE_DIM = 1
COURT_FEATURE_DIM = 5
NODE_FEATURE_DIM = (
    NODE_TYPE_DIM
    + POSITION_DIM
    + VELOCITY_DIM
    + TEAM_DIM
    + POSE_FEATURE_DIM
    + POSE_AVAILABLE_DIM
    + CONFIDENCE_DIM
    + COURT_FEATURE_DIM
)

PLAYER_NODE_TYPE = np.array([1.0, 0.0], dtype=np.float32)
BALL_NODE_TYPE = np.array([0.0, 1.0], dtype=np.float32)
PLAYER_PLAYER_EDGE_TYPE = 0
PLAYER_BALL_EDGE_TYPE = 1

assert NODE_FEATURE_DIM == 49


class VolleyballGraphBuilder:
    """Convert a volleyball frame into a PyTorch Geometric graph."""

    def __init__(self):
        self.feature_dim = NODE_FEATURE_DIM

    def validate_frame_inputs(self, players, ball, court):
        """Validate basic frame inputs before graph creation."""
        if not isinstance(players, (list, tuple)) or len(players) == 0:
            raise ValueError("players must be a non-empty list of player dictionaries.")
        if not isinstance(ball, dict):
            raise ValueError("ball must be a dictionary with x and y coordinates.")
        if not isinstance(court, dict):
            raise ValueError("court must be a dictionary with width and height.")

        width = court.get("width")
        height = court.get("height")
        if width is None or height is None or width <= 0 or height <= 0:
            raise ValueError("court width and height must be positive numbers.")

        for idx, player in enumerate(players):
            if not isinstance(player, dict):
                raise ValueError(f"player at index {idx} is not a dictionary.")
            if "x" not in player or "y" not in player:
                raise ValueError(f"player at index {idx} is missing x or y coordinates.")
            if not np.isfinite(float(player["x"])) or not np.isfinite(float(player["y"])):
                raise ValueError(f"player at index {idx} has non-finite coordinates.")

        if "x" not in ball or "y" not in ball:
            raise ValueError("ball is missing x or y coordinates.")
        if not np.isfinite(float(ball["x"])) or not np.isfinite(float(ball["y"])):
            raise ValueError("ball has non-finite coordinates.")

    def normalize_position(self, x, y, width, height):
        """Convert image coordinates to normalized values in [0, 1]."""
        x_norm = np.clip(float(x) / width, 0.0, 1.0)
        y_norm = np.clip(float(y) / height, 0.0, 1.0)
        return x_norm, y_norm

    def normalize_velocity(self, vx, vy, width, height):
        """Normalize velocity from image-space units to normalized units per frame."""
        vx_norm = float(vx) / width
        vy_norm = float(vy) / height
        return vx_norm, vy_norm

    def normalize_pose(self, pose, width, height):
        """Normalize a 17-keypoint pose array of shape [34]."""
        if pose is None:
            return np.zeros(POSE_FEATURE_DIM, dtype=np.float32), 0.0

        pose_array = np.asarray(pose, dtype=np.float32).reshape(-1)
        if pose_array.size == 0:
            return np.zeros(POSE_FEATURE_DIM, dtype=np.float32), 0.0
        if pose_array.size != POSE_FEATURE_DIM:
            raise ValueError(
                f"Pose length must be exactly {POSE_FEATURE_DIM} values "
                f"(17 keypoints x 2 coordinates), got {pose_array.size}."
            )

        pose_xy = pose_array.reshape(-1, 2)
        pose_xy[:, 0] = np.clip(pose_xy[:, 0] / width, 0.0, 1.0)
        pose_xy[:, 1] = np.clip(pose_xy[:, 1] / height, 0.0, 1.0)
        return pose_xy.reshape(-1).astype(np.float32), 1.0

    def get_team_features(self, team):
        """Encode team IDs as a two-class one-hot vector."""
        if team == 0:
            return np.array([1.0, 0.0], dtype=np.float32)
        if team == 1:
            return np.array([0.0, 1.0], dtype=np.float32)
        return np.array([0.0, 0.0], dtype=np.float32)

    def get_court_features(self, x, y, court):
        """Compute normalized court-relative spatial features for a player or ball."""
        width = float(court["width"])
        height = float(court["height"])
        net_x = float(court.get("net_x", width / 2.0))

        distance_left = np.clip(float(x) / width, 0.0, 1.0)
        distance_right = np.clip((width - float(x)) / width, 0.0, 1.0)
        distance_top = np.clip(float(y) / height, 0.0, 1.0)
        distance_bottom = np.clip((height - float(y)) / height, 0.0, 1.0)
        distance_net = np.clip(abs(float(x) - net_x) / max(width, 1.0), 0.0, 1.0)

        return np.asarray(
            [
                distance_left,
                distance_right,
                distance_top,
                distance_bottom,
                distance_net,
            ],
            dtype=np.float32,
        )

    def build_player_features(self, player, court):
        """Create a 49-dimensional feature vector for one player."""
        width = float(court["width"])
        height = float(court["height"])

        x = float(player["x"])
        y = float(player["y"])
        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError(f"Player {player.get('id', 'unknown')} has invalid coordinates.")

        x_norm, y_norm = self.normalize_position(x, y, width, height)
        vx_norm, vy_norm = self.normalize_velocity(
            player.get("vx", 0.0),
            player.get("vy", 0.0),
            width,
            height,
        )

        team_features = self.get_team_features(player.get("team", -1))
        pose, pose_available = self.normalize_pose(player.get("pose"), width, height)
        confidence = float(np.clip(player.get("confidence", 0.0), 0.0, 1.0))
        court_features = self.get_court_features(x, y, court)

        feature_vector = np.concatenate(
            [
                PLAYER_NODE_TYPE,
                np.asarray([x_norm, y_norm], dtype=np.float32),
                np.asarray([vx_norm, vy_norm], dtype=np.float32),
                team_features,
                pose,
                np.asarray([pose_available], dtype=np.float32),
                np.asarray([confidence], dtype=np.float32),
                court_features,
            ]
        ).astype(np.float32)

        if feature_vector.shape[0] != self.feature_dim:
            raise ValueError(
                f"Player feature size mismatch: expected {self.feature_dim}, got {feature_vector.shape[0]}."
            )

        return feature_vector

    def build_ball_features(self, ball, court):
        """Create a 49-dimensional feature vector for the ball."""
        width = float(court["width"])
        height = float(court["height"])

        x = float(ball["x"])
        y = float(ball["y"])
        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError("Ball has invalid coordinates.")

        x_norm, y_norm = self.normalize_position(x, y, width, height)
        vx_norm, vy_norm = self.normalize_velocity(
            ball.get("vx", 0.0),
            ball.get("vy", 0.0),
            width,
            height,
        )

        pose = np.zeros(POSE_FEATURE_DIM, dtype=np.float32)
        confidence = float(np.clip(ball.get("confidence", 0.0), 0.0, 1.0))
        court_features = self.get_court_features(x, y, court)

        feature_vector = np.concatenate(
            [
                BALL_NODE_TYPE,
                np.asarray([x_norm, y_norm], dtype=np.float32),
                np.asarray([vx_norm, vy_norm], dtype=np.float32),
                np.asarray([0.0, 0.0], dtype=np.float32),
                pose,
                np.asarray([0.0], dtype=np.float32),
                np.asarray([confidence], dtype=np.float32),
                court_features,
            ]
        ).astype(np.float32)

        if feature_vector.shape[0] != self.feature_dim:
            raise ValueError(
                f"Ball feature size mismatch: expected {self.feature_dim}, got {feature_vector.shape[0]}."
            )

        return feature_vector

    def build_edges(self, players, ball, court):
        """Build directed edges with normalized relative_x/relative_y values.

        The relative_x and relative_y fields are normalized by court width and height,
        not clipped to [0, 1]. As a result, the Euclidean distance is computed on the
        normalized coordinates and is therefore approximately in [0, sqrt(2)].
        """
        edge_list = []
        edge_attr = []
        edge_types = []
        width = float(court["width"])
        height = float(court["height"])

        for i in range(len(players)):
            for j in range(len(players)):
                if i == j:
                    continue

                p1 = players[i]
                p2 = players[j]
                relative_x = (float(p2["x"]) - float(p1["x"])) / width
                relative_y = (float(p2["y"]) - float(p1["y"])) / height
                distance = math.sqrt(relative_x ** 2 + relative_y ** 2)

                team_a = p1.get("team")
                team_b = p2.get("team")
                same_team = (
                    1.0
                    if team_a in (0, 1) and team_b in (0, 1) and team_a == team_b
                    else 0.0
                )

                edge_list.append([i, j])
                edge_attr.append([relative_x, relative_y, distance, same_team])
                edge_types.append(PLAYER_PLAYER_EDGE_TYPE)

        ball_index = len(players)
        for i, player in enumerate(players):
            relative_x = (float(ball["x"]) - float(player["x"])) / width
            relative_y = (float(ball["y"]) - float(player["y"])) / height
            distance = math.sqrt(relative_x ** 2 + relative_y ** 2)

            edge_list.append([i, ball_index])
            edge_attr.append([relative_x, relative_y, distance, 0.0])
            edge_types.append(PLAYER_BALL_EDGE_TYPE)

            edge_list.append([ball_index, i])
            edge_attr.append([-relative_x, -relative_y, distance, 0.0])
            edge_types.append(PLAYER_BALL_EDGE_TYPE)

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32)
        edge_type_tensor = torch.tensor(edge_types, dtype=torch.long)

        return edge_index, edge_attr_tensor, edge_type_tensor

    def build_graph(self, players, ball, court, frame_id=None, timestamp=None):
        """Construct a PyTorch Geometric graph for a single volleyball frame."""
        self.validate_frame_inputs(players, ball, court)

        player_ids = [player.get("id", idx) for idx, player in enumerate(players)]
        node_features = []
        node_types = []

        for player in players:
            node_features.append(self.build_player_features(player, court))
            node_types.append(0)

        node_features.append(self.build_ball_features(ball, court))
        node_types.append(1)

        x = torch.tensor(np.asarray(node_features, dtype=np.float32), dtype=torch.float32)
        edge_index, edge_attr, edge_type = self.build_edges(players, ball, court)

        assert x.dim() == 2
        assert x.size(1) == NODE_FEATURE_DIM
        assert edge_index.dim() == 2
        assert edge_index.size(0) == 2
        assert edge_attr.size(0) == edge_index.size(1)
        assert edge_type.size(0) == edge_index.size(1)
        assert torch.isfinite(x).all()
        assert torch.isfinite(edge_attr).all()

        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        graph.edge_type = edge_type
        graph.node_type = torch.tensor(node_types, dtype=torch.long)
        graph.player_ids = player_ids
        graph.ball_index = len(players)

        if frame_id is not None:
            graph.frame_id = frame_id
        if timestamp is not None:
            graph.timestamp = timestamp

        return graph