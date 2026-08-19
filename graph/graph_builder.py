import math
import numpy as np
import torch
from torch_geometric.data import Data


class VolleyballGraphBuilder:
    """
    Converts one volleyball frame into a PyTorch Geometric graph.

    Node types:
        0 -> Player
        1 -> Ball

    The graph is designed for future GAT/HRN processing.
    """

    def __init__(self):
        # Player and ball nodes use the same fixed feature size.
        # 2 node-type + 2 normalized coords + 2 velocity + 2 team one-hot
        # + 34 pose + 1 confidence + 5 court features = 48.
        self.feature_dim = 48

    def validate_frame_inputs(self, players, ball, court):
        """Validate the basic frame inputs before building a graph."""
        if not isinstance(players, (list, tuple)) or len(players) == 0:
            raise ValueError("players must be a non-empty list of player dictionaries.")

        if not isinstance(ball, dict):
            raise ValueError("ball must be a dictionary containing x and y coordinates.")

        if not isinstance(court, dict):
            raise ValueError("court must be a dictionary containing width and height.")

        for idx, player in enumerate(players):
            if not isinstance(player, dict):
                raise ValueError(f"player at index {idx} is not a dictionary.")
            if "x" not in player or "y" not in player:
                raise ValueError(f"player at index {idx} is missing x or y coordinates.")
            if not np.isfinite(player["x"]) or not np.isfinite(player["y"]):
                raise ValueError(f"player at index {idx} has non-finite coordinates.")

        if "x" not in ball or "y" not in ball:
            raise ValueError("ball is missing x or y coordinates.")
        if not np.isfinite(ball["x"]) or not np.isfinite(ball["y"]):
            raise ValueError("ball has non-finite coordinates.")

        width = court.get("width")
        height = court.get("height")
        if width is None or height is None or width <= 0 or height <= 0:
            raise ValueError("court width and height must be positive numbers.")

    def normalize_position(self, x, y, width, height):
        """Convert image coordinates to [0, 1]."""

        x_norm = np.clip(x / width, 0.0, 1.0)
        y_norm = np.clip(y / height, 0.0, 1.0)

        return x_norm, y_norm

    def get_court_features(self, x, y, court):
        """
        Calculate simple court-relative spatial features.

        Features:
        - distance from left boundary
        - distance from right boundary
        - distance from top boundary
        - distance from bottom boundary
        - distance from net

        All values are normalized.
        """

        width = court["width"]
        height = court["height"]
        net_x = court["net_x"]

        left = x / width
        right = (width - x) / width
        top = y / height
        bottom = (height - y) / height
        net = abs(x - net_x) / width

        return [
            left,
            right,
            top,
            bottom,
            net,
        ]

    def create_player_features(self, player, court):
        """Create the fixed-size feature vector for a player."""

        width = court["width"]
        height = court["height"]

        x = player["x"]
        y = player["y"]

        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError(f"Player {player.get('id', 'unknown')} has invalid coordinates.")

        x_norm, y_norm = self.normalize_position(
            x, y, width, height
        )

        vx = player.get("vx", 0.0)
        vy = player.get("vy", 0.0)

        team = player.get("team", 0)

        # One-hot encoding for two teams.
        team_one_hot = [0.0, 0.0]

        if team in [0, 1]:
            team_one_hot[team] = 1.0

        pose = player.get(
            "pose",
            np.zeros(34, dtype=np.float32)
        )

        pose = np.asarray(pose, dtype=np.float32)

        if pose.size != 34:
            pose = np.zeros(34, dtype=np.float32)

        confidence = player.get("confidence", 1.0)

        court_features = self.get_court_features(
            x, y, court
        )

        # Node type: [player, ball]
        node_type = [1.0, 0.0]

        features = (
            node_type
            + [x_norm, y_norm]
            + [vx, vy]
            + team_one_hot
            + pose.tolist()
            + [confidence]
            + court_features
        )

        feature_vector = np.asarray(features, dtype=np.float32)
        if feature_vector.shape[0] != self.feature_dim:
            raise ValueError(
                f"Player feature size mismatch: expected {self.feature_dim}, got {feature_vector.shape[0]}."
            )

        return feature_vector

    def create_ball_features(self, ball, court):
        """Create the fixed-size feature vector for the ball."""

        width = court["width"]
        height = court["height"]

        x = ball["x"]
        y = ball["y"]

        x_norm, y_norm = self.normalize_position(
            x, y, width, height
        )

        vx = ball.get("vx", 0.0)
        vy = ball.get("vy", 0.0)

        confidence = ball.get("confidence", 1.0)

        court_features = self.get_court_features(
            x, y, court
        )

        # Node type: [player, ball]
        node_type = [0.0, 1.0]

        # Ball does not have team or pose features.
        team_features = [0.0, 0.0]
        pose_features = [0.0] * 34

        features = (
            node_type
            + [x_norm, y_norm]
            + [vx, vy]
            + team_features
            + pose_features
            + [confidence]
            + court_features
        )

        feature_vector = np.asarray(features, dtype=np.float32)
        if feature_vector.shape[0] != self.feature_dim:
            raise ValueError(
                f"Ball feature size mismatch: expected {self.feature_dim}, got {feature_vector.shape[0]}."
            )

        return feature_vector

    def build_edges(self, players, ball):
        """
        Build initial graph connectivity.

        Currently:
        - every player connects to every other player
        - every player connects to the ball

        More sophisticated relationship construction
        will be handled in the next task.
        """

        edge_list = []
        edge_features = []

        # Player-player relationships.
        for i in range(len(players)):
            for j in range(i + 1, len(players)):

                p1 = players[i]
                p2 = players[j]

                dx = p2["x"] - p1["x"]
                dy = p2["y"] - p1["y"]

                distance = math.sqrt(dx ** 2 + dy ** 2)

                same_team = float(
                    p1.get("team", 0) == p2.get("team", 0)
                )

                attributes = [
                    dx,
                    dy,
                    distance,
                    same_team,
                ]

                # Add both directions.
                edge_list.append([i, j])
                edge_features.append(attributes)

                edge_list.append([j, i])
                edge_features.append(
                    [-dx, -dy, distance, same_team]
                )

        # Player-ball relationships.
        ball_index = len(players)

        for i, player in enumerate(players):

            dx = ball["x"] - player["x"]
            dy = ball["y"] - player["y"]

            distance = math.sqrt(dx ** 2 + dy ** 2)

            attributes = [
                dx,
                dy,
                distance,
                0.0,
            ]

            edge_list.append([i, ball_index])
            edge_features.append(attributes)

            edge_list.append([ball_index, i])
            edge_features.append(
                [-dx, -dy, distance, 0.0]
            )

        edge_index = torch.tensor(
            edge_list,
            dtype=torch.long
        ).t().contiguous()

        edge_attr = torch.tensor(
            edge_features,
            dtype=torch.float32
        )

        return edge_index, edge_attr

    def build_graph(self, players, ball, court):
        """Convert a volleyball frame into a PyTorch Geometric graph."""

        self.validate_frame_inputs(players, ball, court)

        node_features = []

        # Player nodes.
        for player in players:
            features = self.create_player_features(
                player, court
            )
            node_features.append(features)

        # Ball node.
        ball_features = self.create_ball_features(
            ball, court
        )
        node_features.append(ball_features)

        x = torch.tensor(
            np.array(node_features),
            dtype=torch.float32
        )

        edge_index, edge_attr = self.build_edges(
            players,
            ball
        )

        if x.shape[1] != self.feature_dim:
            raise ValueError(
                f"Node feature matrix has inconsistent feature dimension: expected {self.feature_dim}, got {x.shape[1]}."
            )

        graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr
        )

        return graph