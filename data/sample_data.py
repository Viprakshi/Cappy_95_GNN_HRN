import numpy as np


IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080


def build_pose_for_player(player_id: int):
    """Build a deterministic synthetic 17-keypoint pose for one player."""
    base_x = 420 + (player_id - 1) * 90
    base_y = 650 + ((player_id - 1) % 2) * 25

    keypoints = [
        (base_x, base_y),
        (base_x - 25, base_y - 40),
        (base_x + 25, base_y - 40),
        (base_x - 50, base_y - 80),
        (base_x + 50, base_y - 80),
        (base_x - 60, base_y - 150),
        (base_x + 60, base_y - 150),
        (base_x - 75, base_y - 220),
        (base_x + 75, base_y - 220),
        (base_x - 80, base_y - 260),
        (base_x + 80, base_y - 260),
        (base_x - 25, base_y - 300),
        (base_x + 25, base_y - 300),
        (base_x - 35, base_y - 360),
        (base_x + 35, base_y - 360),
        (base_x - 55, base_y - 420),
        (base_x + 55, base_y - 420),
    ]

    flat_pose = []
    for x, y in keypoints:
        flat_pose.extend([x, y])

    return np.asarray(flat_pose, dtype=np.float32)


def create_sample_data():
    """
    Create one synthetic volleyball frame with six players and one ball.

    Each player pose is a normalized 17-keypoint representation of the form:
        [x1, y1, x2, y2, ..., x17, y17]
    """

    players = [
        {
            "id": 1,
            "x": 500,
            "y": 700,
            "vx": 2.0,
            "vy": -1.0,
            "team": 0,
            "confidence": 0.95,
            "pose": build_pose_for_player(1),
        },
        {
            "id": 2,
            "x": 750,
            "y": 600,
            "vx": 1.0,
            "vy": 0.5,
            "team": 0,
            "confidence": 0.94,
            "pose": build_pose_for_player(2),
        },
        {
            "id": 3,
            "x": 900,
            "y": 750,
            "vx": -1.0,
            "vy": 0.2,
            "team": 0,
            "confidence": 0.96,
            "pose": build_pose_for_player(3),
        },
        {
            "id": 4,
            "x": 1200,
            "y": 700,
            "vx": -1.5,
            "vy": 0.5,
            "team": 1,
            "confidence": 0.93,
            "pose": build_pose_for_player(4),
        },
        {
            "id": 5,
            "x": 1400,
            "y": 600,
            "vx": 0.5,
            "vy": -0.5,
            "team": 1,
            "confidence": 0.95,
            "pose": build_pose_for_player(5),
        },
        {
            "id": 6,
            "x": 1550,
            "y": 750,
            "vx": -0.5,
            "vy": -1.0,
            "team": 1,
            "confidence": 0.92,
            "pose": build_pose_for_player(6),
        },
    ]

    ball = {
        "x": 1020,
        "y": 500,
        "vx": 3.0,
        "vy": -2.0,
        "confidence": 0.98,
    }

    court = {
        "width": IMAGE_WIDTH,
        "height": IMAGE_HEIGHT,
        "net_x": IMAGE_WIDTH / 2,
    }

    return players, ball, court