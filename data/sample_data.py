import numpy as np


IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080


def create_sample_data():
    """
    Creates one synthetic volleyball frame.

    The coordinates are intentionally simple and can later
    be replaced by real detection/pose outputs.
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
            "pose": np.zeros(34, dtype=np.float32),
        },
        {
            "id": 2,
            "x": 750,
            "y": 600,
            "vx": 1.0,
            "vy": 0.5,
            "team": 0,
            "confidence": 0.94,
            "pose": np.zeros(34, dtype=np.float32),
        },
        {
            "id": 3,
            "x": 900,
            "y": 750,
            "vx": -1.0,
            "vy": 0.2,
            "team": 0,
            "confidence": 0.96,
            "pose": np.zeros(34, dtype=np.float32),
        },
        {
            "id": 4,
            "x": 1200,
            "y": 700,
            "vx": -1.5,
            "vy": 0.5,
            "team": 1,
            "confidence": 0.93,
            "pose": np.zeros(34, dtype=np.float32),
        },
        {
            "id": 5,
            "x": 1400,
            "y": 600,
            "vx": 0.5,
            "vy": -0.5,
            "team": 1,
            "confidence": 0.95,
            "pose": np.zeros(34, dtype=np.float32),
        },
        {
            "id": 6,
            "x": 1550,
            "y": 750,
            "vx": -0.5,
            "vy": -1.0,
            "team": 1,
            "confidence": 0.92,
            "pose": np.zeros(34, dtype=np.float32),
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