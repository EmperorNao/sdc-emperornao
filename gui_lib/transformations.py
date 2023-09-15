import math

import numpy as np
from typing import *


def rotate_points_3d(points: np.array, angle: float, axis: str) -> np.array:
    assert axis in ['x', 'y', 'z']
    if axis == 'z':
        rot = np.array([
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1]
        ])
    if axis == 'y':
        rot = np.array([
            [math.cos(angle), 0, math.sin(angle)],
            [0, 1, 0],
            [-math.sin(angle), 0, math.cos(angle)],
        ])
    if axis == 'x':
        rot = np.array([
            [1, 0, 0],
            [0, math.cos(angle), -math.sin(angle)],
            [0, math.sin(angle), math.cos(angle)],
        ])

    return points @ rot


def boxes_straight2rotated_3d(boxes: np.array) -> np.array:
    """Rotate 3d boxes in z dim
        :param boxes: np.array with shape (n,8) in format [cx, cy, cz, dx, dy, dz, yaw, label]
        :return np.array with 8 rotated points for each box, shape (n,8,3)
    """

    rotated_boxes = []
    for box in boxes:
        cx, cy, cz, dx, dy, dz, yaw = box
        dx, dy, dz = dx / 2, dy / 2, dz / 2
        new_d = rotate_points_3d(np.array([
            [-dx, -dy, -dz],
            [-dx, dy, -dz],
            [-dx, dy, dz],
            [-dx, -dy, dz],
            [dx, -dy, -dz],
            [dx, dy, -dz],
            [dx, dy, dz],
            [dx, -dy, dz]
        ]), yaw, axis='z')
        rotated_boxes.append(new_d + np.array([cx, cy, cz]))

    return np.stack(rotated_boxes, axis=0)
