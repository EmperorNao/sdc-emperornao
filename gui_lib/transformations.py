import math

import numpy as np
from typing import *


def translate_points_2d(points: np.array, angle: np.array) -> np.array:
    """

    :param points: array of shape (b, n, 2)
        b - batch, n - points to translate, 2 - dimensions (x, y)
    :param angle: array of size b with angles
    :return: return array of shape (b, n, 2) in which all points in batch rotated by angle
    """
    rot = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
    ])

    return points.transpose(2, 0, 1) @ rot.transpose(2, 0, 1)


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


def boxes_straight2rotated(boxes: np.array) -> np.array:
    """Rotate 2b boxes
        :param boxes: np.array with shape (n,5) in format [cx, cy, dx, dy, yaw]
        :return np.array with 4 rotated points for each box, shape (n,4,2)
    """

    cx, cy, dx, dy, yaw = boxes[:, :5].transpose(1, 0)
    dx, dy = dx / 2, dy / 2
    new_d = translate_points_2d(np.array([[-dx, -dy], [-dx, dy], [dx, dy], [dx, -dy]]), yaw)
    rotated_boxes = new_d + np.expand_dims(np.array([cx, cy]), 0).transpose(2, 0, 1)

    return rotated_boxes


def boxes_straight2rotated_3d(boxes: np.array) -> np.array:
    """Rotate 3d boxes in z dim
        :param boxes: np.array with shape (n,7) in format [cx, cy, cz, dx, dy, dz, yaw]
        :return np.array with 8 rotated points for each box, shape (n,8,3)
    """

    rotated_boxes = []
    for box in boxes:
        cx, cy, cz, dx, dy, dz, yaw = box[:7]
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
