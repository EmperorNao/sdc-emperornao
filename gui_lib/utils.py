import math
import torch
import numpy as np
from utils.types import *
from scipy.spatial.transform import Rotation as R


def translate_boxes_lidar2bev(boxes: np.array,
                              scene_size: Tuple[int, int]
                              ) -> Tuple[np.array, np.array]:

    filtered_boxes = []
    filtered_idx = []
    for box in boxes:

        yaw = box[6]

        proj_lidar_box = np.eye(4)
        proj_lidar_box[0:3, 0:3] = R.from_euler('z', yaw, degrees=False).as_matrix()

        x, y, z = box[0:3]
        proj_lidar_box[0][3], proj_lidar_box[1][3], proj_lidar_box[2][3] = x, y, z

        if -scene_size[0] < x < scene_size[0] and -scene_size[1] < y < scene_size[1]:

            width, length, height = box[3:6]

            front_right_top = np.array(
                [[1, 0, 0, length / 2], [0, 1, 0, width / 2], [0, 0, 1, height / 2], [0, 0, 0, 1]])

            front_left_top = np.array(
                [[1, 0, 0, length / 2], [0, 1, 0, -width / 2], [0, 0, 1, height / 2], [0, 0, 0, 1]])

            back_right_top = np.array(
                [[1, 0, 0, -length / 2], [0, 1, 0, width / 2], [0, 0, 1, height / 2], [0, 0, 0, 1]])

            back_left_top = np.array(
                [[1, 0, 0, -length / 2], [0, 1, 0, -width / 2], [0, 0, 1, height / 2], [0, 0, 0, 1]])

            f_r_t = np.matmul(proj_lidar_box, front_right_top)
            f_l_t = np.matmul(proj_lidar_box, front_left_top)
            b_r_t = np.matmul(proj_lidar_box, back_right_top)
            b_l_t = np.matmul(proj_lidar_box, back_left_top)

            filtered_boxes.append(np.array([
                [f_r_t[0][3], f_r_t[1][3]],
                [f_l_t[0][3], f_l_t[1][3]],
                [b_l_t[0][3], b_l_t[1][3]],
                [b_r_t[0][3], b_r_t[1][3]],
            ])
            )

            filtered_idx.append(True)
        else:
            filtered_idx.append(False)

    if filtered_boxes:
        filtered_boxes = np.stack(filtered_boxes)

        translated_boxes = np.copy(filtered_boxes)
        for i in range(4):
            translated_boxes[:, i, 0] = -filtered_boxes[:, i, 1]
            translated_boxes[:, i, 1] = filtered_boxes[:, i, 0]

        translated_boxes += np.array([scene_size, scene_size, scene_size, scene_size])

        return np.array(filtered_idx), translated_boxes

    else:
        return np.array([]), np.array([])


def boxes_bev2image(boxes: np.array,
                    scene_size: Tuple[int, int],
                    image_size: Tuple[int, int]):

    x1 = np.expand_dims(np.min(boxes[:, :, 0], axis=1), -1)
    x2 = np.expand_dims(np.max(boxes[:, :, 0], axis=1), -1)
    y1 = np.expand_dims(np.min(boxes[:, :, 1], axis=1), -1)
    y2 = np.expand_dims(np.max(boxes[:, :, 1], axis=1), -1)

    wdx = np.abs(x2 - x1) / 2
    wdy = np.abs(y2 - y1) / 2
    cx = x1 + wdx
    cy = np.abs(scene_size[1] * 2 - y1) - wdy

    dx = np.expand_dims(np.linalg.norm(boxes[:, 0] - boxes[:, 1], ord=2, axis=1), -1) / 2
    dy = np.expand_dims(np.linalg.norm(boxes[:, 1] - boxes[:, 2], ord=2, axis=1), -1) / 2

    xs, ys = image_size[0] / (scene_size[0] * 2), image_size[1] / (scene_size[1] * 2)

    cx = np.rint(cx * xs).astype(int)
    dx = np.rint(dx * xs).astype(int)
    cy = np.rint(cy * ys).astype(int)
    dy = np.rint(dy * ys).astype(int)

    return np.concatenate(
        (cx, cy, dx, dy),
        axis=1
    )


def translate_points(points: np.array, yaw: float) -> np.array:
    rot = np.array([
        [math.cos(yaw), math.sin(yaw)],
        [-math.sin(yaw), math.cos(yaw)]
    ])

    return points @ rot


def boxes_straight2rotated(boxes: np.array) -> np.array:

    rotated_boxes = []
    for box in boxes:
        cx, cy, dx, dy, yaw = box
        new_d = translate_points(np.array([[-dx, -dy], [-dx, dy], [dx, dy], [dx, -dy]]), -yaw)
        rotated_boxes.append(new_d + np.array([cx, cy]))

    return np.stack(rotated_boxes, axis=0)


def transform_boxes_cxcy2ltrb(boxes: Tensor):

    return torch.cat(
        [
            boxes[:2] - 0.5 * boxes[2:4],
            boxes[:2] + 0.5 * boxes[2:4]
        ]
    )


def ltrb2shape(box):
    return [[box[0], box[1]],
                      [box[0], box[3]],
                      [box[2], box[3]],
                      [box[2], box[1]]
        ]


def translate_points_3d(points: np.array, yaw: float) -> np.array:
    rot = np.array([
        [math.cos(yaw), math.sin(yaw), 0],
        [-math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
        # y
        # [math.cos(yaw), 0, math.sin(yaw)],
        # [0, 1, 0],
        # [-math.sin(yaw), 0, math.cos(yaw)],
        # x
        # [1, 0, 0,],
        # [0, math.cos(yaw), -math.sin(yaw)],
        # [0, math.sin(yaw), math.cos(yaw)],
    ])

    return points @ rot


def boxes_straight2rotated_3d(boxes: np.array) -> np.array:

    rotated_boxes = []
    for box in boxes:
        cx, cy, cz, dx, dy, dz, yaw = box
        dx, dy, dz = dx / 2, dy / 2, dz / 2
        new_d = translate_points_3d(np.array([
            [-dx, -dy, -dz],
            [-dx, dy, -dz],
            [-dx, dy, dz],
            [-dx, -dy, dz],
            [dx, -dy, -dz],
            [dx, dy, -dz],
            [dx, dy, dz],
            [dx, -dy, dz]
        ]), yaw)
        # new_d = np.array([
        #     [-cx, -cy, -cz],
        #     [-cx, cy, -cz],
        #     [-cx, cy, cz],
        #     [-cx, -cy, cz],
        #     [cx, -cy, -cz],
        #     [cx, cy, -cz],
        #     [cx, cy, cz],
        #     [cx, -cy, cz]
        # ])
        rotated_boxes.append(new_d + np.array([cx, cy, cz]))

    return np.stack(rotated_boxes, axis=0)
