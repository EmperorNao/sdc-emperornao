import math

import numpy as np
from matplotlib import pyplot as plt, patches

from typing import *

from gui_lib.transformations import boxes_straight2rotated_3d


def translate_boxes_lidar2bev(original_boxes: np.array,
                              scene_size: Tuple[int, int]
                              ) -> Tuple[np.array, np.array]:

    boxes = np.copy(original_boxes)

    indexes = np.logical_and(
        np.logical_and(-scene_size[0] < boxes[:, 0], boxes[:, 0] < scene_size[0]),
        np.logical_and(-scene_size[1] < boxes[:, 1], -scene_size[1] < boxes[:, 1])
    )
    boxes = boxes[indexes]

    boxes[:, 6] = math.radians(90) - boxes[:, 6]
    rot = boxes_straight2rotated_3d(boxes[:, :7])

    # first indexes are dependent on order that boxes_straight2rotated_3d returns
    filtered_boxes = np.array([
        [rot[:, 5, 0], rot[:, 5, 1]],
        [rot[:, 2, 0], rot[:, 2, 1]],
        [rot[:, 0, 0], rot[:, 0, 1]],
        [rot[:, 4, 0], rot[:, 4, 1]],
    ])
    filtered_boxes = filtered_boxes.transpose(2, 0, 1)

    if indexes.any():

        translated_boxes = np.copy(filtered_boxes)

        translated_boxes[:, :, 0] = -filtered_boxes[:, :, 1]
        translated_boxes[:, :, 1] = filtered_boxes[:, :, 0]

        translated_boxes += np.array([scene_size, scene_size, scene_size, scene_size])

        return np.arange(0, indexes.shape[0])[indexes], translated_boxes

    else:
        return np.array([]), np.array([])


def create_bev_image(real_shape, points, path_to_save, original_boxes=None):

    fb_points = np.logical_and(points[:, 0] > -real_shape[1], points[:, 0] < real_shape[1])
    lr_points = np.logical_and(points[:, 1] > -real_shape[0], points[:, 1] < real_shape[0])

    truncated_points = points[np.logical_and(fb_points, lr_points), :]

    y_img = truncated_points[:, 0] + real_shape[1]
    x_img = -truncated_points[:, 1] + real_shape[0]
    pixels = truncated_points[:, 0]

    fig, ax = plt.subplots(figsize=(2000 / 200, 2000 / 200), dpi=100)

    ax.scatter(x_img, y_img, s=1, c=pixels, alpha=1.0, cmap='jet')
    ax.set_facecolor((0, 0, 0))
    ax.axis('scaled')

    if original_boxes is not None:
        idx, translated_boxes = translate_boxes_lidar2bev(original_boxes, real_shape)
        for poly in translated_boxes:
            polys = patches.Polygon(poly, closed=True, fill=False, edgecolor='r', linewidth=1)
            ax.add_patch(polys)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    plt.xlim([0, real_shape[1] * 2])
    plt.ylim([0, real_shape[0] * 2])
    plt.savefig(path_to_save, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    if original_boxes is not None:
        return idx, translated_boxes
