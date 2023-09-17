import math

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt, patches

from utils.types import *
from utils.types import make_tuple
from gui_lib.transformations import boxes_straight2rotated_3d, boxes_straight2rotated, boxes_rotated2cxcydxdy


def translate_boxes_lidar2bev(
        original_boxes: np.array,
        scene_size: Tuple[int, int]
) -> Tuple[np.array, np.array]:
    """
    :param original_boxes: 3d bounding boxes with shape (n, 8)
    :param scene_size: size of scene for boxes to take, all boxes which center lies outside scene wouldn't be taken
    :return: pair (indexes, bboxes), indexes with shape (filtered_n) that was taken and
        bboxes with shape (filtered_n, 4, 2) for bird's-eye view
    """

    boxes = np.copy(original_boxes)

    indexes = np.logical_and(
        np.logical_and(-scene_size[0] < boxes[:, 0], boxes[:, 0] < scene_size[0]),
        np.logical_and(-scene_size[1] < boxes[:, 1], boxes[:, 1] < scene_size[1])
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


def translate_boxes_lidar2bev_image(
        original_boxes: np.array,
        scene_size: Tuple[int, int],
        image_size: Tuple[int, int],
) -> np.array:
    """
    :param original_boxes: 3d bounding boxes with shape (n, 8),
    :param scene_size: size of scene for boxes to take, all boxes which center lies outside scene wouldn't be taken
    :param image_size: size of image for which we need to translate our boxes
    :return: boxes with shape (filtered_n, 6) in image coordinates w.r.t left top coordinate system's origin in format;
        each box presented in format (cx, cy, dx, dy, yaw, shape)
    """

    boxes = np.copy(original_boxes)

    indexes = np.logical_and(
        np.logical_and(-scene_size[0] < boxes[:, 0], boxes[:, 0] < scene_size[0]),
        np.logical_and(-scene_size[1] < boxes[:, 1], boxes[:, 1] < scene_size[1])
    )
    boxes = boxes[indexes]

    cx = boxes[:, 0] + scene_size[0]
    dx = boxes[:, 3]

    cy = boxes[:, 1] + scene_size[1]
    dy = boxes[:, 4]

    yaw = boxes[:, 6]

    label = boxes[:, 7]

    xs, ys = image_size[0] / (scene_size[0] * 2), image_size[1] / (scene_size[1] * 2)

    cx = np.rint(cx * xs).astype(int)
    cy = np.rint(cy * ys).astype(int)
    dx = np.rint(dx * xs).astype(int)
    dy = np.rint(dy * ys).astype(int)

    # for shifting (0, 0) to left top side
    image_cx = image_size[0] - cy
    image_cy = image_size[1] - cx
    image_dx = dy
    image_dy = dx
    image_yaw = -yaw

    # rotate -90 degree for shifting to (0, 0)
    boxes = np.stack(
        (image_cx, image_cy, image_dx, image_dy, np.full_like(image_cx, -math.radians(90)), label),
        axis=-1
    )

    rot = boxes_straight2rotated(boxes)
    boxes = boxes_rotated2cxcydxdy(rot)

    return np.concatenate(
        (boxes, np.expand_dims(image_yaw, -1), np.expand_dims(label, -1)),
        axis=1
    )


def create_bev_image(
        scene_size: Union[int, Tuple[int, int]],
        image_size: Union[int, Tuple[int, int]],
        points: np.array,
        path_to_save: str,
        original_boxes=None,
        draw_boxes=False
) -> Tuple[np.array, np.array]:
    """
    :param scene_size: sizes (x,y) or x -> (x,x) of square from which points will be taken
    :param image_size: size (x,y) or x -> (x,x) of image
    :param points: lidar points
    :param path_to_save: path to save bev image
    :param original_boxes: bounding boxes, if this param provided they will be translated to 2d rot bev
    :param draw_boxes: draw boxes on image or not
    :return: pair (indexes, boxes) of boxes that are taken and translated to 2d rot bev
    """
    scene_size = make_tuple(scene_size)
    image_size = make_tuple(image_size)

    fb_points = np.logical_and(-scene_size[1] < points[:, 0], points[:, 0] < scene_size[1])
    lr_points = np.logical_and(-scene_size[0] < points[:, 1], points[:, 1] < scene_size[0])

    truncated_points = points[np.logical_and(fb_points, lr_points), :]

    y_img = truncated_points[:, 0] + scene_size[1]
    x_img = -truncated_points[:, 1] + scene_size[0]
    pixels = truncated_points[:, 0]

    fig, ax = plt.subplots(figsize=(2000 / 200, 2000 / 200), dpi=100)

    ax.scatter(x_img, y_img, s=1, c=pixels, alpha=1.0, cmap='jet')
    ax.set_facecolor((0, 0, 0))
    ax.axis('scaled')

    if original_boxes is not None:
        boxes_indexes, translated_boxes = translate_boxes_lidar2bev(original_boxes, scene_size)
        if draw_boxes:
            for poly in translated_boxes:
                polys = patches.Polygon(poly, closed=True, fill=False, edgecolor='r', linewidth=1)
                ax.add_patch(polys)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    plt.xlim([0, scene_size[1] * 2])
    plt.ylim([0, scene_size[0] * 2])
    plt.savefig(path_to_save, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    resized_image = Image.open(path_to_save).resize(image_size, Image.Resampling.LANCZOS)
    resized_image.save(path_to_save)

    if original_boxes is not None:
        image_boxes = translate_boxes_lidar2bev_image(original_boxes[boxes_indexes], scene_size, image_size)
        return boxes_indexes, image_boxes
