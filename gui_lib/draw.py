import cv2
from utils import qt_cv2_fix  # noqa
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

from utils.types import *
from gui_lib.transformations import boxes_straight2rotated


def get_image(image_path: str) -> np.array:
    return np.array(Image.open(image_path).convert('RGB'))


def get_image_with_boxes(
        image: np.array,
        boxes: np.array,
        rotate: bool = True,
        color: Tuple[int, int, int] = (100, 100, 100)
):

    new_image = np.ascontiguousarray(np.array(image, dtype="uint8"))
    if rotate:
        rot_boxes = boxes_straight2rotated(boxes)
        for box in rot_boxes:
            for i in range(4):
                cv2.line(new_image, tuple_to_int(box[i - 1]), tuple_to_int(box[i]), color, thickness=1, lineType=16)
    else:
        for box in boxes:
            cx, cy, dx, dy = box[:4]
            box = [[cx - dx / 2, cy - dy / 2],
                   [cx - dx / 2, cy + dy / 2],
                   [cx + dx / 2, cy + dy / 2],
                   [cx + dx / 2, cy - dy / 2]
            ]
            for i in range(4):
                cv2.line(new_image, tuple_to_int(box[i - 1]), tuple_to_int(box[i]), color, thickness=1, lineType=16)

    return new_image


def draw_image_with_boxes(
        image: np.array,
        boxes: np.array,
        rotate: bool = True,
        color: Tuple[int, int, int] = (255, 255, 255)
):
    image = get_image_with_boxes(image, boxes, rotate, color)
    plt.imshow(image)
    plt.show()
