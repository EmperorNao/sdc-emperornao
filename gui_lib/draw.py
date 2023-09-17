import cv2
from utils import qt_cv2_fix  # noqa
import numpy as np
from matplotlib import pyplot as plt

from utils.types import *
from gui_lib.transformations import boxes_straight2rotated


def draw_with_boxes(image: np.array, true_boxes: np.array, color: Tuple[int, int, int] = (100, 100, 100)):

    new_image = np.ascontiguousarray(np.array(image, dtype="uint8"))
    rot_boxes = boxes_straight2rotated(true_boxes)
    for box in rot_boxes:
        for i in range(4):
            cv2.line(new_image, tuple_to_int(box[i - 1]), tuple_to_int(box[i]), color, thickness=1, lineType=16)

    plt.imshow(new_image)
    plt.show()
