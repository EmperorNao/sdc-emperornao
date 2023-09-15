import cv2
import numpy as np
from matplotlib import pyplot as plt, patches
from scipy.spatial.transform import Rotation as R

from utils.types import *
from gui_lib.utils import translate_boxes_lidar2bev, boxes_straight2rotated, transform_boxes_cxcy2ltrb, ltrb2shape
from utils import qt_cv2_fix  # noqa

def project_boxes_to_image(img: np.array, boxes: list, calib: dict, cam_number: str):
    img = np.copy(img)

    proj_img_cam = np.eye(4)
    proj_img_cam[0:3, 0:3] = np.array(calib[cam_number]['camera_matrix']['data']).reshape(-1, 3)
    proj_img_cam = proj_img_cam[0:3, 0:4]

    proj_cam_lidar = np.linalg.inv(np.array(calib['extrinsics']['T_LIDAR_CAM' + cam_number]))

    for box in boxes:

        proj_lidar_box = np.eye(4)
        proj_lidar_box[0:3, 0:3] = R.from_euler('z', box['yaw'], degrees=False).as_matrix()
        proj_lidar_box[0][3] = box['position']['x']
        proj_lidar_box[1][3] = box['position']['y']
        proj_lidar_box[2][3] = box['position']['z']

        width = box['dimensions']['x']
        length = box['dimensions']['y']
        height = box['dimensions']['z']

        front_right_bottom = np.array(
            [[1, 0, 0, length / 2], [0, 1, 0, -width / 2], [0, 0, 1, -height / 2], [0, 0, 0, 1]]);
        front_right_top = np.array([[1, 0, 0, length / 2], [0, 1, 0, -width / 2], [0, 0, 1, height / 2], [0, 0, 0, 1]]);
        front_left_bottom = np.array(
            [[1, 0, 0, length / 2], [0, 1, 0, width / 2], [0, 0, 1, -height / 2], [0, 0, 0, 1]]);
        front_left_top = np.array([[1, 0, 0, length / 2], [0, 1, 0, width / 2], [0, 0, 1, height / 2], [0, 0, 0, 1]]);

        back_right_bottom = np.array(
            [[1, 0, 0, -length / 2], [0, 1, 0, -width / 2], [0, 0, 1, -height / 2], [0, 0, 0, 1]]);
        back_right_top = np.array([[1, 0, 0, -length / 2], [0, 1, 0, -width / 2], [0, 0, 1, height / 2], [0, 0, 0, 1]]);
        back_left_bottom = np.array(
            [[1, 0, 0, -length / 2], [0, 1, 0, width / 2], [0, 0, 1, -height / 2], [0, 0, 0, 1]]);
        back_left_top = np.array([[1, 0, 0, -length / 2], [0, 1, 0, width / 2], [0, 0, 1, height / 2], [0, 0, 0, 1]]);

        tmp = np.matmul(proj_cam_lidar, np.matmul(proj_lidar_box, front_right_bottom))
        f_r_b = np.matmul(proj_img_cam, tmp)

        tmp = np.matmul(proj_cam_lidar, np.matmul(proj_lidar_box, front_right_top))
        f_r_t = np.matmul(proj_img_cam, tmp)

        tmp = np.matmul(proj_cam_lidar, np.matmul(proj_lidar_box, front_left_bottom))
        f_l_b = np.matmul(proj_img_cam, tmp)

        tmp = np.matmul(proj_cam_lidar, np.matmul(proj_lidar_box, front_left_top))
        f_l_t = np.matmul(proj_img_cam, tmp)

        tmp = np.matmul(proj_cam_lidar, np.matmul(proj_lidar_box, back_right_bottom))
        b_r_b = np.matmul(proj_img_cam, tmp)

        tmp = np.matmul(proj_cam_lidar, np.matmul(proj_lidar_box, back_right_top))
        b_r_t = np.matmul(proj_img_cam, tmp)

        tmp = np.matmul(proj_cam_lidar, np.matmul(proj_lidar_box, back_left_bottom))
        b_l_b = np.matmul(proj_img_cam, tmp)

        tmp = np.matmul(proj_cam_lidar, np.matmul(proj_lidar_box, back_left_top))
        b_l_t = np.matmul(proj_img_cam, tmp)

        f_r_b_coord = (int(f_r_b[0][3] / f_r_b[2][3]), int(f_r_b[1][3] / f_r_b[2][3]))
        f_r_t_coord = (int(f_r_t[0][3] / f_r_t[2][3]), int(f_r_t[1][3] / f_r_t[2][3]))
        f_l_b_coord = (int(f_l_b[0][3] / f_l_b[2][3]), int(f_l_b[1][3] / f_l_b[2][3]))
        f_l_t_coord = (int(f_l_t[0][3] / f_l_t[2][3]), int(f_l_t[1][3] / f_l_t[2][3]))

        b_r_b_coord = (int(b_r_b[0][3] / b_r_b[2][3]), int(b_r_b[1][3] / b_r_b[2][3]))
        b_r_t_coord = (int(b_r_t[0][3] / b_r_t[2][3]), int(b_r_t[1][3] / b_r_t[2][3]))
        b_l_b_coord = (int(b_l_b[0][3] / b_l_b[2][3]), int(b_l_b[1][3] / b_l_b[2][3]))
        b_l_t_coord = (int(b_l_t[0][3] / b_l_t[2][3]), int(b_l_t[1][3] / b_l_t[2][3]))

        # Draw  12 lines
        # Front
        cv2.line(img, f_r_b_coord, f_r_t_coord, [0, 0, 255], thickness=2, lineType=8, shift=0)
        cv2.line(img, f_r_b_coord, f_l_b_coord, [0, 0, 255], thickness=2, lineType=8, shift=0)
        cv2.line(img, f_l_b_coord, f_l_t_coord, [0, 0, 255], thickness=2, lineType=8, shift=0)
        cv2.line(img, f_l_t_coord, f_r_t_coord, [0, 0, 255], thickness=2, lineType=8, shift=0)
        # back
        cv2.line(img, b_r_b_coord, b_r_t_coord, [0, 0, 100], thickness=2, lineType=8, shift=0)
        cv2.line(img, b_r_b_coord, b_l_b_coord, [0, 0, 100], thickness=2, lineType=8, shift=0)
        cv2.line(img, b_l_b_coord, b_l_t_coord, [0, 0, 100], thickness=2, lineType=8, shift=0)
        cv2.line(img, b_l_t_coord, b_r_t_coord, [0, 0, 100], thickness=2, lineType=8, shift=0)
        # connect front to back
        cv2.line(img, f_r_b_coord, b_r_b_coord, [0, 0, 150], thickness=2, lineType=8, shift=0)
        cv2.line(img, f_r_t_coord, b_r_t_coord, [0, 0, 150], thickness=2, lineType=8, shift=0)
        cv2.line(img, f_l_b_coord, b_l_b_coord, [0, 0, 150], thickness=2, lineType=8, shift=0)
        cv2.line(img, f_l_t_coord, b_l_t_coord, [0, 0, 150], thickness=2, lineType=8, shift=0)

    return img


def create_bev_image(real_shape, points, original_boxes, path_to_save, draw_boxes=True):

    fb_points = np.logical_and(points[:, 0] > -real_shape[1], points[:, 0] < real_shape[1])
    lr_points = np.logical_and(points[:, 1] > -real_shape[0], points[:, 1] < real_shape[0])

    truncated_points = points[np.logical_and(fb_points, lr_points), :]

    y_img = truncated_points[:, 0] + real_shape[1]
    x_img = -truncated_points[:, 1] + real_shape[0]
    pixels = truncated_points[:, 0]

    idx, translated_boxes = translate_boxes_lidar2bev(original_boxes, real_shape)

    fig, ax = plt.subplots(figsize=(2000 / 200, 2000 / 200), dpi=100)

    ax.scatter(x_img, y_img, s=1, c=pixels, alpha=1.0, cmap='jet')
    ax.set_facecolor((0, 0, 0))
    ax.axis('scaled')

    if draw_boxes:
        for poly in translated_boxes:
            polys = patches.Polygon(poly, closed=True, fill=False, edgecolor='r', linewidth=1)
            ax.add_patch(polys)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    plt.xlim([0, real_shape[1] * 2])
    plt.ylim([0, real_shape[0] * 2])
    plt.savefig(path_to_save, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return idx, translated_boxes


def show_and_trasform2(image: np.array, true_boxes: List[Tensor], predictions: List[Tensor]):

    new_image = np.ascontiguousarray(np.array(image * 256, dtype="uint8"))
    rot_true_boxes = boxes_straight2rotated(true_boxes) * 770/300
    rot_preds = boxes_straight2rotated(predictions) * 770/300
    for box in rot_true_boxes:

        def to_int(l):
            return tuple(map(int, l))

        for i in range(4):
            cv2.line(new_image, to_int(box[i - 1]), to_int(box[i]), [100, 100, 100], thickness=3, lineType=8)

    for box in rot_preds:

        def to_int(l):
            return tuple(map(int, l))

        for i in range(4):
            cv2.line(new_image, to_int(box[i - 1]), to_int(box[i]), [200, 0, 200], thickness=3, lineType=8)

    plt.imshow(new_image)
    plt.show()


def show_boxes(image: np.array, boxes: List[Tensor]):

    new_image = np.ascontiguousarray(np.array(image * 256, dtype="uint8")).transpose((1, 2, 0))
    print(boxes.min(axis=0)[0], boxes.max(axis=0)[0])
    boxes = boxes_straight2rotated(boxes)

    for box in boxes:

        def to_int(l):
            return tuple(map(int, l))

        for i in range(4):
            cv2.line(new_image, to_int(box[i - 1]), to_int(box[i]), [200, 0, 200], thickness=3, lineType=8)

    plt.imshow(new_image)
    plt.show()
    plt.close()


def show_boxes_straight(image: np.array, boxes: List[Tensor]):

    new_image = np.ascontiguousarray(np.transpose(np.array(image * 256, dtype="uint8"), (1, 2, 0)))

    boxes *= 512/300

    for box in boxes:
        box = transform_boxes_cxcy2ltrb(box)
        true_boxes = ltrb2shape(box)

        def to_int(l):
            return tuple(map(lambda x: int(x.item()), l))

        for i in range(4):
            cv2.line(new_image, to_int(true_boxes[i - 1]), to_int(true_boxes[i]), [200, 0, 200], thickness=3, lineType=8)

    plt.imshow(new_image)
    plt.show()
    plt.close()

def show_boxes_straight2(image: np.array, boxes: List[Tensor], pred, save: str = None):

    new_image = np.ascontiguousarray(np.transpose(np.array(image * 256, dtype="uint8"), (1, 2, 0)))

    for box in boxes:
        box = transform_boxes_cxcy2ltrb(box)
        true_boxes = ltrb2shape(box)

        def to_int(l):
            return tuple(map(lambda x: int(x.item()), l))

        for i in range(4):
            cv2.line(new_image, to_int(true_boxes[i - 1]), to_int(true_boxes[i]), [200, 0, 200], thickness=3, lineType=8)

    for box in pred:
        box = transform_boxes_cxcy2ltrb(box)
        true_boxes = ltrb2shape(box)

        def to_int(l):
            return tuple(map(lambda x: int(x.item()), l))

        for i in range(4):
            cv2.line(new_image, to_int(true_boxes[i - 1]), to_int(true_boxes[i]), [0, 0, 200], thickness=3, lineType=8)

    if save:
        plt.imsave(save, new_image)
    else:
        plt.imshow(new_image)
    plt.show()
    plt.close()



def show_boxes2(image: np.array, true_boxes: List[Tensor], predictions: List[Tensor]):
    def to_int(l):
        return tuple(map(int, l))

    new_image = np.ascontiguousarray(np.array(image * 256, dtype="uint8"))

    boxes = boxes_straight2rotated(true_boxes)
    for box in boxes:

        def to_int(l):
            return tuple(map(int, l))

        for i in range(4):
            cv2.line(new_image, to_int(box[i - 1]), to_int(box[i]), [200, 0, 200], thickness=3, lineType=8)


    boxes = boxes_straight2rotated(predictions)
    # boxes *= 770.0/300
    for box in boxes:

        def to_int(l):
            return tuple(map(int, l))

        for i in range(4):
            cv2.line(new_image, to_int(box[i - 1]), to_int(box[i]), [0, 200, 0], thickness=3, lineType=8)

    fig = plt.figure()
    plt.imshow(new_image)
    plt.show(block=True)
    plt.close()


def show_boxes3(image: np.array, true_boxes: List[Tensor], predictions: List[Tensor]):
    def to_int(l):
        return tuple(map(int, l))

    new_image = np.ascontiguousarray(np.array(image * 256, dtype="uint8"))
    # boxes = boxes_straight2rotated(true_boxes)
    #
    # for box in boxes:
    #
    #     def to_int(l):
    #         return tuple(map(int, l))
    #
    #     for i in range(4):
    #         cv2.line(new_image, to_int(box[i - 1]), to_int(box[i]), [200, 0, 200], thickness=3, lineType=8)
    for box in true_boxes:
        cv2.circle(new_image, to_int(box[:2]), 3, [200, 0, 200], thickness=3)

    boxes = boxes_straight2rotated(predictions)
    # for box in boxes:
    #
    #     def to_int(l):
    #         return tuple(map(int, l))
    #
    #     for i in range(4):
    #         cv2.line(new_image, to_int(box[i - 1]), to_int(box[i]), [0, 0, 200], thickness=3, lineType=8)
    for box in predictions:
        cv2.circle(new_image, to_int(box[:2]), 3, [0, 200, 0], thickness=3)

    fig = plt.figure()
    plt.imshow(new_image)
    plt.show(block=True)
    plt.close()