import os
import math
from tqdm import tqdm
from os.path import join

import torch

from utils.types import *
from utils.filesystem import mkdir

from gui_lib.bev import create_bev_image
from datasets.cadc.torch import CADCDatasetLidarOnly


def transform_cadc_to_2d_rot(dataset_path: str,
                             transformed_path: str,
                             scene_size: Union[int, Tuple[int, int]],
                             image_size: Union[int, Tuple[int, int]],
                             train_val_test_ratios: Tuple[float, float, float]
                             ):

    assert abs(sum(train_val_test_ratios) - 1.0) < 1e-5, "train + val + test ratios must be 1.0 in sum"
    only_train = abs(train_val_test_ratios[0] - 1.0) < 1e-5
    scene_size = make_tuple(scene_size)
    image_size = make_tuple(image_size)

    mkdir(transformed_path, True)
    for mode in ["train", "val", "test"]:
        mkdir(join(transformed_path, mode), True)
        for folder in ["images", "target"]:
            mkdir(join(transformed_path, mode, folder), True)

    data_in_dir = {
        day.name: [seq.name for seq in os.scandir(day.path) if seq.is_dir() and seq.name != "calib"] for day in
        os.scandir(dataset_path) if day.is_dir()
    }

    dataset = CADCDatasetLidarOnly(dataset_path, data_in_dir)

    if not only_train:
        last_train = math.ceil(len(dataset) * train_val_test_ratios[0])
        last_val = math.ceil(len(dataset) * sum(train_val_test_ratios[:2]))

    for idx in tqdm(range(len(dataset))):

        current_path = transformed_path
        if not only_train:
            if idx < last_train:
                current_path = join(current_path, "train")
            elif idx < last_val:
                current_path = join(current_path, "val")
            else:
                current_path = join(current_path, "test")
        else:
            current_path = join(current_path, "train")

        points, target, calib = dataset[idx]

        image_path = join(current_path, "images", str(idx) + ".png")
        boxes_idxs, translated_boxes = create_bev_image(
            scene_size,
            image_size,
            points,
            image_path,
            target,
            draw_boxes=False
        )

        translated_boxes = torch.tensor(translated_boxes)
        torch.save(translated_boxes, join(current_path, "target", str(idx) + ".pt"))
