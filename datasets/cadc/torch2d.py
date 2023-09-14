import os
import math
from os.path import join, isfile

from PIL import Image
from tqdm import tqdm

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor

from utils.types import *
from utils.filesystem import mkdir
from gui_lib.image import create_bev_image
from gui_lib.utils import boxes_bev2image

from datasets.cadc.torch import CADCDatasetLidarOnly


# TODO: refactor torch2d
class CADCPreprocessor:

    def __init__(self, path):

        d = {
            day.name: [seq.name for seq in os.scandir(day.path) if seq.is_dir() and seq.name != "calib"] for day in
            os.scandir(path) if day.is_dir()
        }

        self.dataset = CADCDatasetLidarOnly(path, d)

    def to_2d_rot_dataset(self,
                          new_path: str,
                          scene_size: Tuple[int, int],
                          train_val_test_ratios: Tuple[float, float, float]
                          ):

        assert abs(sum(train_val_test_ratios) - 1.0) < 1e-6
        only_train = abs(train_val_test_ratios[0] - 1.0) < 1e-6

        mkdir(new_path, True)
        for mode in ["train", "val", "test"]:
            mkdir(join(new_path, mode), True)
            for folder in ["images", "targets"]:
                mkdir(join(new_path, mode, folder), True)

        if not only_train:
            last_train = math.ceil(len(self.dataset) * train_val_test_ratios[0])
            last_val = math.ceil(len(self.dataset) * sum(train_val_test_ratios[:2]))

        for idx in tqdm(range(len(self.dataset))):

            data = self.dataset[idx]

            current_path = new_path
            if not only_train:
                if idx < last_train:
                    current_path = join(current_path, "train")
                elif idx < last_val:
                    current_path = join(current_path, "val")
                else:
                    current_path = join(current_path, "test")
            else:
                current_path = join(current_path, "train")

            points, target, calib = data
            idx_boxes, translated_boxes = create_bev_image(
                scene_size,
                points,
                target['boxes'],
                join(current_path, "images", str(idx) + ".png"),
                draw_boxes=False
            )

            if len(translated_boxes):
                image_boxes = boxes_bev2image(translated_boxes, scene_size, (512, 512))

                boxes = np.concatenate(
                    (
                        image_boxes,
                        np.expand_dims(target['boxes'][idx_boxes][:, 6], -1)
                    ),
                    axis=1
                )
            else:
                boxes = np.array([])

            target['boxes'] = torch.Tensor(boxes)
            target['labels'] = target['labels'][idx_boxes]

            torch.save(target, join(current_path, "targets", str(idx) + ".pt"))


class CADCRot2dDataset(Dataset):

    def __init__(self,
                 dataset_path: str,
                 transform: Compose = None,
                 device: str = 'cpu',
                 float_format_transformation: bool = False):

        self.dataset_path = dataset_path
        self.device = device
        self.float_format_transformation = float_format_transformation

        self.samples = [f.rsplit(".png", 1)[0] for f in os.listdir(join(self.dataset_path, "images")) if
                        isfile(join(self.dataset_path, "images", f))]

        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:

        image = Image.open(join(self.dataset_path, "images", self.samples[idx] + ".png"))
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)

        if self.float_format_transformation:
            image /= 256

        pt_targets = torch.load(join(self.dataset_path, "targets", self.samples[idx] + ".pt"))

        targets = {'boxes': pt_targets['boxes'].to(self.device), 'labels': pt_targets['labels'].to(self.device)}

        return image.to(self.device), targets

    @staticmethod
    def get_dataloaders(
            dataset_path: str,
            batch_size: int = 1,
            device: str = 'cpu'
    ) -> Dict[str, DataLoader]:

        transformation = Compose(
            [ToTensor()]
        )

        def collate(data):
            return [x for x, _ in data], [y for _, y in data]

        train_dataset = CADCRot2dDataset(join(dataset_path, "train"), transform=transformation, device=device,
                                         float_format_transformation=transformation is not None)
        val_dataset = CADCRot2dDataset(join(dataset_path, "val"), transform=transformation, device=device,
                                       float_format_transformation=transformation is not None)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

        return {
            "train": train_dataloader,
            "val": val_dataloader
        }
