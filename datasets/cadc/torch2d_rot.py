import os
import math

import torchvision.transforms
from tqdm import tqdm
from os.path import join

import torch
from torch.utils.data import Dataset, DataLoader

from utils.types import *
from utils.filesystem import mkdir

from gui_lib.draw import get_image
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
        for folder in ["images", "targets"]:
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

        points, target, _ = dataset[idx]

        image_path = join(current_path, "images", str(idx) + ".png")
        target_path = join(current_path, "targets", str(idx) + ".pt")

        boxes_idxs, translated_boxes = create_bev_image(
            scene_size,
            image_size,
            points,
            image_path,
            target,
            draw_boxes=False
        )

        translated_boxes = torch.tensor(translated_boxes)
        torch.save(translated_boxes, target_path)


class CADCBevDataset(Dataset):

    def __init__(self,
                 base_path: str,
                 mode: str,
                 image_transform: Callable = None,
                 target_transform: Callable = None,
                 device: str = 'cpu'
                 ):

        assert mode in ["train", "val", "test"], "mode need to be in ['train', 'val', 'test'], provided '%s'" % mode

        self.path = join(base_path, mode)
        self.images_path = join(self.path, "images")
        self.targets_path = join(self.path, "targets")
        assert os.path.exists(self.path), "Path % don't exist" % self.path

        self.image_transform = image_transform
        self.target_transform = target_transform
        self.device = device

        self.im2target = [(image, image.rsplit('.', 1)[0] + ".pt") for image in os.listdir(self.images_path)]

    def __len__(self) -> int:
        return len(self.im2target)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:

        image_path, target_path = self.im2target[idx]

        image = get_image(join(self.images_path, image_path))
        target = torch.load(join(self.targets_path, target_path))

        if self.image_transform:
            image = self.image_transform(image)
        if self.target_transform:
            target = self.target_transform

        return image.to(self.device), target.to(self.device)

    @staticmethod
    def get_dataloaders(base_dir: str,
                        batch_size: int,
                        device: str = "cpu",
                        ) -> Tuple[Dict[str, Dataset], Dict[str, DataLoader]]:

        image_transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]
        )

        datasets = {}
        dataloaders = {}
        for mode in ["train", "val", "test"]:
            datasets[mode] = CADCBevDataset(base_dir, mode, image_transform=image_transform, device=device)
            dataloaders[mode] = DataLoader(datasets[mode],
                                           batch_size=batch_size,
                                           shuffle=True if mode != "test" else False)

        return datasets, dataloaders
