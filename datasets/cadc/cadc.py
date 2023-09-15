import yaml
import json
import numpy as np

from PIL import Image
from os.path import join
from utils.types import *


cadc_class2label = {
    "Car": 0,
    "Truck": 1,
    "Pedestrian": 2,
    "Traffic_Guidance_Objects": 3,
    "Bus": 4,
    "Garbage_Containers_on_Wheels": 5,
    "Bicycle": 6
}


class CADCProxySequence:

    def __init__(self,
                 data_dir: str,
                 calib_dir: str,
                 n_cams: int = 8
                 ):
        self.data_dir = data_dir
        self.calib_dir = calib_dir
        self.n_cams = n_cams

        with open(join(self.data_dir, "3d_ann.json")) as file:
            self.annotation = json.load(file)

        self.calib = dict()

        self.calib['extrinsics'] = yaml.load(open(join(self.calib_dir, "extrinsics.yaml")), Loader=yaml.Loader)
        for cam_idx in range(self.n_cams):
            cam_name = CADCProxySequence.get_cam_number(cam_idx)
            self.calib[cam_name] = yaml.load(open(join(self.calib_dir, cam_name + ".yaml")), Loader=yaml.Loader)

    def __len__(self) -> int:
        return len(self.annotation)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @staticmethod
    def get_cam_number(idx: int) -> str:
        return str(idx).rjust(2, '0')

    def __getitem__(self, idx: int):

        def get_frame_number(frame_idx: int) -> str:
            return str(frame_idx).rjust(10, '0')

        boxes = self.annotation[idx]["cuboids"]

        return {
            "images": {CADCProxySequence.get_cam_number(cam_idx): np.array(Image.open(
                join(self.data_dir,
                     "labeled",
                     "image_" + CADCProxySequence.get_cam_number(cam_idx),
                     "data",
                     get_frame_number(idx) + ".png")
            )) for cam_idx in range(self.n_cams)
            },
            "points": np.fromfile(
                join(self.data_dir,
                     "labeled",
                     "lidar_points",
                     "data",
                     get_frame_number(idx) + ".bin"
                     ),
                dtype=np.float32
            ).reshape((-1, 4)),
            "boxes": np.array([
                [
                    box['position']['x'], box['position']['y'], box['position']['z'],
                    box['dimensions']['x'], box['dimensions']['y'], box['dimensions']['z'],
                    box['yaw'], cadc_class2label[box['label']]
                ]
                for box in boxes
            ])
        }

    def get_calib(self) -> Dict[str, yaml.YAMLObject]:
        return self.calib
