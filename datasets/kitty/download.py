import os
import wget
import logging
from os.path import join

from utils import types
from utils.filesystem import mkdir, unzip


def download_kitty(base_path: str):

    base_path = join(base_path, "kitty_detecton")
    mkdir(base_path, True)

    dir2url = types.OrderedDict(
        {
            "calib": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip",
            "label": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip",
            "velo": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip",
            "image": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"
        }
    )

    for directory, url in dir2url.items():

        path = os.path.join(base_path, directory)
        mkdir(path, True)

        logging.info("Downloading file '{}' in directory '{}'".format(url, path))

        filename = wget.download(url, join(path, url.rsplit("/", 1)[1]))
        unzip(filename, True)
